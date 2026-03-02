"""Tests for POST /api/v1/gdrive/upload endpoint."""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.db.models import (
    FaceInstance,
    ImageAsset,
    Person,
    TrainingStatus,
)

from .conftest import _enable_drive


class TestStartUpload:
    """Tests for POST /api/v1/gdrive/upload."""

    @pytest.mark.asyncio
    async def test_returns_503_when_drive_disabled(self, test_client: AsyncClient) -> None:
        """Upload endpoint must return 503 when Google Drive is disabled."""
        response = await test_client.post(
            "/api/v1/gdrive/upload",
            json={
                "personId": str(uuid.uuid4()),
                "photoIds": [1, 2],
                "folderId": "some-folder-id",
            },
        )
        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_returns_404_for_unknown_person(
        self,
        test_client: AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Upload must return 404 for a person that does not exist."""
        _enable_drive(monkeypatch)

        response = await test_client.post(
            "/api/v1/gdrive/upload",
            json={
                "personId": str(uuid.uuid4()),
                "photoIds": [1],
                "folderId": "folder-id",
            },
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_returns_400_for_invalid_person_uuid(
        self,
        test_client: AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Upload must return 400 for a malformed person UUID."""
        _enable_drive(monkeypatch)

        response = await test_client.post(
            "/api/v1/gdrive/upload",
            json={
                "personId": "not-a-valid-uuid",
                "photoIds": [1],
                "folderId": "folder-id",
            },
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_returns_404_when_person_has_no_photos(
        self,
        test_client: AsyncClient,
        db_session: AsyncSession,
        monkeypatch: pytest.MonkeyPatch,
        test_person: Person,
    ) -> None:
        """Upload must return 404 when person exists but has no associated photos."""
        _enable_drive(monkeypatch)

        response = await test_client.post(
            "/api/v1/gdrive/upload",
            json={
                "personId": str(test_person.id),
                "photoIds": [],  # Empty = query all, but person has none
                "folderId": "folder-id",
            },
        )
        assert response.status_code == 404
        assert "no photos" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_empty_photo_ids_resolves_all_person_photos(
        self,
        test_client: AsyncClient,
        db_session: AsyncSession,
        monkeypatch: pytest.MonkeyPatch,
        test_person: Person,
    ) -> None:
        """Empty photoIds list must resolve all FaceInstance-linked assets for the person.

        When photoIds is an empty list, the endpoint queries FaceInstance records
        assigned to the person and collects the distinct asset IDs.  The resolved
        total must appear in the response's totalPhotos field.
        """
        _enable_drive(monkeypatch)

        # Create 3 ImageAssets and FaceInstances linked to the test person.
        # Two FaceInstances share asset_id 1 (same photo, two faces) to verify
        # that DISTINCT de-duplication works — totalPhotos must be 3, not 4.
        assets: list[ImageAsset] = []
        for i in range(3):
            asset = ImageAsset(
                path=f"/test/images/person_photo_{i}.jpg",
                training_status=TrainingStatus.TRAINED.value,
                mime_type="image/jpeg",
            )
            db_session.add(asset)
            assets.append(asset)
        await db_session.flush()  # Assign IDs without committing

        face_instances = [
            FaceInstance(
                asset_id=assets[0].id,
                person_id=test_person.id,
                bbox_x=10,
                bbox_y=10,
                bbox_w=50,
                bbox_h=50,
                detection_confidence=0.95,
            ),
            # Second face on the same asset — DISTINCT should deduplicate
            FaceInstance(
                asset_id=assets[0].id,
                person_id=test_person.id,
                bbox_x=80,
                bbox_y=10,
                bbox_w=50,
                bbox_h=50,
                detection_confidence=0.90,
            ),
            FaceInstance(
                asset_id=assets[1].id,
                person_id=test_person.id,
                bbox_x=10,
                bbox_y=10,
                bbox_w=50,
                bbox_h=50,
                detection_confidence=0.92,
            ),
            FaceInstance(
                asset_id=assets[2].id,
                person_id=test_person.id,
                bbox_x=10,
                bbox_y=10,
                bbox_w=50,
                bbox_h=50,
                detection_confidence=0.88,
            ),
        ]
        for fi in face_instances:
            db_session.add(fi)
        await db_session.commit()

        # Mock enqueue to capture the resolved asset_ids without touching Redis
        enqueued_calls: list[dict[str, Any]] = []

        def mock_enqueue_chunked(
            person_id: str,
            asset_ids: list[int],
            remote_base_path: str,
            batch_id: str,
            chunk_size: int = 50,
            start_folder_id: str | None = None,
        ) -> list[str]:
            enqueued_calls.append({"asset_ids": asset_ids})
            return [f"mock-job-{uuid.uuid4().hex[:8]}"]

        monkeypatch.setattr(
            "image_search_service.queue.storage_jobs.enqueue_person_upload_chunked",
            mock_enqueue_chunked,
        )

        response = await test_client.post(
            "/api/v1/gdrive/upload",
            json={
                "personId": str(test_person.id),
                "photoIds": [],  # Empty = resolve all via FaceInstance query
                "folderId": "resolved-folder-id",
            },
        )

        assert response.status_code == 202
        data = response.json()

        # 4 FaceInstances across 3 distinct assets — totalPhotos must be 3 (DISTINCT)
        assert data["totalPhotos"] == 3

        # The enqueued job must receive exactly the 3 distinct asset IDs
        assert len(enqueued_calls) == 1
        resolved_ids = sorted(enqueued_calls[0]["asset_ids"])
        expected_ids = sorted(a.id for a in assets)
        assert resolved_ids == expected_ids

    @pytest.mark.asyncio
    async def test_enqueues_jobs_for_valid_request_with_photo_ids(
        self,
        test_client: AsyncClient,
        db_session: AsyncSession,
        monkeypatch: pytest.MonkeyPatch,
        test_person: Person,
        test_asset: ImageAsset,
    ) -> None:
        """Valid upload request with explicit photoIds must return 202 with batchId."""
        _enable_drive(monkeypatch)

        # Mock the enqueue function to avoid real Redis dependency
        enqueued_calls: list[dict[str, Any]] = []

        def mock_enqueue_chunked(
            person_id: str,
            asset_ids: list[int],
            remote_base_path: str,
            batch_id: str,
            chunk_size: int = 50,
            start_folder_id: str | None = None,
        ) -> list[str]:
            enqueued_calls.append(
                {
                    "person_id": person_id,
                    "asset_ids": asset_ids,
                    "remote_base_path": remote_base_path,
                    "batch_id": batch_id,
                    "start_folder_id": start_folder_id,
                }
            )
            return [f"mock-job-{uuid.uuid4().hex[:8]}"]

        monkeypatch.setattr(
            "image_search_service.queue.storage_jobs.enqueue_person_upload_chunked",
            mock_enqueue_chunked,
        )

        response = await test_client.post(
            "/api/v1/gdrive/upload",
            json={
                "personId": str(test_person.id),
                "photoIds": [test_asset.id],
                "folderId": "target-drive-folder-id",
            },
        )

        assert response.status_code == 202
        data = response.json()
        assert "batchId" in data
        assert data["totalPhotos"] == 1
        assert len(data["jobIds"]) == 1
        assert data["message"] is not None

        # Verify correct arguments were passed to enqueue.
        # start_folder_id must carry the Drive folder ID; remote_base_path
        # must contain only name segments (the person subfolder name).
        assert len(enqueued_calls) == 1
        call = enqueued_calls[0]
        assert call["person_id"] == str(test_person.id)
        assert call["asset_ids"] == [test_asset.id]
        # The Drive folder ID must travel as start_folder_id, NOT embedded in the path
        assert call["start_folder_id"] == "target-drive-folder-id"
        # remote_base_path is the person name (subfolder created inside the Drive folder)
        assert call["remote_base_path"] == test_person.name

    @pytest.mark.asyncio
    async def test_enqueue_with_person_name_override(
        self,
        test_client: AsyncClient,
        db_session: AsyncSession,
        monkeypatch: pytest.MonkeyPatch,
        test_person: Person,
        test_asset: ImageAsset,
    ) -> None:
        """Person name override should be used as remote_base_path when subfolder is created."""
        _enable_drive(monkeypatch)

        enqueued_calls: list[dict[str, Any]] = []

        def mock_enqueue_chunked(
            person_id: str,
            asset_ids: list[int],
            remote_base_path: str,
            batch_id: str,
            chunk_size: int = 50,
            start_folder_id: str | None = None,
        ) -> list[str]:
            enqueued_calls.append(
                {"remote_base_path": remote_base_path, "start_folder_id": start_folder_id}
            )
            return ["mock-job-1"]

        monkeypatch.setattr(
            "image_search_service.queue.storage_jobs.enqueue_person_upload_chunked",
            mock_enqueue_chunked,
        )

        response = await test_client.post(
            "/api/v1/gdrive/upload",
            json={
                "personId": str(test_person.id),
                "photoIds": [test_asset.id],
                "folderId": "target-folder",
                "options": {
                    "personNameOverride": "Custom Name",
                    "createPersonSubfolder": True,
                },
            },
        )

        assert response.status_code == 202
        assert len(enqueued_calls) == 1
        call = enqueued_calls[0]
        # The override name must be the remote_base_path (subfolder name), not the Drive ID
        assert call["remote_base_path"] == "Custom Name"
        # The Drive folder ID must be the start_folder_id
        assert call["start_folder_id"] == "target-folder"

    @pytest.mark.asyncio
    async def test_enqueue_without_person_subfolder(
        self,
        test_client: AsyncClient,
        db_session: AsyncSession,
        monkeypatch: pytest.MonkeyPatch,
        test_person: Person,
        test_asset: ImageAsset,
    ) -> None:
        """With createPersonSubfolder=False, remote_base_path is empty.

        The Drive folder ID travels as start_folder_id so files are uploaded
        directly into the selected folder with no subfolder created.
        """
        _enable_drive(monkeypatch)

        enqueued_calls: list[dict[str, Any]] = []

        def mock_enqueue_chunked(
            person_id: str,
            asset_ids: list[int],
            remote_base_path: str,
            batch_id: str,
            chunk_size: int = 50,
            start_folder_id: str | None = None,
        ) -> list[str]:
            enqueued_calls.append(
                {"remote_base_path": remote_base_path, "start_folder_id": start_folder_id}
            )
            return ["mock-job-1"]

        monkeypatch.setattr(
            "image_search_service.queue.storage_jobs.enqueue_person_upload_chunked",
            mock_enqueue_chunked,
        )

        response = await test_client.post(
            "/api/v1/gdrive/upload",
            json={
                "personId": str(test_person.id),
                "photoIds": [test_asset.id],
                "folderId": "target-folder-id",
                "options": {
                    "createPersonSubfolder": False,
                },
            },
        )

        assert response.status_code == 202
        assert len(enqueued_calls) == 1
        call = enqueued_calls[0]
        # With no subfolder, remote_base_path has no name segments (upload directly
        # into the Drive folder identified by start_folder_id)
        assert call["remote_base_path"] == ""
        # The Drive folder ID travels as start_folder_id (not embedded in the path)
        assert call["start_folder_id"] == "target-folder-id"

    @pytest.mark.asyncio
    async def test_start_upload_response_has_all_contract_fields(
        self,
        test_client: AsyncClient,
        db_session: AsyncSession,
        monkeypatch: pytest.MonkeyPatch,
        test_person: Person,
        test_asset: ImageAsset,
    ) -> None:
        """Upload response must include all fields from the API contract."""
        _enable_drive(monkeypatch)

        def _mock_enqueue(**kw: object) -> list[str]:
            return ["mock-job-1"]

        monkeypatch.setattr(
            "image_search_service.queue.storage_jobs.enqueue_person_upload_chunked",
            _mock_enqueue,
        )

        response = await test_client.post(
            "/api/v1/gdrive/upload",
            json={
                "personId": str(test_person.id),
                "photoIds": [test_asset.id],
                "folderId": "folder",
            },
        )

        assert response.status_code == 202
        data = response.json()
        required_fields = {"batchId", "jobIds", "totalPhotos", "estimatedTimeSeconds", "message"}
        missing = required_fields - set(data.keys())
        assert not missing, f"StartUploadResponse missing fields: {missing}"
