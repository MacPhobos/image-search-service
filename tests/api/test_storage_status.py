"""Tests for GET /api/v1/gdrive/upload/{batch_id}/status endpoint."""

from __future__ import annotations

import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.db.models import StorageUpload, StorageUploadStatus


class TestUploadStatus:
    """Tests for GET /api/v1/gdrive/upload/{batch_id}/status."""

    @pytest.mark.asyncio
    async def test_returns_404_for_unknown_batch(
        self,
        test_client: AsyncClient,
    ) -> None:
        """Unknown batch_id must return 404."""
        response = await test_client.get(f"/api/v1/gdrive/upload/{uuid.uuid4()}/status")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_returns_progress_for_known_batch(
        self,
        test_client: AsyncClient,
        db_session: AsyncSession,
    ) -> None:
        """Known batch must return progress data with correct counts."""
        batch_id = str(uuid.uuid4())

        # Insert 5 upload records: 3 completed, 2 pending
        # SQLite does NOT enforce FK constraints by default, so asset_id values
        # do not need to exist in image_assets table.
        for i in range(5):
            upload = StorageUpload(
                batch_id=batch_id,
                asset_id=i + 1,
                status=(
                    StorageUploadStatus.COMPLETED.value
                    if i < 3
                    else StorageUploadStatus.PENDING.value
                ),
                remote_path=f"/people/test/photo_{i}.jpg" if i < 3 else None,
                remote_file_id=f"drive-file-{i}" if i < 3 else None,
            )
            db_session.add(upload)
        await db_session.commit()

        response = await test_client.get(f"/api/v1/gdrive/upload/{batch_id}/status")

        assert response.status_code == 200
        data = response.json()
        assert data["batchId"] == batch_id
        assert data["total"] == 5
        assert data["completed"] == 3
        assert data["failed"] == 0
        assert data["inProgress"] == 2
        assert data["status"] == "in_progress"
        assert data["percentage"] == 60.0

    @pytest.mark.asyncio
    async def test_status_completed_when_all_done(
        self,
        test_client: AsyncClient,
        db_session: AsyncSession,
    ) -> None:
        """Batch with all records completed must report status=completed."""
        batch_id = str(uuid.uuid4())
        for i in range(3):
            db_session.add(
                StorageUpload(
                    batch_id=batch_id,
                    asset_id=i + 100,
                    status=StorageUploadStatus.COMPLETED.value,
                    remote_file_id=f"drive-{i}",
                )
            )
        await db_session.commit()

        response = await test_client.get(f"/api/v1/gdrive/upload/{batch_id}/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["percentage"] == 100.0

    @pytest.mark.asyncio
    async def test_status_partial_failure_when_some_failed(
        self,
        test_client: AsyncClient,
        db_session: AsyncSession,
    ) -> None:
        """Batch with some completed and some failed must report status=partial_failure."""
        batch_id = str(uuid.uuid4())
        db_session.add(
            StorageUpload(
                batch_id=batch_id,
                asset_id=200,
                status=StorageUploadStatus.COMPLETED.value,
            )
        )
        db_session.add(
            StorageUpload(
                batch_id=batch_id,
                asset_id=201,
                status=StorageUploadStatus.FAILED.value,
                error_message="Network error",
            )
        )
        await db_session.commit()

        response = await test_client.get(f"/api/v1/gdrive/upload/{batch_id}/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "partial_failure"
        assert data["failed"] == 1
        assert data["completed"] == 1

    @pytest.mark.asyncio
    async def test_status_response_includes_file_list(
        self,
        test_client: AsyncClient,
        db_session: AsyncSession,
    ) -> None:
        """Status response must include per-file status entries."""
        batch_id = str(uuid.uuid4())
        db_session.add(
            StorageUpload(
                batch_id=batch_id,
                asset_id=300,
                status=StorageUploadStatus.COMPLETED.value,
                remote_path="/people/Jane Doe/photo.jpg",
                remote_file_id="drive-file-abc",
            )
        )
        await db_session.commit()

        response = await test_client.get(f"/api/v1/gdrive/upload/{batch_id}/status")
        assert response.status_code == 200
        data = response.json()
        assert len(data["files"]) == 1
        file_entry = data["files"][0]
        assert file_entry["assetId"] == 300
        assert file_entry["status"] == "completed"
        assert file_entry["remoteFileId"] == "drive-file-abc"
        assert file_entry["filename"] == "photo.jpg"

    @pytest.mark.asyncio
    async def test_status_response_has_all_contract_fields(
        self,
        test_client: AsyncClient,
        db_session: AsyncSession,
    ) -> None:
        """Status response shape must match the API contract exactly."""
        batch_id = str(uuid.uuid4())
        db_session.add(
            StorageUpload(
                batch_id=batch_id,
                asset_id=400,
                status=StorageUploadStatus.COMPLETED.value,
            )
        )
        await db_session.commit()

        response = await test_client.get(f"/api/v1/gdrive/upload/{batch_id}/status")
        assert response.status_code == 200
        data = response.json()

        required_fields = {
            "batchId",
            "status",
            "total",
            "completed",
            "failed",
            "inProgress",
            "percentage",
            "etaSeconds",
            "files",
            "startedAt",
            "completedAt",
        }
        missing = required_fields - set(data.keys())
        assert not missing, f"UploadStatusResponse missing fields: {missing}"

        # Verify per-file entry shape
        if data["files"]:
            file_fields = {
                "assetId",
                "filename",
                "status",
                "remoteFileId",
                "errorMessage",
                "completedAt",
            }
            file_missing = file_fields - set(data["files"][0].keys())
            assert not file_missing, f"UploadFileStatus missing fields: {file_missing}"
