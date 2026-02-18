"""Integration tests for Google Drive storage API endpoints.

All tests use a FakeStorageBackend or mock — no real Google Drive API calls.
All tests work with SQLite in-memory and no external services.

Test coverage:
    GET  /api/v1/gdrive/health               - DriveHealth tests
    POST /api/v1/gdrive/upload               - StartUpload tests
    GET  /api/v1/gdrive/upload/{id}/status   - UploadStatus tests
    DELETE /api/v1/gdrive/upload/{id}        - CancelUpload tests
    GET  /api/v1/gdrive/folders              - ListFolders tests
    POST /api/v1/gdrive/folders              - CreateFolder tests
    Feature flag (GOOGLE_DRIVE_ENABLED)      - FeatureFlag tests
    Response shape contract                  - Contract tests
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.db.models import (
    FaceInstance,
    ImageAsset,
    Person,
    PersonStatus,
    StorageUpload,
    StorageUploadStatus,
    TrainingStatus,
)
from image_search_service.storage.base import EntryType, StorageEntry, UploadResult

# ============================================================================
# Fake storage backend
# ============================================================================


class FakeAsyncStorageWrapper:
    """Async test double for AsyncStorageWrapper.

    Implements ONLY the 5 protocol methods exposed by AsyncStorageWrapper.
    Records calls for assertion in tests.
    """

    def __init__(self, folders: list[StorageEntry] | None = None) -> None:
        self._folders = folders or []
        self._id_counter = 0
        self.created_folders: list[dict[str, Any]] = []

    def _next_id(self) -> str:
        self._id_counter += 1
        return f"fake_folder_id_{self._id_counter}"

    async def upload_file(
        self,
        content: bytes,
        filename: str,
        mime_type: str,
        folder_id: str | None = None,
    ) -> UploadResult:
        return UploadResult(
            file_id=self._next_id(),
            name=filename,
            size=len(content),
            mime_type=mime_type,
        )

    async def create_folder(self, name: str, parent_id: str | None = None) -> str:
        folder_id = self._next_id()
        self.created_folders.append({"name": name, "parent_id": parent_id, "id": folder_id})
        return folder_id

    async def file_exists(self, file_id: str) -> bool:
        return False

    async def list_folder(self, folder_id: str) -> list[StorageEntry]:
        return list(self._folders)

    async def delete_file(self, file_id: str, *, trash: bool = True) -> None:
        pass


# ============================================================================
# Helper fixtures
# ============================================================================


@pytest.fixture
def fake_storage() -> FakeAsyncStorageWrapper:
    """Create a FakeAsyncStorageWrapper with sample folders."""
    sample_folders = [
        StorageEntry(
            name="Family Photos",
            entry_type=EntryType.FOLDER,
            id="folder_family_photos",
            size=None,
            modified_at=datetime(2026, 1, 15, 10, 0, tzinfo=UTC),
        ),
        StorageEntry(
            name="Vacation 2025",
            entry_type=EntryType.FOLDER,
            id="folder_vacation_2025",
            size=None,
            modified_at=datetime(2026, 2, 1, 8, 0, tzinfo=UTC),
        ),
        StorageEntry(
            name="photo.jpg",
            entry_type=EntryType.FILE,
            id="file_photo_jpg",
            size=102400,
            modified_at=datetime(2026, 1, 20, 12, 0, tzinfo=UTC),
        ),
    ]
    return FakeAsyncStorageWrapper(folders=sample_folders)


@pytest.fixture
async def test_person(db_session: AsyncSession) -> Person:
    """Create a test Person record in the database."""
    person = Person(
        id=uuid.uuid4(),
        name="Jane Doe",
        status=PersonStatus.ACTIVE.value,
    )
    db_session.add(person)
    await db_session.commit()
    await db_session.refresh(person)
    return person


@pytest.fixture
async def test_asset(db_session: AsyncSession) -> ImageAsset:
    """Create a test ImageAsset record in the database."""
    asset = ImageAsset(
        path="/test/images/jane_001.jpg",
        training_status=TrainingStatus.TRAINED.value,
        mime_type="image/jpeg",
    )
    db_session.add(asset)
    await db_session.commit()
    await db_session.refresh(asset)
    return asset


def _enable_drive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Helper: set env vars to enable Google Drive in the test environment.

    Must be called BEFORE test_client is created in fixtures where the
    feature flag affects app startup. For function-level tests, call this
    then clear the settings cache so the new env vars are picked up.
    """
    monkeypatch.setenv("GOOGLE_DRIVE_ENABLED", "true")
    monkeypatch.setenv("GOOGLE_DRIVE_ROOT_ID", "fake-root-folder-id")
    monkeypatch.setenv("GOOGLE_DRIVE_SA_JSON", "/dev/null")
    # Clear LRU caches so new settings are picked up
    from image_search_service.core.config import get_settings
    from image_search_service.storage import get_storage

    get_settings.cache_clear()
    get_storage.cache_clear()


def _mock_async_storage(
    monkeypatch: pytest.MonkeyPatch,
    storage: FakeAsyncStorageWrapper,
) -> None:
    """Inject a fake async storage into the storage module.

    Because the route handlers use lazy ``from image_search_service.storage import
    get_async_storage`` inside the function body (to avoid import-time side effects),
    we must patch the source module attribute — NOT a module-level alias on the
    routes module (which doesn't exist).
    """
    monkeypatch.setattr(
        "image_search_service.storage.get_async_storage",
        lambda: storage,
    )


# ============================================================================
# Health endpoint tests
# ============================================================================


class TestDriveHealth:
    """Tests for GET /api/v1/gdrive/health."""

    @pytest.mark.asyncio
    async def test_health_returns_200_when_disabled(
        self, test_client: AsyncClient
    ) -> None:
        """Health endpoint must return 200 even when Drive is disabled."""
        response = await test_client.get("/api/v1/gdrive/health")
        assert response.status_code == 200
        data = response.json()
        assert data["connected"] is False
        assert data["enabled"] is False

    @pytest.mark.asyncio
    async def test_health_returns_200_when_enabled_and_connected(
        self,
        test_client: AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
        fake_storage: FakeAsyncStorageWrapper,
    ) -> None:
        """Health endpoint should show connected=true when Drive works."""
        _enable_drive(monkeypatch)
        _mock_async_storage(monkeypatch, fake_storage)

        response = await test_client.get("/api/v1/gdrive/health")
        assert response.status_code == 200
        data = response.json()
        assert data["connected"] is True
        assert data["enabled"] is True
        assert data["error"] is None

    @pytest.mark.asyncio
    async def test_health_returns_200_when_drive_raises(
        self,
        test_client: AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Health endpoint must return 200 even if Drive connection fails."""
        _enable_drive(monkeypatch)

        # Fake storage that raises on list_folder
        class FailingStorage(FakeAsyncStorageWrapper):
            async def list_folder(self, folder_id: str) -> list[StorageEntry]:
                from image_search_service.storage.exceptions import NotFoundError

                raise NotFoundError(folder_id)

        _mock_async_storage(monkeypatch, FailingStorage())

        response = await test_client.get("/api/v1/gdrive/health")
        assert response.status_code == 200
        data = response.json()
        assert data["connected"] is False
        assert data["enabled"] is True
        assert data["error"] is not None

    @pytest.mark.asyncio
    async def test_health_response_has_all_contract_fields(
        self, test_client: AsyncClient
    ) -> None:
        """Health response must include all fields defined in the API contract."""
        response = await test_client.get("/api/v1/gdrive/health")
        data = response.json()

        required_fields = {
            "connected",
            "enabled",
            "serviceAccountEmail",
            "rootFolderId",
            "rootFolderName",
            "storageUsedBytes",
            "storageTotalBytes",
            "storageUsagePercentage",
            "lastUploadAt",
            "error",
        }
        missing = required_fields - set(data.keys())
        assert not missing, f"Health response missing fields: {missing}"

    @pytest.mark.asyncio
    async def test_health_uses_get_user_email_for_oauth_backend(
        self,
        test_client: AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
        fake_storage: FakeAsyncStorageWrapper,
    ) -> None:
        """CRITICAL-1: Health endpoint calls get_user_email() for OAuth backends.

        When the storage backend is a GoogleDriveOAuthV3Storage instance,
        the health endpoint must call get_user_email() (not _get_sa_email())
        so that OAuth users can confirm their authentication is working.
        """
        from unittest.mock import MagicMock

        from image_search_service.storage.google_drive_oauth_v3 import (
            GoogleDriveOAuthV3Storage,
        )

        _enable_drive(monkeypatch)

        # Create a real GoogleDriveOAuthV3Storage subclass instance so that
        # isinstance(storage, GoogleDriveOAuthV3Storage) returns True inside
        # the route. We override _build_service to avoid Drive API calls and
        # provide get_user_email() and an async list_folder() mock.
        class _FakeOAuthStorage(GoogleDriveOAuthV3Storage):
            def _build_service(self) -> MagicMock:  # type: ignore[override]
                return MagicMock()

            def get_user_email(self) -> str:
                return "alice@example.com"

            async def list_folder(self, folder_id: str) -> list[Any]:  # type: ignore[override]
                return []

        oauth_storage = _FakeOAuthStorage(
            client_id="test-id",
            client_secret="test-secret",
            refresh_token="test-token",
            root_folder_id="test-root",
        )

        monkeypatch.setattr(
            "image_search_service.storage.get_async_storage",
            lambda: oauth_storage,
        )

        response = await test_client.get("/api/v1/gdrive/health")
        assert response.status_code == 200
        data = response.json()
        assert data["connected"] is True
        assert data["serviceAccountEmail"] == "alice@example.com"

    @pytest.mark.asyncio
    async def test_health_uses_sa_email_for_service_account_backend(
        self,
        test_client: AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
        fake_storage: FakeAsyncStorageWrapper,
    ) -> None:
        """CRITICAL-1: Health endpoint calls _get_sa_email() for SA backends.

        When the backend is NOT a GoogleDriveOAuthV3Storage instance (i.e. the
        standard SA backend), _get_sa_email() must be used as before, so
        existing deployments are not broken.
        """
        _enable_drive(monkeypatch)
        _mock_async_storage(monkeypatch, fake_storage)

        # Patch _get_sa_email to return a known value
        monkeypatch.setattr(
            "image_search_service.api.routes.storage._get_sa_email",
            lambda: "sa-robot@my-project.iam.gserviceaccount.com",
        )

        response = await test_client.get("/api/v1/gdrive/health")
        assert response.status_code == 200
        data = response.json()
        assert data["connected"] is True
        assert data["serviceAccountEmail"] == "sa-robot@my-project.iam.gserviceaccount.com"


# ============================================================================
# Start upload endpoint tests
# ============================================================================


class TestStartUpload:
    """Tests for POST /api/v1/gdrive/upload."""

    @pytest.mark.asyncio
    async def test_returns_503_when_drive_disabled(
        self, test_client: AsyncClient
    ) -> None:
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
                bbox_x=10, bbox_y=10, bbox_w=50, bbox_h=50,
                detection_confidence=0.95,
            ),
            # Second face on the same asset — DISTINCT should deduplicate
            FaceInstance(
                asset_id=assets[0].id,
                person_id=test_person.id,
                bbox_x=80, bbox_y=10, bbox_w=50, bbox_h=50,
                detection_confidence=0.90,
            ),
            FaceInstance(
                asset_id=assets[1].id,
                person_id=test_person.id,
                bbox_x=10, bbox_y=10, bbox_w=50, bbox_h=50,
                detection_confidence=0.92,
            ),
            FaceInstance(
                asset_id=assets[2].id,
                person_id=test_person.id,
                bbox_x=10, bbox_y=10, bbox_w=50, bbox_h=50,
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


# ============================================================================
# Upload status endpoint tests
# ============================================================================


class TestUploadStatus:
    """Tests for GET /api/v1/gdrive/upload/{batch_id}/status."""

    @pytest.mark.asyncio
    async def test_returns_404_for_unknown_batch(
        self,
        test_client: AsyncClient,
    ) -> None:
        """Unknown batch_id must return 404."""
        response = await test_client.get(
            f"/api/v1/gdrive/upload/{uuid.uuid4()}/status"
        )
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
                    StorageUploadStatus.COMPLETED.value if i < 3
                    else StorageUploadStatus.PENDING.value
                ),
                remote_path=f"/people/test/photo_{i}.jpg" if i < 3 else None,
                remote_file_id=f"drive-file-{i}" if i < 3 else None,
            )
            db_session.add(upload)
        await db_session.commit()

        response = await test_client.get(
            f"/api/v1/gdrive/upload/{batch_id}/status"
        )

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

        response = await test_client.get(
            f"/api/v1/gdrive/upload/{batch_id}/status"
        )
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

        response = await test_client.get(
            f"/api/v1/gdrive/upload/{batch_id}/status"
        )
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

        response = await test_client.get(
            f"/api/v1/gdrive/upload/{batch_id}/status"
        )
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

        response = await test_client.get(
            f"/api/v1/gdrive/upload/{batch_id}/status"
        )
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


# ============================================================================
# Cancel upload endpoint tests
# ============================================================================


class TestCancelUpload:
    """Tests for DELETE /api/v1/gdrive/upload/{batch_id}."""

    @pytest.mark.asyncio
    async def test_returns_503_when_drive_disabled(
        self, test_client: AsyncClient
    ) -> None:
        """Cancel must return 503 when Google Drive is disabled."""
        response = await test_client.delete(
            f"/api/v1/gdrive/upload/{uuid.uuid4()}"
        )
        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_returns_404_for_unknown_batch(
        self,
        test_client: AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cancel must return 404 for a batch that does not exist."""
        _enable_drive(monkeypatch)

        response = await test_client.delete(
            f"/api/v1/gdrive/upload/{uuid.uuid4()}"
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_returns_409_when_batch_already_completed(
        self,
        test_client: AsyncClient,
        db_session: AsyncSession,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cancel must return 409 when no pending uploads remain."""
        _enable_drive(monkeypatch)
        batch_id = str(uuid.uuid4())

        db_session.add(
            StorageUpload(
                batch_id=batch_id,
                asset_id=500,
                status=StorageUploadStatus.COMPLETED.value,
            )
        )
        await db_session.commit()

        response = await test_client.delete(
            f"/api/v1/gdrive/upload/{batch_id}"
        )
        assert response.status_code == 409

    @pytest.mark.asyncio
    async def test_cancel_marks_pending_records_as_cancelled(
        self,
        test_client: AsyncClient,
        db_session: AsyncSession,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cancel must mark pending uploads as cancelled and return correct counts."""
        _enable_drive(monkeypatch)
        batch_id = str(uuid.uuid4())

        # 2 completed, 2 pending
        for i in range(4):
            db_session.add(
                StorageUpload(
                    batch_id=batch_id,
                    asset_id=600 + i,
                    status=(
                        StorageUploadStatus.COMPLETED.value if i < 2
                        else StorageUploadStatus.PENDING.value
                    ),
                )
            )
        await db_session.commit()

        response = await test_client.delete(
            f"/api/v1/gdrive/upload/{batch_id}"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["batchId"] == batch_id
        assert data["status"] == "cancelled"
        assert data["completedBeforeCancel"] == 2
        assert data["cancelledCount"] == 2
        assert data["message"] is not None

    @pytest.mark.asyncio
    async def test_cancel_response_has_all_contract_fields(
        self,
        test_client: AsyncClient,
        db_session: AsyncSession,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cancel response shape must match the API contract."""
        _enable_drive(monkeypatch)
        batch_id = str(uuid.uuid4())

        db_session.add(
            StorageUpload(
                batch_id=batch_id,
                asset_id=700,
                status=StorageUploadStatus.PENDING.value,
            )
        )
        await db_session.commit()

        response = await test_client.delete(
            f"/api/v1/gdrive/upload/{batch_id}"
        )
        assert response.status_code == 200
        data = response.json()

        required_fields = {
            "batchId", "status", "completedBeforeCancel", "cancelledCount", "message"
        }
        missing = required_fields - set(data.keys())
        assert not missing, f"CancelUploadResponse missing fields: {missing}"


# ============================================================================
# Folder list endpoint tests
# ============================================================================


class TestListFolders:
    """Tests for GET /api/v1/gdrive/folders."""

    @pytest.mark.asyncio
    async def test_returns_503_when_drive_disabled(
        self, test_client: AsyncClient
    ) -> None:
        """Folder listing must return 503 when Drive is not enabled."""
        response = await test_client.get("/api/v1/gdrive/folders")
        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_returns_only_folders_not_files(
        self,
        test_client: AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
        fake_storage: FakeAsyncStorageWrapper,
    ) -> None:
        """Folder listing must filter out file entries, returning only folders."""
        _enable_drive(monkeypatch)
        _mock_async_storage(monkeypatch, fake_storage)

        response = await test_client.get("/api/v1/gdrive/folders")

        assert response.status_code == 200
        data = response.json()
        # fake_storage has 2 folders and 1 file; only folders should appear
        assert data["total"] == 2
        assert len(data["folders"]) == 2
        folder_names = {f["name"] for f in data["folders"]}
        assert "Family Photos" in folder_names
        assert "Vacation 2025" in folder_names

    @pytest.mark.asyncio
    async def test_uses_root_id_when_no_parent_provided(
        self,
        test_client: AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
        fake_storage: FakeAsyncStorageWrapper,
    ) -> None:
        """Without parentId query param, the configured root folder ID is used."""
        _enable_drive(monkeypatch)
        _mock_async_storage(monkeypatch, fake_storage)

        # Track which folder_id was passed to list_folder
        listed_ids: list[str] = []
        original_list = fake_storage.list_folder

        async def tracking_list(folder_id: str) -> list[StorageEntry]:
            listed_ids.append(folder_id)
            return await original_list(folder_id)

        fake_storage.list_folder = tracking_list  # type: ignore[method-assign]

        response = await test_client.get("/api/v1/gdrive/folders")
        assert response.status_code == 200
        assert listed_ids == ["fake-root-folder-id"]

    @pytest.mark.asyncio
    async def test_uses_provided_parent_id(
        self,
        test_client: AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
        fake_storage: FakeAsyncStorageWrapper,
    ) -> None:
        """When parentId is provided, it is passed directly to list_folder."""
        _enable_drive(monkeypatch)
        _mock_async_storage(monkeypatch, fake_storage)

        listed_ids: list[str] = []
        original_list = fake_storage.list_folder

        async def tracking_list(folder_id: str) -> list[StorageEntry]:
            listed_ids.append(folder_id)
            return await original_list(folder_id)

        fake_storage.list_folder = tracking_list  # type: ignore[method-assign]

        response = await test_client.get(
            "/api/v1/gdrive/folders?parentId=custom-parent-folder-id"
        )
        assert response.status_code == 200
        assert listed_ids == ["custom-parent-folder-id"]

    @pytest.mark.asyncio
    async def test_folder_list_response_has_all_contract_fields(
        self,
        test_client: AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
        fake_storage: FakeAsyncStorageWrapper,
    ) -> None:
        """Folder list response shape must match the API contract."""
        _enable_drive(monkeypatch)
        _mock_async_storage(monkeypatch, fake_storage)

        response = await test_client.get("/api/v1/gdrive/folders")
        assert response.status_code == 200
        data = response.json()

        # Top-level response fields
        assert "parentId" in data
        assert "folders" in data
        assert "total" in data

        # Per-folder entry fields
        if data["folders"]:
            folder = data["folders"][0]
            folder_fields = {"id", "name", "parentId", "hasChildren", "createdAt"}
            missing = folder_fields - set(folder.keys())
            assert not missing, f"DriveFolderInfo missing fields: {missing}"

    @pytest.mark.asyncio
    async def test_returns_storage_error_as_http_status(
        self,
        test_client: AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """StorageError from Drive should be translated to an HTTP error response."""
        _enable_drive(monkeypatch)

        class PermissionDeniedStorage(FakeAsyncStorageWrapper):
            async def list_folder(self, folder_id: str) -> list[StorageEntry]:
                from image_search_service.storage.exceptions import StoragePermissionError

                raise StoragePermissionError(folder_id, "Not shared with service account")

        _mock_async_storage(monkeypatch, PermissionDeniedStorage())

        response = await test_client.get("/api/v1/gdrive/folders")
        assert response.status_code == 403


# ============================================================================
# Create folder endpoint tests
# ============================================================================


class TestCreateFolder:
    """Tests for POST /api/v1/gdrive/folders."""

    @pytest.mark.asyncio
    async def test_returns_503_when_drive_disabled(
        self, test_client: AsyncClient
    ) -> None:
        """Folder creation must return 503 when Drive is not enabled."""
        response = await test_client.post(
            "/api/v1/gdrive/folders",
            json={"name": "New Folder"},
        )
        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_returns_422_for_empty_name(
        self,
        test_client: AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Empty folder name must fail Pydantic validation (422 Unprocessable Entity)."""
        _enable_drive(monkeypatch)

        response = await test_client.post(
            "/api/v1/gdrive/folders",
            json={"name": "", "parentId": "some-parent"},
        )
        assert response.status_code == 422  # Pydantic min_length=1

    @pytest.mark.asyncio
    async def test_creates_folder_and_returns_201(
        self,
        test_client: AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
        fake_storage: FakeAsyncStorageWrapper,
    ) -> None:
        """Valid create folder request must return 201 with the new folder ID."""
        _enable_drive(monkeypatch)
        _mock_async_storage(monkeypatch, fake_storage)

        response = await test_client.post(
            "/api/v1/gdrive/folders",
            json={"name": "New Album", "parentId": "parent-folder-id"},
        )

        assert response.status_code == 201
        data = response.json()
        assert "folderId" in data
        assert data["name"] == "New Album"
        assert data["parentId"] == "parent-folder-id"

    @pytest.mark.asyncio
    async def test_uses_root_when_no_parent_provided(
        self,
        test_client: AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
        fake_storage: FakeAsyncStorageWrapper,
    ) -> None:
        """Without parentId, the configured root folder ID is used as parent."""
        _enable_drive(monkeypatch)
        _mock_async_storage(monkeypatch, fake_storage)

        response = await test_client.post(
            "/api/v1/gdrive/folders",
            json={"name": "Root Level Folder"},
        )

        assert response.status_code == 201
        data = response.json()
        # parent_id should default to the configured root
        assert data["parentId"] == "fake-root-folder-id"

    @pytest.mark.asyncio
    async def test_create_folder_response_has_all_contract_fields(
        self,
        test_client: AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
        fake_storage: FakeAsyncStorageWrapper,
    ) -> None:
        """Create folder response shape must match the API contract."""
        _enable_drive(monkeypatch)
        _mock_async_storage(monkeypatch, fake_storage)

        response = await test_client.post(
            "/api/v1/gdrive/folders",
            json={"name": "Contract Test Folder"},
        )
        assert response.status_code == 201
        data = response.json()

        required_fields = {"folderId", "name", "parentId", "path"}
        missing = required_fields - set(data.keys())
        assert not missing, f"CreateFolderResponse missing fields: {missing}"

    @pytest.mark.asyncio
    async def test_storage_error_translated_to_http_status(
        self,
        test_client: AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """StorageError from Drive should be translated to an HTTP error response."""
        _enable_drive(monkeypatch)

        class QuotaExceededStorage(FakeAsyncStorageWrapper):
            async def create_folder(
                self, name: str, parent_id: str | None = None
            ) -> str:
                from image_search_service.storage.exceptions import StorageQuotaError

                raise StorageQuotaError(
                    quota_bytes=15_000_000_000,
                    used_bytes=15_000_000_000,
                )

        _mock_async_storage(monkeypatch, QuotaExceededStorage())

        response = await test_client.post(
            "/api/v1/gdrive/folders",
            json={"name": "Will Fail"},
        )
        assert response.status_code == 507


# ============================================================================
# Feature flag behavior tests
# ============================================================================


class TestFeatureFlag:
    """Tests verifying GOOGLE_DRIVE_ENABLED feature flag behavior."""

    @pytest.mark.asyncio
    async def test_upload_returns_503_when_disabled(
        self, test_client: AsyncClient
    ) -> None:
        """All upload endpoints must return 503 when Drive is disabled."""
        response = await test_client.post(
            "/api/v1/gdrive/upload",
            json={"personId": str(uuid.uuid4()), "photoIds": [1], "folderId": "f"},
        )
        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_folders_post_returns_503_when_disabled(
        self, test_client: AsyncClient
    ) -> None:
        """Folder creation must return 503 when Drive is disabled."""
        response = await test_client.post(
            "/api/v1/gdrive/folders",
            json={"name": "test"},
        )
        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_folders_get_returns_503_when_disabled(
        self, test_client: AsyncClient
    ) -> None:
        """Folder listing must return 503 when Drive is disabled."""
        response = await test_client.get("/api/v1/gdrive/folders")
        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_cancel_returns_503_when_disabled(
        self, test_client: AsyncClient
    ) -> None:
        """Cancel endpoint must return 503 when Drive is disabled."""
        response = await test_client.delete(
            f"/api/v1/gdrive/upload/{uuid.uuid4()}"
        )
        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_health_works_when_disabled(
        self, test_client: AsyncClient
    ) -> None:
        """Health endpoint MUST return 200 even when Drive is disabled."""
        response = await test_client.get("/api/v1/gdrive/health")
        assert response.status_code == 200
        data = response.json()
        assert data["enabled"] is False
        assert data["connected"] is False

    @pytest.mark.asyncio
    async def test_status_endpoint_works_without_drive_enabled(
        self,
        test_client: AsyncClient,
        db_session: AsyncSession,
    ) -> None:
        """Status polling endpoint does NOT require Drive enabled — it only queries DB."""
        batch_id = str(uuid.uuid4())
        db_session.add(
            StorageUpload(
                batch_id=batch_id,
                asset_id=999,
                status=StorageUploadStatus.PENDING.value,
            )
        )
        await db_session.commit()

        # No _require_drive_enabled() call in the status endpoint
        response = await test_client.get(
            f"/api/v1/gdrive/upload/{batch_id}/status"
        )
        assert response.status_code == 200
