"""Shared fixtures and helpers for API integration tests."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.db.models import (
    ImageAsset,
    Person,
    PersonStatus,
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
# Helper functions
# ============================================================================


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
