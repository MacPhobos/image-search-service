"""Unit tests for AsyncStorageWrapper.

Verifies that each async method correctly delegates to the sync
StorageBackend, and that errors propagate across thread boundaries.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from image_search_service.storage.async_wrapper import AsyncStorageWrapper
from image_search_service.storage.base import EntryType, StorageEntry, UploadResult
from image_search_service.storage.exceptions import (
    NotFoundError,
    StorageQuotaError,
    UploadError,
)

# ─── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_sync_storage() -> MagicMock:
    """Create a mock synchronous StorageBackend."""
    storage = MagicMock()
    # CANONICAL: UploadResult(file_id, name, size, mime_type) — see INTERFACES.md §2.1
    storage.upload_file.return_value = UploadResult(
        file_id="async-upload-id",
        name="test.jpg",  # NOT filename=
        size=1024,  # NOT size_bytes=
        mime_type="image/jpeg",
    )
    storage.create_folder.return_value = "async-folder-id"
    storage.file_exists.return_value = True
    storage.list_folder.return_value = []
    storage.delete_file.return_value = None
    return storage


@pytest.fixture
def async_storage(mock_sync_storage: MagicMock) -> AsyncStorageWrapper:
    """Create an AsyncStorageWrapper around mock sync storage."""
    return AsyncStorageWrapper(mock_sync_storage)


# ─── TestAsyncWrapper ──────────────────────────────────────────────────────────


class TestAsyncWrapper:
    """Tests for AsyncStorageWrapper delegation and error propagation."""

    @pytest.mark.asyncio
    async def test_upload_file_delegates_to_sync(
        self, async_storage: AsyncStorageWrapper, mock_sync_storage: MagicMock
    ) -> None:
        """Async upload_file calls sync storage.upload_file in executor."""
        result = await async_storage.upload_file(b"data", "test.jpg", "image/jpeg", "folder-id")
        assert result.file_id == "async-upload-id"
        assert result.name == "test.jpg"
        assert result.size == 1024
        mock_sync_storage.upload_file.assert_called_once_with(
            b"data", "test.jpg", "image/jpeg", "folder-id"
        )

    @pytest.mark.asyncio
    async def test_upload_file_with_none_folder_id(
        self, async_storage: AsyncStorageWrapper, mock_sync_storage: MagicMock
    ) -> None:
        """Async upload_file passes None folder_id to sync."""
        await async_storage.upload_file(b"data", "test.jpg", "image/jpeg", None)
        mock_sync_storage.upload_file.assert_called_once_with(
            b"data", "test.jpg", "image/jpeg", None
        )

    @pytest.mark.asyncio
    async def test_create_folder_delegates_to_sync(
        self, async_storage: AsyncStorageWrapper, mock_sync_storage: MagicMock
    ) -> None:
        """Async create_folder calls sync storage.create_folder."""
        result = await async_storage.create_folder("Test Folder", "parent-id")
        assert result == "async-folder-id"
        mock_sync_storage.create_folder.assert_called_once_with("Test Folder", "parent-id")

    @pytest.mark.asyncio
    async def test_create_folder_with_none_parent(
        self, async_storage: AsyncStorageWrapper, mock_sync_storage: MagicMock
    ) -> None:
        """Async create_folder passes None parent_id to sync."""
        await async_storage.create_folder("Root Folder")
        mock_sync_storage.create_folder.assert_called_once_with("Root Folder", None)

    @pytest.mark.asyncio
    async def test_file_exists_delegates_to_sync(
        self, async_storage: AsyncStorageWrapper, mock_sync_storage: MagicMock
    ) -> None:
        """Async file_exists calls sync storage.file_exists."""
        result = await async_storage.file_exists("some-file-id")
        assert result is True
        mock_sync_storage.file_exists.assert_called_once_with("some-file-id")

    @pytest.mark.asyncio
    async def test_file_exists_returns_false(
        self, async_storage: AsyncStorageWrapper, mock_sync_storage: MagicMock
    ) -> None:
        """Async file_exists propagates False from sync."""
        mock_sync_storage.file_exists.return_value = False
        result = await async_storage.file_exists("missing-id")
        assert result is False

    @pytest.mark.asyncio
    async def test_list_folder_delegates_to_sync(
        self, async_storage: AsyncStorageWrapper, mock_sync_storage: MagicMock
    ) -> None:
        """Async list_folder calls sync storage.list_folder."""
        entries = [
            StorageEntry(
                name="photo.jpg",
                entry_type=EntryType.FILE,
                id="f1",
                size=1024,
                modified_at=None,
                mime_type="image/jpeg",
            )
        ]
        mock_sync_storage.list_folder.return_value = entries

        result = await async_storage.list_folder("folder-id")
        assert result == entries
        mock_sync_storage.list_folder.assert_called_once_with("folder-id")

    @pytest.mark.asyncio
    async def test_list_folder_returns_empty_list(
        self, async_storage: AsyncStorageWrapper, mock_sync_storage: MagicMock
    ) -> None:
        """Async list_folder returns empty list for empty folder."""
        result = await async_storage.list_folder("empty-folder")
        assert result == []

    @pytest.mark.asyncio
    async def test_delete_file_defaults_to_trash(
        self, async_storage: AsyncStorageWrapper, mock_sync_storage: MagicMock
    ) -> None:
        """Async delete_file delegates with trash=True by default."""
        await async_storage.delete_file("file-id")
        mock_sync_storage.delete_file.assert_called_once_with("file-id", trash=True)

    @pytest.mark.asyncio
    async def test_delete_file_passes_trash_false(
        self, async_storage: AsyncStorageWrapper, mock_sync_storage: MagicMock
    ) -> None:
        """Async delete_file forwards trash=False to sync."""
        await async_storage.delete_file("file-id", trash=False)
        mock_sync_storage.delete_file.assert_called_once_with("file-id", trash=False)

    @pytest.mark.asyncio
    async def test_upload_error_propagates(
        self, async_storage: AsyncStorageWrapper, mock_sync_storage: MagicMock
    ) -> None:
        """UploadError from sync storage propagates through async wrapper."""
        mock_sync_storage.upload_file.side_effect = UploadError("test.jpg", detail="boom")

        with pytest.raises(UploadError) as exc_info:
            await async_storage.upload_file(b"data", "test.jpg", "image/jpeg")

        assert exc_info.value.filename == "test.jpg"

    @pytest.mark.asyncio
    async def test_quota_error_propagates(
        self, async_storage: AsyncStorageWrapper, mock_sync_storage: MagicMock
    ) -> None:
        """StorageQuotaError from sync storage propagates through async wrapper.

        CANONICAL: StorageQuotaError() with NO positional string arg.
        See INTERFACES.md §3.
        """
        mock_sync_storage.upload_file.side_effect = StorageQuotaError()

        with pytest.raises(StorageQuotaError):
            await async_storage.upload_file(b"data", "test.jpg", "image/jpeg")

    @pytest.mark.asyncio
    async def test_not_found_error_propagates(
        self, async_storage: AsyncStorageWrapper, mock_sync_storage: MagicMock
    ) -> None:
        """NotFoundError from sync storage propagates through async wrapper."""
        mock_sync_storage.list_folder.side_effect = NotFoundError("folder-id")

        with pytest.raises(NotFoundError):
            await async_storage.list_folder("folder-id")

    @pytest.mark.asyncio
    async def test_create_folder_error_propagates(
        self, async_storage: AsyncStorageWrapper, mock_sync_storage: MagicMock
    ) -> None:
        """Errors from create_folder propagate correctly."""
        from image_search_service.storage.exceptions import PathAmbiguousError

        mock_sync_storage.create_folder.side_effect = PathAmbiguousError("TestFolder", 3)

        with pytest.raises(PathAmbiguousError) as exc_info:
            await async_storage.create_folder("TestFolder")

        assert exc_info.value.count == 3


# ─── TestExecutorConfiguration ────────────────────────────────────────────────


class TestExecutorConfiguration:
    """Tests for ThreadPoolExecutor configuration."""

    def test_executor_is_module_level(self) -> None:
        """The ThreadPoolExecutor is shared at module level."""
        from image_search_service.storage.async_wrapper import _storage_executor

        assert _storage_executor is not None
        assert _storage_executor._max_workers == 4  # type: ignore[attr-defined]

    def test_executor_has_correct_thread_prefix(self) -> None:
        """ThreadPoolExecutor uses 'storage-io' thread name prefix."""
        from image_search_service.storage.async_wrapper import _storage_executor

        assert _storage_executor._thread_name_prefix == "storage-io"  # type: ignore[attr-defined]

    def test_two_wrappers_share_executor(self) -> None:
        """Multiple AsyncStorageWrapper instances share the same executor."""
        from image_search_service.storage.async_wrapper import _storage_executor

        storage1 = AsyncStorageWrapper(MagicMock())
        storage2 = AsyncStorageWrapper(MagicMock())

        # Both wrappers use the same module-level executor
        # (Verified by checking it's the same object referenced in the module)
        import image_search_service.storage.async_wrapper as aw_module

        assert aw_module._storage_executor is _storage_executor
        # The wrappers themselves don't store a reference to the executor
        assert not hasattr(storage1, "_executor")
        assert not hasattr(storage2, "_executor")
