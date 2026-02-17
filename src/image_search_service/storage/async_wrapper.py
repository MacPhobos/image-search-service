"""Async wrapper for synchronous StorageBackend implementations.

Delegates blocking I/O to a bounded ThreadPoolExecutor via
asyncio.run_in_executor(), keeping the FastAPI event loop unblocked.

Pattern matches existing codebase usage of async wrappers:
- DB sync operations wrapped in asyncio.to_thread for RQ compatibility
- Embedding service runs sync inference, wrapped for async routes
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import TYPE_CHECKING, Any

from image_search_service.storage.base import StorageEntry, UploadResult

if TYPE_CHECKING:
    from image_search_service.storage.base import StorageBackend


# Module-level executor shared across all AsyncStorageWrapper instances.
# Bounded to 4 workers to:
# 1. Respect Drive API rate limits (~3 writes/sec sustained)
# 2. Limit resource consumption (each thread holds a Drive connection)
# 3. Prevent thread pool exhaustion in the FastAPI process
_storage_executor = ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="storage-io",
)


class AsyncStorageWrapper:
    """Async facade over a synchronous StorageBackend.

    Wraps each sync method with asyncio.run_in_executor() to prevent
    blocking the FastAPI event loop during Drive API calls.

    Exposes ONLY the 5 protocol methods (no mkdirp, no listdir).
    See INTERFACES.md §10.

    Usage in FastAPI routes:
        storage = get_async_storage()
        result = await storage.upload_file(content, "photo.jpg", "image/jpeg", folder_id)

    Usage in RQ jobs:
        storage = get_storage()  # Use sync directly, no wrapper needed
        result = storage.upload_file(content, "photo.jpg", "image/jpeg", folder_id)
    """

    def __init__(self, storage: StorageBackend) -> None:
        """Initialize the wrapper.

        Args:
            storage: Synchronous StorageBackend implementation to wrap.
        """
        self._storage = storage

    async def _run_in_executor(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Run a synchronous function in the thread pool executor.

        Args:
            fn: Sync callable to execute.
            *args: Positional arguments forwarded to fn.
            **kwargs: Keyword arguments forwarded to fn.

        Returns:
            Whatever fn returns.

        Raises:
            Whatever fn raises (propagated across thread boundary).
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _storage_executor,
            partial(fn, *args, **kwargs),
        )

    async def upload_file(
        self,
        content: bytes,
        filename: str,
        mime_type: str,
        folder_id: str | None = None,
    ) -> UploadResult:
        """Async upload file to storage.

        Args:
            content: Raw file bytes.
            filename: Name for the file in storage.
            mime_type: MIME type (e.g., "image/jpeg").
            folder_id: Provider-specific parent folder ID. None = upload to root.

        Returns:
            UploadResult with file_id, name, size, mime_type.
        """
        result: UploadResult = await self._run_in_executor(
            self._storage.upload_file,
            content,
            filename,
            mime_type,
            folder_id,
        )
        return result

    async def create_folder(self, name: str, parent_id: str | None = None) -> str:
        """Async create folder in storage.

        Args:
            name: Folder name (not a path — a single segment).
            parent_id: Provider-specific parent folder ID. None = create under root.

        Returns:
            Provider-specific folder ID (new or existing).
        """
        result: str = await self._run_in_executor(
            self._storage.create_folder,
            name,
            parent_id,
        )
        return result

    async def file_exists(self, file_id: str) -> bool:
        """Async check if file exists.

        Args:
            file_id: Provider-specific file/folder ID.

        Returns:
            True if the file exists and is not trashed.
        """
        result: bool = await self._run_in_executor(
            self._storage.file_exists,
            file_id,
        )
        return result

    async def list_folder(self, folder_id: str) -> list[StorageEntry]:
        """Async list folder contents.

        Args:
            folder_id: Provider-specific folder ID (NOT a virtual path).

        Returns:
            List of StorageEntry objects for direct children.
        """
        result: list[StorageEntry] = await self._run_in_executor(
            self._storage.list_folder,
            folder_id,
        )
        return result

    async def delete_file(self, file_id: str, *, trash: bool = True) -> None:
        """Async delete (or permanently delete) file.

        CANONICAL: accepts trash parameter to match StorageBackend protocol.
        See INTERFACES.md §10.

        Args:
            file_id: Provider-specific file/folder ID.
            trash: If True (default), move to trash. If False, permanently delete.
        """
        await self._run_in_executor(
            self._storage.delete_file,
            file_id,
            trash=trash,
        )
