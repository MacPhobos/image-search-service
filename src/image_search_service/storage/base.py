"""Cloud storage abstraction protocol and data types.

Defines the vendor-neutral interface for cloud storage operations.
Implementations include GoogleDriveV3Storage (Phase 2) and
FakeStorageBackend (testing).

All paths are virtual paths relative to a configured root.
"/" represents the provider-specific root (e.g., shared Drive folder).
Paths use forward slashes: "/people/John/photo1.jpg"
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Protocol, runtime_checkable


class EntryType(str, Enum):
    """Type of storage entry."""

    FILE = "file"
    FOLDER = "folder"


@dataclass(frozen=True)
class StorageEntry:
    """Represents an item in cloud storage.

    Attributes:
        name: Display name of the file or folder.
        entry_type: Whether this is a file or folder.
        id: Provider-specific opaque ID (Drive file ID, S3 key, etc.).
        size: Size in bytes. None for folders.
        modified_at: Last modification timestamp. None if unavailable.
        mime_type: MIME type string. None for folders or unknown types.
    """

    name: str
    entry_type: EntryType
    id: str
    size: int | None
    modified_at: datetime | None
    mime_type: str | None = None


@dataclass(frozen=True)
class UploadResult:
    """Result of a file upload operation.

    Canonical field names (use EXACTLY these -- no aliases):
        file_id   (NOT: id, drive_id)
        name      (NOT: filename, file_name)
        size      (NOT: size_bytes, file_size, file_size_bytes)
        mime_type (NOT: mimeType, content_type)

    Attributes:
        file_id: Provider-specific file ID.
        name: Filename as stored in the provider.
        size: Size in bytes of the uploaded file.
        mime_type: MIME type of the uploaded file.
    """

    file_id: str
    name: str
    size: int
    mime_type: str


@runtime_checkable
class StorageBackend(Protocol):
    """Cloud storage abstraction protocol.

    All paths are virtual paths relative to a configured root.
    "/" represents the provider-specific root (e.g., shared Drive folder).
    Paths use forward slashes: "/people/John/photo1.jpg"

    Implementations must be sync. For async FastAPI routes, wrap with
    AsyncStorageWrapper (Phase 2).

    5 core operations:
    - upload_file: Upload file content to storage
    - create_folder: Create a folder (with parents)
    - file_exists: Check if a file exists by ID
    - list_folder: List contents of a folder
    - delete_file: Delete a file or folder
    """

    def upload_file(
        self,
        content: bytes,
        filename: str,
        mime_type: str,
        folder_id: str | None = None,
    ) -> UploadResult:
        """Upload file content to storage.

        Args:
            content: Raw file bytes.
            filename: Name for the file in storage.
            mime_type: MIME type of the file (e.g., "image/jpeg").
            folder_id: Provider-specific parent folder ID.
                       None means upload to root.

        Returns:
            UploadResult with file ID, name, size, and MIME type.

        Raises:
            UploadError: On upload failure.
            RateLimitError: If provider rate-limits the request.
            StorageQuotaError: If storage quota is exceeded.
            StoragePermissionError: If permission is denied.
        """
        ...

    def create_folder(self, name: str, parent_id: str | None = None) -> str:
        """Create a folder in storage. Return the folder ID.

        Idempotent: if a folder with the same name already exists
        in the parent, returns the existing folder's ID.

        Args:
            name: Folder name (not a path -- a single segment).
            parent_id: Provider-specific parent folder ID.
                       None means create under root.

        Returns:
            Provider-specific folder ID (new or existing).

        Raises:
            PathAmbiguousError: If multiple folders with the same name
                                exist in the parent.
            StoragePermissionError: If permission is denied.
        """
        ...

    def file_exists(self, file_id: str) -> bool:
        """Check if a file or folder exists by its provider-specific ID.

        Args:
            file_id: Provider-specific file/folder ID.

        Returns:
            True if the file exists and is not trashed.
        """
        ...

    def list_folder(self, folder_id: str) -> list[StorageEntry]:
        """List direct children of a folder.

        Args:
            folder_id: Provider-specific folder ID (NOT a virtual path).
                       Use PathResolver.resolve() first if you have a path.

        Returns:
            List of StorageEntry objects for direct children.
            Empty list if folder is empty.

        Raises:
            NotFoundError: If folder_id does not exist.
            StoragePermissionError: If permission is denied.
        """
        ...

    def delete_file(self, file_id: str, *, trash: bool = True) -> None:
        """Delete a file or folder.

        Args:
            file_id: Provider-specific file/folder ID.
            trash: If True (default), move to trash (recoverable).
                   If False, permanently delete. Implementations that
                   do not support permanent deletion may ignore this flag
                   and always trash. Document this in the implementation.

        Raises:
            NotFoundError: If file_id does not exist.
            StoragePermissionError: If permission is denied.
        """
        ...
