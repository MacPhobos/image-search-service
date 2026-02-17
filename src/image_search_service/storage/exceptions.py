"""Exception hierarchy for cloud storage operations.

All storage exceptions inherit from StorageError.
Google API HttpError codes are translated to these exceptions
in the GoogleDriveV3Storage implementation (Phase 2).

Exception Tree:
    StorageError (base)
    +-- NotFoundError           (404)
    +-- PathAmbiguousError      (Drive name collision)
    +-- RateLimitError          (429, retryable)
    +-- StorageQuotaError       (403 quotaExceeded, non-retryable)
    +-- RootBoundaryError       (path traversal attempt)
    +-- StoragePermissionError  (403 permission denied)
    +-- ConfigurationError      (invalid config at startup)
    +-- UploadError             (upload-specific failure)
"""

from __future__ import annotations


class StorageError(Exception):
    """Base exception for all storage operations."""

    pass


class NotFoundError(StorageError):
    """Raised when a file or directory does not exist.

    Attributes:
        path: The path or ID that was not found.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(f"Not found: {path}")


class PathAmbiguousError(StorageError):
    """Raised when a path resolves to multiple items.

    Google Drive allows multiple files/folders with the same name
    in the same parent folder. This exception signals that ambiguity.

    Attributes:
        path: The ambiguous path segment.
        count: Number of items found with the same name.
    """

    def __init__(self, path: str, count: int) -> None:
        self.path = path
        self.count = count
        super().__init__(f"Ambiguous path '{path}': found {count} items with the same name")


class RateLimitError(StorageError):
    """Raised when the storage provider rate-limits the request.

    Corresponds to HTTP 429 or 403 with reason 'rateLimitExceeded'.
    This error is retryable with exponential backoff.

    Attributes:
        retry_after: Suggested wait time in seconds, or None if unknown.
    """

    def __init__(self, retry_after: int | None = None) -> None:
        self.retry_after = retry_after
        msg = "Rate limited by storage provider"
        if retry_after is not None:
            msg += f" (retry after {retry_after}s)"
        super().__init__(msg)


class StorageQuotaError(StorageError):
    """Raised when storage quota is exceeded.

    Corresponds to HTTP 403 with reason 'storageQuotaExceeded'.
    This error is NOT retryable -- requires manual cleanup.

    Canonical constructor: keyword-only quota_bytes and used_bytes.

    WRONG:  StorageQuotaError("Google Drive storage quota exceeded")
    RIGHT:  StorageQuotaError()
    RIGHT:  StorageQuotaError(quota_bytes=15_000_000_000, used_bytes=14_900_000_000)

    Attributes:
        quota_bytes: Total quota in bytes, or None if unknown.
        used_bytes: Used quota in bytes, or None if unknown.
    """

    def __init__(
        self,
        quota_bytes: int | None = None,
        used_bytes: int | None = None,
    ) -> None:
        self.quota_bytes = quota_bytes
        self.used_bytes = used_bytes
        msg = "Storage quota exceeded"
        if quota_bytes is not None and used_bytes is not None:
            used_gb = used_bytes / (1024**3)
            total_gb = quota_bytes / (1024**3)
            msg += f" ({used_gb:.1f} GB / {total_gb:.1f} GB)"
        super().__init__(msg)


class RootBoundaryError(StorageError):
    """Raised when an operation attempts to escape the root folder.

    Triggered by path traversal attempts (e.g., "/../../../etc").

    Attributes:
        path: The offending path.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(f"Path escapes root boundary: {path}")


class StoragePermissionError(StorageError):
    """Raised on permission denied from the storage provider.

    Corresponds to HTTP 403 (non-rate-limit, non-quota reasons).
    Common cause: folder not shared with service account.

    Note: Named StoragePermissionError (not PermissionError)
    to avoid shadowing the Python builtin.

    Attributes:
        path: The path or ID where permission was denied.
        detail: Additional context from the provider.
    """

    def __init__(self, path: str, detail: str = "") -> None:
        self.path = path
        self.detail = detail
        msg = f"Permission denied: {path}"
        if detail:
            msg += f". {detail}"
        super().__init__(msg)


class ConfigurationError(StorageError):
    """Raised when storage configuration is invalid.

    Triggered during startup validation:
    - Missing service account key file
    - Invalid file permissions on key file
    - Missing required environment variables

    Attributes:
        field: The configuration field that is invalid.
        detail: Description of what's wrong.
    """

    def __init__(self, field: str, detail: str) -> None:
        self.field = field
        self.detail = detail
        super().__init__(f"Storage configuration error [{field}]: {detail}")


class UploadError(StorageError):
    """Raised when a file upload fails.

    Wraps provider-specific upload errors with context about
    what was being uploaded.

    Canonical constructor: positional filename, optional detail.

    WRONG:  UploadError(local_path="<bytes>", remote_path="...", detail=str(e))
    RIGHT:  UploadError(filename, detail=str(e))
    RIGHT:  UploadError(filename)

    Attributes:
        filename: The filename being uploaded.
        detail: Description of the failure.
    """

    def __init__(self, filename: str, detail: str = "") -> None:
        self.filename = filename
        self.detail = detail
        msg = f"Upload failed: {filename}"
        if detail:
            msg += f". {detail}"
        super().__init__(msg)
