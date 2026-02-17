"""Cloud storage abstraction package.

Provides vendor-neutral storage operations starting with Google Drive.

Phase 1 exports (Foundation):
    - StorageBackend protocol
    - StorageEntry, EntryType, UploadResult data types
    - Exception hierarchy (StorageError and subclasses)
    - Path normalization utilities
    - validate_google_drive_config() startup validation

Phase 2 will add:
    - GoogleDriveV3Storage implementation
    - AsyncStorageWrapper
    - get_storage() / get_async_storage() factory functions
"""

from image_search_service.storage.base import (
    EntryType,
    StorageBackend,
    StorageEntry,
    UploadResult,
)
from image_search_service.storage.config_validation import validate_google_drive_config
from image_search_service.storage.exceptions import (
    ConfigurationError,
    NotFoundError,
    PathAmbiguousError,
    RateLimitError,
    RootBoundaryError,
    StorageError,
    StoragePermissionError,
    StorageQuotaError,
    UploadError,
)
from image_search_service.storage.path_resolver import (
    LookupFn,
    PathResolver,
    normalize_path,
    sanitize_folder_name,
)

__all__ = [
    # Protocol & types
    "StorageBackend",
    "StorageEntry",
    "EntryType",
    "UploadResult",
    # Exceptions
    "StorageError",
    "NotFoundError",
    "PathAmbiguousError",
    "RateLimitError",
    "StorageQuotaError",
    "RootBoundaryError",
    "StoragePermissionError",
    "ConfigurationError",
    "UploadError",
    # Path utilities
    "LookupFn",
    "PathResolver",
    "normalize_path",
    "sanitize_folder_name",
    # Startup validation
    "validate_google_drive_config",
]
