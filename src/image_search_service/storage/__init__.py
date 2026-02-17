"""Cloud storage abstraction layer.

Provides factory functions for storage backend instances following
the existing codebase pattern of @lru_cache singletons.

Pattern matches:
- services/embedding.py: get_embedding_service() -> EmbeddingService
- core/config.py: get_settings() -> Settings
- db/session.py: get_engine() -> AsyncEngine

Phase 1 exports (Foundation):
    - StorageBackend protocol
    - StorageEntry, EntryType, UploadResult data types
    - Exception hierarchy (StorageError and subclasses)
    - Path normalization utilities
    - validate_google_drive_config() startup validation

Phase 2 adds:
    - GoogleDriveV3Storage implementation
    - AsyncStorageWrapper
    - get_storage() / get_async_storage() factory functions

Usage:
    # In FastAPI routes (async):
    from image_search_service.storage import get_async_storage
    storage = get_async_storage()
    if storage is None:
        raise HTTPException(503, "Google Drive not enabled")
    result = await storage.upload_file(...)

    # In RQ background jobs (sync):
    from image_search_service.storage import get_storage
    storage = get_storage()
    if storage is None:
        return  # Drive disabled, skip
    result = storage.upload_file(...)
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from image_search_service.storage.async_wrapper import AsyncStorageWrapper
    from image_search_service.storage.google_drive import GoogleDriveV3Storage

__all__ = [
    # Protocol & types
    "StorageBackend",
    "StorageEntry",
    "EntryType",
    "UploadResult",
    # Exceptions
    "StorageError",
    "NotFoundError",
    # AlreadyExistsError removed — does not exist (see INTERFACES.md §3)
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
    # Factory functions (Phase 2)
    "get_storage",
    "get_async_storage",
]


@lru_cache(maxsize=1)
def get_storage() -> GoogleDriveV3Storage | None:
    """Get cached sync storage client (for RQ workers and CLI).

    Returns GoogleDriveV3Storage if Google Drive is enabled and configured,
    or None if disabled via feature toggle.

    Follows the existing @lru_cache(maxsize=1) singleton pattern from:
    - services/embedding.py (get_embedding_service)
    - core/config.py (get_settings)

    Returns:
        GoogleDriveV3Storage instance if enabled, None if disabled.

    Raises:
        ConfigurationError: If enabled but misconfigured (missing SA key, etc.).
    """
    from image_search_service.core.config import get_settings
    from image_search_service.storage.google_drive import GoogleDriveV3Storage

    settings = get_settings()

    if not settings.google_drive_enabled:
        return None

    if not settings.google_drive_sa_json:
        raise ConfigurationError(
            field="GOOGLE_DRIVE_SA_JSON",
            detail="Must be set when GOOGLE_DRIVE_ENABLED=true",
        )
    if not settings.google_drive_root_id:
        raise ConfigurationError(
            field="GOOGLE_DRIVE_ROOT_ID",
            detail="Must be set when GOOGLE_DRIVE_ENABLED=true",
        )

    return GoogleDriveV3Storage(
        service_account_json_path=settings.google_drive_sa_json,
        root_folder_id=settings.google_drive_root_id,
        path_cache_maxsize=settings.google_drive_path_cache_maxsize,
        path_cache_ttl=settings.google_drive_path_cache_ttl,
    )


def get_async_storage() -> AsyncStorageWrapper | None:
    """Get async storage wrapper (for FastAPI routes).

    NOT cached with @lru_cache — creates a new wrapper each call, but the
    underlying sync storage IS cached via get_storage(). Per INTERFACES.md §8:
    "get_async_storage() is NOT cached with @lru_cache — shares the cached
    sync backend underneath."

    Returns:
        AsyncStorageWrapper instance if enabled, None if disabled.

    Raises:
        ConfigurationError: If enabled but misconfigured.
    """
    storage = get_storage()
    if storage is None:
        return None

    from image_search_service.storage.async_wrapper import AsyncStorageWrapper

    return AsyncStorageWrapper(storage)
