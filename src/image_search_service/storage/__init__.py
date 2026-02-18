"""Cloud storage abstraction layer.

Provides factory functions for storage backend instances following
the existing codebase pattern of @lru_cache singletons.

Pattern matches:
- services/embedding.py: get_embedding_service() -> EmbeddingService
- core/config.py: get_settings() -> Settings
- db/session.py: get_engine() -> AsyncEngine

Supported backends:
    - GoogleDriveV3Storage        — service account (SA) authentication
    - GoogleDriveOAuthV3Storage   — OAuth 2.0 user credentials (personal accounts)

The active backend is selected by GOOGLE_DRIVE_AUTH_MODE:
    "service_account" (default) → GoogleDriveV3Storage
    "oauth"                      → GoogleDriveOAuthV3Storage

Exports (Foundation):
    - StorageBackend protocol
    - StorageEntry, EntryType, UploadResult data types
    - Exception hierarchy (StorageError and subclasses)
    - Path normalization utilities
    - validate_google_drive_config() startup validation
    - GoogleDriveV3Storage, GoogleDriveOAuthV3Storage implementations
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
    from image_search_service.storage.google_drive_oauth_v3 import GoogleDriveOAuthV3Storage

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
    # Concrete backends (for isinstance checks in health endpoints)
    "GoogleDriveV3Storage",
    "GoogleDriveOAuthV3Storage",
]


@lru_cache(maxsize=1)
def get_storage() -> GoogleDriveV3Storage | None:
    """Get cached sync storage client (for RQ workers and CLI).

    Returns a GoogleDriveV3Storage (or GoogleDriveOAuthV3Storage subclass)
    if Google Drive is enabled and configured, or None if disabled.

    The concrete backend is selected by GOOGLE_DRIVE_AUTH_MODE:
        "service_account" (default) → GoogleDriveV3Storage (SA JSON key)
        "oauth"                      → GoogleDriveOAuthV3Storage (refresh token)

    Follows the existing @lru_cache(maxsize=1) singleton pattern from:
    - services/embedding.py (get_embedding_service)
    - core/config.py (get_settings)

    Returns:
        GoogleDriveV3Storage (or subclass) if enabled, None if disabled.

    Raises:
        ConfigurationError: If enabled but misconfigured (missing credentials).
    """
    from image_search_service.core.config import get_settings
    from image_search_service.storage.google_drive import GoogleDriveV3Storage

    settings = get_settings()

    if not settings.google_drive_enabled:
        return None

    # Root folder ID is required for both auth modes.
    if not settings.google_drive_root_id:
        raise ConfigurationError(
            field="GOOGLE_DRIVE_ROOT_ID",
            detail="Must be set when GOOGLE_DRIVE_ENABLED=true",
        )

    auth_mode = settings.google_drive_auth_mode

    if auth_mode == "oauth":
        # OAuth 2.0 user credentials path (personal Google accounts).
        from image_search_service.storage.google_drive_oauth_v3 import (
            GoogleDriveOAuthV3Storage,
        )

        if not settings.google_drive_client_id:
            raise ConfigurationError(
                field="GOOGLE_DRIVE_CLIENT_ID",
                detail="Must be set when GOOGLE_DRIVE_AUTH_MODE=oauth",
            )
        if not settings.google_drive_client_secret:
            raise ConfigurationError(
                field="GOOGLE_DRIVE_CLIENT_SECRET",
                detail="Must be set when GOOGLE_DRIVE_AUTH_MODE=oauth",
            )
        if not settings.google_drive_refresh_token:
            raise ConfigurationError(
                field="GOOGLE_DRIVE_REFRESH_TOKEN",
                detail=(
                    "Must be set when GOOGLE_DRIVE_AUTH_MODE=oauth. "
                    "Run: python scripts/gdrive_oauth_bootstrap.py"
                ),
            )

        return GoogleDriveOAuthV3Storage(
            client_id=settings.google_drive_client_id,
            client_secret=settings.google_drive_client_secret,
            refresh_token=settings.google_drive_refresh_token,
            root_folder_id=settings.google_drive_root_id,
            path_cache_maxsize=settings.google_drive_path_cache_maxsize,
            path_cache_ttl=settings.google_drive_path_cache_ttl,
        )

    # Default: service_account path (Google Workspace / shared folders).
    if not settings.google_drive_sa_json:
        raise ConfigurationError(
            field="GOOGLE_DRIVE_SA_JSON",
            detail="Must be set when GOOGLE_DRIVE_AUTH_MODE=service_account",
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
