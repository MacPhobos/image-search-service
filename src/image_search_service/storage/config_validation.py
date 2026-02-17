"""Startup validation for Google Drive configuration.

Provides validate_google_drive_config() which is called lazily when
Google Drive features are first used -- not at import time.
This follows the project's lazy initialization pattern.
"""

from __future__ import annotations

import logging

from image_search_service.storage.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


def validate_google_drive_config(settings: object) -> None:
    """Validate Google Drive configuration at startup.

    Called lazily when Google Drive features are first used,
    not at import time. Follows the project's lazy initialization pattern.

    Args:
        settings: Application settings (image_search_service.core.config.Settings).

    Raises:
        ConfigurationError: If configuration is invalid.
    """
    import stat
    from pathlib import Path

    # Use getattr for forward-compatibility and to avoid circular imports
    if not getattr(settings, "google_drive_enabled", False):
        return  # Nothing to validate

    # Validate SA JSON path
    sa_json: str = getattr(settings, "google_drive_sa_json", "")
    if not sa_json:
        raise ConfigurationError(
            "GOOGLE_DRIVE_SA_JSON",
            "Service account JSON path is required when Google Drive is enabled. "
            "Set GOOGLE_DRIVE_SA_JSON=/path/to/service-account-key.json",
        )

    sa_path = Path(sa_json)
    if not sa_path.exists():
        raise ConfigurationError(
            "GOOGLE_DRIVE_SA_JSON",
            f"Service account key file not found: {sa_path}",
        )

    if not sa_path.is_file():
        raise ConfigurationError(
            "GOOGLE_DRIVE_SA_JSON",
            f"Path is not a file: {sa_path}",
        )

    # Check file permissions (warn if too permissive)
    mode = sa_path.stat().st_mode
    if mode & (stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH):
        logger.warning(
            "Service account key file has group/other permissions. "
            "Security recommendation: chmod 600 %s",
            sa_path,
        )

    # Validate root folder ID
    root_id: str = getattr(settings, "google_drive_root_id", "")
    if not root_id:
        raise ConfigurationError(
            "GOOGLE_DRIVE_ROOT_ID",
            "Root folder ID is required when Google Drive is enabled. "
            "Share a Google Drive folder with the service account and "
            "set GOOGLE_DRIVE_ROOT_ID to the folder's ID.",
        )
