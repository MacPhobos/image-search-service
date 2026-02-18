"""Startup validation for Google Drive configuration.

Provides validate_google_drive_config() which is called lazily when
Google Drive features are first used -- not at import time.
This follows the project's lazy initialization pattern.

Supported auth modes:
    "service_account" (default): Validates SA JSON key file presence and format.
    "oauth":                     Validates client_id, client_secret, refresh_token.
"""

from __future__ import annotations

import json
import logging
import stat
from pathlib import Path

from image_search_service.core.config import Settings
from image_search_service.storage.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


def validate_google_drive_config(settings: Settings) -> None:
    """Validate Google Drive configuration at startup.

    Dispatches to the appropriate auth-mode validator based on
    GOOGLE_DRIVE_AUTH_MODE. Called lazily when Google Drive features are
    first used, not at import time. Follows the project's lazy init pattern.

    Args:
        settings: Application settings.

    Raises:
        ConfigurationError: If configuration is invalid for the selected mode.
    """
    if not settings.google_drive_enabled:
        return  # Nothing to validate

    # Root folder ID is required regardless of auth mode.
    root_id = settings.google_drive_root_id
    if not root_id:
        raise ConfigurationError(
            "GOOGLE_DRIVE_ROOT_ID",
            "Root folder ID is required when Google Drive is enabled. "
            "Set GOOGLE_DRIVE_ROOT_ID to the Google Drive folder ID for uploads.",
        )

    if settings.google_drive_auth_mode == "oauth":
        _validate_oauth_config(settings)
    else:
        _validate_service_account_config(settings)


def _validate_service_account_config(settings: Settings) -> None:
    """Validate service account configuration.

    Args:
        settings: Application settings.

    Raises:
        ConfigurationError: If SA JSON configuration is invalid.
    """
    sa_json = settings.google_drive_sa_json
    if not sa_json:
        raise ConfigurationError(
            "GOOGLE_DRIVE_SA_JSON",
            "Service account JSON path is required when GOOGLE_DRIVE_AUTH_MODE=service_account. "
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

    # Validate that the file is valid JSON and contains type=service_account.
    try:
        with sa_path.open() as f:
            sa_data = json.load(f)
        if sa_data.get("type") != "service_account":
            raise ConfigurationError(
                "GOOGLE_DRIVE_SA_JSON",
                "File does not appear to be a service account key "
                f"(missing type=service_account): {sa_path}",
            )
    except json.JSONDecodeError as e:
        raise ConfigurationError(
            "GOOGLE_DRIVE_SA_JSON",
            f"File is not valid JSON: {sa_path}: {e}",
        ) from e

    # Check file permissions (warn if too permissive).
    mode = sa_path.stat().st_mode
    if mode & (stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH):
        logger.warning(
            "Service account key file has group/other permissions. "
            "Security recommendation: chmod 600 %s",
            sa_path,
        )


def _validate_oauth_config(settings: Settings) -> None:
    """Validate OAuth 2.0 user credential configuration.

    Checks that all three required OAuth credentials are present.
    Does NOT attempt a live token refresh (that happens in _build_service()).

    Args:
        settings: Application settings.

    Raises:
        ConfigurationError: If any required OAuth credential is missing.
    """
    if not settings.google_drive_client_id:
        raise ConfigurationError(
            "GOOGLE_DRIVE_CLIENT_ID",
            "OAuth client ID is required when GOOGLE_DRIVE_AUTH_MODE=oauth. "
            "See Google Cloud Console > APIs & Services > Credentials.",
        )
    if not settings.google_drive_client_secret:
        raise ConfigurationError(
            "GOOGLE_DRIVE_CLIENT_SECRET",
            "OAuth client secret is required when GOOGLE_DRIVE_AUTH_MODE=oauth.",
        )
    if not settings.google_drive_refresh_token:
        raise ConfigurationError(
            "GOOGLE_DRIVE_REFRESH_TOKEN",
            "OAuth refresh token is required when GOOGLE_DRIVE_AUTH_MODE=oauth. "
            "Run: python scripts/gdrive_oauth_bootstrap.py",
        )
