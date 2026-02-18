"""Tests for Google Drive configuration settings and validation."""

from __future__ import annotations

import os
import stat
import tempfile

import pytest
from pydantic import ValidationError

from image_search_service.core.config import Settings
from image_search_service.storage.config_validation import validate_google_drive_config
from image_search_service.storage.exceptions import ConfigurationError


class TestGoogleDriveSettings:
    """Tests for Google Drive fields in Settings.

    Note: Fields use aliases (e.g. GOOGLE_DRIVE_ENABLED) for env var loading.
    Tests use monkeypatch.setenv() to override env vars and instantiate
    Settings() without arguments, following the project's existing pattern.
    """

    def test_default_disabled(self) -> None:
        settings = Settings()
        assert settings.google_drive_enabled is False

    def test_default_empty_strings(self) -> None:
        settings = Settings()
        assert settings.google_drive_sa_json == ""
        assert settings.google_drive_root_id == ""

    def test_default_batch_size(self) -> None:
        settings = Settings()
        assert settings.google_drive_upload_batch_size == 10

    def test_default_cache_ttl(self) -> None:
        settings = Settings()
        assert settings.google_drive_path_cache_ttl == 300

    def test_default_cache_maxsize(self) -> None:
        settings = Settings()
        assert settings.google_drive_path_cache_maxsize == 1024

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOGLE_DRIVE_ENABLED", "true")
        monkeypatch.setenv("GOOGLE_DRIVE_SA_JSON", "/tmp/sa.json")
        monkeypatch.setenv("GOOGLE_DRIVE_ROOT_ID", "abc123")
        monkeypatch.setenv("GOOGLE_DRIVE_UPLOAD_BATCH_SIZE", "25")

        settings = Settings()
        assert settings.google_drive_enabled is True
        assert settings.google_drive_sa_json == "/tmp/sa.json"
        assert settings.google_drive_root_id == "abc123"
        assert settings.google_drive_upload_batch_size == 25

    def test_batch_size_minimum_validation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """batch_size must be >= 1."""
        monkeypatch.setenv("GOOGLE_DRIVE_UPLOAD_BATCH_SIZE", "0")
        with pytest.raises(ValidationError):
            Settings()

    def test_batch_size_maximum_validation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """batch_size must be <= 50."""
        monkeypatch.setenv("GOOGLE_DRIVE_UPLOAD_BATCH_SIZE", "51")
        with pytest.raises(ValidationError):
            Settings()

    def test_cache_ttl_minimum_validation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """cache_ttl must be >= 60."""
        monkeypatch.setenv("GOOGLE_DRIVE_PATH_CACHE_TTL", "59")
        with pytest.raises(ValidationError):
            Settings()

    def test_cache_ttl_maximum_validation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """cache_ttl must be <= 3600."""
        monkeypatch.setenv("GOOGLE_DRIVE_PATH_CACHE_TTL", "3601")
        with pytest.raises(ValidationError):
            Settings()

    def test_cache_maxsize_minimum_validation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """cache_maxsize must be >= 128."""
        monkeypatch.setenv("GOOGLE_DRIVE_PATH_CACHE_MAXSIZE", "127")
        with pytest.raises(ValidationError):
            Settings()

    def test_cache_maxsize_maximum_validation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """cache_maxsize must be <= 10000."""
        monkeypatch.setenv("GOOGLE_DRIVE_PATH_CACHE_MAXSIZE", "10001")
        with pytest.raises(ValidationError):
            Settings()

    def test_env_var_cache_ttl_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOGLE_DRIVE_PATH_CACHE_TTL", "600")
        settings = Settings()
        assert settings.google_drive_path_cache_ttl == 600

    def test_env_var_cache_maxsize_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOGLE_DRIVE_PATH_CACHE_MAXSIZE", "512")
        settings = Settings()
        assert settings.google_drive_path_cache_maxsize == 512


class TestValidateGoogleDriveConfig:
    """Tests for validate_google_drive_config()."""

    def test_disabled_skips_validation(self) -> None:
        """No validation when google_drive_enabled=False."""
        settings = Settings()
        assert settings.google_drive_enabled is False
        # Should not raise even though SA JSON is empty
        validate_google_drive_config(settings)  # No exception

    def test_enabled_missing_sa_json_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ConfigurationError when enabled but SA JSON path missing."""
        monkeypatch.setenv("GOOGLE_DRIVE_ENABLED", "true")
        monkeypatch.setenv("GOOGLE_DRIVE_SA_JSON", "")
        monkeypatch.setenv("GOOGLE_DRIVE_ROOT_ID", "abc123")
        settings = Settings()

        with pytest.raises(ConfigurationError, match="GOOGLE_DRIVE_SA_JSON"):
            validate_google_drive_config(settings)

    def test_enabled_nonexistent_sa_file_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ConfigurationError when SA JSON file doesn't exist."""
        monkeypatch.setenv("GOOGLE_DRIVE_ENABLED", "true")
        monkeypatch.setenv("GOOGLE_DRIVE_SA_JSON", "/nonexistent/path/service-account.json")
        monkeypatch.setenv("GOOGLE_DRIVE_ROOT_ID", "abc123")
        settings = Settings()

        with pytest.raises(ConfigurationError, match="GOOGLE_DRIVE_SA_JSON"):
            validate_google_drive_config(settings)

    def test_enabled_missing_root_id_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ConfigurationError when enabled but root folder ID missing."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b'{"type": "service_account"}')
            sa_path = f.name
        try:
            monkeypatch.setenv("GOOGLE_DRIVE_ENABLED", "true")
            monkeypatch.setenv("GOOGLE_DRIVE_SA_JSON", sa_path)
            monkeypatch.setenv("GOOGLE_DRIVE_ROOT_ID", "")
            settings = Settings()

            with pytest.raises(ConfigurationError, match="GOOGLE_DRIVE_ROOT_ID"):
                validate_google_drive_config(settings)
        finally:
            os.unlink(sa_path)

    def test_valid_config_does_not_raise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Valid configuration passes without raising."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b'{"type": "service_account"}')
            sa_path = f.name
        try:
            monkeypatch.setenv("GOOGLE_DRIVE_ENABLED", "true")
            monkeypatch.setenv("GOOGLE_DRIVE_SA_JSON", sa_path)
            monkeypatch.setenv("GOOGLE_DRIVE_ROOT_ID", "folder_abc123")
            settings = Settings()
            # Should not raise
            validate_google_drive_config(settings)
        finally:
            os.unlink(sa_path)

    def test_sa_path_not_a_file_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ConfigurationError when SA JSON path points to a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("GOOGLE_DRIVE_ENABLED", "true")
            monkeypatch.setenv("GOOGLE_DRIVE_SA_JSON", tmpdir)  # Directory, not a file
            monkeypatch.setenv("GOOGLE_DRIVE_ROOT_ID", "folder_abc123")
            settings = Settings()

            with pytest.raises(ConfigurationError, match="GOOGLE_DRIVE_SA_JSON"):
                validate_google_drive_config(settings)

    def test_permissive_file_permissions_logs_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Permissive file permissions log a warning (not an error)."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b'{"type": "service_account"}')
            sa_path = f.name
        try:
            # Make file group-readable (too permissive)
            os.chmod(sa_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP)

            monkeypatch.setenv("GOOGLE_DRIVE_ENABLED", "true")
            monkeypatch.setenv("GOOGLE_DRIVE_SA_JSON", sa_path)
            monkeypatch.setenv("GOOGLE_DRIVE_ROOT_ID", "folder_abc123")
            settings = Settings()

            import logging

            with caplog.at_level(logging.WARNING):
                validate_google_drive_config(settings)

            # Should complete without error (warning only)
            assert any("permission" in record.message.lower() for record in caplog.records)
        finally:
            os.unlink(sa_path)

    def test_configuration_error_has_field_attribute(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ConfigurationError stores the field name as an attribute."""
        monkeypatch.setenv("GOOGLE_DRIVE_ENABLED", "true")
        monkeypatch.setenv("GOOGLE_DRIVE_SA_JSON", "")
        monkeypatch.setenv("GOOGLE_DRIVE_ROOT_ID", "abc")
        settings = Settings()

        with pytest.raises(ConfigurationError) as exc_info:
            validate_google_drive_config(settings)
        assert exc_info.value.field == "GOOGLE_DRIVE_SA_JSON"

    def test_sa_file_invalid_json_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ConfigurationError when SA JSON file contains invalid JSON."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b"this is not valid json {{{")
            sa_path = f.name
        try:
            monkeypatch.setenv("GOOGLE_DRIVE_ENABLED", "true")
            monkeypatch.setenv("GOOGLE_DRIVE_SA_JSON", sa_path)
            monkeypatch.setenv("GOOGLE_DRIVE_ROOT_ID", "folder_abc123")
            settings = Settings()

            with pytest.raises(ConfigurationError, match="GOOGLE_DRIVE_SA_JSON"):
                validate_google_drive_config(settings)
        finally:
            os.unlink(sa_path)

    def test_sa_file_wrong_type_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ConfigurationError when SA JSON file has wrong type field."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b'{"type": "authorized_user", "client_id": "foo"}')
            sa_path = f.name
        try:
            monkeypatch.setenv("GOOGLE_DRIVE_ENABLED", "true")
            monkeypatch.setenv("GOOGLE_DRIVE_SA_JSON", sa_path)
            monkeypatch.setenv("GOOGLE_DRIVE_ROOT_ID", "folder_abc123")
            settings = Settings()

            with pytest.raises(ConfigurationError, match="GOOGLE_DRIVE_SA_JSON"):
                validate_google_drive_config(settings)
        finally:
            os.unlink(sa_path)


class TestGoogleDriveOAuthSettings:
    """Tests for OAuth config fields in Settings."""

    def test_default_auth_mode_is_service_account(self) -> None:
        """Default auth mode is service_account (backward compatible)."""
        settings = Settings()
        assert settings.google_drive_auth_mode == "service_account"

    def test_default_oauth_credentials_are_empty(self) -> None:
        """OAuth credential fields default to empty strings."""
        settings = Settings()
        assert settings.google_drive_client_id == ""
        assert settings.google_drive_client_secret == ""
        assert settings.google_drive_refresh_token == ""

    def test_oauth_env_vars_loaded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """OAuth credentials are loaded from environment variables."""
        monkeypatch.setenv("GOOGLE_DRIVE_AUTH_MODE", "oauth")
        monkeypatch.setenv("GOOGLE_DRIVE_CLIENT_ID", "my-client-id.apps.googleusercontent.com")
        monkeypatch.setenv("GOOGLE_DRIVE_CLIENT_SECRET", "my-client-secret")
        monkeypatch.setenv("GOOGLE_DRIVE_REFRESH_TOKEN", "1//my-refresh-token")

        settings = Settings()
        assert settings.google_drive_auth_mode == "oauth"
        assert settings.google_drive_client_id == "my-client-id.apps.googleusercontent.com"
        assert settings.google_drive_client_secret == "my-client-secret"
        assert settings.google_drive_refresh_token == "1//my-refresh-token"

    def test_invalid_auth_mode_raises_validation_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CRITICAL-3: Typo in GOOGLE_DRIVE_AUTH_MODE must raise ValidationError at startup.

        Previously auth_mode was plain str, so typos like 'oAuth' or
        'service-account' silently fell through to the SA path with a
        confusing error. Now it is Literal["service_account", "oauth"] so
        pydantic-settings raises ValidationError immediately.
        """
        monkeypatch.setenv("GOOGLE_DRIVE_AUTH_MODE", "oAuth")  # wrong case
        with pytest.raises(ValidationError):
            Settings()

    def test_another_invalid_auth_mode_raises_validation_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Hyphenated variant 'service-account' must also raise ValidationError."""
        monkeypatch.setenv("GOOGLE_DRIVE_AUTH_MODE", "service-account")
        with pytest.raises(ValidationError):
            Settings()

    def test_service_account_literal_is_valid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """'service_account' (exact) is accepted by the Literal constraint."""
        monkeypatch.setenv("GOOGLE_DRIVE_AUTH_MODE", "service_account")
        settings = Settings()
        assert settings.google_drive_auth_mode == "service_account"

    def test_oauth_literal_is_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """'oauth' (exact) is accepted by the Literal constraint."""
        monkeypatch.setenv("GOOGLE_DRIVE_AUTH_MODE", "oauth")
        settings = Settings()
        assert settings.google_drive_auth_mode == "oauth"

    def test_root_id_description_is_auth_mode_neutral(self) -> None:
        """MINOR-4: google_drive_root_id description must not mention service account."""
        # Access the field metadata through model_fields
        field_info = Settings.model_fields.get("google_drive_root_id")
        assert field_info is not None
        description = field_info.description or ""
        assert "service account" not in description.lower(), (
            f"Description should be auth-mode neutral, got: {description!r}"
        )


class TestValidateGoogleDriveOAuthConfig:
    """Tests for validate_google_drive_config() in OAuth mode."""

    def test_oauth_mode_missing_root_id_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ConfigurationError when oauth mode but root ID is missing."""
        monkeypatch.setenv("GOOGLE_DRIVE_ENABLED", "true")
        monkeypatch.setenv("GOOGLE_DRIVE_AUTH_MODE", "oauth")
        monkeypatch.setenv("GOOGLE_DRIVE_ROOT_ID", "")
        monkeypatch.setenv("GOOGLE_DRIVE_CLIENT_ID", "id")
        monkeypatch.setenv("GOOGLE_DRIVE_CLIENT_SECRET", "secret")
        monkeypatch.setenv("GOOGLE_DRIVE_REFRESH_TOKEN", "token")
        settings = Settings()

        with pytest.raises(ConfigurationError, match="GOOGLE_DRIVE_ROOT_ID"):
            validate_google_drive_config(settings)

    def test_oauth_mode_missing_client_id_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ConfigurationError when oauth mode but GOOGLE_DRIVE_CLIENT_ID is missing."""
        monkeypatch.setenv("GOOGLE_DRIVE_ENABLED", "true")
        monkeypatch.setenv("GOOGLE_DRIVE_AUTH_MODE", "oauth")
        monkeypatch.setenv("GOOGLE_DRIVE_ROOT_ID", "root-id")
        monkeypatch.setenv("GOOGLE_DRIVE_CLIENT_ID", "")
        monkeypatch.setenv("GOOGLE_DRIVE_CLIENT_SECRET", "secret")
        monkeypatch.setenv("GOOGLE_DRIVE_REFRESH_TOKEN", "token")
        settings = Settings()

        with pytest.raises(ConfigurationError, match="GOOGLE_DRIVE_CLIENT_ID"):
            validate_google_drive_config(settings)

    def test_oauth_mode_missing_client_secret_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ConfigurationError when oauth mode but GOOGLE_DRIVE_CLIENT_SECRET is missing."""
        monkeypatch.setenv("GOOGLE_DRIVE_ENABLED", "true")
        monkeypatch.setenv("GOOGLE_DRIVE_AUTH_MODE", "oauth")
        monkeypatch.setenv("GOOGLE_DRIVE_ROOT_ID", "root-id")
        monkeypatch.setenv("GOOGLE_DRIVE_CLIENT_ID", "id")
        monkeypatch.setenv("GOOGLE_DRIVE_CLIENT_SECRET", "")
        monkeypatch.setenv("GOOGLE_DRIVE_REFRESH_TOKEN", "token")
        settings = Settings()

        with pytest.raises(ConfigurationError, match="GOOGLE_DRIVE_CLIENT_SECRET"):
            validate_google_drive_config(settings)

    def test_oauth_mode_missing_refresh_token_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ConfigurationError when oauth mode but GOOGLE_DRIVE_REFRESH_TOKEN is missing."""
        monkeypatch.setenv("GOOGLE_DRIVE_ENABLED", "true")
        monkeypatch.setenv("GOOGLE_DRIVE_AUTH_MODE", "oauth")
        monkeypatch.setenv("GOOGLE_DRIVE_ROOT_ID", "root-id")
        monkeypatch.setenv("GOOGLE_DRIVE_CLIENT_ID", "id")
        monkeypatch.setenv("GOOGLE_DRIVE_CLIENT_SECRET", "secret")
        monkeypatch.setenv("GOOGLE_DRIVE_REFRESH_TOKEN", "")
        settings = Settings()

        with pytest.raises(ConfigurationError, match="GOOGLE_DRIVE_REFRESH_TOKEN"):
            validate_google_drive_config(settings)

    def test_oauth_mode_valid_config_passes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Valid OAuth configuration passes without raising."""
        monkeypatch.setenv("GOOGLE_DRIVE_ENABLED", "true")
        monkeypatch.setenv("GOOGLE_DRIVE_AUTH_MODE", "oauth")
        monkeypatch.setenv("GOOGLE_DRIVE_ROOT_ID", "root-folder-id")
        monkeypatch.setenv("GOOGLE_DRIVE_CLIENT_ID", "id.apps.googleusercontent.com")
        monkeypatch.setenv("GOOGLE_DRIVE_CLIENT_SECRET", "GOCSPX-secret")
        monkeypatch.setenv("GOOGLE_DRIVE_REFRESH_TOKEN", "1//refresh-token-value")
        settings = Settings()

        # Should not raise
        validate_google_drive_config(settings)

    def test_disabled_skips_oauth_validation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No validation when Drive is disabled, regardless of auth mode."""
        monkeypatch.setenv("GOOGLE_DRIVE_ENABLED", "false")
        monkeypatch.setenv("GOOGLE_DRIVE_AUTH_MODE", "oauth")
        # All OAuth fields missing — still no error because Drive is disabled
        settings = Settings()
        validate_google_drive_config(settings)  # Must not raise
