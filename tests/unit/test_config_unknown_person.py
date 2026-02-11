"""Unit tests for unknown person discovery configuration settings.

Tests validation, default values, and environment variable loading for
unknown person discovery settings added to core/config.py.
"""

from typing import Any

import pytest
from pydantic import ValidationError

from image_search_service.core.config import Settings
from image_search_service.services.config_service import ConfigService


class TestUnknownPersonConfigDefaults:
    """Test default values for unknown person discovery settings."""

    def test_unknown_person_min_display_count_default(self) -> None:
        """Test that unknown_person_min_display_count has correct default value."""
        settings = Settings()
        assert settings.unknown_person_min_display_count == 5

    def test_unknown_person_default_threshold_default(self) -> None:
        """Test that unknown_person_default_threshold has correct default value."""
        settings = Settings()
        assert settings.unknown_person_default_threshold == 0.70

    def test_unknown_person_max_faces_default(self) -> None:
        """Test that unknown_person_max_faces has correct default value."""
        settings = Settings()
        assert settings.unknown_person_max_faces == 50000

    def test_unknown_person_chunk_size_default(self) -> None:
        """Test that unknown_person_chunk_size has correct default value."""
        settings = Settings()
        assert settings.unknown_person_chunk_size == 10000


class TestUnknownPersonConfigValidation:
    """Test field validation for unknown person discovery settings.

    Note: Settings uses extra="ignore", so validation only applies when
    loading from environment variables, not constructor arguments.
    """

    def test_min_display_count_must_be_at_least_2(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that min_display_count cannot be less than 2."""
        monkeypatch.setenv("UNKNOWN_PERSON_MIN_DISPLAY_COUNT", "1")
        with pytest.raises(ValidationError) as exc_info:
            Settings()

        errors = exc_info.value.errors()
        # Pydantic uses the alias (env var name) in error location
        assert any(
            error["loc"] == ("UNKNOWN_PERSON_MIN_DISPLAY_COUNT",)
            and "greater than or equal to 2" in str(error["msg"])
            for error in errors
        )

    def test_min_display_count_must_be_at_most_50(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that min_display_count cannot exceed 50."""
        monkeypatch.setenv("UNKNOWN_PERSON_MIN_DISPLAY_COUNT", "51")
        with pytest.raises(ValidationError) as exc_info:
            Settings()

        errors = exc_info.value.errors()
        assert any(
            error["loc"] == ("UNKNOWN_PERSON_MIN_DISPLAY_COUNT",)
            and "less than or equal to 50" in str(error["msg"])
            for error in errors
        )

    def test_default_threshold_must_be_between_0_and_1(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that default_threshold must be in range [0.0, 1.0]."""
        # Test below minimum
        monkeypatch.setenv("UNKNOWN_PERSON_DEFAULT_THRESHOLD", "-0.1")
        with pytest.raises(ValidationError) as exc_info:
            Settings()

        errors = exc_info.value.errors()
        assert any(
            error["loc"] == ("UNKNOWN_PERSON_DEFAULT_THRESHOLD",)
            and "greater than or equal to 0" in str(error["msg"])
            for error in errors
        )

        # Test above maximum (need to clear previous env var)
        monkeypatch.setenv("UNKNOWN_PERSON_DEFAULT_THRESHOLD", "1.1")
        with pytest.raises(ValidationError) as exc_info:
            Settings()

        errors = exc_info.value.errors()
        assert any(
            error["loc"] == ("UNKNOWN_PERSON_DEFAULT_THRESHOLD",)
            and "less than or equal to 1" in str(error["msg"])
            for error in errors
        )

    def test_max_faces_must_be_at_least_100(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that max_faces cannot be less than 100."""
        monkeypatch.setenv("UNKNOWN_PERSON_MAX_FACES", "99")
        with pytest.raises(ValidationError) as exc_info:
            Settings()

        errors = exc_info.value.errors()
        assert any(
            error["loc"] == ("UNKNOWN_PERSON_MAX_FACES",)
            and "greater than or equal to 100" in str(error["msg"])
            for error in errors
        )

    def test_max_faces_must_be_at_most_100000(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that max_faces cannot exceed 100000."""
        monkeypatch.setenv("UNKNOWN_PERSON_MAX_FACES", "100001")
        with pytest.raises(ValidationError) as exc_info:
            Settings()

        errors = exc_info.value.errors()
        assert any(
            error["loc"] == ("UNKNOWN_PERSON_MAX_FACES",)
            and "less than or equal to 100000" in str(error["msg"])
            for error in errors
        )

    def test_chunk_size_must_be_at_least_1000(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that chunk_size cannot be less than 1000."""
        monkeypatch.setenv("UNKNOWN_PERSON_CHUNK_SIZE", "999")
        with pytest.raises(ValidationError) as exc_info:
            Settings()

        errors = exc_info.value.errors()
        assert any(
            error["loc"] == ("UNKNOWN_PERSON_CHUNK_SIZE",)
            and "greater than or equal to 1000" in str(error["msg"])
            for error in errors
        )

    def test_chunk_size_must_be_at_most_20000(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that chunk_size cannot exceed 20000."""
        monkeypatch.setenv("UNKNOWN_PERSON_CHUNK_SIZE", "20001")
        with pytest.raises(ValidationError) as exc_info:
            Settings()

        errors = exc_info.value.errors()
        assert any(
            error["loc"] == ("UNKNOWN_PERSON_CHUNK_SIZE",)
            and "less than or equal to 20000" in str(error["msg"])
            for error in errors
        )


class TestUnknownPersonConfigEnvironmentVariables:
    """Test that settings can be loaded from environment variables."""

    def test_min_display_count_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test loading unknown_person_min_display_count from environment."""
        monkeypatch.setenv("UNKNOWN_PERSON_MIN_DISPLAY_COUNT", "10")
        settings = Settings()
        assert settings.unknown_person_min_display_count == 10

    def test_default_threshold_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test loading unknown_person_default_threshold from environment."""
        monkeypatch.setenv("UNKNOWN_PERSON_DEFAULT_THRESHOLD", "0.80")
        settings = Settings()
        assert settings.unknown_person_default_threshold == 0.80

    def test_max_faces_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test loading unknown_person_max_faces from environment."""
        monkeypatch.setenv("UNKNOWN_PERSON_MAX_FACES", "25000")
        settings = Settings()
        assert settings.unknown_person_max_faces == 25000

    def test_chunk_size_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test loading unknown_person_chunk_size from environment."""
        monkeypatch.setenv("UNKNOWN_PERSON_CHUNK_SIZE", "5000")
        settings = Settings()
        assert settings.unknown_person_chunk_size == 5000

    def test_all_unknown_person_settings_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test loading all unknown person settings from environment at once."""
        monkeypatch.setenv("UNKNOWN_PERSON_MIN_DISPLAY_COUNT", "15")
        monkeypatch.setenv("UNKNOWN_PERSON_DEFAULT_THRESHOLD", "0.75")
        monkeypatch.setenv("UNKNOWN_PERSON_MAX_FACES", "30000")
        monkeypatch.setenv("UNKNOWN_PERSON_CHUNK_SIZE", "15000")

        settings = Settings()
        assert settings.unknown_person_min_display_count == 15
        assert settings.unknown_person_default_threshold == 0.75
        assert settings.unknown_person_max_faces == 30000
        assert settings.unknown_person_chunk_size == 15000


class TestUnknownPersonConfigServiceDefaults:
    """Test that ConfigService.DEFAULTS contains unknown person settings."""

    def test_all_unknown_person_keys_in_defaults(self) -> None:
        """Test that all unknown person config keys are in ConfigService.DEFAULTS."""
        required_keys = {
            "unknown_person_min_display_count",
            "unknown_person_default_threshold",
            "unknown_person_max_faces",
            "unknown_person_chunk_size",
        }

        defaults_keys = set(ConfigService.DEFAULTS.keys())
        missing_keys = required_keys - defaults_keys

        assert not missing_keys, (
            f"Missing unknown person config keys in ConfigService.DEFAULTS: {missing_keys}. "
            "These keys are required for database-backed configuration."
        )

    def test_unknown_person_defaults_have_correct_types(self) -> None:
        """Test that unknown person DEFAULTS values have correct types."""
        expected_types: dict[str, type[Any]] = {
            "unknown_person_min_display_count": int,
            "unknown_person_default_threshold": float,
            "unknown_person_max_faces": int,
            "unknown_person_chunk_size": int,
        }

        for key, expected_type in expected_types.items():
            value = ConfigService.DEFAULTS.get(key)
            assert value is not None, f"Key '{key}' missing from ConfigService.DEFAULTS"
            assert isinstance(value, expected_type), (
                f"DEFAULTS['{key}'] has type {type(value).__name__}, "
                f"expected {expected_type.__name__}"
            )

    def test_unknown_person_defaults_values_in_valid_ranges(self) -> None:
        """Test that DEFAULTS values are within documented constraint ranges."""
        validations = [
            # (key, min_value, max_value)
            ("unknown_person_min_display_count", 2, 50),
            ("unknown_person_default_threshold", 0.0, 1.0),
            ("unknown_person_max_faces", 100, 100000),
            ("unknown_person_chunk_size", 1000, 20000),
        ]

        for key, min_val, max_val in validations:
            value = ConfigService.DEFAULTS.get(key)
            assert value is not None, f"Key '{key}' missing from ConfigService.DEFAULTS"

            # Convert to float for comparison (handles both int and float)
            numeric_value = float(value)

            assert min_val <= numeric_value <= max_val, (
                f"DEFAULTS['{key}'] = {value} is outside valid range "
                f"[{min_val}, {max_val}]. Update DEFAULTS or migration constraints."
            )


class TestUnknownPersonConfigBusinessLogic:
    """Test business logic constraints for unknown person settings."""

    def test_chunk_size_less_than_or_equal_to_max_faces(self) -> None:
        """Test that chunk_size should be <= max_faces for efficient processing."""
        chunk_size = ConfigService.DEFAULTS.get("unknown_person_chunk_size")
        max_faces = ConfigService.DEFAULTS.get("unknown_person_max_faces")

        assert chunk_size is not None and max_faces is not None, (
            "Missing required keys in ConfigService.DEFAULTS"
        )

        # Type narrowing: we know these are ints from previous tests
        assert isinstance(chunk_size, int) and isinstance(max_faces, int), (
            "Chunk size and max faces must be integers"
        )

        assert chunk_size <= max_faces, (
            f"DEFAULTS violates business constraint: "
            f"unknown_person_chunk_size ({chunk_size}) should be <= "
            f"unknown_person_max_faces ({max_faces})"
        )

    def test_default_threshold_matches_cluster_confidence(self) -> None:
        """Test that default_threshold matches existing cluster confidence setting.

        This ensures consistency across unknown face clustering and unknown person
        discovery features.
        """
        # Get the new unknown person threshold
        person_threshold = ConfigService.DEFAULTS.get("unknown_person_default_threshold")

        # Note: The existing unknown_face_cluster_min_confidence is in core/config.py
        # but not in ConfigService.DEFAULTS. This test documents the expected alignment.
        settings = Settings()
        cluster_confidence = settings.unknown_face_cluster_min_confidence

        assert person_threshold is not None, (
            "unknown_person_default_threshold missing from ConfigService.DEFAULTS"
        )

        assert isinstance(person_threshold, (int, float)), (
            "unknown_person_default_threshold must be numeric"
        )

        # These should be aligned for consistent user experience
        assert float(person_threshold) == cluster_confidence, (
            f"Inconsistent thresholds: "
            f"unknown_person_default_threshold ({person_threshold}) should match "
            f"unknown_face_cluster_min_confidence ({cluster_confidence})"
        )
