"""Unit tests for storage factory functions.

Tests get_storage() and get_async_storage() behavior including:
- Feature toggle (returns None when disabled)
- Configuration validation errors
- Singleton behavior via @lru_cache
- AsyncStorageWrapper wrapping
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from image_search_service.storage import get_async_storage, get_storage
from image_search_service.storage.exceptions import ConfigurationError, StorageError


def _make_settings(
    *,
    enabled: bool = False,
    sa_json: str = "",
    root_id: str = "",
    cache_maxsize: int = 1024,
    cache_ttl: int = 300,
) -> MagicMock:
    """Create a mock Settings object for testing."""
    settings = MagicMock()
    settings.google_drive_enabled = enabled
    settings.google_drive_sa_json = sa_json
    settings.google_drive_root_id = root_id
    settings.google_drive_path_cache_maxsize = cache_maxsize
    settings.google_drive_path_cache_ttl = cache_ttl
    return settings


# ─── TestGetStorage ────────────────────────────────────────────────────────────


class TestGetStorage:
    """Tests for get_storage() factory function."""

    def setup_method(self) -> None:
        """Clear lru_cache between tests to ensure isolation."""
        get_storage.cache_clear()
        # get_async_storage has no @lru_cache (per INTERFACES.md §8)
        # Also clear get_settings cache if it's an lru_cache singleton,
        # to prevent cross-test pollution from patched settings.
        try:
            from image_search_service.core.config import get_settings as _gs

            if hasattr(_gs, "cache_clear"):
                _gs.cache_clear()
        except Exception:
            pass

    def test_returns_none_when_disabled(self) -> None:
        """get_storage returns None when google_drive_enabled=False."""
        # Patch get_settings in the core.config module (where it's defined)
        # and inside the storage factory which imports it lazily.
        with patch(
            "image_search_service.core.config.get_settings",
            return_value=_make_settings(enabled=False),
        ):
            result = get_storage()
        assert result is None

    def test_raises_when_enabled_but_missing_sa_json(self) -> None:
        """get_storage raises ConfigurationError when SA JSON path not set."""
        with patch(
            "image_search_service.core.config.get_settings",
            return_value=_make_settings(enabled=True, sa_json="", root_id="root"),
        ):
            with pytest.raises(ConfigurationError, match="GOOGLE_DRIVE_SA_JSON"):
                get_storage()

    def test_raises_when_enabled_but_missing_root_id(self) -> None:
        """get_storage raises ConfigurationError when root folder ID not set."""
        with patch(
            "image_search_service.core.config.get_settings",
            return_value=_make_settings(
                enabled=True, sa_json="/fake/sa.json", root_id=""
            ),
        ):
            with pytest.raises(ConfigurationError, match="GOOGLE_DRIVE_ROOT_ID"):
                get_storage()

    def test_returns_drive_storage_when_properly_configured(self) -> None:
        """get_storage returns GoogleDriveV3Storage when properly configured."""
        from image_search_service.storage.google_drive import GoogleDriveV3Storage

        with patch(
            "image_search_service.core.config.get_settings",
            return_value=_make_settings(
                enabled=True, sa_json="/fake/sa.json", root_id="root-id"
            ),
        ):
            result = get_storage()

        # Result should be a GoogleDriveV3Storage (lazy init, no API calls yet)
        assert result is not None
        assert isinstance(result, GoogleDriveV3Storage)
        assert result._sa_json_path == "/fake/sa.json"
        assert result._root_folder_id == "root-id"

    def test_singleton_behavior_when_disabled(self) -> None:
        """get_storage returns same instance on repeated calls (lru_cache)."""
        mock_settings = _make_settings(enabled=False)
        with patch(
            "image_search_service.core.config.get_settings",
            return_value=mock_settings,
        ):
            result1 = get_storage()
            result2 = get_storage()

        assert result1 is result2
        assert result1 is None

    def test_cache_clear_allows_fresh_call(self) -> None:
        """cache_clear() resets the singleton so next call re-evaluates."""
        mock_settings = _make_settings(enabled=False)

        with patch(
            "image_search_service.core.config.get_settings",
            return_value=mock_settings,
        ) as mock_get_settings:
            get_storage()  # First call (cached)
            get_storage()  # Second call (from cache)

        # get_settings was only called once (cached by lru_cache)
        assert mock_get_settings.call_count == 1

        get_storage.cache_clear()

        with patch(
            "image_search_service.core.config.get_settings",
            return_value=mock_settings,
        ) as mock_get_settings2:
            get_storage()  # Fresh call after cache_clear

        # Called again after cache_clear
        assert mock_get_settings2.call_count == 1


# ─── TestGetAsyncStorage ───────────────────────────────────────────────────────


class TestGetAsyncStorage:
    """Tests for get_async_storage() factory function."""

    def setup_method(self) -> None:
        """Clear lru_cache between tests to ensure isolation."""
        get_storage.cache_clear()
        # get_async_storage has no @lru_cache (per INTERFACES.md §8)
        # Also clear get_settings cache if it's an lru_cache singleton,
        # to prevent cross-test pollution from patched settings.
        try:
            from image_search_service.core.config import get_settings as _gs

            if hasattr(_gs, "cache_clear"):
                _gs.cache_clear()
        except Exception:
            pass

    def test_returns_none_when_sync_storage_none(self) -> None:
        """get_async_storage returns None when get_storage returns None."""
        with patch(
            "image_search_service.core.config.get_settings",
            return_value=_make_settings(enabled=False),
        ):
            result = get_async_storage()
        assert result is None

    def test_wraps_sync_storage_when_enabled(self) -> None:
        """get_async_storage wraps get_storage result in AsyncStorageWrapper."""
        from image_search_service.storage.async_wrapper import AsyncStorageWrapper

        mock_sync = MagicMock()

        with patch("image_search_service.storage.get_storage", return_value=mock_sync):
            result = get_async_storage()

        assert result is not None
        assert isinstance(result, AsyncStorageWrapper)
        assert result._storage is mock_sync

    def test_returns_none_when_storage_disabled(self) -> None:
        """get_async_storage returns None when Drive is disabled."""
        with patch("image_search_service.storage.get_storage", return_value=None):
            result = get_async_storage()
        assert result is None

    def test_creates_new_wrapper_each_call(self) -> None:
        """get_async_storage creates a new wrapper each call (NOT cached).

        Per INTERFACES.md §8: get_async_storage() is NOT cached — it creates
        a new AsyncStorageWrapper each time, but wraps the same cached sync
        backend from get_storage().
        """
        from image_search_service.storage.async_wrapper import AsyncStorageWrapper

        mock_sync = MagicMock()

        with patch("image_search_service.storage.get_storage", return_value=mock_sync):
            result1 = get_async_storage()
            result2 = get_async_storage()

        # Both are valid wrappers but are distinct instances
        assert result1 is not None
        assert result2 is not None
        assert isinstance(result1, AsyncStorageWrapper)
        assert isinstance(result2, AsyncStorageWrapper)
        # Both wrap the same underlying sync storage (the cached singleton)
        assert result1._storage is mock_sync
        assert result2._storage is mock_sync

    def test_raises_if_sync_storage_raises(self) -> None:
        """get_async_storage propagates StorageError from get_storage."""
        with patch(
            "image_search_service.storage.get_storage",
            side_effect=StorageError("misconfigured"),
        ):
            with pytest.raises(StorageError, match="misconfigured"):
                get_async_storage()


# ─── TestFactoryIntegration ───────────────────────────────────────────────────


class TestFactoryIntegration:
    """Integration tests for factory functions together."""

    def setup_method(self) -> None:
        """Clear lru_cache between tests."""
        get_storage.cache_clear()
        # get_async_storage has no @lru_cache (per INTERFACES.md §8)
        try:
            from image_search_service.core.config import get_settings as _gs

            if hasattr(_gs, "cache_clear"):
                _gs.cache_clear()
        except Exception:
            pass

    def test_disabled_drive_gives_none_for_both(self) -> None:
        """Both factories return None when Drive is disabled."""
        with patch(
            "image_search_service.core.config.get_settings",
            return_value=_make_settings(enabled=False),
        ):
            sync_result = get_storage()
            async_result = get_async_storage()

        assert sync_result is None
        assert async_result is None

    def test_missing_sa_json_raises_for_sync(self) -> None:
        """StorageError from missing SA JSON raised by sync factory."""
        with patch(
            "image_search_service.core.config.get_settings",
            return_value=_make_settings(enabled=True, sa_json="", root_id="root"),
        ):
            with pytest.raises(StorageError):
                get_storage()

    def test_missing_root_id_raises_for_sync(self) -> None:
        """StorageError from missing root ID raised by sync factory."""
        with patch(
            "image_search_service.core.config.get_settings",
            return_value=_make_settings(
                enabled=True, sa_json="/fake/sa.json", root_id=""
            ),
        ):
            with pytest.raises(StorageError):
                get_storage()
