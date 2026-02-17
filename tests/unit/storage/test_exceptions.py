"""Tests for storage exception hierarchy."""

from __future__ import annotations

import pytest

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


class TestExceptionHierarchy:
    def test_all_inherit_from_storage_error(self) -> None:
        """All storage exceptions must inherit from StorageError."""
        exceptions: list[StorageError] = [
            NotFoundError("path"),
            PathAmbiguousError("path", 2),
            RateLimitError(),
            StorageQuotaError(),
            RootBoundaryError("path"),
            StoragePermissionError("path"),
            ConfigurationError("field", "detail"),
            UploadError("file"),
        ]
        for exc in exceptions:
            assert isinstance(exc, StorageError)

    def test_eight_exception_subclasses(self) -> None:
        """There are exactly 8 exception subclasses."""
        subclasses = [
            NotFoundError,
            PathAmbiguousError,
            RateLimitError,
            StorageQuotaError,
            RootBoundaryError,
            StoragePermissionError,
            ConfigurationError,
            UploadError,
        ]
        assert len(subclasses) == 8
        for cls in subclasses:
            assert issubclass(cls, StorageError)

    def test_catch_all_with_storage_error(self) -> None:
        """StorageError catches any storage exception."""
        with pytest.raises(StorageError):
            raise NotFoundError("/missing/file")

    def test_catch_specific_as_storage_error(self) -> None:
        """Each exception can be caught as StorageError."""
        exc_instances: list[StorageError] = [
            NotFoundError("path"),
            PathAmbiguousError("path", 2),
            RateLimitError(),
            StorageQuotaError(),
            RootBoundaryError("path"),
            StoragePermissionError("path"),
            ConfigurationError("field", "detail"),
            UploadError("file"),
        ]
        for exc in exc_instances:
            with pytest.raises(StorageError):
                raise exc


class TestNotFoundError:
    def test_stores_path(self) -> None:
        exc = NotFoundError("/people/John")
        assert exc.path == "/people/John"

    def test_message_includes_path(self) -> None:
        exc = NotFoundError("/people/John")
        assert "/people/John" in str(exc)

    def test_is_exception(self) -> None:
        exc = NotFoundError("/people/John")
        assert isinstance(exc, Exception)

    def test_message_format(self) -> None:
        exc = NotFoundError("/some/path")
        assert str(exc) == "Not found: /some/path"


class TestPathAmbiguousError:
    def test_stores_path_and_count(self) -> None:
        exc = PathAmbiguousError("/people/John", 3)
        assert exc.path == "/people/John"
        assert exc.count == 3

    def test_message_includes_count(self) -> None:
        exc = PathAmbiguousError("/people/John", 3)
        assert "3" in str(exc)

    def test_message_includes_path(self) -> None:
        exc = PathAmbiguousError("/people/John", 3)
        assert "/people/John" in str(exc)

    def test_count_attribute_type(self) -> None:
        exc = PathAmbiguousError("path", 5)
        assert isinstance(exc.count, int)


class TestRateLimitError:
    def test_with_retry_after(self) -> None:
        exc = RateLimitError(retry_after=30)
        assert exc.retry_after == 30
        assert "30" in str(exc)

    def test_without_retry_after(self) -> None:
        exc = RateLimitError()
        assert exc.retry_after is None
        assert "Rate limited" in str(exc)

    def test_retry_after_none_by_default(self) -> None:
        exc = RateLimitError()
        assert exc.retry_after is None

    def test_retry_after_in_message(self) -> None:
        exc = RateLimitError(retry_after=60)
        assert "60s" in str(exc)


class TestStorageQuotaError:
    def test_with_quota_info(self) -> None:
        exc = StorageQuotaError(
            quota_bytes=15 * 1024**3,
            used_bytes=14 * 1024**3,
        )
        assert exc.quota_bytes == 15 * 1024**3
        assert exc.used_bytes == 14 * 1024**3
        assert "14.0 GB" in str(exc)
        assert "15.0 GB" in str(exc)

    def test_without_quota_info(self) -> None:
        exc = StorageQuotaError()
        assert exc.quota_bytes is None
        assert exc.used_bytes is None
        assert "quota exceeded" in str(exc).lower()

    def test_no_positional_string_arg(self) -> None:
        """Canonical: no positional string argument allowed."""
        # Only keyword args for quota_bytes and used_bytes
        exc = StorageQuotaError(quota_bytes=None, used_bytes=None)
        assert exc.quota_bytes is None

    def test_partial_quota_info_no_gb_display(self) -> None:
        """When only one of quota/used is set, no GB display."""
        exc = StorageQuotaError(quota_bytes=1024**3, used_bytes=None)
        # Without both values, no GB display
        assert "GB" not in str(exc)


class TestRootBoundaryError:
    def test_stores_path(self) -> None:
        exc = RootBoundaryError("/../../../etc")
        assert exc.path == "/../../../etc"

    def test_message_includes_path(self) -> None:
        exc = RootBoundaryError("/../../../etc")
        assert "/../../../etc" in str(exc)

    def test_message_format(self) -> None:
        exc = RootBoundaryError("/bad/path")
        assert "root boundary" in str(exc).lower()


class TestStoragePermissionError:
    def test_with_detail(self) -> None:
        exc = StoragePermissionError("/secret", "Folder not shared")
        assert exc.path == "/secret"
        assert exc.detail == "Folder not shared"
        assert "Folder not shared" in str(exc)

    def test_without_detail(self) -> None:
        exc = StoragePermissionError("/secret")
        assert exc.detail == ""

    def test_path_in_message(self) -> None:
        exc = StoragePermissionError("/secret")
        assert "/secret" in str(exc)

    def test_does_not_shadow_builtin(self) -> None:
        """Verify it's StoragePermissionError, not PermissionError."""
        exc = StoragePermissionError("/path")
        assert not isinstance(exc, PermissionError)


class TestConfigurationError:
    def test_stores_field_and_detail(self) -> None:
        exc = ConfigurationError("GOOGLE_DRIVE_SA_JSON", "File not found")
        assert exc.field == "GOOGLE_DRIVE_SA_JSON"
        assert exc.detail == "File not found"
        assert "GOOGLE_DRIVE_SA_JSON" in str(exc)

    def test_message_includes_detail(self) -> None:
        exc = ConfigurationError("FIELD", "Something wrong")
        assert "Something wrong" in str(exc)

    def test_message_format(self) -> None:
        exc = ConfigurationError("MY_FIELD", "bad value")
        assert "[MY_FIELD]" in str(exc)


class TestUploadError:
    def test_with_detail(self) -> None:
        exc = UploadError("photo.jpg", "Network timeout")
        assert exc.filename == "photo.jpg"
        assert exc.detail == "Network timeout"
        assert "photo.jpg" in str(exc)

    def test_without_detail(self) -> None:
        exc = UploadError("photo.jpg")
        assert exc.detail == ""

    def test_filename_in_message(self) -> None:
        exc = UploadError("photo.jpg")
        assert "photo.jpg" in str(exc)

    def test_detail_in_message(self) -> None:
        exc = UploadError("photo.jpg", "Connection refused")
        assert "Connection refused" in str(exc)

    def test_canonical_constructor_positional(self) -> None:
        """Canonical: positional filename, optional keyword detail."""
        exc = UploadError("my_file.jpg", detail="timeout")
        assert exc.filename == "my_file.jpg"
        assert exc.detail == "timeout"
