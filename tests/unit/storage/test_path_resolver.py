"""Tests for path normalization and PathResolver cache."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from image_search_service.storage.exceptions import RootBoundaryError
from image_search_service.storage.path_resolver import (
    LookupFn,
    PathResolver,
    normalize_path,
    sanitize_folder_name,
)


class TestNormalizePath:
    """Test path normalization rules."""

    @pytest.mark.parametrize(
        "input_path,expected",
        [
            pytest.param("/", "/", id="root-unchanged"),
            pytest.param("", "/", id="empty-becomes-root"),
            pytest.param("a/b", "/a/b", id="add-leading-slash"),
            pytest.param("/a/b/", "/a/b", id="remove-trailing-slash"),
            pytest.param("//a///b", "/a/b", id="collapse-multiple-slashes"),
            pytest.param("/a/b/c/d", "/a/b/c/d", id="deep-path"),
            pytest.param("photos", "/photos", id="single-segment"),
            pytest.param("/a/file..name", "/a/file..name", id="double-dot-in-filename"),
            pytest.param("   ", "/", id="whitespace-only-becomes-root"),
            pytest.param("/John Doe/photos", "/John Doe/photos", id="spaces-in-segment"),
        ],
    )
    def test_normalize_path(self, input_path: str, expected: str) -> None:
        """Should normalize paths according to rules."""
        assert normalize_path(input_path) == expected

    @pytest.mark.parametrize(
        "input_path",
        [
            pytest.param("/a/../b", id="parent-traversal"),
            pytest.param("/..", id="root-traversal"),
            pytest.param("/../../../etc/passwd", id="deep-traversal"),
        ],
    )
    def test_reject_traversal(self, input_path: str) -> None:
        """Should reject path traversal attempts with RootBoundaryError."""
        with pytest.raises(RootBoundaryError):
            normalize_path(input_path)

    def test_whitespace_only_segment_in_path_raises(self) -> None:
        """Path like /a/   /b should raise ValueError."""
        with pytest.raises(ValueError, match="whitespace-only"):
            normalize_path("/a/   /b")


class TestSanitizeFolderName:
    @pytest.mark.parametrize(
        "input_name,expected",
        [
            pytest.param("John Doe", "John Doe", id="normal-name"),
            pytest.param("John/Doe", "John_Doe", id="replace-forward-slash"),
            pytest.param("John\\Doe", "John_Doe", id="replace-backslash"),
            pytest.param('John<>:"|?*Doe', "JohnDoe", id="remove-special-chars"),
            pytest.param("John   Doe", "John Doe", id="collapse-whitespace"),
            pytest.param("  John Doe  ", "John Doe", id="strip-whitespace"),
            pytest.param("José García", "José García", id="unicode-preserved"),
        ],
    )
    def test_sanitize_folder_name(self, input_name: str, expected: str) -> None:
        """Should sanitize folder names according to rules."""
        assert sanitize_folder_name(input_name) == expected

    def test_truncate_long_name(self) -> None:
        long_name = "A" * 300
        result = sanitize_folder_name(long_name)
        assert len(result) == 255

    @pytest.mark.parametrize(
        "input_name",
        [
            pytest.param('<>:"|?*', id="empty-after-sanitize"),
            pytest.param("", id="empty-string"),
        ],
    )
    def test_empty_raises(self, input_name: str) -> None:
        """Should raise ValueError for names that are or become empty."""
        with pytest.raises(ValueError):
            sanitize_folder_name(input_name)


class TestPathResolver:
    """Test PathResolver cache behavior."""

    def test_resolve_root(self) -> None:
        resolver = PathResolver(root_folder_id="root123")
        assert resolver.resolve("/") == "root123"

    def test_root_folder_id_property(self) -> None:
        resolver = PathResolver(root_folder_id="root123")
        assert resolver.root_folder_id == "root123"

    def test_cache_size_starts_empty(self) -> None:
        resolver = PathResolver(root_folder_id="root123")
        assert resolver.cache_size == 0

    def test_resolve_uses_lookup_fn(self) -> None:
        lookup: MagicMock = MagicMock(return_value=("child456", True))
        resolver = PathResolver(
            root_folder_id="root123",
            lookup_fn=lookup,
        )
        result = resolver.resolve("/people")
        assert result == "child456"
        lookup.assert_called_once_with("root123", "people")

    def test_cache_hit_avoids_lookup(self) -> None:
        lookup: MagicMock = MagicMock(return_value=("child456", True))
        resolver = PathResolver(
            root_folder_id="root123",
            lookup_fn=lookup,
        )
        # First call: cache miss
        resolver.resolve("/people")
        # Second call: cache hit
        resolver.resolve("/people")
        # Lookup should only be called once
        assert lookup.call_count == 1

    def test_multi_segment_path(self) -> None:
        def lookup(parent_id: str, name: str) -> tuple[str, bool]:
            mapping: dict[tuple[str, str], tuple[str, bool]] = {
                ("root123", "people"): ("folder1", True),
                ("folder1", "John"): ("folder2", True),
                ("folder2", "photo.jpg"): ("file3", False),
            }
            return mapping[(parent_id, name)]

        resolver = PathResolver(
            root_folder_id="root123",
            lookup_fn=lookup,
        )
        result = resolver.resolve("/people/John/photo.jpg")
        assert result == "file3"

    def test_cache_ttl_expiry(self) -> None:
        """Cache entries expire after TTL using deterministic time mocking."""
        lookup: MagicMock = MagicMock(return_value=("child456", True))
        with patch("image_search_service.storage.path_resolver.monotonic") as mock_time:
            mock_time.return_value = 100.0
            resolver = PathResolver(
                root_folder_id="root123",
                lookup_fn=lookup,
                cache_ttl=60,
            )
            resolver.resolve("/people")
            assert lookup.call_count == 1

            # Same time -- should be cache hit
            resolver.resolve("/people")
            assert lookup.call_count == 1

            # Jump past TTL
            mock_time.return_value = 161.0
            resolver.resolve("/people")
            assert lookup.call_count == 2  # Re-fetched after expiry

    def test_cache_maxsize_eviction(self) -> None:
        call_count = 0

        def lookup(parent_id: str, name: str) -> tuple[str, bool]:
            nonlocal call_count
            call_count += 1
            return (f"id_{name}", True)

        resolver = PathResolver(
            root_folder_id="root123",
            lookup_fn=lookup,
            cache_maxsize=2,
        )
        # Fill cache to max (2 entries)
        resolver.resolve("/a")
        resolver.resolve("/b")
        assert resolver.cache_size == 2

        # Add third entry, should evict oldest (a)
        resolver.resolve("/c")
        assert resolver.cache_size == 2

    def test_invalidate_parent(self) -> None:
        lookup: MagicMock = MagicMock(return_value=("child456", True))
        resolver = PathResolver(
            root_folder_id="root123",
            lookup_fn=lookup,
        )
        resolver.resolve("/people")
        assert lookup.call_count == 1

        resolver.invalidate_parent("root123")

        resolver.resolve("/people")
        assert lookup.call_count == 2  # Re-fetched after invalidation

    def test_invalidate_all(self) -> None:
        lookup: MagicMock = MagicMock(return_value=("child456", True))
        resolver = PathResolver(
            root_folder_id="root123",
            lookup_fn=lookup,
        )
        resolver.resolve("/people")
        resolver.invalidate_all()
        assert resolver.cache_size == 0

    def test_cache_put_manual_insert(self) -> None:
        resolver = PathResolver(root_folder_id="root123")
        resolver.cache_put("root123", "people", "folder456", True)
        assert resolver.cache_size == 1
        # Resolve should use cached value (no lookup_fn needed)
        result = resolver.resolve("/people")
        assert result == "folder456"

    def test_no_lookup_fn_raises(self) -> None:
        resolver = PathResolver(root_folder_id="root123")
        with pytest.raises(NotImplementedError):
            resolver.resolve("/people")

    def test_thread_safety(self) -> None:
        """Verify concurrent access doesn't corrupt cache."""
        lookup: MagicMock = MagicMock(side_effect=lambda p, n: (f"id_{n}", True))
        resolver = PathResolver(
            root_folder_id="root123",
            lookup_fn=lookup,
            cache_maxsize=100,
        )
        errors: list[Exception] = []

        def worker(name: str) -> None:
            try:
                for _ in range(50):
                    resolver.resolve(f"/{name}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(f"folder_{i}",)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert resolver.cache_size <= 100  # maxsize respected

    def test_invalidate_parent_only_removes_children(self) -> None:
        """invalidate_parent removes only children of specified parent."""

        def lookup(parent_id: str, name: str) -> tuple[str, bool]:
            return (f"id_{parent_id}_{name}", True)

        resolver = PathResolver(root_folder_id="root", lookup_fn=lookup)
        # Populate two different parent's children
        resolver.resolve("/a")  # (root, a) -> id_root_a
        resolver.cache_put("other_parent", "b", "id_other_b", True)
        assert resolver.cache_size == 2

        # Invalidate root's children
        resolver.invalidate_parent("root")
        assert resolver.cache_size == 1  # other_parent's entry remains

    def test_cache_put_respects_maxsize(self) -> None:
        """cache_put evicts oldest when at maxsize."""
        resolver = PathResolver(root_folder_id="root", cache_maxsize=2)
        resolver.cache_put("root", "a", "id_a", True)
        resolver.cache_put("root", "b", "id_b", True)
        assert resolver.cache_size == 2

        resolver.cache_put("root", "c", "id_c", True)
        assert resolver.cache_size == 2  # Oldest evicted

    def test_lookup_fn_type_alias(self) -> None:
        """LookupFn type alias is exported and usable."""

        def my_lookup(parent_id: str, child_name: str) -> tuple[str, bool]:
            return ("some_id", True)

        # This should be assignable to LookupFn
        fn: LookupFn = my_lookup
        assert callable(fn)

    def test_multi_segment_resolution_from_fixture(
        self, resolver_with_lookup: PathResolver
    ) -> None:
        """Verify full three-level path resolution using shared fixture."""
        result = resolver_with_lookup.resolve("/people/John Doe/photo.jpg")
        assert result == "file_photo"
