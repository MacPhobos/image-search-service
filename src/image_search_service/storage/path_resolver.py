"""Path normalization and ID resolution for cloud storage.

Google Drive (and other cloud providers) use opaque IDs instead of
filesystem-style paths. This module provides:

1. normalize_path(): Pure function for path normalization
2. sanitize_folder_name(): Pure function for sanitizing folder names
3. PathResolver: Thread-safe LRU cache for path-to-ID resolution

Path Resolution Example:
    Resolving "/people/John Doe/photo.jpg" requires:
    1. Start at root folder ID (configured)
    2. Query children of root named "people" -> folder ID
    3. Query children of "people" named "John Doe" -> folder ID
    4. Query children of "John Doe" named "photo.jpg" -> file ID

    Without caching: 100 photos to same person = 300+ API calls
    With caching: 100 photos to same person = ~3 API calls (folder cached)
"""

from __future__ import annotations

import re
import threading
from collections import OrderedDict
from collections.abc import Callable
from time import monotonic

from image_search_service.storage.exceptions import (
    NotFoundError,
    PathAmbiguousError,
    RootBoundaryError,
)

# Type alias for the lookup function signature.
# Callable[[parent_id, child_name], (child_id, is_dir)]
# Must be defined BEFORE PathResolver to use in type annotations.
LookupFn = Callable[[str, str], tuple[str, bool]]


def normalize_path(path: str) -> str:
    """Normalize a virtual storage path.

    Rules:
    - Collapse multiple slashes: "//a///b" -> "/a/b"
    - Remove trailing slash: "/a/b/" -> "/a/b"
    - Ensure leading slash: "a/b" -> "/a/b"
    - Reject ".." segments: raises RootBoundaryError
    - "/" remains as root
    - Empty string becomes "/"
    - Whitespace-only segments are rejected

    Args:
        path: Raw path string.

    Returns:
        Normalized path string.

    Raises:
        RootBoundaryError: If path contains ".." segments.
        ValueError: If path contains whitespace-only segments.
    """
    # Handle empty/whitespace
    if not path or not path.strip():
        return "/"

    # Split on slashes, filter empty segments from multiple slashes
    segments = [s for s in path.split("/") if s]

    # Check for parent traversal
    if ".." in segments:
        raise RootBoundaryError(path)

    # Check for whitespace-only segments
    for seg in segments:
        if not seg.strip():
            raise ValueError(f"Path contains whitespace-only segment: '{path}'")

    # Rebuild normalized path
    if not segments:
        return "/"

    return "/" + "/".join(segments)


def sanitize_folder_name(name: str) -> str:
    """Sanitize a person name for use as a Drive folder name.

    Rules:
    - Replace path separators with underscores
    - Remove characters problematic in Drive names
    - Collapse whitespace
    - Strip leading/trailing whitespace
    - Truncate to 255 characters

    Args:
        name: Raw folder name (e.g., person name).

    Returns:
        Sanitized folder name safe for cloud storage.

    Raises:
        ValueError: If name is empty after sanitization.
    """
    # Replace path separators
    name = name.replace("/", "_").replace("\\", "_")

    # Remove problematic characters
    name = re.sub(r'[<>:"|?*]', "", name)

    # Collapse whitespace
    name = re.sub(r"\s+", " ", name).strip()

    # Truncate
    name = name[:255]

    if not name:
        raise ValueError("Folder name is empty after sanitization")

    return name


class PathResolver:
    """Thread-safe LRU cache for path-to-ID resolution.

    Resolves virtual paths (e.g., "/people/John") to provider-specific
    IDs by walking path segments from the root folder.

    The cache uses (parent_id, child_name) as keys, which matches
    Google Drive's hierarchy model. Each cache entry stores:
    - child_id: The resolved ID
    - is_dir: Whether the child is a folder
    - timestamp: When the entry was cached (for TTL expiry)

    Thread Safety:
        All cache operations are protected by threading.Lock.
        Required because AsyncStorageWrapper delegates to a
        ThreadPoolExecutor with multiple workers.

    Cache Invalidation:
        - invalidate_parent(parent_id): Clear all children of a folder
          (call after create/delete/rename in that folder)
        - invalidate_all(): Clear entire cache
          (call on bulk operations or error recovery)

    Usage:
        resolver = PathResolver(root_folder_id="1ABC...", lookup_fn=my_lookup)
        file_id = resolver.resolve("/people/John/photo.jpg")
    """

    def __init__(
        self,
        root_folder_id: str,
        lookup_fn: LookupFn | None = None,
        cache_maxsize: int = 1024,
        cache_ttl: int = 300,
    ) -> None:
        """Initialize the path resolver.

        Args:
            root_folder_id: Provider-specific ID of the root folder.
            lookup_fn: Callable that resolves (parent_id, child_name) to
                       (child_id, is_dir). Injected by the storage
                       implementation (Phase 2). If None, resolve() will
                       raise NotImplementedError for API-dependent lookups.
            cache_maxsize: Maximum number of cache entries (default 1024).
            cache_ttl: Time-to-live in seconds for cache entries (default 300).
        """
        self._root_id = root_folder_id
        self._lookup_fn = lookup_fn
        self._cache_maxsize = cache_maxsize
        self._cache_ttl = cache_ttl
        # Cache: (parent_id, child_name) -> (child_id, is_dir, timestamp)
        self._cache: OrderedDict[
            tuple[str, str], tuple[str, bool, float]
        ] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def root_folder_id(self) -> str:
        """The configured root folder ID."""
        return self._root_id

    @property
    def cache_size(self) -> int:
        """Current number of entries in the cache."""
        with self._lock:
            return len(self._cache)

    def resolve(self, path: str) -> str:
        """Resolve a normalized path to a provider-specific ID.

        Walks path segments from root, using cache where possible.
        Cache misses trigger the lookup function (injected at construction).

        Args:
            path: Normalized virtual path (e.g., "/people/John").
                  Must be normalized via normalize_path() first.

        Returns:
            Provider-specific file/folder ID.

        Raises:
            NotFoundError: If any path segment doesn't exist.
            PathAmbiguousError: If any segment matches multiple items.
            NotImplementedError: If no lookup_fn was provided and a
                                 cache miss occurs.
        """
        if path == "/":
            return self._root_id

        segments = path.strip("/").split("/")
        current_id = self._root_id

        for segment in segments:
            current_id = self._resolve_segment(current_id, segment)

        return current_id

    def cache_put(
        self, parent_id: str, child_name: str, child_id: str, is_dir: bool
    ) -> None:
        """Manually insert an entry into the cache.

        Useful after create_folder operations where we already know
        the ID without needing a lookup.

        Args:
            parent_id: Parent folder ID.
            child_name: Child name.
            child_id: Resolved child ID.
            is_dir: Whether the child is a folder.
        """
        cache_key = (parent_id, child_name)
        with self._lock:
            self._cache[cache_key] = (child_id, is_dir, monotonic())
            self._cache.move_to_end(cache_key)
            # Evict oldest if over capacity
            while len(self._cache) > self._cache_maxsize:
                self._cache.popitem(last=False)

    def invalidate_parent(self, parent_id: str) -> None:
        """Invalidate all cached children of a parent folder.

        Call after create/delete/rename operations within the parent.

        Args:
            parent_id: The parent folder ID whose children to invalidate.
        """
        with self._lock:
            keys_to_remove = [key for key in self._cache if key[0] == parent_id]
            for key in keys_to_remove:
                del self._cache[key]

    def invalidate_all(self) -> None:
        """Clear entire cache.

        Call on bulk operations or error recovery.
        """
        with self._lock:
            self._cache.clear()

    def _resolve_segment(self, parent_id: str, name: str) -> str:
        """Resolve a single path segment within a parent folder.

        Checks cache first (with TTL validation), then falls back
        to the lookup function.

        Args:
            parent_id: Parent folder ID.
            name: Child name to resolve.

        Returns:
            Child ID.

        Raises:
            NotFoundError: If segment does not exist.
            PathAmbiguousError: If segment matches multiple items.
            NotImplementedError: If no lookup_fn was provided and a
                                 cache miss occurs.
        """
        cache_key = (parent_id, name)

        # Check cache
        with self._lock:
            if cache_key in self._cache:
                child_id, _, timestamp = self._cache[cache_key]
                if monotonic() - timestamp < self._cache_ttl:
                    # Cache hit, move to end (LRU)
                    self._cache.move_to_end(cache_key)
                    return child_id
                else:
                    # Expired, remove
                    del self._cache[cache_key]

        # Cache miss: use lookup function
        if self._lookup_fn is None:
            raise NotImplementedError(
                "No lookup function provided. "
                "PathResolver requires a lookup_fn for cache misses. "
                "This is injected by the storage implementation (Phase 2)."
            )

        child_id, is_dir = self._lookup_fn(parent_id, name)

        # Store in cache
        with self._lock:
            self._cache[cache_key] = (child_id, is_dir, monotonic())
            self._cache.move_to_end(cache_key)
            # Evict oldest if over capacity
            while len(self._cache) > self._cache_maxsize:
                self._cache.popitem(last=False)

        return child_id


# Suppress unused import warnings -- NotFoundError and PathAmbiguousError
# are imported for use by lookup functions injected at runtime (Phase 2).
__all__ = [
    "LookupFn",
    "PathResolver",
    "normalize_path",
    "sanitize_folder_name",
    # Re-exported for convenience of lookup function implementations
    "NotFoundError",
    "PathAmbiguousError",
    "RootBoundaryError",
]
