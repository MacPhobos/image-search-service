"""Shared fixtures for storage unit tests."""

from __future__ import annotations

import pytest

from image_search_service.storage.exceptions import NotFoundError
from image_search_service.storage.path_resolver import PathResolver


@pytest.fixture
def resolver_with_lookup() -> PathResolver:
    """PathResolver with a simple in-memory lookup function."""
    tree: dict[tuple[str, str], tuple[str, bool]] = {
        ("root", "people"): ("folder_people", True),
        ("folder_people", "John Doe"): ("folder_john", True),
        ("folder_john", "photo.jpg"): ("file_photo", False),
    }

    def lookup(parent_id: str, name: str) -> tuple[str, bool]:
        key = (parent_id, name)
        if key not in tree:
            raise NotFoundError(f"{name} not found in {parent_id}")
        return tree[key]

    return PathResolver(
        root_folder_id="root",
        lookup_fn=lookup,
        cache_maxsize=100,
        cache_ttl=300,
    )
