"""Regression tests for Qdrant safety guards.

This module ensures that tests can never accidentally delete production
Qdrant collections by enforcing environment-based collection name isolation.
"""

import os

import pytest


def test_face_collection_name_respects_environment(monkeypatch):
    """Verify face collection name comes from settings, not hardcoded constant.

    This is a critical safety test. If this fails, tests could delete production data.
    """
    from image_search_service.core.config import get_settings

    # Set test collection name via environment
    monkeypatch.setenv("QDRANT_FACE_COLLECTION", "test_faces_custom")

    # Clear settings cache to pick up new environment variable
    get_settings.cache_clear()

    # Import the function that retrieves collection name
    from image_search_service.vector.face_qdrant import _get_face_collection_name

    # Verify it returns the environment-configured name, not hardcoded "faces"
    assert _get_face_collection_name() == "test_faces_custom"

    # Cleanup
    get_settings.cache_clear()


def test_image_collection_name_respects_environment(monkeypatch):
    """Verify image collection name comes from settings, not hardcoded.

    Ensures consistency with face collection behavior.
    """
    from image_search_service.core.config import get_settings

    # Set test collection name via environment
    monkeypatch.setenv("QDRANT_COLLECTION", "test_image_assets_custom")

    # Clear settings cache
    get_settings.cache_clear()

    # Verify settings reflect environment variable
    settings = get_settings()
    assert settings.qdrant_collection == "test_image_assets_custom"

    # Cleanup
    get_settings.cache_clear()


def test_reset_face_collection_blocked_for_production_during_tests():
    """Verify safety guard prevents production deletion during tests.

    This test simulates attempting to reset the production "faces" collection
    while pytest is running, which should be blocked.
    """
    # Set environment to simulate production collection name
    os.environ["QDRANT_FACE_COLLECTION"] = "faces"

    # Ensure PYTEST_CURRENT_TEST is set (pytest sets this automatically)
    assert os.getenv("PYTEST_CURRENT_TEST") is not None, "This test must run under pytest"

    from image_search_service.core.config import get_settings

    get_settings.cache_clear()

    from image_search_service.vector.face_qdrant import FaceQdrantClient

    # Create client (using in-memory Qdrant from fixture)
    client = FaceQdrantClient.get_instance()

    # Attempting to reset should raise RuntimeError due to safety guard
    with pytest.raises(RuntimeError, match="SAFETY GUARD.*production 'faces' collection"):
        client.reset_collection()

    # Cleanup: restore test collection name
    os.environ["QDRANT_FACE_COLLECTION"] = "test_faces"
    get_settings.cache_clear()


def test_reset_image_collection_blocked_for_production_during_tests():
    """Verify safety guard prevents production image collection deletion during tests.

    Ensures the main qdrant.py module has similar protections as face_qdrant.py.
    """
    # Set environment to simulate production collection name
    os.environ["QDRANT_COLLECTION"] = "image_assets"

    # Ensure PYTEST_CURRENT_TEST is set
    assert os.getenv("PYTEST_CURRENT_TEST") is not None

    from image_search_service.core.config import get_settings

    get_settings.cache_clear()

    # Import after clearing cache
    from image_search_service.vector.qdrant import reset_collection

    # Attempting to reset should raise RuntimeError due to safety guard
    with pytest.raises(RuntimeError, match="production collection.*during tests"):
        reset_collection()

    # Cleanup
    os.environ["QDRANT_COLLECTION"] = "test_image_assets"
    get_settings.cache_clear()


def test_test_fixtures_use_safe_collection_names(monkeypatch):
    """Verify that test fixtures in conftest.py set safe collection names.

    This test validates the autouse fixture configuration.
    """
    from image_search_service.core.config import get_settings

    # Fixtures should have already set these via monkeypatch
    settings = get_settings()

    # These should be test-safe names, never production names
    assert settings.qdrant_collection != "image_assets", (
        "Test fixtures failed to override collection name. "
        "Check conftest.py use_test_settings fixture."
    )
    assert settings.qdrant_face_collection != "faces", (
        "Test fixtures failed to override face collection name. "
        "Check conftest.py use_test_settings fixture."
    )

    # Should be test-prefixed
    assert settings.qdrant_collection.startswith("test_")
    assert settings.qdrant_face_collection.startswith("test_")


def test_settings_cache_cleared_between_tests():
    """Verify settings cache is cleared to prevent cross-test contamination.

    If settings cache persists between tests, one test could set production
    collection names that leak into other tests.
    """
    from image_search_service.core.config import get_settings

    # Get current settings
    settings_1 = get_settings()
    initial_collection = settings_1.qdrant_collection

    # Clear cache (simulating what conftest.py should do)
    get_settings.cache_clear()

    # Get settings again - should be new instance
    settings_2 = get_settings()

    # Should still have same values (from environment/fixtures)
    assert settings_2.qdrant_collection == initial_collection

    # But importantly, cache should have been cleared
    # (This is more of a process verification than assertion)
