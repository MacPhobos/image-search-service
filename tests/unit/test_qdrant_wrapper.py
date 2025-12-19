"""Test Qdrant wrapper functions."""

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from image_search_service.core.config import get_settings
from image_search_service.vector.qdrant import ensure_collection, search_vectors, upsert_vector
from tests.conftest import MockEmbeddingService


def test_upsert_vector_stores_point(
    qdrant_client: QdrantClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that upsert_vector successfully stores a point."""
    # Patch get_qdrant_client to return our test fixture
    monkeypatch.setattr(
        "image_search_service.vector.qdrant.get_qdrant_client", lambda: qdrant_client
    )

    # Generate test vector
    mock_emb = MockEmbeddingService()
    vector = mock_emb.embed_text("test image")

    # Upsert point
    upsert_vector(
        asset_id=123,
        vector=vector,
        payload={"path": "/test/image.jpg"},
    )

    # Verify point was stored by searching for it
    settings = get_settings()
    points = qdrant_client.retrieve(
        collection_name=settings.qdrant_collection,
        ids=[123],
    )

    assert len(points) == 1
    assert points[0].id == 123
    assert points[0].payload is not None
    assert points[0].payload["asset_id"] == "123"
    assert points[0].payload["path"] == "/test/image.jpg"


def test_search_vectors_returns_results(
    qdrant_client: QdrantClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that search_vectors finds similar vectors."""
    # Patch get_qdrant_client
    monkeypatch.setattr(
        "image_search_service.vector.qdrant.get_qdrant_client", lambda: qdrant_client
    )

    mock_emb = MockEmbeddingService()

    # Insert test vectors
    vector1 = mock_emb.embed_text("sunset on beach")
    vector2 = mock_emb.embed_text("mountain peak")

    upsert_vector(asset_id=1, vector=vector1, payload={"path": "/test/sunset.jpg"})
    upsert_vector(asset_id=2, vector=vector2, payload={"path": "/test/mountain.jpg"})

    # Search with query similar to first vector
    query_vector = mock_emb.embed_text("sunset on beach")
    results = search_vectors(query_vector=query_vector, limit=10)

    # Should return results
    assert len(results) > 0

    # Results should have expected structure
    for result in results:
        assert "asset_id" in result
        assert "score" in result
        assert "payload" in result
        assert isinstance(result["score"], (int, float))


def test_search_vectors_respects_limit(
    qdrant_client: QdrantClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that search_vectors respects limit parameter."""
    # Patch get_qdrant_client
    monkeypatch.setattr(
        "image_search_service.vector.qdrant.get_qdrant_client", lambda: qdrant_client
    )

    mock_emb = MockEmbeddingService()

    # Insert 5 test vectors
    for i in range(5):
        vector = mock_emb.embed_text(f"image {i}")
        upsert_vector(asset_id=i, vector=vector, payload={"path": f"/test/img{i}.jpg"})

    # Search with limit=3
    query_vector = mock_emb.embed_text("image query")
    results = search_vectors(query_vector=query_vector, limit=3)

    # Should return at most 3 results
    assert len(results) <= 3


def test_search_vectors_respects_offset(
    qdrant_client: QdrantClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that search_vectors respects offset parameter for pagination."""
    # Patch get_qdrant_client
    monkeypatch.setattr(
        "image_search_service.vector.qdrant.get_qdrant_client", lambda: qdrant_client
    )

    mock_emb = MockEmbeddingService()

    # Insert test vectors
    for i in range(5):
        vector = mock_emb.embed_text(f"image {i}")
        upsert_vector(asset_id=i, vector=vector, payload={"path": f"/test/img{i}.jpg"})

    # Get first page
    query_vector = mock_emb.embed_text("image")
    results1 = search_vectors(query_vector=query_vector, limit=2, offset=0)

    # Get second page
    results2 = search_vectors(query_vector=query_vector, limit=2, offset=2)

    # If we have enough results, pages should be different
    if len(results1) > 0 and len(results2) > 0:
        ids1 = {r["asset_id"] for r in results1}
        ids2 = {r["asset_id"] for r in results2}
        assert ids1 != ids2


def test_ensure_collection_creates_if_missing() -> None:
    """Test that ensure_collection creates collection if it doesn't exist."""
    # Create fresh in-memory client
    client = QdrantClient(":memory:")

    # Verify collection doesn't exist
    collections = client.get_collections().collections
    settings = get_settings()
    collection_names = [c.name for c in collections]
    assert settings.qdrant_collection not in collection_names

    # Ensure collection (should create it)
    # Temporarily patch get_qdrant_client to return our fresh client
    import image_search_service.vector.qdrant as qdrant_module

    original_get_client = qdrant_module.get_qdrant_client
    qdrant_module.get_qdrant_client = lambda: client

    try:
        ensure_collection(embedding_dim=512)

        # Verify collection was created
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        assert settings.qdrant_collection in collection_names
    finally:
        # Restore original function
        qdrant_module.get_qdrant_client = original_get_client


def test_ensure_collection_idempotent() -> None:
    """Test that ensure_collection is idempotent (safe to call multiple times)."""
    client = QdrantClient(":memory:")
    settings = get_settings()

    # Manually create collection
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )

    # Patch get_qdrant_client
    import image_search_service.vector.qdrant as qdrant_module

    original_get_client = qdrant_module.get_qdrant_client
    qdrant_module.get_qdrant_client = lambda: client

    try:
        # Should not raise error when collection already exists
        ensure_collection(embedding_dim=512)

        # Collection should still exist
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        assert settings.qdrant_collection in collection_names
    finally:
        qdrant_module.get_qdrant_client = original_get_client


def test_search_empty_collection_returns_empty(
    qdrant_client: QdrantClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that searching empty collection returns empty list."""
    # Patch get_qdrant_client
    monkeypatch.setattr(
        "image_search_service.vector.qdrant.get_qdrant_client", lambda: qdrant_client
    )

    mock_emb = MockEmbeddingService()
    query_vector = mock_emb.embed_text("test query")

    results = search_vectors(query_vector=query_vector, limit=10)

    # Empty collection should return empty results
    assert results == []
