"""Test Qdrant wrapper functions."""

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from image_search_service.core.config import get_settings
from image_search_service.vector.qdrant import (
    ensure_collection,
    search_vectors,
    update_vector_payload,
    upsert_vector,
)
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


def test_ensure_collection_creates_if_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that ensure_collection creates collection if it doesn't exist."""
    import image_search_service.vector.qdrant as qdrant_module

    # Create fresh in-memory client
    client = QdrantClient(":memory:")

    # Verify collection doesn't exist in our fresh client
    settings = get_settings()
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    assert settings.qdrant_collection not in collection_names

    # _collection_ensured is a module-level set that persists across tests in the
    # same process. If a previous test (e.g. via the qdrant_client fixture) already
    # called ensure_collection, the fast-path guard returns early and our fresh
    # in-memory client never gets the collection.
    # monkeypatch replaces the set with an empty one and auto-restores it after
    # the test, preventing both false negatives here and pollution of other tests.
    monkeypatch.setattr(qdrant_module, "_collection_ensured", set())
    monkeypatch.setattr(qdrant_module, "get_qdrant_client", lambda: client)

    # ensure_collection must now create the collection (cache was empty)
    ensure_collection(embedding_dim=768)  # Image search uses 768-dim

    # Verify collection was created in our test client
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    assert settings.qdrant_collection in collection_names


def test_ensure_collection_idempotent() -> None:
    """Test that ensure_collection is idempotent (safe to call multiple times)."""
    client = QdrantClient(":memory:")
    settings = get_settings()

    # Manually create collection
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),  # Image search uses 768-dim
    )

    # Patch get_qdrant_client
    import image_search_service.vector.qdrant as qdrant_module

    original_get_client = qdrant_module.get_qdrant_client
    qdrant_module.get_qdrant_client = lambda: client

    try:
        # Should not raise error when collection already exists
        ensure_collection(embedding_dim=768)  # Image search uses 768-dim

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


# ========== Person Filter Tests (New Feature) ==========


def test_upsert_vector_with_person_ids(
    qdrant_client: QdrantClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that upsert_vector includes person_ids in payload."""
    monkeypatch.setattr(
        "image_search_service.vector.qdrant.get_qdrant_client", lambda: qdrant_client
    )

    mock_emb = MockEmbeddingService()
    vector = mock_emb.embed_text("test image")

    # Upsert with person_ids
    upsert_vector(
        asset_id=123,
        vector=vector,
        payload={"path": "/test/image.jpg"},
        person_ids=["person-uuid-1", "person-uuid-2"],
    )

    # Verify person_ids were stored in payload
    settings = get_settings()
    points = qdrant_client.retrieve(
        collection_name=settings.qdrant_collection,
        ids=[123],
    )

    assert len(points) == 1
    assert points[0].payload is not None
    assert "person_ids" in points[0].payload
    assert points[0].payload["person_ids"] == ["person-uuid-1", "person-uuid-2"]


def test_upsert_vector_without_person_ids(
    qdrant_client: QdrantClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that upsert_vector uses empty array when person_ids not provided."""
    monkeypatch.setattr(
        "image_search_service.vector.qdrant.get_qdrant_client", lambda: qdrant_client
    )

    mock_emb = MockEmbeddingService()
    vector = mock_emb.embed_text("test image")

    # Upsert without person_ids parameter
    upsert_vector(
        asset_id=456,
        vector=vector,
        payload={"path": "/test/no-faces.jpg"},
    )

    # Verify person_ids defaults to empty array
    settings = get_settings()
    points = qdrant_client.retrieve(
        collection_name=settings.qdrant_collection,
        ids=[456],
    )

    assert len(points) == 1
    assert points[0].payload is not None
    assert "person_ids" in points[0].payload
    assert points[0].payload["person_ids"] == []


def test_search_with_person_id_filter(
    qdrant_client: QdrantClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that search_vectors applies personId filter correctly."""
    monkeypatch.setattr(
        "image_search_service.vector.qdrant.get_qdrant_client", lambda: qdrant_client
    )

    mock_emb = MockEmbeddingService()

    # Insert test vectors with different person_ids
    vector1 = mock_emb.embed_text("image with person A")
    upsert_vector(
        asset_id=1,
        vector=vector1,
        payload={"path": "/test/person-a.jpg"},
        person_ids=["person-a-uuid"],
    )

    vector2 = mock_emb.embed_text("image with person B")
    upsert_vector(
        asset_id=2,
        vector=vector2,
        payload={"path": "/test/person-b.jpg"},
        person_ids=["person-b-uuid"],
    )

    vector3 = mock_emb.embed_text("image with both persons")
    upsert_vector(
        asset_id=3,
        vector=vector3,
        payload={"path": "/test/both.jpg"},
        person_ids=["person-a-uuid", "person-b-uuid"],
    )

    # Search with personId filter for person A
    query_vector = mock_emb.embed_text("search query")
    results = search_vectors(
        query_vector=query_vector,
        limit=10,
        filters={"personId": "person-a-uuid"},
    )

    # Should only return images with person A (assets 1 and 3)
    asset_ids = {r["asset_id"] for r in results}
    assert asset_ids == {"1", "3"}


def test_search_with_person_id_snake_case(
    qdrant_client: QdrantClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that search_vectors also accepts person_id (snake_case)."""
    monkeypatch.setattr(
        "image_search_service.vector.qdrant.get_qdrant_client", lambda: qdrant_client
    )

    mock_emb = MockEmbeddingService()

    # Insert test vector
    vector = mock_emb.embed_text("test image")
    upsert_vector(
        asset_id=1,
        vector=vector,
        payload={"path": "/test/person.jpg"},
        person_ids=["person-uuid-123"],
    )

    # Search with person_id (snake_case)
    query_vector = mock_emb.embed_text("search query")
    results = search_vectors(
        query_vector=query_vector,
        limit=10,
        filters={"person_id": "person-uuid-123"},
    )

    # Should return the image
    assert len(results) == 1
    assert results[0]["asset_id"] == "1"


def test_search_without_person_filter_returns_all(
    qdrant_client: QdrantClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that search_vectors returns all results when no personId filter."""
    monkeypatch.setattr(
        "image_search_service.vector.qdrant.get_qdrant_client", lambda: qdrant_client
    )

    mock_emb = MockEmbeddingService()

    # Insert vectors with different person_ids
    for i in range(3):
        vector = mock_emb.embed_text(f"image {i}")
        upsert_vector(
            asset_id=i,
            vector=vector,
            payload={"path": f"/test/img{i}.jpg"},
            person_ids=[f"person-{i}"],
        )

    # Search without person filter
    query_vector = mock_emb.embed_text("search query")
    results = search_vectors(query_vector=query_vector, limit=10, filters={})

    # Should return all 3 images
    assert len(results) == 3


def test_search_with_combined_filters(
    qdrant_client: QdrantClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that search_vectors applies personId with other filters."""
    monkeypatch.setattr(
        "image_search_service.vector.qdrant.get_qdrant_client", lambda: qdrant_client
    )

    mock_emb = MockEmbeddingService()

    # Insert vectors with category_id and person_ids
    vector1 = mock_emb.embed_text("cat1 person A")
    upsert_vector(
        asset_id=1,
        vector=vector1,
        payload={"path": "/test/1.jpg", "category_id": 5},
        person_ids=["person-a"],
    )

    vector2 = mock_emb.embed_text("cat2 person A")
    upsert_vector(
        asset_id=2,
        vector=vector2,
        payload={"path": "/test/2.jpg", "category_id": 10},
        person_ids=["person-a"],
    )

    vector3 = mock_emb.embed_text("cat1 person B")
    upsert_vector(
        asset_id=3,
        vector=vector3,
        payload={"path": "/test/3.jpg", "category_id": 5},
        person_ids=["person-b"],
    )

    # Search with both category_id and personId
    query_vector = mock_emb.embed_text("search query")
    results = search_vectors(
        query_vector=query_vector,
        limit=10,
        filters={"category_id": 5, "personId": "person-a"},
    )

    # Should only return asset 1 (category 5 AND person A)
    asset_ids = {r["asset_id"] for r in results}
    assert asset_ids == {"1"}


def test_search_with_nonexistent_person_returns_empty(
    qdrant_client: QdrantClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that searching for nonexistent person returns empty results."""
    monkeypatch.setattr(
        "image_search_service.vector.qdrant.get_qdrant_client", lambda: qdrant_client
    )

    mock_emb = MockEmbeddingService()

    # Insert vector with person
    vector = mock_emb.embed_text("test image")
    upsert_vector(
        asset_id=1,
        vector=vector,
        payload={"path": "/test/1.jpg"},
        person_ids=["person-a"],
    )

    # Search for different person
    query_vector = mock_emb.embed_text("search query")
    results = search_vectors(
        query_vector=query_vector,
        limit=10,
        filters={"personId": "person-nonexistent"},
    )

    # Should return empty
    assert results == []


def test_update_vector_payload_success(
    qdrant_client: QdrantClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that update_vector_payload updates Qdrant payload without changing vector."""
    monkeypatch.setattr(
        "image_search_service.vector.qdrant.get_qdrant_client", lambda: qdrant_client
    )

    mock_emb = MockEmbeddingService()

    # Insert initial vector with empty person_ids
    vector = mock_emb.embed_text("test image")
    upsert_vector(
        asset_id=123,
        vector=vector,
        payload={"path": "/test/image.jpg"},
        person_ids=[],
    )

    # Update payload with person_ids
    update_vector_payload(
        asset_id=123,
        payload_updates={"person_ids": ["person-1", "person-2"]},
    )

    # Verify payload was updated
    settings = get_settings()
    points = qdrant_client.retrieve(
        collection_name=settings.qdrant_collection,
        ids=[123],
    )

    assert len(points) == 1
    assert points[0].payload is not None
    assert points[0].payload["person_ids"] == ["person-1", "person-2"]
    # Original payload should still be there
    assert points[0].payload["path"] == "/test/image.jpg"


def test_update_vector_payload_preserves_other_fields(
    qdrant_client: QdrantClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that update_vector_payload preserves other payload fields."""
    monkeypatch.setattr(
        "image_search_service.vector.qdrant.get_qdrant_client", lambda: qdrant_client
    )

    mock_emb = MockEmbeddingService()

    # Insert vector with multiple payload fields
    vector = mock_emb.embed_text("test image")
    upsert_vector(
        asset_id=456,
        vector=vector,
        payload={"path": "/test/image.jpg", "category_id": 5},
        person_ids=["old-person"],
    )

    # Update only person_ids
    update_vector_payload(
        asset_id=456,
        payload_updates={"person_ids": ["new-person-1", "new-person-2"]},
    )

    # Verify all fields are preserved
    settings = get_settings()
    points = qdrant_client.retrieve(
        collection_name=settings.qdrant_collection,
        ids=[456],
    )

    assert len(points) == 1
    payload = points[0].payload
    assert payload is not None
    assert payload["person_ids"] == ["new-person-1", "new-person-2"]
    assert payload["path"] == "/test/image.jpg"
    assert payload["category_id"] == 5
    assert payload["asset_id"] == "456"


def test_update_vector_payload_with_empty_array(
    qdrant_client: QdrantClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that update_vector_payload can clear person_ids with empty array."""
    monkeypatch.setattr(
        "image_search_service.vector.qdrant.get_qdrant_client", lambda: qdrant_client
    )

    mock_emb = MockEmbeddingService()

    # Insert vector with person_ids
    vector = mock_emb.embed_text("test image")
    upsert_vector(
        asset_id=789,
        vector=vector,
        payload={"path": "/test/image.jpg"},
        person_ids=["person-1", "person-2"],
    )

    # Clear person_ids
    update_vector_payload(
        asset_id=789,
        payload_updates={"person_ids": []},
    )

    # Verify person_ids is now empty
    settings = get_settings()
    points = qdrant_client.retrieve(
        collection_name=settings.qdrant_collection,
        ids=[789],
    )

    assert len(points) == 1
    assert points[0].payload is not None
    assert points[0].payload["person_ids"] == []
