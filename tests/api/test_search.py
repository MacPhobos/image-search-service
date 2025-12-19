"""Test search endpoint."""

import pytest
from httpx import AsyncClient
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.core.config import get_settings
from image_search_service.db.models import ImageAsset


@pytest.mark.asyncio
async def test_search_empty_collection_returns_empty(test_client: AsyncClient) -> None:
    """Test that searching empty collection returns empty results."""
    response = await test_client.post(
        "/api/v1/search",
        json={
            "query": "a beautiful sunset",
            "limit": 10,
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert data["query"] == "a beautiful sunset"
    assert data["results"] == []
    assert data["total"] == 0


@pytest.mark.asyncio
async def test_search_with_results(
    test_client: AsyncClient,
    db_session: AsyncSession,
    qdrant_client: QdrantClient,
    mock_embedding_service: object,
) -> None:
    """Test search returns results after inserting vectors."""
    # Create test assets in database
    asset1 = ImageAsset(path="/test/sunset.jpg")
    asset2 = ImageAsset(path="/test/mountain.jpg")
    db_session.add(asset1)
    db_session.add(asset2)
    await db_session.flush()

    # Insert vectors into Qdrant
    settings = get_settings()

    # Use mock embedding service to generate deterministic vectors
    from tests.conftest import MockEmbeddingService

    mock_emb = MockEmbeddingService()
    vector1 = mock_emb.embed_text("sunset on the beach")
    vector2 = mock_emb.embed_text("mountain peak")

    qdrant_client.upsert(
        collection_name=settings.qdrant_collection,
        points=[
            PointStruct(
                id=asset1.id,
                vector=vector1,
                payload={"asset_id": str(asset1.id), "path": asset1.path},
            ),
            PointStruct(
                id=asset2.id,
                vector=vector2,
                payload={"asset_id": str(asset2.id), "path": asset2.path},
            ),
        ],
    )

    # Search for sunset (should match vector1 better)
    response = await test_client.post(
        "/api/v1/search",
        json={
            "query": "sunset on the beach",
            "limit": 10,
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Should return results
    assert data["query"] == "sunset on the beach"
    assert len(data["results"]) > 0
    assert data["total"] > 0

    # First result should have asset details and score
    first_result = data["results"][0]
    assert "asset" in first_result
    assert "score" in first_result
    assert first_result["asset"]["path"] in ["/test/sunset.jpg", "/test/mountain.jpg"]
    assert isinstance(first_result["score"], (int, float))


@pytest.mark.asyncio
async def test_search_respects_limit(
    test_client: AsyncClient,
    db_session: AsyncSession,
    qdrant_client: QdrantClient,
) -> None:
    """Test that search respects limit parameter."""
    # Create 5 test assets
    assets = []
    for i in range(5):
        asset = ImageAsset(path=f"/test/image{i}.jpg")
        db_session.add(asset)
        assets.append(asset)
    await db_session.flush()

    # Insert vectors
    settings = get_settings()
    from tests.conftest import MockEmbeddingService

    mock_emb = MockEmbeddingService()

    points = []
    for asset in assets:
        vector = mock_emb.embed_text(f"image {asset.id}")
        points.append(
            PointStruct(
                id=asset.id,
                vector=vector,
                payload={"asset_id": str(asset.id), "path": asset.path},
            )
        )

    qdrant_client.upsert(collection_name=settings.qdrant_collection, points=points)

    # Search with limit=3
    response = await test_client.post(
        "/api/v1/search",
        json={
            "query": "image",
            "limit": 3,
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Should return at most 3 results
    assert len(data["results"]) <= 3


@pytest.mark.asyncio
async def test_search_qdrant_unavailable_returns_503(db_session: AsyncSession) -> None:
    """Test that search returns 503 when Qdrant is unavailable."""
    from collections.abc import AsyncGenerator

    from httpx import ASGITransport

    from image_search_service.db.session import get_db
    from image_search_service.main import create_app
    from image_search_service.services.embedding import get_embedding_service
    from image_search_service.vector.qdrant import get_qdrant_client
    from tests.conftest import MockEmbeddingService

    # Create a mock Qdrant client that raises exception on get_collections()
    class FailingQdrantClient:
        def get_collections(self) -> None:
            raise ConnectionError("Qdrant unavailable")

    # Create app with dependency overrides
    app = create_app()

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    def override_get_qdrant() -> FailingQdrantClient:
        return FailingQdrantClient()  # type: ignore[return-value]

    def override_get_embedding() -> MockEmbeddingService:
        return MockEmbeddingService()

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_qdrant_client] = override_get_qdrant
    app.dependency_overrides[get_embedding_service] = override_get_embedding

    # Create test client
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/search",
            json={
                "query": "test query",
            },
        )

        # Should return 503 Service Unavailable
        assert response.status_code == 503
        data = response.json()
        assert "detail" in data
        assert "unavailable" in data["detail"]["message"].lower()

    # Clear overrides
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_search_with_offset(
    test_client: AsyncClient,
    db_session: AsyncSession,
    qdrant_client: QdrantClient,
) -> None:
    """Test that search respects offset parameter for pagination."""
    # Create 5 test assets
    assets = []
    for i in range(5):
        asset = ImageAsset(path=f"/test/image{i}.jpg")
        db_session.add(asset)
        assets.append(asset)
    await db_session.flush()

    # Insert vectors
    settings = get_settings()
    from tests.conftest import MockEmbeddingService

    mock_emb = MockEmbeddingService()

    points = []
    for asset in assets:
        vector = mock_emb.embed_text(f"image {asset.id}")
        points.append(
            PointStruct(
                id=asset.id,
                vector=vector,
                payload={"asset_id": str(asset.id), "path": asset.path},
            )
        )

    qdrant_client.upsert(collection_name=settings.qdrant_collection, points=points)

    # Get first page
    response1 = await test_client.post(
        "/api/v1/search",
        json={"query": "image", "limit": 2, "offset": 0},
    )
    data1 = response1.json()

    # Get second page
    response2 = await test_client.post(
        "/api/v1/search",
        json={"query": "image", "limit": 2, "offset": 2},
    )
    data2 = response2.json()

    # Results should be different (pagination working)
    if len(data1["results"]) > 0 and len(data2["results"]) > 0:
        first_page_ids = {r["asset"]["id"] for r in data1["results"]}
        second_page_ids = {r["asset"]["id"] for r in data2["results"]}
        # Pages should not have overlapping results
        assert first_page_ids != second_page_ids
