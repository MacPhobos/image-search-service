"""Tests for similar image search endpoint."""

import pytest
from httpx import AsyncClient
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.core.config import get_settings
from image_search_service.db.models import ImageAsset


@pytest.mark.asyncio
async def test_search_similar_success(
    test_client: AsyncClient,
    db_session: AsyncSession,
    qdrant_client: QdrantClient,
) -> None:
    """Test that similar search returns results for valid asset_id."""
    # Create test assets
    asset1 = ImageAsset(path="/test/image1.jpg")
    asset2 = ImageAsset(path="/test/image2.jpg")
    asset3 = ImageAsset(path="/test/image3.jpg")
    db_session.add(asset1)
    db_session.add(asset2)
    db_session.add(asset3)
    await db_session.flush()

    # Insert vectors
    settings = get_settings()
    from tests.conftest import MockEmbeddingService

    mock_emb = MockEmbeddingService()

    # Use similar embeddings for asset1 and asset2
    vector1 = mock_emb.embed_text("similar image")
    vector2 = mock_emb.embed_text("similar image")
    vector3 = mock_emb.embed_text("different image")

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
            PointStruct(
                id=asset3.id,
                vector=vector3,
                payload={"asset_id": str(asset3.id), "path": asset3.path},
            ),
        ],
    )

    # Search for similar images to asset1
    response = await test_client.get(f"/api/v1/search/similar/{asset1.id}")

    assert response.status_code == 200
    data = response.json()

    assert "results" in data
    assert "total" in data
    assert "query" in data
    assert f"similar to asset {asset1.id}" in data["query"]
    assert len(data["results"]) > 0

    # First result should have asset details and score
    first_result = data["results"][0]
    assert "asset" in first_result
    assert "score" in first_result
    assert isinstance(first_result["score"], int | float)


@pytest.mark.asyncio
async def test_search_similar_not_found(test_client: AsyncClient) -> None:
    """Test that similar search returns 404 for invalid asset_id."""
    response = await test_client.get("/api/v1/search/similar/99999")

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert data["detail"]["error"] == "not_found"


@pytest.mark.asyncio
async def test_search_similar_excludes_self(
    test_client: AsyncClient,
    db_session: AsyncSession,
    qdrant_client: QdrantClient,
) -> None:
    """Test that similar search excludes source image by default."""
    # Create test asset
    asset1 = ImageAsset(path="/test/image1.jpg")
    asset2 = ImageAsset(path="/test/image2.jpg")
    db_session.add(asset1)
    db_session.add(asset2)
    await db_session.flush()

    # Insert vectors
    settings = get_settings()
    from tests.conftest import MockEmbeddingService

    mock_emb = MockEmbeddingService()
    vector = mock_emb.embed_text("test image")

    qdrant_client.upsert(
        collection_name=settings.qdrant_collection,
        points=[
            PointStruct(
                id=asset1.id,
                vector=vector,
                payload={"asset_id": str(asset1.id), "path": asset1.path},
            ),
            PointStruct(
                id=asset2.id,
                vector=vector,
                payload={"asset_id": str(asset2.id), "path": asset2.path},
            ),
        ],
    )

    # Search for similar images with excludeSelf=True (default)
    response = await test_client.get(f"/api/v1/search/similar/{asset1.id}")

    assert response.status_code == 200
    data = response.json()

    # Source image should not be in results
    result_ids = [r["asset"]["id"] for r in data["results"]]
    assert asset1.id not in result_ids


@pytest.mark.asyncio
async def test_search_similar_includes_self_when_requested(
    test_client: AsyncClient,
    db_session: AsyncSession,
    qdrant_client: QdrantClient,
) -> None:
    """Test that similar search includes source image when excludeSelf=False."""
    # Create test asset
    asset1 = ImageAsset(path="/test/image1.jpg")
    db_session.add(asset1)
    await db_session.flush()

    # Insert vector
    settings = get_settings()
    from tests.conftest import MockEmbeddingService

    mock_emb = MockEmbeddingService()
    vector = mock_emb.embed_text("test image")

    qdrant_client.upsert(
        collection_name=settings.qdrant_collection,
        points=[
            PointStruct(
                id=asset1.id,
                vector=vector,
                payload={"asset_id": str(asset1.id), "path": asset1.path},
            ),
        ],
    )

    # Search for similar images with excludeSelf=False
    response = await test_client.get(
        f"/api/v1/search/similar/{asset1.id}",
        params={"exclude_self": False},
    )

    assert response.status_code == 200
    data = response.json()

    # Source image should be in results
    result_ids = [r["asset"]["id"] for r in data["results"]]
    assert asset1.id in result_ids


@pytest.mark.asyncio
async def test_search_similar_vector_not_found(
    test_client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Test that similar search returns 404 when asset exists but has no vector."""
    # Create asset without vector in Qdrant
    asset = ImageAsset(path="/test/image1.jpg")
    db_session.add(asset)
    await db_session.flush()

    response = await test_client.get(f"/api/v1/search/similar/{asset.id}")

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert data["detail"]["error"] == "vector_not_found"


@pytest.mark.asyncio
async def test_search_similar_respects_limit(
    test_client: AsyncClient,
    db_session: AsyncSession,
    qdrant_client: QdrantClient,
) -> None:
    """Test that similar search respects limit parameter."""
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
        vector = mock_emb.embed_text("test image")
        points.append(
            PointStruct(
                id=asset.id,
                vector=vector,
                payload={"asset_id": str(asset.id), "path": asset.path},
            )
        )

    qdrant_client.upsert(collection_name=settings.qdrant_collection, points=points)

    # Search with limit=2
    response = await test_client.get(
        f"/api/v1/search/similar/{assets[0].id}",
        params={"limit": 2},
    )

    assert response.status_code == 200
    data = response.json()

    # Should return at most 2 results (excluding source)
    assert len(data["results"]) <= 2
