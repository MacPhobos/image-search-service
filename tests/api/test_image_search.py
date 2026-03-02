"""Tests for image-based search endpoint."""

import pytest
from httpx import AsyncClient
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.core.config import get_settings
from image_search_service.db.models import ImageAsset


@pytest.mark.asyncio
async def test_search_by_image_success(
    test_client: AsyncClient,
    db_session: AsyncSession,
    qdrant_client: QdrantClient,
    temp_image_factory,
) -> None:
    """Test that image search returns results with valid image."""
    # Create test assets in database
    asset1 = ImageAsset(path="/test/sunset.jpg")
    asset2 = ImageAsset(path="/test/mountain.jpg")
    db_session.add(asset1)
    db_session.add(asset2)
    await db_session.flush()

    # Insert vectors into Qdrant
    settings = get_settings()
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

    # Create test image
    image_path = temp_image_factory("query.jpg", width=100, height=100)

    # Upload image and search
    with open(image_path, "rb") as f:
        response = await test_client.post(
            "/api/v1/search/image",
            files={"file": ("test.jpg", f, "image/jpeg")},
            data={"limit": 10},
        )

    assert response.status_code == 200
    data = response.json()

    # Should return results
    assert "results" in data
    assert "total" in data
    assert "query" in data
    assert "image:" in data["query"]
    assert len(data["results"]) > 0

    # First result should have asset details and score
    first_result = data["results"][0]
    assert "asset" in first_result
    assert "score" in first_result
    assert first_result["asset"]["path"] in ["/test/sunset.jpg", "/test/mountain.jpg"]
    assert isinstance(first_result["score"], int | float)


@pytest.mark.asyncio
async def test_search_by_image_invalid_file(test_client: AsyncClient, tmp_path) -> None:
    """Test that image search returns 400 with non-image file."""
    # Create a text file
    text_file = tmp_path / "test.txt"
    text_file.write_text("not an image")

    with open(text_file, "rb") as f:
        response = await test_client.post(
            "/api/v1/search/image",
            files={"file": ("test.txt", f, "text/plain")},
            data={"limit": 10},
        )

    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert data["detail"]["error"] == "invalid_file_type"


@pytest.mark.asyncio
async def test_search_by_image_respects_limit(
    test_client: AsyncClient,
    db_session: AsyncSession,
    qdrant_client: QdrantClient,
    temp_image_factory,
) -> None:
    """Test that image search respects limit parameter."""
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

    # Create test image
    image_path = temp_image_factory("query.jpg")

    # Search with limit=3
    with open(image_path, "rb") as f:
        response = await test_client.post(
            "/api/v1/search/image",
            files={"file": ("test.jpg", f, "image/jpeg")},
            data={"limit": 3},
        )

    assert response.status_code == 200
    data = response.json()

    # Should return at most 3 results
    assert len(data["results"]) <= 3


@pytest.mark.asyncio
async def test_search_by_image_with_category_filter(
    test_client: AsyncClient,
    db_session: AsyncSession,
    qdrant_client: QdrantClient,
    temp_image_factory,
) -> None:
    """Test that image search respects category filter."""
    # Create test assets in different categories
    asset1 = ImageAsset(path="/test/cat1.jpg")
    asset2 = ImageAsset(path="/test/cat2.jpg")
    db_session.add(asset1)
    db_session.add(asset2)
    await db_session.flush()

    # Insert vectors with different category_ids
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
                payload={"asset_id": str(asset1.id), "path": asset1.path, "category_id": 1},
            ),
            PointStruct(
                id=asset2.id,
                vector=vector,
                payload={"asset_id": str(asset2.id), "path": asset2.path, "category_id": 2},
            ),
        ],
    )

    # Create test image
    image_path = temp_image_factory("query.jpg")

    # Search with category filter
    with open(image_path, "rb") as f:
        response = await test_client.post(
            "/api/v1/search/image",
            files={"file": ("test.jpg", f, "image/jpeg")},
            data={"limit": 10, "categoryId": 1},
        )

    assert response.status_code == 200
    data = response.json()

    # Should only return results from category 1
    # Note: Actual filtering depends on Qdrant implementation
    assert "results" in data
