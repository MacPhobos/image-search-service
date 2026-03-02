"""Tests for hybrid search endpoint (text + image fusion)."""

import pytest
from httpx import AsyncClient
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.core.config import get_settings
from image_search_service.db.models import ImageAsset


@pytest.mark.asyncio
async def test_hybrid_search_text_only(
    test_client: AsyncClient,
    db_session: AsyncSession,
    qdrant_client: QdrantClient,
) -> None:
    """Test hybrid search with text query only."""
    # Create test assets
    asset1 = ImageAsset(path="/test/sunset.jpg")
    db_session.add(asset1)
    await db_session.flush()

    # Insert vector
    settings = get_settings()
    from tests.conftest import MockEmbeddingService

    mock_emb = MockEmbeddingService()
    vector = mock_emb.embed_text("sunset on beach")

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

    # Hybrid search with text only
    response = await test_client.post(
        "/api/v1/search/hybrid",
        data={"textQuery": "sunset on beach", "limit": 10},
    )

    assert response.status_code == 200
    data = response.json()

    assert "results" in data
    assert "textQuery" in data
    assert data["textQuery"] == "sunset on beach"
    assert data["imageFilename"] is None
    assert len(data["results"]) > 0

    # Check result structure
    result = data["results"][0]
    assert "asset" in result
    assert "textScore" in result
    assert "imageScore" in result
    assert "combinedScore" in result
    assert "rank" in result


@pytest.mark.asyncio
async def test_hybrid_search_image_only(
    test_client: AsyncClient,
    db_session: AsyncSession,
    qdrant_client: QdrantClient,
    temp_image_factory,
) -> None:
    """Test hybrid search with image file only."""
    # Create test assets
    asset1 = ImageAsset(path="/test/mountain.jpg")
    db_session.add(asset1)
    await db_session.flush()

    # Insert vector
    settings = get_settings()
    from tests.conftest import MockEmbeddingService

    mock_emb = MockEmbeddingService()
    vector = mock_emb.embed_text("mountain peak")

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

    # Create test image
    image_path = temp_image_factory("query.jpg")

    # Hybrid search with image only
    with open(image_path, "rb") as f:
        response = await test_client.post(
            "/api/v1/search/hybrid",
            files={"file": ("query.jpg", f, "image/jpeg")},
            data={"limit": 10},
        )

    assert response.status_code == 200
    data = response.json()

    assert "results" in data
    assert data["textQuery"] is None
    assert data["imageFilename"] == "query.jpg"
    assert len(data["results"]) > 0


@pytest.mark.asyncio
async def test_hybrid_search_both_modalities(
    test_client: AsyncClient,
    db_session: AsyncSession,
    qdrant_client: QdrantClient,
    temp_image_factory,
) -> None:
    """Test hybrid search with both text and image (RRF fusion)."""
    # Create test assets
    asset1 = ImageAsset(path="/test/sunset.jpg")
    asset2 = ImageAsset(path="/test/mountain.jpg")
    db_session.add(asset1)
    db_session.add(asset2)
    await db_session.flush()

    # Insert vectors
    settings = get_settings()
    from tests.conftest import MockEmbeddingService

    mock_emb = MockEmbeddingService()
    vector1 = mock_emb.embed_text("sunset on beach")
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
    image_path = temp_image_factory("query.jpg")

    # Hybrid search with both text and image
    with open(image_path, "rb") as f:
        response = await test_client.post(
            "/api/v1/search/hybrid",
            files={"file": ("query.jpg", f, "image/jpeg")},
            data={"textQuery": "sunset on beach", "textWeight": 0.5, "limit": 10},
        )

    assert response.status_code == 200
    data = response.json()

    assert "results" in data
    assert data["textQuery"] == "sunset on beach"
    assert data["imageFilename"] == "query.jpg"
    assert len(data["results"]) > 0

    # Results should have scores from both modalities
    result = data["results"][0]
    assert result["textScore"] is not None or result["imageScore"] is not None
    assert result["combinedScore"] > 0
    assert result["rank"] >= 1


@pytest.mark.asyncio
async def test_hybrid_search_missing_both_inputs(test_client: AsyncClient) -> None:
    """Test hybrid search returns 400 when neither text nor image provided."""
    response = await test_client.post(
        "/api/v1/search/hybrid",
        data={"limit": 10},
    )

    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert data["detail"]["error"] == "missing_query"


@pytest.mark.asyncio
async def test_hybrid_search_invalid_file_type(test_client: AsyncClient, tmp_path) -> None:
    """Test hybrid search returns 400 with invalid file type."""
    # Create text file
    text_file = tmp_path / "test.txt"
    text_file.write_text("not an image")

    with open(text_file, "rb") as f:
        response = await test_client.post(
            "/api/v1/search/hybrid",
            files={"file": ("test.txt", f, "text/plain")},
            data={"limit": 10},
        )

    assert response.status_code == 400
    data = response.json()
    assert data["detail"]["error"] == "invalid_file_type"


@pytest.mark.asyncio
async def test_hybrid_search_respects_text_weight(
    test_client: AsyncClient,
    db_session: AsyncSession,
    qdrant_client: QdrantClient,
    temp_image_factory,
) -> None:
    """Test that hybrid search respects textWeight parameter."""
    # Create test asset
    asset1 = ImageAsset(path="/test/test.jpg")
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

    image_path = temp_image_factory("query.jpg")

    # Test with different text weights
    weights = [0.0, 0.5, 1.0]
    for weight in weights:
        with open(image_path, "rb") as f:
            response = await test_client.post(
                "/api/v1/search/hybrid",
                files={"file": ("query.jpg", f, "image/jpeg")},
                data={"textQuery": "test", "textWeight": weight, "limit": 10},
            )

        assert response.status_code == 200
