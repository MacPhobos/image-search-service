"""Tests for compose search endpoint (image + modifier text)."""

import pytest
from httpx import AsyncClient
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.core.config import get_settings
from image_search_service.db.models import ImageAsset


@pytest.mark.asyncio
async def test_compose_search_success(
    test_client: AsyncClient,
    db_session: AsyncSession,
    qdrant_client: QdrantClient,
    temp_image_factory,
) -> None:
    """Test composed image retrieval with reference image and modifier text."""
    # Create test assets
    asset1 = ImageAsset(path="/test/beach.jpg")
    asset2 = ImageAsset(path="/test/sunset.jpg")
    db_session.add(asset1)
    db_session.add(asset2)
    await db_session.flush()

    # Insert vectors
    settings = get_settings()
    from tests.conftest import MockEmbeddingService

    mock_emb = MockEmbeddingService()
    vector1 = mock_emb.embed_text("beach scene")
    vector2 = mock_emb.embed_text("sunset colors")

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

    # Create reference image
    image_path = temp_image_factory("beach.jpg")

    # Compose search: beach photo + "at sunset"
    with open(image_path, "rb") as f:
        response = await test_client.post(
            "/api/v1/search/compose",
            files={"file": ("beach.jpg", f, "image/jpeg")},
            data={"modifierText": "at sunset", "alpha": 0.3, "limit": 10},
        )

    assert response.status_code == 200
    data = response.json()

    assert "results" in data
    assert "total" in data
    assert "referenceImage" in data
    assert "modifierText" in data
    assert "alpha" in data

    assert data["referenceImage"] == "beach.jpg"
    assert data["modifierText"] == "at sunset"
    assert data["alpha"] == 0.3
    assert len(data["results"]) > 0

    # Check result structure
    result = data["results"][0]
    assert "asset" in result
    assert "score" in result


@pytest.mark.asyncio
async def test_compose_search_invalid_file(test_client: AsyncClient, tmp_path) -> None:
    """Test compose search returns 400 with invalid file type."""
    text_file = tmp_path / "test.txt"
    text_file.write_text("not an image")

    with open(text_file, "rb") as f:
        response = await test_client.post(
            "/api/v1/search/compose",
            files={"file": ("test.txt", f, "text/plain")},
            data={"modifierText": "at sunset"},
        )

    assert response.status_code == 400
    data = response.json()
    assert data["detail"]["error"] == "invalid_file_type"


@pytest.mark.asyncio
async def test_compose_search_respects_alpha(
    test_client: AsyncClient,
    db_session: AsyncSession,
    qdrant_client: QdrantClient,
    temp_image_factory,
) -> None:
    """Test that compose search respects alpha parameter."""
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

    image_path = temp_image_factory("reference.jpg")

    # Test with different alpha values
    alphas = [0.0, 0.3, 0.7, 1.0]
    for alpha in alphas:
        with open(image_path, "rb") as f:
            response = await test_client.post(
                "/api/v1/search/compose",
                files={"file": ("ref.jpg", f, "image/jpeg")},
                data={"modifierText": "at sunset", "alpha": alpha, "limit": 10},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["alpha"] == alpha


@pytest.mark.asyncio
async def test_compose_search_respects_limit(
    test_client: AsyncClient,
    db_session: AsyncSession,
    qdrant_client: QdrantClient,
    temp_image_factory,
) -> None:
    """Test that compose search respects limit parameter."""
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

    image_path = temp_image_factory("reference.jpg")

    # Search with limit=3
    with open(image_path, "rb") as f:
        response = await test_client.post(
            "/api/v1/search/compose",
            files={"file": ("ref.jpg", f, "image/jpeg")},
            data={"modifierText": "at sunset", "limit": 3},
        )

    assert response.status_code == 200
    data = response.json()

    # Should return at most 3 results
    assert len(data["results"]) <= 3
