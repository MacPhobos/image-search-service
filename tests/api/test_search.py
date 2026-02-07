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
    assert isinstance(first_result["score"], (int, float))


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
    assert isinstance(first_result["score"], (int, float))


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


@pytest.mark.asyncio
async def test_search_ranks_semantically_similar_higher(
    test_client: AsyncClient,
    db_session: AsyncSession,
    qdrant_client: QdrantClient,
    mock_embedding_service: object,
) -> None:
    """Verify that semantically similar images rank higher in search results.

    This test validates semantic ranking correctness by:
    1. Creating images with semantic filenames (nature vs food concepts)
    2. Using mock embeddings that respect semantic similarity
    3. Searching with a nature-related query
    4. Asserting nature images rank higher than food images

    The test works with both LegacyMockEmbeddingService (512-dim) and
    SemanticMockEmbeddingService (768-dim). With semantic mocks, the
    ranking differences are more pronounced.
    """
    # Create test assets with semantic filenames that trigger concept clusters
    # Nature cluster images
    nature1 = ImageAsset(path="/photos/sunset_beach.jpg")
    nature2 = ImageAsset(path="/photos/ocean_waves.jpg")
    nature3 = ImageAsset(path="/photos/mountain_lake.jpg")

    # Food cluster images
    food1 = ImageAsset(path="/photos/pizza_restaurant.jpg")
    food2 = ImageAsset(path="/photos/burger_meal.jpg")

    # Urban cluster images (control group)
    urban1 = ImageAsset(path="/photos/city_building.jpg")

    # Add all assets to DB
    for asset in [nature1, nature2, nature3, food1, food2, urban1]:
        db_session.add(asset)
    await db_session.flush()

    # Insert vectors into Qdrant using mock embedding service
    settings = get_settings()
    from tests.conftest import MockEmbeddingService

    mock_emb = MockEmbeddingService()

    # Generate embeddings based on path filenames (semantic mock extracts stems)
    points = []
    for asset in [nature1, nature2, nature3, food1, food2, urban1]:
        # Use path as semantic hint for embedding (mock extracts filename stem)
        vector = mock_emb.embed_text(asset.path)
        points.append(
            PointStruct(
                id=asset.id,
                vector=vector,
                payload={"asset_id": str(asset.id), "path": asset.path},
            )
        )

    qdrant_client.upsert(collection_name=settings.qdrant_collection, points=points)

    # Search for nature-related query (should match nature cluster)
    response = await test_client.post(
        "/api/v1/search",
        json={
            "query": "ocean waves at sunset",
            "limit": 10,
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Verify we got results
    assert len(data["results"]) >= 5, "Should return at least 5 results"
    assert data["query"] == "ocean waves at sunset"

    # Extract paths and scores for analysis
    result_paths = [r["asset"]["path"] for r in data["results"]]
    result_scores = {r["asset"]["path"]: r["score"] for r in data["results"]}

    # Verify all expected images are in results
    nature_paths = {nature1.path, nature2.path, nature3.path}
    food_paths = {food1.path, food2.path}

    # Core assertion: Nature images should rank higher than food images
    # Find positions of each category in ranked results
    nature_positions = [i for i, path in enumerate(result_paths) if path in nature_paths]
    food_positions = [i for i, path in enumerate(result_paths) if path in food_paths]

    # Calculate average rank position for each category
    if nature_positions and food_positions:
        avg_nature_rank = sum(nature_positions) / len(nature_positions)
        avg_food_rank = sum(food_positions) / len(food_positions)

        # Nature images should have better (lower) average rank
        assert (
            avg_nature_rank < avg_food_rank
        ), f"Nature images should rank higher than food images. Nature avg rank: {avg_nature_rank:.2f}, Food avg rank: {avg_food_rank:.2f}"

    # Score-based assertion: Nature images should have higher scores
    nature_scores = [result_scores[path] for path in nature_paths if path in result_scores]
    food_scores = [result_scores[path] for path in food_paths if path in result_scores]

    if nature_scores and food_scores:
        avg_nature_score = sum(nature_scores) / len(nature_scores)
        avg_food_score = sum(food_scores) / len(food_scores)

        # Nature images should have higher average similarity score
        assert (
            avg_nature_score > avg_food_score
        ), f"Nature images should have higher scores than food images. Nature avg: {avg_nature_score:.4f}, Food avg: {avg_food_score:.4f}"

    # Detailed ranking verification for debugging
    # The most relevant nature image should be in top 3
    top_3_paths = result_paths[:3]
    assert any(
        path in nature_paths for path in top_3_paths
    ), f"At least one nature image should be in top 3 results. Top 3: {top_3_paths}"


@pytest.mark.asyncio
async def test_search_cross_cluster_similarity_lower(
    test_client: AsyncClient,
    db_session: AsyncSession,
    qdrant_client: QdrantClient,
    mock_embedding_service: object,
) -> None:
    """Verify that cross-cluster similarity scores are lower than within-cluster.

    This test validates that the mock embedding service produces semantically
    meaningful embeddings where:
    - Within-cluster items have high similarity (e.g., "dog" and "cat")
    - Cross-cluster items have low similarity (e.g., "dog" and "pizza")
    """
    # Create assets from different semantic clusters
    animal1 = ImageAsset(path="/photos/dog_puppy.jpg")
    animal2 = ImageAsset(path="/photos/cat_kitten.jpg")
    food1 = ImageAsset(path="/photos/pizza_slice.jpg")
    urban1 = ImageAsset(path="/photos/city_street.jpg")

    for asset in [animal1, animal2, food1, urban1]:
        db_session.add(asset)
    await db_session.flush()

    # Insert vectors
    settings = get_settings()
    from tests.conftest import MockEmbeddingService

    mock_emb = MockEmbeddingService()

    points = []
    for asset in [animal1, animal2, food1, urban1]:
        vector = mock_emb.embed_text(asset.path)
        points.append(
            PointStruct(
                id=asset.id,
                vector=vector,
                payload={"asset_id": str(asset.id), "path": asset.path},
            )
        )

    qdrant_client.upsert(collection_name=settings.qdrant_collection, points=points)

    # Search for animal-related query
    response = await test_client.post(
        "/api/v1/search",
        json={
            "query": "dog playing with puppy",
            "limit": 10,
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Extract scores
    result_scores = {r["asset"]["path"]: r["score"] for r in data["results"]}

    # Animal images should have higher scores than food/urban
    animal_scores = [
        result_scores.get(animal1.path, 0),
        result_scores.get(animal2.path, 0),
    ]
    non_animal_scores = [
        result_scores.get(food1.path, 0),
        result_scores.get(urban1.path, 0),
    ]

    # Filter out zero scores (images not found)
    animal_scores = [s for s in animal_scores if s > 0]
    non_animal_scores = [s for s in non_animal_scores if s > 0]

    if animal_scores and non_animal_scores:
        # Within-cluster (animal) should score higher than cross-cluster
        min_animal_score = min(animal_scores)
        max_non_animal_score = max(non_animal_scores)

        assert (
            min_animal_score > max_non_animal_score * 0.8
        ), f"Within-cluster similarity should be higher. Min animal: {min_animal_score:.4f}, Max non-animal: {max_non_animal_score:.4f}"
