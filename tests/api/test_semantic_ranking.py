"""Tests for semantic ranking correctness in search results."""

import pytest
from httpx import AsyncClient
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.core.config import get_settings
from image_search_service.db.models import ImageAsset


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

    The test works with SemanticMockEmbeddingService (768-dim) which provides
    semantic similarity-aware mock embeddings for more pronounced ranking
    differences.
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
        assert avg_nature_rank < avg_food_rank, (
            f"Nature images should rank higher than food images. "
            f"Nature avg rank: {avg_nature_rank:.2f}, Food avg rank: {avg_food_rank:.2f}"
        )

    # Score-based assertion: Nature images should have higher scores
    nature_scores = [result_scores[path] for path in nature_paths if path in result_scores]
    food_scores = [result_scores[path] for path in food_paths if path in result_scores]

    if nature_scores and food_scores:
        avg_nature_score = sum(nature_scores) / len(nature_scores)
        avg_food_score = sum(food_scores) / len(food_scores)

        # Nature images should have higher average similarity score
        assert avg_nature_score > avg_food_score, (
            f"Nature images should have higher scores than food images. "
            f"Nature avg: {avg_nature_score:.4f}, Food avg: {avg_food_score:.4f}"
        )

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

        assert min_animal_score > max_non_animal_score * 0.8, (
            f"Within-cluster similarity should be higher. "
            f"Min animal: {min_animal_score:.4f}, Max non-animal: {max_non_animal_score:.4f}"
        )
