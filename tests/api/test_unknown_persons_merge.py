"""Tests for unknown person group merge endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest
from httpx import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.db.models import FaceInstance, ImageAsset


@pytest.fixture
async def sample_asset(db_session: AsyncSession) -> ImageAsset:
    """Create sample image asset."""
    asset = ImageAsset(
        path="/test/image.jpg",
        training_status="pending",
    )
    db_session.add(asset)
    await db_session.commit()
    await db_session.refresh(asset)
    return asset


@pytest.fixture
async def group_a_faces(
    db_session: AsyncSession,
    sample_asset: ImageAsset,
) -> list[FaceInstance]:
    """Create faces for group A."""
    faces = []
    for i in range(3):
        face = FaceInstance(
            id=uuid4(),
            asset_id=sample_asset.id,
            bbox_x=100 + i * 10,
            bbox_y=100,
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.9,
            quality_score=0.8,
            qdrant_point_id=uuid4(),
            cluster_id="unknown_1",
            person_id=None,
        )
        db_session.add(face)
        faces.append(face)

    await db_session.commit()
    return faces


@pytest.fixture
async def group_b_faces(
    db_session: AsyncSession,
    sample_asset: ImageAsset,
) -> list[FaceInstance]:
    """Create faces for group B."""
    faces = []
    for i in range(2):
        face = FaceInstance(
            id=uuid4(),
            asset_id=sample_asset.id,
            bbox_x=300 + i * 10,
            bbox_y=100,
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.9,
            quality_score=0.8,
            qdrant_point_id=uuid4(),
            cluster_id="unknown_2",
            person_id=None,
        )
        db_session.add(face)
        faces.append(face)

    await db_session.commit()
    return faces


@pytest.fixture
async def group_c_faces(
    db_session: AsyncSession,
    sample_asset: ImageAsset,
) -> list[FaceInstance]:
    """Create faces for group C (for multi-group testing)."""
    faces = []
    for i in range(4):
        face = FaceInstance(
            id=uuid4(),
            asset_id=sample_asset.id,
            bbox_x=500 + i * 10,
            bbox_y=100,
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.9,
            quality_score=0.8,
            qdrant_point_id=uuid4(),
            cluster_id="unknown_3",
            person_id=None,
        )
        db_session.add(face)
        faces.append(face)

    await db_session.commit()
    return faces


@pytest.mark.asyncio
async def test_merge_suggestions_returns_similar_groups(
    test_client: AsyncClient,
    group_a_faces: list[FaceInstance],
    group_b_faces: list[FaceInstance],
) -> None:
    """Test merge-suggestions endpoint returns suggestions above threshold."""
    with patch(
        "image_search_service.api.routes.unknown_persons.get_face_qdrant_client"
    ) as mock_get_qdrant:
        # Mock Qdrant client
        mock_qdrant = MagicMock()
        mock_get_qdrant.return_value = mock_qdrant

        # Mock embeddings (similar vectors for high similarity)
        embedding_a = np.random.rand(512).tolist()
        embedding_b = (np.array(embedding_a) + 0.01).tolist()  # Very similar

        # Build embeddings maps for each cluster (called separately per cluster)
        # The endpoint calls get_embeddings_batch() once per cluster
        def get_embeddings_for_cluster(point_ids):
            result = {}
            group_a_ids = {f.qdrant_point_id for f in group_a_faces}
            for pid in point_ids:
                if pid in group_a_ids:
                    result[pid] = embedding_a
                else:
                    result[pid] = embedding_b
            return result

        mock_qdrant.get_embeddings_batch.side_effect = get_embeddings_for_cluster

        response = await test_client.get(
            "/api/v1/faces/unknown-persons/candidates/merge-suggestions"
            "?maxSuggestions=10&minSimilarity=0.60"
        )

        assert response.status_code == 200
        data = response.json()

        # Should return suggestions
        assert "suggestions" in data
        assert "totalGroupsCompared" in data
        assert data["totalGroupsCompared"] == 2

        # Should have at least one suggestion (groups are similar)
        suggestions = data["suggestions"]
        assert len(suggestions) > 0

        # First suggestion should have required fields
        suggestion = suggestions[0]
        assert "groupAId" in suggestion
        assert "groupBId" in suggestion
        assert "similarity" in suggestion
        assert "groupAFaceCount" in suggestion
        assert "groupBFaceCount" in suggestion

        # Similarity should be in valid range
        assert 0.0 <= suggestion["similarity"] <= 1.0


@pytest.mark.asyncio
async def test_merge_suggestions_respects_min_similarity_threshold(
    test_client: AsyncClient,
    group_a_faces: list[FaceInstance],
    group_b_faces: list[FaceInstance],
) -> None:
    """Test that suggestions below min_similarity are filtered out."""
    with patch(
        "image_search_service.api.routes.unknown_persons.get_face_qdrant_client"
    ) as mock_get_qdrant:
        mock_qdrant = MagicMock()
        mock_get_qdrant.return_value = mock_qdrant

        # Mock dissimilar embeddings (low similarity)
        embedding_a = np.zeros(512).tolist()
        embedding_b = np.ones(512).tolist()

        # Build embeddings maps for each cluster (called separately per cluster)
        def get_embeddings_for_cluster(point_ids):
            result = {}
            group_a_ids = {f.qdrant_point_id for f in group_a_faces}
            for pid in point_ids:
                if pid in group_a_ids:
                    result[pid] = embedding_a
                else:
                    result[pid] = embedding_b
            return result

        mock_qdrant.get_embeddings_batch.side_effect = get_embeddings_for_cluster

        response = await test_client.get(
            "/api/v1/faces/unknown-persons/candidates/merge-suggestions"
            "?maxSuggestions=10&minSimilarity=0.90"  # High threshold
        )

        assert response.status_code == 200
        data = response.json()

        # Should return no suggestions (dissimilar groups, high threshold)
        assert len(data["suggestions"]) == 0


@pytest.mark.asyncio
async def test_merge_groups_success(
    test_client: AsyncClient,
    db_session: AsyncSession,
    group_a_faces: list[FaceInstance],
    group_b_faces: list[FaceInstance],
) -> None:
    """Test successful merge of two groups."""
    # Merge group B into group A
    response = await test_client.post(
        "/api/v1/faces/unknown-persons/candidates/merge",
        json={
            "groupAId": "unknown_1",
            "groupBId": "unknown_2",
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Verify response
    assert data["mergedGroupId"] == "unknown_1"
    assert data["totalFaces"] == 5  # 3 from A + 2 from B
    assert data["facesMoved"] == 2  # 2 from B

    # Verify database state
    # All faces should now be in group A
    group_a_query = select(func.count(FaceInstance.id)).where(
        FaceInstance.cluster_id == "unknown_1"
    )
    group_a_result = await db_session.execute(group_a_query)
    group_a_count = group_a_result.scalar_one()
    assert group_a_count == 5

    # Group B should be empty
    group_b_query = select(func.count(FaceInstance.id)).where(
        FaceInstance.cluster_id == "unknown_2"
    )
    group_b_result = await db_session.execute(group_b_query)
    group_b_count = group_b_result.scalar_one()
    assert group_b_count == 0


@pytest.mark.asyncio
async def test_merge_groups_invalid_group_a(
    test_client: AsyncClient,
    group_b_faces: list[FaceInstance],
) -> None:
    """Test merge with non-existent group A."""
    response = await test_client.post(
        "/api/v1/faces/unknown-persons/candidates/merge",
        json={
            "groupAId": "unknown_nonexistent",
            "groupBId": "unknown_2",
        },
    )

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_merge_groups_invalid_group_b(
    test_client: AsyncClient,
    group_a_faces: list[FaceInstance],
) -> None:
    """Test merge with non-existent group B."""
    response = await test_client.post(
        "/api/v1/faces/unknown-persons/candidates/merge",
        json={
            "groupAId": "unknown_1",
            "groupBId": "unknown_nonexistent",
        },
    )

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_merge_groups_same_group(
    test_client: AsyncClient,
    group_a_faces: list[FaceInstance],
) -> None:
    """Test that merging a group with itself is rejected."""
    response = await test_client.post(
        "/api/v1/faces/unknown-persons/candidates/merge",
        json={
            "groupAId": "unknown_1",
            "groupBId": "unknown_1",
        },
    )

    assert response.status_code == 400
    assert "cannot merge a group with itself" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_merge_suggestions_no_groups(
    test_client: AsyncClient,
) -> None:
    """Test merge-suggestions with no candidate groups."""
    response = await test_client.get(
        "/api/v1/faces/unknown-persons/candidates/merge-suggestions"
    )

    assert response.status_code == 200
    data = response.json()

    assert data["suggestions"] == []
    assert data["totalGroupsCompared"] == 0


