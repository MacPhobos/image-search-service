"""Tests for GET /api/v1/faces/unknown-persons/candidates/{group_id} endpoint."""

from __future__ import annotations

from unittest.mock import patch
from uuid import uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.db.models import FaceInstance, ImageAsset


@pytest.fixture
async def sample_asset(db_session: AsyncSession) -> ImageAsset:
    """Create sample image asset."""
    asset = ImageAsset(path="/test/image.jpg", training_status="pending")
    db_session.add(asset)
    await db_session.commit()
    await db_session.refresh(asset)
    return asset


@pytest.fixture
async def test_cluster(db_session: AsyncSession, sample_asset: ImageAsset) -> list[FaceInstance]:
    """Create test cluster with 5 faces."""
    faces = []
    for i in range(5):
        face = FaceInstance(
            id=uuid4(),
            asset_id=sample_asset.id,
            bbox_x=100 + i * 50,
            bbox_y=100,
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.95 - (i * 0.05),  # Varying confidence
            quality_score=0.9 - (i * 0.1),  # Decreasing quality
            qdrant_point_id=uuid4(),
            cluster_id="unknown_test",
            person_id=None,
        )
        db_session.add(face)
        faces.append(face)

    await db_session.commit()
    return faces


@pytest.mark.asyncio
async def test_detail_success(
    test_client: AsyncClient,
    test_cluster: list[FaceInstance],
) -> None:
    """Test detail endpoint returns all faces in group."""
    with patch("image_search_service.api.routes.unknown_persons.get_redis") as mock_redis:
        mock_redis_conn = mock_redis.return_value
        mock_redis_conn.get.return_value = b"0.82"  # Mocked confidence

        response = await test_client.get("/api/v1/faces/unknown-persons/candidates/unknown_test")

        assert response.status_code == 200
        data = response.json()

        assert data["groupId"] == "unknown_test"
        assert data["faceCount"] == 5
        assert data["clusterConfidence"] == 0.82
        assert "avgQuality" in data
        assert "membershipHash" in data

        # All 5 faces should be included
        assert len(data["faces"]) == 5

        # Faces should be sorted by quality (highest first)
        qualities = [f["qualityScore"] for f in data["faces"]]
        assert qualities == sorted(qualities, reverse=True)


@pytest.mark.asyncio
async def test_detail_group_not_found(test_client: AsyncClient) -> None:
    """Test detail endpoint returns 404 for non-existent group."""
    with patch("image_search_service.api.routes.unknown_persons.get_redis") as mock_redis:
        mock_redis_conn = mock_redis.return_value
        mock_redis_conn.get.return_value = None

        response = await test_client.get(
            "/api/v1/faces/unknown-persons/candidates/unknown_nonexistent"
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_detail_faces_structure(
    test_client: AsyncClient,
    test_cluster: list[FaceInstance],
) -> None:
    """Test face objects have correct structure."""
    with patch("image_search_service.api.routes.unknown_persons.get_redis") as mock_redis:
        mock_redis_conn = mock_redis.return_value
        mock_redis_conn.get.return_value = b"0.75"

        response = await test_client.get("/api/v1/faces/unknown-persons/candidates/unknown_test")

        assert response.status_code == 200
        data = response.json()

        face = data["faces"][0]
        assert "faceInstanceId" in face
        assert "assetId" in face
        assert "qualityScore" in face
        assert "detectionConfidence" in face
        assert "bboxX" in face
        assert "bboxY" in face
        assert "bboxW" in face
        assert "bboxH" in face
        assert "thumbnailUrl" in face  # May be None


@pytest.mark.asyncio
async def test_detail_avg_quality_calculation(
    test_client: AsyncClient,
    test_cluster: list[FaceInstance],
) -> None:
    """Test avgQuality is correctly calculated."""
    with patch("image_search_service.api.routes.unknown_persons.get_redis") as mock_redis:
        mock_redis_conn = mock_redis.return_value
        mock_redis_conn.get.return_value = b"0.80"

        response = await test_client.get("/api/v1/faces/unknown-persons/candidates/unknown_test")

        assert response.status_code == 200
        data = response.json()

        # Manually calculate expected average
        # quality_scores: 0.9, 0.8, 0.7, 0.6, 0.5
        expected_avg = (0.9 + 0.8 + 0.7 + 0.6 + 0.5) / 5
        assert abs(data["avgQuality"] - expected_avg) < 0.01


@pytest.mark.asyncio
async def test_detail_redis_confidence_fallback(
    test_client: AsyncClient,
    test_cluster: list[FaceInstance],
) -> None:
    """Test detail handles missing Redis confidence gracefully."""
    with patch("image_search_service.api.routes.unknown_persons.get_redis") as mock_redis:
        mock_redis_conn = mock_redis.return_value
        mock_redis_conn.get.return_value = None  # No cached confidence

        response = await test_client.get("/api/v1/faces/unknown-persons/candidates/unknown_test")

        assert response.status_code == 200
        data = response.json()

        # Should default to 0.0
        assert data["clusterConfidence"] == 0.0


@pytest.mark.asyncio
async def test_detail_redis_failure(
    test_client: AsyncClient,
    test_cluster: list[FaceInstance],
) -> None:
    """Test detail handles Redis failure gracefully."""
    with patch("image_search_service.api.routes.unknown_persons.get_redis") as mock_redis:
        mock_redis_conn = mock_redis.return_value
        mock_redis_conn.get.side_effect = Exception("Redis connection failed")

        response = await test_client.get("/api/v1/faces/unknown-persons/candidates/unknown_test")

        assert response.status_code == 200
        data = response.json()

        # Should fallback to 0.0 confidence
        assert data["clusterConfidence"] == 0.0


@pytest.mark.asyncio
async def test_detail_membership_hash_consistency(
    test_client: AsyncClient,
    test_cluster: list[FaceInstance],
) -> None:
    """Test membership hash is consistent across calls."""
    with patch("image_search_service.api.routes.unknown_persons.get_redis") as mock_redis:
        mock_redis_conn = mock_redis.return_value
        mock_redis_conn.get.return_value = b"0.75"

        response1 = await test_client.get("/api/v1/faces/unknown-persons/candidates/unknown_test")
        response2 = await test_client.get("/api/v1/faces/unknown-persons/candidates/unknown_test")

        assert response1.status_code == 200
        assert response2.status_code == 200

        hash1 = response1.json()["membershipHash"]
        hash2 = response2.json()["membershipHash"]

        # Same group should have same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex digest
