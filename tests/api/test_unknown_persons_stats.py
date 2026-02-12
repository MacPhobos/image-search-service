"""Tests for GET /api/v1/faces/unknown-persons/stats endpoint."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch
from uuid import uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.db.models import (
    DismissedUnknownPersonGroup,
    FaceInstance,
    ImageAsset,
)


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
async def unassigned_faces_clustered(
    db_session: AsyncSession, sample_asset: ImageAsset
) -> list[FaceInstance]:
    """Create clustered unassigned faces (unknown_ cluster_ids)."""
    faces = []
    for i in range(5):
        face = FaceInstance(
            id=uuid4(),
            asset_id=sample_asset.id,
            bbox_x=100,
            bbox_y=100 + i,  # Unique y coordinate per face
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.9,
            quality_score=0.8,
            qdrant_point_id=uuid4(),
            cluster_id=f"unknown_{i % 2}",  # 2 clusters
            person_id=None,
        )
        db_session.add(face)
        faces.append(face)

    await db_session.commit()
    return faces


@pytest.fixture
async def noise_faces(db_session: AsyncSession, sample_asset: ImageAsset) -> list[FaceInstance]:
    """Create noise faces (cluster_id = '-1')."""
    faces = []
    for i in range(3):
        face = FaceInstance(
            id=uuid4(),
            asset_id=sample_asset.id,
            bbox_x=300,
            bbox_y=100 + i,  # Unique y coordinate per face
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.9,
            quality_score=0.6,
            qdrant_point_id=uuid4(),
            cluster_id="-1",
            person_id=None,
        )
        db_session.add(face)
        faces.append(face)

    await db_session.commit()
    return faces


@pytest.fixture
async def unclustered_faces(
    db_session: AsyncSession, sample_asset: ImageAsset
) -> list[FaceInstance]:
    """Create unclustered faces (cluster_id = None)."""
    faces = []
    for i in range(2):
        face = FaceInstance(
            id=uuid4(),
            asset_id=sample_asset.id,
            bbox_x=500,
            bbox_y=100 + i,  # Unique y coordinate per face
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.9,
            quality_score=0.7,
            qdrant_point_id=uuid4(),
            cluster_id=None,
            person_id=None,
        )
        db_session.add(face)
        faces.append(face)

    await db_session.commit()
    return faces


@pytest.mark.asyncio
async def test_stats_zero_state(test_client: AsyncClient) -> None:
    """Test stats endpoint with no faces."""
    response = await test_client.get("/api/v1/faces/unknown-persons/stats")

    assert response.status_code == 200
    data = response.json()

    assert data["totalUnassignedFaces"] == 0
    assert data["totalClusteredFaces"] == 0
    assert data["totalNoiseFaces"] == 0
    assert data["totalUnclusteredFaces"] == 0
    assert data["candidateGroups"] == 0
    assert data["avgGroupSize"] == 0.0
    assert data["avgGroupConfidence"] == 0.0
    assert data["totalDismissedGroups"] == 0
    assert data["lastDiscoveryAt"] is None


@pytest.mark.asyncio
async def test_stats_with_faces(
    test_client: AsyncClient,
    unassigned_faces_clustered: list[FaceInstance],
    noise_faces: list[FaceInstance],
    unclustered_faces: list[FaceInstance],
) -> None:
    """Test stats endpoint with various face types."""
    with patch("image_search_service.api.routes.unknown_persons.get_redis") as mock_redis:
        # Mock Redis responses
        mock_redis_conn = mock_redis.return_value
        mock_redis_conn.get.return_value = None  # No cached data

        response = await test_client.get("/api/v1/faces/unknown-persons/stats")

        assert response.status_code == 200
        data = response.json()

        # 5 clustered + 3 noise + 2 unclustered = 10 total unassigned
        assert data["totalUnassignedFaces"] == 10
        assert data["totalClusteredFaces"] == 5  # unknown_0, unknown_1
        assert data["totalNoiseFaces"] == 3  # cluster_id = '-1'
        assert data["totalUnclusteredFaces"] == 2  # cluster_id = None
        assert data["candidateGroups"] == 2  # 2 distinct unknown_ clusters
        assert data["avgGroupSize"] == 2.5  # 5 faces / 2 groups


@pytest.mark.asyncio
async def test_stats_with_dismissed_groups(
    test_client: AsyncClient,
    db_session: AsyncSession,
    unassigned_faces_clustered: list[FaceInstance],
) -> None:
    """Test stats includes dismissed group count."""
    # Create dismissed group
    dismissed = DismissedUnknownPersonGroup(
        id=uuid4(),
        membership_hash="abc123hash",
        cluster_id="unknown_dismissed",
        face_count=3,
        reason="Not a real person",
        marked_as_noise=False,
    )
    db_session.add(dismissed)
    await db_session.commit()

    with patch("image_search_service.api.routes.unknown_persons.get_redis") as mock_redis:
        mock_redis_conn = mock_redis.return_value
        mock_redis_conn.get.return_value = None

        response = await test_client.get("/api/v1/faces/unknown-persons/stats")

        assert response.status_code == 200
        data = response.json()

        assert data["totalDismissedGroups"] == 1


@pytest.mark.asyncio
async def test_stats_with_redis_timestamp(
    test_client: AsyncClient,
    unassigned_faces_clustered: list[FaceInstance],
) -> None:
    """Test stats includes last discovery timestamp from Redis."""
    discovery_time = datetime(2026, 2, 11, 10, 30, 0, tzinfo=UTC)

    with patch("image_search_service.api.routes.unknown_persons.get_redis") as mock_redis:
        mock_redis_conn = mock_redis.return_value
        mock_redis_conn.get.side_effect = lambda key: (
            discovery_time.isoformat().encode("utf-8")
            if key == "unknown_persons:last_discovery"
            else None
        )

        response = await test_client.get("/api/v1/faces/unknown-persons/stats")

        assert response.status_code == 200
        data = response.json()

        assert data["lastDiscoveryAt"] is not None
        # Check timestamp format (ISO 8601)
        assert "2026-02-11T10:30:00" in data["lastDiscoveryAt"]


@pytest.mark.asyncio
async def test_stats_with_redis_avg_confidence(
    test_client: AsyncClient,
    unassigned_faces_clustered: list[FaceInstance],
) -> None:
    """Test stats includes avg confidence from Redis cache."""
    with patch("image_search_service.api.routes.unknown_persons.get_redis") as mock_redis:
        mock_redis_conn = mock_redis.return_value
        mock_redis_conn.get.side_effect = lambda key: (
            b"0.85" if key == "unknown_persons:avg_confidence" else None
        )

        response = await test_client.get("/api/v1/faces/unknown-persons/stats")

        assert response.status_code == 200
        data = response.json()

        assert data["avgGroupConfidence"] == 0.85


@pytest.mark.asyncio
async def test_stats_redis_failure_graceful(
    test_client: AsyncClient,
    unassigned_faces_clustered: list[FaceInstance],
) -> None:
    """Test stats handles Redis failures gracefully."""
    with patch("image_search_service.api.routes.unknown_persons.get_redis") as mock_redis:
        mock_redis_conn = mock_redis.return_value
        mock_redis_conn.get.side_effect = Exception("Redis connection failed")

        response = await test_client.get("/api/v1/faces/unknown-persons/stats")

        assert response.status_code == 200
        data = response.json()

        # Should return 0 for avg confidence on Redis failure
        assert data["avgGroupConfidence"] == 0.0
        assert data["lastDiscoveryAt"] is None
