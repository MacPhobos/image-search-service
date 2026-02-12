"""Tests for GET /api/v1/faces/unknown-persons/candidates endpoint."""

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
from image_search_service.services.unknown_person_service import compute_membership_hash


@pytest.fixture
async def sample_asset(db_session: AsyncSession) -> ImageAsset:
    """Create sample image asset."""
    asset = ImageAsset(path="/test/image.jpg", training_status="pending")
    db_session.add(asset)
    await db_session.commit()
    await db_session.refresh(asset)
    return asset


@pytest.fixture
async def clustered_faces(
    db_session: AsyncSession, sample_asset: ImageAsset
) -> dict[str, list[FaceInstance]]:
    """Create faces in multiple unknown clusters.

    Note: Uses different bbox coordinates per cluster to avoid UNIQUE constraint violations.
    """
    clusters = {
        "unknown_1": [],
        "unknown_2": [],
        "unknown_3": [],
    }

    # Cluster 1: 8 faces (high quality) - use y=100-107
    for i in range(8):
        face = FaceInstance(
            id=uuid4(),
            asset_id=sample_asset.id,
            bbox_x=100,
            bbox_y=100 + i,  # Unique y coordinate per face
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.95,
            quality_score=0.9,
            qdrant_point_id=uuid4(),
            cluster_id="unknown_1",
            person_id=None,
        )
        db_session.add(face)
        clusters["unknown_1"].append(face)

    # Cluster 2: 5 faces (medium quality) - use y=200-204
    for i in range(5):
        face = FaceInstance(
            id=uuid4(),
            asset_id=sample_asset.id,
            bbox_x=200,
            bbox_y=200 + i,  # Unique y coordinate per face
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.90,
            quality_score=0.75,
            qdrant_point_id=uuid4(),
            cluster_id="unknown_2",
            person_id=None,
        )
        db_session.add(face)
        clusters["unknown_2"].append(face)

    # Cluster 3: 3 faces (low quality, below default threshold) - use y=300-302
    for i in range(3):
        face = FaceInstance(
            id=uuid4(),
            asset_id=sample_asset.id,
            bbox_x=300,
            bbox_y=300 + i,  # Unique y coordinate per face
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.85,
            quality_score=0.60,
            qdrant_point_id=uuid4(),
            cluster_id="unknown_3",
            person_id=None,
        )
        db_session.add(face)
        clusters["unknown_3"].append(face)

    await db_session.commit()
    return clusters


def mock_redis_for_clusters() -> dict:
    """Mock Redis responses for cluster confidence scores."""
    return {
        "unknown_persons:cluster:unknown_1": b"0.85",
        "unknown_persons:cluster:unknown_2": b"0.72",
        "unknown_persons:cluster:unknown_3": b"0.65",  # Below 0.70 threshold
        "unknown_persons:last_discovery": datetime(2026, 2, 11, 12, 0, 0, tzinfo=UTC)
        .isoformat()
        .encode("utf-8"),
    }


@pytest.mark.asyncio
async def test_candidates_empty(test_client: AsyncClient) -> None:
    """Test candidates endpoint with no clusters."""
    with patch("image_search_service.api.routes.unknown_persons.get_redis") as mock_redis:
        mock_redis_conn = mock_redis.return_value
        mock_redis_conn.get.return_value = None

        response = await test_client.get("/api/v1/faces/unknown-persons/candidates")

        assert response.status_code == 200
        data = response.json()

        assert data["groups"] == []
        assert data["totalGroups"] == 0
        assert data["page"] == 1
        assert data["groupsPerPage"] == 50
        assert data["facesPerGroup"] == 6


@pytest.mark.asyncio
async def test_candidates_basic(
    test_client: AsyncClient,
    clustered_faces: dict[str, list[FaceInstance]],
) -> None:
    """Test candidates endpoint returns clusters above threshold."""
    with patch("image_search_service.api.routes.unknown_persons.get_redis") as mock_redis:
        mock_redis_conn = mock_redis.return_value
        redis_data = mock_redis_for_clusters()
        mock_redis_conn.get.side_effect = lambda key: redis_data.get(key)

        response = await test_client.get("/api/v1/faces/unknown-persons/candidates")

        assert response.status_code == 200
        data = response.json()

        # Should return 2 groups (unknown_1 and unknown_2)
        # unknown_3 filtered out due to low confidence (0.65 < 0.70)
        assert len(data["groups"]) == 2
        assert data["totalGroups"] == 2

        # Check group structure
        group = data["groups"][0]
        assert "groupId" in group
        assert "membershipHash" in group
        assert "faceCount" in group
        assert "clusterConfidence" in group
        assert "avgQuality" in group
        assert "representativeFace" in group
        assert "sampleFaces" in group
        assert "isDismissed" in group


@pytest.mark.asyncio
async def test_candidates_pagination(
    test_client: AsyncClient,
    clustered_faces: dict[str, list[FaceInstance]],
) -> None:
    """Test candidates pagination."""
    with patch("image_search_service.api.routes.unknown_persons.get_redis") as mock_redis:
        mock_redis_conn = mock_redis.return_value
        redis_data = mock_redis_for_clusters()
        mock_redis_conn.get.side_effect = lambda key: redis_data.get(key)

        # Page 1, 1 group per page
        response = await test_client.get(
            "/api/v1/faces/unknown-persons/candidates?page=1&groupsPerPage=1"
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data["groups"]) == 1
        assert data["page"] == 1
        assert data["groupsPerPage"] == 1

        # Page 2
        response = await test_client.get(
            "/api/v1/faces/unknown-persons/candidates?page=2&groupsPerPage=1"
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data["groups"]) == 1
        assert data["page"] == 2


@pytest.mark.asyncio
async def test_candidates_min_confidence_override(
    test_client: AsyncClient,
    clustered_faces: dict[str, list[FaceInstance]],
) -> None:
    """Test candidates with custom min_confidence filter."""
    with patch("image_search_service.api.routes.unknown_persons.get_redis") as mock_redis:
        mock_redis_conn = mock_redis.return_value
        redis_data = mock_redis_for_clusters()
        mock_redis_conn.get.side_effect = lambda key: redis_data.get(key)

        # Lower threshold to 0.60 AND minGroupSize to 1 (should include unknown_3 with 3 faces)
        response = await test_client.get(
            "/api/v1/faces/unknown-persons/candidates?minConfidence=0.60&minGroupSize=1"
        )

        assert response.status_code == 200
        data = response.json()

        # Should now return 3 groups (including unknown_3)
        assert len(data["groups"]) == 3
        assert data["minConfidenceSetting"] == 0.60
        assert data["minGroupSizeSetting"] == 1


@pytest.mark.asyncio
async def test_candidates_min_group_size_filter(
    test_client: AsyncClient,
    clustered_faces: dict[str, list[FaceInstance]],
) -> None:
    """Test candidates with min_group_size filter."""
    with patch("image_search_service.api.routes.unknown_persons.get_redis") as mock_redis:
        mock_redis_conn = mock_redis.return_value
        redis_data = mock_redis_for_clusters()
        mock_redis_conn.get.side_effect = lambda key: redis_data.get(key)

        # Only groups with 6+ faces (should filter out unknown_2 and unknown_3)
        response = await test_client.get(
            "/api/v1/faces/unknown-persons/candidates?minGroupSize=6"
        )

        assert response.status_code == 200
        data = response.json()

        # Only unknown_1 has 8 faces
        assert len(data["groups"]) == 1
        assert data["groups"][0]["faceCount"] == 8
        assert data["minGroupSizeSetting"] == 6


@pytest.mark.asyncio
async def test_candidates_sort_by_face_count(
    test_client: AsyncClient,
    clustered_faces: dict[str, list[FaceInstance]],
) -> None:
    """Test candidates sorting by face count."""
    with patch("image_search_service.api.routes.unknown_persons.get_redis") as mock_redis:
        mock_redis_conn = mock_redis.return_value
        redis_data = mock_redis_for_clusters()
        mock_redis_conn.get.side_effect = lambda key: redis_data.get(key)

        # Sort by face count descending
        response = await test_client.get(
            "/api/v1/faces/unknown-persons/candidates?sortBy=face_count&sortOrder=desc"
        )

        assert response.status_code == 200
        data = response.json()

        # unknown_1 (8 faces) should be first, unknown_2 (5 faces) second
        assert data["groups"][0]["faceCount"] == 8
        assert data["groups"][1]["faceCount"] == 5


@pytest.mark.asyncio
async def test_candidates_sort_by_confidence(
    test_client: AsyncClient,
    clustered_faces: dict[str, list[FaceInstance]],
) -> None:
    """Test candidates sorting by confidence."""
    with patch("image_search_service.api.routes.unknown_persons.get_redis") as mock_redis:
        mock_redis_conn = mock_redis.return_value
        redis_data = mock_redis_for_clusters()
        mock_redis_conn.get.side_effect = lambda key: redis_data.get(key)

        # Sort by confidence ascending
        response = await test_client.get(
            "/api/v1/faces/unknown-persons/candidates?sortBy=confidence&sortOrder=asc"
        )

        assert response.status_code == 200
        data = response.json()

        # unknown_2 (0.72) should be first, unknown_1 (0.85) second
        assert data["groups"][0]["clusterConfidence"] == 0.72
        assert data["groups"][1]["clusterConfidence"] == 0.85


@pytest.mark.asyncio
async def test_candidates_with_dismissed_group(
    test_client: AsyncClient,
    db_session: AsyncSession,
    clustered_faces: dict[str, list[FaceInstance]],
) -> None:
    """Test candidates filters out dismissed groups by default."""
    # Dismiss unknown_1
    unknown_1_faces = clustered_faces["unknown_1"]
    face_ids = [f.id for f in unknown_1_faces]
    membership_hash = compute_membership_hash(face_ids)

    dismissed = DismissedUnknownPersonGroup(
        id=uuid4(),
        membership_hash=membership_hash,
        cluster_id="unknown_1",
        face_count=len(face_ids),
        reason="Test dismissal",
        marked_as_noise=False,
    )
    db_session.add(dismissed)
    await db_session.commit()

    with patch("image_search_service.api.routes.unknown_persons.get_redis") as mock_redis:
        mock_redis_conn = mock_redis.return_value
        redis_data = mock_redis_for_clusters()
        mock_redis_conn.get.side_effect = lambda key: redis_data.get(key)

        # Default: exclude dismissed
        response = await test_client.get("/api/v1/faces/unknown-persons/candidates")

        assert response.status_code == 200
        data = response.json()

        # Should only return unknown_2 (unknown_1 dismissed, unknown_3 filtered)
        assert len(data["groups"]) == 1
        assert data["groups"][0]["groupId"] == "unknown_2"


@pytest.mark.asyncio
async def test_candidates_include_dismissed(
    test_client: AsyncClient,
    db_session: AsyncSession,
    clustered_faces: dict[str, list[FaceInstance]],
) -> None:
    """Test candidates can include dismissed groups."""
    # Dismiss unknown_1
    unknown_1_faces = clustered_faces["unknown_1"]
    face_ids = [f.id for f in unknown_1_faces]
    membership_hash = compute_membership_hash(face_ids)

    dismissed = DismissedUnknownPersonGroup(
        id=uuid4(),
        membership_hash=membership_hash,
        cluster_id="unknown_1",
        face_count=len(face_ids),
        reason="Test dismissal",
        marked_as_noise=False,
    )
    db_session.add(dismissed)
    await db_session.commit()

    with patch("image_search_service.api.routes.unknown_persons.get_redis") as mock_redis:
        mock_redis_conn = mock_redis.return_value
        redis_data = mock_redis_for_clusters()
        mock_redis_conn.get.side_effect = lambda key: redis_data.get(key)

        # Include dismissed
        response = await test_client.get(
            "/api/v1/faces/unknown-persons/candidates?includeDismissed=true"
        )

        assert response.status_code == 200
        data = response.json()

        # Should return both unknown_1 and unknown_2
        assert len(data["groups"]) == 2

        # Check dismissed flag
        dismissed_group = next(g for g in data["groups"] if g["groupId"] == "unknown_1")
        assert dismissed_group["isDismissed"] is True


@pytest.mark.asyncio
async def test_candidates_faces_per_group_limit(
    test_client: AsyncClient,
    clustered_faces: dict[str, list[FaceInstance]],
) -> None:
    """Test candidates respects facesPerGroup limit."""
    with patch("image_search_service.api.routes.unknown_persons.get_redis") as mock_redis:
        mock_redis_conn = mock_redis.return_value
        redis_data = mock_redis_for_clusters()
        mock_redis_conn.get.side_effect = lambda key: redis_data.get(key)

        # Request 3 sample faces
        response = await test_client.get(
            "/api/v1/faces/unknown-persons/candidates?facesPerGroup=3"
        )

        assert response.status_code == 200
        data = response.json()

        # Each group should have representative + max 3 sample faces
        for group in data["groups"]:
            assert len(group["sampleFaces"]) <= 3
            assert "representativeFace" in group


@pytest.mark.asyncio
async def test_candidates_representative_face_highest_quality(
    test_client: AsyncClient,
    clustered_faces: dict[str, list[FaceInstance]],
) -> None:
    """Test representative face is highest quality face."""
    with patch("image_search_service.api.routes.unknown_persons.get_redis") as mock_redis:
        mock_redis_conn = mock_redis.return_value
        redis_data = mock_redis_for_clusters()
        mock_redis_conn.get.side_effect = lambda key: redis_data.get(key)

        response = await test_client.get("/api/v1/faces/unknown-persons/candidates")

        assert response.status_code == 200
        data = response.json()

        # unknown_1 has quality 0.9, should be representative
        unknown_1_group = next(g for g in data["groups"] if g["groupId"] == "unknown_1")
        assert unknown_1_group["representativeFace"]["qualityScore"] == 0.9
