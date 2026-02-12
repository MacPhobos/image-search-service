"""Tests for cluster_confidence always being populated in list_clusters endpoint."""

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
    asset = ImageAsset(
        path="/test/image.jpg",
        training_status="pending",
    )
    db_session.add(asset)
    await db_session.commit()
    await db_session.refresh(asset)
    return asset


@pytest.fixture
async def unlabeled_clustered_faces(
    db_session: AsyncSession,
    sample_asset: ImageAsset,
) -> list[FaceInstance]:
    """Create unlabeled clustered faces for testing."""
    cluster_id = "test_cluster_123"
    faces = []

    for i in range(3):
        face = FaceInstance(
            id=uuid4(),
            asset_id=sample_asset.id,
            bbox_x=100 + i * 10,
            bbox_y=100,
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.95,
            quality_score=0.85,
            qdrant_point_id=uuid4(),
            cluster_id=cluster_id,
            person_id=None,  # Unlabeled
        )
        db_session.add(face)
        faces.append(face)

    await db_session.commit()
    return faces


@pytest.mark.asyncio
async def test_list_clusters_returns_confidence_without_min_confidence_param(
    test_client: AsyncClient,
    unlabeled_clustered_faces: list[FaceInstance],
) -> None:
    """Test that cluster_confidence is populated even without min_confidence param.

    This verifies the fix where clustering_service is always initialized,
    not just when min_confidence is provided.
    """
    with patch(
        "image_search_service.services.face_clustering_service.FaceClusteringService.calculate_cluster_confidence"
    ) as mock_calculate_confidence:
        # Mock the confidence calculation to return a known value (as async coroutine)
        async def mock_calc(*args, **kwargs):  # type: ignore[no-untyped-def]
            return 0.85

        mock_calculate_confidence.side_effect = mock_calc

        # Call endpoint WITHOUT min_confidence query param (default includeLabeled=false)
        response = await test_client.get("/api/v1/faces/clusters")

        assert response.status_code == 200
        data = response.json()

        # Should have items
        assert len(data["items"]) > 0

        # First cluster should have cluster_confidence populated
        first_cluster = data["items"][0]
        assert "clusterConfidence" in first_cluster
        assert first_cluster["clusterConfidence"] == 0.85

        # Verify the confidence calculation was called
        assert mock_calculate_confidence.called


@pytest.mark.asyncio
async def test_list_clusters_confidence_handles_calculation_error(
    test_client: AsyncClient,
    unlabeled_clustered_faces: list[FaceInstance],
) -> None:
    """Test that cluster_confidence is None when calculation fails.

    Verifies graceful error handling when confidence calculation raises exception.
    """
    with patch(
        "image_search_service.services.face_clustering_service.FaceClusteringService.calculate_cluster_confidence"
    ) as mock_calculate_confidence:
        # Mock calculation failure (as async coroutine that raises)
        async def mock_calc_error(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise Exception("Qdrant connection failed")

        mock_calculate_confidence.side_effect = mock_calc_error

        # Call endpoint without min_confidence
        response = await test_client.get("/api/v1/faces/clusters")

        assert response.status_code == 200
        data = response.json()

        # Should still return cluster, but with null confidence
        assert len(data["items"]) > 0
        first_cluster = data["items"][0]
        assert first_cluster["clusterConfidence"] is None
