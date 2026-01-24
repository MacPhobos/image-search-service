"""Tests for cluster filtering parameters (min_confidence, min_cluster_size)."""

import uuid
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
async def mock_image_asset(db_session):
    """Create a mock ImageAsset in the database."""
    from image_search_service.db.models import ImageAsset, TrainingStatus

    asset = ImageAsset(
        path="/test/images/photo.jpg",
        training_status=TrainingStatus.PENDING.value,
        width=640,
        height=480,
        file_size=102400,
        mime_type="image/jpeg",
    )
    db_session.add(asset)
    await db_session.commit()
    await db_session.refresh(asset)
    return asset


@pytest.fixture
def mock_qdrant_client(monkeypatch):
    """Create a mock FaceQdrantClient."""
    mock_client = MagicMock()
    mock_client.ensure_collection.return_value = None
    mock_client.upsert_face.return_value = None
    mock_client.upsert_faces_batch.return_value = None
    mock_client.search_similar_faces.return_value = []
    mock_client.search_against_prototypes.return_value = []
    mock_client.update_cluster_ids.return_value = None
    mock_client.update_person_ids.return_value = None

    def get_mock_client():
        return mock_client

    monkeypatch.setattr(
        "image_search_service.vector.face_qdrant.get_face_qdrant_client",
        get_mock_client,
    )
    return mock_client


class TestClustersFiltering:
    """Tests for /api/v1/faces/clusters filtering parameters."""

    @pytest.mark.asyncio
    async def test_clusters_with_min_confidence_filter(
        self, test_client, db_session, mock_image_asset, mock_qdrant_client
    ):
        """Test filtering clusters by minimum confidence."""
        from image_search_service.db.models import FaceInstance

        # Create two clusters with different confidences
        cluster_high = "cluster_high_conf"
        cluster_low = "cluster_low_conf"

        # High confidence cluster (3 faces)
        for i in range(3):
            face = FaceInstance(
                id=uuid.uuid4(),
                asset_id=mock_image_asset.id,
                bbox_x=100 + i * 10,
                bbox_y=100,
                bbox_w=80,
                bbox_h=80,
                detection_confidence=0.95,
                quality_score=0.75,
                qdrant_point_id=uuid.uuid4(),
                cluster_id=cluster_high,
            )
            db_session.add(face)

        # Low confidence cluster (3 faces)
        for i in range(3):
            face = FaceInstance(
                id=uuid.uuid4(),
                asset_id=mock_image_asset.id,
                bbox_x=200 + i * 10,
                bbox_y=100,
                bbox_w=80,
                bbox_h=80,
                detection_confidence=0.85,
                quality_score=0.65,
                qdrant_point_id=uuid.uuid4(),
                cluster_id=cluster_low,
            )
            db_session.add(face)

        await db_session.commit()

        # Mock Qdrant to return high similarity for high_conf cluster
        # and low similarity for low_conf cluster
        def mock_get_embedding(point_id):
            # Return similar embeddings for high_conf, dissimilar for low_conf
            # (This is simplified; real test would use actual embeddings)
            return [0.5] * 512

        mock_qdrant_client.get_embedding_by_point_id.side_effect = mock_get_embedding

        # Mock the FaceClusteringService at the import location
        with patch(
            "image_search_service.services.face_clustering_service.FaceClusteringService.calculate_cluster_confidence"
        ) as mock_calc:
            # High confidence cluster returns 0.95
            # Low confidence cluster returns 0.75
            async def mock_calculate(cluster_id, qdrant_point_ids=None, max_faces_for_calculation=20):  # noqa: E501
                if cluster_id == cluster_high:
                    return 0.95
                elif cluster_id == cluster_low:
                    return 0.75
                return 0.0

            mock_calc.side_effect = mock_calculate

            # When: get clusters with min_confidence=0.85
            response = await test_client.get(
                "/api/v1/faces/clusters",
                params={
                    "include_labeled": False,
                    "min_confidence": 0.85,
                    "page": 1,
                    "page_size": 20,
                },
            )

            # Then: returns only high confidence cluster
            assert response.status_code == 200
            data = response.json()

            # Should only include cluster with confidence >= 0.85
            assert data["total"] == 1
            assert len(data["items"]) == 1
            assert data["items"][0]["clusterId"] == cluster_high
            assert data["items"][0]["clusterConfidence"] >= 0.85

    @pytest.mark.asyncio
    async def test_clusters_with_min_size_filter(
        self, test_client, db_session, mock_image_asset
    ):
        """Test filtering clusters by minimum size."""
        from image_search_service.db.models import FaceInstance

        # Create two clusters with different sizes
        cluster_large = "cluster_large"
        cluster_small = "cluster_small"

        # Large cluster (10 faces)
        for i in range(10):
            face = FaceInstance(
                id=uuid.uuid4(),
                asset_id=mock_image_asset.id,
                bbox_x=100 + i * 10,
                bbox_y=100,
                bbox_w=80,
                bbox_h=80,
                detection_confidence=0.95,
                quality_score=0.75,
                qdrant_point_id=uuid.uuid4(),
                cluster_id=cluster_large,
            )
            db_session.add(face)

        # Small cluster (3 faces)
        for i in range(3):
            face = FaceInstance(
                id=uuid.uuid4(),
                asset_id=mock_image_asset.id,
                bbox_x=200 + i * 10,
                bbox_y=100,
                bbox_w=80,
                bbox_h=80,
                detection_confidence=0.85,
                quality_score=0.65,
                qdrant_point_id=uuid.uuid4(),
                cluster_id=cluster_small,
            )
            db_session.add(face)

        await db_session.commit()

        # When: get clusters with min_cluster_size=5
        response = await test_client.get(
            "/api/v1/faces/clusters",
            params={
                "include_labeled": False,
                "min_cluster_size": 5,
                "page": 1,
                "page_size": 20,
            },
        )

        # Then: returns only large cluster
        assert response.status_code == 200
        data = response.json()

        # Should only include cluster with faceCount >= 5
        assert data["total"] == 1
        assert len(data["items"]) == 1
        assert data["items"][0]["clusterId"] == cluster_large
        assert data["items"][0]["faceCount"] >= 5

    @pytest.mark.asyncio
    async def test_clusters_response_includes_new_fields(
        self, test_client, db_session, mock_image_asset, mock_qdrant_client
    ):
        """Test that response includes clusterConfidence and representativeFaceId."""
        from image_search_service.db.models import FaceInstance

        cluster_id = "test_cluster"

        # Create 3 faces in cluster
        face_ids = []
        for i in range(3):
            face = FaceInstance(
                id=uuid.uuid4(),
                asset_id=mock_image_asset.id,
                bbox_x=100 + i * 10,
                bbox_y=100,
                bbox_w=80,
                bbox_h=80,
                detection_confidence=0.95,
                quality_score=0.75,
                qdrant_point_id=uuid.uuid4(),
                cluster_id=cluster_id,
            )
            db_session.add(face)
            face_ids.append(face.id)

        await db_session.commit()

        # Mock confidence calculation
        with patch(
            "image_search_service.services.face_clustering_service.FaceClusteringService.calculate_cluster_confidence"
        ) as mock_calc:
            mock_calc.return_value = 0.92

            # When: get clusters
            response = await test_client.get(
                "/api/v1/faces/clusters", params={"include_labeled": False}
            )

            # Then: response includes new fields
            assert response.status_code == 200
            data = response.json()

            assert len(data["items"]) == 1
            cluster = data["items"][0]

            # Check that new optional fields are present
            assert "clusterConfidence" in cluster
            assert "representativeFaceId" in cluster

            # clusterConfidence should be populated when calculated
            # representativeFaceId should be the highest quality face
            assert cluster["faceCount"] == 3

    @pytest.mark.asyncio
    async def test_clusters_backward_compatibility(self, test_client, db_session, mock_image_asset):
        """Test that existing clients without new params still work."""
        from image_search_service.db.models import FaceInstance

        # Create a cluster
        cluster_id = "test_cluster"
        for i in range(3):
            face = FaceInstance(
                id=uuid.uuid4(),
                asset_id=mock_image_asset.id,
                bbox_x=100 + i * 10,
                bbox_y=100,
                bbox_w=80,
                bbox_h=80,
                detection_confidence=0.95,
                quality_score=0.75,
                qdrant_point_id=uuid.uuid4(),
                cluster_id=cluster_id,
            )
            db_session.add(face)

        await db_session.commit()

        # When: get clusters without new parameters
        response = await test_client.get(
            "/api/v1/faces/clusters", params={"include_labeled": False}
        )

        # Then: should work without min_confidence or min_cluster_size
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 1

    @pytest.mark.asyncio
    async def test_clusters_combined_filters(
        self, test_client, db_session, mock_image_asset, mock_qdrant_client
    ):
        """Test combining min_confidence and min_cluster_size filters."""
        from image_search_service.db.models import FaceInstance

        # Create three clusters with different characteristics
        cluster_large_high = "cluster_large_high"  # 10 faces, high confidence
        cluster_large_low = "cluster_large_low"  # 10 faces, low confidence
        cluster_small_high = "cluster_small_high"  # 3 faces, high confidence

        # Large + high confidence
        for i in range(10):
            face = FaceInstance(
                id=uuid.uuid4(),
                asset_id=mock_image_asset.id,
                bbox_x=100 + i * 10,
                bbox_y=100,
                bbox_w=80,
                bbox_h=80,
                detection_confidence=0.95,
                quality_score=0.75,
                qdrant_point_id=uuid.uuid4(),
                cluster_id=cluster_large_high,
            )
            db_session.add(face)

        # Large + low confidence
        for i in range(10):
            face = FaceInstance(
                id=uuid.uuid4(),
                asset_id=mock_image_asset.id,
                bbox_x=200 + i * 10,
                bbox_y=100,
                bbox_w=80,
                bbox_h=80,
                detection_confidence=0.85,
                quality_score=0.65,
                qdrant_point_id=uuid.uuid4(),
                cluster_id=cluster_large_low,
            )
            db_session.add(face)

        # Small + high confidence
        for i in range(3):
            face = FaceInstance(
                id=uuid.uuid4(),
                asset_id=mock_image_asset.id,
                bbox_x=300 + i * 10,
                bbox_y=100,
                bbox_w=80,
                bbox_h=80,
                detection_confidence=0.95,
                quality_score=0.75,
                qdrant_point_id=uuid.uuid4(),
                cluster_id=cluster_small_high,
            )
            db_session.add(face)

        await db_session.commit()

        # Mock confidence calculation
        with patch(
            "image_search_service.services.face_clustering_service.FaceClusteringService.calculate_cluster_confidence"
        ) as mock_calc:

            async def mock_calculate(cluster_id, qdrant_point_ids=None, max_faces_for_calculation=20):  # noqa: E501
                if cluster_id == cluster_large_high:
                    return 0.95
                elif cluster_id == cluster_large_low:
                    return 0.70
                elif cluster_id == cluster_small_high:
                    return 0.92
                return 0.0

            mock_calc.side_effect = mock_calculate

            # When: filter by both min_confidence=0.85 and min_cluster_size=5
            response = await test_client.get(
                "/api/v1/faces/clusters",
                params={
                    "include_labeled": False,
                    "min_confidence": 0.85,
                    "min_cluster_size": 5,
                    "page": 1,
                    "page_size": 20,
                },
            )

            # Then: should only return cluster_large_high
            assert response.status_code == 200
            data = response.json()

            assert data["total"] == 1
            assert len(data["items"]) == 1
            assert data["items"][0]["clusterId"] == cluster_large_high
            assert data["items"][0]["faceCount"] >= 5
            assert data["items"][0]["clusterConfidence"] >= 0.85
