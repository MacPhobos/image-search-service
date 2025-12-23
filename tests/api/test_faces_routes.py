"""Tests for face API routes."""

import uuid

import pytest


class TestClusterEndpoints:
    """Tests for cluster API endpoints."""

    @pytest.mark.asyncio
    async def test_list_clusters_empty(self, test_client, db_session):
        """Test listing clusters when none exist."""
        response = await test_client.get("/api/v1/faces/clusters")

        assert response.status_code == 200
        data = response.json()
        assert data["items"] == []
        assert data["total"] == 0
        assert data["page"] == 1

    @pytest.mark.asyncio
    async def test_list_clusters_pagination(self, test_client, db_session):
        """Test cluster listing with pagination parameters."""
        response = await test_client.get(
            "/api/v1/faces/clusters",
            params={"page": 2, "page_size": 10}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 2
        assert data["page_size"] == 10

    @pytest.mark.asyncio
    async def test_list_clusters_invalid_page(self, test_client, db_session):
        """Test listing clusters with invalid page number."""
        response = await test_client.get(
            "/api/v1/faces/clusters",
            params={"page": 0}
        )

        # Should return validation error
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_get_cluster_not_found(self, test_client):
        """Test getting non-existent cluster."""
        response = await test_client.get("/api/v1/faces/clusters/nonexistent")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_get_cluster_success(self, test_client, db_session, mock_face_instance):
        """Test getting existing cluster."""
        from image_search_service.db.models import FaceInstance

        # Set cluster_id for the mock face
        mock_face_instance.cluster_id = "test_cluster_123"
        await db_session.commit()

        response = await test_client.get("/api/v1/faces/clusters/test_cluster_123")

        assert response.status_code == 200
        data = response.json()
        assert data["cluster_id"] == "test_cluster_123"
        assert len(data["faces"]) >= 1

    @pytest.mark.asyncio
    async def test_label_cluster_not_found(self, test_client):
        """Test labeling non-existent cluster."""
        response = await test_client.post(
            "/api/v1/faces/clusters/nonexistent/label",
            json={"personName": "John Doe"}
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_split_cluster_not_found(self, test_client):
        """Test splitting non-existent cluster."""
        response = await test_client.post(
            "/api/v1/faces/clusters/nonexistent/split",
            json={"minClusterSize": 3}
        )

        assert response.status_code == 404


class TestPersonEndpoints:
    """Tests for person API endpoints."""

    @pytest.mark.asyncio
    async def test_list_persons_empty(self, test_client, db_session):
        """Test listing persons when none exist."""
        response = await test_client.get("/api/v1/faces/persons")

        assert response.status_code == 200
        data = response.json()
        assert data["items"] == []
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_list_persons_pagination(self, test_client, db_session):
        """Test person listing with pagination."""
        response = await test_client.get(
            "/api/v1/faces/persons",
            params={"page": 1, "page_size": 20}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 1
        assert data["page_size"] == 20

    @pytest.mark.asyncio
    async def test_get_person_not_found(self, test_client):
        """Test getting non-existent person."""
        fake_id = str(uuid.uuid4())
        response = await test_client.get(f"/api/v1/faces/persons/{fake_id}")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_person_success(self, test_client, db_session, mock_person):
        """Test getting existing person."""
        response = await test_client.get(f"/api/v1/faces/persons/{mock_person.id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(mock_person.id)
        assert data["name"] == mock_person.name

    @pytest.mark.asyncio
    async def test_update_person_not_found(self, test_client):
        """Test updating non-existent person."""
        fake_id = str(uuid.uuid4())
        response = await test_client.patch(
            f"/api/v1/faces/persons/{fake_id}",
            json={"name": "New Name"}
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_merge_persons_source_not_found(self, test_client, db_session, mock_person):
        """Test merging when source person doesn't exist."""
        fake_id = str(uuid.uuid4())
        response = await test_client.post(
            f"/api/v1/faces/persons/{fake_id}/merge",
            json={"intoPersonId": str(mock_person.id)}
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_merge_persons_target_not_found(self, test_client, db_session, mock_person):
        """Test merging when target person doesn't exist."""
        fake_id = str(uuid.uuid4())
        response = await test_client.post(
            f"/api/v1/faces/persons/{mock_person.id}/merge",
            json={"intoPersonId": fake_id}
        )

        assert response.status_code == 404


class TestFaceDetectionEndpoints:
    """Tests for face detection API endpoints."""

    @pytest.mark.asyncio
    async def test_detect_faces_asset_not_found(self, test_client):
        """Test detecting faces for non-existent asset."""
        fake_id = 999999
        response = await test_client.post(
            f"/api/v1/faces/detect/{fake_id}",
            json={"minConfidence": 0.5}
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_detect_faces_invalid_confidence(self, test_client, db_session, mock_image_asset):
        """Test face detection with invalid confidence value."""
        response = await test_client.post(
            f"/api/v1/faces/detect/{mock_image_asset.id}",
            json={"minConfidence": 1.5}  # Invalid: > 1.0
        )

        # Should return validation error
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_trigger_clustering_invalid_params(self, test_client):
        """Test triggering clustering with invalid parameters."""
        response = await test_client.post(
            "/api/v1/faces/cluster",
            json={"minClusterSize": -1}  # Invalid: negative
        )

        # Should return validation error
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_list_face_instances_empty(self, test_client, db_session):
        """Test listing face instances when none exist."""
        response = await test_client.get("/api/v1/faces/instances")

        assert response.status_code == 200
        data = response.json()
        assert data["items"] == []
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_list_face_instances_pagination(self, test_client):
        """Test face instances listing with pagination."""
        response = await test_client.get(
            "/api/v1/faces/instances",
            params={"page": 1, "page_size": 50}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 1
        assert data["page_size"] == 50

    @pytest.mark.asyncio
    async def test_get_face_instance_not_found(self, test_client):
        """Test getting non-existent face instance."""
        fake_id = str(uuid.uuid4())
        response = await test_client.get(f"/api/v1/faces/instances/{fake_id}")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_face_instance_success(self, test_client, db_session, mock_face_instance):
        """Test getting existing face instance."""
        response = await test_client.get(f"/api/v1/faces/instances/{mock_face_instance.id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(mock_face_instance.id)
        assert data["asset_id"] == mock_face_instance.asset_id


class TestAssignmentEndpoints:
    """Tests for face assignment API endpoints."""

    @pytest.mark.asyncio
    async def test_trigger_assignment_success(self, test_client):
        """Test triggering face assignment."""
        response = await test_client.post(
            "/api/v1/faces/assign",
            json={"maxFaces": 100}
        )

        # Should accept the request (even if no faces to assign)
        assert response.status_code in [200, 202]

    @pytest.mark.asyncio
    async def test_trigger_assignment_invalid_params(self, test_client):
        """Test triggering assignment with invalid parameters."""
        response = await test_client.post(
            "/api/v1/faces/assign",
            json={"maxFaces": -10}  # Invalid: negative
        )

        # Should return validation error
        assert response.status_code == 422


class TestPrototypeEndpoints:
    """Tests for person prototype endpoints."""

    @pytest.mark.asyncio
    async def test_create_prototype_person_not_found(self, test_client):
        """Test creating prototype for non-existent person."""
        fake_id = str(uuid.uuid4())
        fake_face_id = str(uuid.uuid4())

        response = await test_client.post(
            f"/api/v1/faces/persons/{fake_id}/prototypes",
            json={"faceInstanceId": fake_face_id, "role": "exemplar"}
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_list_prototypes_person_not_found(self, test_client):
        """Test listing prototypes for non-existent person."""
        fake_id = str(uuid.uuid4())

        response = await test_client.get(f"/api/v1/faces/persons/{fake_id}/prototypes")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_prototype_not_found(self, test_client):
        """Test deleting non-existent prototype."""
        fake_person_id = str(uuid.uuid4())
        fake_proto_id = str(uuid.uuid4())

        response = await test_client.delete(
            f"/api/v1/faces/persons/{fake_person_id}/prototypes/{fake_proto_id}"
        )

        assert response.status_code == 404


class TestClusteringWorkflow:
    """Integration tests for clustering workflow."""

    @pytest.mark.asyncio
    async def test_clustering_workflow_no_faces(self, test_client, db_session):
        """Test clustering workflow when no faces exist."""
        # Trigger clustering
        response = await test_client.post(
            "/api/v1/faces/cluster",
            json={"minClusterSize": 5, "qualityThreshold": 0.5}
        )

        # Should complete without error (0 clusters found)
        assert response.status_code in [200, 202]

        # List clusters (should be empty)
        response = await test_client.get("/api/v1/faces/clusters")
        assert response.status_code == 200
        assert response.json()["total"] == 0
