"""Tests for face API routes."""

import uuid
from unittest.mock import MagicMock, patch

import pytest


# Fixtures for face tests
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
async def mock_face_instance(db_session, mock_image_asset):
    """Create a mock FaceInstance in the database."""
    from image_search_service.db.models import FaceInstance

    face = FaceInstance(
        id=uuid.uuid4(),
        asset_id=mock_image_asset.id,
        bbox_x=100,
        bbox_y=150,
        bbox_w=80,
        bbox_h=80,
        detection_confidence=0.95,
        quality_score=0.75,
        qdrant_point_id=uuid.uuid4(),
    )
    db_session.add(face)
    await db_session.commit()
    await db_session.refresh(face)
    return face


@pytest.fixture
async def mock_person(db_session):
    """Create a mock Person in the database."""
    from image_search_service.db.models import Person, PersonStatus

    person = Person(
        id=uuid.uuid4(),
        name="Test Person",
        status=PersonStatus.ACTIVE.value,
    )
    db_session.add(person)
    await db_session.commit()
    await db_session.refresh(person)
    return person


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
    async def test_list_clusters_partially_labeled(self, test_client, db_session, mock_image_asset, mock_person):
        """Test listing clusters with mixed labeled/unlabeled faces (regression test for duplicate bug).

        This tests the fix for the bug where grouping by both cluster_id AND person_id
        caused duplicate cluster rows when a cluster had some faces labeled and some unlabeled.
        """
        from image_search_service.db.models import FaceInstance

        cluster_id = "test_cluster_mixed"

        # Create 3 faces in the same cluster: 2 labeled, 1 unlabeled
        face1 = FaceInstance(
            id=uuid.uuid4(),
            asset_id=mock_image_asset.id,
            bbox_x=100,
            bbox_y=100,
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.95,
            quality_score=0.85,
            qdrant_point_id=uuid.uuid4(),
            cluster_id=cluster_id,
            person_id=mock_person.id,  # Labeled
        )
        face2 = FaceInstance(
            id=uuid.uuid4(),
            asset_id=mock_image_asset.id,
            bbox_x=200,
            bbox_y=100,
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.92,
            quality_score=0.78,
            qdrant_point_id=uuid.uuid4(),
            cluster_id=cluster_id,
            person_id=mock_person.id,  # Labeled
        )
        face3 = FaceInstance(
            id=uuid.uuid4(),
            asset_id=mock_image_asset.id,
            bbox_x=300,
            bbox_y=100,
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.88,
            quality_score=0.72,
            qdrant_point_id=uuid.uuid4(),
            cluster_id=cluster_id,
            person_id=None,  # Unlabeled
        )
        db_session.add(face1)
        db_session.add(face2)
        db_session.add(face3)
        await db_session.commit()

        # Test with include_labeled=True (should see cluster once with all 3 faces)
        response = await test_client.get(
            "/api/v1/faces/clusters",
            params={"include_labeled": True}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1, "Should have exactly 1 cluster (no duplicates)"

        cluster = data["items"][0]
        assert cluster["clusterId"] == cluster_id
        assert cluster["faceCount"] == 3, "Should aggregate all 3 faces in the cluster"
        assert cluster["personId"] == str(mock_person.id), "Should use MAX person_id (labeled over unlabeled)"
        assert cluster["personName"] == mock_person.name

        # Test with include_labeled=False (should NOT see this cluster since it has labeled faces)
        response = await test_client.get(
            "/api/v1/faces/clusters",
            params={"include_labeled": False}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0, "Should not include clusters with any labeled faces"

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
        assert data["pageSize"] == 10

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

        # Set cluster_id for the mock face
        mock_face_instance.cluster_id = "test_cluster_123"
        await db_session.commit()

        response = await test_client.get("/api/v1/faces/clusters/test_cluster_123")

        assert response.status_code == 200
        data = response.json()
        assert data["clusterId"] == "test_cluster_123"
        assert len(data["faces"]) >= 1

    @pytest.mark.asyncio
    async def test_label_cluster_not_found(self, test_client):
        """Test labeling non-existent cluster."""
        response = await test_client.post(
            "/api/v1/faces/clusters/nonexistent/label",
            json={"name": "John Doe"}
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_split_cluster_not_found(self, test_client):
        """Test splitting non-existent cluster returns 400 (cluster too small)."""
        response = await test_client.post(
            "/api/v1/faces/clusters/nonexistent/split",
            json={"minClusterSize": 3}
        )

        # Non-existent cluster returns 400 (cluster too small to split)
        assert response.status_code == 400
        assert "too small" in response.json()["detail"].lower()


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
        assert data["pageSize"] == 20

    @pytest.mark.asyncio
    async def test_get_person_not_found(self, test_client):
        """Test getting non-existent person."""
        fake_id = str(uuid.uuid4())
        response = await test_client.get(f"/api/v1/faces/persons/{fake_id}")

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()

    @pytest.mark.asyncio
    async def test_get_person_success(self, test_client, db_session, mock_person):
        """Test getting existing person with basic fields."""
        response = await test_client.get(f"/api/v1/faces/persons/{mock_person.id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(mock_person.id)
        assert data["name"] == mock_person.name
        assert data["status"] == mock_person.status
        assert "faceCount" in data
        assert "photoCount" in data
        assert "thumbnailUrl" in data
        assert "createdAt" in data
        assert "updatedAt" in data

    @pytest.mark.asyncio
    async def test_get_person_no_faces(self, test_client, db_session, mock_person):
        """Test getting person with no face instances."""
        response = await test_client.get(f"/api/v1/faces/persons/{mock_person.id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(mock_person.id)
        assert data["name"] == mock_person.name
        assert data["faceCount"] == 0
        assert data["photoCount"] == 0
        assert data["thumbnailUrl"] is None

    @pytest.mark.asyncio
    async def test_get_person_with_faces_counts(self, test_client, db_session, mock_person):
        """Test getting person with multiple faces across multiple photos."""
        from image_search_service.db.models import FaceInstance, ImageAsset

        # Create 2 photos
        photo1 = ImageAsset(
            path="/test/photo1.jpg",
            training_status="pending",
        )
        photo2 = ImageAsset(
            path="/test/photo2.jpg",
            training_status="pending",
        )
        db_session.add(photo1)
        db_session.add(photo2)
        await db_session.flush()

        # Create 3 faces in photo1, 2 faces in photo2
        for i in range(3):
            face = FaceInstance(
                id=uuid.uuid4(),
                asset_id=photo1.id,
                bbox_x=100 + i * 100,
                bbox_y=100,
                bbox_w=80,
                bbox_h=80,
                detection_confidence=0.95,
                quality_score=0.85 - i * 0.1,
                qdrant_point_id=uuid.uuid4(),
                person_id=mock_person.id,
            )
            db_session.add(face)

        for i in range(2):
            face = FaceInstance(
                id=uuid.uuid4(),
                asset_id=photo2.id,
                bbox_x=100 + i * 100,
                bbox_y=100,
                bbox_w=80,
                bbox_h=80,
                detection_confidence=0.92,
                quality_score=0.78 - i * 0.05,
                qdrant_point_id=uuid.uuid4(),
                person_id=mock_person.id,
            )
            db_session.add(face)

        await db_session.commit()

        response = await test_client.get(f"/api/v1/faces/persons/{mock_person.id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(mock_person.id)
        assert data["faceCount"] == 5  # 3 + 2 faces
        assert data["photoCount"] == 2  # 2 distinct photos

    @pytest.mark.asyncio
    async def test_get_person_with_thumbnail(self, test_client, db_session, mock_person, mock_image_asset):
        """Test getting person with thumbnail URL from highest quality face."""
        from image_search_service.db.models import FaceInstance

        # Create 3 faces with different quality scores
        face1 = FaceInstance(
            id=uuid.uuid4(),
            asset_id=mock_image_asset.id,
            bbox_x=100,
            bbox_y=100,
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.95,
            quality_score=0.65,  # Lower quality
            qdrant_point_id=uuid.uuid4(),
            person_id=mock_person.id,
        )
        face2 = FaceInstance(
            id=uuid.uuid4(),
            asset_id=mock_image_asset.id,
            bbox_x=200,
            bbox_y=100,
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.98,
            quality_score=0.95,  # Highest quality
            qdrant_point_id=uuid.uuid4(),
            person_id=mock_person.id,
        )
        face3 = FaceInstance(
            id=uuid.uuid4(),
            asset_id=mock_image_asset.id,
            bbox_x=300,
            bbox_y=100,
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.90,
            quality_score=0.70,  # Medium quality
            qdrant_point_id=uuid.uuid4(),
            person_id=mock_person.id,
        )
        db_session.add(face1)
        db_session.add(face2)
        db_session.add(face3)
        await db_session.commit()

        response = await test_client.get(f"/api/v1/faces/persons/{mock_person.id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(mock_person.id)
        assert data["faceCount"] == 3
        assert data["photoCount"] == 1
        # Thumbnail URL should point to the image containing the highest quality face
        # Implementation uses /api/v1/images/{asset_id}/thumbnail, not face-specific thumbnail
        assert data["thumbnailUrl"] == f"/api/v1/images/{mock_image_asset.id}/thumbnail"

    @pytest.mark.asyncio
    async def test_get_person_timestamps(self, test_client, db_session, mock_person):
        """Test that timestamps are returned correctly."""
        response = await test_client.get(f"/api/v1/faces/persons/{mock_person.id}")

        assert response.status_code == 200
        data = response.json()
        assert "createdAt" in data
        assert "updatedAt" in data
        # Validate ISO format (basic check)
        from datetime import datetime
        datetime.fromisoformat(data["createdAt"].replace("Z", "+00:00"))
        datetime.fromisoformat(data["updatedAt"].replace("Z", "+00:00"))

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

    @pytest.mark.asyncio
    async def test_get_person_photos_not_found(self, test_client):
        """Test getting photos for non-existent person."""
        fake_id = str(uuid.uuid4())
        response = await test_client.get(f"/api/v1/faces/persons/{fake_id}/photos")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_get_person_photos_empty(self, test_client, db_session, mock_person):
        """Test getting photos for person with no faces."""
        response = await test_client.get(f"/api/v1/faces/persons/{mock_person.id}/photos")

        assert response.status_code == 200
        data = response.json()
        assert data["items"] == []
        assert data["total"] == 0
        assert data["page"] == 1
        assert data["personId"] == str(mock_person.id)
        assert data["personName"] == mock_person.name

    @pytest.mark.asyncio
    async def test_get_person_photos_success(self, test_client, db_session, mock_person, mock_image_asset):
        """Test getting photos for person with faces."""
        from image_search_service.db.models import FaceInstance

        # Create 2 faces for the person in the same photo
        face1 = FaceInstance(
            id=uuid.uuid4(),
            asset_id=mock_image_asset.id,
            bbox_x=100,
            bbox_y=100,
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.95,
            quality_score=0.85,
            qdrant_point_id=uuid.uuid4(),
            person_id=mock_person.id,
        )
        face2 = FaceInstance(
            id=uuid.uuid4(),
            asset_id=mock_image_asset.id,
            bbox_x=300,
            bbox_y=100,
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.92,
            quality_score=0.78,
            qdrant_point_id=uuid.uuid4(),
            person_id=None,  # Unlabeled face
        )
        db_session.add(face1)
        db_session.add(face2)
        await db_session.commit()

        response = await test_client.get(f"/api/v1/faces/persons/{mock_person.id}/photos")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["items"]) == 1

        photo = data["items"][0]
        assert photo["photoId"] == mock_image_asset.id
        assert photo["faceCount"] == 2
        assert photo["hasNonPersonFaces"] is True  # face2 has no person_id
        assert len(photo["faces"]) == 2
        assert photo["thumbnailUrl"] == f"/api/v1/images/{mock_image_asset.id}/thumbnail"
        assert photo["fullUrl"] == f"/api/v1/images/{mock_image_asset.id}/full"

    @pytest.mark.asyncio
    async def test_get_person_photos_pagination(self, test_client, db_session, mock_person):
        """Test pagination of person photos."""
        from image_search_service.db.models import FaceInstance, ImageAsset

        # Create 3 different photos with faces for this person
        for i in range(3):
            asset = ImageAsset(
                path=f"/test/photo_{i}.jpg",
                training_status="pending",
            )
            db_session.add(asset)
            await db_session.flush()

            face = FaceInstance(
                id=uuid.uuid4(),
                asset_id=asset.id,
                bbox_x=100,
                bbox_y=100,
                bbox_w=80,
                bbox_h=80,
                detection_confidence=0.95,
                quality_score=0.85,
                qdrant_point_id=uuid.uuid4(),
                person_id=mock_person.id,
            )
            db_session.add(face)

        await db_session.commit()

        # Get first page with page_size=2
        response = await test_client.get(
            f"/api/v1/faces/persons/{mock_person.id}/photos",
            params={"page": 1, "page_size": 2}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert len(data["items"]) == 2
        assert data["page"] == 1
        assert data["pageSize"] == 2

        # Get second page
        response = await test_client.get(
            f"/api/v1/faces/persons/{mock_person.id}/photos",
            params={"page": 2, "page_size": 2}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert len(data["items"]) == 1
        assert data["page"] == 2


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

    @pytest.mark.skip(reason="Endpoint GET /faces/instances not implemented")
    @pytest.mark.asyncio
    async def test_list_face_instances_empty(self, test_client, db_session):
        """Test listing face instances when none exist."""
        response = await test_client.get("/api/v1/faces/instances")

        assert response.status_code == 200
        data = response.json()
        assert data["items"] == []
        assert data["total"] == 0

    @pytest.mark.skip(reason="Endpoint GET /faces/instances not implemented")
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
        assert data["pageSize"] == 50

    @pytest.mark.skip(reason="Endpoint GET /faces/instances/{id} not implemented")
    @pytest.mark.asyncio
    async def test_get_face_instance_not_found(self, test_client):
        """Test getting non-existent face instance."""
        fake_id = str(uuid.uuid4())
        response = await test_client.get(f"/api/v1/faces/instances/{fake_id}")

        assert response.status_code == 404

    @pytest.mark.skip(reason="Endpoint GET /faces/instances/{id} not implemented")
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

    @pytest.mark.skip(reason="Endpoint POST /faces/assign not implemented")
    @pytest.mark.asyncio
    async def test_trigger_assignment_success(self, test_client):
        """Test triggering face assignment."""
        response = await test_client.post(
            "/api/v1/faces/assign",
            json={"maxFaces": 100}
        )

        # Should accept the request (even if no faces to assign)
        assert response.status_code in [200, 202]

    @pytest.mark.skip(reason="Endpoint POST /faces/assign not implemented")
    @pytest.mark.asyncio
    async def test_trigger_assignment_invalid_params(self, test_client):
        """Test triggering assignment with invalid parameters."""
        response = await test_client.post(
            "/api/v1/faces/assign",
            json={"maxFaces": -10}  # Invalid: negative
        )

        # Should return validation error
        assert response.status_code == 422


class TestUnassignFaceEndpoint:
    """Tests for unassigning faces from persons."""

    @pytest.mark.asyncio
    async def test_unassign_face_not_found(self, test_client):
        """Test unassigning non-existent face."""
        fake_id = str(uuid.uuid4())
        response = await test_client.delete(f"/api/v1/faces/faces/{fake_id}/person")

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()

    @pytest.mark.asyncio
    async def test_unassign_face_not_assigned(
        self, test_client, db_session, mock_face_instance, mock_qdrant_client
    ):
        """Test unassigning face that has no person assigned."""
        # Ensure face has no person_id
        mock_face_instance.person_id = None
        await db_session.commit()

        response = await test_client.delete(
            f"/api/v1/faces/faces/{mock_face_instance.id}/person"
        )

        assert response.status_code == 400
        data = response.json()
        assert "not assigned" in data["detail"].lower()

    @pytest.mark.asyncio
    async def test_unassign_face_success(
        self, test_client, db_session, mock_face_instance, mock_person, mock_qdrant_client
    ):
        """Test successful face unassignment."""
        # Assign face to person
        mock_face_instance.person_id = mock_person.id
        await db_session.commit()

        response = await test_client.delete(
            f"/api/v1/faces/faces/{mock_face_instance.id}/person"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["faceId"] == str(mock_face_instance.id)
        assert data["previousPersonId"] == str(mock_person.id)
        assert data["previousPersonName"] == mock_person.name

        # Verify Qdrant was updated (person_id removed)
        mock_qdrant_client.update_person_ids.assert_called_once_with(
            [mock_face_instance.qdrant_point_id], None
        )

        # Verify database was updated
        await db_session.refresh(mock_face_instance)
        assert mock_face_instance.person_id is None


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


class TestBulkOperations:
    """Tests for bulk face assignment operations."""

    @pytest.mark.asyncio
    async def test_bulk_remove_person_not_found(self, test_client):
        """Test bulk remove with non-existent person."""
        fake_id = str(uuid.uuid4())
        response = await test_client.post(
            f"/api/v1/faces/persons/{fake_id}/photos/bulk-remove",
            json={"photoIds": [1, 2, 3]}
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_bulk_remove_empty_photo_ids(self, test_client, db_session, mock_person):
        """Test bulk remove with empty photo list."""
        response = await test_client.post(
            f"/api/v1/faces/persons/{mock_person.id}/photos/bulk-remove",
            json={"photoIds": []}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["updatedFaces"] == 0
        assert data["updatedPhotos"] == 0
        assert data["skippedFaces"] == 0

    @pytest.mark.asyncio
    async def test_bulk_remove_no_matching_faces(self, test_client, db_session, mock_person, mock_image_asset):
        """Test bulk remove when no faces match."""
        # Photo exists but has no faces from this person
        response = await test_client.post(
            f"/api/v1/faces/persons/{mock_person.id}/photos/bulk-remove",
            json={"photoIds": [mock_image_asset.id]}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["updatedFaces"] == 0
        assert data["updatedPhotos"] == 0

    @pytest.mark.asyncio
    async def test_bulk_remove_success(self, test_client, db_session, mock_person, mock_image_asset):
        """Test successful bulk remove operation."""

        from image_search_service.db.models import FaceInstance

        # Create 2 faces for the person
        face1 = FaceInstance(
            id=uuid.uuid4(),
            asset_id=mock_image_asset.id,
            bbox_x=100,
            bbox_y=100,
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.95,
            quality_score=0.85,
            qdrant_point_id=uuid.uuid4(),
            person_id=mock_person.id,
        )
        face2 = FaceInstance(
            id=uuid.uuid4(),
            asset_id=mock_image_asset.id,
            bbox_x=300,
            bbox_y=100,
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.92,
            quality_score=0.78,
            qdrant_point_id=uuid.uuid4(),
            person_id=mock_person.id,
        )
        db_session.add(face1)
        db_session.add(face2)
        await db_session.commit()

        # Mock Qdrant update
        with patch("image_search_service.vector.face_qdrant.get_face_qdrant_client") as mock_qdrant:
            mock_qdrant.return_value.update_person_ids.return_value = None

            response = await test_client.post(
                f"/api/v1/faces/persons/{mock_person.id}/photos/bulk-remove",
                json={"photoIds": [mock_image_asset.id]}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["updatedFaces"] == 2
            assert data["updatedPhotos"] == 1

            # Verify Qdrant was called with None
            mock_qdrant.return_value.update_person_ids.assert_called_once()
            call_args = mock_qdrant.return_value.update_person_ids.call_args
            assert call_args[0][1] is None  # person_id should be None

    @pytest.mark.asyncio
    async def test_bulk_move_person_not_found(self, test_client):
        """Test bulk move with non-existent source person."""
        fake_id = str(uuid.uuid4())
        response = await test_client.post(
            f"/api/v1/faces/persons/{fake_id}/photos/bulk-move",
            json={"photoIds": [1, 2], "toPersonName": "New Person"}
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_bulk_move_missing_destination(self, test_client, db_session, mock_person):
        """Test bulk move without destination person."""
        response = await test_client.post(
            f"/api/v1/faces/persons/{mock_person.id}/photos/bulk-move",
            json={"photoIds": [1, 2]}  # Missing both toPersonId and toPersonName
        )

        # Should return validation error
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_bulk_move_both_destinations(self, test_client, db_session, mock_person):
        """Test bulk move with both toPersonId and toPersonName."""
        response = await test_client.post(
            f"/api/v1/faces/persons/{mock_person.id}/photos/bulk-move",
            json={
                "photoIds": [1, 2],
                "toPersonId": str(uuid.uuid4()),
                "toPersonName": "New Person"
            }
        )

        # Should return validation error
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_bulk_move_to_nonexistent_person(self, test_client, db_session, mock_person):
        """Test bulk move to non-existent target person."""
        fake_target_id = str(uuid.uuid4())
        response = await test_client.post(
            f"/api/v1/faces/persons/{mock_person.id}/photos/bulk-move",
            json={"photoIds": [1, 2], "toPersonId": fake_target_id}
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_bulk_move_create_new_person(self, test_client, db_session, mock_person, mock_image_asset):
        """Test bulk move creating a new person."""

        from image_search_service.db.models import FaceInstance

        # Create a face for the source person
        face = FaceInstance(
            id=uuid.uuid4(),
            asset_id=mock_image_asset.id,
            bbox_x=100,
            bbox_y=100,
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.95,
            quality_score=0.85,
            qdrant_point_id=uuid.uuid4(),
            person_id=mock_person.id,
        )
        db_session.add(face)
        await db_session.commit()

        # Mock Qdrant update
        with patch("image_search_service.vector.face_qdrant.get_face_qdrant_client") as mock_qdrant:
            mock_qdrant.return_value.update_person_ids.return_value = None

            response = await test_client.post(
                f"/api/v1/faces/persons/{mock_person.id}/photos/bulk-move",
                json={"photoIds": [mock_image_asset.id], "toPersonName": "New Person"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["toPersonName"] == "New Person"
            assert data["updatedFaces"] == 1
            assert data["updatedPhotos"] == 1
            assert data["personCreated"] is True

    @pytest.mark.asyncio
    async def test_bulk_move_to_existing_person(self, test_client, db_session, mock_person, mock_image_asset):
        """Test bulk move to existing person."""

        from image_search_service.db.models import FaceInstance, Person

        # Create target person
        target_person = Person(name="Target Person")
        db_session.add(target_person)
        await db_session.flush()

        # Create a face for the source person
        face = FaceInstance(
            id=uuid.uuid4(),
            asset_id=mock_image_asset.id,
            bbox_x=100,
            bbox_y=100,
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.95,
            quality_score=0.85,
            qdrant_point_id=uuid.uuid4(),
            person_id=mock_person.id,
        )
        db_session.add(face)
        await db_session.commit()

        # Mock Qdrant update
        with patch("image_search_service.vector.face_qdrant.get_face_qdrant_client") as mock_qdrant:
            mock_qdrant.return_value.update_person_ids.return_value = None

            response = await test_client.post(
                f"/api/v1/faces/persons/{mock_person.id}/photos/bulk-move",
                json={"photoIds": [mock_image_asset.id], "toPersonId": str(target_person.id)}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["toPersonId"] == str(target_person.id)
            assert data["toPersonName"] == "Target Person"
            assert data["updatedFaces"] == 1
            assert data["updatedPhotos"] == 1
            assert data["personCreated"] is False


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
