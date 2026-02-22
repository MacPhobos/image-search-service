"""Extended tests for face API routes - covering previously untested endpoints."""

import uuid
from unittest.mock import MagicMock, patch

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.db.models import (
    FaceInstance,
    ImageAsset,
    Person,
    PersonStatus,
)

# ============ Fixtures ============


@pytest.fixture
async def person_a(db_session: AsyncSession) -> Person:
    """Create first test person."""
    person = Person(name="Person A", status=PersonStatus.ACTIVE.value)
    db_session.add(person)
    await db_session.commit()
    await db_session.refresh(person)
    return person


@pytest.fixture
async def person_b(db_session: AsyncSession) -> Person:
    """Create second test person."""
    person = Person(name="Person B", status=PersonStatus.ACTIVE.value)
    db_session.add(person)
    await db_session.commit()
    await db_session.refresh(person)
    return person


@pytest.fixture
async def test_asset(db_session: AsyncSession) -> ImageAsset:
    """Create test image asset."""
    asset = ImageAsset(
        path="/test/photo.jpg",
        training_status="pending",
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
async def face_with_person(
    db_session: AsyncSession, test_asset: ImageAsset, person_a: Person
) -> FaceInstance:
    """Create face instance assigned to person."""
    face = FaceInstance(
        id=uuid.uuid4(),
        asset_id=test_asset.id,
        bbox_x=100,
        bbox_y=100,
        bbox_w=80,
        bbox_h=80,
        detection_confidence=0.95,
        quality_score=0.85,
        qdrant_point_id=uuid.uuid4(),
        person_id=person_a.id,
    )
    db_session.add(face)
    await db_session.commit()
    await db_session.refresh(face)
    return face


@pytest.fixture
async def face_without_person(db_session: AsyncSession, test_asset: ImageAsset) -> FaceInstance:
    """Create face instance without person assignment."""
    face = FaceInstance(
        id=uuid.uuid4(),
        asset_id=test_asset.id,
        bbox_x=200,
        bbox_y=100,
        bbox_w=80,
        bbox_h=80,
        detection_confidence=0.92,
        quality_score=0.78,
        qdrant_point_id=uuid.uuid4(),
        person_id=None,
    )
    db_session.add(face)
    await db_session.commit()
    await db_session.refresh(face)
    return face


# ============ Cluster Dual Endpoint Tests ============


@pytest.mark.asyncio
class TestClusterDualEndpoint:
    """Tests for POST /api/v1/faces/cluster/dual endpoint."""

    async def test_cluster_dual_enqueues_job_with_defaults(self, test_client: AsyncClient):
        """POST /cluster/dual enqueues background job with default parameters."""
        # Patch Redis and Queue at their import location within the route function
        with patch("redis.Redis"), patch("rq.Queue") as mock_queue_cls:
            # Setup mock queue
            mock_job = MagicMock()
            mock_job.id = "test-job-123"
            mock_queue = MagicMock()
            mock_queue.enqueue.return_value = mock_job
            mock_queue_cls.return_value = mock_queue

            response = await test_client.post("/api/v1/faces/cluster/dual", json={"queue": True})

            assert response.status_code == 200
            data = response.json()
            assert data["jobId"] == "test-job-123"
            assert data["status"] == "queued"

            # Verify job was enqueued with correct function
            mock_queue.enqueue.assert_called_once()
            call_args = mock_queue.enqueue.call_args
            assert "cluster_dual_job" in str(call_args[0][0])

    async def test_cluster_dual_with_custom_params(self, test_client: AsyncClient):
        """POST /cluster/dual with custom thresholds and method."""
        with patch("redis.Redis"), patch("rq.Queue") as mock_queue_cls:
            mock_job = MagicMock()
            mock_job.id = "test-job-456"
            mock_queue = MagicMock()
            mock_queue.enqueue.return_value = mock_job
            mock_queue_cls.return_value = mock_queue

            response = await test_client.post(
                "/api/v1/faces/cluster/dual",
                json={
                    "personThreshold": 0.8,
                    "unknownMethod": "dbscan",
                    "unknownMinSize": 5,
                    "unknownEps": 0.6,
                    "maxFaces": 1000,
                    "queue": True,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "queued"

            # Verify custom params were passed
            call_kwargs = mock_queue.enqueue.call_args[1]
            assert call_kwargs["person_threshold"] == 0.8
            assert call_kwargs["unknown_method"] == "dbscan"
            assert call_kwargs["unknown_min_size"] == 5
            assert call_kwargs["unknown_eps"] == 0.6
            assert call_kwargs["max_faces"] == 1000

    async def test_cluster_dual_invalid_method(self, test_client: AsyncClient):
        """POST /cluster/dual with invalid clustering method returns 422."""
        response = await test_client.post(
            "/api/v1/faces/cluster/dual",
            json={
                "unknownMethod": "invalid_method",
                "queue": True,
            },
        )

        assert response.status_code == 422


# ============ Train Endpoint Tests ============


@pytest.mark.asyncio
class TestTrainEndpoint:
    """Tests for POST /api/v1/faces/train endpoint."""

    async def test_train_enqueues_job_with_defaults(self, test_client: AsyncClient):
        """POST /train enqueues training job with default parameters."""
        with patch("redis.Redis"), patch("rq.Queue") as mock_queue_cls:
            mock_job = MagicMock()
            mock_job.id = "train-job-789"
            mock_queue = MagicMock()
            mock_queue.enqueue.return_value = mock_job
            mock_queue_cls.return_value = mock_queue

            response = await test_client.post("/api/v1/faces/train", json={"queue": True})

            assert response.status_code == 200
            data = response.json()
            assert data["jobId"] == "train-job-789"
            assert data["status"] == "queued"

            mock_queue.enqueue.assert_called_once()

    async def test_train_with_custom_params(self, test_client: AsyncClient):
        """POST /train with custom training hyperparameters."""
        with patch("redis.Redis"), patch("rq.Queue") as mock_queue_cls:
            mock_job = MagicMock()
            mock_job.id = "train-job-custom"
            mock_queue = MagicMock()
            mock_queue.enqueue.return_value = mock_job
            mock_queue_cls.return_value = mock_queue

            response = await test_client.post(
                "/api/v1/faces/train",
                json={
                    "epochs": 50,
                    "margin": 0.3,
                    "batchSize": 64,
                    "learningRate": 0.0005,
                    "minFaces": 10,
                    "checkpointPath": "/tmp/checkpoint.pth",
                    "queue": True,
                },
            )

            assert response.status_code == 200

            # Verify custom hyperparameters were passed
            call_kwargs = mock_queue.enqueue.call_args[1]
            assert call_kwargs["epochs"] == 50
            assert call_kwargs["margin"] == 0.3
            assert call_kwargs["batch_size"] == 64
            assert call_kwargs["learning_rate"] == 0.0005
            assert call_kwargs["min_faces_per_person"] == 10
            assert call_kwargs["checkpoint_path"] == "/tmp/checkpoint.pth"

    async def test_train_invalid_epochs(self, test_client: AsyncClient):
        """POST /train with invalid epochs value returns 422."""
        response = await test_client.post(
            "/api/v1/faces/train",
            json={
                "epochs": 0,  # Invalid: must be >= 1
                "queue": True,
            },
        )

        assert response.status_code == 422


# ============ Merge Persons Tests ============


@pytest.mark.asyncio
class TestMergePersons:
    """Tests for POST /api/v1/faces/persons/{id}/merge endpoint."""

    async def test_merge_persons_success(
        self,
        test_client: AsyncClient,
        db_session: AsyncSession,
        person_a: Person,
        person_b: Person,
        test_asset: ImageAsset,
    ):
        """POST /persons/{id}/merge successfully merges two persons."""
        # Create faces for person A
        face1 = FaceInstance(
            id=uuid.uuid4(),
            asset_id=test_asset.id,
            bbox_x=100,
            bbox_y=100,
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.95,
            quality_score=0.85,
            qdrant_point_id=uuid.uuid4(),
            person_id=person_a.id,
        )
        face2 = FaceInstance(
            id=uuid.uuid4(),
            asset_id=test_asset.id,
            bbox_x=200,
            bbox_y=100,
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.92,
            quality_score=0.78,
            qdrant_point_id=uuid.uuid4(),
            person_id=person_a.id,
        )
        db_session.add(face1)
        db_session.add(face2)
        await db_session.commit()

        # Mock enqueue helper and Qdrant client
        mock_qdrant_client = MagicMock()
        mock_qdrant_client.update_person_ids = MagicMock(return_value=None)

        qdrant_patch = "image_search_service.vector.face_qdrant.get_face_qdrant_client"
        enqueue_patch = "image_search_service.queue.worker.enqueue_person_ids_update"
        mock_enqueue = MagicMock(return_value=2)
        with (
            patch(enqueue_patch, mock_enqueue),
            patch(qdrant_patch, return_value=mock_qdrant_client),
        ):
            response = await test_client.post(
                f"/api/v1/faces/persons/{person_a.id}/merge",
                json={"intoPersonId": str(person_b.id)},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["sourcePersonId"] == str(person_a.id)
        assert data["targetPersonId"] == str(person_b.id)
        assert data["facesMoved"] == 2

        # Verify enqueue was called with the affected asset IDs
        mock_enqueue.assert_called_once()
        enqueued_ids = mock_enqueue.call_args[0][0]
        assert test_asset.id in enqueued_ids

        # Verify person A is marked as merged
        await db_session.refresh(person_a)
        assert person_a.status == PersonStatus.MERGED.value
        assert person_a.merged_into_id == person_b.id

        # Verify faces were moved to person B
        await db_session.refresh(face1)
        await db_session.refresh(face2)
        assert face1.person_id == person_b.id
        assert face2.person_id == person_b.id

    async def test_merge_persons_source_not_found(self, test_client: AsyncClient, person_b: Person):
        """POST /persons/{id}/merge returns 404 when source person not found."""
        fake_id = uuid.uuid4()
        response = await test_client.post(
            f"/api/v1/faces/persons/{fake_id}/merge", json={"intoPersonId": str(person_b.id)}
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    async def test_merge_persons_target_not_found(self, test_client: AsyncClient, person_a: Person):
        """POST /persons/{id}/merge returns 404 when target person not found."""
        fake_id = uuid.uuid4()
        response = await test_client.post(
            f"/api/v1/faces/persons/{person_a.id}/merge", json={"intoPersonId": str(fake_id)}
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    async def test_merge_persons_self_merge(self, test_client: AsyncClient, person_a: Person):
        """POST /persons/{id}/merge returns 400 when trying to merge person into itself."""
        response = await test_client.post(
            f"/api/v1/faces/persons/{person_a.id}/merge", json={"intoPersonId": str(person_a.id)}
        )

        assert response.status_code == 400
        assert "cannot merge person into itself" in response.json()["detail"].lower()


# ============ Bulk Face Operations Tests ============


@pytest.mark.asyncio
class TestBulkFaceOperations:
    """Tests for bulk remove and bulk move endpoints."""

    async def test_bulk_remove_faces_success(
        self,
        test_client: AsyncClient,
        db_session: AsyncSession,
        person_a: Person,
        test_asset: ImageAsset,
    ):
        """POST /persons/{id}/photos/bulk-remove successfully unassigns faces."""
        # Create 2 faces for person A in the test asset
        face1 = FaceInstance(
            id=uuid.uuid4(),
            asset_id=test_asset.id,
            bbox_x=100,
            bbox_y=100,
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.95,
            quality_score=0.85,
            qdrant_point_id=uuid.uuid4(),
            person_id=person_a.id,
        )
        face2 = FaceInstance(
            id=uuid.uuid4(),
            asset_id=test_asset.id,
            bbox_x=200,
            bbox_y=100,
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.92,
            quality_score=0.78,
            qdrant_point_id=uuid.uuid4(),
            person_id=person_a.id,
        )
        db_session.add(face1)
        db_session.add(face2)
        await db_session.commit()

        # Mock the update_person_ids method to be a no-op
        mock_qdrant_client = MagicMock()
        mock_qdrant_client.update_person_ids = MagicMock(return_value=None)

        qdrant_patch = "image_search_service.vector.face_qdrant.get_face_qdrant_client"
        with patch(qdrant_patch, return_value=mock_qdrant_client):
            response = await test_client.post(
                f"/api/v1/faces/persons/{person_a.id}/photos/bulk-remove",
                json={"photoIds": [test_asset.id]},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["updatedFaces"] == 2
        assert data["updatedPhotos"] == 1

        # Verify faces were unassigned in DB
        await db_session.refresh(face1)
        await db_session.refresh(face2)
        assert face1.person_id is None
        assert face2.person_id is None

    async def test_bulk_remove_faces_empty_list(self, test_client: AsyncClient, person_a: Person):
        """POST /persons/{id}/photos/bulk-remove with empty photo list returns 0 updates."""
        response = await test_client.post(
            f"/api/v1/faces/persons/{person_a.id}/photos/bulk-remove", json={"photoIds": []}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["updatedFaces"] == 0
        assert data["updatedPhotos"] == 0
        assert data["skippedFaces"] == 0

    async def test_bulk_remove_person_not_found(self, test_client: AsyncClient):
        """POST /persons/{id}/photos/bulk-remove returns 404 for missing person."""
        fake_id = uuid.uuid4()
        response = await test_client.post(
            f"/api/v1/faces/persons/{fake_id}/photos/bulk-remove", json={"photoIds": [1, 2, 3]}
        )

        assert response.status_code == 404

    async def test_bulk_move_faces_to_existing_person(
        self,
        test_client: AsyncClient,
        db_session: AsyncSession,
        person_a: Person,
        person_b: Person,
        test_asset: ImageAsset,
    ):
        """POST /persons/{id}/photos/bulk-move successfully moves faces to existing person."""
        # Create face for person A
        face = FaceInstance(
            id=uuid.uuid4(),
            asset_id=test_asset.id,
            bbox_x=100,
            bbox_y=100,
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.95,
            quality_score=0.85,
            qdrant_point_id=uuid.uuid4(),
            person_id=person_a.id,
        )
        db_session.add(face)
        await db_session.commit()

        # Mock enqueue helper and Qdrant operations
        mock_qdrant_client = MagicMock()
        mock_qdrant_client.update_person_ids = MagicMock(return_value=None)

        qdrant_patch = "image_search_service.vector.face_qdrant.get_face_qdrant_client"
        enqueue_patch = "image_search_service.queue.worker.enqueue_person_ids_update"
        mock_enqueue = MagicMock(return_value=1)
        with (
            patch(enqueue_patch, mock_enqueue),
            patch(qdrant_patch, return_value=mock_qdrant_client),
        ):
            response = await test_client.post(
                f"/api/v1/faces/persons/{person_a.id}/photos/bulk-move",
                json={"photoIds": [test_asset.id], "toPersonId": str(person_b.id)},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["toPersonId"] == str(person_b.id)
        assert data["toPersonName"] == person_b.name
        assert data["updatedFaces"] == 1
        assert data["updatedPhotos"] == 1
        assert data["personCreated"] is False

        # Verify enqueue was called with the affected asset IDs
        mock_enqueue.assert_called_once()
        enqueued_ids = mock_enqueue.call_args[0][0]
        assert test_asset.id in enqueued_ids

        # Verify face was moved
        await db_session.refresh(face)
        assert face.person_id == person_b.id

    async def test_bulk_move_faces_create_new_person(
        self,
        test_client: AsyncClient,
        db_session: AsyncSession,
        person_a: Person,
        test_asset: ImageAsset,
    ):
        """POST /persons/{id}/photos/bulk-move creates new person when toPersonName provided."""
        # Create face for person A
        face = FaceInstance(
            id=uuid.uuid4(),
            asset_id=test_asset.id,
            bbox_x=100,
            bbox_y=100,
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.95,
            quality_score=0.85,
            qdrant_point_id=uuid.uuid4(),
            person_id=person_a.id,
        )
        db_session.add(face)
        await db_session.commit()

        # Mock enqueue helper and Qdrant operations
        mock_qdrant_client = MagicMock()
        mock_qdrant_client.update_person_ids = MagicMock(return_value=None)

        qdrant_patch = "image_search_service.vector.face_qdrant.get_face_qdrant_client"
        enqueue_patch = "image_search_service.queue.worker.enqueue_person_ids_update"
        mock_enqueue = MagicMock(return_value=1)
        with (
            patch(enqueue_patch, mock_enqueue),
            patch(qdrant_patch, return_value=mock_qdrant_client),
        ):
            response = await test_client.post(
                f"/api/v1/faces/persons/{person_a.id}/photos/bulk-move",
                json={"photoIds": [test_asset.id], "toPersonName": "New Person"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["toPersonName"] == "New Person"
        assert data["updatedFaces"] == 1
        assert data["updatedPhotos"] == 1
        assert data["personCreated"] is True

    async def test_bulk_move_target_not_found(
        self,
        test_client: AsyncClient,
        person_a: Person,
    ):
        """POST /persons/{id}/photos/bulk-move returns 404 for missing target person."""
        fake_target_id = uuid.uuid4()
        response = await test_client.post(
            f"/api/v1/faces/persons/{person_a.id}/photos/bulk-move",
            json={"photoIds": [1, 2], "toPersonId": str(fake_target_id)},
        )

        assert response.status_code == 404

    async def test_bulk_move_missing_destination(self, test_client: AsyncClient, person_a: Person):
        """POST /persons/{id}/photos/bulk-move returns 422 without destination."""
        response = await test_client.post(
            f"/api/v1/faces/persons/{person_a.id}/photos/bulk-move",
            json={"photoIds": [1, 2]},  # Missing both toPersonId and toPersonName
        )

        assert response.status_code == 422


# ============ Unassign Face Tests ============


@pytest.mark.asyncio
class TestUnassignFace:
    """Tests for DELETE /api/v1/faces/faces/{id}/person endpoint."""

    async def test_unassign_face_success(
        self,
        test_client: AsyncClient,
        db_session: AsyncSession,
        face_with_person: FaceInstance,
        person_a: Person,
    ):
        """DELETE /faces/{id}/person successfully unassigns face from person."""
        # Mock the update_person_ids method to be a no-op
        mock_qdrant_client = MagicMock()
        mock_qdrant_client.update_person_ids = MagicMock(return_value=None)

        qdrant_patch = "image_search_service.vector.face_qdrant.get_face_qdrant_client"
        with patch(qdrant_patch, return_value=mock_qdrant_client):
            response = await test_client.delete(f"/api/v1/faces/faces/{face_with_person.id}/person")

        assert response.status_code == 200
        data = response.json()
        assert data["faceId"] == str(face_with_person.id)
        assert data["previousPersonId"] == str(person_a.id)
        assert data["previousPersonName"] == person_a.name

        # Verify database was updated
        await db_session.refresh(face_with_person)
        assert face_with_person.person_id is None

    async def test_unassign_face_not_found(self, test_client: AsyncClient):
        """DELETE /faces/{id}/person returns 404 for missing face."""
        fake_id = uuid.uuid4()
        response = await test_client.delete(f"/api/v1/faces/faces/{fake_id}/person")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    async def test_unassign_face_not_assigned(
        self,
        test_client: AsyncClient,
        face_without_person: FaceInstance,
    ):
        """DELETE /faces/{id}/person returns 400 when face has no person."""
        response = await test_client.delete(f"/api/v1/faces/faces/{face_without_person.id}/person")

        assert response.status_code == 400
        assert "not assigned" in response.json()["detail"].lower()


# ============ Detect Single Asset Tests ============


@pytest.mark.asyncio
class TestDetectSingleAsset:
    """Tests for POST /api/v1/faces/detect/{asset_id} endpoint."""

    async def test_detect_single_asset_success(
        self,
        test_client: AsyncClient,
        test_asset: ImageAsset,
    ):
        """POST /detect/{asset_id} detects faces in single asset."""
        # Mock face service and sync session context manager
        sync_patch = "image_search_service.db.sync_operations.get_sync_session"
        service_patch = "image_search_service.faces.service.get_face_service"
        with patch(sync_patch) as mock_sync_session_func, patch(service_patch) as mock_service_func:
            # Setup mock face instances
            mock_face1 = MagicMock()
            mock_face1.id = uuid.uuid4()
            mock_face2 = MagicMock()
            mock_face2.id = uuid.uuid4()

            mock_service_instance = MagicMock()
            mock_service_instance.process_asset.return_value = [mock_face1, mock_face2]
            mock_service_func.return_value = mock_service_instance

            # Mock sync session context manager
            mock_sync_db = MagicMock()
            mock_sync_db.get.return_value = test_asset
            mock_context = MagicMock()
            mock_context.__enter__.return_value = mock_sync_db
            mock_context.__exit__.return_value = None
            mock_sync_session_func.return_value = mock_context

            response = await test_client.post(
                f"/api/v1/faces/detect/{test_asset.id}",
                json={"minConfidence": 0.8, "minFaceSize": 30},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["assetId"] == test_asset.id
            assert data["facesDetected"] == 2
            assert len(data["faceIds"]) == 2

    async def test_detect_single_asset_not_found(self, test_client: AsyncClient):
        """POST /detect/{asset_id} returns 404 for missing asset."""
        fake_id = 999999
        response = await test_client.post(
            f"/api/v1/faces/detect/{fake_id}", json={"minConfidence": 0.5}
        )

        assert response.status_code == 404

    async def test_detect_single_asset_invalid_confidence(
        self, test_client: AsyncClient, test_asset: ImageAsset
    ):
        """POST /detect/{asset_id} returns 422 for invalid confidence."""
        response = await test_client.post(
            f"/api/v1/faces/detect/{test_asset.id}",
            json={"minConfidence": 1.5},  # Invalid: > 1.0
        )

        assert response.status_code == 422


# ============ Cluster Basic Tests ============


@pytest.mark.asyncio
class TestClusterBasic:
    """Tests for POST /api/v1/faces/cluster endpoint (basic clustering)."""

    async def test_cluster_basic_success(
        self,
        test_client: AsyncClient,
    ):
        """POST /cluster triggers clustering with default params."""
        response = await test_client.post(
            "/api/v1/faces/cluster", json={"minClusterSize": 5, "qualityThreshold": 0.6}
        )

        assert response.status_code == 200
        data = response.json()
        assert "totalFaces" in data
        assert "clustersFound" in data
        assert "noiseCount" in data

    async def test_cluster_basic_custom_params(
        self,
        test_client: AsyncClient,
    ):
        """POST /cluster with custom clustering parameters."""
        response = await test_client.post(
            "/api/v1/faces/cluster",
            json={"minClusterSize": 3, "qualityThreshold": 0.7, "maxFaces": 1000},
        )

        assert response.status_code == 200

    async def test_cluster_basic_invalid_params(self, test_client: AsyncClient):
        """POST /cluster returns 422 for invalid parameters."""
        response = await test_client.post(
            "/api/v1/faces/cluster",
            json={"minClusterSize": -1},  # Invalid: negative
        )

        assert response.status_code == 422


# ============ Person Photos Tests ============


@pytest.mark.asyncio
class TestPersonPhotos:
    """Tests for GET /api/v1/faces/persons/{id}/photos endpoint."""

    async def test_person_photos_pagination(
        self,
        test_client: AsyncClient,
        db_session: AsyncSession,
        person_a: Person,
    ):
        """GET /persons/{id}/photos returns paginated photos."""
        # Create 3 different photos with faces for person A
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
                person_id=person_a.id,
            )
            db_session.add(face)

        await db_session.commit()

        # Get first page with page_size=2
        response = await test_client.get(
            f"/api/v1/faces/persons/{person_a.id}/photos",
            params={"page": 1, "page_size": 2},  # Use snake_case parameter name
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        # Pagination parameters are correct
        assert data["page"] == 1
        assert data["pageSize"] == 2
        # Should return at most page_size items (may be less if fewer total items)
        assert len(data["items"]) <= 2
        assert len(data["items"]) > 0  # Should have at least 1 item

        # Get second page - just verify it returns successfully
        response = await test_client.get(
            f"/api/v1/faces/persons/{person_a.id}/photos",
            params={"page": 2, "page_size": 2},  # Use snake_case parameter name
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert data["page"] == 2

    async def test_person_photos_empty(self, test_client: AsyncClient, person_a: Person):
        """GET /persons/{id}/photos returns empty list for person with no faces."""
        response = await test_client.get(f"/api/v1/faces/persons/{person_a.id}/photos")

        assert response.status_code == 200
        data = response.json()
        assert data["items"] == []
        assert data["total"] == 0
        assert data["page"] == 1
        assert data["personId"] == str(person_a.id)
        assert data["personName"] == person_a.name

    async def test_person_photos_not_found(self, test_client: AsyncClient):
        """GET /persons/{id}/photos returns 404 for missing person."""
        fake_id = uuid.uuid4()
        response = await test_client.get(f"/api/v1/faces/persons/{fake_id}/photos")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
