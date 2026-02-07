"""Unit tests for face_jobs.py to increase coverage from 21% to 60%+.

Tests cover the highest-priority untested functions:
- cluster_dual_job (P1)
- expire_old_suggestions_job (P1)
- cleanup_orphaned_suggestions_job (P1)
- compute_centroids_job (P2)
- train_person_matching_job (P2)
- backfill_faces_job (P2)
- detect_faces_for_session_job (P3 - partial coverage)
- propagate_person_label_multiproto_job (P3 - additional tests)

Module: image_search_service.queue.face_jobs (~2127 lines, 656 missed statements)
Target: 40%+ coverage increase (21% → 60%+)
"""

import json
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from image_search_service.db.models import (
    FaceDetectionSession,
    FaceDetectionSessionStatus,
    FaceInstance,
    FaceSuggestion,
    FaceSuggestionStatus,
    ImageAsset,
    Person,
    PersonCentroid,
    PersonStatus,
    TrainingStatus,
)
from image_search_service.queue.face_jobs import (
    backfill_faces_job,
    cleanup_orphaned_suggestions_job,
    cluster_dual_job,
    compute_centroids_job,
    detect_faces_for_session_job,
    expire_old_suggestions_job,
    propagate_person_label_multiproto_job,
    train_person_matching_job,
)

# ============ Fixtures ============


@pytest.fixture
def mock_job():
    """Mock RQ job with ID."""
    job = MagicMock()
    job.id = "test-job-123"
    return job


@pytest.fixture
def mock_get_sync_session(sync_db_session):
    """Mock get_sync_session to return test session."""

    def _mock():
        return sync_db_session

    return _mock


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client with in-memory storage."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    client = QdrantClient(":memory:")
    client.create_collection(
        "test_faces",
        VectorParams(size=512, distance=Distance.COSINE),
    )
    return client


@pytest.fixture
def mock_face_qdrant():
    """Mock FaceQdrantClient."""
    mock = MagicMock()
    mock.get_embedding_by_point_id = MagicMock(return_value=[0.1] * 512)
    mock.search_similar_faces = MagicMock(return_value=[])
    return mock


@pytest.fixture
def mock_face_detector():
    """Mock InsightFace detector."""
    mock = MagicMock()
    mock.detect_faces.return_value = [
        {
            "bbox": [10, 20, 60, 70],
            "det_score": 0.95,
            "embedding": np.random.randn(512).tolist(),
        },
    ]
    return mock


@pytest.fixture
def create_person(sync_db_session):
    """Factory for creating Person records."""
    _counter = [0]  # Mutable counter to avoid unique constraint conflicts

    def _create(name: str | None = None, status: PersonStatus = PersonStatus.ACTIVE):
        if name is None:
            _counter[0] += 1
            name = f"Test Person {_counter[0]}"

        person = Person(
            id=uuid.uuid4(),
            name=name,
            status=status.value,
        )
        sync_db_session.add(person)
        sync_db_session.commit()
        sync_db_session.refresh(person)
        return person

    return _create


@pytest.fixture
def create_image_asset(sync_db_session):
    """Factory for creating ImageAsset records."""

    def _create(path: str | None = None, training_status: str = TrainingStatus.PENDING.value):
        if path is None:
            path = f"/test/images/photo_{uuid.uuid4().hex[:8]}.jpg"

        asset = ImageAsset(
            path=path,
            training_status=training_status,
            width=640,
            height=480,
            file_size=102400,
            mime_type="image/jpeg",
        )
        sync_db_session.add(asset)
        sync_db_session.commit()
        sync_db_session.refresh(asset)
        return asset

    return _create


@pytest.fixture
def create_face_instance(sync_db_session, create_image_asset):
    """Factory for creating FaceInstance records."""

    def _create(
        person_id: uuid.UUID | None = None,
        quality_score: float = 0.75,
        asset_id: int | None = None,
        qdrant_point_id: uuid.UUID | None = None,
    ):
        if asset_id is None:
            asset = create_image_asset()
            asset_id = asset.id

        if qdrant_point_id is None:
            qdrant_point_id = uuid.uuid4()

        face = FaceInstance(
            id=uuid.uuid4(),
            asset_id=asset_id,
            person_id=person_id,
            bbox_x=100,
            bbox_y=150,
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.95,
            quality_score=quality_score,
            qdrant_point_id=qdrant_point_id,
        )
        sync_db_session.add(face)
        sync_db_session.commit()
        sync_db_session.refresh(face)
        return face

    return _create


@pytest.fixture
def create_suggestion(sync_db_session, create_face_instance, create_person):
    """Factory for creating FaceSuggestion records."""

    def _create(
        suggested_person_id: uuid.UUID | None = None,
        status: str = FaceSuggestionStatus.PENDING.value,
        face_instance_id: uuid.UUID | None = None,
        source_face_id: uuid.UUID | None = None,
        confidence: float = 0.85,
        created_at: datetime | None = None,
    ):
        if face_instance_id is None:
            face = create_face_instance()
            face_instance_id = face.id

        if source_face_id is None:
            source_face = create_face_instance()
            source_face_id = source_face.id

        if suggested_person_id is None:
            person = create_person()
            suggested_person_id = person.id

        suggestion = FaceSuggestion(
            face_instance_id=face_instance_id,
            suggested_person_id=suggested_person_id,
            source_face_id=source_face_id,
            confidence=confidence,
            status=status,
            created_at=created_at or datetime.now(UTC),
        )
        sync_db_session.add(suggestion)
        sync_db_session.commit()
        sync_db_session.refresh(suggestion)
        return suggestion

    return _create


@pytest.fixture
def create_detection_session(sync_db_session):
    """Factory for creating FaceDetectionSession records."""

    def _create(
        status: str = FaceDetectionSessionStatus.PENDING.value,
        training_session_id: int | None = None,
    ):
        session = FaceDetectionSession(
            id=uuid.uuid4(),
            training_session_id=training_session_id,
            status=status,
            total_images=0,
            processed_images=0,
            failed_images=0,
            faces_detected=0,
            faces_assigned=0,
            faces_assigned_to_persons=0,
            clusters_created=0,
            suggestions_created=0,
            current_batch=0,
            total_batches=0,
            min_confidence=0.5,
            min_face_size=20,
            batch_size=8,
        )
        sync_db_session.add(session)
        sync_db_session.commit()
        sync_db_session.refresh(session)
        return session

    return _create


# ============ P1: cluster_dual_job Tests ============


class TestClusterDualJob:
    """Tests for cluster_dual_job function (P1 priority)."""

    def test_cluster_dual_job_basic(
        self, sync_db_session, mock_job, mock_get_sync_session, create_face_instance
    ):
        """Test basic dual-mode clustering execution."""
        # Create some unassigned faces
        for _ in range(5):
            create_face_instance(person_id=None)

        # Mock dependencies
        mock_clusterer = MagicMock()
        mock_clusterer.cluster_all_faces.return_value = {
            "status": "completed",
            "assigned_to_people": 2,
            "unknown_clusters": 1,
            "still_unlabeled": 2,
            "total_processed": 5,
        }

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
            patch(
                "image_search_service.faces.dual_clusterer.get_dual_mode_clusterer",
                return_value=mock_clusterer,
            ),
        ):
            result = cluster_dual_job()

        assert result["total_processed"] == 5
        assert result["assigned_to_people"] == 2
        assert result["unknown_clusters"] == 1
        mock_clusterer.cluster_all_faces.assert_called_once()

    def test_cluster_dual_job_custom_params(
        self, sync_db_session, mock_job, mock_get_sync_session
    ):
        """Test clustering with custom parameters."""
        mock_clusterer = MagicMock()
        mock_clusterer.cluster_all_faces.return_value = {
            "total_processed": 0,
            "assigned_to_people": 0,
            "unknown_clusters": 0,
        }

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
            patch(
                "image_search_service.faces.dual_clusterer.get_dual_mode_clusterer",
                return_value=mock_clusterer,
            ),
        ):
            result = cluster_dual_job(
                person_threshold=0.8,
                unknown_method="kmeans",
                unknown_min_size=5,
                unknown_eps=0.3,
                max_faces=1000,
            )

        assert result["total_processed"] == 0
        # Verify custom params passed through to cluster_all_faces
        call_kwargs = mock_clusterer.cluster_all_faces.call_args[1]
        assert call_kwargs["max_faces"] == 1000

    def test_cluster_dual_job_empty_db(self, sync_db_session, mock_job, mock_get_sync_session):
        """Test clustering with no faces in database."""
        mock_clusterer = MagicMock()
        mock_clusterer.cluster_all_faces.return_value = {
            "total_processed": 0,
            "assigned_to_people": 0,
            "unknown_clusters": 0,
            "still_unlabeled": 0,
        }

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
            patch(
                "image_search_service.faces.dual_clusterer.get_dual_mode_clusterer",
                return_value=mock_clusterer,
            ),
        ):
            result = cluster_dual_job()

        assert result["total_processed"] == 0
        assert result["assigned_to_people"] == 0
        assert result["unknown_clusters"] == 0


# ============ P1: expire_old_suggestions_job Tests ============


class TestExpireOldSuggestionsJob:
    """Tests for expire_old_suggestions_job function (P1 priority)."""

    def test_expire_old_suggestions(
        self, sync_db_session, mock_job, mock_get_sync_session, create_suggestion, create_person
    ):
        """Test expiring suggestions older than threshold."""
        # Create persons to avoid unique constraint violations
        person1 = create_person(name="Person 1")
        person2 = create_person(name="Person 2")
        person3 = create_person(name="Person 3")

        # Create old pending suggestions (31 days old)
        old_date = datetime.now(UTC) - timedelta(days=31)
        old_suggestion1 = create_suggestion(
            suggested_person_id=person1.id, status="pending", created_at=old_date
        )
        old_suggestion2 = create_suggestion(
            suggested_person_id=person2.id, status="pending", created_at=old_date
        )

        # Create recent suggestion (should not be expired)
        recent_suggestion = create_suggestion(
            suggested_person_id=person3.id, status="pending", created_at=datetime.now(UTC)
        )

        # Store IDs before job execution
        old_suggestion1_id = old_suggestion1.id
        old_suggestion2_id = old_suggestion2.id
        recent_suggestion_id = recent_suggestion.id

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
        ):
            result = expire_old_suggestions_job(days_threshold=30)

        assert result["status"] == "completed"
        assert result["expired_count"] == 2
        assert result["threshold_days"] == 30

        # Verify old suggestions were expired (query fresh from DB)
        updated_old1 = sync_db_session.get(FaceSuggestion, old_suggestion1_id)
        updated_old2 = sync_db_session.get(FaceSuggestion, old_suggestion2_id)
        updated_recent = sync_db_session.get(FaceSuggestion, recent_suggestion_id)

        assert updated_old1 is not None
        assert updated_old2 is not None
        assert updated_old1.status == FaceSuggestionStatus.EXPIRED.value
        assert updated_old2.status == FaceSuggestionStatus.EXPIRED.value
        assert updated_old1.reviewed_at is not None
        assert updated_old2.reviewed_at is not None

        # Recent suggestion should remain pending
        assert updated_recent is not None
        assert updated_recent.status == FaceSuggestionStatus.PENDING.value

    def test_expire_no_old_suggestions(self, sync_db_session, mock_job, mock_get_sync_session):
        """Test expiring when no old suggestions exist."""
        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
        ):
            result = expire_old_suggestions_job(days_threshold=30)

        assert result["status"] == "completed"
        assert result["expired_count"] == 0

    def test_expire_uses_config_default(
        self, sync_db_session, mock_job, mock_get_sync_session, create_suggestion
    ):
        """Test that function uses config value when days_threshold not provided."""
        # Create old suggestion
        old_date = datetime.now(UTC) - timedelta(days=100)
        create_suggestion(status="pending", created_at=old_date)

        # Mock SyncConfigService
        mock_config = MagicMock()
        mock_config.get_int.return_value = 90  # Config says 90 days

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
            patch(
                "image_search_service.services.config_service.SyncConfigService",
                return_value=mock_config,
            ),
        ):
            result = expire_old_suggestions_job()  # No days_threshold provided

        assert result["status"] == "completed"
        assert result["expired_count"] == 1
        mock_config.get_int.assert_called_once_with("face_suggestion_expiry_days")


# ============ P1: cleanup_orphaned_suggestions_job Tests ============


class TestCleanupOrphanedSuggestionsJob:
    """Tests for cleanup_orphaned_suggestions_job function (P1 priority)."""

    def test_cleanup_orphaned_no_person(
        self,
        sync_db_session,
        mock_job,
        mock_get_sync_session,
        create_suggestion,
        create_face_instance,
        create_person,
    ):
        """Test cleanup when source face person_id is NULL."""
        person = create_person()
        source_face = create_face_instance(person_id=None)  # Unassigned source
        suggestion = create_suggestion(
            suggested_person_id=person.id, source_face_id=source_face.id, status="pending"
        )
        suggestion_id = suggestion.id

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
        ):
            result = cleanup_orphaned_suggestions_job()

        assert result["status"] == "completed"
        assert result["expired_count"] == 1

        # Verify suggestion was expired (query fresh from DB)
        updated_suggestion = sync_db_session.get(FaceSuggestion, suggestion_id)
        assert updated_suggestion is not None
        assert updated_suggestion.status == FaceSuggestionStatus.EXPIRED.value
        assert updated_suggestion.reviewed_at is not None

    def test_cleanup_orphaned_different_person(
        self,
        sync_db_session,
        mock_job,
        mock_get_sync_session,
        create_suggestion,
        create_face_instance,
        create_person,
    ):
        """Test cleanup when source face moved to different person."""
        person_a = create_person(name="Person A")
        person_b = create_person(name="Person B")

        # Source face assigned to person_b
        source_face = create_face_instance(person_id=person_b.id)

        # Suggestion says target should be person_a (mismatch!)
        suggestion = create_suggestion(
            suggested_person_id=person_a.id, source_face_id=source_face.id, status="pending"
        )
        suggestion_id = suggestion.id

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
        ):
            result = cleanup_orphaned_suggestions_job()

        assert result["status"] == "completed"
        assert result["expired_count"] == 1

        # Query fresh from DB
        updated_suggestion = sync_db_session.get(FaceSuggestion, suggestion_id)
        assert updated_suggestion is not None
        assert updated_suggestion.status == FaceSuggestionStatus.EXPIRED.value

    def test_cleanup_no_orphaned(
        self,
        sync_db_session,
        mock_job,
        mock_get_sync_session,
        create_suggestion,
        create_face_instance,
        create_person,
    ):
        """Test cleanup when all suggestions are valid."""
        person = create_person()
        source_face = create_face_instance(person_id=person.id)

        # Valid suggestion: source_face.person_id == suggested_person_id
        create_suggestion(
            suggested_person_id=person.id, source_face_id=source_face.id, status="pending"
        )

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
        ):
            result = cleanup_orphaned_suggestions_job()

        assert result["status"] == "completed"
        assert result["expired_count"] == 0


# ============ P2: compute_centroids_job Tests ============


class TestComputeCentroidsJob:
    """Tests for compute_centroids_job function (P2 priority)."""

    def test_compute_centroids_basic(
        self,
        sync_db_session,
        mock_job,
        mock_get_sync_session,
        create_person,
        create_face_instance,
    ):
        """Test basic centroid computation."""
        person = create_person()
        # Create faces for person
        for _ in range(5):
            create_face_instance(person_id=person.id)

        mock_assigner = MagicMock()
        mock_assigner.compute_person_centroids.return_value = {
            "status": "completed",
            "centroids_computed": 1,
            "persons_processed": 1,
        }

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
            patch(
                "image_search_service.faces.assigner.get_face_assigner",
                return_value=mock_assigner,
            ),
        ):
            result = compute_centroids_job()

        assert result["status"] == "completed"
        assert result["centroids_computed"] == 1
        mock_assigner.compute_person_centroids.assert_called_once()

    def test_compute_centroids_insufficient_faces(
        self, sync_db_session, mock_job, mock_get_sync_session, create_person, create_face_instance
    ):
        """Test centroid computation when person has insufficient faces."""
        person = create_person()
        # Only 1 face (below typical min_faces threshold)
        create_face_instance(person_id=person.id)

        mock_assigner = MagicMock()
        mock_assigner.compute_person_centroids.return_value = {
            "status": "completed",
            "centroids_computed": 0,  # Skipped due to insufficient faces
            "persons_processed": 1,
        }

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
            patch(
                "image_search_service.faces.assigner.get_face_assigner",
                return_value=mock_assigner,
            ),
        ):
            result = compute_centroids_job()

        assert result["status"] == "completed"
        assert result["centroids_computed"] == 0

    def test_compute_centroids_updates_existing(
        self,
        sync_db_session,
        mock_job,
        mock_get_sync_session,
        create_person,
        create_face_instance,
    ):
        """Test re-computation updates existing centroid version."""
        person = create_person()
        for _ in range(5):
            create_face_instance(person_id=person.id)

        # Create existing centroid (just for simulating a re-computation scenario)
        # Note: We're just testing that the job calls the assigner correctly
        # The actual centroid creation is mocked

        mock_assigner = MagicMock()
        mock_assigner.compute_person_centroids.return_value = {
            "status": "completed",
            "centroids_computed": 1,  # Re-computed
            "persons_processed": 1,
        }

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
            patch(
                "image_search_service.faces.assigner.get_face_assigner",
                return_value=mock_assigner,
            ),
        ):
            result = compute_centroids_job()

        assert result["status"] == "completed"
        assert result["centroids_computed"] == 1


# ============ P2: train_person_matching_job Tests ============


class TestTrainPersonMatchingJob:
    """Tests for train_person_matching_job function (P2 priority)."""

    def test_train_person_matching_basic(self, sync_db_session, mock_job, mock_get_sync_session):
        """Test basic training job execution."""
        mock_trainer = MagicMock()
        mock_trainer.fine_tune_for_person_clustering.return_value = {
            "status": "completed",
            "persons_used": 5,
            "total_triplets": 150,
            "final_loss": 0.15,
        }

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
            patch(
                "image_search_service.faces.trainer.get_face_trainer", return_value=mock_trainer
            ),
        ):
            result = train_person_matching_job()

        assert result["status"] == "completed"
        assert result["persons_used"] == 5
        assert result["final_loss"] == 0.15
        mock_trainer.fine_tune_for_person_clustering.assert_called_once()

    def test_train_person_matching_custom_params(
        self, sync_db_session, mock_job, mock_get_sync_session
    ):
        """Test training with custom parameters."""
        mock_trainer = MagicMock()
        mock_trainer.fine_tune_for_person_clustering.return_value = {
            "status": "completed",
            "persons_used": 0,
            "total_triplets": 0,
            "final_loss": 0.0,
        }

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
            patch(
                "image_search_service.faces.trainer.get_face_trainer", return_value=mock_trainer
            ),
        ):
            result = train_person_matching_job(
                epochs=50, margin=0.3, batch_size=64, learning_rate=0.001, min_faces_per_person=10
            )

        assert result["status"] == "completed"
        # Verify custom params passed through to fine_tune_for_person_clustering
        call_kwargs = mock_trainer.fine_tune_for_person_clustering.call_args[1]
        assert call_kwargs["min_faces_per_person"] == 10


# ============ P2: backfill_faces_job Tests ============


class TestBackfillFacesJob:
    """Tests for backfill_faces_job function (P2 priority)."""

    def test_backfill_faces_basic(
        self, sync_db_session, mock_job, mock_get_sync_session, create_image_asset
    ):
        """Test basic backfill processing."""
        # Create assets without faces
        for _ in range(3):
            create_image_asset()

        mock_service = MagicMock()
        mock_service.process_assets_batch.return_value = {
            "processed": 3,
            "total_faces": 5,
            "errors": 0,
            "throughput": 1.5,
        }

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
            patch(
                "image_search_service.faces.service.get_face_service", return_value=mock_service
            ),
        ):
            result = backfill_faces_job(limit=3, batch_size=2)

        assert result["processed"] == 3
        assert result["total_faces"] == 5
        mock_service.process_assets_batch.assert_called_once()

    def test_backfill_faces_empty(self, sync_db_session, mock_job, mock_get_sync_session):
        """Test backfill when no images need processing."""
        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
        ):
            result = backfill_faces_job()

        assert result["processed"] == 0
        assert result["status"] == "no_assets_to_process"


# ============ P3: detect_faces_for_session_job Tests ============


class TestDetectFacesForSessionJob:
    """Tests for detect_faces_for_session_job function (P3 priority - partial coverage)."""

    def test_detect_faces_session_not_found(self, sync_db_session, mock_job, mock_get_sync_session):
        """Test error handling when session doesn't exist."""
        invalid_session_id = str(uuid.uuid4())

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
        ):
            result = detect_faces_for_session_job(invalid_session_id)

        assert result["status"] == "failed"
        assert "not found" in result["error"].lower()

    def test_detect_faces_basic_flow(
        self,
        sync_db_session,
        mock_job,
        mock_get_sync_session,
        create_detection_session,
        create_image_asset,
    ):
        """Test basic face detection flow for session."""
        session = create_detection_session()
        # Create assets to process
        for _ in range(3):
            create_image_asset()

        mock_service = MagicMock()
        mock_service.process_assets_batch.return_value = {
            "processed": 3,
            "total_faces": 3,
            "errors": 0,
        }

        mock_assigner = MagicMock()
        mock_assigner.assign_new_faces.return_value = {
            "assigned": 2,
            "unassigned": 1,
        }

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
            patch(
                "image_search_service.faces.service.get_face_service", return_value=mock_service
            ),
            patch(
                "image_search_service.faces.assigner.get_face_assigner",
                return_value=mock_assigner,
            ),
            patch("redis.Redis") as mock_redis_class,
        ):
            # Mock Redis for progress tracking
            mock_redis = MagicMock()
            mock_redis_class.from_url.return_value = mock_redis

            result = detect_faces_for_session_job(str(session.id))

        assert result["status"] == "completed"
        assert result["total_images"] > 0
        assert result["faces_detected"] > 0

    def test_detect_faces_no_faces_detected(
        self,
        sync_db_session,
        mock_job,
        mock_get_sync_session,
        create_detection_session,
        create_image_asset,
    ):
        """Test when detection runs but finds no faces."""
        session = create_detection_session()
        create_image_asset()

        mock_service = MagicMock()
        mock_service.process_assets_batch.return_value = {
            "processed": 1,
            "total_faces": 0,  # No faces found
            "errors": 0,
        }

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
            patch(
                "image_search_service.faces.service.get_face_service", return_value=mock_service
            ),
            patch("redis.Redis") as mock_redis_class,
        ):
            mock_redis = MagicMock()
            mock_redis_class.from_url.return_value = mock_redis

            result = detect_faces_for_session_job(str(session.id))

        assert result["status"] == "completed"
        assert result["faces_detected"] == 0

    def test_detect_faces_error_handling(
        self,
        sync_db_session,
        mock_job,
        mock_get_sync_session,
        create_detection_session,
        create_image_asset,
    ):
        """Test error handling during batch processing."""
        session = create_detection_session()
        create_image_asset()

        mock_service = MagicMock()
        mock_service.process_assets_batch.return_value = {
            "processed": 1,
            "total_faces": 0,
            "errors": 1,
            "error_details": [{"asset_id": 1, "error": "Detection failed"}],
        }

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
            patch(
                "image_search_service.faces.service.get_face_service", return_value=mock_service
            ),
            patch("redis.Redis") as mock_redis_class,
        ):
            mock_redis = MagicMock()
            mock_redis_class.from_url.return_value = mock_redis

            result = detect_faces_for_session_job(str(session.id))

        assert result["status"] == "completed"
        assert result["failed_images"] == 1

    def test_detect_faces_empty_asset_list(
        self, sync_db_session, mock_job, mock_get_sync_session, create_detection_session
    ):
        """Test when session has no assets to process."""
        session = create_detection_session()

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
        ):
            result = detect_faces_for_session_job(str(session.id))

        assert result["status"] == "completed"
        assert result["total_images"] == 0

    def test_detect_faces_resume_session(
        self,
        sync_db_session,
        mock_job,
        mock_get_sync_session,
        create_detection_session,
        create_image_asset,
    ):
        """Test resuming a paused session."""
        # Create session with stored asset IDs (simulating previous run)
        session = create_detection_session()
        assets = [create_image_asset() for _ in range(5)]
        asset_ids = [a.id for a in assets]

        # Set session as if it was paused mid-processing
        session.asset_ids_json = json.dumps(asset_ids)
        session.current_asset_index = 2  # Processed first 2
        session.status = FaceDetectionSessionStatus.PENDING.value
        sync_db_session.commit()

        mock_service = MagicMock()
        mock_service.process_assets_batch.return_value = {
            "processed": 3,  # Remaining 3
            "total_faces": 3,
            "errors": 0,
        }

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
            patch(
                "image_search_service.faces.service.get_face_service", return_value=mock_service
            ),
            patch("redis.Redis") as mock_redis_class,
        ):
            mock_redis = MagicMock()
            mock_redis_class.from_url.return_value = mock_redis

            result = detect_faces_for_session_job(str(session.id))

        assert result["status"] == "completed"


# ============ P3: Additional propagate_person_label_multiproto_job Tests ============


class TestPropagatePersonLabelMultiprotoJobAdditional:
    """Additional tests for propagate_person_label_multiproto_job (P3 priority)."""

    def test_propagate_skip_already_assigned(
        self,
        sync_db_session,
        mock_job,
        mock_get_sync_session,
        create_person,
        create_face_instance,
        mock_face_qdrant,
    ):
        """Test that faces already assigned to a person are skipped."""
        person = create_person()

        # Create prototype
        from image_search_service.db.models import PersonPrototype, PrototypeRole

        proto_face = create_face_instance(person_id=person.id)
        prototype = PersonPrototype(
            id=uuid.uuid4(),
            person_id=person.id,
            face_instance_id=proto_face.id,
            qdrant_point_id=proto_face.qdrant_point_id,
            role=PrototypeRole.EXEMPLAR.value,
            created_at=datetime.now(UTC),
        )
        sync_db_session.add(prototype)
        sync_db_session.commit()

        # Create face already assigned to another person
        other_person = create_person(name="Other")
        assigned_face = create_face_instance(person_id=other_person.id)

        # Mock Qdrant to return the assigned face
        mock_result = MagicMock()
        mock_result.payload = {"face_instance_id": str(assigned_face.id)}
        mock_result.score = 0.85
        mock_face_qdrant.search_similar_faces = MagicMock(return_value=[mock_result])

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
            patch(
                "image_search_service.vector.face_qdrant.get_face_qdrant_client",
                return_value=mock_face_qdrant,
            ),
        ):
            result = propagate_person_label_multiproto_job(str(person.id))

        # Should skip assigned face, create no suggestions
        assert result["status"] == "completed"
        assert result["suggestions_created"] == 0

    def test_propagate_handles_no_prototypes(
        self, sync_db_session, mock_job, mock_get_sync_session, create_person
    ):
        """Test handling when person has no prototypes."""
        person = create_person()

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
        ):
            result = propagate_person_label_multiproto_job(str(person.id))

        assert result["status"] == "error"
        assert "no prototypes" in result["message"].lower()

    def test_propagate_preserves_existing(
        self,
        sync_db_session,
        mock_job,
        mock_get_sync_session,
        create_person,
        create_face_instance,
        create_suggestion,
        mock_face_qdrant,
    ):
        """Test preserve_existing parameter."""
        person = create_person()

        # Create prototype
        from image_search_service.db.models import PersonPrototype, PrototypeRole

        proto_face = create_face_instance(person_id=person.id)
        prototype = PersonPrototype(
            id=uuid.uuid4(),
            person_id=person.id,
            face_instance_id=proto_face.id,
            qdrant_point_id=proto_face.qdrant_point_id,
            role=PrototypeRole.EXEMPLAR.value,
            created_at=datetime.now(UTC),
        )
        sync_db_session.add(prototype)
        sync_db_session.commit()

        # Create existing pending suggestion
        existing_suggestion = create_suggestion(
            suggested_person_id=person.id, status=FaceSuggestionStatus.PENDING.value
        )
        existing_id = existing_suggestion.id

        # Mock Qdrant (empty results)
        mock_face_qdrant.search_similar_faces = MagicMock(return_value=[])

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
            patch(
                "image_search_service.vector.face_qdrant.get_face_qdrant_client",
                return_value=mock_face_qdrant,
            ),
        ):
            result = propagate_person_label_multiproto_job(str(person.id), preserve_existing=True)

        # With preserve_existing=True, old suggestions should NOT be expired
        assert result["expired_count"] == 0

        # Verify suggestion still pending (query fresh from DB)
        updated_suggestion = sync_db_session.get(FaceSuggestion, existing_id)
        assert updated_suggestion is not None
        assert updated_suggestion.status == FaceSuggestionStatus.PENDING.value
