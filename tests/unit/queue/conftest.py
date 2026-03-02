"""Shared fixtures for queue job tests."""

import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock

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
    PersonStatus,
    TrainingStatus,
)
from tests.constants import FACE_EMBEDDING_DIM


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
        VectorParams(size=FACE_EMBEDDING_DIM, distance=Distance.COSINE),
    )
    return client


@pytest.fixture
def mock_face_qdrant():
    """Mock FaceQdrantClient."""
    mock = MagicMock()
    mock.get_embedding_by_point_id = MagicMock(return_value=[0.1] * FACE_EMBEDDING_DIM)
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
            "embedding": np.random.randn(FACE_EMBEDDING_DIM).tolist(),
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
