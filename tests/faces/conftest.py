"""Fixtures for face pipeline tests."""

import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture
def mock_face_embedding():
    """Generate a random 512-dim normalized embedding."""
    embedding = np.random.randn(512).astype(np.float32)
    return (embedding / np.linalg.norm(embedding)).tolist()


@pytest.fixture
def mock_detected_face(mock_face_embedding):
    """Create a mock DetectedFace object."""
    from image_search_service.faces.detector import DetectedFace

    return DetectedFace(
        bbox=(100, 150, 80, 80),
        confidence=0.95,
        landmarks=np.array(
            [
                [120, 170],
                [160, 170],  # Eyes
                [140, 195],  # Nose
                [125, 220],
                [155, 220],  # Mouth
            ]
        ),
        embedding=np.array(mock_face_embedding),
        aligned_face=np.zeros((112, 112, 3), dtype=np.uint8),
    )


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
def mock_qdrant_client():
    """Create a mock FaceQdrantClient."""
    with patch("image_search_service.vector.face_qdrant.get_face_qdrant_client") as mock:
        client = MagicMock()
        client.ensure_collection.return_value = None
        client.upsert_face.return_value = None
        client.upsert_faces_batch.return_value = None
        client.search_similar_faces.return_value = []
        client.search_against_prototypes.return_value = []
        client.update_cluster_ids.return_value = None
        client.update_person_ids.return_value = None
        mock.return_value = client
        yield client


@pytest.fixture
def mock_insightface():
    """Mock InsightFace model loading."""
    with patch("image_search_service.faces.detector._ensure_model_loaded") as mock:
        app = MagicMock()
        mock.return_value = app
        yield app
