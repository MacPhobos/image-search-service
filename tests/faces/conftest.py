"""Fixtures for face pipeline tests."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# mock_image_asset, mock_face_instance, mock_person from root conftest.py


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
