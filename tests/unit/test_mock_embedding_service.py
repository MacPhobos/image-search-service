"""Test infrastructure validation for MockEmbeddingService (SemanticMockEmbeddingService).

This file validates the mock embedding service used as test infrastructure.
It does NOT test production code. For production EmbeddingService tests,
see test_embedding_service.py.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np

from tests.conftest import MockEmbeddingService
from tests.constants import CLIP_EMBEDDING_DIM


def test_mock_embedding_service_embed_text_returns_vector() -> None:
    """Test that mock embedding service returns CLIP_EMBEDDING_DIM-dim vector for text."""
    service = MockEmbeddingService()

    vector = service.embed_text("a beautiful sunset")

    # Should return CLIP_EMBEDDING_DIM-dimensional vector
    assert len(vector) == CLIP_EMBEDDING_DIM
    # All values should be floats
    assert all(isinstance(v, float) for v in vector)
    # Vector should be L2-normalized (norm ≈ 1.0)
    assert abs(np.linalg.norm(vector) - 1.0) < 0.01


def test_mock_embedding_service_embed_image_returns_vector(
    temp_image_factory: Callable[..., Path],
) -> None:
    """Test that mock embedding service returns CLIP_EMBEDDING_DIM-dim vector for image."""
    service = MockEmbeddingService()

    # Create test image
    image_path = temp_image_factory("test.jpg", width=100, height=100)

    vector = service.embed_image(image_path)

    # Should return CLIP_EMBEDDING_DIM-dimensional vector
    assert len(vector) == CLIP_EMBEDDING_DIM
    # All values should be floats
    assert all(isinstance(v, float) for v in vector)
    # Vector should be L2-normalized (norm ≈ 1.0)
    assert abs(np.linalg.norm(vector) - 1.0) < 0.01


def test_mock_embedding_dim_correct() -> None:
    """Test that embedding_dim property returns correct dimension."""
    service = MockEmbeddingService()

    assert service.embedding_dim == CLIP_EMBEDDING_DIM


def test_mock_embedding_deterministic() -> None:
    """Test that same input produces same output (deterministic)."""
    service = MockEmbeddingService()

    # Same text should produce same vector
    vector1 = service.embed_text("test query")
    vector2 = service.embed_text("test query")

    assert vector1 == vector2


def test_mock_embedding_different_inputs_produce_different_vectors() -> None:
    """Test that different inputs produce different vectors."""
    service = MockEmbeddingService()

    vector1 = service.embed_text("sunset")
    vector2 = service.embed_text("mountain")

    # Vectors should be different
    assert vector1 != vector2


def test_mock_embedding_image_path_affects_output(temp_image_factory: Callable[..., Path]) -> None:
    """Test that different image paths produce different vectors."""
    service = MockEmbeddingService()

    # Create two different images
    image1 = temp_image_factory("img1.jpg")
    image2 = temp_image_factory("img2.jpg")

    vector1 = service.embed_image(image1)
    vector2 = service.embed_image(image2)

    # Different paths should produce different vectors
    assert vector1 != vector2
