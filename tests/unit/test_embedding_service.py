"""Test embedding service (using mock to avoid loading OpenCLIP)."""

from collections.abc import Callable
from pathlib import Path

from tests.conftest import MockEmbeddingService


def test_mock_embedding_service_embed_text_returns_vector() -> None:
    """Test that mock embedding service returns 512-dim vector for text."""
    service = MockEmbeddingService()

    vector = service.embed_text("a beautiful sunset")

    # Should return 512-dimensional vector
    assert len(vector) == 512
    # All values should be floats
    assert all(isinstance(v, float) for v in vector)
    # Values should be normalized (between 0 and 1)
    assert all(0.0 <= v <= 1.0 for v in vector)


def test_mock_embedding_service_embed_image_returns_vector(
    temp_image_factory: Callable[..., Path]
) -> None:
    """Test that mock embedding service returns 512-dim vector for image."""
    service = MockEmbeddingService()

    # Create test image
    image_path = temp_image_factory("test.jpg", width=100, height=100)

    vector = service.embed_image(image_path)

    # Should return 512-dimensional vector
    assert len(vector) == 512
    # All values should be floats
    assert all(isinstance(v, float) for v in vector)
    # Values should be normalized
    assert all(0.0 <= v <= 1.0 for v in vector)


def test_mock_embedding_dim_correct() -> None:
    """Test that embedding_dim property returns correct dimension."""
    service = MockEmbeddingService()

    assert service.embedding_dim == 512


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


def test_mock_embedding_image_path_affects_output(
    temp_image_factory: Callable[..., Path]
) -> None:
    """Test that different image paths produce different vectors."""
    service = MockEmbeddingService()

    # Create two different images
    image1 = temp_image_factory("img1.jpg")
    image2 = temp_image_factory("img2.jpg")

    vector1 = service.embed_image(image1)
    vector2 = service.embed_image(image2)

    # Different paths should produce different vectors
    assert vector1 != vector2
