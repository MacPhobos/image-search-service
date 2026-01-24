"""Test SigLIP embedding service (using mock to avoid loading model)."""

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image


class MockSigLIPEmbeddingService:
    """Mock SigLIP embedding service that returns deterministic 768-dim vectors."""

    @property
    def embedding_dim(self) -> int:
        """Return SigLIP embedding dimension (768)."""
        return 768

    def embed_text(self, text: str) -> list[float]:
        """Generate deterministic 768-dim vector from text.

        Args:
            text: Text to embed

        Returns:
            Deterministic 768-dim vector normalized to [0, 1]
        """
        import hashlib

        # Use SHA256 hash for more entropy (need 768 values)
        h = hashlib.sha256(text.encode()).hexdigest()

        # Generate 768 values from hash
        vector = []
        for i in range(768):
            # Take 2 hex chars, convert to int, normalize to [0, 1]
            idx = (i * 2) % len(h)
            val = int(h[idx : idx + 2], 16) / 255.0
            vector.append(val)

        return vector

    def embed_image(self, image_path: str | Path) -> list[float]:
        """Generate deterministic 768-dim vector from image path.

        Args:
            image_path: Path to image file

        Returns:
            Deterministic 768-dim vector
        """
        # Use path as deterministic seed
        return self.embed_text(str(image_path))

    def embed_image_from_pil(self, image: "Image.Image") -> list[float]:  # type: ignore[name-defined]
        """Generate deterministic vector from PIL image.

        Args:
            image: PIL Image object

        Returns:
            Deterministic 768-dim vector
        """
        # Use image size as seed for deterministic output
        return self.embed_text(f"{image.width}x{image.height}")

    def embed_images_batch(
        self, images: list["Image.Image"], batch_size: int = 32  # type: ignore[name-defined]
    ) -> list[list[float]]:
        """Generate deterministic vectors for batch of images.

        Args:
            images: List of PIL Image objects
            batch_size: Batch size (ignored in mock)

        Returns:
            List of deterministic 768-dim vectors
        """
        return [self.embed_image_from_pil(img) for img in images]


def test_mock_siglip_embed_text_returns_768_dim_vector() -> None:
    """Test that mock SigLIP service returns 768-dim vector for text."""
    service = MockSigLIPEmbeddingService()

    vector = service.embed_text("a beautiful sunset")

    # Should return 768-dimensional vector (SigLIP dimension)
    assert len(vector) == 768
    # All values should be floats
    assert all(isinstance(v, float) for v in vector)
    # Values should be normalized (between 0 and 1)
    assert all(0.0 <= v <= 1.0 for v in vector)


def test_mock_siglip_embed_image_returns_768_dim_vector(
    temp_image_factory: Callable[..., Path]
) -> None:
    """Test that mock SigLIP service returns 768-dim vector for image."""
    service = MockSigLIPEmbeddingService()

    # Create test image
    image_path = temp_image_factory("test.jpg", width=100, height=100)

    vector = service.embed_image(image_path)

    # Should return 768-dimensional vector
    assert len(vector) == 768
    # All values should be floats
    assert all(isinstance(v, float) for v in vector)
    # Values should be normalized
    assert all(0.0 <= v <= 1.0 for v in vector)


def test_mock_siglip_embedding_dim_correct() -> None:
    """Test that embedding_dim property returns 768 for SigLIP."""
    service = MockSigLIPEmbeddingService()

    assert service.embedding_dim == 768


def test_mock_siglip_deterministic() -> None:
    """Test that same input produces same output (deterministic)."""
    service = MockSigLIPEmbeddingService()

    # Same text should produce same vector
    vector1 = service.embed_text("test query")
    vector2 = service.embed_text("test query")

    assert vector1 == vector2


def test_mock_siglip_different_inputs_produce_different_vectors() -> None:
    """Test that different inputs produce different vectors."""
    service = MockSigLIPEmbeddingService()

    vector1 = service.embed_text("sunset")
    vector2 = service.embed_text("mountain")

    # Vectors should be different
    assert vector1 != vector2


def test_mock_siglip_batch_embedding_returns_correct_count() -> None:
    """Test that batch embedding returns correct number of vectors."""
    from PIL import Image

    service = MockSigLIPEmbeddingService()

    # Create mock images
    images = [
        Image.new("RGB", (100, 100)),
        Image.new("RGB", (200, 200)),
        Image.new("RGB", (150, 150)),
    ]

    vectors = service.embed_images_batch(images)

    # Should return same number of vectors as images
    assert len(vectors) == 3
    # Each vector should be 768-dimensional
    assert all(len(v) == 768 for v in vectors)


@patch("image_search_service.services.siglip_embedding._load_model")
def test_siglip_service_lazy_loading(mock_load_model: MagicMock) -> None:
    """Test that SigLIP model is lazy loaded."""
    from image_search_service.services.siglip_embedding import SigLIPEmbeddingService

    # Create mock return values
    mock_model = MagicMock()
    mock_model.visual.output_dim = 768
    mock_preprocess = MagicMock()
    mock_tokenizer = MagicMock()
    mock_load_model.return_value = (mock_model, mock_preprocess, mock_tokenizer)

    service = SigLIPEmbeddingService()

    # Model should not be loaded yet
    assert mock_load_model.call_count == 0

    # Access embedding_dim property (triggers load)
    _ = service.embedding_dim

    # Now model should be loaded
    assert mock_load_model.call_count == 1


@patch("image_search_service.services.siglip_embedding._load_model")
@patch("image_search_service.services.siglip_embedding.get_settings")
def test_siglip_service_uses_correct_config(
    mock_get_settings: MagicMock, mock_load_model: MagicMock
) -> None:
    """Test that SigLIP service uses correct configuration values."""
    from image_search_service.services.siglip_embedding import SigLIPEmbeddingService

    # Mock settings
    mock_settings = MagicMock()
    mock_settings.siglip_model_name = "ViT-B-16-SigLIP"
    mock_settings.siglip_pretrained = "webli"
    mock_settings.siglip_embedding_dim = 768
    mock_get_settings.return_value = mock_settings

    # Mock model
    mock_model = MagicMock()
    mock_model.visual.output_dim = 768
    mock_preprocess = MagicMock()
    mock_tokenizer = MagicMock()
    mock_load_model.return_value = (mock_model, mock_preprocess, mock_tokenizer)

    service = SigLIPEmbeddingService()

    # Access embedding_dim to trigger lazy loading
    _ = service.embedding_dim

    # Verify model was loaded
    assert mock_load_model.called


def test_siglip_service_singleton() -> None:
    """Test that get_siglip_service returns singleton instance."""
    from image_search_service.services.siglip_embedding import get_siglip_service

    # Get service twice
    service1 = get_siglip_service()
    service2 = get_siglip_service()

    # Should be same instance
    assert service1 is service2
