"""Test SigLIP embedding service: real class with mocked ML models + mock infrastructure.

Section 1: Tests for the REAL SigLIPEmbeddingService class from siglip_embedding.py,
with only the underlying OpenCLIP model layer mocked.

Section 2: Tests for MockSigLIPEmbeddingService (test infrastructure validation).
"""

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from tests.constants import SIGLIP_EMBEDDING_DIM

# ===========================================================================
# Helpers for real SigLIPEmbeddingService tests
# ===========================================================================


def _make_siglip_mock_model(embedding_dim: int = SIGLIP_EMBEDDING_DIM) -> MagicMock:
    """Create a mock SigLIP model that returns tensors of the given dimension."""
    rng = np.random.RandomState(99)

    model = MagicMock()
    model.visual.output_dim = embedding_dim

    def _encode_text(tokens: torch.Tensor) -> torch.Tensor:
        return torch.tensor(rng.randn(1, embedding_dim), dtype=torch.float32)

    def _encode_image(tensor: torch.Tensor) -> torch.Tensor:
        batch_size = tensor.shape[0]
        return torch.tensor(rng.randn(batch_size, embedding_dim), dtype=torch.float32)

    model.encode_text = MagicMock(side_effect=_encode_text)
    model.encode_image = MagicMock(side_effect=_encode_image)
    model.eval.return_value = model
    model.to.return_value = model

    return model


def _make_siglip_mock_preprocess() -> MagicMock:
    """Create a mock preprocess function that returns a plausible image tensor."""

    def _preprocess(image: Image.Image) -> torch.Tensor:
        return torch.randn(3, 224, 224)

    return MagicMock(side_effect=_preprocess)


def _make_siglip_mock_tokenizer() -> MagicMock:
    """Create a mock tokenizer that returns a token tensor."""

    def _tokenize(texts: list[str]) -> torch.Tensor:
        return torch.randint(0, 1000, (len(texts), 77))

    return MagicMock(side_effect=_tokenize)


# ===========================================================================
# Fixtures for real SigLIPEmbeddingService tests
# ===========================================================================


@pytest.fixture(autouse=True)
def _reset_siglip_globals():
    """Reset the module-level globals in siglip_embedding.py between tests."""
    import image_search_service.services.siglip_embedding as siglip_mod

    original_model = siglip_mod._model
    original_preprocess = siglip_mod._preprocess
    original_tokenizer = siglip_mod._tokenizer

    siglip_mod._model = None
    siglip_mod._preprocess = None
    siglip_mod._tokenizer = None

    yield

    siglip_mod._model = original_model
    siglip_mod._preprocess = original_preprocess
    siglip_mod._tokenizer = original_tokenizer


@pytest.fixture()
def mock_siglip_stack():
    """Provide a full mocked SigLIP stack (model, preprocess, tokenizer).

    Patches _load_model in siglip_embedding.py so the real SigLIPEmbeddingService
    uses our mocks instead of loading a real SigLIP model.
    """
    mock_model = _make_siglip_mock_model(embedding_dim=SIGLIP_EMBEDDING_DIM)
    mock_preprocess = _make_siglip_mock_preprocess()
    mock_tokenizer = _make_siglip_mock_tokenizer()

    with patch(
        "image_search_service.services.siglip_embedding._load_model",
        return_value=(mock_model, mock_preprocess, mock_tokenizer),
    ) as mock_load:
        yield {
            "model": mock_model,
            "preprocess": mock_preprocess,
            "tokenizer": mock_tokenizer,
            "load_model": mock_load,
        }


# ===========================================================================
# Section 1: Real SigLIPEmbeddingService tests (mocked ML layer only)
# ===========================================================================


class TestSigLIPEmbedText:
    """Tests for SigLIPEmbeddingService.embed_text through the real class."""

    def test_returns_list_of_floats(self, mock_siglip_stack: dict) -> None:
        """embed_text should return a list[float] when given valid text."""
        from image_search_service.services.siglip_embedding import SigLIPEmbeddingService

        service = SigLIPEmbeddingService()
        result = service.embed_text("a beautiful sunset")

        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_returns_correct_dimension(self, mock_siglip_stack: dict) -> None:
        """embed_text should return a vector with 768 dimensions."""
        from image_search_service.services.siglip_embedding import SigLIPEmbeddingService

        service = SigLIPEmbeddingService()
        result = service.embed_text("test query")

        assert len(result) == SIGLIP_EMBEDDING_DIM

    def test_output_is_normalized(self, mock_siglip_stack: dict) -> None:
        """embed_text should return an L2-normalized vector (norm approx 1.0)."""
        from image_search_service.services.siglip_embedding import SigLIPEmbeddingService

        service = SigLIPEmbeddingService()
        result = service.embed_text("normalization check")

        norm = float(np.linalg.norm(result))
        assert abs(norm - 1.0) < 0.01, f"Expected norm ~1.0, got {norm}"

    def test_calls_tokenizer_with_input(self, mock_siglip_stack: dict) -> None:
        """embed_text should pass the input text through the tokenizer."""
        from image_search_service.services.siglip_embedding import SigLIPEmbeddingService

        service = SigLIPEmbeddingService()
        service.embed_text("hello world")

        mock_siglip_stack["tokenizer"].assert_called_once()
        call_args = mock_siglip_stack["tokenizer"].call_args[0][0]
        assert call_args == ["hello world"]

    def test_calls_model_encode_text(self, mock_siglip_stack: dict) -> None:
        """embed_text should call model.encode_text with tokenized input."""
        from image_search_service.services.siglip_embedding import SigLIPEmbeddingService

        service = SigLIPEmbeddingService()
        service.embed_text("test")

        mock_siglip_stack["model"].encode_text.assert_called_once()

    def test_triggers_model_load(self, mock_siglip_stack: dict) -> None:
        """embed_text should trigger _load_model to obtain the model."""
        from image_search_service.services.siglip_embedding import SigLIPEmbeddingService

        service = SigLIPEmbeddingService()
        service.embed_text("trigger load")

        mock_siglip_stack["load_model"].assert_called()


class TestSigLIPEmbedImage:
    """Tests for SigLIPEmbeddingService.embed_image through the real class."""

    def test_returns_list_of_floats(
        self, mock_siglip_stack: dict, temp_image_factory: Callable[..., Path]
    ) -> None:
        """embed_image should return a list[float] when given a valid image path."""
        from image_search_service.services.siglip_embedding import SigLIPEmbeddingService

        service = SigLIPEmbeddingService()
        image_path = temp_image_factory("photo.jpg", width=100, height=100)
        result = service.embed_image(image_path)

        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_returns_correct_dimension(
        self, mock_siglip_stack: dict, temp_image_factory: Callable[..., Path]
    ) -> None:
        """embed_image should return a 768-dimensional vector."""
        from image_search_service.services.siglip_embedding import SigLIPEmbeddingService

        service = SigLIPEmbeddingService()
        image_path = temp_image_factory("photo.jpg", width=64, height=64)
        result = service.embed_image(image_path)

        assert len(result) == SIGLIP_EMBEDDING_DIM

    def test_output_is_normalized(
        self, mock_siglip_stack: dict, temp_image_factory: Callable[..., Path]
    ) -> None:
        """embed_image should return an L2-normalized vector."""
        from image_search_service.services.siglip_embedding import SigLIPEmbeddingService

        service = SigLIPEmbeddingService()
        image_path = temp_image_factory("photo.jpg", width=50, height=50)
        result = service.embed_image(image_path)

        norm = float(np.linalg.norm(result))
        assert abs(norm - 1.0) < 0.01

    def test_applies_preprocessing(
        self, mock_siglip_stack: dict, temp_image_factory: Callable[..., Path]
    ) -> None:
        """embed_image should apply the preprocess transform to the opened image."""
        from image_search_service.services.siglip_embedding import SigLIPEmbeddingService

        service = SigLIPEmbeddingService()
        image_path = temp_image_factory("photo.jpg", width=100, height=100)
        service.embed_image(image_path)

        mock_siglip_stack["preprocess"].assert_called_once()
        call_arg = mock_siglip_stack["preprocess"].call_args[0][0]
        assert isinstance(call_arg, Image.Image)

    def test_calls_model_encode_image(
        self, mock_siglip_stack: dict, temp_image_factory: Callable[..., Path]
    ) -> None:
        """embed_image should call model.encode_image with the preprocessed tensor."""
        from image_search_service.services.siglip_embedding import SigLIPEmbeddingService

        service = SigLIPEmbeddingService()
        image_path = temp_image_factory("photo.jpg", width=100, height=100)
        service.embed_image(image_path)

        mock_siglip_stack["model"].encode_image.assert_called_once()


class TestSigLIPEmbedImageFromPil:
    """Tests for SigLIPEmbeddingService.embed_image_from_pil."""

    def test_returns_correct_dimension(self, mock_siglip_stack: dict) -> None:
        """embed_image_from_pil should return a 768-dimensional vector."""
        from image_search_service.services.siglip_embedding import SigLIPEmbeddingService

        service = SigLIPEmbeddingService()
        pil_image = Image.new("RGB", (100, 100))
        result = service.embed_image_from_pil(pil_image)

        assert len(result) == SIGLIP_EMBEDDING_DIM
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_output_is_normalized(self, mock_siglip_stack: dict) -> None:
        """embed_image_from_pil should return an L2-normalized vector."""
        from image_search_service.services.siglip_embedding import SigLIPEmbeddingService

        service = SigLIPEmbeddingService()
        pil_image = Image.new("RGB", (100, 100))
        result = service.embed_image_from_pil(pil_image)

        norm = float(np.linalg.norm(result))
        assert abs(norm - 1.0) < 0.01


class TestSigLIPEmbedImagesBatch:
    """Tests for SigLIPEmbeddingService.embed_images_batch."""

    def test_returns_correct_count(self, mock_siglip_stack: dict) -> None:
        """embed_images_batch should return one vector per input image."""
        from image_search_service.services.siglip_embedding import SigLIPEmbeddingService

        service = SigLIPEmbeddingService()
        images = [
            Image.new("RGB", (100, 100)),
            Image.new("RGB", (200, 200)),
            Image.new("RGB", (150, 150)),
        ]
        results = service.embed_images_batch(images)

        assert len(results) == 3
        assert all(len(v) == SIGLIP_EMBEDDING_DIM for v in results)

    def test_empty_list_returns_empty(self, mock_siglip_stack: dict) -> None:
        """embed_images_batch should return empty list for empty input."""
        from image_search_service.services.siglip_embedding import SigLIPEmbeddingService

        service = SigLIPEmbeddingService()
        results = service.embed_images_batch([])

        assert results == []

    def test_vectors_are_normalized(self, mock_siglip_stack: dict) -> None:
        """embed_images_batch should return L2-normalized vectors."""
        from image_search_service.services.siglip_embedding import SigLIPEmbeddingService

        service = SigLIPEmbeddingService()
        images = [Image.new("RGB", (100, 100)), Image.new("RGB", (50, 50))]
        results = service.embed_images_batch(images)

        for vec in results:
            norm = float(np.linalg.norm(vec))
            assert abs(norm - 1.0) < 0.01

    def test_respects_batch_size_parameter(self, mock_siglip_stack: dict) -> None:
        """embed_images_batch should process images in batches of given size."""
        from image_search_service.services.siglip_embedding import SigLIPEmbeddingService

        service = SigLIPEmbeddingService()
        # Create 5 images, batch_size=2 should result in 3 encode_image calls
        images = [Image.new("RGB", (100, 100)) for _ in range(5)]
        results = service.embed_images_batch(images, batch_size=2)

        assert len(results) == 5
        # With batch_size=2 and 5 images: batches of [2, 2, 1] = 3 calls
        assert mock_siglip_stack["model"].encode_image.call_count == 3


class TestSigLIPEmbeddingDim:
    """Tests for SigLIPEmbeddingService.embedding_dim property."""

    def test_returns_model_output_dim(self, mock_siglip_stack: dict) -> None:
        """embedding_dim should return the model's visual.output_dim."""
        from image_search_service.services.siglip_embedding import SigLIPEmbeddingService

        service = SigLIPEmbeddingService()
        assert service.embedding_dim == SIGLIP_EMBEDDING_DIM

    def test_triggers_model_load(self, mock_siglip_stack: dict) -> None:
        """Accessing embedding_dim should trigger lazy model loading."""
        from image_search_service.services.siglip_embedding import SigLIPEmbeddingService

        service = SigLIPEmbeddingService()
        _ = service.embedding_dim

        mock_siglip_stack["load_model"].assert_called()


class TestSigLIPLazyInit:
    """Tests for lazy initialization and singleton patterns."""

    def test_construction_does_not_load_model(self) -> None:
        """SigLIPEmbeddingService construction should NOT load the model."""
        with patch("image_search_service.services.siglip_embedding._load_model") as mock_load:
            from image_search_service.services.siglip_embedding import SigLIPEmbeddingService

            _ = SigLIPEmbeddingService()
            mock_load.assert_not_called()

    @patch("image_search_service.services.siglip_embedding._load_model")
    def test_lazy_loading(self, mock_load_model: MagicMock) -> None:
        """SigLIP model should be lazy loaded on first use."""
        from image_search_service.services.siglip_embedding import SigLIPEmbeddingService

        mock_model = MagicMock()
        mock_model.visual.output_dim = SIGLIP_EMBEDDING_DIM
        mock_load_model.return_value = (mock_model, MagicMock(), MagicMock())

        service = SigLIPEmbeddingService()

        # Model should not be loaded yet
        assert mock_load_model.call_count == 0

        # Access embedding_dim property (triggers load)
        _ = service.embedding_dim

        # Now model should be loaded
        assert mock_load_model.call_count == 1

    @patch("image_search_service.services.siglip_embedding._load_model")
    @patch("image_search_service.services.siglip_embedding.get_settings")
    def test_uses_correct_config(
        self, mock_get_settings: MagicMock, mock_load_model: MagicMock
    ) -> None:
        """SigLIP service should use correct configuration values."""
        from image_search_service.services.siglip_embedding import SigLIPEmbeddingService

        mock_settings = MagicMock()
        mock_settings.siglip_model_name = "ViT-B-16-SigLIP"
        mock_settings.siglip_pretrained = "webli"
        mock_settings.siglip_embedding_dim = SIGLIP_EMBEDDING_DIM
        mock_get_settings.return_value = mock_settings

        mock_model = MagicMock()
        mock_model.visual.output_dim = SIGLIP_EMBEDDING_DIM
        mock_load_model.return_value = (mock_model, MagicMock(), MagicMock())

        service = SigLIPEmbeddingService()
        _ = service.embedding_dim

        assert mock_load_model.called

    def test_singleton(self) -> None:
        """get_siglip_service should return the same instance on repeated calls."""
        from image_search_service.services.siglip_embedding import get_siglip_service

        get_siglip_service.cache_clear()

        service1 = get_siglip_service()
        service2 = get_siglip_service()

        assert service1 is service2

        get_siglip_service.cache_clear()


class TestSigLIPErrorHandling:
    """Tests for SigLIPEmbeddingService error handling paths."""

    def test_embed_image_when_invalid_path_then_raises(self, mock_siglip_stack: dict) -> None:
        """embed_image should raise when given a nonexistent image path."""
        from image_search_service.services.siglip_embedding import SigLIPEmbeddingService

        service = SigLIPEmbeddingService()

        with pytest.raises((FileNotFoundError, OSError)):
            service.embed_image("/nonexistent/path/to/image.jpg")

    def test_embed_image_when_corrupt_file_then_raises(
        self, mock_siglip_stack: dict, tmp_path: Path
    ) -> None:
        """embed_image should raise when given a file that is not a valid image."""
        from image_search_service.services.siglip_embedding import SigLIPEmbeddingService

        corrupt_file = tmp_path / "corrupt.jpg"
        corrupt_file.write_text("this is not an image")

        service = SigLIPEmbeddingService()

        with pytest.raises(Exception):
            service.embed_image(corrupt_file)

    def test_model_loading_failure_propagates(self) -> None:
        """When _load_model raises, the error should propagate to callers."""
        from image_search_service.services.siglip_embedding import SigLIPEmbeddingService

        with patch(
            "image_search_service.services.siglip_embedding._load_model",
            side_effect=RuntimeError("SigLIP model file not found"),
        ):
            service = SigLIPEmbeddingService()

            with pytest.raises(RuntimeError, match="SigLIP model file not found"):
                service.embed_text("test")


class TestSigLIPDeviceProperty:
    """Tests for SigLIPEmbeddingService.device property."""

    def test_returns_string(self, mock_siglip_stack: dict) -> None:
        """device property should return a string device identifier."""
        from image_search_service.services.siglip_embedding import SigLIPEmbeddingService

        service = SigLIPEmbeddingService()
        device = service.device

        assert isinstance(device, str)
        assert device in ("cpu", "cuda", "mps") or device.startswith("cuda:")

    def test_is_cached(self, mock_siglip_stack: dict) -> None:
        """device property should cache the result after first access."""
        from image_search_service.services.siglip_embedding import SigLIPEmbeddingService

        service = SigLIPEmbeddingService()
        device1 = service.device
        device2 = service.device

        assert device1 == device2
        assert service._device is not None


# ===========================================================================
# Section 2: MockSigLIPEmbeddingService (test infrastructure validation)
# ===========================================================================


class MockSigLIPEmbeddingService:
    """Mock SigLIP embedding service that returns deterministic SIGLIP_EMBEDDING_DIM-dim vectors.

    Test infrastructure only -- NOT production code.
    """

    @property
    def embedding_dim(self) -> int:
        """Return SigLIP embedding dimension."""
        return SIGLIP_EMBEDDING_DIM

    def embed_text(self, text: str) -> list[float]:
        """Generate deterministic SIGLIP_EMBEDDING_DIM-dim vector from text."""
        import hashlib

        h = hashlib.sha256(text.encode()).hexdigest()

        vector = []
        for i in range(SIGLIP_EMBEDDING_DIM):
            idx = (i * 2) % len(h)
            val = int(h[idx : idx + 2], 16) / 255.0
            vector.append(val)

        return vector

    def embed_image(self, image_path: str | Path) -> list[float]:
        """Generate deterministic SIGLIP_EMBEDDING_DIM-dim vector from image path."""
        return self.embed_text(str(image_path))

    def embed_image_from_pil(self, image: "Image.Image") -> list[float]:  # type: ignore[name-defined]
        """Generate deterministic vector from PIL image."""
        return self.embed_text(f"{image.width}x{image.height}")

    def embed_images_batch(
        self,
        images: list["Image.Image"],
        batch_size: int = 32,  # type: ignore[name-defined]
    ) -> list[list[float]]:
        """Generate deterministic vectors for batch of images."""
        return [self.embed_image_from_pil(img) for img in images]


class TestMockSigLIPInfrastructure:
    """Test infrastructure validation for MockSigLIPEmbeddingService.

    These tests validate that the mock itself works correctly.
    They do NOT test production code.
    """

    def test_embed_text_returns_768_dim_vector(self) -> None:
        """Mock SigLIP service should return 768-dim vector for text."""
        service = MockSigLIPEmbeddingService()
        vector = service.embed_text("a beautiful sunset")

        assert len(vector) == SIGLIP_EMBEDDING_DIM
        assert all(isinstance(v, float) for v in vector)
        assert all(0.0 <= v <= 1.0 for v in vector)

    def test_embed_image_returns_768_dim_vector(
        self, temp_image_factory: Callable[..., Path]
    ) -> None:
        """Mock SigLIP service should return 768-dim vector for image."""
        service = MockSigLIPEmbeddingService()
        image_path = temp_image_factory("test.jpg", width=100, height=100)
        vector = service.embed_image(image_path)

        assert len(vector) == SIGLIP_EMBEDDING_DIM
        assert all(isinstance(v, float) for v in vector)
        assert all(0.0 <= v <= 1.0 for v in vector)

    def test_embedding_dim_correct(self) -> None:
        """embedding_dim property should return 768 for SigLIP."""
        service = MockSigLIPEmbeddingService()
        assert service.embedding_dim == SIGLIP_EMBEDDING_DIM

    def test_deterministic(self) -> None:
        """Same input should produce same output (deterministic)."""
        service = MockSigLIPEmbeddingService()
        vector1 = service.embed_text("test query")
        vector2 = service.embed_text("test query")

        assert vector1 == vector2

    def test_different_inputs_produce_different_vectors(self) -> None:
        """Different inputs should produce different vectors."""
        service = MockSigLIPEmbeddingService()
        vector1 = service.embed_text("sunset")
        vector2 = service.embed_text("mountain")

        assert vector1 != vector2

    def test_batch_embedding_returns_correct_count(self) -> None:
        """Batch embedding should return correct number of vectors."""
        service = MockSigLIPEmbeddingService()
        images = [
            Image.new("RGB", (100, 100)),
            Image.new("RGB", (200, 200)),
            Image.new("RGB", (150, 150)),
        ]
        vectors = service.embed_images_batch(images)

        assert len(vectors) == 3
        assert all(len(v) == SIGLIP_EMBEDDING_DIM for v in vectors)
