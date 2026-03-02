"""Test the real EmbeddingService class with mocked ML model layer.

Tests the production EmbeddingService from embedding.py by mocking only the
underlying OpenCLIP model, tokenizer, and preprocessing functions. This verifies
that the service correctly orchestrates tokenization, model inference,
normalization, and output conversion.
"""

import sys
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from tests.constants import CLIP_EMBEDDING_DIM

# Tests that exercise the real EmbeddingService inference path require CUDA.
# On macOS (no NVIDIA GPU), these are skipped.
requires_cuda = pytest.mark.skipif(
    sys.platform == "darwin",
    reason="Requires CUDA — Linux with NVIDIA GPU only",
)

# Save original (unpatched) method references at import time, BEFORE conftest's
# autouse clear_embedding_cache fixture replaces them with mock delegates.
from image_search_service.services.embedding import EmbeddingService as _EmbeddingServiceCls  # noqa: E402, I001

_ORIGINAL_EMBED_TEXT = _EmbeddingServiceCls.embed_text
_ORIGINAL_EMBED_IMAGE = _EmbeddingServiceCls.embed_image
_ORIGINAL_EMBED_IMAGES_BATCH = _EmbeddingServiceCls.embed_images_batch
_ORIGINAL_EMBED_IMAGE_FROM_PIL = _EmbeddingServiceCls.embed_image_from_pil
_ORIGINAL_EMBEDDING_DIM = _EmbeddingServiceCls.__dict__["embedding_dim"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_model(embedding_dim: int = CLIP_EMBEDDING_DIM) -> MagicMock:
    """Create a mock OpenCLIP model that returns tensors of the given dimension.

    The model provides:
    - encode_text(tokens) -> tensor (1, embedding_dim)
    - encode_image(tensor) -> tensor (N, embedding_dim)
    - visual.output_dim -> embedding_dim
    - eval() -> self

    The returned tensors are random but deterministic (seeded) so tests
    can verify normalization and shape without loading a real model.
    """
    rng = np.random.RandomState(42)

    model = MagicMock()
    model.visual.output_dim = embedding_dim

    def _encode_text(tokens: torch.Tensor) -> torch.Tensor:
        raw = torch.tensor(rng.randn(1, embedding_dim), dtype=torch.float32)
        return raw

    def _encode_image(tensor: torch.Tensor) -> torch.Tensor:
        batch_size = tensor.shape[0]
        raw = torch.tensor(rng.randn(batch_size, embedding_dim), dtype=torch.float32)
        return raw

    model.encode_text = MagicMock(side_effect=_encode_text)
    model.encode_image = MagicMock(side_effect=_encode_image)
    model.eval.return_value = model
    model.to.return_value = model

    return model


def _make_mock_preprocess() -> MagicMock:
    """Create a mock preprocess function that returns a plausible image tensor."""

    def _preprocess(image: Image.Image) -> torch.Tensor:
        return torch.randn(3, 224, 224)

    mock = MagicMock(side_effect=_preprocess)
    return mock


def _make_mock_tokenizer() -> MagicMock:
    """Create a mock tokenizer that returns a token tensor."""

    def _tokenize(texts: list[str]) -> torch.Tensor:
        return torch.randint(0, 1000, (len(texts), 77))

    mock = MagicMock(side_effect=_tokenize)
    return mock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_embedding_globals():
    """Reset the module-level globals in embedding.py between tests.

    The production code caches the model in module globals (_model, _preprocess,
    _tokenizer). We reset them before each test so our mocked _load_model is
    always invoked.
    """
    import image_search_service.services.embedding as emb_mod

    original_model = emb_mod._model
    original_preprocess = emb_mod._preprocess
    original_tokenizer = emb_mod._tokenizer

    emb_mod._model = None
    emb_mod._preprocess = None
    emb_mod._tokenizer = None

    yield

    emb_mod._model = original_model
    emb_mod._preprocess = original_preprocess
    emb_mod._tokenizer = original_tokenizer


@pytest.fixture()
def mock_clip_stack(monkeypatch: pytest.MonkeyPatch):
    """Provide a full mocked CLIP stack (model, preprocess, tokenizer).

    Patches _load_model in embedding.py so the real EmbeddingService uses our
    mocks instead of loading a real OpenCLIP model.

    Also restores the original EmbeddingService methods that the autouse
    clear_embedding_cache fixture in conftest.py replaces with
    SemanticMockEmbeddingService delegates. The originals were captured at
    module import time (before conftest runs).
    """
    from image_search_service.services.embedding import EmbeddingService

    # Restore original (unpatched) methods on the class.
    # The conftest autouse fixture replaces these with SemanticMock delegates.
    monkeypatch.setattr(EmbeddingService, "embed_text", _ORIGINAL_EMBED_TEXT)
    monkeypatch.setattr(EmbeddingService, "embed_image", _ORIGINAL_EMBED_IMAGE)
    monkeypatch.setattr(EmbeddingService, "embed_images_batch", _ORIGINAL_EMBED_IMAGES_BATCH)
    monkeypatch.setattr(EmbeddingService, "embed_image_from_pil", _ORIGINAL_EMBED_IMAGE_FROM_PIL)
    monkeypatch.setattr(EmbeddingService, "embedding_dim", _ORIGINAL_EMBEDDING_DIM)

    mock_model = _make_mock_model(embedding_dim=CLIP_EMBEDDING_DIM)
    mock_preprocess = _make_mock_preprocess()
    mock_tokenizer = _make_mock_tokenizer()

    with patch(
        "image_search_service.services.embedding._load_model",
        return_value=(mock_model, mock_preprocess, mock_tokenizer),
    ) as mock_load:
        yield {
            "model": mock_model,
            "preprocess": mock_preprocess,
            "tokenizer": mock_tokenizer,
            "load_model": mock_load,
        }


# ---------------------------------------------------------------------------
# Tests: embed_text
# ---------------------------------------------------------------------------


@requires_cuda
def test_embed_text_returns_list_of_floats(mock_clip_stack: dict) -> None:
    """embed_text should return a list[float] when given valid text."""
    from image_search_service.services.embedding import EmbeddingService

    service = EmbeddingService()
    result = service.embed_text("a beautiful sunset")

    assert isinstance(result, list)
    assert all(isinstance(v, float) for v in result)


def test_embed_text_returns_correct_dimension(mock_clip_stack: dict) -> None:
    """embed_text should return a vector with the model's embedding dimension."""
    from image_search_service.services.embedding import EmbeddingService

    service = EmbeddingService()
    result = service.embed_text("test query")

    assert len(result) == CLIP_EMBEDDING_DIM


@requires_cuda
def test_embed_text_output_is_normalized(mock_clip_stack: dict) -> None:
    """embed_text should return an L2-normalized vector (norm approx 1.0)."""
    from image_search_service.services.embedding import EmbeddingService

    service = EmbeddingService()
    result = service.embed_text("normalization check")

    norm = float(np.linalg.norm(result))
    assert abs(norm - 1.0) < 0.01, f"Expected norm ~1.0, got {norm}"


def test_embed_text_calls_tokenizer_with_input(mock_clip_stack: dict) -> None:
    """embed_text should pass the input text through the tokenizer."""
    from image_search_service.services.embedding import EmbeddingService

    service = EmbeddingService()
    service.embed_text("hello world")

    mock_clip_stack["tokenizer"].assert_called_once()
    call_args = mock_clip_stack["tokenizer"].call_args[0][0]
    assert call_args == ["hello world"]


@requires_cuda
def test_embed_text_calls_model_encode_text(mock_clip_stack: dict) -> None:
    """embed_text should call model.encode_text with tokenized input."""
    from image_search_service.services.embedding import EmbeddingService

    service = EmbeddingService()
    service.embed_text("test")

    mock_clip_stack["model"].encode_text.assert_called_once()


@requires_cuda
def test_embed_text_triggers_model_load(mock_clip_stack: dict) -> None:
    """embed_text should trigger _load_model to obtain the model."""
    from image_search_service.services.embedding import EmbeddingService

    service = EmbeddingService()
    service.embed_text("trigger load")

    mock_clip_stack["load_model"].assert_called()


# ---------------------------------------------------------------------------
# Tests: embed_image
# ---------------------------------------------------------------------------


@requires_cuda
def test_embed_image_returns_list_of_floats(
    mock_clip_stack: dict, temp_image_factory: Callable[..., Path]
) -> None:
    """embed_image should return a list[float] when given a valid image path."""
    from image_search_service.services.embedding import EmbeddingService

    service = EmbeddingService()
    image_path = temp_image_factory("photo.jpg", width=100, height=100)
    result = service.embed_image(image_path)

    assert isinstance(result, list)
    assert all(isinstance(v, float) for v in result)


def test_embed_image_returns_correct_dimension(
    mock_clip_stack: dict, temp_image_factory: Callable[..., Path]
) -> None:
    """embed_image should return a vector with the model's embedding dimension."""
    from image_search_service.services.embedding import EmbeddingService

    service = EmbeddingService()
    image_path = temp_image_factory("photo.jpg", width=64, height=64)
    result = service.embed_image(image_path)

    assert len(result) == CLIP_EMBEDDING_DIM


@requires_cuda
def test_embed_image_output_is_normalized(
    mock_clip_stack: dict, temp_image_factory: Callable[..., Path]
) -> None:
    """embed_image should return an L2-normalized vector (norm approx 1.0)."""
    from image_search_service.services.embedding import EmbeddingService

    service = EmbeddingService()
    image_path = temp_image_factory("photo.jpg", width=50, height=50)
    result = service.embed_image(image_path)

    norm = float(np.linalg.norm(result))
    assert abs(norm - 1.0) < 0.01, f"Expected norm ~1.0, got {norm}"


def test_embed_image_applies_preprocessing(
    mock_clip_stack: dict, temp_image_factory: Callable[..., Path]
) -> None:
    """embed_image should apply the preprocess transform to the opened image."""
    from image_search_service.services.embedding import EmbeddingService

    service = EmbeddingService()
    image_path = temp_image_factory("photo.jpg", width=100, height=100)
    service.embed_image(image_path)

    mock_clip_stack["preprocess"].assert_called_once()
    # The argument should be a PIL Image
    call_arg = mock_clip_stack["preprocess"].call_args[0][0]
    assert isinstance(call_arg, Image.Image)


@requires_cuda
def test_embed_image_calls_model_encode_image(
    mock_clip_stack: dict, temp_image_factory: Callable[..., Path]
) -> None:
    """embed_image should call model.encode_image with the preprocessed tensor."""
    from image_search_service.services.embedding import EmbeddingService

    service = EmbeddingService()
    image_path = temp_image_factory("photo.jpg", width=100, height=100)
    service.embed_image(image_path)

    mock_clip_stack["model"].encode_image.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: embed_image_from_pil
# ---------------------------------------------------------------------------


@requires_cuda
def test_embed_image_from_pil_returns_correct_dimension(
    mock_clip_stack: dict,
) -> None:
    """embed_image_from_pil should return a correctly dimensioned vector."""
    from image_search_service.services.embedding import EmbeddingService

    service = EmbeddingService()
    pil_image = Image.new("RGB", (100, 100))
    result = service.embed_image_from_pil(pil_image)

    assert len(result) == CLIP_EMBEDDING_DIM
    assert isinstance(result, list)
    assert all(isinstance(v, float) for v in result)


@requires_cuda
def test_embed_image_from_pil_output_is_normalized(
    mock_clip_stack: dict,
) -> None:
    """embed_image_from_pil should return an L2-normalized vector."""
    from image_search_service.services.embedding import EmbeddingService

    service = EmbeddingService()
    pil_image = Image.new("RGB", (100, 100))
    result = service.embed_image_from_pil(pil_image)

    norm = float(np.linalg.norm(result))
    assert abs(norm - 1.0) < 0.01


# ---------------------------------------------------------------------------
# Tests: embed_images_batch
# ---------------------------------------------------------------------------


@requires_cuda
def test_embed_images_batch_returns_correct_count(
    mock_clip_stack: dict,
) -> None:
    """embed_images_batch should return one vector per input image."""
    from image_search_service.services.embedding import EmbeddingService

    service = EmbeddingService()
    images = [
        Image.new("RGB", (100, 100)),
        Image.new("RGB", (200, 200)),
        Image.new("RGB", (150, 150)),
    ]
    results = service.embed_images_batch(images)

    assert len(results) == 3
    assert all(len(v) == CLIP_EMBEDDING_DIM for v in results)


def test_embed_images_batch_empty_list_returns_empty(
    mock_clip_stack: dict,
) -> None:
    """embed_images_batch should return empty list for empty input."""
    from image_search_service.services.embedding import EmbeddingService

    service = EmbeddingService()
    results = service.embed_images_batch([])

    assert results == []


@requires_cuda
def test_embed_images_batch_vectors_are_normalized(
    mock_clip_stack: dict,
) -> None:
    """embed_images_batch should return L2-normalized vectors."""
    from image_search_service.services.embedding import EmbeddingService

    service = EmbeddingService()
    images = [Image.new("RGB", (100, 100)), Image.new("RGB", (50, 50))]
    results = service.embed_images_batch(images)

    for vec in results:
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 0.01


# ---------------------------------------------------------------------------
# Tests: embedding_dim property
# ---------------------------------------------------------------------------


def test_embedding_dim_returns_model_output_dim(mock_clip_stack: dict) -> None:
    """embedding_dim should return the model's visual.output_dim."""
    from image_search_service.services.embedding import EmbeddingService

    service = EmbeddingService()
    assert service.embedding_dim == CLIP_EMBEDDING_DIM


def test_embedding_dim_triggers_model_load(mock_clip_stack: dict) -> None:
    """Accessing embedding_dim should trigger lazy model loading."""
    from image_search_service.services.embedding import EmbeddingService

    service = EmbeddingService()
    _ = service.embedding_dim

    mock_clip_stack["load_model"].assert_called()


# ---------------------------------------------------------------------------
# Tests: lazy initialization / singleton
# ---------------------------------------------------------------------------


def test_lazy_initialization_when_model_not_accessed_then_no_load() -> None:
    """EmbeddingService construction should NOT load the model."""
    from image_search_service.services.embedding import EmbeddingService

    with patch("image_search_service.services.embedding._load_model") as mock_load:
        _ = EmbeddingService()
        mock_load.assert_not_called()


def test_get_embedding_service_returns_singleton() -> None:
    """get_embedding_service should return the same instance on repeated calls."""
    from image_search_service.services.embedding import get_embedding_service

    # Clear cache to start fresh
    get_embedding_service.cache_clear()

    service1 = get_embedding_service()
    service2 = get_embedding_service()

    assert service1 is service2

    # Cleanup
    get_embedding_service.cache_clear()


# ---------------------------------------------------------------------------
# Tests: error cases
# ---------------------------------------------------------------------------


def test_embed_image_when_invalid_path_then_raises(
    mock_clip_stack: dict,
) -> None:
    """embed_image should raise when given a nonexistent image path."""
    from image_search_service.services.embedding import EmbeddingService

    service = EmbeddingService()

    with pytest.raises((FileNotFoundError, OSError)):
        service.embed_image("/nonexistent/path/to/image.jpg")


def test_embed_image_when_corrupt_file_then_raises(mock_clip_stack: dict, tmp_path: Path) -> None:
    """embed_image should raise when given a file that is not a valid image."""
    from image_search_service.services.embedding import EmbeddingService

    corrupt_file = tmp_path / "corrupt.jpg"
    corrupt_file.write_text("this is not an image")

    service = EmbeddingService()

    with pytest.raises(Exception):
        service.embed_image(corrupt_file)


def test_model_loading_failure_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When _load_model raises, the error should propagate to callers."""
    from image_search_service.services.embedding import EmbeddingService

    # Restore real embed_text (conftest replaces it with SemanticMock)
    monkeypatch.setattr(EmbeddingService, "embed_text", _ORIGINAL_EMBED_TEXT)

    with patch(
        "image_search_service.services.embedding._load_model",
        side_effect=RuntimeError("Model file not found"),
    ):
        service = EmbeddingService()

        with pytest.raises(RuntimeError, match="Model file not found"):
            service.embed_text("test")


# ---------------------------------------------------------------------------
# Tests: device property
# ---------------------------------------------------------------------------


def test_device_property_returns_string(mock_clip_stack: dict) -> None:
    """device property should return a string device identifier."""
    from image_search_service.services.embedding import EmbeddingService

    service = EmbeddingService()
    device = service.device

    assert isinstance(device, str)
    assert device in ("cpu", "cuda", "mps") or device.startswith("cuda:")


def test_device_property_is_cached(mock_clip_stack: dict) -> None:
    """device property should cache the result after first access."""
    from image_search_service.services.embedding import EmbeddingService

    service = EmbeddingService()
    device1 = service.device
    device2 = service.device

    assert device1 == device2
    # After first access, _device should be set
    assert service._device is not None
