"""Embedding service using OpenCLIP for image and text embeddings."""

from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image

from image_search_service.core.config import get_settings
from image_search_service.core.device import get_device, get_device_info
from image_search_service.core.logging import get_logger

logger = get_logger(__name__)

# Lazy load heavy imports
_model = None
_preprocess = None
_tokenizer = None


def _load_model() -> tuple[Any, Any, Any]:
    """Lazy load OpenCLIP model.

    Returns:
        Tuple of (model, preprocess, tokenizer)
    """
    global _model, _preprocess, _tokenizer

    if _model is not None:
        return _model, _preprocess, _tokenizer

    import open_clip

    settings = get_settings()
    device = get_device()
    device_info = get_device_info()

    logger.info(
        f"Loading OpenCLIP model {settings.clip_model_name} on {device} ({device_info['type']})"
    )

    model, _, preprocess = open_clip.create_model_and_transforms(
        settings.clip_model_name, pretrained=settings.clip_pretrained
    )
    model = model.to(device)
    model.eval()

    tokenizer = open_clip.get_tokenizer(settings.clip_model_name)

    _model = model
    _preprocess = preprocess
    _tokenizer = tokenizer

    logger.info(f"Model loaded. Embedding dim: {model.visual.output_dim}")
    return model, preprocess, tokenizer


class EmbeddingService:
    """Service for generating CLIP embeddings."""

    def __init__(self) -> None:
        """Initialize embedding service."""
        self._device: str | None = None

    @property
    def device(self) -> str:
        """Get device for model inference."""
        if self._device is None:
            self._device = get_device()
        return self._device

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension from loaded model."""
        model, _, _ = _load_model()
        return int(model.visual.output_dim)

    def embed_text(self, text: str) -> list[float]:
        """Embed text query using CLIP.

        Args:
            text: Text to embed

        Returns:
            Normalized embedding vector as list of floats
        """
        import torch

        model, _, tokenizer = _load_model()

        with torch.no_grad():
            tokens = tokenizer([text]).to(self.device)
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        result: list[float] = text_features[0].cpu().numpy().tolist()
        return result

    def embed_image(self, image_path: str | Path) -> list[float]:
        """Embed image using CLIP.

        Args:
            image_path: Path to image file

        Returns:
            Normalized embedding vector as list of floats
        """
        import torch

        model, preprocess, _ = _load_model()

        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        result: list[float] = image_features[0].cpu().numpy().tolist()
        return result


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """Get singleton embedding service."""
    return EmbeddingService()
