"""Embedding service using OpenCLIP for image and text embeddings."""

from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image

from image_search_service.core.config import get_settings
from image_search_service.core.device import get_device
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

    logger.info(f"Loading OpenCLIP model {settings.clip_model_name} on {device}")

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

    def embed_image_from_pil(self, image: "Image.Image") -> list[float]:
        """Embed a pre-loaded PIL image using CLIP.

        Args:
            image: PIL Image object (already loaded)

        Returns:
            Normalized embedding vector as list of floats
        """
        import torch

        model, preprocess, _ = _load_model()

        image_rgb = image.convert("RGB")
        image_tensor = preprocess(image_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        result: list[float] = image_features[0].cpu().numpy().tolist()
        return result

    def embed_images_batch(
        self, images: list["Image.Image"]
    ) -> list[list[float]]:
        """Embed multiple pre-loaded PIL images in a single GPU batch.

        This is significantly faster than calling embed_image() multiple times
        as it batches the GPU inference.

        Args:
            images: List of PIL Image objects (already loaded)

        Returns:
            List of normalized embedding vectors
        """
        import torch

        if not images:
            return []

        model, preprocess, _ = _load_model()

        # Preprocess all images and stack into batch tensor
        tensors = []
        for img in images:
            img_rgb = img.convert("RGB")
            tensor = preprocess(img_rgb)
            tensors.append(tensor)

        # Stack into batch: (N, C, H, W)
        batch_tensor = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            image_features = model.encode_image(batch_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Convert to list of lists
        results: list[list[float]] = image_features.cpu().numpy().tolist()
        return results


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """Get singleton embedding service."""
    return EmbeddingService()
