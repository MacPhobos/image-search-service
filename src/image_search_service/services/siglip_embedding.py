"""SigLIP embedding service for improved semantic search.

SigLIP provides significantly better retrieval accuracy compared to CLIP:
- 15-25% improvement in retrieval metrics
- Better fine-grained similarity matching
- More efficient inference

This service mirrors the existing CLIP embedding service (embedding.py) but uses
the SigLIP model from open_clip with 768-dimensional embeddings instead of 512.

Model Caching
=============
Similar to CLIP, the SigLIP model is lazily loaded and cached globally to avoid
reloading on each request. The model is loaded once on first use and reused
across all subsequent calls.
"""

import multiprocessing
from functools import lru_cache
from pathlib import Path
from threading import Lock
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
_model_lock = Lock()


def _is_main_process() -> bool:
    """Check if running in main process (not a subprocess).

    Returns:
        True if in main process, False if in subprocess
    """
    try:
        return multiprocessing.current_process().name == "MainProcess"
    except Exception:
        # If we can't determine, assume main process (safe default)
        return True


def preload_siglip_model() -> None:
    """Preload SigLIP model in main process before workers fork.

    This function should be called during application startup in the main
    FastAPI process. It ensures the model is loaded and cached in memory
    before RQ workers fork, so workers inherit the already-loaded model
    object. This avoids Metal compiler service initialization issues in
    subprocess contexts on macOS.

    Safe to call multiple times - only loads once due to global state check.
    """
    if not _is_main_process():
        logger.debug("preload_siglip_model called in subprocess, skipping")
        return

    global _model, _preprocess, _tokenizer

    with _model_lock:
        if _model is not None:
            logger.debug("SigLIP model already preloaded, skipping")
            return

        logger.info("Preloading SigLIP model in main process")
        _load_model()
        logger.info("SigLIP model preloaded successfully")


def _load_model() -> tuple[Any, Any, Any]:
    """Lazy load SigLIP model from open_clip.

    Loads the model on first call and caches globally. In the main process,
    this is called during startup via preload_siglip_model(). In worker
    subprocesses, the model should already be cached from inheritance.

    Returns:
        Tuple of (model, preprocess, tokenizer)
    """
    global _model, _preprocess, _tokenizer

    if _model is not None:
        return _model, _preprocess, _tokenizer

    import open_clip

    settings = get_settings()
    device = get_device()

    logger.info(
        f"Loading SigLIP model {settings.siglip_model_name} "
        f"(pretrained={settings.siglip_pretrained}) on {device}"
    )

    model, _, preprocess = open_clip.create_model_and_transforms(
        settings.siglip_model_name, pretrained=settings.siglip_pretrained
    )
    model = model.to(device)
    model.eval()

    tokenizer = open_clip.get_tokenizer(settings.siglip_model_name)

    _model = model
    _preprocess = preprocess
    _tokenizer = tokenizer

    logger.info(f"SigLIP model loaded. Embedding dim: {model.visual.output_dim}")
    return model, preprocess, tokenizer


class SigLIPEmbeddingService:
    """Service for generating SigLIP embeddings with improved retrieval accuracy."""

    def __init__(self) -> None:
        """Initialize SigLIP embedding service."""
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
        """Embed text query using SigLIP.

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
        """Embed image using SigLIP.

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
        """Embed a pre-loaded PIL image using SigLIP.

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
        self, images: list["Image.Image"], batch_size: int = 32
    ) -> list[list[float]]:
        """Embed multiple pre-loaded PIL images in batches for efficiency.

        This is significantly faster than calling embed_image() multiple times
        as it batches the GPU inference.

        Memory Management:
        - Explicitly deletes batch tensors after inference to prevent GPU memory
          accumulation on MPS (Metal Performance Shaders on macOS)
        - MPS relies on Python garbage collection and doesn't have cuda.empty_cache()
        - Explicit deletion + gc.collect() ensures timely memory release

        Args:
            images: List of PIL Image objects (already loaded)
            batch_size: Number of images to process per batch (default: 32)

        Returns:
            List of normalized embedding vectors
        """
        import gc

        import torch

        if not images:
            return []

        model, preprocess, _ = _load_model()

        all_embeddings: list[list[float]] = []

        # Process in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i : i + batch_size]

            # Preprocess batch
            tensors = []
            for img in batch_images:
                img_rgb = img.convert("RGB")
                tensor = preprocess(img_rgb)
                tensors.append(tensor)

            # Stack into batch: (N, C, H, W)
            batch_tensor = torch.stack(tensors).to(self.device)

            with torch.no_grad():
                image_features = model.encode_image(batch_tensor)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )

            # Convert to list of lists
            batch_embeddings: list[list[float]] = (
                image_features.cpu().numpy().tolist()
            )
            all_embeddings.extend(batch_embeddings)

            # Explicit GPU memory cleanup (critical for MPS on macOS)
            # Delete intermediate tensors immediately
            del batch_tensor
            del image_features
            del tensors

            # Force garbage collection to free GPU memory
            # This is especially important for MPS which doesn't have empty_cache()
            # Safe on CUDA (just adds small overhead) and CPU (no-op)
            gc.collect()

        return all_embeddings


@lru_cache(maxsize=1)
def get_siglip_service() -> SigLIPEmbeddingService:
    """Get singleton SigLIP embedding service."""
    return SigLIPEmbeddingService()
