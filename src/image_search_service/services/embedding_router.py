"""Router to select between CLIP and SigLIP based on feature flags.

This module provides intelligent routing between the legacy CLIP embedding service
and the newer SigLIP service based on configuration flags. It supports:

1. Full SigLIP mode (use_siglip=True): All traffic uses SigLIP
2. Gradual rollout (siglip_rollout_percentage > 0): Percentage-based routing
3. Legacy CLIP mode (default): All traffic uses CLIP

Gradual Rollout Strategy
========================
When siglip_rollout_percentage is set (e.g., 10 for 10%), the router will:
- Use deterministic bucketing for user_id (if provided) for consistent experience
- Use random bucketing if no user_id (e.g., anonymous searches)
- Route to SigLIP if bucket < rollout_percentage

Example:
    - siglip_rollout_percentage=50
    - user_id=123 → bucket=23 → 23 < 50 → SigLIP
    - user_id=456 → bucket=56 → 56 >= 50 → CLIP

This allows A/B testing and gradual migration from CLIP to SigLIP.
"""

import random

from image_search_service.core.config import get_settings
from image_search_service.services.embedding import (
    EmbeddingService,
    get_embedding_service,
)
from image_search_service.services.siglip_embedding import (
    SigLIPEmbeddingService,
    get_siglip_service,
)


def get_search_embedding_service(
    user_id: int | None = None,
) -> tuple[EmbeddingService | SigLIPEmbeddingService, str]:
    """Get embedding service and collection based on feature flags and rollout.

    This function routes requests to either CLIP or SigLIP based on:
    1. Full SigLIP mode (settings.use_siglip)
    2. Gradual rollout percentage (settings.siglip_rollout_percentage)
    3. Default CLIP mode

    For gradual rollout, routing is deterministic per user_id to ensure
    consistent experience (same user always gets same model). If no user_id
    is provided, random routing is used.

    Args:
        user_id: Optional user ID for deterministic A/B bucketing

    Returns:
        Tuple of (embedding_service, collection_name)
        - embedding_service: Either EmbeddingService (CLIP) or SigLIPEmbeddingService
        - collection_name: Qdrant collection to search ("image_assets" or "image_assets_siglip")

    Examples:
        # Full SigLIP mode
        USE_SIGLIP=true → Always returns (SigLIPService, "image_assets_siglip")

        # Gradual rollout (50%)
        SIGLIP_ROLLOUT_PERCENTAGE=50, user_id=123
        → bucket=23 → Returns (SigLIPService, "image_assets_siglip")

        # Legacy CLIP mode
        (default settings) → Returns (EmbeddingService, "image_assets")
    """
    settings = get_settings()

    # Full SigLIP mode
    if settings.use_siglip:
        return get_siglip_service(), settings.siglip_collection

    # Gradual rollout by percentage
    if settings.siglip_rollout_percentage > 0:
        # Deterministic bucketing for same user (consistent experience)
        if user_id is not None:
            bucket = user_id % 100
        else:
            # Random bucket for anonymous/no user_id
            bucket = random.randint(0, 99)

        # Route to SigLIP if bucket falls within rollout percentage
        if bucket < settings.siglip_rollout_percentage:
            return get_siglip_service(), settings.siglip_collection

    # Default to CLIP
    return get_embedding_service(), settings.qdrant_collection
