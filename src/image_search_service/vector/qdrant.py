"""Qdrant vector database client wrapper with lazy initialization."""


from qdrant_client import QdrantClient

from image_search_service.core.config import get_settings
from image_search_service.core.logging import get_logger

logger = get_logger(__name__)

_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    """Get or create Qdrant client (lazy initialization)."""
    global _client

    if _client is None:
        settings = get_settings()
        _client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key if settings.qdrant_api_key else None,
        )
        logger.info("Qdrant client initialized")

    return _client


def ping() -> bool:
    """Check if Qdrant server is accessible.

    Returns:
        True if server is accessible, False otherwise
    """
    try:
        client = get_qdrant_client()
        # Try to get collections to verify connection
        client.get_collections()
        return True
    except Exception as e:
        logger.warning("Qdrant ping failed: %s", e)
        return False


def close_qdrant() -> None:
    """Close Qdrant client and cleanup resources."""
    global _client

    if _client is not None:
        _client.close()
        _client = None
        logger.info("Qdrant client closed")
