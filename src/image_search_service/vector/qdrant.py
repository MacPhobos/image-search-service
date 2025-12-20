"""Qdrant vector database client wrapper with lazy initialization."""

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)

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


def ensure_collection(embedding_dim: int) -> None:
    """Create collection if it doesn't exist.

    Args:
        embedding_dim: Dimension of embedding vectors
    """
    settings = get_settings()
    client = get_qdrant_client()

    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    if settings.qdrant_collection not in collection_names:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
        )
        logger.info(
            f"Created Qdrant collection '{settings.qdrant_collection}' with dim={embedding_dim}"
        )


def upsert_vector(asset_id: int, vector: list[float], payload: dict[str, str | int]) -> None:
    """Upsert a vector point into Qdrant.

    Args:
        asset_id: Asset ID (used as point ID)
        vector: Embedding vector
        payload: Additional metadata (strings and integers)
    """
    settings = get_settings()
    client = get_qdrant_client()

    client.upsert(
        collection_name=settings.qdrant_collection,
        points=[
            PointStruct(id=asset_id, vector=vector, payload={**payload, "asset_id": str(asset_id)})
        ],
    )


def search_vectors(
    query_vector: list[float],
    limit: int = 50,
    offset: int = 0,
    filters: dict[str, str | int] | None = None,
    client: QdrantClient | None = None,
) -> list[dict[str, Any]]:
    """Search for similar vectors.

    Args:
        query_vector: Query embedding vector
        limit: Maximum number of results
        offset: Offset for pagination
        filters: Optional filters (from_date, to_date, category_id)
        client: Optional Qdrant client (uses default if not provided)

    Returns:
        List of search results with asset_id, score, and payload
    """
    settings = get_settings()
    if client is None:
        client = get_qdrant_client()

    # Build filter if date range or category_id provided
    qdrant_filter = None
    if filters:
        conditions: list[FieldCondition] = []
        if filters.get("from_date"):
            # Note: Qdrant Range filter expects numeric values. If using date strings,
            # they should be converted to timestamps. For now, we type: ignore this.
            conditions.append(
                FieldCondition(key="created_at", range=Range(gte=filters["from_date"]))  # type: ignore[arg-type]
            )
        if filters.get("to_date"):
            conditions.append(
                FieldCondition(key="created_at", range=Range(lte=filters["to_date"]))  # type: ignore[arg-type]
            )
        if filters.get("category_id"):
            conditions.append(
                FieldCondition(key="category_id", match=MatchValue(value=filters["category_id"]))
            )
        if conditions:
            # Filter.must accepts various condition types, use type: ignore for simplicity
            qdrant_filter = Filter(must=conditions)  # type: ignore[arg-type]

    # Use query_points() instead of search() for modern qdrant-client API
    results = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=limit,
        offset=offset,
        query_filter=qdrant_filter,
        with_payload=True,
    )

    return [
        {
            "asset_id": hit.payload.get("asset_id") if hit.payload else None,
            "score": hit.score,
            "payload": hit.payload or {},
        }
        for hit in results.points
    ]


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
