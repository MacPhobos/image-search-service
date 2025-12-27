"""Qdrant vector database client wrapper with lazy initialization."""

import os
from typing import Any, cast
from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
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


_collection_ensured: set[str] = set()


def ensure_collection(embedding_dim: int) -> None:
    """Create collection if it doesn't exist.

    Uses module-level cache to avoid redundant API calls.

    Args:
        embedding_dim: Dimension of embedding vectors
    """
    settings = get_settings()

    # Fast path: already ensured in this process
    if settings.qdrant_collection in _collection_ensured:
        return

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

    # Mark as ensured
    _collection_ensured.add(settings.qdrant_collection)


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


def upsert_vectors_batch(
    points: list[dict[str, Any]],
    wait: bool = True,
) -> int:
    """Batch upsert multiple vector points into Qdrant.

    Args:
        points: List of dicts with keys: asset_id, vector, payload
        wait: Whether to wait for operation to complete (default: True)

    Returns:
        Number of points upserted
    """
    if not points:
        return 0

    settings = get_settings()
    client = get_qdrant_client()

    qdrant_points = [
        PointStruct(
            id=p["asset_id"],
            vector=p["vector"],
            payload={**p.get("payload", {}), "asset_id": str(p["asset_id"])},
        )
        for p in points
    ]

    client.upsert(
        collection_name=settings.qdrant_collection,
        points=qdrant_points,
        wait=wait,
    )

    return len(qdrant_points)


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


# ========== Deletion Methods ==========


def delete_vectors_by_directory(
    path_prefix: str,
    batch_size: int = 100,
    client: QdrantClient | None = None,
) -> int:
    """Delete all vectors where path starts with path_prefix.

    Uses scroll API to find matching vectors, then deletes by point IDs.

    Args:
        path_prefix: Directory path prefix (e.g., "/photos/2024/")
        batch_size: Number of vectors to delete per batch
        client: Optional Qdrant client override

    Returns:
        Total number of vectors deleted
    """
    settings = get_settings()
    if client is None:
        client = get_qdrant_client()

    try:
        deleted_count = 0
        offset = None

        while True:
            # Scroll through all points and filter by path prefix in Python
            # (Qdrant doesn't support prefix matching in filters natively)
            scroll_result = client.scroll(
                collection_name=settings.qdrant_collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
            )

            points = scroll_result[0]
            next_offset = scroll_result[1]

            if not points:
                break

            # Filter points by path prefix
            matching_ids = [
                point.id
                for point in points
                if point.payload
                and isinstance(point.payload.get("path"), str)
                and point.payload["path"].startswith(path_prefix)
            ]

            # Delete matching points
            if matching_ids:
                client.delete(
                    collection_name=settings.qdrant_collection,
                    points_selector=PointIdsList(points=matching_ids),
                )
                deleted_count += len(matching_ids)
                logger.info(f"Deleted {len(matching_ids)} vectors with path prefix '{path_prefix}'")

            # Check if we've scrolled through all points
            if next_offset is None:
                break

            offset = next_offset

        logger.info(f"Total deleted {deleted_count} vectors with path prefix '{path_prefix}'")
        return deleted_count

    except Exception as e:
        logger.error(f"Failed to delete vectors by directory '{path_prefix}': {e}")
        raise


def delete_vectors_by_asset(
    asset_id: int,
    client: QdrantClient | None = None,
) -> int:
    """Delete vector for a specific asset_id.

    Args:
        asset_id: The asset ID (point ID in Qdrant)
        client: Optional Qdrant client override

    Returns:
        Number of vectors deleted (0 or 1)
    """
    settings = get_settings()
    if client is None:
        client = get_qdrant_client()

    try:
        client.delete(
            collection_name=settings.qdrant_collection,
            points_selector=PointIdsList(points=[asset_id]),
        )
        logger.info(f"Deleted vector for asset_id {asset_id}")
        return 1

    except Exception as e:
        logger.error(f"Failed to delete vector for asset_id {asset_id}: {e}")
        raise


def delete_vectors_by_session(
    session_id: int,
    asset_ids: list[int],
    client: QdrantClient | None = None,
) -> int:
    """Delete all vectors for assets in a training session.

    Args:
        session_id: Training session ID (for logging)
        asset_ids: List of asset IDs to delete
        client: Optional Qdrant client override

    Returns:
        Total number of vectors deleted
    """
    settings = get_settings()
    if client is None:
        client = get_qdrant_client()

    try:
        if not asset_ids:
            logger.info(f"No assets to delete for session {session_id}")
            return 0

        client.delete(
            collection_name=settings.qdrant_collection,
            points_selector=PointIdsList(points=cast(list[int | str | UUID], asset_ids)),
        )
        deleted_count = len(asset_ids)
        logger.info(f"Deleted {deleted_count} vectors for session {session_id}")
        return deleted_count

    except Exception as e:
        logger.error(f"Failed to delete vectors for session {session_id}: {e}")
        raise


def delete_vectors_by_category(
    category_id: int,
    batch_size: int = 100,
    client: QdrantClient | None = None,
) -> int:
    """Delete all vectors with matching category_id in payload.

    Uses scroll API with category filter, then deletes by point IDs.

    Args:
        category_id: Category ID to match
        batch_size: Number of vectors to delete per batch
        client: Optional Qdrant client override

    Returns:
        Total number of vectors deleted
    """
    settings = get_settings()
    if client is None:
        client = get_qdrant_client()

    try:
        deleted_count = 0
        offset = None

        # Build filter for category_id
        category_filter = Filter(
            must=[FieldCondition(key="category_id", match=MatchValue(value=category_id))]
        )

        while True:
            # Scroll with category filter
            scroll_result = client.scroll(
                collection_name=settings.qdrant_collection,
                scroll_filter=category_filter,
                limit=batch_size,
                offset=offset,
                with_payload=False,  # We only need IDs
            )

            points = scroll_result[0]
            next_offset = scroll_result[1]

            if not points:
                break

            # Extract point IDs
            point_ids = [point.id for point in points]

            # Delete batch
            if point_ids:
                client.delete(
                    collection_name=settings.qdrant_collection,
                    points_selector=PointIdsList(points=point_ids),
                )
                deleted_count += len(point_ids)
                logger.info(f"Deleted {len(point_ids)} vectors with category_id {category_id}")

            # Check if we've scrolled through all matching points
            if next_offset is None:
                break

            offset = next_offset

        logger.info(f"Total deleted {deleted_count} vectors with category_id {category_id}")
        return deleted_count

    except Exception as e:
        logger.error(f"Failed to delete vectors by category {category_id}: {e}")
        raise


def get_directory_stats(
    client: QdrantClient | None = None,
) -> list[dict[str, Any]]:
    """Get statistics about directories with vectors.

    Scrolls through all vectors and groups by directory path.

    Args:
        client: Optional Qdrant client override

    Returns:
        List of dicts with: path_prefix, vector_count, last_indexed
    """
    settings = get_settings()
    if client is None:
        client = get_qdrant_client()

    try:
        directory_stats: dict[str, dict[str, Any]] = {}
        offset = None

        while True:
            scroll_result = client.scroll(
                collection_name=settings.qdrant_collection,
                limit=100,
                offset=offset,
                with_payload=True,
            )

            points = scroll_result[0]
            next_offset = scroll_result[1]

            if not points:
                break

            # Group by directory (extract directory from path)
            for point in points:
                if point.payload and isinstance(point.payload.get("path"), str):
                    path = point.payload["path"]
                    # Extract directory (everything before the last /)
                    directory = path.rsplit("/", 1)[0] if "/" in path else "/"

                    if directory not in directory_stats:
                        directory_stats[directory] = {
                            "path_prefix": directory,
                            "vector_count": 0,
                            "last_indexed": None,
                        }

                    directory_stats[directory]["vector_count"] += 1

                    # Track most recent created_at
                    created_at = point.payload.get("created_at")
                    if created_at:
                        current_last = directory_stats[directory]["last_indexed"]
                        if current_last is None or created_at > current_last:
                            directory_stats[directory]["last_indexed"] = created_at

            # Check if we've scrolled through all points
            if next_offset is None:
                break

            offset = next_offset

        # Convert to list and sort by vector count descending
        stats_list = sorted(
            directory_stats.values(),
            key=lambda x: x["vector_count"],
            reverse=True,
        )

        logger.info(f"Retrieved stats for {len(stats_list)} directories")
        return stats_list

    except Exception as e:
        logger.error(f"Failed to get directory stats: {e}")
        raise


def delete_orphan_vectors(
    valid_asset_ids: set[int],
    batch_size: int = 100,
    client: QdrantClient | None = None,
) -> int:
    """Delete vectors whose asset_id is not in valid_asset_ids.

    Used for cleanup when database records are deleted but vectors remain.

    Args:
        valid_asset_ids: Set of asset IDs that exist in database
        batch_size: Number of vectors to process per batch
        client: Optional Qdrant client override

    Returns:
        Total number of orphan vectors deleted
    """
    settings = get_settings()
    if client is None:
        client = get_qdrant_client()

    try:
        deleted_count = 0
        offset = None

        while True:
            scroll_result = client.scroll(
                collection_name=settings.qdrant_collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
            )

            points = scroll_result[0]
            next_offset = scroll_result[1]

            if not points:
                break

            # Find orphan points (point.id not in valid_asset_ids)
            orphan_ids = [
                point.id
                for point in points
                if isinstance(point.id, int) and point.id not in valid_asset_ids
            ]

            # Delete orphan points
            if orphan_ids:
                client.delete(
                    collection_name=settings.qdrant_collection,
                    points_selector=PointIdsList(points=cast(list[int | str | UUID], orphan_ids)),
                )
                deleted_count += len(orphan_ids)
                logger.info(f"Deleted {len(orphan_ids)} orphan vectors")

            # Check if we've scrolled through all points
            if next_offset is None:
                break

            offset = next_offset

        logger.info(f"Total deleted {deleted_count} orphan vectors")
        return deleted_count

    except Exception as e:
        logger.error(f"Failed to delete orphan vectors: {e}")
        raise


def reset_collection(
    client: QdrantClient | None = None,
) -> int:
    """Delete all vectors in the collection.

    WARNING: This is destructive and cannot be undone.

    Args:
        client: Optional Qdrant client override

    Returns:
        Total number of vectors deleted
    """
    settings = get_settings()
    collection_name = settings.qdrant_collection

    # CRITICAL SAFETY GUARD: Prevent deletion of production collections during tests
    if os.getenv("PYTEST_CURRENT_TEST") and collection_name in ["image_assets", "faces"]:
        raise RuntimeError(
            f"Refusing to delete production collection '{collection_name}' during tests. "
            "Ensure test settings are properly configured with test-safe collection names."
        )

    if client is None:
        client = get_qdrant_client()

    try:
        # Get count before deletion
        collection_info = client.get_collection(collection_name=collection_name)
        vector_count = collection_info.points_count or 0

        # Delete all points by recreating the collection
        # This is more efficient than scrolling and deleting batches
        client.delete_collection(collection_name=collection_name)
        logger.warning(f"Deleted collection '{collection_name}'")

        # Recreate empty collection with same parameters
        # Get vector size from the deleted collection config
        vector_size = collection_info.config.params.vectors.size  # type: ignore[union-attr]
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        logger.info(f"Recreated collection '{collection_name}' with dim={vector_size}")

        logger.warning(f"Reset collection: deleted {vector_count} vectors")
        return vector_count

    except Exception as e:
        logger.error(f"Failed to reset collection: {e}")
        raise
