"""RQ Worker entry point - routes to Listener-based worker.

This module is maintained for import compatibility.
All work is delegated to listener_worker.py which implements the RQ Listener pattern.

The queue management functions (get_queue, get_redis, etc.) are preserved for
backward compatibility with existing code.
"""

from collections.abc import Collection

from redis import Redis
from rq import Queue

from image_search_service.core.config import get_settings
from image_search_service.core.logging import get_logger
from image_search_service.queue.listener_worker import main

logger = get_logger(__name__)

# Queue names by priority
QUEUE_HIGH = "training-high"  # User-initiated training
QUEUE_NORMAL = "training-normal"  # Scheduled tasks
QUEUE_LOW = "training-low"  # Thumbnails, cleanup
QUEUE_DEFAULT = "default"  # Legacy support

_queues: dict[str, Queue] = {}
_redis_conn: Redis | None = None


def get_redis() -> Redis:
    """Get or create Redis connection (lazy initialization).

    Returns:
        Redis connection instance
    """
    global _redis_conn

    if _redis_conn is None:
        settings = get_settings()
        _redis_conn = Redis.from_url(settings.redis_url)
        logger.info("Redis connection initialized")

    return _redis_conn


def get_queue(priority: str = QUEUE_NORMAL) -> Queue:
    """Get or create RQ queue by priority (lazy initialization).

    Args:
        priority: Queue priority name (QUEUE_HIGH, QUEUE_NORMAL, QUEUE_LOW, QUEUE_DEFAULT)

    Returns:
        RQ Queue instance connected to Redis
    """
    global _queues

    if priority not in _queues:
        redis_conn = get_redis()
        _queues[priority] = Queue(priority, connection=redis_conn)
        logger.info(f"RQ queue '{priority}' initialized")

    return _queues[priority]


def enqueue_person_ids_update(asset_ids: Collection[int]) -> int:
    """Enqueue person_ids update jobs for the given asset IDs.

    Each asset gets a separate job that refreshes the person_ids payload
    in the Qdrant image_assets collection. The job is idempotent, so
    duplicate enqueues are harmless.

    This helper swallows all exceptions so that a queueing failure never
    breaks the caller's HTTP response.

    Args:
        asset_ids: Asset IDs whose person_ids payloads need refreshing.

    Returns:
        Number of jobs successfully enqueued (0 on failure).
    """
    if not asset_ids:
        return 0

    try:
        from image_search_service.queue.jobs import update_asset_person_ids_job

        queue = get_queue(QUEUE_DEFAULT)
        for asset_id in asset_ids:
            queue.enqueue(
                update_asset_person_ids_job,
                asset_id=asset_id,
                job_timeout="5m",
            )
        logger.info(
            "Enqueued %d person_ids update job(s)",
            len(asset_ids),
        )
        return len(asset_ids)
    except Exception as e:
        logger.warning("Failed to enqueue person_ids update jobs: %s", e)
        return 0


if __name__ == "__main__":
    main()
