"""RQ worker entry point with priority queues."""

from redis import Redis
from rq import Queue, Worker

from image_search_service.core.config import get_settings
from image_search_service.core.logging import configure_logging, get_logger

configure_logging()
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


def get_all_queues() -> list[Queue]:
    """Get all queues in priority order.

    Returns:
        List of Queue instances ordered by priority (high to low)
    """
    return [
        get_queue(QUEUE_HIGH),
        get_queue(QUEUE_NORMAL),
        get_queue(QUEUE_LOW),
        get_queue(QUEUE_DEFAULT),
    ]


def main() -> None:
    """Start RQ worker to process background jobs from all queues."""
    settings = get_settings()

    logger.info("Starting RQ worker with priority queues")
    logger.info("Redis URL: %s", settings.redis_url)

    # Connect to Redis
    redis_conn = get_redis()

    # Create worker with all queues in priority order
    queue_names = [QUEUE_HIGH, QUEUE_NORMAL, QUEUE_LOW, QUEUE_DEFAULT]
    logger.info(f"Processing queues in order: {queue_names}")

    worker = Worker(queue_names, connection=redis_conn)
    worker.work()


if __name__ == "__main__":
    main()
