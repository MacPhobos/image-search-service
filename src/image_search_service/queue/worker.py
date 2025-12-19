"""RQ worker entry point."""

from redis import Redis
from rq import Queue, Worker

from image_search_service.core.config import get_settings
from image_search_service.core.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

_queue: Queue | None = None


def get_queue() -> Queue:
    """Get or create RQ queue (lazy initialization).

    Returns:
        RQ Queue instance connected to Redis
    """
    global _queue

    if _queue is None:
        settings = get_settings()
        redis_conn = Redis.from_url(settings.redis_url)
        _queue = Queue("default", connection=redis_conn)
        logger.info("RQ queue initialized")

    return _queue


def main() -> None:
    """Start RQ worker to process background jobs."""
    settings = get_settings()

    logger.info("Starting RQ worker")
    logger.info("Redis URL: %s", settings.redis_url)

    # Connect to Redis
    redis_conn = Redis.from_url(settings.redis_url)

    # Create worker and start processing
    worker = Worker(["default"], connection=redis_conn)
    worker.work()


if __name__ == "__main__":
    main()
