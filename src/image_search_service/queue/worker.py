"""RQ worker entry point."""

from redis import Redis
from rq import Worker

from image_search_service.core.config import get_settings
from image_search_service.core.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)


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
