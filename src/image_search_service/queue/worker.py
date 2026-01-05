"""RQ worker entry point with priority queues.

Custom Worker Implementation
=============================
To fix MPS crashes on macOS, we use a custom Worker class that preloads
the embedding model in the work-horse subprocess (where the actual job runs).

The root cause: RQ spawns work-horse subprocesses with fresh Python interpreters
using spawn() on macOS (not fork()). This means a preload in the main worker
process doesn't help the work-horse since it has completely separate module globals.

The solution: Override execute_job() to preload the model in the work-horse
subprocess right before the job runs. This happens in the correct process context
where Metal compiler service is available.
"""

from redis import Redis
from rq import Queue, Worker
from rq.job import Job
from rq.queue import Queue as QueueType

from image_search_service.core.config import get_settings
from image_search_service.core.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)


class EmbeddingPreloadWorker(Worker):
    """Custom RQ Worker that preloads embedding model in work-horse subprocess.

    This worker overrides execute_job() to preload the embedding model
    in the work-horse subprocess context BEFORE the job runs. This prevents
    Metal compiler service initialization issues on macOS when using MPS.

    The work-horse subprocess is where RQ actually executes jobs. By preloading
    the model there (rather than in the main worker process), we ensure the
    model initialization happens in the correct process context.
    """

    def execute_job(self, job: Job, queue: QueueType) -> bool:
        """Execute a job with embedding model preloaded in work-horse context.

        Preloads the embedding model in the work-horse subprocess before
        executing the job. This ensures OpenCLIP model initialization happens
        in the work-horse process before any job needs the embedding service.

        Args:
            job: RQ Job to execute
            queue: Queue the job came from

        Returns:
            True if job succeeded, False if job failed
        """
        # Preload embedding model in work-horse subprocess context
        # This MUST happen here (in work-horse), not in main worker process
        try:
            from image_search_service.services.embedding import (
                preload_embedding_model,
            )

            logger.debug("Preloading embedding model in work-horse subprocess")
            preload_embedding_model()
            logger.debug("Embedding model preloaded successfully in work-horse")
        except Exception as e:
            logger.warning(
                f"Failed to preload embedding model in work-horse: {e}. "
                "Will load on first use (may cause MPS crash on macOS)."
            )

        # Execute the job with model preloaded in work-horse context
        return super().execute_job(job, queue)  # type: ignore[no-any-return]

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
    """Start RQ worker to process background jobs from all queues.

    Uses EmbeddingPreloadWorker to ensure embedding model is preloaded
    in the work-horse subprocess context (not main worker process).
    """
    settings = get_settings()

    logger.info("Starting RQ worker with priority queues")
    logger.info("Redis URL: %s", settings.redis_url)

    # Connect to Redis
    redis_conn = get_redis()

    # Create custom worker with all queues in priority order
    # Uses EmbeddingPreloadWorker to preload model in work-horse subprocess
    queue_names = [QUEUE_HIGH, QUEUE_NORMAL, QUEUE_LOW, QUEUE_DEFAULT]
    logger.info(f"Processing queues in order: {queue_names}")

    worker = EmbeddingPreloadWorker(queue_names, connection=redis_conn)
    worker.work()


if __name__ == "__main__":
    main()
