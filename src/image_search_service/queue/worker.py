"""RQ worker entry point with platform-aware single-worker enforcement.

On macOS, enforce single worker to prevent GPU fork-safety crashes.
On Linux, allow multiple workers (no GPU fork-safety issues).

CRITICAL: Proxy detection disabled at module import time
============================================================
System proxy detection is disabled immediately at module import time (before any
other imports) to prevent fork-safety crashes in work-horse subprocesses.

This must happen at module level because:
- Work-horse processes spawned via spawn() reload the module
- Proxy detection via urllib can fork in multi-threaded context
- Setting env vars at module level ensures they're set in subprocess too
"""

# CRITICAL: Disable proxy detection FIRST, before ANY other imports
# This prevents urllib from forking when detecting system proxies
import os
import platform

if platform.system() == "Darwin":  # Only on macOS
    # Disable environment-based proxy detection
    os.environ["HTTP_PROXY"] = ""
    os.environ["HTTPS_PROXY"] = ""
    os.environ["FTP_PROXY"] = ""
    os.environ["http_proxy"] = ""
    os.environ["https_proxy"] = ""
    os.environ["ftp_proxy"] = ""
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
    os.environ["all_proxy"] = ""
    os.environ["ALL_PROXY"] = ""
    os.environ.pop("REQUEST_METHOD", None)

# Now safe to import other modules
"""
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

Single Worker Enforcement on macOS
===================================
Additionally, we enforce a single worker process on macOS to prevent GPU fork-safety
crashes from multiple concurrent fork operations. The Metal GPU driver is not fork-safe
when called from multiple threads simultaneously. By limiting to 1 worker:
- Single thread pool = safer fork context
- Still spawns work-horse per job (RQ requirement)
- Work-horse environment is cleaner (fewer parent threads)
- MPS GPU remains fully enabled and functional
"""

import os
import platform
import tempfile
from pathlib import Path

from redis import Redis
from rq import Queue, Worker
from rq.job import Job
from rq.queue import Queue as QueueType

from image_search_service.core.config import get_settings
from image_search_service.core.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)


def _get_worker_lock_file() -> Path:
    """Get platform-specific worker lock file path.

    On macOS, this creates a lock to ensure only one worker runs.
    On Linux, lock file is not used (multiple workers allowed).

    Returns:
        Path to lock file on macOS, empty Path on Linux
    """
    if platform.system() == "Darwin":  # macOS
        lock_dir = Path(tempfile.gettempdir()) / "image_search"
        lock_dir.mkdir(parents=True, exist_ok=True)
        return lock_dir / "rq_worker.lock"
    return Path()  # Return empty path on Linux


def _check_single_worker_on_macos() -> bool:
    """Check if another worker is already running on macOS.

    On macOS, enforce single worker to prevent GPU fork-safety crashes
    from multiple concurrent fork operations. The Metal GPU driver crashes
    when work-horse subprocesses are forked from multiple thread pools.

    Returns:
        True if on macOS and lock acquired successfully
        False if on macOS and another worker already running
        True if on Linux (no single-worker restriction)
    """
    if platform.system() != "Darwin":
        return True  # Allow multiple workers on Linux

    lock_file = _get_worker_lock_file()

    # Check if lock file exists (another worker running)
    if lock_file.exists():
        try:
            # Try to read PID from lock file
            existing_pid = lock_file.read_text().strip()
            logger.warning(
                f"Another RQ worker is already running (PID: {existing_pid}). "
                f"On macOS, only one worker is allowed to prevent GPU fork-safety crashes. "
                f"Kill existing worker with: kill {existing_pid}"
            )
            return False
        except Exception as e:
            logger.warning(f"Could not read lock file: {e}. Attempting to start anyway.")

    # Create lock file with current PID
    try:
        lock_file.write_text(str(os.getpid()))
        logger.info(f"macOS worker lock acquired (PID: {os.getpid()})")
        return True
    except Exception as e:
        logger.error(f"Failed to create worker lock file: {e}")
        return False


def _cleanup_worker_lock() -> None:
    """Clean up worker lock file on exit.

    Called in finally block to ensure lock is released even if worker crashes.

    Lock File Management (macOS):
    =============================
    - Location: /tmp/image_search/rq_worker.lock
    - Contains: PID of running worker process
    - Lifetime: From worker startup → graceful shutdown or crash
    - Cleanup: Automatic via finally block in main()
    - Manual restart: rm /tmp/image_search/rq_worker.lock && make worker

    Why lock file matters:
    - Prevents multiple workers from initializing GPU simultaneously
    - Each fork operation with active GPU state could crash Metal driver
    - Single worker + lock file = safe GPU fork operations
    """
    if platform.system() == "Darwin":  # macOS
        lock_file = _get_worker_lock_file()
        try:
            if lock_file.exists():
                lock_file.unlink()
                logger.info("macOS worker lock released")
        except Exception as e:
            logger.warning(f"Failed to clean up lock file: {e}")


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

    CRITICAL: Fork-Safety on macOS
    ===============================
    This function must disable system proxy detection immediately to prevent
    fork-safety crashes in work-horse subprocesses. urllib's proxy detection
    on macOS calls fork() when in a multi-threaded context, causing Signal 11.

    macOS CONSTRAINTS & FORK-SAFETY REQUIREMENTS
    =============================================
    On macOS with Apple Silicon GPU (MPS):

    1. Single Worker Enforcement:
       - Only ONE worker process allowed per machine
       - Prevents concurrent fork operations from multiple thread pools
       - Metal GPU driver is not fork-safe when called from multiple threads
       - Lock file: /tmp/image_search/rq_worker.lock (cleaned up on exit)
       - To restart: kill <PID> && rm /tmp/image_search/rq_worker.lock

    2. Fork-Unsafe Libraries (Explicitly Disabled on macOS):
       - Kerberos GSS Authentication (libgssapi_krb5)
       - PostgreSQL GSS mode (gssencmode=disable in connection URL)
       - Any fork-unsafe C extensions
       - Reason: These libraries hold thread state that corrupts across fork()

       Library Detection at Startup:
       - Detects gRPC and greenlet loading (warnings, not errors)
       - Verifies GSS auth is disabled in database URL (must contain gssencmode=disable)
       - gRPC/greenlet warnings are OK when jobs are sync-only (no async/await)
       - Single worker + sync jobs + spawn() = safe execution

    3. Threading Model:
       - Parent worker uses thread pool (safe, single process)
       - Child work-horse processes spawned via spawn() (not fork())
       - GPU model preloaded in child context
       - GPU memory cleanup aggressive (gc.collect() frequent)

    On Linux:
       - Multiple workers allowed (fork is GPU-safe on Linux)
       - No single-worker enforcement
       - Same code runs but without macOS restrictions

    Uses EmbeddingPreloadWorker to ensure embedding model is preloaded
    in the work-horse subprocess context (not main worker process).
    """
    settings = get_settings()
    import multiprocessing

    # Platform-specific enforcement
    if platform.system() == "Darwin":  # macOS
        logger.info("=" * 70)
        logger.info("macOS detected - enforcing single worker for GPU fork-safety")
        logger.info("=" * 70)

        # Verify fork method
        fork_method = multiprocessing.get_start_method()
        if fork_method != "spawn":
            logger.error(
                f"DANGEROUS: Using '{fork_method}' as start method on macOS. "
                f"Should be 'spawn'. This may cause GPU crashes."
            )
        else:
            logger.info(f"✓ Multiprocessing start method: {fork_method} (correct for macOS)")

        if not _check_single_worker_on_macos():
            logger.error("Cannot start worker: another worker already running on macOS")
            logger.error("To kill the existing worker and restart:")
            lock_file = _get_worker_lock_file()
            if lock_file.exists():
                try:
                    existing_pid = lock_file.read_text().strip()
                    logger.error(f"  1. kill {existing_pid}")
                    logger.error(f"  2. rm {lock_file}")
                except Exception:
                    pass
            return

    logger.info("Starting RQ worker with priority queues")
    logger.info("Redis URL: %s", settings.redis_url)

    # Connect to Redis
    redis_conn = get_redis()

    # Create custom worker with all queues in priority order
    # Uses EmbeddingPreloadWorker to preload model in work-horse subprocess
    queue_names = [QUEUE_HIGH, QUEUE_NORMAL, QUEUE_LOW, QUEUE_DEFAULT]
    logger.info(f"Processing queues in order: {queue_names}")
    logger.info(f"Platform: {platform.system()}")

    # Detailed GPU and device logging
    logger.info("=" * 70)
    logger.info("GPU & Device Information")
    logger.info("=" * 70)
    try:
        from image_search_service.core.device import get_device_info, get_device

        device_info = get_device_info()
        device = get_device()

        logger.info(f"Selected device: {device}")
        logger.info(f"Platform: {device_info.get('platform')}")
        logger.info(f"Machine: {device_info.get('machine')}")
        logger.info(f"Python version: {device_info.get('python_version')}")
        logger.info(f"PyTorch version: {device_info.get('pytorch_version')}")

        if device_info.get("mps_available"):
            logger.info("MPS GPU available: True ✓")
            logger.info("MPS workaround: High watermark disabled (PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0)")
        else:
            logger.info("MPS GPU available: False")

        if device_info.get("cuda_available"):
            logger.info(f"CUDA available: True (Device: {device_info.get('cuda_device_name')})")
        else:
            logger.info("CUDA available: False")

    except Exception as e:
        logger.warning(f"Could not retrieve device information: {e}")

    # Monitor for fork-unsafe libraries
    logger.info("=" * 70)
    logger.info("Fork-Safety Checks")
    logger.info("=" * 70)
    try:
        import grpc
        logger.warning("⚠️ gRPC library loaded - monitor for threading issues in child processes")
    except ImportError:
        logger.info("✓ gRPC not loaded (good for fork safety)")

    try:
        import greenlet
        logger.warning("⚠️ Greenlet library loaded - verify no async/await in job functions")
    except ImportError:
        logger.info("✓ Greenlet not loaded")

    # Check for GSS authentication disabling on macOS (should use gssencmode=disable)
    if platform.system() == "Darwin":
        try:
            # Import the sync engine to check its database URL configuration
            from image_search_service.db.session import get_sync_engine

            # Get the sync engine (used for workers) and check its URL
            sync_engine = get_sync_engine()
            db_url = str(sync_engine.url)

            # Verify GSS auth is disabled in the connection URL
            if "gssencmode=disable" in db_url.lower():
                logger.info("✓ GSS authentication disabled on macOS (correct)")
            else:
                logger.error("⚠️ UNEXPECTED: GSS authentication not disabled in worker database URL!")
                logger.error(f"   Worker DB URL does not contain 'gssencmode=disable'")
        except Exception as e:
            logger.warning(f"Could not verify GSS auth status in database URL: {e}")
            logger.warning("   Assuming GSS auth is properly configured")
    logger.info("=" * 70)

    worker = EmbeddingPreloadWorker(queue_names, connection=redis_conn)

    try:
        worker.work()
    finally:
        _cleanup_worker_lock()


if __name__ == "__main__":
    main()
