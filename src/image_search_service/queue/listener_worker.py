"""RQ Listener-based worker - processes jobs in main process with GPU initialized.

CROSS-PLATFORM SUPPORT
======================
Works on:
- macOS with Apple Silicon (MPS GPU)
- Linux with NVIDIA GPU (CUDA)
- Any system with CPU-only fallback

ARCHITECTURE
============
1. Initialize GPU once in main process (auto-detect platform)
2. Load embedding model once
3. Listen to Redis queues for job messages
4. Process jobs in-process (GPU available)
5. No subprocess spawning = No fork = No crashes ✅
"""

import gc
import os
import signal
import socket
import time
from datetime import UTC, datetime
from typing import Any

from redis import Redis
from rq import Queue
from rq.job import Job, JobStatus
from rq.registry import FailedJobRegistry, FinishedJobRegistry
from rq.worker import Worker as RQWorker

from image_search_service.core.config import get_settings
from image_search_service.core.device import get_device, get_device_info
from image_search_service.core.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)


class ListenerWorker:
    """RQ Listener-based worker processing jobs in main process with GPU initialized.

    Works on macOS (MPS), Linux (CUDA), and CPU-only systems.
    """

    def __init__(
        self,
        queues: list[str] | list[Queue],
        connection: Redis | None = None,
    ):
        """Initialize listener worker.

        Args:
            queues: List of queue names or Queue objects to listen to
            connection: Redis connection instance
        """
        self.queues = self._get_queue_objects(queues, connection)
        self.connection = connection or Redis()
        self.running = False
        self._gpu_initialized = False
        self._model_loaded = False

        # Worker identification for RQ registration
        self.worker_name = f"listener-{socket.gethostname()}-{os.getpid()}"
        self.pid = os.getpid()
        self.hostname = socket.gethostname()

        # Create RQWorker instance for registration only (not for job execution)
        # We still process jobs in-process, but this allows RQ monitoring tools
        # to see the worker via Worker.all()
        self._rq_worker = RQWorker(
            queues=self.queues,
            connection=self.connection,
            name=self.worker_name,
        )

        # Track current job for monitoring
        self._current_job: Job | None = None
        self._is_busy = False

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

    def _get_queue_objects(
        self,
        queues: list[str] | list[Queue],
        connection: Redis | None = None,
    ) -> list[Queue]:
        """Convert queue names or objects to Queue instances."""
        queue_objects = []
        conn = connection or Redis()

        for q in queues:
            if isinstance(q, str):
                queue_objects.append(Queue(q, connection=conn))
            elif isinstance(q, Queue):
                queue_objects.append(q)
            else:
                raise ValueError(f"Invalid queue type: {type(q)}")

        return queue_objects

    def _initialize_gpu(self) -> None:
        """Initialize GPU in main process (called once at startup).

        Auto-detects platform and initializes appropriate GPU:
        - macOS: Metal Performance Shaders (MPS)
        - Linux: NVIDIA CUDA
        - Other: CPU fallback
        """
        if self._gpu_initialized:
            logger.debug("GPU already initialized, skipping")
            return

        logger.info("=" * 70)
        logger.info("Initializing GPU in Main Process")
        logger.info("=" * 70)

        try:
            device = get_device()
            device_info = get_device_info()

            logger.info(f"Device: {device}")
            logger.info(f"Platform: {device_info.get('platform')}")
            logger.info(f"Machine: {device_info.get('machine')}")
            logger.info(f"PyTorch version: {device_info.get('pytorch_version')}")

            if device_info.get("mps_available"):
                logger.info("✓ Apple Silicon GPU (MPS) available")
                logger.info("  Metal Performance Shaders will be used")
                logger.info("  GPU is now initialized in main process")
            elif device_info.get("cuda_available"):
                logger.info("✓ NVIDIA CUDA GPU available")
                logger.info(f"  Device: {device_info.get('cuda_device_name')}")
                logger.info("  GPU is now initialized in main process")
            else:
                logger.warning("⚠️ No GPU available (falling back to CPU)")
                logger.info("  Using CPU for tensor operations (slower but works)")

            self._gpu_initialized = True
            logger.info("✓ GPU initialization complete")

        except Exception as e:
            logger.error(f"Failed to initialize GPU: {e}")
            self._gpu_initialized = False
            raise

    def _preload_embedding_model(self) -> None:
        """Preload embedding model in main process (called once at startup).

        Must happen after GPU initialization. Model stays loaded for all jobs.
        Works identically on all platforms (uses whatever GPU was initialized).
        """
        if self._model_loaded:
            logger.debug("Embedding model already loaded, skipping")
            return

        logger.info("=" * 70)
        logger.info("Preloading Embedding Model")
        logger.info("=" * 70)

        try:
            from image_search_service.services.embedding import (
                preload_embedding_model,
            )

            logger.debug("Loading embedding model in main process...")
            preload_embedding_model()

            self._model_loaded = True
            logger.info("✓ Embedding model preloaded successfully")
            logger.info("  Model is now available for all jobs in main process")

        except Exception as e:
            logger.error(f"Failed to preload embedding model: {e}")
            self._model_loaded = False
            raise

    def _register_birth(self) -> None:
        """Register worker birth with Redis for RQ monitoring visibility.

        Makes this worker visible to Worker.all() and monitoring tools.
        Non-fatal if registration fails (worker can still process jobs).
        """
        try:
            self._rq_worker.register_birth()  # type: ignore[no-untyped-call]
            logger.info(f"✓ Worker registered: {self.worker_name}")
            logger.debug(f"  PID: {self.pid}, Hostname: {self.hostname}")
        except Exception as e:
            logger.warning(f"Failed to register worker birth: {e}")

    def _register_death(self) -> None:
        """Unregister worker from Redis on shutdown.

        Removes worker from Worker.all() listings.
        Non-fatal if unregistration fails.
        """
        try:
            self._rq_worker.register_death()  # type: ignore[no-untyped-call]
            logger.info(f"✓ Worker unregistered: {self.worker_name}")
        except Exception as e:
            logger.warning(f"Failed to register worker death: {e}")

    def _heartbeat(self) -> None:
        """Send heartbeat to Redis to update last_heartbeat timestamp.

        Allows monitoring tools to detect stale/dead workers.
        Non-fatal if heartbeat fails.
        """
        try:
            self._rq_worker.heartbeat()
            logger.debug(f"Heartbeat sent for worker {self.worker_name}")
        except Exception as e:
            logger.debug(f"Heartbeat failed: {e}")

    def _set_state(self, state: str) -> None:
        """Update worker state in Redis (idle or busy).

        Args:
            state: Worker state ('idle' or 'busy')

        Raises:
            ValueError: If state is not 'idle' or 'busy'
        """
        if state not in ("idle", "busy"):
            raise ValueError(f"Invalid worker state: {state} (expected 'idle' or 'busy')")

        try:
            self._rq_worker.set_state(state)
            self._is_busy = state == "busy"
            logger.debug(f"Worker state changed to: {state}")
        except Exception as e:
            logger.warning(f"Failed to set worker state to {state}: {e}")

    def _set_current_job(self, job: Job | None) -> None:
        """Track current job for monitoring visibility.

        Args:
            job: Current job being processed (or None if idle)
        """
        self._current_job = job
        try:
            job_id = job.id if job else None
            self._rq_worker.set_current_job_id(job_id)
            if job:
                logger.debug(f"Current job set to: {job.id}")
            else:
                logger.debug("Current job cleared")
        except Exception as e:
            logger.warning(f"Failed to set current job: {e}")

    def _cleanup_job_from_registries(self, job: Job) -> None:
        """Remove job from all registries on worker shutdown.

        Idempotent and non-fatal. In RQ 2.x, StartedJobRegistry no longer
        supports add/remove — no-op for registry cleanup on interrupted jobs.

        Args:
            job: Job to clean up
        """
        logger.debug(f"Cleanup called for interrupted job {job.id} (no registry ops in RQ 2.x)")

    def _startup_checks(self) -> None:
        """Perform startup checks (device, libraries, etc)."""
        logger.info("=" * 70)
        logger.info("Startup Checks")
        logger.info("=" * 70)

        import platform

        logger.info(f"Platform: {platform.system()}")
        logger.info(f"Processor: {platform.machine()}")

        # Check multiprocessing start method (macOS specific)
        if platform.system() == "Darwin":
            import multiprocessing

            fork_method = multiprocessing.get_start_method()
            logger.info(f"Multiprocessing start method: {fork_method}")

            if fork_method != "spawn":
                logger.warning(f"⚠️ Using '{fork_method}' (expected 'spawn' on macOS)")

        # Check for fork-unsafe libraries (informational)
        try:
            import grpc  # noqa: F401  # type: ignore[import-untyped]

            logger.warning("⚠️ gRPC library loaded (monitor for threading issues)")
        except ImportError:
            logger.info("✓ gRPC not loaded")

        try:
            import greenlet  # noqa: F401  # type: ignore[import-untyped]

            logger.warning("⚠️ Greenlet library loaded (verify no async/await in jobs)")
        except ImportError:
            logger.info("✓ Greenlet not loaded")

        logger.info("=" * 70)

    def _get_next_job(self) -> Job | None:
        """Get next job from queues in priority order.

        Checks queues in order, returns first available job.
        """
        # Use dequeue_any to poll multiple queues
        # This returns (job, queue) tuple or None
        try:
            result = Queue.dequeue_any(self.queues, timeout=None, connection=self.connection)
            if result is None:
                return None
            job, _ = result
            return job
        except Exception as e:
            logger.warning(f"Error dequeuing from queues: {e}")
            return None

    def process_job(self, job: Job) -> bool:
        """Process a single job in main process (GPU available).

        This is the core advantage over RQ Worker - no subprocess spawning.
        Job runs directly in this process where GPU is already initialized.

        Args:
            job: RQ Job to process

        Returns:
            True if successful, False if failed
        """
        logger.info(f"Processing job {job.id} ({job.func_name})")

        # Update worker state to busy and track current job
        self._set_state("busy")
        self._set_current_job(job)

        # Get queue name for registry management
        queue_name = job.origin or "default"

        try:
            job.set_status(JobStatus.STARTED)
            job.save()

            start_time = time.time()

            # Execute the job function in main process
            # GPU is already initialized, works with MPS/CUDA/CPU transparently
            result = job.func(*job.args, **job.kwargs)

            elapsed = time.time() - start_time

            # Save result and mark as finished
            job._result = result
            job.set_status(JobStatus.FINISHED)
            job.ended_at = datetime.now(UTC)
            job.save()

            # Add job to FinishedJobRegistry
            # Note: StartedJobRegistry.add/remove raises NotImplementedError in RQ 2.x
            try:
                finished_registry = FinishedJobRegistry(queue_name, connection=self.connection)
                finished_registry.add(job, -1)
            except Exception as e:
                logger.warning(f"Failed to add job to FinishedJobRegistry: {e}")

            logger.info(f"✓ Job {job.id} completed successfully ({elapsed:.1f}s)")
            gc.collect()

            # Clear current job and return to idle
            self._set_current_job(None)
            self._set_state("idle")
            return True

        except Exception as e:
            elapsed = time.time() - start_time

            # Save exception info and mark as failed
            job._exc_info = str(e)
            job.set_status(JobStatus.FAILED)
            job.ended_at = datetime.now(UTC)
            job.save()

            # Add job to FailedJobRegistry
            # Note: StartedJobRegistry.add/remove raises NotImplementedError in RQ 2.x
            try:
                failed_registry = FailedJobRegistry(queue_name, connection=self.connection)
                failed_registry.add(job, -1)
            except Exception as reg_error:
                logger.warning(f"Failed to add job to FailedJobRegistry: {reg_error}")

            logger.error(
                f"✗ Job {job.id} failed after {elapsed:.1f}s: {e}",
                exc_info=True,
            )

            gc.collect()

            # Clear current job and return to idle
            self._set_current_job(None)
            self._set_state("idle")
            return False

    def _handle_shutdown(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals (SIGTERM, SIGINT)."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False

    def listen(self) -> None:
        """Main listener loop - listen to queues and process jobs in-process.

        KEY DIFFERENCES FROM RQ WORKER:
        - RQ Worker: Spawns work-horse per job (causes fork)
        - RQ Listener: Processes in main process (no fork)
        - GPU: Stays initialized throughout (not re-initialized per job)
        """
        self.running = True

        logger.info("=" * 70)
        logger.info("RQ Listener Worker Starting")
        logger.info("=" * 70)

        self._startup_checks()
        logger.info("")
        self._initialize_gpu()
        logger.info("")
        self._preload_embedding_model()
        logger.info("")

        # Register worker with Redis for monitoring visibility
        self._register_birth()
        self._set_state("idle")

        queue_names = [q.name for q in self.queues]
        logger.info("")
        logger.info(f"Listening to queues: {queue_names}")
        logger.info("Processing jobs in main process (no fork/spawn)")
        logger.info("")

        # Heartbeat tracking
        last_heartbeat = time.time()
        heartbeat_interval = 30.0  # Send heartbeat every 30 seconds

        try:
            while self.running:
                # Send periodic heartbeat to Redis
                now = time.time()
                if now - last_heartbeat >= heartbeat_interval:
                    self._heartbeat()
                    last_heartbeat = now

                job = self._get_next_job()

                if job:
                    self.process_job(job)
                else:
                    time.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            # Cleanup: ensure worker state is idle and current job is cleared
            logger.info("RQ Listener Worker shutting down")
            self._set_state("idle")

            # Cleanup interrupted job from registries if any
            if self._current_job is not None:
                self._cleanup_job_from_registries(self._current_job)
                self._set_current_job(None)

            # Unregister worker from Redis
            self._register_death()


def main() -> None:
    """Entry point for RQ Listener-based worker.

    Direct replacement for RQ Worker.
    Works on macOS (MPS), Linux (CUDA), and CPU fallback.
    """
    settings = get_settings()

    logger.info(f"Connecting to Redis: {settings.redis_url}")
    redis_conn = Redis.from_url(settings.redis_url)

    # Queue names in priority order
    queue_names = [
        "training-high",  # User-initiated training
        "training-normal",  # Scheduled tasks
        "training-low",  # Thumbnails, cleanup
        "default",  # Legacy support
    ]

    worker = ListenerWorker(queue_names, connection=redis_conn)
    worker.listen()


if __name__ == "__main__":
    main()
