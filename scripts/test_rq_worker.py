#!/usr/bin/env python
"""Test RQ worker with real GPU processing on macOS.

This standalone test script reproduces the RQ worker crash on macOS by:
1. Creating an isolated test environment (test Redis DB, test Qdrant collection)
2. Using a real sample image from the test fixtures
3. Enqueueing a single-image training job via RQ
4. Monitoring job execution with detailed DEBUG logging
5. Capturing crashes and GPU memory metrics

Usage:
    # Terminal 1: Start the worker (handles OBJC_DISABLE_INITIALIZE_FORK_SAFETY on macOS)
    # x86:
    make worker

    # macOS (the case we're debugging):
    OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES make worker

    # Terminal 2: Run this script to enqueue and monitor jobs
    uv run python scripts/test_rq_worker.py --mode client

Environment:
    REDIS_TEST_DB: Redis database index for test (default: 1)
    QDRANT_TEST_COLLECTION: Qdrant collection name (default: test_rq_worker_assets)
    WORKER_DEBUG: If set, enables DEBUG logging in worker
"""

import argparse
import logging
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from redis import Redis
from rq import Queue
from rq.job import Job, JobStatus
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# Add src to path so we can import the app
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from image_search_service.core.config import get_settings
from image_search_service.core.logging import configure_logging, get_logger
from image_search_service.db.models import (
    Base,
    ImageAsset,
    SessionStatus,
    TrainingJob,
    TrainingSession,
)
from image_search_service.db.models import (
    JobStatus as JobStatusEnum,
)
from image_search_service.queue.training_jobs import train_single_asset

# Configure logging at module level
configure_logging()
logger = get_logger(__name__)

# Test configuration
REDIS_TEST_DB = int(os.environ.get("REDIS_TEST_DB", "0"))  # Use same DB as worker (default is 0)
QDRANT_TEST_COLLECTION = os.environ.get("QDRANT_TEST_COLLECTION", "test_rq_worker_assets")
SAMPLE_IMAGE_PATH = Path(__file__).parent.parent / "tests" / "sample-images" / "Charlize Theron_30.jpg"


class TestEnvironment:
    """Isolated test environment for RQ worker testing."""

    def __init__(self):
        """Initialize test environment."""
        self.redis_conn: Redis | None = None
        self.db_session: Session | None = None
        self.db_engine: Any | None = None
        self.session_factory: Any | None = None
        self.test_training_session: TrainingSession | None = None
        self.test_asset: ImageAsset | None = None
        self.test_training_job: TrainingJob | None = None
        self.job_id: str | None = None

    def setup(self) -> None:
        """Set up test environment."""
        logger.info("=" * 70)
        logger.info("Test RQ Worker - Initialization")
        logger.info("=" * 70)

        # 1. Connect to Redis (test DB)
        self._setup_redis()

        # 2. Set up test database
        self._setup_database()

        # 3. Override Qdrant collection name
        self._override_qdrant_collection()

        # 4. Create test data
        self._create_test_data()

        logger.info("=" * 70)
        logger.info("Test environment ready")
        logger.info("=" * 70)

    def _setup_redis(self) -> None:
        """Connect to Redis test database."""
        try:
            redis_url = f"redis://localhost:6379/{REDIS_TEST_DB}"
            self.redis_conn = Redis.from_url(redis_url)
            # Test connection
            self.redis_conn.ping()
            logger.info(f"✓ Connected to Redis: {redis_url}")
        except Exception as e:
            logger.error(f"✗ Failed to connect to Redis: {e}")
            raise

    def _setup_database(self) -> None:
        """Set up SQLite test database."""
        try:
            db_url = "sqlite:///:memory:"
            self.db_engine = create_engine(db_url, echo=False)
            self.session_factory = sessionmaker(bind=self.db_engine)

            # Create all tables
            Base.metadata.create_all(self.db_engine)
            logger.info(f"✓ Created test database: {db_url}")

            # Create session
            self.db_session = self.session_factory()
        except Exception as e:
            logger.error(f"✗ Failed to set up database: {e}")
            raise

    def _override_qdrant_collection(self) -> None:
        """Override Qdrant collection name for testing."""
        # Force re-evaluation of settings with test collection
        os.environ["QDRANT_COLLECTION"] = QDRANT_TEST_COLLECTION
        get_settings.cache_clear()
        settings = get_settings()
        logger.info(f"✓ Using Qdrant collection: {settings.qdrant_collection}")

    def _create_test_data(self) -> None:
        """Create test training session, asset, and job."""
        if not self.db_session:
            raise RuntimeError("Database session not initialized")

        try:
            # Create training session
            now = datetime.now(UTC)
            self.test_training_session = TrainingSession(
                name=f"test_rq_worker_{now.timestamp()}",
                root_path=str(SAMPLE_IMAGE_PATH.parent),  # Use test image directory as root
                category_id=1,
                status=SessionStatus.PENDING.value,
                created_at=now,
            )
            self.db_session.add(self.test_training_session)
            self.db_session.flush()  # Get the ID

            logger.info(f"✓ Created training session: ID={self.test_training_session.id}")

            # Verify sample image exists
            if not SAMPLE_IMAGE_PATH.exists():
                raise FileNotFoundError(
                    f"Sample image not found: {SAMPLE_IMAGE_PATH}\n"
                    "Expected: tests/sample-images/Charlize Theron_30.jpg"
                )

            # Create image asset
            self.test_asset = ImageAsset(
                path=str(SAMPLE_IMAGE_PATH),
                created_at=now,
            )
            self.db_session.add(self.test_asset)
            self.db_session.flush()  # Get the ID

            logger.info(
                f"✓ Created image asset: ID={self.test_asset.id}, "
                f"path={SAMPLE_IMAGE_PATH.name}"
            )

            # Create training job
            self.test_training_job = TrainingJob(
                session_id=self.test_training_session.id,
                asset_id=self.test_asset.id,
                status=JobStatusEnum.PENDING.value,
                created_at=now,
            )
            self.db_session.add(self.test_training_job)
            self.db_session.commit()

            logger.info(f"✓ Created training job: ID={self.test_training_job.id}")

        except Exception as e:
            logger.error(f"✗ Failed to create test data: {e}")
            if self.db_session:
                self.db_session.rollback()
            raise

    def enqueue_job(self) -> str:
        """Enqueue the training job via RQ."""
        if not self.redis_conn or not self.test_training_job or not self.test_asset:
            raise RuntimeError("Test environment not properly initialized")

        try:
            queue = Queue("training-high", connection=self.redis_conn)

            # Enqueue train_single_asset job
            job = queue.enqueue(
                train_single_asset,
                self.test_training_job.id,
                self.test_asset.id,
                self.test_training_session.id,
                job_timeout=300,  # 5 minutes
            )

            self.job_id = job.id
            logger.info(f"✓ Enqueued job: {job.id}")
            logger.info(f"  Status: {job.get_status()}")
            logger.info("  Queue: training-high")

            return job.id

        except Exception as e:
            logger.error(f"✗ Failed to enqueue job: {e}")
            raise

    def monitor_job(self, timeout: int = 120) -> bool:
        """Monitor job execution with debug logging.

        Args:
            timeout: Maximum time to wait for job completion (seconds)

        Returns:
            True if job succeeded, False if failed or timed out
        """
        if not self.redis_conn or not self.job_id:
            raise RuntimeError("Job not enqueued")

        logger.info("=" * 70)
        logger.info("Monitoring Job Execution")
        logger.info("=" * 70)

        start_time = time.time()
        last_status = None
        gpu_memory_before = self._get_gpu_memory()

        logger.info(f"GPU Memory before job: {gpu_memory_before} MB")
        logger.info(f"Timeout: {timeout} seconds")
        logger.info("Polling job status (every 2 seconds)...")

        while time.time() - start_time < timeout:
            try:
                job = Job.fetch(self.job_id, connection=self.redis_conn)
                current_status = job.get_status()

                # Log status changes
                if current_status != last_status:
                    elapsed = int(time.time() - start_time)
                    logger.info(f"[{elapsed:03d}s] Job status: {current_status}")

                    if current_status == JobStatus.STARTED:
                        logger.info(f"      Worker PID: {job.worker_name}")
                    elif current_status == JobStatus.FINISHED:
                        self._log_job_success(job, start_time)
                        gpu_memory_after = self._get_gpu_memory()
                        logger.info(f"GPU Memory after job: {gpu_memory_after} MB")
                        logger.info(
                            f"GPU Memory delta: {gpu_memory_before - gpu_memory_after:+d} MB"
                        )
                        return True
                    elif current_status == JobStatus.FAILED:
                        self._log_job_failure(job, start_time)
                        return False

                    last_status = current_status

                time.sleep(2)

            except Exception as e:
                elapsed = int(time.time() - start_time)
                logger.error(f"[{elapsed:03d}s] Error fetching job: {e}")

        # Timeout
        elapsed = int(time.time() - start_time)
        logger.error(f"[{elapsed:03d}s] Job did not complete within {timeout} seconds")
        return False

    def _log_job_success(self, job: Job, start_time: float) -> None:
        """Log successful job completion."""
        elapsed = time.time() - start_time
        logger.info("=" * 70)
        logger.info("✓ JOB SUCCEEDED")
        logger.info("=" * 70)
        logger.info(f"Elapsed time: {elapsed:.2f}s")
        logger.info(f"Result: {job.result}")

    def _log_job_failure(self, job: Job, start_time: float) -> None:
        """Log job failure with debug info."""
        elapsed = time.time() - start_time
        logger.error("=" * 70)
        logger.error("✗ JOB FAILED")
        logger.error("=" * 70)
        logger.error(f"Elapsed time: {elapsed:.2f}s")
        logger.error(f"Status: {job.get_status()}")
        logger.error(f"Error message: {job.exc_info}")

        # Log worker info if available
        if hasattr(job, "worker_name") and job.worker_name:
            logger.error(f"Worker: {job.worker_name}")

    def _get_gpu_memory(self) -> int:
        """Get current GPU memory usage in MB."""
        try:
            if torch.cuda.is_available():
                return int(torch.cuda.memory_allocated() / 1024 / 1024)
            elif torch.backends.mps.is_available():
                # MPS doesn't expose memory usage easily, return 0
                return 0
        except Exception:
            pass
        return 0

    def cleanup(self) -> None:
        """Clean up test environment."""
        logger.info("=" * 70)
        logger.info("Cleaning up test environment")
        logger.info("=" * 70)

        # Close database session
        if self.db_session:
            try:
                self.db_session.close()
                logger.info("✓ Closed database session")
            except Exception as e:
                logger.warning(f"Warning closing database: {e}")

        # Clean Redis (optional - only delete our test job)
        if self.redis_conn and self.job_id:
            try:
                self.redis_conn.delete(f"rq:job:{self.job_id}")
                logger.info(f"✓ Cleaned up Redis job: {self.job_id}")
            except Exception as e:
                logger.warning(f"Warning cleaning Redis: {e}")


def print_environment_info() -> None:
    """Print debug information about the environment."""
    logger.info("=" * 70)
    logger.info("Environment Information")
    logger.info("=" * 70)

    # Python and platform
    logger.info(f"Platform: {platform.system()} {platform.release()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"Working directory: {os.getcwd()}")

    # PyTorch device info
    logger.info(f"PyTorch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()
    logger.info(f"CUDA available: {cuda_available}")
    logger.info(f"MPS available: {mps_available}")

    if cuda_available:
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.1f} GB")

    # Environment variables
    logger.info(f"Redis DB: {REDIS_TEST_DB}")
    logger.info(f"Qdrant collection: {QDRANT_TEST_COLLECTION}")
    logger.info(f"Sample image: {SAMPLE_IMAGE_PATH.name}")

    # RQ configuration
    logger.info(f"OBJC_DISABLE_INITIALIZE_FORK_SAFETY: {os.environ.get('OBJC_DISABLE_INITIALIZE_FORK_SAFETY', 'not set')}")
    logger.info(f"WORKER_DEBUG: {os.environ.get('WORKER_DEBUG', 'not set')}")


def run_client_mode() -> None:
    """Run test client: enqueue job and monitor."""
    print_environment_info()

    env = TestEnvironment()

    try:
        env.setup()
        job_id = env.enqueue_job()

        logger.info("")
        logger.info("Waiting for worker to process job...")
        logger.info("Make sure the worker is running in another terminal:")
        logger.info("  make worker  (or with OBJC_DISABLE_INITIALIZE_FORK_SAFETY on macOS)")
        logger.info("")

        success = env.monitor_job(timeout=120)

        logger.info("")
        logger.info("=" * 70)
        logger.info("Test Results")
        logger.info("=" * 70)
        logger.info(f"Job ID: {job_id}")
        logger.info(f"Result: {'✓ SUCCESS' if success else '✗ FAILED'}")
        logger.info("=" * 70)

        sys.exit(0 if success else 1)

    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        env.cleanup()


def run_worker_mode() -> None:
    """Run test worker: enhanced logging version of make worker."""
    print_environment_info()

    # Set up enhanced debug logging
    os.environ.setdefault("WORKER_DEBUG", "true")

    logger.info("Starting test RQ worker with enhanced logging...")
    logger.info("Processing from Redis DB: %d", REDIS_TEST_DB)

    try:
        from image_search_service.queue.worker import main
        main()
    except KeyboardInterrupt:
        logger.info("Worker stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Worker failed: {e}", exc_info=True)
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test RQ worker with real GPU processing on macOS"
    )
    parser.add_argument(
        "--mode",
        choices=["client", "worker"],
        default="client",
        help="Run mode: client (enqueue jobs) or worker (process jobs)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG logging",
    )

    args = parser.parse_args()

    # Set log level
    if args.debug:
        os.environ["LOG_LEVEL"] = "DEBUG"
        get_logger("image_search_service").setLevel(logging.DEBUG)

    if args.mode == "client":
        run_client_mode()
    elif args.mode == "worker":
        run_worker_mode()


if __name__ == "__main__":
    main()
