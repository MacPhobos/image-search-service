"""Integration tests for RQ Listener-based worker."""

import gc
import os
import time
from pathlib import Path

import pytest
import torch
from redis import Redis
from rq import Queue

from image_search_service.queue.listener_worker import ListenerWorker


# Module-level job functions (needed for RQ serialization)
# Note: Must not start with 'test_' to avoid pytest collection
def job_returning_pid():
    """Simple job that returns current process ID."""
    import os

    return os.getpid()


def job_with_delay(job_id):
    """Job that simulates work with sleep."""
    time.sleep(0.1)
    return job_id


def job_that_fails():
    """Job that raises an exception."""
    raise ValueError("Test error")


class TestListenerWorker:
    """Test RQ Listener-based worker functionality."""

    @pytest.fixture
    def redis_conn(self):
        """Get Redis connection."""
        return Redis()

    @pytest.fixture
    def worker(self, redis_conn):
        """Create listener worker."""
        return ListenerWorker(["test-queue"], connection=redis_conn)

    def test_worker_initialization(self, worker):
        """Test worker initializes correctly."""
        assert worker.queues is not None
        assert len(worker.queues) == 1
        assert worker.connection is not None

    def test_gpu_initialization(self, worker):
        """Test GPU initializes once in main process."""
        assert not worker._gpu_initialized

        worker._initialize_gpu()

        assert worker._gpu_initialized

        # Calling again should skip (idempotent)
        worker._initialize_gpu()
        assert worker._gpu_initialized

    def test_model_preload(self, worker):
        """Test embedding model preloads in main process."""
        assert not worker._model_loaded

        # Must initialize GPU first
        worker._initialize_gpu()

        # Now preload model
        worker._preload_embedding_model()

        assert worker._model_loaded

    def test_no_subprocess_spawning(self, worker):
        """Test that listener doesn't spawn subprocesses."""
        parent_pid = os.getpid()

        # Initialize (no subprocess spawning)
        worker._initialize_gpu()
        worker._preload_embedding_model()

        # Verify still in same process
        assert os.getpid() == parent_pid

    def test_gpu_operations_dont_fork(self, redis_conn):
        """Test that GPU operations don't cause fork crashes.

        CRITICAL TEST: Verifies no Signal 11 crashes from GPU operations.
        """
        # Get baseline crash dumps
        crash_dir = Path("/Users/mac/Library/Logs/DiagnosticReports")
        initial_crashes = {c.name for c in crash_dir.glob("*.ips")} if crash_dir.exists() else set()

        # Create worker and initialize
        worker = ListenerWorker(["test-queue"], connection=redis_conn)
        worker._initialize_gpu()
        worker._preload_embedding_model()

        # Execute GPU operations in main process
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        for i in range(5):
            # Create tensor on GPU
            x = torch.randn(10, 10, device=device)
            y = torch.randn(10, 10, device=device)
            z = x @ y
            del x, y, z

        gc.collect()

        # Check for new crash dumps
        if crash_dir.exists():
            current_crashes = {c.name for c in crash_dir.glob("*.ips")}
            new_crashes = current_crashes - initial_crashes
        else:
            new_crashes = set()

        # CRITICAL ASSERTION
        assert len(new_crashes) == 0, (
            f"GPU operations caused {len(new_crashes)} new crash(es)! Signal 11 still occurring."
        )

    def test_job_processing_in_main_process(self, redis_conn):
        """Test job processes in main process (not subprocess)."""
        parent_pid = os.getpid()

        # Enqueue job (use module-level function)
        queue = Queue("test-queue", connection=redis_conn)
        job = queue.enqueue(job_returning_pid)

        # Process job using listener
        worker = ListenerWorker(["test-queue"], connection=redis_conn)
        worker._initialize_gpu()
        worker.process_job(job)

        # Verify job ran in main process (same PID)
        job.refresh()  # Reload from Redis
        assert job.get_status() == "finished"
        assert job.result == parent_pid  # Job ran in parent process!

    def test_sequential_job_processing(self, redis_conn):
        """Test multiple jobs process sequentially."""
        # Enqueue multiple jobs (use module-level function)
        queue = Queue("test-queue", connection=redis_conn)
        job_ids = []

        for i in range(3):
            job = queue.enqueue(job_with_delay, i)
            job_ids.append(job)

        # Process all jobs
        worker = ListenerWorker(["test-queue"], connection=redis_conn)
        worker._initialize_gpu()

        start = time.time()
        for job in job_ids:
            worker.process_job(job)
        elapsed = time.time() - start

        # Verify all jobs completed
        for job in job_ids:
            job.refresh()  # Reload from Redis
            assert job.get_status() == "finished"

        # Jobs processed sequentially (should take ~0.3s minimum)
        assert elapsed >= 0.25

    def test_error_handling_in_job(self, redis_conn):
        """Test listener handles job errors gracefully."""
        # Enqueue failing job (use module-level function)
        queue = Queue("test-queue", connection=redis_conn)
        job = queue.enqueue(job_that_fails)

        # Process job - should handle error gracefully
        worker = ListenerWorker(["test-queue"], connection=redis_conn)
        worker._initialize_gpu()

        result = worker.process_job(job)

        # Should return False for failed job
        assert result is False

        # Job status should be 'failed'
        job.refresh()  # Reload from Redis
        assert job.get_status() == "failed"

        # Job should have exception info stored
        # Note: RQ stores exception info in _exc_info private attribute
        assert hasattr(job, "_exc_info")
        assert job._exc_info is not None
        assert "Test error" in str(job._exc_info)

    def test_graceful_shutdown(self, worker):
        """Test graceful shutdown handling."""
        import signal

        # Set running state
        worker.running = True

        # Simulate SIGTERM
        worker._handle_shutdown(signal.SIGTERM, None)

        # Should stop running
        assert worker.running is False

    def test_queue_priority_ordering(self, redis_conn):
        """Test queues are processed in priority order."""
        # Create worker with queue order
        worker = ListenerWorker(
            ["training-high", "training-normal"],
            connection=redis_conn,
        )

        # Queue should respect order
        assert worker.queues[0].name == "training-high"
        assert worker.queues[1].name == "training-normal"
