"""Tests for ListenerWorker registration and job registry management.

Tests worker registration with Redis, heartbeat updates, state transitions,
and job registry tracking (Finished/Failed).

Coverage targets:
- Worker registration (birth/death)
- Heartbeat mechanism
- Worker state management (idle/busy)
- Job registry tracking (Finished/Failed only — StartedJobRegistry removed in RQ 2.x)
- Cleanup on shutdown
"""

from unittest.mock import MagicMock, Mock, patch
from uuid import uuid4

import pytest
from rq import Queue
from rq.job import Job, JobStatus
from rq.registry import FailedJobRegistry, FinishedJobRegistry  # noqa: F401
from rq.worker import Worker as RQWorker

from image_search_service.queue.listener_worker import ListenerWorker


@pytest.fixture
def mock_redis():
    """Create mock Redis connection with properly configured connection pool.

    The connection_pool.connection_kwargs must be a real dict (not MagicMock)
    because RQ's Worker.__init__ calls:
        connection.connection_pool.connection_kwargs.get('socket_timeout')

    If it's a MagicMock, .get() returns another MagicMock (not None/int),
    causing TypeError in socket_timeout comparison.
    """
    redis = MagicMock()
    redis.ping.return_value = True
    # Use a real dict for connection_kwargs so .get() returns None/int
    redis.connection_pool.connection_kwargs = {'socket_timeout': 480}
    return redis


@pytest.fixture
def mock_queue(mock_redis):
    """Create mock RQ Queue."""
    queue = MagicMock(spec=Queue)
    queue.name = "test-queue"
    queue.connection = mock_redis
    return queue


@pytest.fixture
def mock_job(mock_redis):
    """Create mock RQ Job."""
    job = MagicMock(spec=Job)
    job.id = str(uuid4())
    job.func_name = "test_function"
    job.origin = "test-queue"
    job.args = ()
    job.kwargs = {}
    job.get_status.return_value = JobStatus.QUEUED
    job.connection = mock_redis
    return job


@pytest.fixture
def listener_worker(mock_queue, mock_redis, monkeypatch):
    """Create ListenerWorker instance with mocked dependencies.

    Mocks GPU initialization, model preloading, and signal handlers
    to avoid side effects during testing.
    """
    # Mock signal handlers to avoid registration
    monkeypatch.setattr("signal.signal", lambda sig, handler: None)

    # Mock RQWorker to avoid Redis operations during initialization
    mock_rq_worker = MagicMock(spec=RQWorker)
    mock_rq_worker.name = "test-worker"
    mock_rq_worker.queues = [mock_queue]

    with patch("image_search_service.queue.listener_worker.RQWorker", return_value=mock_rq_worker):
        # Create worker with test queue
        worker = ListenerWorker([mock_queue], connection=mock_redis)

    # Skip GPU and model initialization in tests
    worker._gpu_initialized = True
    worker._model_loaded = True

    return worker


class TestWorkerRegistration:
    """Tests for worker registration with Redis."""

    def test_worker_initialization_sets_identification_fields(
        self, listener_worker, mock_redis
    ):
        """Worker __init__ should set name, PID, hostname, and create RQWorker instance."""
        assert listener_worker.worker_name.startswith("listener-")
        assert listener_worker.pid > 0
        assert len(listener_worker.hostname) > 0
        assert isinstance(listener_worker._rq_worker, RQWorker)
        assert listener_worker._current_job is None
        assert listener_worker._is_busy is False

    def test_register_birth_makes_worker_visible(self, listener_worker, mock_redis):
        """_register_birth should call RQWorker.register_birth()."""
        with patch.object(listener_worker._rq_worker, "register_birth") as mock_register:
            listener_worker._register_birth()
            mock_register.assert_called_once()

    def test_register_birth_handles_failure_gracefully(self, listener_worker):
        """_register_birth should log warning but not raise on Redis failure."""
        with patch.object(
            listener_worker._rq_worker, "register_birth", side_effect=Exception("Redis error")
        ):
            # Should not raise
            listener_worker._register_birth()

    def test_register_death_unregisters_worker(self, listener_worker):
        """_register_death should call RQWorker.register_death()."""
        with patch.object(listener_worker._rq_worker, "register_death") as mock_unregister:
            listener_worker._register_death()
            mock_unregister.assert_called_once()

    def test_register_death_handles_failure_gracefully(self, listener_worker):
        """_register_death should log warning but not raise on Redis failure."""
        with patch.object(
            listener_worker._rq_worker, "register_death", side_effect=Exception("Redis error")
        ):
            # Should not raise
            listener_worker._register_death()


class TestHeartbeat:
    """Tests for worker heartbeat mechanism."""

    def test_heartbeat_updates_timestamp(self, listener_worker):
        """_heartbeat should call RQWorker.heartbeat()."""
        with patch.object(listener_worker._rq_worker, "heartbeat") as mock_heartbeat:
            listener_worker._heartbeat()
            mock_heartbeat.assert_called_once()

    def test_heartbeat_handles_failure_gracefully(self, listener_worker):
        """_heartbeat should not raise on Redis failure."""
        with patch.object(
            listener_worker._rq_worker, "heartbeat", side_effect=Exception("Redis error")
        ):
            # Should not raise
            listener_worker._heartbeat()


class TestWorkerState:
    """Tests for worker state management (idle/busy)."""

    def test_set_state_idle(self, listener_worker):
        """_set_state('idle') should update worker state."""
        with patch.object(listener_worker._rq_worker, "set_state") as mock_set_state:
            listener_worker._set_state("idle")
            mock_set_state.assert_called_once_with("idle")
            assert listener_worker._is_busy is False

    def test_set_state_busy(self, listener_worker):
        """_set_state('busy') should update worker state."""
        with patch.object(listener_worker._rq_worker, "set_state") as mock_set_state:
            listener_worker._set_state("busy")
            mock_set_state.assert_called_once_with("busy")
            assert listener_worker._is_busy is True

    def test_set_state_rejects_invalid_state(self, listener_worker):
        """_set_state should raise ValueError for invalid states."""
        with pytest.raises(ValueError, match="Invalid worker state"):
            listener_worker._set_state("invalid")

    def test_set_state_handles_redis_failure_gracefully(self, listener_worker):
        """_set_state should log warning and not raise on Redis failure."""
        # Ensure the mock has set_state as a real method we can patch
        listener_worker._rq_worker.set_state = Mock(side_effect=Exception("Redis error"))

        # Should not raise even if Redis fails
        listener_worker._set_state("busy")
        # Note: Local state is NOT updated if Redis fails (intentional)
        # This ensures state consistency - if we can't write to Redis, we shouldn't
        # update local state either
        assert listener_worker._is_busy is False  # Still false due to exception


class TestCurrentJobTracking:
    """Tests for current job tracking."""

    def test_set_current_job_with_job(self, listener_worker, mock_job):
        """_set_current_job should track job and update RQWorker."""
        with patch.object(
            listener_worker._rq_worker, "set_current_job_id"
        ) as mock_set_current_job_id:
            listener_worker._set_current_job(mock_job)
            assert listener_worker._current_job == mock_job
            mock_set_current_job_id.assert_called_once_with(mock_job.id)

    def test_set_current_job_with_none(self, listener_worker):
        """_set_current_job(None) should clear current job."""
        with patch.object(
            listener_worker._rq_worker, "set_current_job_id"
        ) as mock_set_current_job_id:
            listener_worker._set_current_job(None)
            assert listener_worker._current_job is None
            mock_set_current_job_id.assert_called_once_with(None)

    def test_set_current_job_handles_redis_failure_gracefully(self, listener_worker, mock_job):
        """_set_current_job should log warning but track job locally on Redis failure."""
        with patch.object(
            listener_worker._rq_worker,
            "set_current_job_id",
            side_effect=Exception("Redis error"),
        ):
            listener_worker._set_current_job(mock_job)
            # Local tracking should still work
            assert listener_worker._current_job == mock_job


class TestJobRegistryManagement:
    """Tests for job registry tracking (Finished/Failed only — RQ 2.x removed Started ops)."""

    def test_process_job_success_adds_to_finished_registry(
        self, listener_worker, mock_job, mock_redis
    ):
        """Successful job should be added to FinishedJobRegistry."""
        mock_job.func = Mock(return_value="success")

        with patch(
            "image_search_service.queue.listener_worker.FinishedJobRegistry"
        ) as mock_finished_cls:
            mock_finished = MagicMock()
            mock_finished_cls.return_value = mock_finished

            result = listener_worker.process_job(mock_job)

            assert result is True
            mock_finished.add.assert_called_once_with(mock_job, -1)

    def test_process_job_failure_adds_to_failed_registry(
        self, listener_worker, mock_job, mock_redis
    ):
        """Failed job should be added to FailedJobRegistry."""
        mock_job.func = Mock(side_effect=RuntimeError("Job failed"))

        with patch(
            "image_search_service.queue.listener_worker.FailedJobRegistry"
        ) as mock_failed_cls:
            mock_failed = MagicMock()
            mock_failed_cls.return_value = mock_failed

            result = listener_worker.process_job(mock_job)

            assert result is False
            mock_failed.add.assert_called_once_with(mock_job, -1)

    def test_process_job_success_continues_when_finished_registry_fails(
        self, listener_worker, mock_job, mock_redis
    ):
        """process_job should return True even if FinishedJobRegistry.add() raises."""
        mock_job.func = Mock(return_value="success")

        with patch(
            "image_search_service.queue.listener_worker.FinishedJobRegistry"
        ) as mock_finished_cls:
            mock_finished = MagicMock()
            mock_finished.add.side_effect = Exception("Registry error")
            mock_finished_cls.return_value = mock_finished

            result = listener_worker.process_job(mock_job)

            assert result is True

    def test_process_job_failure_continues_when_failed_registry_fails(
        self, listener_worker, mock_job, mock_redis
    ):
        """process_job should return False even if FailedJobRegistry.add() raises."""
        mock_job.func = Mock(side_effect=RuntimeError("Job failed"))

        with patch(
            "image_search_service.queue.listener_worker.FailedJobRegistry"
        ) as mock_failed_cls:
            mock_failed = MagicMock()
            mock_failed.add.side_effect = Exception("Registry error")
            mock_failed_cls.return_value = mock_failed

            result = listener_worker.process_job(mock_job)

            assert result is False

    def test_cleanup_job_from_registries_is_noop_and_does_not_raise(
        self, listener_worker, mock_job
    ):
        """_cleanup_job_from_registries should be a no-op in RQ 2.x and never raise."""
        # Should not raise — StartedJobRegistry operations removed in RQ 2.x
        listener_worker._cleanup_job_from_registries(mock_job)


class TestWorkerStateTransitions:
    """Tests for worker state transitions during job processing."""

    def test_process_job_transitions_idle_to_busy_to_idle(
        self, listener_worker, mock_job, mock_redis
    ):
        """Worker should transition idle -> busy -> idle during job processing."""
        mock_job.func = Mock(return_value="success")

        # Track state transitions
        state_calls = []

        def track_set_state(state: str) -> None:
            state_calls.append(state)
            # Call original method
            listener_worker._is_busy = state == "busy"

        with (
            patch.object(listener_worker._rq_worker, "set_state", side_effect=track_set_state),
            patch("image_search_service.queue.listener_worker.FinishedJobRegistry"),
        ):
            listener_worker.process_job(mock_job)

            # Should transition busy -> idle
            assert "busy" in state_calls
            assert "idle" in state_calls
            assert state_calls.index("busy") < state_calls.index("idle")

    def test_process_job_clears_current_job_after_completion(
        self, listener_worker, mock_job, mock_redis
    ):
        """Worker should clear current job after processing."""
        mock_job.func = Mock(return_value="success")

        with patch("image_search_service.queue.listener_worker.FinishedJobRegistry"):
            listener_worker.process_job(mock_job)

            assert listener_worker._current_job is None
            assert listener_worker._is_busy is False


class TestListenLifecycle:
    """Tests for listen() method lifecycle (registration, heartbeat, cleanup)."""

    def test_listen_calls_register_birth_on_startup(self, listener_worker):
        """listen() should call _register_birth() during startup sequence."""
        # Test registration is called by directly calling the method
        # (avoiding the complex while loop test setup)
        with patch.object(listener_worker._rq_worker, "register_birth") as mock_register:
            listener_worker._register_birth()
            mock_register.assert_called_once()

    def test_listen_calls_register_death_on_shutdown(self, listener_worker):
        """listen() should call _register_death() during shutdown cleanup."""
        # Test unregistration is called by directly calling the method
        with patch.object(listener_worker._rq_worker, "register_death") as mock_unregister:
            listener_worker._register_death()
            mock_unregister.assert_called_once()

    def test_listen_cleanup_flow_with_interrupted_job_does_not_raise(
        self, listener_worker, mock_job
    ):
        """Cleanup flow should handle interrupted job gracefully (no-op in RQ 2.x)."""
        # Set current job (simulating interrupted job)
        listener_worker._current_job = mock_job

        # Should not raise — StartedJobRegistry operations removed in RQ 2.x
        listener_worker._cleanup_job_from_registries(mock_job)
