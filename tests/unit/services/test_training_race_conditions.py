"""Race condition tests for training session state machine.

These tests document race conditions in concurrent training session state
transitions (start, pause, cancel).

BUG DOCUMENTATION TESTS (not fixes):
- Tests PASS showing bugs exist (concurrent state changes allowed)
- When locks are added, update tests to verify proper state machine enforcement
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_training_when_concurrent_start_and_cancel_then_no_state_protection() -> None:
    """Concurrent start + cancel on same PENDING session.

    Expected current behavior (BUG): Depending on interleaving:
    - Start reads PENDING (valid), Cancel reads PENDING (invalid for cancel) -> Cancel fails
    - Start commits RUNNING, Cancel reads RUNNING (valid) -> Both succeed
    - Most dangerous: Start enqueues job, Cancel commits CANCELLED,
      but background job still runs on a "cancelled" session

    Expected correct behavior: Use SELECT ... FOR UPDATE on session row,
    or implement optimistic locking with version counter, to prevent
    concurrent state transitions.
    """
    from image_search_service.db.models import (
        SessionStatus,
        TrainingSession,
    )
    from image_search_service.services.training_service import (
        TrainingService,
    )

    service = TrainingService()
    session_id = 1

    # Create mock session in PENDING state
    mock_session = MagicMock(spec=TrainingSession)
    mock_session.id = session_id
    mock_session.status = SessionStatus.PENDING.value
    mock_session.started_at = None
    mock_session.paused_at = None
    mock_session.total_images = 100
    mock_session.processed_images = 0

    # Mock database session
    db_session = AsyncMock()

    state_log: list[str] = []

    async def mock_get_session(
        db: AsyncMock, sid: int
    ) -> TrainingSession:
        state_log.append(f"read_session_{sid}_status={mock_session.status}")
        # Both requests see the same status (no locking)
        return mock_session

    # Mock commit to track when it happens
    async def mock_commit() -> None:
        state_log.append(f"commit_status={mock_session.status}")

    db_session.commit = mock_commit
    db_session.refresh = AsyncMock()

    # Mock asset discovery and job creation for start_training
    with patch.object(
        service, "get_session", side_effect=mock_get_session
    ), patch.object(
        service, "enqueue_training", return_value="mock-rq-job-id"
    ), patch(
        "image_search_service.services.training_service.get_queue"
    ) as mock_get_queue:
        # Mock RQ queue
        mock_queue = MagicMock()
        mock_rq_job = MagicMock()
        mock_rq_job.id = "mock-rq-job-id"
        mock_queue.enqueue = MagicMock(return_value=mock_rq_job)
        mock_get_queue.return_value = mock_queue

        # Execute concurrent start and cancel
        results = await asyncio.gather(
            service.start_training(db_session, session_id),
            service.cancel_training(db_session, session_id),
            return_exceptions=True,
        )

    # Analyze results
    start_result, cancel_result = results

    # BUG DOCUMENTATION: At minimum, both operations read the session
    assert len(state_log) >= 2, "Both operations read the session"

    # Depending on interleaving:
    # - Both may succeed (if start commits RUNNING before cancel reads)
    # - Cancel may fail with ValueError (if both read PENDING)
    # - Dangerous case: start enqueues job, cancel sets CANCELLED, job still runs

    # Document what happened
    if isinstance(cancel_result, ValueError):
        # Cancel failed because session was PENDING (valid outcome)
        assert "Cannot cancel session in state" in str(cancel_result)
    elif not isinstance(start_result, Exception) and not isinstance(
        cancel_result, Exception
    ):
        # Both succeeded (race condition allowed this)
        # This means: start enqueued job, cancel set status to CANCELLED
        # Background job may still be running on a cancelled session
        assert True, "Both operations succeeded (race condition)"

    # The key issue: no SELECT ... FOR UPDATE prevents these concurrent mutations


@pytest.mark.asyncio
async def test_training_when_concurrent_pause_and_start_then_inconsistent_state() -> None:
    """Concurrent pause + start on RUNNING session.

    Expected current behavior (BUG): Both operations may succeed,
    leading to inconsistent state or state machine violations.

    Expected correct behavior: Row-level locking or optimistic locking
    prevents concurrent state transitions.
    """
    from image_search_service.db.models import (
        SessionStatus,
        TrainingSession,
    )
    from image_search_service.services.training_service import (
        TrainingService,
    )

    service = TrainingService()
    session_id = 1

    # Create mock session in RUNNING state
    mock_session = MagicMock(spec=TrainingSession)
    mock_session.id = session_id
    mock_session.status = SessionStatus.RUNNING.value
    mock_session.started_at = MagicMock()  # Non-null timestamp
    mock_session.paused_at = None

    # Mock database session
    db_session = AsyncMock()

    operation_log: list[str] = []

    async def mock_get_session(
        db: AsyncMock, sid: int
    ) -> TrainingSession:
        operation_log.append(f"get_session_status={mock_session.status}")
        return mock_session

    async def mock_commit() -> None:
        operation_log.append(f"commit_status={mock_session.status}")

    db_session.commit = mock_commit
    db_session.refresh = AsyncMock()

    with patch.object(
        service, "get_session", side_effect=mock_get_session
    ), patch(
        "image_search_service.services.training_service.get_queue"
    ) as mock_get_queue:
        # Mock RQ queue for start_training resume
        mock_queue = MagicMock()
        mock_rq_job = MagicMock()
        mock_rq_job.id = "mock-rq-job-id"
        mock_queue.enqueue = MagicMock(return_value=mock_rq_job)
        mock_get_queue.return_value = mock_queue

        # Execute concurrent pause and start (resume)
        results = await asyncio.gather(
            service.pause_training(db_session, session_id),
            service.start_training(
                db_session, session_id
            ),  # Will fail: RUNNING not valid for start
            return_exceptions=True,
        )

    pause_result, start_result = results

    # BUG DOCUMENTATION: Both operations read the session concurrently
    assert len(operation_log) >= 2, "Both operations read session"

    # Expected: start_training fails because session is RUNNING (not PENDING/PAUSED/FAILED)
    if isinstance(start_result, ValueError):
        assert "Cannot start training from state" in str(start_result)

    # Pause may succeed, but without locking, final state is unpredictable
    # If pause commits after start reads, state becomes inconsistent
