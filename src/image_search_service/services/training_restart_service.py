"""Training restart service for restarting failed or completed training sessions."""

import time
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.core.logging import get_logger
from image_search_service.db.models import JobStatus, SessionStatus, TrainingJob, TrainingSession
from image_search_service.queue.worker import QUEUE_HIGH, get_queue
from image_search_service.services.restart_service_base import CleanupStats, RestartServiceBase

logger = get_logger(__name__)


class TrainingRestartService(RestartServiceBase):
    """Restart training session (Phase 1: CLIP embeddings).

    Options:
        failed_only: If True, reset only FAILED jobs to PENDING.
                    If False, reset ALL jobs to PENDING (full restart).

    Safety features:
    - Validates state before restart (no restart if RUNNING)
    - Idempotent: safe to call multiple times
    - Uses existing start_training workflow (no new code path)
    - Snapshot/rollback mechanism for safety
    - Audit logging with reset_at and reset_reason
    """

    def __init__(self, failed_only: bool = True):
        """Initialize training restart service.

        Args:
            failed_only: Reset only failed jobs (True) or all jobs (False)
        """
        self.failed_only = failed_only

    async def validate_state(
        self,
        db: AsyncSession,
        session_id: int,
    ) -> None:
        """Validate training session can be restarted.

        Allowed states: COMPLETED, FAILED, CANCELLED
        Forbidden states: RUNNING, PENDING

        Args:
            db: Database session
            session_id: Training session ID

        Raises:
            ValueError: If session not found or in invalid state
            RuntimeError: If jobs still active
        """
        # Get session
        result = await db.execute(
            select(TrainingSession).where(TrainingSession.id == session_id)
        )
        session = result.scalar_one_or_none()

        if not session:
            raise ValueError(f"Training session {session_id} not found")

        # Check status
        if session.status == SessionStatus.RUNNING.value:
            raise ValueError(
                f"Cannot restart training session {session_id}: "
                "training is currently running. Cancel or wait for completion first."
            )

        if session.status == SessionStatus.PENDING.value:
            raise ValueError(
                f"Cannot restart training session {session_id}: "
                "session is already pending. Start it instead of restarting."
            )

        # Check for active jobs (RUNNING status)
        active_jobs_result = await db.execute(
            select(TrainingJob)
            .where(
                TrainingJob.session_id == session_id,
                TrainingJob.status == JobStatus.RUNNING.value,
            )
            .limit(1)
        )
        if active_jobs_result.scalar_one_or_none():
            raise RuntimeError(
                f"Cannot restart: some jobs are still running for session {session_id}. "
                "Wait for completion or cancel jobs first."
            )

    async def cleanup(
        self,
        db: AsyncSession,
        session_id: int,
    ) -> CleanupStats:
        """Reset training jobs to PENDING state.

        Options:
        - failed_only=True: Reset only FAILED jobs
        - failed_only=False: Reset ALL jobs (full restart)

        Idempotent: Resetting already-PENDING jobs is safe.

        Args:
            db: Database session
            session_id: Training session ID

        Returns:
            CleanupStats with reset counts
        """
        start_time = time.time()

        if self.failed_only:
            reset_count = await self._reset_failed_jobs(db, session_id)
        else:
            reset_count = await self._reset_all_jobs(db, session_id)

        # Count preserved jobs (not reset)
        total_jobs_result = await db.execute(
            select(TrainingJob).where(TrainingJob.session_id == session_id)
        )
        total_jobs = len(total_jobs_result.scalars().all())
        preserved_count = total_jobs - reset_count

        duration_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"Cleanup completed: reset {reset_count} jobs, preserved {preserved_count} jobs",
            extra={
                "session_id": session_id,
                "reset_count": reset_count,
                "preserved_count": preserved_count,
                "failed_only": self.failed_only,
            },
        )

        return CleanupStats(
            operation="training_restart",
            session_id=session_id,
            items_deleted=0,  # We don't delete jobs, just reset status
            items_reset=reset_count,
            items_preserved=preserved_count,
            duration_ms=duration_ms,
        )

    async def reset_state(
        self,
        db: AsyncSession,
        session_id: int,
    ) -> None:
        """Reset session counters and status to PENDING.

        Args:
            db: Database session
            session_id: Training session ID
        """
        result = await db.execute(
            select(TrainingSession).where(TrainingSession.id == session_id)
        )
        session = result.scalar_one_or_none()

        if not session:
            raise ValueError(f"Training session {session_id} not found")

        # Reset counters
        session.processed_images = 0
        session.failed_images = 0
        session.status = SessionStatus.PENDING.value

        # Clear timestamps
        session.started_at = None
        session.completed_at = None
        session.paused_at = None

        # Set restart metadata
        session.reset_at = datetime.now(UTC)
        session.reset_reason = "Failed jobs only" if self.failed_only else "Full restart"

        logger.info(
            "Reset session state to PENDING",
            extra={
                "session_id": session_id,
                "reset_reason": session.reset_reason,
            },
        )

    async def enqueue_job(
        self,
        db: AsyncSession,
        session_id: int,
    ) -> str:
        """Start training using existing workflow.

        CRITICAL: Uses existing train_session job function to maintain compatibility.

        Args:
            db: Database session
            session_id: Training session ID

        Returns:
            RQ job ID
        """
        from image_search_service.queue.training_jobs import train_session

        # Enqueue training job (high priority)
        queue = get_queue(QUEUE_HIGH)
        rq_job = queue.enqueue(train_session, session_id, job_timeout="1h")

        logger.info(
            "Enqueued training job",
            extra={
                "session_id": session_id,
                "job_id": rq_job.id,
            },
        )

        return str(rq_job.id)

    async def _capture_snapshot(
        self,
        db: AsyncSession,
        session_id: int,
    ) -> dict[str, Any]:
        """Capture training session state for rollback.

        Args:
            db: Database session
            session_id: Training session ID

        Returns:
            Snapshot dictionary with session and job states
        """
        # Get session state
        result = await db.execute(
            select(TrainingSession).where(TrainingSession.id == session_id)
        )
        session = result.scalar_one_or_none()

        if not session:
            raise ValueError(f"Training session {session_id} not found")

        # Get all job states
        jobs_result = await db.execute(
            select(TrainingJob.id, TrainingJob.status, TrainingJob.progress).where(
                TrainingJob.session_id == session_id
            )
        )
        job_states = {
            row.id: {"status": row.status, "progress": row.progress} for row in jobs_result
        }

        snapshot = {
            "session_id": session_id,
            "session_status": session.status,
            "processed_images": session.processed_images,
            "failed_images": session.failed_images,
            "started_at": session.started_at.isoformat() if session.started_at else None,
            "completed_at": session.completed_at.isoformat() if session.completed_at else None,
            "paused_at": session.paused_at.isoformat() if session.paused_at else None,
            "job_states": job_states,
        }

        logger.debug(
            f"Captured snapshot for session {session_id}",
            extra={"session_id": session_id, "job_count": len(job_states)},
        )

        return snapshot

    async def _restore_snapshot(
        self,
        db: AsyncSession,
        snapshot: dict[str, Any],
    ) -> None:
        """Restore training session state from snapshot.

        Args:
            db: Database session
            snapshot: Snapshot dictionary from _capture_snapshot
        """
        session_id = snapshot["session_id"]

        # Restore session state
        result = await db.execute(
            select(TrainingSession).where(TrainingSession.id == session_id)
        )
        session = result.scalar_one_or_none()

        if not session:
            logger.warning(f"Session {session_id} not found during rollback")
            return

        session.status = snapshot["session_status"]
        session.processed_images = snapshot["processed_images"]
        session.failed_images = snapshot["failed_images"]

        # Restore timestamps
        session.started_at = (
            datetime.fromisoformat(snapshot["started_at"]) if snapshot["started_at"] else None
        )
        session.completed_at = (
            datetime.fromisoformat(snapshot["completed_at"]) if snapshot["completed_at"] else None
        )
        session.paused_at = (
            datetime.fromisoformat(snapshot["paused_at"]) if snapshot["paused_at"] else None
        )

        # Restore job states
        for job_id, state in snapshot["job_states"].items():
            await db.execute(
                update(TrainingJob)
                .where(TrainingJob.id == job_id)
                .values(status=state["status"], progress=state["progress"])
            )

        logger.info(
            f"Restored snapshot for session {session_id}",
            extra={"session_id": session_id, "job_count": len(snapshot["job_states"])},
        )

    async def _reset_failed_jobs(
        self,
        db: AsyncSession,
        session_id: int,
    ) -> int:
        """Reset failed jobs to PENDING state.

        Args:
            db: Database session
            session_id: Training session ID

        Returns:
            Number of jobs reset
        """
        result = await db.execute(
            update(TrainingJob)
            .where(
                TrainingJob.session_id == session_id,
                TrainingJob.status == JobStatus.FAILED.value,
            )
            .values(
                status=JobStatus.PENDING.value,
                progress=0,
                error_message=None,
                started_at=None,
                completed_at=None,
                processing_time_ms=None,
            )
        )

        rowcount = result.rowcount or 0  # type: ignore[attr-defined]
        logger.info(
            f"Reset {rowcount} failed jobs to PENDING",
            extra={"session_id": session_id, "count": rowcount},
        )

        return rowcount

    async def _reset_all_jobs(
        self,
        db: AsyncSession,
        session_id: int,
    ) -> int:
        """Reset all jobs to PENDING state (full restart).

        Args:
            db: Database session
            session_id: Training session ID

        Returns:
            Number of jobs reset
        """
        result = await db.execute(
            update(TrainingJob)
            .where(TrainingJob.session_id == session_id)
            .values(
                status=JobStatus.PENDING.value,
                progress=0,
                error_message=None,
                started_at=None,
                completed_at=None,
                processing_time_ms=None,
            )
        )

        rowcount = result.rowcount or 0  # type: ignore[attr-defined]
        logger.info(
            f"Reset {rowcount} jobs to PENDING (full restart)",
            extra={"session_id": session_id, "count": rowcount},
        )

        return rowcount
