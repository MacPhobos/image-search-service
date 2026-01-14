"""Base service for restart operations with safety guarantees.

This module provides abstract base classes and utilities for implementing
safe, idempotent restart operations for training, face detection, and clustering.

All restart services follow a consistent workflow:
1. State validation (ensure restart is allowed)
2. Snapshot capture (for rollback)
3. Cleanup (delete old data)
4. State reset (update session counters/status)
5. Job enqueue (trigger background processing)
6. Rollback on failure (restore snapshot if any step fails)
"""

import time
from abc import ABC, abstractmethod
from typing import Any, TypedDict

from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.core.logging import get_logger

logger = get_logger(__name__)


class CleanupStats(TypedDict, total=False):
    """Standard cleanup statistics returned by all restart operations.

    Required fields:
        operation: Name of the restart operation (e.g., "training_restart")
        session_id: ID of the session being restarted
        items_deleted: Number of items deleted
        items_reset: Number of items reset to initial state
        items_preserved: Number of items preserved
        duration_ms: Operation duration in milliseconds

    Optional fields:
        job_id: Background job ID (if enqueued)
        Any operation-specific fields (e.g., face_instances_deleted)
    """

    operation: str
    session_id: int
    items_deleted: int
    items_reset: int
    items_preserved: int
    duration_ms: int
    job_id: str  # Optional, added during enqueue


class RestartServiceBase(ABC):
    """Base class for all restart operations.

    Provides common functionality:
    - State validation (prevent invalid restarts)
    - Transaction management (atomic operations)
    - Rollback on failure (restore snapshot)
    - Audit logging (track all restart events)
    - Idempotency checks (safe to call multiple times)

    Subclasses must implement:
    - validate_state(): Check if restart is allowed
    - cleanup(): Delete old data (idempotent)
    - reset_state(): Reset session status and counters
    - enqueue_job(): Start background job
    - _capture_snapshot(): Capture state for rollback
    - _restore_snapshot(): Restore from snapshot
    """

    @abstractmethod
    async def validate_state(
        self,
        db: AsyncSession,
        session_id: int,
    ) -> None:
        """Validate that restart is allowed.

        Args:
            db: Database session
            session_id: Session ID to restart

        Raises:
            ValueError: If session in invalid state for restart
            RuntimeError: If job currently running
        """
        pass

    @abstractmethod
    async def cleanup(
        self,
        db: AsyncSession,
        session_id: int,
    ) -> CleanupStats:
        """Delete old data from previous run.

        CRITICAL: Must be idempotent (calling twice produces same result).

        Args:
            db: Database session
            session_id: Session ID to clean up

        Returns:
            CleanupStats with deletion/reset counts
        """
        pass

    @abstractmethod
    async def reset_state(
        self,
        db: AsyncSession,
        session_id: int,
    ) -> None:
        """Reset session status and counters.

        Args:
            db: Database session
            session_id: Session ID to reset
        """
        pass

    @abstractmethod
    async def enqueue_job(
        self,
        db: AsyncSession,
        session_id: int,
    ) -> str:
        """Enqueue background job to re-run operation.

        CRITICAL: Must use existing job functions (no new code paths).

        Args:
            db: Database session
            session_id: Session ID to process

        Returns:
            job_id: Background job identifier
        """
        pass

    async def restart(
        self,
        db: AsyncSession,
        session_id: int,
    ) -> CleanupStats:
        """Execute full restart workflow with rollback on failure.

        Workflow:
        1. Validate state (cannot restart RUNNING sessions)
        2. Capture snapshot for rollback
        3. Cleanup old data
        4. Reset session state
        5. Enqueue new job
        6. Log success

        On failure at any step: rollback to snapshot and re-raise exception.

        Args:
            db: Database session
            session_id: Session ID to restart

        Returns:
            CleanupStats with operation results

        Raises:
            ValueError: If validation fails
            RuntimeError: If job still running
            Exception: If cleanup/reset/enqueue fails
        """
        start_time = time.time()

        # 1. Validate state
        logger.info(f"Validating restart for session {session_id}")
        await self.validate_state(db, session_id)

        # 2. Capture snapshot for rollback
        logger.info(f"Capturing snapshot for session {session_id}")
        snapshot = await self._capture_snapshot(db, session_id)

        try:
            # 3. Cleanup old data
            logger.info(f"Cleaning up old data for session {session_id}")
            stats = await self.cleanup(db, session_id)

            # 4. Reset session state
            logger.info(f"Resetting session state for session {session_id}")
            await self.reset_state(db, session_id)
            await db.commit()

            # 5. Enqueue new job
            logger.info(f"Enqueueing job for session {session_id}")
            job_id = await self.enqueue_job(db, session_id)
            stats["job_id"] = job_id

            # 6. Log success
            total_duration_ms = int((time.time() - start_time) * 1000)
            await self._log_restart_event(db, session_id, stats, total_duration_ms)

            logger.info(
                f"Restart completed for session {session_id} in {total_duration_ms}ms",
                extra={
                    "session_id": session_id,
                    "operation": stats["operation"],
                    "items_deleted": stats["items_deleted"],
                    "items_reset": stats["items_reset"],
                    "job_id": job_id,
                },
            )

            return stats

        except Exception as e:
            # Rollback to snapshot
            logger.error(
                f"Restart failed for session {session_id}, rolling back: {e}",
                exc_info=True,
            )

            await db.rollback()

            try:
                await self._restore_snapshot(db, snapshot)
                await db.commit()
                logger.info(f"Successfully rolled back session {session_id} to previous state")
            except Exception as rollback_error:
                logger.error(
                    f"Rollback failed for session {session_id}: {rollback_error}",
                    exc_info=True,
                )
                # Still raise original error

            raise

    @abstractmethod
    async def _capture_snapshot(
        self,
        db: AsyncSession,
        session_id: int,
    ) -> dict[str, Any]:
        """Capture current state for rollback.

        NOTE: Snapshot is limited to session state (status, counters).
        We cannot snapshot deleted records (face instances, persons).

        Args:
            db: Database session
            session_id: Session ID to snapshot

        Returns:
            Snapshot dictionary with state to restore
        """
        pass

    @abstractmethod
    async def _restore_snapshot(
        self,
        db: AsyncSession,
        snapshot: dict[str, Any],
    ) -> None:
        """Restore state from snapshot.

        NOTE: Only restores session state. Deleted records cannot be restored.

        Args:
            db: Database session
            snapshot: Snapshot dictionary from _capture_snapshot
        """
        pass

    async def _log_restart_event(
        self,
        db: AsyncSession,
        session_id: int,
        stats: CleanupStats,
        total_duration_ms: int,
    ) -> None:
        """Log restart event to audit trail.

        Uses structured logging with context. Could be extended to write
        to a RestartEvent database table for audit purposes.

        Args:
            db: Database session
            session_id: Session ID that was restarted
            stats: Cleanup statistics
            total_duration_ms: Total operation duration in milliseconds
        """
        logger.info(
            f"Restart audit: {stats['operation']} for session {session_id}",
            extra={
                "session_id": session_id,
                "operation": stats["operation"],
                "stats": {
                    "items_deleted": stats["items_deleted"],
                    "items_reset": stats["items_reset"],
                    "items_preserved": stats["items_preserved"],
                    "cleanup_duration_ms": stats["duration_ms"],
                    "total_duration_ms": total_duration_ms,
                },
                "job_id": stats.get("job_id"),
            },
        )
