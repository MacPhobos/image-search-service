"""Progress tracking utilities for background jobs."""

from datetime import UTC, datetime

from sqlalchemy.orm import Session

from image_search_service.core.logging import get_logger
from image_search_service.db.models import SessionStatus
from image_search_service.db.sync_operations import (
    check_session_status_sync,
    get_session_by_id_sync,
    update_session_progress_sync,
)

logger = get_logger(__name__)


class ProgressTracker:
    """Track and update training progress for a session."""

    def __init__(self, session_id: int) -> None:
        """Initialize progress tracker.

        Args:
            session_id: Training session ID
        """
        self.session_id = session_id
        self._last_check_time = datetime.now(UTC)

    def update_progress(
        self, db_session: Session, processed: int, failed: int
    ) -> None:
        """Update session progress in database.

        Args:
            db_session: Database session
            processed: Total number of processed images
            failed: Total number of failed images
        """
        update_session_progress_sync(db_session, self.session_id, processed, failed)

    def check_cancelled(self, db_session: Session) -> bool:
        """Check if session has been cancelled.

        Args:
            db_session: Database session

        Returns:
            True if session is cancelled, False otherwise
        """
        status = check_session_status_sync(db_session, self.session_id)
        return status == SessionStatus.CANCELLED.value

    def check_paused(self, db_session: Session) -> bool:
        """Check if session has been paused.

        Args:
            db_session: Database session

        Returns:
            True if session is paused, False otherwise
        """
        status = check_session_status_sync(db_session, self.session_id)
        return status == SessionStatus.PAUSED.value

    def should_stop(self, db_session: Session) -> bool:
        """Check if processing should stop (cancelled or paused).

        Args:
            db_session: Database session

        Returns:
            True if processing should stop, False otherwise
        """
        return self.check_cancelled(db_session) or self.check_paused(db_session)

    def get_current_progress(self, db_session: Session) -> dict[str, int | float]:
        """Get current progress statistics.

        Args:
            db_session: Database session

        Returns:
            Dictionary with progress statistics
        """
        session = get_session_by_id_sync(db_session, self.session_id)

        if not session:
            logger.error(f"Session {self.session_id} not found")
            return {
                "total": 0,
                "processed": 0,
                "failed": 0,
                "remaining": 0,
                "percentage": 0.0,
            }

        total = session.total_images
        processed = session.processed_images
        failed = session.failed_images
        remaining = max(0, total - processed)
        percentage = (processed / total * 100) if total > 0 else 0.0

        return {
            "total": total,
            "processed": processed,
            "failed": failed,
            "remaining": remaining,
            "percentage": round(percentage, 2),
        }

    def calculate_eta(
        self, start_time: datetime, processed: int, total: int
    ) -> datetime | None:
        """Calculate estimated time of arrival (ETA).

        Args:
            start_time: Job start time
            processed: Number of processed images
            total: Total number of images

        Returns:
            Estimated completion datetime, or None if not enough data
        """
        if processed <= 0 or total <= 0:
            return None

        elapsed = (datetime.now(UTC) - start_time).total_seconds()
        if elapsed <= 0:
            return None

        # Calculate processing rate (images per second)
        rate = processed / elapsed

        # Calculate remaining time
        remaining = total - processed
        if rate <= 0:
            return None

        eta_seconds = remaining / rate
        eta = datetime.now(UTC).timestamp() + eta_seconds

        return datetime.fromtimestamp(eta, UTC)

    def calculate_rate(
        self, start_time: datetime, processed: int
    ) -> float:
        """Calculate processing rate in images per minute.

        Args:
            start_time: Job start time
            processed: Number of processed images

        Returns:
            Processing rate in images per minute
        """
        if processed <= 0:
            return 0.0

        elapsed = (datetime.now(UTC) - start_time).total_seconds()
        if elapsed <= 0:
            return 0.0

        # Convert to images per minute
        rate = (processed / elapsed) * 60

        return round(rate, 2)
