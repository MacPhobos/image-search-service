"""Tests for three critical bugs fixed in training session discovery offload.

Bug 1 — Generic exception handler does not write FAILED status to DB
    train_session() generic ``except Exception`` block must update session
    status to FAILED in the database, not just return a "failed" dict.

Bug 2 — restart_training() has no guard for DISCOVERING state
    restart_training() must raise ValueError when the session is in
    DISCOVERING state so the original worker is not raced by a second one.

Bug 3 — No mid-discovery cancellation checkpoint
    discover_assets_sync() must honour a cancellation_check callback and
    return early (with partial results) when the check returns True.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from image_search_service.db.models import (
    SessionStatus,
    TrainingSession,
    TrainingSubdirectory,
)
from image_search_service.db.sync_operations import discover_assets_sync

# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------


def _make_sync_session(
    sync_db_session: Session,
    *,
    status: str = SessionStatus.DISCOVERING.value,
    name: str = "Test Session",
    root_path: str = "/tmp/photos",
) -> TrainingSession:
    """Create and persist a TrainingSession using the sync session."""
    ts = TrainingSession(
        name=name,
        root_path=root_path,
        status=status,
        total_images=0,
        processed_images=0,
        failed_images=0,
    )
    sync_db_session.add(ts)
    sync_db_session.commit()
    sync_db_session.refresh(ts)
    return ts


def _add_subdir(
    sync_db_session: Session,
    session_id: int,
    path: str,
    *,
    name: str = "photos",
    image_count: int = 0,
) -> TrainingSubdirectory:
    """Create and persist a TrainingSubdirectory."""
    subdir = TrainingSubdirectory(
        session_id=session_id,
        path=path,
        name=name,
        selected=True,
        image_count=image_count,
        trained_count=0,
    )
    sync_db_session.add(subdir)
    sync_db_session.commit()
    return subdir


# ===========================================================================
# Bug 1: Generic exception handler must write FAILED status to DB
# ===========================================================================


class TestBug1GenericExceptionHandlerWritesFailedStatus:
    """train_session() generic exception handler writes FAILED to the DB."""

    def test_oserror_during_discovery_sets_session_failed_in_db(
        self, sync_db_session: Session
    ) -> None:
        """When discover_assets_sync raises OSError the session must be FAILED in DB.

        Previously, only ValueError was caught with a DB update.  Any other
        exception (OSError, PermissionError, SQLAlchemyError, …) left the
        session stuck in DISCOVERING forever.
        """
        from image_search_service.db.sync_operations import get_session_by_id_sync
        from image_search_service.queue.training_jobs import train_session

        # Arrange: create a session in DISCOVERING state
        ts = _make_sync_session(sync_db_session, status=SessionStatus.DISCOVERING.value)
        session_id = ts.id

        # Patch get_sync_session to return the shared in-memory session so the
        # job function and our assertions use the same database connection.
        with (
            patch(
                "image_search_service.queue.training_jobs.get_sync_session",
                return_value=sync_db_session,
            ),
            patch(
                "image_search_service.queue.training_jobs.discover_assets_sync",
                side_effect=OSError("Permission denied: /photos"),
            ),
        ):
            result = train_session(session_id)

        # Assert: return value signals failure
        assert result["status"] == "failed"
        assert session_id == result["session_id"]

        # Assert: DB reflects FAILED status (not still DISCOVERING).
        # Expire the in-session cache so we get the committed value.
        sync_db_session.expire_all()
        refreshed = get_session_by_id_sync(sync_db_session, session_id)
        assert refreshed is not None
        assert refreshed.status == SessionStatus.FAILED.value, (
            f"Expected FAILED, got {refreshed.status!r}. "
            "The generic exception handler must commit FAILED status to DB."
        )

    def test_oserror_during_running_phase_returns_failed_dict(
        self, sync_db_session: Session
    ) -> None:
        """RUNNING sessions that hit an unhandled exception return a failed dict."""
        from image_search_service.queue.training_jobs import train_session

        # Arrange: session already in RUNNING state (past discovery)
        ts = _make_sync_session(sync_db_session, status=SessionStatus.RUNNING.value)
        session_id = ts.id

        with (
            patch(
                "image_search_service.queue.training_jobs.get_sync_session",
                return_value=sync_db_session,
            ),
            # Blow up on the `select(TrainingJob)` query that follows discovery
            patch.object(
                sync_db_session,
                "execute",
                side_effect=RuntimeError("Simulated DB failure"),
            ),
        ):
            result = train_session(session_id)

        assert result["status"] == "failed"
        assert result["error"] is not None


# ===========================================================================
# Bug 2: restart_training() must guard against DISCOVERING state
# ===========================================================================


class TestBug2RestartTrainingDiscoveringGuard:
    """restart_training() must raise ValueError when session is DISCOVERING."""

    @pytest.mark.asyncio
    async def test_restart_when_discovering_raises_value_error(
        self, db_session: AsyncSession
    ) -> None:
        """Restarting a DISCOVERING session must raise ValueError immediately."""
        from image_search_service.services.training_service import TrainingService

        service = TrainingService()

        # Arrange: create a session in DISCOVERING state
        ts = TrainingSession(
            name="Discovering Session",
            root_path="/tmp/photos",
            status=SessionStatus.DISCOVERING.value,
            total_images=0,
            processed_images=0,
            failed_images=0,
        )
        db_session.add(ts)
        await db_session.commit()
        await db_session.refresh(ts)

        # Act + Assert
        with pytest.raises(ValueError, match="currently discovering assets"):
            await service.restart_training(db_session, ts.id)

    @pytest.mark.asyncio
    async def test_restart_when_failed_does_not_raise(self, db_session: AsyncSession) -> None:
        """Restarting a FAILED session must succeed (guard must not block it)."""
        from image_search_service.services.training_service import TrainingService

        service = TrainingService()

        ts = TrainingSession(
            name="Failed Session",
            root_path="/tmp/photos",
            status=SessionStatus.FAILED.value,
            total_images=5,
            processed_images=0,
            failed_images=5,
        )
        db_session.add(ts)
        await db_session.commit()
        await db_session.refresh(ts)

        # Patch the queue so we don't need a real Redis/RQ instance
        with patch(
            "image_search_service.services.training_service.get_queue"
        ) as mock_queue_factory:
            mock_queue = MagicMock()
            mock_rq_job = MagicMock()
            mock_rq_job.id = "mock-rq-job-id"
            mock_queue.enqueue = MagicMock(return_value=mock_rq_job)
            mock_queue_factory.return_value = mock_queue

            # Should NOT raise
            result = await service.restart_training(db_session, ts.id)

        # After restart, session transitions to DISCOVERING (via enqueue_training)
        assert result is not None

    @pytest.mark.asyncio
    async def test_restart_when_not_found_raises_value_error(
        self, db_session: AsyncSession
    ) -> None:
        """Restarting a non-existent session must raise ValueError."""
        from image_search_service.services.training_service import TrainingService

        service = TrainingService()

        with pytest.raises(ValueError, match="not found"):
            await service.restart_training(db_session, 999_999)


# ===========================================================================
# Bug 3: discover_assets_sync cancellation checkpoint
# ===========================================================================


class TestBug3DiscoveryCancellationCheckpoint:
    """discover_assets_sync() respects cancellation_check during rglob."""

    def test_cancellation_check_returns_early_with_partial_results(
        self, sync_db_session: Session, tmp_path: Path
    ) -> None:
        """When cancellation_check returns True mid-scan, discovery stops early.

        We create 1 001 image files so the first 500-file boundary is crossed.
        The callback returns True on the second call (after 1 000 files), which
        must abort the scan before all files are processed.
        """
        ts = _make_sync_session(sync_db_session, status=SessionStatus.DISCOVERING.value)

        # Create a directory with 1 001 JPEG files (crossing the 500-file boundary twice)
        photo_dir = tmp_path / "photos"
        photo_dir.mkdir()

        for i in range(1001):
            img_path = photo_dir / f"img_{i:04d}.jpg"
            Image.new("RGB", (4, 4), color=(i % 256, 0, 0)).save(img_path, format="JPEG")

        _add_subdir(sync_db_session, ts.id, str(photo_dir), image_count=1001)

        # Cancellation callback: allow first 500 files, cancel at second check (1 000)
        call_count = 0

        def cancellation_check() -> bool:
            nonlocal call_count
            call_count += 1
            # First call at file 500: don't cancel yet
            # Second call at file 1 000: cancel
            return call_count >= 2

        assets = discover_assets_sync(sync_db_session, ts.id, cancellation_check=cancellation_check)

        # Scan was interrupted before processing all 1 001 files
        assert len(assets) < 1001, (
            f"Expected fewer than 1 001 assets (early exit), got {len(assets)}. "
            "The cancellation checkpoint must abort the scan when check() is True."
        )
        # The callback must have been called at least once (at the 500-file mark)
        assert call_count >= 1

    def test_no_cancellation_check_scans_all_files(
        self, sync_db_session: Session, tmp_path: Path
    ) -> None:
        """Without a cancellation_check, all files are discovered (backward compat)."""
        ts = _make_sync_session(sync_db_session, status=SessionStatus.DISCOVERING.value)

        photo_dir = tmp_path / "photos_full"
        photo_dir.mkdir()
        for i in range(5):
            Image.new("RGB", (4, 4)).save(photo_dir / f"img_{i}.jpg", format="JPEG")

        _add_subdir(sync_db_session, ts.id, str(photo_dir), name="photos_full", image_count=5)

        assets = discover_assets_sync(sync_db_session, ts.id, cancellation_check=None)

        assert len(assets) == 5

    def test_cancellation_check_never_true_scans_all_files(
        self, sync_db_session: Session, tmp_path: Path
    ) -> None:
        """When cancellation_check always returns False, all files are scanned."""
        ts = _make_sync_session(sync_db_session, status=SessionStatus.DISCOVERING.value)

        photo_dir = tmp_path / "photos_nocancell"
        photo_dir.mkdir()
        for i in range(5):
            Image.new("RGB", (4, 4)).save(photo_dir / f"img_{i}.jpg", format="JPEG")

        _add_subdir(sync_db_session, ts.id, str(photo_dir), name="photos_nocancell", image_count=5)

        assets = discover_assets_sync(sync_db_session, ts.id, cancellation_check=lambda: False)

        assert len(assets) == 5
