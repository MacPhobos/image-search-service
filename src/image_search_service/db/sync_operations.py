"""Synchronous database operations for RQ workers.

RQ workers run in a synchronous context, so they need sync database access.
This module provides synchronous database operations for background jobs.
"""

from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from image_search_service.core.logging import get_logger
from image_search_service.db.models import (
    ImageAsset,
    JobStatus,
    SessionStatus,
    TrainingEvidence,
    TrainingJob,
    TrainingSession,
    TrainingSubdirectory,
)
from image_search_service.db.session import get_sync_engine

logger = get_logger(__name__)


def get_sync_session() -> Session:
    """Get synchronous database session for RQ workers.

    Returns:
        Synchronous SQLAlchemy session
    """
    engine = get_sync_engine()
    return Session(engine)


def update_training_job_sync(
    session: Session, job_id: int, status: str, error: str | None = None
) -> None:
    """Update training job status synchronously.

    Args:
        session: Database session
        job_id: Training job ID
        status: New job status
        error: Optional error message
    """
    stmt = update(TrainingJob).where(TrainingJob.id == job_id)

    values: dict[str, object] = {"status": status}

    if status == JobStatus.RUNNING.value:
        values["started_at"] = datetime.now(UTC)
    elif status in (JobStatus.COMPLETED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value):
        values["completed_at"] = datetime.now(UTC)

    if error:
        values["error_message"] = error

    stmt = stmt.values(**values)
    session.execute(stmt)
    session.commit()

    logger.debug(f"Updated training job {job_id} to status {status}")


def update_job_progress_sync(
    session: Session, job_id: int, progress: int, processing_time_ms: int | None = None
) -> None:
    """Update training job progress synchronously.

    Args:
        session: Database session
        job_id: Training job ID
        progress: Progress percentage (0-100)
        processing_time_ms: Processing time in milliseconds
    """
    values: dict[str, object] = {"progress": progress}

    if processing_time_ms is not None:
        values["processing_time_ms"] = processing_time_ms

    stmt = update(TrainingJob).where(TrainingJob.id == job_id).values(**values)
    session.execute(stmt)
    session.commit()


def update_session_progress_sync(
    session: Session, session_id: int, processed: int, failed: int
) -> None:
    """Update training session progress synchronously.

    Args:
        session: Database session
        session_id: Training session ID
        processed: Number of processed images
        failed: Number of failed images
    """
    stmt = (
        update(TrainingSession)
        .where(TrainingSession.id == session_id)
        .values(processed_images=processed, failed_images=failed)
    )
    session.execute(stmt)
    session.commit()

    logger.debug(f"Updated session {session_id} progress: {processed} processed, {failed} failed")


def create_evidence_sync(session: Session, evidence_data: dict[str, object]) -> int:
    """Create training evidence record synchronously.

    Args:
        session: Database session
        evidence_data: Dictionary with evidence data

    Returns:
        ID of created evidence record
    """
    evidence = TrainingEvidence(**evidence_data)
    session.add(evidence)
    session.commit()
    session.refresh(evidence)

    logger.debug(f"Created training evidence {evidence.id} for asset {evidence.asset_id}")
    return evidence.id


def check_session_status_sync(session: Session, session_id: int) -> str:
    """Check training session status synchronously.

    Args:
        session: Database session
        session_id: Training session ID

    Returns:
        Current session status string
    """
    stmt = select(TrainingSession.status).where(TrainingSession.id == session_id)
    result = session.execute(stmt).scalar_one_or_none()

    if result is None:
        logger.warning(f"Session {session_id} not found")
        return SessionStatus.FAILED.value

    return result


def get_asset_by_id_sync(session: Session, asset_id: int) -> ImageAsset | None:
    """Get image asset by ID synchronously.

    Args:
        session: Database session
        asset_id: Asset ID

    Returns:
        ImageAsset or None if not found
    """
    stmt = select(ImageAsset).where(ImageAsset.id == asset_id)
    return session.execute(stmt).scalar_one_or_none()


def update_asset_indexed_at_sync(session: Session, asset_id: int) -> None:
    """Update asset indexed_at timestamp synchronously.

    Args:
        session: Database session
        asset_id: Asset ID
    """
    stmt = (
        update(ImageAsset)
        .where(ImageAsset.id == asset_id)
        .values(indexed_at=datetime.now(UTC))
    )
    session.execute(stmt)
    session.commit()

    logger.debug(f"Updated asset {asset_id} indexed_at timestamp")


def get_training_job_sync(session: Session, job_id: int) -> TrainingJob | None:
    """Get training job by ID synchronously.

    Args:
        session: Database session
        job_id: Training job ID

    Returns:
        TrainingJob or None if not found
    """
    stmt = select(TrainingJob).where(TrainingJob.id == job_id)
    return session.execute(stmt).scalar_one_or_none()


def get_session_by_id_sync(session: Session, session_id: int) -> TrainingSession | None:
    """Get training session by ID synchronously.

    Args:
        session: Database session
        session_id: Training session ID

    Returns:
        TrainingSession or None if not found
    """
    stmt = select(TrainingSession).where(TrainingSession.id == session_id)
    return session.execute(stmt).scalar_one_or_none()


def increment_subdirectory_trained_count_sync(
    session: Session, session_id: int, asset_file_path: str
) -> None:
    """Increment trained_count for the subdirectory containing the asset.

    This function is called when a training job completes successfully.
    It finds the TrainingSubdirectory record matching the asset's parent
    directory and session_id, then increments its trained_count.

    Args:
        session: Database session
        session_id: Training session ID
        asset_file_path: Full file path of the trained asset

    Returns:
        None (silently skips if subdirectory not found)
    """
    from pathlib import Path

    # Extract parent directory from asset path
    parent_dir = str(Path(asset_file_path).parent)

    # Find matching subdirectory for this session
    stmt = (
        select(TrainingSubdirectory)
        .where(TrainingSubdirectory.session_id == session_id)
        .where(TrainingSubdirectory.path == parent_dir)
    )
    subdir = session.execute(stmt).scalar_one_or_none()

    if subdir:
        # Increment trained_count
        subdir.trained_count = (subdir.trained_count or 0) + 1
        session.commit()

        logger.debug(
            f"Incremented trained_count for subdirectory {subdir.id} "
            f"(path: {parent_dir}, count: {subdir.trained_count})"
        )
    else:
        # This can happen if:
        # 1. Asset was discovered but not part of selected subdirectories
        # 2. Subdirectory record was deleted
        # 3. Path mismatch due to symlinks or path normalization
        logger.debug(
            f"Subdirectory not found for asset path {asset_file_path} "
            f"in session {session_id} (parent: {parent_dir})"
        )
