"""Synchronous database operations for RQ workers.

RQ workers run in a synchronous context, so they need sync database access.
This module provides synchronous database operations for background jobs.
"""

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

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


# ============================================================================
# Sync asset discovery helpers (used by RQ worker during DISCOVERING phase)
# ============================================================================

_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp"}
)


def ensure_asset_exists_sync(db_session: Session, file_path: str) -> ImageAsset:
    """Create or return existing ImageAsset for a file path (sync).

    Args:
        db_session: Synchronous database session
        file_path: Absolute file path for the image

    Returns:
        Existing or newly created ImageAsset record
    """
    stmt = select(ImageAsset).where(ImageAsset.path == file_path)
    existing = db_session.execute(stmt).scalar_one_or_none()
    if existing:
        return existing

    path_obj = Path(file_path)
    stat = path_obj.stat()
    asset = ImageAsset(
        path=file_path,
        file_size=stat.st_size,
        file_modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
    )
    db_session.add(asset)
    db_session.flush()
    logger.debug(f"Created new asset record for {file_path}")
    return asset


def discover_assets_sync(
    db_session: Session,
    session_id: int,
    cancellation_check: Callable[[], bool] | None = None,
) -> list[ImageAsset]:
    """Discover all image assets for selected subdirectories of a session (sync).

    Scans each selected TrainingSubdirectory recursively for image files and
    creates (or retrieves) ImageAsset records.  All writes are flushed but
    a single commit is made at the end for efficiency.

    The optional ``cancellation_check`` callback is invoked every 500 files
    during the filesystem scan so that a user-triggered cancellation is
    honoured without waiting for a potentially very long rglob to finish.
    When the callback returns ``True`` the scan stops early and whatever
    assets have been found so far are returned (the caller is responsible
    for handling a partial or empty result).

    Args:
        db_session: Synchronous database session
        session_id: Training session ID
        cancellation_check: Optional zero-argument callable that returns True
            when the scan should be aborted (e.g. session was cancelled).

    Returns:
        List of ImageAsset records discovered (new + existing).
        May be a partial list if ``cancellation_check`` returned True.

    Raises:
        ValueError: If the training session does not exist
    """
    training_session = get_session_by_id_sync(db_session, session_id)
    if not training_session:
        raise ValueError(f"Training session {session_id} not found")

    subdirs_stmt = (
        select(TrainingSubdirectory)
        .where(TrainingSubdirectory.session_id == session_id)
        .where(TrainingSubdirectory.selected.is_(True))
    )
    selected_subdirs: list[TrainingSubdirectory] = list(
        db_session.execute(subdirs_stmt).scalars().all()
    )

    if not selected_subdirs:
        logger.warning(f"No selected subdirectories for session {session_id}")
        return []

    all_assets: list[ImageAsset] = []

    for subdir in selected_subdirs:
        dir_path = Path(subdir.path)
        if not dir_path.exists() or not dir_path.is_dir():
            logger.warning(f"Subdirectory does not exist: {subdir.path}")
            continue

        logger.info(f"Scanning directory (sync): {subdir.path}")
        dir_assets: list[ImageAsset] = []
        file_count = 0

        for file_path in dir_path.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in _IMAGE_EXTENSIONS:
                continue

            file_count += 1

            # Check for cancellation every 500 files to avoid running an
            # arbitrarily long filesystem scan after the user has cancelled.
            if cancellation_check is not None and file_count % 500 == 0:
                if cancellation_check():
                    logger.info(
                        f"Discovery cancelled for session {session_id} "
                        f"after {file_count} files scanned"
                    )
                    # Commit assets found so far and return early.
                    db_session.commit()
                    return all_assets

            asset = ensure_asset_exists_sync(db_session, str(file_path.absolute()))
            dir_assets.append(asset)

        logger.info(f"Found {len(dir_assets)} images in {subdir.path}")
        all_assets.extend(dir_assets)

    # Commit all new asset records together
    db_session.commit()

    logger.info(
        f"Total assets discovered for session {session_id}: {len(all_assets)}"
    )
    return all_assets


def create_training_jobs_sync(
    db_session: Session,
    session_id: int,
    assets: list[ImageAsset],
) -> dict[str, int]:
    """Create TrainingJob records with perceptual-hash deduplication (sync).

    Mirrors the logic of TrainingService.create_training_jobs() but uses
    synchronous DB operations so it can run inside the RQ worker.

    Algorithm:
    1. Skip assets that already have jobs for this session.
    2. Compute missing perceptual hashes.
    3. Group assets by hash; oldest asset per group gets a PENDING job,
       duplicates get SKIPPED jobs.
    4. Increment session.skipped_images counter.

    Args:
        db_session: Synchronous database session
        session_id: Training session ID
        assets: List of ImageAsset records to process

    Returns:
        Dict with keys: jobs_created, unique, skipped
    """
    from collections import defaultdict

    from image_search_service.services.perceptual_hash import compute_perceptual_hash

    asset_ids = [a.id for a in assets]

    # Find assets that already have jobs
    existing_stmt = (
        select(TrainingJob.asset_id)
        .where(TrainingJob.session_id == session_id)
        .where(TrainingJob.asset_id.in_(asset_ids))
    )
    existing_asset_ids: set[int] = set(
        db_session.execute(existing_stmt).scalars().all()
    )

    new_assets = [a for a in assets if a.id not in existing_asset_ids]

    if not new_assets:
        logger.debug(
            f"All {len(asset_ids)} assets already have jobs for session {session_id}"
        )
        return {"jobs_created": 0, "unique": 0, "skipped": 0}

    # Compute missing perceptual hashes
    for asset in new_assets:
        if asset.perceptual_hash is None:
            try:
                asset.perceptual_hash = compute_perceptual_hash(asset.path)
                logger.debug(
                    f"Computed hash for asset {asset.id}: {asset.perceptual_hash}"
                )
            except Exception as exc:
                logger.warning(f"Failed to compute hash for asset {asset.id}: {exc}")

    db_session.commit()

    # Refresh to get stored hashes
    for asset in new_assets:
        db_session.refresh(asset)

    # Group by perceptual hash (None → unique group per asset)
    hash_groups: dict[str | None, list[ImageAsset]] = defaultdict(list)
    for asset in new_assets:
        hash_groups[asset.perceptual_hash].append(asset)

    # Sort each group oldest-first so the oldest is the representative
    for hash_val in hash_groups:
        hash_groups[hash_val].sort(key=lambda a: a.created_at)

    jobs_created = 0
    unique_count = 0
    skipped_count = 0

    for hash_val, group in hash_groups.items():
        if hash_val is None or len(group) == 1:
            for asset in group:
                db_session.add(
                    TrainingJob(
                        session_id=session_id,
                        asset_id=asset.id,
                        status=JobStatus.PENDING.value,
                        progress=0,
                        image_path=asset.path,
                    )
                )
                jobs_created += 1
                unique_count += 1
        else:
            representative = group[0]
            duplicates = group[1:]

            db_session.add(
                TrainingJob(
                    session_id=session_id,
                    asset_id=representative.id,
                    status=JobStatus.PENDING.value,
                    progress=0,
                    image_path=representative.path,
                )
            )
            jobs_created += 1
            unique_count += 1

            for dup in duplicates:
                db_session.add(
                    TrainingJob(
                        session_id=session_id,
                        asset_id=dup.id,
                        status=JobStatus.SKIPPED.value,
                        progress=100,
                        image_path=dup.path,
                        skip_reason=f"Duplicate of asset {representative.id}",
                    )
                )
                jobs_created += 1
                skipped_count += 1

    # Update session skipped_images counter
    session_obj = get_session_by_id_sync(db_session, session_id)
    if session_obj and skipped_count > 0:
        session_obj.skipped_images = (session_obj.skipped_images or 0) + skipped_count

    db_session.commit()

    logger.info(
        f"Created {jobs_created} training jobs for session {session_id}: "
        f"{unique_count} unique, {skipped_count} skipped "
        f"({len(existing_asset_ids)} already existed)"
    )

    return {"jobs_created": jobs_created, "unique": unique_count, "skipped": skipped_count}
