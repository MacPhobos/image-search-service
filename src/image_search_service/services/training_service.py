"""Training session management service."""

from collections import defaultdict
from datetime import UTC, datetime
from typing import cast

from sqlalchemy import Integer, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from image_search_service.api.training_schemas import (
    DirectoryInfo,
    JobsSummary,
    ProgressStats,
    TrainingProgressResponse,
    TrainingSessionCreate,
    TrainingSessionUpdate,
)
from image_search_service.core.logging import get_logger
from image_search_service.db.models import (
    ImageAsset,
    JobStatus,
    SessionStatus,
    TrainingJob,
    TrainingSession,
    TrainingSubdirectory,
)
from image_search_service.queue.worker import QUEUE_HIGH, get_queue
from image_search_service.services.asset_discovery import AssetDiscoveryService
from image_search_service.services.perceptual_hash import compute_perceptual_hash

logger = get_logger(__name__)


class TrainingService:
    """Service for managing training sessions."""

    async def create_session(
        self, db: AsyncSession, data: TrainingSessionCreate
    ) -> TrainingSession:
        """Create a new training session.

        Args:
            db: Database session
            data: Session creation data

        Returns:
            Created training session
        """
        # Create session with config
        session = TrainingSession(
            name=data.name,
            root_path=data.root_path,
            category_id=data.category_id,
            status=SessionStatus.PENDING.value,
            config=data.config.model_dump() if data.config else None,
            total_images=0,
            processed_images=0,
            failed_images=0,
        )

        db.add(session)
        await db.flush()

        # Create subdirectory records if provided
        if data.subdirectories:
            for subdir_path in data.subdirectories:
                # Extract directory name from path
                name = subdir_path.split("/")[-1] if "/" in subdir_path else subdir_path

                subdir = TrainingSubdirectory(
                    session_id=session.id,
                    path=subdir_path,
                    name=name,
                    selected=True,  # Provided subdirectories are selected by default
                    image_count=0,  # Will be updated when scanning
                    trained_count=0,
                )
                db.add(subdir)

        await db.commit()

        # Refresh session with category relationship loaded
        refreshed_session = await self.get_session(db, session.id)
        if not refreshed_session:
            raise ValueError(f"Failed to retrieve created session {session.id}")

        logger.info(f"Created training session {session.id}: {session.name}")
        return refreshed_session

    async def get_session(
        self, db: AsyncSession, session_id: int
    ) -> TrainingSession | None:
        """Get a training session by ID.

        Args:
            db: Database session
            session_id: Session ID

        Returns:
            Training session or None if not found
        """
        query = (
            select(TrainingSession)
            .where(TrainingSession.id == session_id)
            .options(
                selectinload(TrainingSession.subdirectories),
                selectinload(TrainingSession.category),
            )
        )
        result = await db.execute(query)
        return result.scalar_one_or_none()

    async def list_sessions(
        self,
        db: AsyncSession,
        page: int = 1,
        page_size: int = 20,
        status: str | None = None,
    ) -> tuple[list[TrainingSession], int]:
        """List training sessions with pagination and optional filtering.

        Args:
            db: Database session
            page: Page number (1-indexed)
            page_size: Number of items per page
            status: Optional status filter

        Returns:
            Tuple of (sessions list, total count)
        """
        query = select(TrainingSession)
        count_query = select(func.count(TrainingSession.id))

        # Apply status filter
        if status:
            query = query.where(TrainingSession.status == status)
            count_query = count_query.where(TrainingSession.status == status)

        # Get total count
        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0

        # Apply pagination and ordering
        offset = (page - 1) * page_size
        query = (
            query.offset(offset)
            .limit(page_size)
            .order_by(TrainingSession.created_at.desc())
            .options(
                selectinload(TrainingSession.subdirectories),
                selectinload(TrainingSession.category),
            )
        )

        result = await db.execute(query)
        sessions = list(result.scalars().all())

        return sessions, total

    async def update_session(
        self, db: AsyncSession, session_id: int, data: TrainingSessionUpdate
    ) -> TrainingSession:
        """Update a training session.

        Args:
            db: Database session
            session_id: Session ID
            data: Update data

        Returns:
            Updated training session

        Raises:
            ValueError: If session not found
        """
        session = await self.get_session(db, session_id)
        if not session:
            raise ValueError(f"Training session {session_id} not found")

        # Update fields if provided
        if data.name is not None:
            session.name = data.name

        if data.config is not None:
            session.config = data.config.model_dump()

        await db.commit()
        await db.refresh(session)

        logger.info(f"Updated training session {session_id}")
        return session

    async def delete_session(self, db: AsyncSession, session_id: int) -> bool:
        """Delete a training session.

        Args:
            db: Database session
            session_id: Session ID

        Returns:
            True if deleted, False if not found
        """
        session = await self.get_session(db, session_id)
        if not session:
            return False

        await db.delete(session)
        await db.commit()

        logger.info(f"Deleted training session {session_id}")
        return True

    async def get_session_progress(
        self, db: AsyncSession, session_id: int
    ) -> TrainingProgressResponse:
        """Get progress information for a training session.

        Args:
            db: Database session
            session_id: Session ID

        Returns:
            Progress response with stats and job summary

        Raises:
            ValueError: If session not found
        """
        session = await self.get_session(db, session_id)
        if not session:
            raise ValueError(f"Training session {session_id} not found")

        # Get job counts by status
        job_counts_query = (
            select(TrainingJob.status, func.count(TrainingJob.id))
            .where(TrainingJob.session_id == session_id)
            .group_by(TrainingJob.status)
        )
        result = await db.execute(job_counts_query)
        job_counts: dict[str, int] = {status: count for status, count in result.all()}

        # Create jobs summary
        jobs_summary = JobsSummary(
            pending=job_counts.get(JobStatus.PENDING.value, 0),
            running=job_counts.get(JobStatus.RUNNING.value, 0),
            completed=job_counts.get(JobStatus.COMPLETED.value, 0),
            failed=job_counts.get(JobStatus.FAILED.value, 0),
            cancelled=job_counts.get(JobStatus.CANCELLED.value, 0),
            skipped=job_counts.get(JobStatus.SKIPPED.value, 0),
        )

        # Calculate progress percentage
        total = session.total_images
        current = session.processed_images
        percentage = (current / total * 100) if total > 0 else 0.0

        # Calculate ETA if session is running
        eta_seconds = None
        images_per_minute = None

        if session.status == SessionStatus.RUNNING.value and session.started_at:
            elapsed = (datetime.now(UTC) - session.started_at).total_seconds()
            if elapsed > 0 and current > 0:
                images_per_minute = (current / elapsed) * 60
                remaining = total - current
                if images_per_minute > 0:
                    eta_seconds = int(remaining / images_per_minute * 60)

        progress = ProgressStats(
            current=current,
            total=total,
            percentage=round(percentage, 2),
            etaSeconds=eta_seconds,
            imagesPerMinute=round(images_per_minute, 2) if images_per_minute else None,
        )

        return TrainingProgressResponse(
            sessionId=session_id,
            status=session.status,
            progress=progress,
            jobsSummary=jobs_summary,
        )

    async def update_subdirectories(
        self,
        db: AsyncSession,
        session_id: int,
        subdirectories: list[TrainingSubdirectory],
    ) -> None:
        """Update subdirectories for a training session.

        Args:
            db: Database session
            session_id: Session ID
            subdirectories: List of subdirectory records to add

        Raises:
            ValueError: If session not found
        """
        session = await self.get_session(db, session_id)
        if not session:
            raise ValueError(f"Training session {session_id} not found")

        # Add all subdirectories
        for subdir in subdirectories:
            db.add(subdir)

        await db.commit()
        logger.info(f"Added {len(subdirectories)} subdirectories to session {session_id}")

    async def enqueue_training(self, db: AsyncSession, session_id: int) -> str:
        """Enqueue training session for background processing.

        This method:
        1. Discovers all assets in selected subdirectories
        2. Creates TrainingJob records for each asset
        3. Enqueues the main training job to RQ
        4. Updates session status to RUNNING

        Args:
            db: Database session
            session_id: Training session ID

        Returns:
            RQ job ID

        Raises:
            ValueError: If session not found or has no selected subdirectories
        """
        # Get session
        session = await self.get_session(db, session_id)
        if not session:
            raise ValueError(f"Training session {session_id} not found")

        # Discover assets in selected subdirectories
        logger.info(f"Discovering assets for session {session_id}")
        discovery_service = AssetDiscoveryService()
        assets = await discovery_service.discover_assets(db, session_id)

        if not assets:
            raise ValueError(f"No assets found for session {session_id}")

        # Update session total_images count
        session.total_images = len(assets)
        session.processed_images = 0
        session.failed_images = 0

        # Create TrainingJob records for each asset
        asset_ids = [asset.id for asset in assets]
        job_count = await self.create_training_jobs(db, session_id, asset_ids)

        logger.info(f"Created {job_count} training jobs for session {session_id}")

        # Update session status and timestamps
        session.status = SessionStatus.RUNNING.value
        session.started_at = datetime.now(UTC)

        await db.commit()

        # Enqueue main training job to RQ (high priority)
        from image_search_service.queue.training_jobs import train_session

        queue = get_queue(QUEUE_HIGH)
        rq_job = queue.enqueue(train_session, session_id, job_timeout="1h")

        logger.info(
            f"Enqueued training session {session_id} as RQ job {rq_job.id} "
            f"with {job_count} assets"
        )

        return str(rq_job.id)

    async def create_training_jobs(
        self, db: AsyncSession, session_id: int, asset_ids: list[int]
    ) -> dict[str, int]:
        """Create TrainingJob records for a list of assets with hash deduplication.

        This method implements perceptual hash-based deduplication:
        1. Query assets and compute missing perceptual hashes
        2. Group assets by perceptual_hash
        3. For each hash group:
           - Create PENDING job for representative (oldest by created_at)
           - Create SKIPPED jobs for duplicates with skip_reason
        4. Store full image_path on ALL jobs for audit trail
        5. Update session.skipped_images counter

        Args:
            db: Database session
            session_id: Training session ID
            asset_ids: List of asset IDs to create jobs for

        Returns:
            Dict with keys:
            - jobs_created: Total jobs created (PENDING + SKIPPED)
            - unique: Number of unique images (PENDING jobs)
            - skipped: Number of duplicate images (SKIPPED jobs)
        """
        # Check which assets already have jobs for this session
        existing_query = (
            select(TrainingJob.asset_id)
            .where(TrainingJob.session_id == session_id)
            .where(TrainingJob.asset_id.in_(asset_ids))
        )
        result = await db.execute(existing_query)
        existing_asset_ids = set(result.scalars().all())

        # Get assets that need jobs
        new_asset_ids = [aid for aid in asset_ids if aid not in existing_asset_ids]

        if not new_asset_ids:
            logger.debug(f"All {len(asset_ids)} assets already have jobs for session {session_id}")
            return {"jobs_created": 0, "unique": 0, "skipped": 0}

        # Query assets with their hashes
        assets_query = select(ImageAsset).where(ImageAsset.id.in_(new_asset_ids))
        result = await db.execute(assets_query)
        assets = cast(list[ImageAsset], list(result.scalars().all()))

        # Compute missing hashes and update database
        for asset in assets:
            if asset.perceptual_hash is None:
                try:
                    asset.perceptual_hash = compute_perceptual_hash(asset.path)
                    logger.debug(f"Computed hash for asset {asset.id}: {asset.perceptual_hash}")
                except Exception as e:
                    logger.warning(f"Failed to compute hash for asset {asset.id}: {e}")
                    # Continue without hash (will be treated as unique)

        await db.commit()

        # Refresh each asset individually to get updated hashes
        for asset in assets:
            await db.refresh(asset)

        # Group assets by perceptual_hash
        hash_groups: dict[str | None, list[ImageAsset]] = defaultdict(list)
        for asset in assets:
            hash_groups[asset.perceptual_hash].append(asset)

        # Sort each group by created_at (oldest first)
        for hash_value in hash_groups:
            hash_groups[hash_value].sort(key=lambda a: a.created_at)

        # Create jobs with deduplication logic
        jobs_created = 0
        unique_count = 0
        skipped_count = 0

        for hash_value, group_assets in hash_groups.items():
            if hash_value is None or len(group_assets) == 1:
                # No hash or single asset: create PENDING job
                for asset in group_assets:
                    job = TrainingJob(
                        session_id=session_id,
                        asset_id=asset.id,
                        status=JobStatus.PENDING.value,
                        progress=0,
                        image_path=asset.path,
                    )
                    db.add(job)
                    jobs_created += 1
                    unique_count += 1
            else:
                # Multiple assets with same hash: representative + duplicates
                representative = group_assets[0]  # Oldest asset
                duplicates = group_assets[1:]

                # Create PENDING job for representative
                representative_job = TrainingJob(
                    session_id=session_id,
                    asset_id=representative.id,
                    status=JobStatus.PENDING.value,
                    progress=0,
                    image_path=representative.path,
                )
                db.add(representative_job)
                jobs_created += 1
                unique_count += 1

                # Create SKIPPED jobs for duplicates
                for duplicate in duplicates:
                    duplicate_job = TrainingJob(
                        session_id=session_id,
                        asset_id=duplicate.id,
                        status=JobStatus.SKIPPED.value,
                        progress=100,  # Mark as complete
                        image_path=duplicate.path,
                        skip_reason=f"Duplicate of asset {representative.id}",
                    )
                    db.add(duplicate_job)
                    jobs_created += 1
                    skipped_count += 1

        # Update session skipped_images counter
        session_query = select(TrainingSession).where(TrainingSession.id == session_id)
        session_result = await db.execute(session_query)
        session = session_result.scalar_one_or_none()

        if session:
            session.skipped_images += skipped_count

        await db.commit()

        logger.info(
            f"Created {jobs_created} training jobs for session {session_id}: "
            f"{unique_count} unique, {skipped_count} skipped duplicates "
            f"({len(existing_asset_ids)} already existed)"
        )

        return {
            "jobs_created": jobs_created,
            "unique": unique_count,
            "skipped": skipped_count,
        }

    async def update_job_status(
        self,
        db: AsyncSession,
        job_id: int,
        status: JobStatus,
        error: str | None = None,
    ) -> None:
        """Update training job status.

        Args:
            db: Database session
            job_id: Training job ID
            status: New job status
            error: Optional error message
        """
        query = select(TrainingJob).where(TrainingJob.id == job_id)
        result = await db.execute(query)
        job = result.scalar_one_or_none()

        if not job:
            logger.warning(f"Training job {job_id} not found")
            return

        job.status = status.value

        if status == JobStatus.RUNNING:
            job.started_at = datetime.now(UTC)
        elif status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            job.completed_at = datetime.now(UTC)

        if error:
            job.error_message = error

        await db.commit()

        logger.debug(f"Updated training job {job_id} to status {status.value}")

    async def get_pending_assets(self, db: AsyncSession, session_id: int) -> list[int]:
        """Get asset IDs that have not been successfully trained yet.

        Args:
            db: Database session
            session_id: Training session ID

        Returns:
            List of asset IDs with pending or failed jobs
        """
        query = (
            select(TrainingJob.asset_id)
            .where(TrainingJob.session_id == session_id)
            .where(
                TrainingJob.status.in_(
                    [JobStatus.PENDING.value, JobStatus.FAILED.value]
                )
            )
        )
        result = await db.execute(query)
        return list(result.scalars().all())

    async def start_training(
        self, db: AsyncSession, session_id: int
    ) -> TrainingSession:
        """Start or resume training for a session.

        Valid state transitions:
        - pending → running (initial start, discovers assets)
        - paused → running (resume existing jobs)
        - failed → running (resume existing jobs)

        Args:
            db: Database session
            session_id: Training session ID

        Returns:
            Updated training session

        Raises:
            ValueError: If session not found or in invalid state
        """
        session = await self.get_session(db, session_id)
        if not session:
            raise ValueError(f"Training session {session_id} not found")

        # Validate state transition
        valid_states = [
            SessionStatus.PENDING.value,
            SessionStatus.PAUSED.value,
            SessionStatus.FAILED.value,
        ]
        if session.status not in valid_states:
            raise ValueError(
                f"Cannot start training from state '{session.status}'. "
                f"Valid states: {valid_states}"
            )

        # Check if session already running
        if session.status == SessionStatus.RUNNING.value:
            logger.warning(f"Session {session_id} already running")
            return session

        # PENDING state: discover assets and create jobs
        if session.status == SessionStatus.PENDING.value:
            rq_job_id = await self.enqueue_training(db, session_id)
            logger.info(
                f"Started new training for session {session_id}, RQ job: {rq_job_id}"
            )
            # Session is already refreshed by enqueue_training
            return await self.get_session(db, session_id) or session

        # PAUSED or FAILED state: resume with existing jobs
        # Clear paused_at when resuming from paused
        if session.status == SessionStatus.PAUSED.value:
            session.paused_at = None

        # Set started_at if not set (failed before first start)
        if not session.started_at:
            session.started_at = datetime.now(UTC)

        # Update status to running
        session.status = SessionStatus.RUNNING.value

        await db.commit()
        await db.refresh(session)

        # Enqueue RQ job to process existing pending jobs
        from image_search_service.queue.training_jobs import train_session

        queue = get_queue(QUEUE_HIGH)
        rq_job = queue.enqueue(train_session, session_id, job_timeout="1h")

        logger.info(
            f"Resumed training for session {session_id}, RQ job: {rq_job.id}"
        )
        return session

    async def pause_training(
        self, db: AsyncSession, session_id: int
    ) -> TrainingSession:
        """Pause a running training session.

        Args:
            db: Database session
            session_id: Training session ID

        Returns:
            Updated training session

        Raises:
            ValueError: If session not found or not running
        """
        session = await self.get_session(db, session_id)
        if not session:
            raise ValueError(f"Training session {session_id} not found")

        # Validate state
        if session.status != SessionStatus.RUNNING.value:
            raise ValueError(
                f"Cannot pause session in state '{session.status}'. "
                "Only running sessions can be paused."
            )

        # Update status
        session.status = SessionStatus.PAUSED.value
        session.paused_at = datetime.now(UTC)

        await db.commit()
        await db.refresh(session)

        logger.info(f"Paused training session {session_id}")
        return session

    async def cancel_training(
        self, db: AsyncSession, session_id: int
    ) -> TrainingSession:
        """Cancel a training session.

        Cancels all pending jobs and updates session status.

        Args:
            db: Database session
            session_id: Training session ID

        Returns:
            Updated training session with jobs cancelled count

        Raises:
            ValueError: If session not found or in invalid state
        """
        session = await self.get_session(db, session_id)
        if not session:
            raise ValueError(f"Training session {session_id} not found")

        # Validate state
        valid_states = [SessionStatus.RUNNING.value, SessionStatus.PAUSED.value]
        if session.status not in valid_states:
            raise ValueError(
                f"Cannot cancel session in state '{session.status}'. "
                f"Valid states: {valid_states}"
            )

        # Cancel all pending jobs
        jobs_cancelled = await self.cancel_pending_jobs(db, session_id)

        # Update session status
        session.status = SessionStatus.CANCELLED.value

        await db.commit()
        await db.refresh(session)

        logger.info(f"Cancelled training session {session_id}, {jobs_cancelled} jobs cancelled")
        return session

    async def restart_training(
        self, db: AsyncSession, session_id: int, failed_only: bool = True
    ) -> TrainingSession:
        """Restart training for failed or all images.

        Args:
            db: Database session
            session_id: Training session ID
            failed_only: If True, only restart failed jobs. If False, restart all jobs.

        Returns:
            Updated training session

        Raises:
            ValueError: If session not found
        """
        session = await self.get_session(db, session_id)
        if not session:
            raise ValueError(f"Training session {session_id} not found")

        # Reset jobs based on failed_only flag
        if failed_only:
            jobs_reset = await self.reset_failed_jobs(db, session_id)
            logger.info(f"Resetting {jobs_reset} failed jobs for session {session_id}")
        else:
            jobs_reset = await self.reset_all_jobs(db, session_id)
            logger.info(f"Resetting all {jobs_reset} jobs for session {session_id}")

        # Update session counters
        session.processed_images = 0
        session.failed_images = 0
        session.status = SessionStatus.PENDING.value
        session.started_at = None
        session.completed_at = None
        session.paused_at = None

        await db.commit()
        await db.refresh(session)

        # Start training
        return await self.start_training(db, session_id)

    async def list_jobs(
        self,
        db: AsyncSession,
        session_id: int,
        page: int,
        page_size: int,
        status: JobStatus | None = None,
    ) -> tuple[list[TrainingJob], int]:
        """List training jobs for a session with pagination and filtering.

        Args:
            db: Database session
            session_id: Training session ID
            page: Page number (1-indexed)
            page_size: Number of items per page
            status: Optional status filter

        Returns:
            Tuple of (jobs list, total count)
        """
        query = select(TrainingJob).where(TrainingJob.session_id == session_id)
        count_query = select(func.count(TrainingJob.id)).where(
            TrainingJob.session_id == session_id
        )

        # Apply status filter
        if status:
            query = query.where(TrainingJob.status == status.value)
            count_query = count_query.where(TrainingJob.status == status.value)

        # Get total count
        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0

        # Apply pagination and ordering
        offset = (page - 1) * page_size
        query = (
            query.offset(offset)
            .limit(page_size)
            .order_by(TrainingJob.created_at.desc())
        )

        result = await db.execute(query)
        jobs = list(result.scalars().all())

        return jobs, total

    async def get_job(self, db: AsyncSession, job_id: int) -> TrainingJob | None:
        """Get a training job by ID.

        Args:
            db: Database session
            job_id: Training job ID

        Returns:
            Training job or None if not found
        """
        query = select(TrainingJob).where(TrainingJob.id == job_id)
        result = await db.execute(query)
        return result.scalar_one_or_none()

    async def cancel_pending_jobs(self, db: AsyncSession, session_id: int) -> int:
        """Cancel all pending jobs for a session.

        Args:
            db: Database session
            session_id: Training session ID

        Returns:
            Number of jobs cancelled
        """
        query = (
            select(TrainingJob)
            .where(TrainingJob.session_id == session_id)
            .where(TrainingJob.status == JobStatus.PENDING.value)
        )
        result = await db.execute(query)
        pending_jobs = list(result.scalars().all())

        for job in pending_jobs:
            job.status = JobStatus.CANCELLED.value
            job.completed_at = datetime.now(UTC)

        await db.commit()

        logger.info(f"Cancelled {len(pending_jobs)} pending jobs for session {session_id}")
        return len(pending_jobs)

    async def reset_failed_jobs(self, db: AsyncSession, session_id: int) -> int:
        """Reset failed jobs to pending status.

        Args:
            db: Database session
            session_id: Training session ID

        Returns:
            Number of jobs reset
        """
        query = (
            select(TrainingJob)
            .where(TrainingJob.session_id == session_id)
            .where(TrainingJob.status == JobStatus.FAILED.value)
        )
        result = await db.execute(query)
        failed_jobs = list(result.scalars().all())

        for job in failed_jobs:
            job.status = JobStatus.PENDING.value
            job.progress = 0
            job.error_message = None
            job.started_at = None
            job.completed_at = None

        await db.commit()

        logger.info(f"Reset {len(failed_jobs)} failed jobs for session {session_id}")
        return len(failed_jobs)

    async def reset_all_jobs(self, db: AsyncSession, session_id: int) -> int:
        """Reset all jobs to pending status for full re-train.

        Args:
            db: Database session
            session_id: Training session ID

        Returns:
            Number of jobs reset
        """
        query = select(TrainingJob).where(TrainingJob.session_id == session_id)
        result = await db.execute(query)
        all_jobs = list(result.scalars().all())

        for job in all_jobs:
            job.status = JobStatus.PENDING.value
            job.progress = 0
            job.error_message = None
            job.started_at = None
            job.completed_at = None
            job.processing_time_ms = None

        await db.commit()

        logger.info(f"Reset all {len(all_jobs)} jobs for session {session_id}")
        return len(all_jobs)

    async def enrich_with_training_status(
        self, db: AsyncSession, subdirs: list[DirectoryInfo], root_path: str
    ) -> list[DirectoryInfo]:
        """Enrich directory list with training status metadata.

        Queries the training_subdirectories table to calculate trained counts
        and training status for each subdirectory across all sessions.

        Args:
            db: Database session
            subdirs: List of DirectoryInfo objects to enrich
            root_path: Root path being scanned (unused, kept for compatibility)

        Returns:
            Enriched list of DirectoryInfo objects with training status
        """
        if not subdirs:
            return subdirs

        # Collect all subdirectory paths to check
        paths_to_check = [subdir.path for subdir in subdirs]

        # Query TrainingSubdirectory.path directly, aggregate across all sessions
        # Join with TrainingSession to check for in-progress status
        # Note: We still query ALL records (even trained_count=0) to check for in_progress status
        query = (
            select(
                TrainingSubdirectory.path,
                func.sum(TrainingSubdirectory.trained_count).label("total_trained"),
                func.sum(TrainingSubdirectory.image_count).label("total_images"),
                func.max(TrainingSubdirectory.created_at).label("last_trained_at"),
                # Check if any linked session is currently running
                # Use MAX(CASE...) for SQLite compatibility (bool_or is PostgreSQL-only)
                func.max(
                    func.cast(
                        TrainingSession.status == SessionStatus.RUNNING.value,
                        Integer,
                    )
                ).label("is_training"),
            )
            .join(TrainingSession)
            .where(TrainingSubdirectory.path.in_(paths_to_check))
            .group_by(TrainingSubdirectory.path)
        )

        result = await db.execute(query)
        trained_records = result.all()

        # Build lookup map: path -> training metadata
        training_map: dict[str, tuple[int, datetime | None, bool]] = {}

        for record in trained_records:
            training_map[record.path] = (
                record.total_trained or 0,
                record.last_trained_at,
                # is_training is 1 if any session is running, 0 otherwise (from MAX(CAST(...)))
                bool(record.is_training),
            )

        # Enrich subdirectories with training status
        for subdir in subdirs:
            if subdir.path in training_map:
                total_trained, last_trained_at, is_training = training_map[subdir.path]
                subdir.trained_count = total_trained

                # Calculate training status
                # Priority: in_progress > complete > partial > never
                if is_training:
                    subdir.training_status = "in_progress"
                    # For in_progress, set last_trained_at even if trained_count is 0
                    subdir.last_trained_at = last_trained_at
                elif total_trained == 0:
                    # Has database record but no training completed
                    subdir.training_status = "never"
                    subdir.last_trained_at = None
                elif total_trained >= subdir.image_count:
                    subdir.training_status = "complete"
                    subdir.last_trained_at = last_trained_at
                else:
                    subdir.training_status = "partial"
                    subdir.last_trained_at = last_trained_at
            else:
                # Never trained (no database record)
                subdir.trained_count = 0
                subdir.last_trained_at = None
                subdir.training_status = "never"

        logger.debug(
            f"Enriched {len(subdirs)} subdirectories with training status "
            f"({len(training_map)} have training records)"
        )

        return subdirs
