"""Tests for training deduplication logic."""

from pathlib import Path

import pytest
from PIL import Image
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.db.models import ImageAsset, JobStatus, TrainingJob, TrainingSession
from image_search_service.services.perceptual_hash import compute_perceptual_hash
from image_search_service.services.training_service import TrainingService


@pytest.fixture
async def training_service() -> TrainingService:
    """Create training service instance."""
    return TrainingService()


@pytest.fixture
async def sample_images(tmp_path: Path) -> list[Path]:
    """Create sample test images with distinct visual patterns.

    Returns:
        List of paths: [original, duplicate, different]
        - original: horizontal gradient (white→black)
        - duplicate: copy of original (same hash)
        - different: vertical gradient (white→black, different hash)
    """
    import numpy as np

    # Create original image with horizontal gradient
    # Gradient left (white) to right (black)
    gradient_h = np.linspace(255, 0, 100, dtype=np.uint8)
    gradient_array = np.tile(gradient_h, (100, 1))  # Repeat for each row
    original = Image.fromarray(gradient_array, mode="L").convert("RGB")
    original_path = tmp_path / "original.jpg"
    original.save(original_path)

    # Create duplicate (same content as original)
    duplicate_path = tmp_path / "duplicate.jpg"
    original.save(duplicate_path)

    # Create different image with vertical gradient
    # Gradient top (white) to bottom (black)
    gradient_v = np.linspace(255, 0, 100, dtype=np.uint8)
    gradient_array_v = np.tile(gradient_v[:, np.newaxis], (1, 100))  # Repeat for each column
    different = Image.fromarray(gradient_array_v, mode="L").convert("RGB")
    different_path = tmp_path / "different.jpg"
    different.save(different_path)

    return [original_path, duplicate_path, different_path]


class TestTrainingDeduplication:
    """Test training job deduplication with perceptual hashes."""

    async def test_creates_pending_job_for_unique_images(
        self,
        db_session: AsyncSession,
        training_service: TrainingService,
        sample_images: list[Path],
    ) -> None:
        """Test that unique images get PENDING jobs."""
        # Create training session
        session = TrainingSession(
            name="Test Session",
            root_path=str(sample_images[0].parent),
            status="pending",
        )
        db_session.add(session)
        await db_session.commit()
        await db_session.refresh(session)

        # Create assets for original and different images
        asset1 = ImageAsset(path=str(sample_images[0]))  # Original
        asset2 = ImageAsset(path=str(sample_images[2]))  # Different
        db_session.add_all([asset1, asset2])
        await db_session.commit()
        await db_session.refresh(asset1)
        await db_session.refresh(asset2)

        # Create training jobs
        result = await training_service.create_training_jobs(
            db_session, session.id, [asset1.id, asset2.id]
        )

        # Both should be unique
        assert result["jobs_created"] == 2
        assert result["unique"] == 2
        assert result["skipped"] == 0

        # Verify jobs in database
        jobs_query = select(TrainingJob).where(TrainingJob.session_id == session.id)
        jobs_result = await db_session.execute(jobs_query)
        jobs = list(jobs_result.scalars().all())

        assert len(jobs) == 2
        assert all(job.status == JobStatus.PENDING.value for job in jobs)
        assert all(job.image_path is not None for job in jobs)
        assert all(job.skip_reason is None for job in jobs)

    async def test_creates_skipped_job_for_duplicates(
        self,
        db_session: AsyncSession,
        training_service: TrainingService,
        sample_images: list[Path],
    ) -> None:
        """Test that duplicate images get SKIPPED jobs."""
        # Create training session
        session = TrainingSession(
            name="Test Session",
            root_path=str(sample_images[0].parent),
            status="pending",
        )
        db_session.add(session)
        await db_session.commit()
        await db_session.refresh(session)

        # Create assets for original and duplicate
        asset1 = ImageAsset(path=str(sample_images[0]))  # Original
        asset2 = ImageAsset(path=str(sample_images[1]))  # Duplicate
        db_session.add_all([asset1, asset2])
        await db_session.commit()
        await db_session.refresh(asset1)
        await db_session.refresh(asset2)

        # Create training jobs
        result = await training_service.create_training_jobs(
            db_session, session.id, [asset1.id, asset2.id]
        )

        # Should detect duplication
        assert result["jobs_created"] == 2
        assert result["unique"] == 1
        assert result["skipped"] == 1

        # Verify jobs in database
        jobs_query = (
            select(TrainingJob)
            .where(TrainingJob.session_id == session.id)
            .order_by(TrainingJob.asset_id)
        )
        jobs_result = await db_session.execute(jobs_query)
        jobs = list(jobs_result.scalars().all())

        assert len(jobs) == 2

        # First asset (oldest) should be PENDING
        pending_job = jobs[0]
        assert pending_job.status == JobStatus.PENDING.value
        assert pending_job.skip_reason is None
        assert pending_job.image_path == str(sample_images[0])

        # Second asset (duplicate) should be SKIPPED
        skipped_job = jobs[1]
        assert skipped_job.status == JobStatus.SKIPPED.value
        assert skipped_job.skip_reason == f"Duplicate of asset {asset1.id}"
        assert skipped_job.image_path == str(sample_images[1])
        assert skipped_job.progress == 100

    async def test_computes_missing_hashes(
        self,
        db_session: AsyncSession,
        training_service: TrainingService,
        sample_images: list[Path],
    ) -> None:
        """Test that missing perceptual hashes are computed."""
        # Create training session
        session = TrainingSession(
            name="Test Session",
            root_path=str(sample_images[0].parent),
            status="pending",
        )
        db_session.add(session)
        await db_session.commit()
        await db_session.refresh(session)

        # Create asset WITHOUT hash
        asset = ImageAsset(path=str(sample_images[0]), perceptual_hash=None)
        db_session.add(asset)
        await db_session.commit()
        await db_session.refresh(asset)

        assert asset.perceptual_hash is None

        # Create training jobs (should compute hash)
        await training_service.create_training_jobs(db_session, session.id, [asset.id])

        # Verify hash was computed
        await db_session.refresh(asset)
        assert asset.perceptual_hash is not None
        assert len(asset.perceptual_hash) == 16
        assert all(c in "0123456789abcdef" for c in asset.perceptual_hash)

    async def test_updates_session_skipped_counter(
        self,
        db_session: AsyncSession,
        training_service: TrainingService,
        sample_images: list[Path],
    ) -> None:
        """Test that session.skipped_images counter is updated."""
        # Create training session
        session = TrainingSession(
            name="Test Session",
            root_path=str(sample_images[0].parent),
            status="pending",
            skipped_images=0,
        )
        db_session.add(session)
        await db_session.commit()
        await db_session.refresh(session)

        # Create duplicate assets
        asset1 = ImageAsset(path=str(sample_images[0]))
        asset2 = ImageAsset(path=str(sample_images[1]))
        db_session.add_all([asset1, asset2])
        await db_session.commit()

        # Create training jobs
        await training_service.create_training_jobs(db_session, session.id, [asset1.id, asset2.id])

        # Verify session counter updated
        await db_session.refresh(session)
        assert session.skipped_images == 1

    async def test_handles_existing_jobs(
        self,
        db_session: AsyncSession,
        training_service: TrainingService,
        sample_images: list[Path],
    ) -> None:
        """Test that existing jobs are not recreated."""
        # Create training session
        session = TrainingSession(
            name="Test Session",
            root_path=str(sample_images[0].parent),
            status="pending",
        )
        db_session.add(session)
        await db_session.commit()
        await db_session.refresh(session)

        # Create asset
        asset = ImageAsset(path=str(sample_images[0]))
        db_session.add(asset)
        await db_session.commit()
        await db_session.refresh(asset)

        # Create jobs first time
        result1 = await training_service.create_training_jobs(db_session, session.id, [asset.id])
        assert result1["jobs_created"] == 1

        # Try to create again
        result2 = await training_service.create_training_jobs(db_session, session.id, [asset.id])
        assert result2["jobs_created"] == 0

    async def test_mixed_unique_and_duplicate_images(
        self,
        db_session: AsyncSession,
        training_service: TrainingService,
        sample_images: list[Path],
    ) -> None:
        """Test handling of mixed unique and duplicate images."""
        # Create training session
        session = TrainingSession(
            name="Test Session",
            root_path=str(sample_images[0].parent),
            status="pending",
        )
        db_session.add(session)
        await db_session.commit()
        await db_session.refresh(session)

        # Create all assets: original, duplicate, different
        assets = [
            ImageAsset(path=str(sample_images[0])),  # Original
            ImageAsset(path=str(sample_images[1])),  # Duplicate of original
            ImageAsset(path=str(sample_images[2])),  # Different
        ]
        db_session.add_all(assets)
        await db_session.commit()
        for asset in assets:
            await db_session.refresh(asset)

        # Create training jobs
        result = await training_service.create_training_jobs(
            db_session, session.id, [a.id for a in assets]
        )

        # Should have 2 unique (original + different) and 1 skipped (duplicate)
        assert result["jobs_created"] == 3
        assert result["unique"] == 2
        assert result["skipped"] == 1

        # Verify job statuses
        jobs_query = (
            select(TrainingJob)
            .where(TrainingJob.session_id == session.id)
            .order_by(TrainingJob.status.desc(), TrainingJob.asset_id)
        )
        jobs_result = await db_session.execute(jobs_query)
        jobs = list(jobs_result.scalars().all())

        pending_count = sum(1 for job in jobs if job.status == JobStatus.PENDING.value)
        skipped_count = sum(1 for job in jobs if job.status == JobStatus.SKIPPED.value)

        assert pending_count == 2
        assert skipped_count == 1
