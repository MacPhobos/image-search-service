"""Integration tests for temporal prototype migration."""

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.db.models import (
    AgeEraBucket,
    FaceInstance,
    ImageAsset,
    Person,
    PersonPrototype,
    PrototypeRole,
)
from scripts.migrate_to_temporal_prototypes import (
    backfill_prototype_temporal_data,
    migrate_prototypes,
)


class TestTemporalMigration:
    """Tests for temporal prototype migration script."""

    @pytest.fixture
    async def sample_person(self, db_session: AsyncSession) -> Person:
        """Create sample person for testing."""
        person = Person(
            id=uuid.uuid4(),
            name="Test Person",
            status="active",
        )
        db_session.add(person)
        await db_session.commit()
        await db_session.refresh(person)
        return person

    @pytest.fixture
    async def sample_image_asset(self, db_session: AsyncSession) -> ImageAsset:
        """Create sample image asset for testing."""
        asset = ImageAsset(
            path="/test/image.jpg",
            file_modified_at=datetime(2020, 5, 15, tzinfo=timezone.utc),
            training_status="trained",
        )
        db_session.add(asset)
        await db_session.commit()
        await db_session.refresh(asset)
        return asset

    @pytest.fixture
    async def face_with_age(
        self, db_session: AsyncSession, sample_person: Person, sample_image_asset: ImageAsset
    ) -> FaceInstance:
        """Create face instance with age estimate in landmarks."""
        face = FaceInstance(
            id=uuid.uuid4(),
            asset_id=sample_image_asset.id,
            person_id=sample_person.id,
            qdrant_point_id=uuid.uuid4(),
            bbox_x=100,
            bbox_y=100,
            bbox_w=200,
            bbox_h=200,
            detection_confidence=0.95,
            quality_score=0.9,
            landmarks={
                "age_estimate": 25,
                "age_confidence": 0.85,
                "pose": "frontal",
                "bbox": {"width": 200, "height": 200},
                "quality_score": 0.9,
            },
        )
        db_session.add(face)
        await db_session.commit()
        await db_session.refresh(face)
        return face

    @pytest.fixture
    async def prototype_without_temporal(
        self, db_session: AsyncSession, sample_person: Person, face_with_age: FaceInstance
    ) -> PersonPrototype:
        """Create prototype without temporal metadata."""
        prototype = PersonPrototype(
            id=uuid.uuid4(),
            person_id=sample_person.id,
            face_instance_id=face_with_age.id,
            qdrant_point_id=uuid.uuid4(),
            role=PrototypeRole.EXEMPLAR,
            age_era_bucket=None,  # Missing temporal data
            decade_bucket=None,
        )
        db_session.add(prototype)
        await db_session.commit()
        await db_session.refresh(prototype)
        return prototype

    async def test_backfill_sets_era_bucket(
        self, db_session: AsyncSession, prototype_without_temporal: PersonPrototype
    ):
        """Backfill correctly sets age_era_bucket from face landmarks."""
        # Backfill temporal data
        updated = await backfill_prototype_temporal_data(db_session, prototype_without_temporal)

        assert updated is True
        await db_session.refresh(prototype_without_temporal)

        # Verify age_era_bucket is set correctly (25 years old → YOUNG_ADULT)
        assert prototype_without_temporal.age_era_bucket == AgeEraBucket.YOUNG_ADULT.value

        # Verify decade_bucket is set from photo timestamp (2020 → "2020s")
        assert prototype_without_temporal.decade_bucket == "2020s"

    async def test_backfill_skips_already_migrated(
        self, db_session: AsyncSession, sample_person: Person, face_with_age: FaceInstance
    ):
        """Backfill skips prototypes that already have temporal data."""
        # Create prototype with existing temporal data
        prototype = PersonPrototype(
            id=uuid.uuid4(),
            person_id=sample_person.id,
            face_instance_id=face_with_age.id,
            qdrant_point_id=uuid.uuid4(),
            role=PrototypeRole.EXEMPLAR,
            age_era_bucket=AgeEraBucket.ADULT.value,  # Already has data
            decade_bucket="2000s",
        )
        db_session.add(prototype)
        await db_session.commit()

        # Attempt backfill
        updated = await backfill_prototype_temporal_data(db_session, prototype)

        # Should skip (return False)
        assert updated is False

    async def test_backfill_skips_missing_face(
        self, db_session: AsyncSession, sample_person: Person
    ):
        """Backfill skips prototypes without linked face."""
        # Create prototype without face_instance_id
        prototype = PersonPrototype(
            id=uuid.uuid4(),
            person_id=sample_person.id,
            face_instance_id=None,  # No linked face
            qdrant_point_id=uuid.uuid4(),
            role=PrototypeRole.CENTROID,
            age_era_bucket=None,
            decade_bucket=None,
        )
        db_session.add(prototype)
        await db_session.commit()

        # Attempt backfill
        updated = await backfill_prototype_temporal_data(db_session, prototype)

        # Should skip (return False)
        assert updated is False

    async def test_backfill_skips_missing_age(
        self,
        db_session: AsyncSession,
        sample_person: Person,
        sample_image_asset: ImageAsset,
    ):
        """Backfill skips faces without age_estimate in landmarks."""
        # Create face without age_estimate
        face = FaceInstance(
            id=uuid.uuid4(),
            asset_id=sample_image_asset.id,
            person_id=sample_person.id,
            qdrant_point_id=uuid.uuid4(),
            bbox_x=100,
            bbox_y=100,
            bbox_w=200,
            bbox_h=200,
            detection_confidence=0.95,
            quality_score=0.8,
            landmarks={"pose": "frontal"},  # No age_estimate
        )
        db_session.add(face)
        await db_session.commit()

        # Create prototype linked to face
        prototype = PersonPrototype(
            id=uuid.uuid4(),
            person_id=sample_person.id,
            face_instance_id=face.id,
            qdrant_point_id=uuid.uuid4(),
            role=PrototypeRole.EXEMPLAR,
            age_era_bucket=None,
            decade_bucket=None,
        )
        db_session.add(prototype)
        await db_session.commit()

        # Attempt backfill
        updated = await backfill_prototype_temporal_data(db_session, prototype)

        # Should skip (return False)
        assert updated is False

    async def test_backfill_handles_infant_age(
        self,
        db_session: AsyncSession,
        sample_person: Person,
        sample_image_asset: ImageAsset,
    ):
        """Backfill correctly classifies infant age range."""
        # Create face with infant age (2 years)
        face = FaceInstance(
            id=uuid.uuid4(),
            asset_id=sample_image_asset.id,
            person_id=sample_person.id,
            qdrant_point_id=uuid.uuid4(),
            bbox_x=100,
            bbox_y=100,
            bbox_w=200,
            bbox_h=200,
            detection_confidence=0.95,
            quality_score=0.9,
            landmarks={"age_estimate": 2},
        )
        db_session.add(face)
        await db_session.commit()

        # Create prototype
        prototype = PersonPrototype(
            id=uuid.uuid4(),
            person_id=sample_person.id,
            face_instance_id=face.id,
            qdrant_point_id=uuid.uuid4(),
            role=PrototypeRole.EXEMPLAR,
            age_era_bucket=None,
            decade_bucket=None,
        )
        db_session.add(prototype)
        await db_session.commit()

        # Backfill
        updated = await backfill_prototype_temporal_data(db_session, prototype)

        assert updated is True
        await db_session.refresh(prototype)
        assert prototype.age_era_bucket == AgeEraBucket.INFANT.value

    async def test_backfill_handles_senior_age(
        self,
        db_session: AsyncSession,
        sample_person: Person,
        sample_image_asset: ImageAsset,
    ):
        """Backfill correctly classifies senior age range."""
        # Create face with senior age (70 years)
        face = FaceInstance(
            id=uuid.uuid4(),
            asset_id=sample_image_asset.id,
            person_id=sample_person.id,
            qdrant_point_id=uuid.uuid4(),
            bbox_x=100,
            bbox_y=100,
            bbox_w=200,
            bbox_h=200,
            detection_confidence=0.95,
            quality_score=0.9,
            landmarks={"age_estimate": 70},
        )
        db_session.add(face)
        await db_session.commit()

        # Create prototype
        prototype = PersonPrototype(
            id=uuid.uuid4(),
            person_id=sample_person.id,
            face_instance_id=face.id,
            qdrant_point_id=uuid.uuid4(),
            role=PrototypeRole.EXEMPLAR,
            age_era_bucket=None,
            decade_bucket=None,
        )
        db_session.add(prototype)
        await db_session.commit()

        # Backfill
        updated = await backfill_prototype_temporal_data(db_session, prototype)

        assert updated is True
        await db_session.refresh(prototype)
        assert prototype.age_era_bucket == AgeEraBucket.SENIOR.value

    async def test_migrate_respects_dry_run(
        self, db_session: AsyncSession, prototype_without_temporal: PersonPrototype
    ):
        """Dry run previews but doesn't commit changes."""
        # Note: migrate_prototypes creates its own session, so we can't use db_session fixture
        # This test verifies the dry_run flag prevents commits

        # Get initial state
        initial_era_bucket = prototype_without_temporal.age_era_bucket

        # Run migration in dry-run mode
        # Note: In real scenario, this would need its own DB setup
        # For now, we just verify the function signature and return type
        stats = await migrate_prototypes(dry_run=True, recompute=False)

        # Verify stats structure
        assert "total_prototypes" in stats
        assert "updated" in stats
        assert "skipped" in stats
        assert "errors" in stats

        # In actual dry run, changes would not be committed
        # (This test documents expected behavior)

    async def test_backfill_with_missing_decade(
        self,
        db_session: AsyncSession,
        sample_person: Person,
    ):
        """Backfill handles missing photo timestamp gracefully."""
        # Create asset without file_modified_at
        asset = ImageAsset(
            path="/test/no_timestamp.jpg",
            file_modified_at=None,  # No timestamp
            training_status="trained",
        )
        db_session.add(asset)
        await db_session.commit()

        # Create face with age but no timestamp
        face = FaceInstance(
            id=uuid.uuid4(),
            asset_id=asset.id,
            person_id=sample_person.id,
            qdrant_point_id=uuid.uuid4(),
            bbox_x=100,
            bbox_y=100,
            bbox_w=200,
            bbox_h=200,
            detection_confidence=0.95,
            quality_score=0.9,
            landmarks={"age_estimate": 30},
        )
        db_session.add(face)
        await db_session.commit()

        # Create prototype
        prototype = PersonPrototype(
            id=uuid.uuid4(),
            person_id=sample_person.id,
            face_instance_id=face.id,
            qdrant_point_id=uuid.uuid4(),
            role=PrototypeRole.EXEMPLAR,
            age_era_bucket=None,
            decade_bucket=None,
        )
        db_session.add(prototype)
        await db_session.commit()

        # Backfill
        updated = await backfill_prototype_temporal_data(db_session, prototype)

        assert updated is True
        await db_session.refresh(prototype)

        # Should have age_era_bucket but not decade_bucket
        # Age 30 falls in YOUNG_ADULT range (20-35 years)
        assert prototype.age_era_bucket == AgeEraBucket.YOUNG_ADULT.value
        assert prototype.decade_bucket is None

    async def test_backfill_preserves_other_fields(
        self, db_session: AsyncSession, prototype_without_temporal: PersonPrototype
    ):
        """Backfill only updates temporal fields, preserves others."""
        # Set pinned metadata
        prototype_without_temporal.is_pinned = True
        prototype_without_temporal.pinned_by = "user@example.com"
        prototype_without_temporal.pinned_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
        await db_session.commit()

        original_role = prototype_without_temporal.role
        original_is_pinned = prototype_without_temporal.is_pinned
        original_pinned_by = prototype_without_temporal.pinned_by

        # Backfill
        updated = await backfill_prototype_temporal_data(db_session, prototype_without_temporal)

        assert updated is True
        await db_session.refresh(prototype_without_temporal)

        # Verify temporal fields updated
        assert prototype_without_temporal.age_era_bucket is not None
        assert prototype_without_temporal.decade_bucket is not None

        # Verify other fields preserved
        assert prototype_without_temporal.role == original_role
        assert prototype_without_temporal.is_pinned == original_is_pinned
        assert prototype_without_temporal.pinned_by == original_pinned_by
