#!/usr/bin/env python3
"""Backfill temporal metadata for existing prototypes.

This script:
1. Finds all existing PersonPrototype records without age_era_bucket
2. Looks up the linked FaceInstance's landmarks for age_estimate
3. Classifies the age into era bucket
4. Updates the prototype with temporal metadata
5. Optionally triggers recomputation for temporal diversity

Usage:
    python scripts/migrate_to_temporal_prototypes.py [--dry-run] [--recompute]

Options:
    --dry-run     Preview changes without applying
    --recompute   Trigger prototype recomputation after migration
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from uuid import UUID

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from image_search_service.core.logging import configure_logging, get_logger
from image_search_service.db.models import FaceInstance, ImageAsset, PersonPrototype
from image_search_service.db.session import get_async_session_context
from image_search_service.services.temporal_service import (
    classify_age_era,
    extract_decade_from_timestamp,
)

logger = get_logger(__name__)


async def backfill_prototype_temporal_data(
    db: AsyncSession,
    prototype: PersonPrototype,
) -> bool:
    """Backfill temporal data for a single prototype.

    1. Get linked FaceInstance
    2. Extract age_estimate from landmarks
    3. Classify into age_era_bucket
    4. Extract photo timestamp for decade_bucket
    5. Update prototype record

    Args:
        db: Database session
        prototype: PersonPrototype to update

    Returns:
        True if updated, False if skipped.
    """
    # Skip if already has temporal data
    if prototype.age_era_bucket is not None:
        logger.debug(
            f"Prototype {prototype.id} already has age_era_bucket={prototype.age_era_bucket}, skipping"
        )
        return False

    # Skip if no linked face instance
    if prototype.face_instance_id is None:
        logger.warning(f"Prototype {prototype.id} has no linked face_instance_id, skipping")
        return False

    # Fetch face instance with landmarks and asset
    face_result = await db.execute(
        select(FaceInstance)
        .options(joinedload(FaceInstance.asset))
        .where(FaceInstance.id == prototype.face_instance_id)
    )
    face = face_result.scalar_one_or_none()

    if face is None:
        logger.warning(
            f"Prototype {prototype.id} references non-existent face {prototype.face_instance_id}, skipping"
        )
        return False

    # Extract age estimate from landmarks
    age_estimate = None
    if face.landmarks and isinstance(face.landmarks, dict):
        age_estimate = face.landmarks.get("age_estimate")

    if age_estimate is None:
        logger.debug(
            f"Prototype {prototype.id} face has no age_estimate in landmarks, skipping"
        )
        return False

    # Classify age era
    age_era = classify_age_era(age_estimate)
    if age_era is None:
        logger.warning(
            f"Could not classify age {age_estimate} for prototype {prototype.id}, skipping"
        )
        return False

    # Extract decade from photo timestamp
    decade_bucket = None
    if face.asset:
        decade_bucket = extract_decade_from_timestamp(face.asset.file_modified_at)

    # Update prototype
    prototype.age_era_bucket = age_era.value
    prototype.decade_bucket = decade_bucket

    # Flush changes to database (caller is responsible for commit)
    await db.flush()

    logger.info(
        f"Updated prototype {prototype.id}: age_era={age_era.value}, decade={decade_bucket}"
    )
    return True


async def migrate_prototypes(dry_run: bool = False, recompute: bool = False) -> dict:
    """Migrate existing prototypes to temporal system.

    Args:
        dry_run: If True, preview changes without committing
        recompute: If True, trigger prototype recomputation after migration

    Returns:
        {
            "total_prototypes": int,
            "updated": int,
            "skipped": int,
            "errors": int,
            "recomputed_persons": int,
        }
    """
    logger.info("Starting temporal prototype migration")
    logger.info(f"Dry run: {dry_run}, Recompute: {recompute}")

    stats = {
        "total_prototypes": 0,
        "updated": 0,
        "skipped": 0,
        "errors": 0,
        "recomputed_persons": 0,
    }

    async with get_async_session_context() as db:
        try:
            # Find all prototypes without age_era_bucket
            result = await db.execute(
                select(PersonPrototype)
                .options(
                    joinedload(PersonPrototype.face_instance).joinedload(FaceInstance.asset)
                )
                .where(PersonPrototype.age_era_bucket.is_(None))
            )
            prototypes = result.scalars().unique().all()

            stats["total_prototypes"] = len(prototypes)
            logger.info(f"Found {stats['total_prototypes']} prototypes to migrate")

            # Process each prototype
            for prototype in prototypes:
                try:
                    updated = await backfill_prototype_temporal_data(db, prototype)
                    if updated:
                        stats["updated"] += 1
                    else:
                        stats["skipped"] += 1
                except Exception as e:
                    logger.error(f"Error processing prototype {prototype.id}: {e}")
                    stats["errors"] += 1

            # Commit changes if not dry run
            if not dry_run:
                await db.commit()
                logger.info(f"Committed {stats['updated']} updates")
            else:
                await db.rollback()
                logger.info("Dry run complete, changes not committed")

            # Optionally trigger recomputation
            if recompute and not dry_run and stats["updated"] > 0:
                logger.info("Triggering prototype recomputation (not implemented yet)")
                # TODO: Implement recomputation trigger
                # This would call prototype_service.recompute_person_prototypes()
                # for each unique person_id that had updates

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            stats["errors"] += 1
            if not dry_run:
                await db.rollback()

    logger.info(f"Migration complete: {stats}")
    return stats


async def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Migrate prototypes to temporal system")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without committing",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Trigger prototype recomputation after migration",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set log level",
    )
    args = parser.parse_args()

    # Setup logging
    import logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    configure_logging()

    try:
        result = await migrate_prototypes(dry_run=args.dry_run, recompute=args.recompute)

        # Print summary
        print("\n" + "=" * 60)
        print("MIGRATION SUMMARY")
        print("=" * 60)
        print(f"Total prototypes: {result['total_prototypes']}")
        print(f"Updated: {result['updated']}")
        print(f"Skipped: {result['skipped']}")
        print(f"Errors: {result['errors']}")
        if args.recompute:
            print(f"Recomputed persons: {result['recomputed_persons']}")
        print("=" * 60)

        # Return exit code based on errors
        return 1 if result["errors"] > 0 else 0

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
