#!/usr/bin/env python3
"""Rollback temporal prototype changes.

Clears temporal metadata fields from prototypes:
- Sets age_era_bucket to NULL
- Sets decade_bucket to NULL
- Preserves is_pinned and other fields

Usage:
    python scripts/rollback_temporal_prototypes.py [--dry-run]

Options:
    --dry-run     Preview changes without applying
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from sqlalchemy import select, update  # noqa: E402

from image_search_service.core.logging import configure_logging, get_logger  # noqa: E402
from image_search_service.db.models import PersonPrototype  # noqa: E402
from image_search_service.db.session import get_async_session_context  # noqa: E402

logger = get_logger(__name__)


async def rollback_temporal_metadata(dry_run: bool = False) -> dict:
    """Clear temporal metadata from all prototypes.

    Args:
        dry_run: If True, preview changes without committing

    Returns:
        {
            "total_prototypes": int,
            "cleared": int,
            "errors": int,
        }
    """
    logger.info("Starting temporal prototype rollback")
    logger.info(f"Dry run: {dry_run}")

    stats = {
        "total_prototypes": 0,
        "cleared": 0,
        "errors": 0,
    }

    async with get_async_session_context() as db:
        try:
            # Count prototypes with temporal data
            result = await db.execute(
                select(PersonPrototype).where(
                    (PersonPrototype.age_era_bucket.is_not(None))
                    | (PersonPrototype.decade_bucket.is_not(None))
                )
            )
            prototypes = result.scalars().all()
            stats["total_prototypes"] = len(prototypes)

            logger.info(f"Found {stats['total_prototypes']} prototypes with temporal data")

            if stats["total_prototypes"] == 0:
                logger.info("No prototypes to rollback")
                return stats

            # Clear temporal metadata
            await db.execute(
                update(PersonPrototype)
                .where(
                    (PersonPrototype.age_era_bucket.is_not(None))
                    | (PersonPrototype.decade_bucket.is_not(None))
                )
                .values(
                    age_era_bucket=None,
                    decade_bucket=None,
                )
            )

            stats["cleared"] = stats["total_prototypes"]

            # Commit changes if not dry run
            if not dry_run:
                await db.commit()
                logger.info(f"Cleared temporal data from {stats['cleared']} prototypes")
            else:
                await db.rollback()
                logger.info("Dry run complete, changes not committed")

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            stats["errors"] += 1
            if not dry_run:
                await db.rollback()

    logger.info(f"Rollback complete: {stats}")
    return stats


async def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Rollback temporal prototype changes")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without committing",
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

    # Confirm rollback
    if not args.dry_run:
        print("\n" + "=" * 60)
        print("WARNING: This will clear temporal metadata from all prototypes!")
        print("=" * 60)
        response = input("\nAre you sure you want to continue? (yes/no): ")
        if response.lower() != "yes":
            print("Rollback cancelled")
            return 0

    try:
        result = await rollback_temporal_metadata(dry_run=args.dry_run)

        # Print summary
        print("\n" + "=" * 60)
        print("ROLLBACK SUMMARY")
        print("=" * 60)
        print(f"Total prototypes with temporal data: {result['total_prototypes']}")
        print(f"Cleared: {result['cleared']}")
        print(f"Errors: {result['errors']}")
        print("=" * 60)

        # Return exit code based on errors
        return 1 if result["errors"] > 0 else 0

    except Exception as e:
        logger.error(f"Rollback failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
