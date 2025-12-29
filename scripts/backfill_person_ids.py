#!/usr/bin/env python3
"""Backfill person_ids into existing Qdrant vector payloads.

This script updates all existing vectors to include person_ids
based on face assignments in the database.

Usage:
    python scripts/backfill_person_ids.py [--dry-run] [--batch-size N]

Arguments:
    --dry-run     Print what would be updated without making changes
    --batch-size  Number of assets to process per batch (default: 100)
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from sqlalchemy import select  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402

from image_search_service.core.logging import configure_logging, get_logger  # noqa: E402
from image_search_service.db.models import FaceInstance, ImageAsset  # noqa: E402
from image_search_service.db.session import get_async_session_context  # noqa: E402
from image_search_service.vector.qdrant import (  # noqa: E402
    get_qdrant_client,
    update_vector_payload,
)

logger = get_logger(__name__)


async def get_person_ids_for_asset(asset_id: int, db: AsyncSession) -> list[str]:
    """Get all person_ids for faces on an asset.

    Args:
        asset_id: The image asset ID
        db: Database session

    Returns:
        List of person_id strings (UUIDs converted to string)
    """
    result = await db.execute(
        select(FaceInstance.person_id)
        .where(FaceInstance.asset_id == asset_id)
        .where(FaceInstance.person_id.isnot(None))
        .distinct()
    )
    # Convert UUID objects to strings for Qdrant payload
    return [str(row[0]) for row in result.fetchall() if row[0]]


async def backfill_person_ids(
    dry_run: bool = False, batch_size: int = 100
) -> dict[str, int]:
    """Backfill person_ids for all existing vectors.

    Iterates through all assets that have been indexed (indexed_at is not null),
    queries the database for person_ids from face assignments, and updates
    the Qdrant payload with the person_ids array.

    Args:
        dry_run: If True, preview changes without committing to Qdrant
        batch_size: Number of assets to process per batch

    Returns:
        dict with counts: {'processed': N, 'updated': N, 'skipped': N, 'errors': N}
    """
    logger.info("Starting person_ids backfill")
    logger.info(f"Dry run: {dry_run}, Batch size: {batch_size}")

    stats = {"processed": 0, "updated": 0, "skipped": 0, "errors": 0}

    async with get_async_session_context() as db:
        try:
            # Get Qdrant client
            qdrant_client = get_qdrant_client()

            # Process assets in batches
            offset = 0
            while True:
                # Fetch batch of indexed assets
                result = await db.execute(
                    select(ImageAsset)
                    .where(ImageAsset.indexed_at.isnot(None))  # Only indexed assets
                    .offset(offset)
                    .limit(batch_size)
                )
                assets = result.scalars().all()

                if not assets:
                    # No more assets to process
                    break

                # Process each asset in batch
                for asset in assets:
                    stats["processed"] += 1

                    try:
                        # Get person_ids from database
                        person_ids = await get_person_ids_for_asset(asset.id, db)

                        if dry_run:
                            logger.info(
                                f"[DRY RUN] Would update asset {asset.id}: person_ids={person_ids}"
                            )
                            stats["updated"] += 1
                        else:
                            # Update Qdrant payload
                            update_vector_payload(
                                asset_id=asset.id,
                                payload_updates={"person_ids": person_ids},
                                client=qdrant_client,
                            )
                            logger.debug(f"Updated asset {asset.id}: person_ids={person_ids}")
                            stats["updated"] += 1

                    except Exception as e:
                        logger.error(f"Error updating asset {asset.id}: {e}")
                        stats["errors"] += 1

                # Progress logging after each batch
                logger.info(
                    f"Progress: {stats['processed']} processed, "
                    f"{stats['updated']} updated, {stats['errors']} errors"
                )

                # Move to next batch
                offset += batch_size

        except Exception as e:
            logger.error(f"Backfill failed: {e}")
            stats["errors"] += 1

    logger.info(f"Backfill complete: {stats}")
    return stats


async def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Backfill person_ids into Qdrant vector payloads")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be updated without making changes",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of assets to process per batch (default: 100)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set log level",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    configure_logging()

    try:
        result = await backfill_person_ids(dry_run=args.dry_run, batch_size=args.batch_size)

        # Print summary
        print("\n" + "=" * 60)
        print("BACKFILL SUMMARY")
        print("=" * 60)
        print(f"Processed: {result['processed']}")
        print(f"Updated:   {result['updated']}")
        print(f"Skipped:   {result['skipped']}")
        print(f"Errors:    {result['errors']}")
        print("=" * 60)

        # Return exit code based on errors
        return 1 if result["errors"] > 0 else 0

    except Exception as e:
        logger.error(f"Backfill failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
