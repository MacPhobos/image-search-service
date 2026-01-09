#!/usr/bin/env python3
"""Backfill EXIF metadata for existing image assets.

This script extracts EXIF metadata (taken_at, camera info, GPS, etc.) from image files
for assets that don't have it yet (where taken_at IS NULL).

Usage:
    python scripts/backfill_exif.py [--limit N] [--batch-size N] [--dry-run]

Arguments:
    --limit N       Process max N images (default: unlimited)
    --batch-size N  Commit every N assets (default: 100)
    --dry-run       Show what would be done without committing
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

from image_search_service.core.logging import configure_logging, get_logger  # noqa: E402
from image_search_service.db.models import ImageAsset  # noqa: E402
from image_search_service.db.session import get_async_session_context  # noqa: E402
from image_search_service.services.exif_service import get_exif_service  # noqa: E402

logger = get_logger(__name__)


async def backfill_exif_metadata(
    limit: int | None = None,
    batch_size: int = 100,
    dry_run: bool = False,
) -> dict[str, int]:
    """Backfill EXIF metadata for assets without taken_at date.

    Queries assets where taken_at IS NULL, extracts EXIF data from image files,
    and updates database with metadata in batches.

    Args:
        limit: Maximum number of assets to process (None = unlimited)
        batch_size: Commit every N assets (prevents memory issues)
        dry_run: If True, show what would be done without committing

    Returns:
        dict with counts: {'processed': N, 'updated': N, 'skipped': N, 'failed': N}
    """
    logger.info("Starting EXIF backfill...")
    logger.info(f"Limit: {limit or 'unlimited'}, Batch size: {batch_size}, Dry run: {dry_run}")

    stats = {"processed": 0, "updated": 0, "skipped": 0, "failed": 0}
    exif_service = get_exif_service()

    async with get_async_session_context() as db:
        try:
            # Query assets without EXIF data (taken_at IS NULL)
            query = select(ImageAsset).where(ImageAsset.taken_at.is_(None))

            if limit:
                query = query.limit(limit)

            result = await db.execute(query)
            assets = result.scalars().all()

            if not assets:
                logger.info("No assets found needing EXIF extraction")
                return stats

            total_assets = len(assets)
            logger.info(f"Found {total_assets} assets needing EXIF extraction")

            # Process in batches for memory efficiency
            batch_count = (total_assets + batch_size - 1) // batch_size

            for batch_idx in range(batch_count):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, total_assets)
                batch_assets = assets[batch_start:batch_end]

                batch_updated = 0
                batch_skipped = 0
                batch_failed = 0

                for asset in batch_assets:
                    stats["processed"] += 1

                    try:
                        # Validate file exists
                        asset_path = Path(asset.path)
                        if not asset_path.exists():
                            logger.warning(f"Asset {asset.id}: file not found at {asset.path}")
                            stats["failed"] += 1
                            batch_failed += 1
                            continue

                        # Extract EXIF metadata
                        exif_data = exif_service.extract_exif(str(asset_path))

                        # Check if any useful EXIF data was found
                        has_data = any(
                            [
                                exif_data.get("taken_at") is not None,
                                exif_data.get("camera_make") is not None,
                                exif_data.get("camera_model") is not None,
                                exif_data.get("gps_latitude") is not None,
                                exif_data.get("gps_longitude") is not None,
                                bool(exif_data.get("exif_metadata")),
                            ]
                        )

                        if not has_data:
                            logger.debug(f"Asset {asset.id}: no EXIF data found")
                            stats["skipped"] += 1
                            batch_skipped += 1
                            continue

                        # Update asset with EXIF metadata
                        if not dry_run:
                            asset.taken_at = exif_data.get("taken_at")
                            asset.camera_make = exif_data.get("camera_make")
                            asset.camera_model = exif_data.get("camera_model")
                            asset.gps_latitude = exif_data.get("gps_latitude")
                            asset.gps_longitude = exif_data.get("gps_longitude")
                            asset.exif_metadata = exif_data.get("exif_metadata")

                        stats["updated"] += 1
                        batch_updated += 1

                        # Log details for dry run or debug
                        if dry_run or logger.level <= logging.DEBUG:
                            taken_str = (
                                exif_data["taken_at"].isoformat()
                                if exif_data.get("taken_at")
                                else "N/A"
                            )
                            camera_str = (
                                " + ".join(
                                    filter(
                                        None,
                                        [
                                            exif_data.get("camera_make"),
                                            exif_data.get("camera_model"),
                                        ],
                                    )
                                )
                                or "N/A"
                            )
                            lat = exif_data.get("gps_latitude")
                            lon = exif_data.get("gps_longitude")
                            gps_str = f"({lat:.6f}, {lon:.6f})" if lat and lon else "N/A"
                            log_prefix = "[DRY RUN] Would update" if dry_run else "Updated"
                            logger.info(
                                f"{log_prefix} asset {asset.id}: taken={taken_str}, "
                                f"camera={camera_str}, gps={gps_str}"
                            )

                    except Exception as e:
                        logger.error(f"Asset {asset.id}: failed to extract EXIF: {e}")
                        stats["failed"] += 1
                        batch_failed += 1
                        continue

                # Commit batch (unless dry run)
                if not dry_run and (batch_updated > 0):
                    try:
                        await db.commit()
                        logger.debug(f"Committed batch {batch_idx + 1}/{batch_count}")
                    except Exception as e:
                        logger.error(f"Failed to commit batch {batch_idx + 1}: {e}")
                        await db.rollback()
                        # Count failed commits as failures
                        stats["failed"] += batch_updated
                        stats["updated"] -= batch_updated
                        continue

                # Progress logging
                logger.info(
                    f"Processing batch {batch_idx + 1}/{batch_count}... "
                    f"({batch_updated} assets updated, {batch_skipped} skipped, "
                    f"{batch_failed} failed)"
                )

        except Exception as e:
            logger.error(f"Backfill failed: {e}", exc_info=True)
            stats["failed"] += 1

    logger.info(
        f"Done! Processed: {stats['processed']}, Updated: {stats['updated']}, "
        f"Skipped: {stats['skipped']}, Failed: {stats['failed']}"
    )
    return stats


async def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Backfill EXIF metadata for existing image assets")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process max N images (default: unlimited)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Commit every N assets (default: 100)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without committing",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set log level (default: INFO)",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    configure_logging()

    try:
        result = await backfill_exif_metadata(
            limit=args.limit,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )

        # Print summary
        print("\n" + "=" * 60)
        print("EXIF BACKFILL SUMMARY")
        print("=" * 60)
        print(f"Processed: {result['processed']}")
        print(f"Updated:   {result['updated']}")
        print(f"Skipped:   {result['skipped']}")
        print(f"Failed:    {result['failed']}")
        print("=" * 60)

        # Return exit code based on errors
        return 1 if result["failed"] > 0 else 0

    except Exception as e:
        logger.error(f"Backfill failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
