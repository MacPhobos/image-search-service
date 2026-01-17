"""Hash backfill jobs for computing perceptual hashes on existing assets.

This module provides functionality to compute perceptual hashes for all assets
that don't already have one. This is useful after migration or when adding new
images to the database.
"""

from typing import Any

from sqlalchemy import select

from image_search_service.core.logging import get_logger
from image_search_service.db.models import ImageAsset
from image_search_service.db.session import get_async_session_context
from image_search_service.services.perceptual_hash import compute_perceptual_hash

logger = get_logger(__name__)


async def backfill_perceptual_hashes(
    batch_size: int = 500, limit: int | None = None
) -> dict[str, Any]:
    """Compute perceptual hashes for all assets without hashes.

    This function:
    1. Queries all assets with perceptual_hash IS NULL
    2. Computes hash for each asset in batches
    3. Updates database with computed hashes
    4. Returns statistics (total, processed, failed)

    Args:
        batch_size: Number of assets to process per batch (default: 500)
        limit: Maximum total images to process (None = all)

    Returns:
        Dictionary with keys:
        - status: "completed" or "partial" (if some failed)
        - total: Total assets needing hashes
        - processed: Successfully processed count
        - failed: Failed count
        - errors: List of error messages (first 10)

    Example:
        >>> result = await backfill_perceptual_hashes(batch_size=1000, limit=5000)
        >>> print(f"Processed {result['processed']}/{result['total']} assets")
    """
    total = 0
    processed = 0
    failed = 0
    errors: list[str] = []

    async with get_async_session_context() as db:
        # Query assets without hashes
        query = select(ImageAsset).where(ImageAsset.perceptual_hash.is_(None))
        if limit is not None:
            query = query.limit(limit)
        result = await db.execute(query)
        assets = result.scalars().all()

        total = len(assets)
        logger.info(
            f"Found {total} assets without perceptual hashes"
            + (f" (limited to {limit})" if limit else "")
        )

        if total == 0:
            return {
                "status": "completed",
                "total": 0,
                "processed": 0,
                "failed": 0,
                "errors": [],
            }

        # Process in batches
        for i in range(0, total, batch_size):
            batch = assets[i : i + batch_size]
            batch_start = i + 1
            batch_end = min(i + batch_size, total)

            logger.info(f"Processing batch {batch_start}-{batch_end}/{total}")

            for asset in batch:
                try:
                    # Compute hash
                    hash_value = compute_perceptual_hash(asset.path)
                    asset.perceptual_hash = hash_value
                    processed += 1

                    if processed % 100 == 0:
                        logger.debug(f"Computed {processed}/{total} hashes")

                except FileNotFoundError:
                    failed += 1
                    error_msg = f"Asset {asset.id}: File not found at {asset.path}"
                    logger.warning(error_msg)
                    if len(errors) < 10:
                        errors.append(error_msg)

                except Exception as e:
                    failed += 1
                    error_msg = f"Asset {asset.id}: {type(e).__name__}: {e}"
                    logger.error(error_msg)
                    if len(errors) < 10:
                        errors.append(error_msg)

            # Commit batch
            try:
                await db.commit()
                logger.info(f"Committed batch {batch_start}-{batch_end}")
            except Exception as e:
                logger.error(f"Failed to commit batch: {e}")
                await db.rollback()
                # Increment failed count for entire batch
                failed += len(batch) - processed
                if len(errors) < 10:
                    errors.append(f"Batch commit failed: {e}")

    # Final status
    status = "completed" if failed == 0 else "partial"

    logger.info(
        f"Hash backfill {status}: {processed} processed, {failed} failed out of {total} total"
    )

    return {
        "status": status,
        "total": total,
        "processed": processed,
        "failed": failed,
        "errors": errors,
    }


def backfill_perceptual_hashes_sync(
    batch_size: int = 500, limit: int | None = None
) -> dict[str, Any]:
    """Synchronous wrapper for backfill_perceptual_hashes.

    This function is used for CLI/Makefile integration.

    Args:
        batch_size: Number of assets to process per batch
        limit: Maximum total images to process (None = all)

    Returns:
        Same as async version
    """
    import asyncio

    return asyncio.run(backfill_perceptual_hashes(batch_size=batch_size, limit=limit))


if __name__ == "__main__":
    # Allow running as script: python -m image_search_service.queue.hash_backfill_jobs
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Backfill perceptual hashes for images without hashes."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Number of assets to process per batch (default: 500)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum total images to process (default: all)",
    )
    args = parser.parse_args()

    result = backfill_perceptual_hashes_sync(batch_size=args.batch_size, limit=args.limit)

    print("\n" + "=" * 60)
    print("HASH BACKFILL RESULTS")
    print("=" * 60)
    print(f"Status: {result['status'].upper()}")
    print(f"Total assets: {result['total']}")
    print(f"Processed: {result['processed']}")
    print(f"Failed: {result['failed']}")

    if result["errors"]:
        print("\nErrors (first 10):")
        for error in result["errors"]:
            print(f"  - {error}")

    print("=" * 60)

    # Exit with error code if any failures
    sys.exit(1 if result["failed"] > 0 else 0)
