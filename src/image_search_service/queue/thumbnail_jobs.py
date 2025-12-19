"""Thumbnail generation background jobs for RQ."""

from sqlalchemy import select

from image_search_service.core.config import get_settings
from image_search_service.core.logging import get_logger
from image_search_service.db.models import ImageAsset, TrainingJob
from image_search_service.db.sync_operations import get_sync_session
from image_search_service.services.thumbnail_service import ThumbnailService

logger = get_logger(__name__)


def generate_thumbnails_batch(session_id: int) -> dict[str, int]:
    """Background job to generate thumbnails for a training session.

    This job processes all assets in a session and generates thumbnails for each.
    Updates asset records with thumbnail path and dimensions.

    Args:
        session_id: Training session ID

    Returns:
        Dictionary with stats: {generated: X, skipped: Y, failed: Z, total: N}
    """
    logger.info(f"Starting batch thumbnail generation for session {session_id}")

    db_session = get_sync_session()
    settings = get_settings()
    thumb_service = ThumbnailService(
        thumbnail_dir=settings.thumbnail_dir, thumbnail_size=settings.thumbnail_size
    )

    stats = {"generated": 0, "skipped": 0, "failed": 0, "total": 0}

    try:
        # Get all assets for this session via training jobs
        query = (
            select(ImageAsset)
            .join(TrainingJob, TrainingJob.asset_id == ImageAsset.id)
            .where(TrainingJob.session_id == session_id)
            .distinct()
        )
        result = db_session.execute(query)
        assets = list(result.scalars().all())

        stats["total"] = len(assets)
        logger.info(f"Found {len(assets)} assets to process for session {session_id}")

        # Process each asset
        for i, asset in enumerate(assets, 1):
            try:
                # Skip if thumbnail already exists
                if thumb_service.thumbnail_exists(asset.id, asset.path):
                    logger.debug(f"Thumbnail already exists for asset {asset.id}")
                    stats["skipped"] += 1
                    continue

                # Generate thumbnail
                thumb_path, width, height = thumb_service.generate_thumbnail(
                    asset.path, asset.id
                )

                # Update asset with thumbnail info
                asset.thumbnail_path = thumb_path
                asset.width = width
                asset.height = height
                db_session.commit()

                stats["generated"] += 1

                # Log progress every 10 assets
                if i % 10 == 0:
                    logger.info(
                        f"Thumbnail generation progress for session {session_id}: "
                        f"{i}/{len(assets)} ({i * 100 // len(assets)}%)"
                    )

            except FileNotFoundError:
                logger.warning(
                    f"Original image not found for asset {asset.id}: {asset.path}"
                )
                stats["failed"] += 1
            except Exception as e:
                logger.error(f"Failed to generate thumbnail for asset {asset.id}: {e}")
                stats["failed"] += 1

        logger.info(
            f"Completed batch thumbnail generation for session {session_id}: "
            f"{stats['generated']} generated, {stats['skipped']} skipped, "
            f"{stats['failed']} failed out of {stats['total']} total"
        )

        return stats

    except Exception as e:
        logger.exception(f"Error in batch thumbnail generation for session {session_id}: {e}")
        raise
    finally:
        db_session.close()


def generate_single_thumbnail(asset_id: int) -> dict[str, object]:
    """Generate thumbnail for a single asset.

    This job can be enqueued for individual thumbnail generation.

    Args:
        asset_id: Image asset ID

    Returns:
        Dictionary with result info
    """
    logger.info(f"Generating thumbnail for asset {asset_id}")

    db_session = get_sync_session()
    settings = get_settings()
    thumb_service = ThumbnailService(
        thumbnail_dir=settings.thumbnail_dir, thumbnail_size=settings.thumbnail_size
    )

    try:
        # Get asset from database
        query = select(ImageAsset).where(ImageAsset.id == asset_id)
        result = db_session.execute(query)
        asset = result.scalar_one_or_none()

        if not asset:
            error_msg = f"Asset {asset_id} not found"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg, "assetId": asset_id}

        # Check if thumbnail already exists
        if thumb_service.thumbnail_exists(asset.id, asset.path):
            logger.info(f"Thumbnail already exists for asset {asset_id}")
            return {
                "status": "skipped",
                "message": "Thumbnail already exists",
                "assetId": asset_id,
                "thumbnailPath": asset.thumbnail_path,
            }

        # Generate thumbnail
        try:
            thumb_path, width, height = thumb_service.generate_thumbnail(
                asset.path, asset.id
            )

            # Update asset
            asset.thumbnail_path = thumb_path
            asset.width = width
            asset.height = height
            db_session.commit()

            logger.info(f"Successfully generated thumbnail for asset {asset_id}")

            return {
                "status": "success",
                "assetId": asset_id,
                "thumbnailPath": thumb_path,
                "width": width,
                "height": height,
            }

        except FileNotFoundError:
            error_msg = f"Original image not found: {asset.path}"
            logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg,
                "assetId": asset_id,
            }

    except Exception as e:
        error_msg = f"Error generating thumbnail for asset {asset_id}: {e}"
        logger.exception(error_msg)
        return {
            "status": "error",
            "message": str(e),
            "assetId": asset_id,
        }

    finally:
        db_session.close()
