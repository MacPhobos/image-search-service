"""Thumbnail generation background jobs for RQ."""

from sqlalchemy import select

from image_search_service.core.config import get_settings
from image_search_service.core.logging import get_logger
from image_search_service.db.models import ImageAsset, TrainingJob
from image_search_service.db.sync_operations import get_sync_session
from image_search_service.services.exif_service import get_exif_service
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

                # Extract EXIF metadata (graceful degradation)
                try:
                    exif_service = get_exif_service()
                    exif_data = exif_service.extract_exif(asset.path)

                    if exif_data:
                        # Update asset with EXIF data (only if values are present)
                        if exif_data.get("taken_at"):
                            asset.taken_at = exif_data["taken_at"]
                        if exif_data.get("camera_make"):
                            asset.camera_make = exif_data["camera_make"]
                        if exif_data.get("camera_model"):
                            asset.camera_model = exif_data["camera_model"]
                        if exif_data.get("gps_latitude") is not None:
                            asset.gps_latitude = exif_data["gps_latitude"]
                        if exif_data.get("gps_longitude") is not None:
                            asset.gps_longitude = exif_data["gps_longitude"]
                        if exif_data.get("exif_metadata"):
                            asset.exif_metadata = exif_data["exif_metadata"]

                        logger.debug(
                            f"Extracted EXIF for asset {asset.id}: "
                            f"taken_at={asset.taken_at}, camera={asset.camera_make}"
                        )
                except Exception as e:
                    # Log debug message - don't spam logs in batch mode
                    logger.debug(f"Could not extract EXIF for asset {asset.id}: {e}")

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


