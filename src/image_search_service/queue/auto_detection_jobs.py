"""Background jobs for auto-detection of new images."""

from pathlib import Path

from sqlalchemy import select

from image_search_service.core.logging import get_logger
from image_search_service.db.models import ImageAsset, TrainingStatus
from image_search_service.db.sync_operations import get_sync_session

logger = get_logger(__name__)


def process_new_image(image_path: str, auto_train: bool = False) -> dict[str, object]:
    """Process a newly detected image.

    - Create ImageAsset record if not exists
    - Optionally enqueue for training

    Args:
        image_path: Absolute path to image file
        auto_train: Whether to auto-enqueue for training

    Returns:
        Dictionary with processing result
    """
    with get_sync_session() as db:
        # Check if already exists
        stmt = select(ImageAsset).where(ImageAsset.path == image_path)
        existing = db.execute(stmt).scalar_one_or_none()

        if existing:
            logger.debug(f"Image already exists: {image_path}")
            return {"status": "exists", "asset_id": existing.id}

        # Get file metadata
        file_path = Path(image_path)
        if not file_path.exists():
            logger.warning(f"File not found: {image_path}")
            return {"status": "error", "error": "File not found"}

        file_stat = file_path.stat()

        # Create new asset
        asset = ImageAsset(
            path=image_path,
            file_size=file_stat.st_size,
            file_modified_at=file_stat.st_mtime,
            training_status=TrainingStatus.PENDING.value,
        )
        db.add(asset)
        db.commit()
        db.refresh(asset)

        logger.info(f"New image detected and added: {image_path} (id={asset.id})")

        result: dict[str, object] = {"status": "created", "asset_id": asset.id, "path": image_path}

        # Optionally enqueue for training
        if auto_train:
            from image_search_service.queue.training_jobs import train_single_asset
            from image_search_service.queue.worker import QUEUE_LOW, get_queue

            queue = get_queue(QUEUE_LOW)
            job = queue.enqueue(
                train_single_asset,
                job_id=None,
                asset_id=asset.id,
                session_id=None,
                job_timeout=300,
            )
            result["training_job_id"] = job.id
            logger.info(f"Auto-enqueued training for asset {asset.id}")

        return result


def scan_directory_incremental(
    directory: str, extensions: list[str] | None = None, auto_train: bool = False
) -> dict[str, object]:
    """Scan directory for new images incrementally.

    - Compare with existing database records
    - Add only new images
    - Optionally enqueue for training

    Args:
        directory: Directory path to scan
        extensions: List of file extensions (with dots)
        auto_train: Whether to auto-enqueue for training

    Returns:
        Dictionary with scan results
    """
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"]

    directory_path = Path(directory)
    if not directory_path.exists():
        return {"error": f"Directory not found: {directory}"}

    discovered = 0
    created = 0
    skipped = 0

    with get_sync_session() as db:
        # Get existing paths for this directory
        path_stmt = select(ImageAsset.path).where(ImageAsset.path.like(f"{directory}%"))
        path_result = db.execute(path_stmt)
        existing_paths = set(path for (path,) in path_result.all())

        # Scan directory
        for ext in extensions:
            for file_path in directory_path.rglob(f"*{ext}"):
                if not file_path.is_file():
                    continue

                discovered += 1
                path_str = str(file_path.absolute())

                if path_str in existing_paths:
                    skipped += 1
                    continue

                # Get file metadata
                file_stat = file_path.stat()

                # Create new asset
                asset = ImageAsset(
                    path=path_str,
                    file_size=file_stat.st_size,
                    file_modified_at=file_stat.st_mtime,
                    training_status=TrainingStatus.PENDING.value,
                )
                db.add(asset)
                created += 1

                # Optionally enqueue for training
                if auto_train and created % 10 == 0:
                    # Flush every 10 assets to get IDs
                    db.flush()

        db.commit()

    logger.info(
        f"Incremental scan of {directory}: "
        f"discovered={discovered}, created={created}, skipped={skipped}"
    )

    # Enqueue training jobs if auto_train is enabled
    if auto_train and created > 0:
        from image_search_service.queue.training_jobs import train_single_asset
        from image_search_service.queue.worker import QUEUE_LOW, get_queue

        queue = get_queue(QUEUE_LOW)

        with get_sync_session() as db:
            # Get newly created assets for this directory
            asset_stmt = (
                select(ImageAsset)
                .where(ImageAsset.path.like(f"{directory}%"))
                .where(ImageAsset.training_status == TrainingStatus.PENDING.value)
                .limit(created)
            )
            asset_result = db.execute(asset_stmt)
            new_assets = list(asset_result.scalars().all())

            enqueued = 0
            for asset in new_assets:
                queue.enqueue(
                    train_single_asset,
                    job_id=None,
                    asset_id=asset.id,
                    session_id=None,
                    job_timeout=300,
                )
                enqueued += 1

            logger.info(f"Auto-enqueued {enqueued} training jobs")

    return {
        "directory": directory,
        "discovered": discovered,
        "created": created,
        "skipped": skipped,
    }
