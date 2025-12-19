"""Thumbnail generation service for image assets."""

import hashlib
from pathlib import Path

from PIL import Image
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.core.logging import get_logger
from image_search_service.db.models import ImageAsset

logger = get_logger(__name__)

# MIME type mapping for supported image formats
MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


class ThumbnailService:
    """Service for generating and managing image thumbnails."""

    def __init__(self, thumbnail_dir: str, thumbnail_size: int = 256):
        """Initialize thumbnail service.

        Args:
            thumbnail_dir: Directory to store thumbnails
            thumbnail_size: Maximum width/height for thumbnails (maintains aspect ratio)
        """
        self.thumbnail_dir = Path(thumbnail_dir)
        self.thumbnail_size = thumbnail_size
        self.thumbnail_dir.mkdir(parents=True, exist_ok=True)

    def get_thumbnail_path(self, asset_id: int, original_path: str) -> Path:
        """Generate consistent thumbnail path based on asset ID.

        Uses MD5 hash sharding to avoid too many files in one directory.

        Args:
            asset_id: Image asset ID
            original_path: Original image path (unused, for future compatibility)

        Returns:
            Path to thumbnail file
        """
        # Use hash for directory sharding to avoid too many files in one dir
        hash_prefix = hashlib.md5(str(asset_id).encode()).hexdigest()[:2]
        shard_dir = self.thumbnail_dir / hash_prefix
        shard_dir.mkdir(parents=True, exist_ok=True)
        return shard_dir / f"{asset_id}.jpg"

    def thumbnail_exists(self, asset_id: int, original_path: str) -> bool:
        """Check if thumbnail already exists.

        Args:
            asset_id: Image asset ID
            original_path: Original image path

        Returns:
            True if thumbnail exists, False otherwise
        """
        thumb_path = self.get_thumbnail_path(asset_id, original_path)
        return thumb_path.exists()

    def get_image_dimensions(self, image_path: str) -> tuple[int, int]:
        """Get original image dimensions.

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (width, height) in pixels

        Raises:
            FileNotFoundError: If image file doesn't exist
            IOError: If file is not a valid image
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(path) as img:
            width, height = img.size
            return (width, height)

    def generate_thumbnail(
        self, original_path: str, asset_id: int
    ) -> tuple[str, int, int]:
        """Generate thumbnail for image.

        Resizes to fit within thumbnail_size x thumbnail_size while maintaining aspect ratio.
        Saves as JPEG for smaller file size.

        Args:
            original_path: Path to original image
            asset_id: Image asset ID

        Returns:
            Tuple of (thumbnail_path, width, height) for the thumbnail

        Raises:
            FileNotFoundError: If original image doesn't exist
            IOError: If image processing fails
        """
        original = Path(original_path)
        if not original.exists():
            raise FileNotFoundError(f"Original image not found: {original_path}")

        # Get thumbnail path with sharding
        thumb_path = self.get_thumbnail_path(asset_id, original_path)

        # Open and resize image
        with Image.open(original) as img:
            # Convert RGBA to RGB for JPEG compatibility
            if img.mode in ("RGBA", "LA", "P"):
                # Create white background
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
                img = background

            # Resize maintaining aspect ratio
            img.thumbnail((self.thumbnail_size, self.thumbnail_size), Image.Resampling.LANCZOS)

            # Save as JPEG with quality 85
            img.save(thumb_path, "JPEG", quality=85, optimize=True)

            # Return thumbnail info
            width, height = img.size
            logger.debug(
                f"Generated thumbnail for asset {asset_id}: "
                f"{thumb_path} ({width}x{height})"
            )
            return (str(thumb_path), width, height)

    async def generate_thumbnails_for_session(
        self, db: AsyncSession, session_id: int
    ) -> dict[str, int]:
        """Generate thumbnails for all assets in a training session.

        Args:
            db: Database session
            session_id: Training session ID

        Returns:
            Dictionary with stats: {generated: X, skipped: Y, failed: Z}
        """
        # Get all assets for session via training jobs
        from image_search_service.db.models import TrainingJob

        query = (
            select(ImageAsset)
            .join(TrainingJob, TrainingJob.asset_id == ImageAsset.id)
            .where(TrainingJob.session_id == session_id)
            .distinct()
        )
        result = await db.execute(query)
        assets = list(result.scalars().all())

        stats = {"generated": 0, "skipped": 0, "failed": 0}

        for asset in assets:
            try:
                # Skip if thumbnail already exists and is up to date
                if self.thumbnail_exists(asset.id, asset.path):
                    stats["skipped"] += 1
                    continue

                # Generate thumbnail
                thumb_path, width, height = self.generate_thumbnail(asset.path, asset.id)

                # Update asset with thumbnail info
                asset.thumbnail_path = thumb_path
                asset.width = width
                asset.height = height

                stats["generated"] += 1

            except Exception as e:
                logger.error(f"Failed to generate thumbnail for asset {asset.id}: {e}")
                stats["failed"] += 1

        # Commit updates
        await db.commit()

        logger.info(
            f"Thumbnail generation for session {session_id} complete: "
            f"{stats['generated']} generated, {stats['skipped']} skipped, "
            f"{stats['failed']} failed"
        )

        return stats

    @staticmethod
    def get_mime_type(path: str) -> str:
        """Get MIME type from file extension.

        Args:
            path: File path

        Returns:
            MIME type string (defaults to 'image/jpeg' if unknown)
        """
        ext = Path(path).suffix.lower()
        return MIME_TYPES.get(ext, "image/jpeg")
