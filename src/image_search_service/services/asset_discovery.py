"""Asset discovery service for scanning directories and creating asset records."""

from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.core.logging import get_logger
from image_search_service.db.models import ImageAsset, TrainingSession, TrainingSubdirectory

logger = get_logger(__name__)


class AssetDiscoveryService:
    """Service for discovering and registering image assets."""

    def __init__(self, extensions: list[str] | None = None) -> None:
        """Initialize asset discovery service.

        Args:
            extensions: List of image file extensions to scan (default: jpg, jpeg, png, webp)
        """
        self.extensions = extensions or ["jpg", "jpeg", "png", "webp"]
        # Convert to lowercase with dots for easier comparison
        self.extensions_set = {f".{ext.lower()}" for ext in self.extensions}

    async def discover_assets(
        self, db: AsyncSession, session_id: int
    ) -> list[ImageAsset]:
        """Discover all images in selected subdirectories for a session.

        Args:
            db: Database session
            session_id: Training session ID

        Returns:
            List of ImageAsset records (newly created and existing)

        Raises:
            ValueError: If session not found
        """
        # Get session with subdirectories
        query = (
            select(TrainingSession)
            .where(TrainingSession.id == session_id)
        )
        result = await db.execute(query)
        session = result.scalar_one_or_none()

        if not session:
            raise ValueError(f"Training session {session_id} not found")

        # Get selected subdirectories
        subdirs_query = (
            select(TrainingSubdirectory)
            .where(TrainingSubdirectory.session_id == session_id)
            .where(TrainingSubdirectory.selected == True)  # noqa: E712
        )
        subdirs_result = await db.execute(subdirs_query)
        selected_subdirs: list[TrainingSubdirectory] = list(subdirs_result.scalars().all())

        if not selected_subdirs:
            logger.warning(f"No subdirectories selected for session {session_id}")
            return []

        # Discover assets in each subdirectory
        all_assets: list[ImageAsset] = []

        for subdir in selected_subdirs:
            logger.info(f"Scanning directory: {subdir.path}")
            assets = await self._scan_directory(db, subdir.path, recursive=True)
            all_assets.extend(assets)

            logger.info(f"Found {len(assets)} images in {subdir.path}")

        logger.info(f"Total assets discovered for session {session_id}: {len(all_assets)}")
        return all_assets

    async def _scan_directory(
        self, db: AsyncSession, directory_path: str, recursive: bool = True
    ) -> list[ImageAsset]:
        """Scan a directory for image files.

        Args:
            db: Database session
            directory_path: Directory path to scan
            recursive: Whether to scan subdirectories recursively

        Returns:
            List of ImageAsset records
        """
        path = Path(directory_path)
        if not path.exists() or not path.is_dir():
            logger.warning(f"Directory does not exist or is not a directory: {directory_path}")
            return []

        assets: list[ImageAsset] = []

        # Use glob pattern based on recursion setting
        pattern = "**/*" if recursive else "*"

        for file_path in path.glob(pattern):
            if not file_path.is_file():
                continue

            # Check if file has valid image extension
            if file_path.suffix.lower() not in self.extensions_set:
                continue

            # Create or get existing asset
            asset = await self.ensure_asset_exists(db, str(file_path.absolute()))
            assets.append(asset)

        return assets

    async def ensure_asset_exists(self, db: AsyncSession, path: str) -> ImageAsset:
        """Create or get existing asset by path.

        Args:
            db: Database session
            path: Absolute file path

        Returns:
            ImageAsset record (existing or newly created)
        """
        # Check if asset already exists
        query = select(ImageAsset).where(ImageAsset.path == path)
        result = await db.execute(query)
        existing_asset = result.scalar_one_or_none()

        if existing_asset:
            return existing_asset

        # Create new asset
        file_path = Path(path)
        file_stat = file_path.stat()

        asset = ImageAsset(
            path=path,
            file_size=file_stat.st_size,
            file_modified_at=datetime.fromtimestamp(file_stat.st_mtime, tz=UTC),
        )

        db.add(asset)
        await db.flush()

        logger.debug(f"Created new asset record for {path}")
        return asset

    async def count_images_in_directory(
        self, directory: str, recursive: bool = True
    ) -> int:
        """Count images in a directory without creating database records.

        Args:
            directory: Directory path to scan
            recursive: Whether to scan subdirectories recursively

        Returns:
            Number of image files found
        """
        path = Path(directory)
        if not path.exists() or not path.is_dir():
            return 0

        count = 0
        pattern = "**/*" if recursive else "*"

        for file_path in path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.extensions_set:
                count += 1

        return count
