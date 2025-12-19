"""Directory scanning service for training system."""

from pathlib import Path

from image_search_service.api.training_schemas import DirectoryInfo
from image_search_service.core.config import get_settings
from image_search_service.core.logging import get_logger

logger = get_logger(__name__)


class DirectoryService:
    """Service for scanning and validating directory paths."""

    def __init__(self) -> None:
        """Initialize directory service with settings."""
        self.settings = get_settings()

    def validate_path(self, path: str) -> bool:
        """Validate path to prevent directory traversal attacks.

        Args:
            path: Path to validate

        Returns:
            True if path is valid and within allowed root

        Raises:
            ValueError: If path is invalid or outside allowed root
        """
        try:
            # Strip whitespace from path
            path = path.strip()

            # Resolve to absolute path
            abs_path = Path(path).resolve()

            # Check if path exists
            if not abs_path.exists():
                raise ValueError(f"Path does not exist: {path}")

            # Check if it's a directory
            if not abs_path.is_dir():
                raise ValueError(f"Path is not a directory: {path}")

            # If image_root_dir is configured, enforce it
            if self.settings.image_root_dir:
                root_dir = Path(self.settings.image_root_dir).resolve()
                try:
                    abs_path.relative_to(root_dir)
                except ValueError:
                    raise ValueError(
                        f"Path {path} is outside allowed root {self.settings.image_root_dir}"
                    )

            return True
        except Exception as e:
            logger.warning(f"Path validation failed for {path}: {e}")
            raise

    def count_images(self, path: str, extensions: list[str]) -> int:
        """Count image files in a directory (non-recursive).

        Args:
            path: Directory path
            extensions: List of file extensions to count

        Returns:
            Count of image files
        """
        try:
            dir_path = Path(path)
            count = 0

            for ext in extensions:
                # Check lowercase extension
                count += len(list(dir_path.glob(f"*.{ext.lower()}")))
                # Check uppercase extension
                count += len(list(dir_path.glob(f"*.{ext.upper()}")))

            return count
        except Exception as e:
            logger.error(f"Failed to count images in {path}: {e}")
            return 0

    def list_subdirectories(self, path: str) -> list[DirectoryInfo]:
        """List immediate subdirectories with image counts.

        Args:
            path: Root directory path

        Returns:
            List of subdirectory information
        """
        self.validate_path(path)
        root_path = Path(path)
        subdirs: list[DirectoryInfo] = []

        # Default extensions for image counting
        default_extensions = ["jpg", "jpeg", "png", "webp"]

        try:
            # Get all immediate subdirectories
            for entry in sorted(root_path.iterdir()):
                if entry.is_dir():
                    # Count images in this subdirectory (non-recursive)
                    image_count = self.count_images(str(entry), default_extensions)

                    subdirs.append(
                        DirectoryInfo(
                            path=str(entry),
                            name=entry.name,
                            imageCount=image_count,
                            selected=False,
                        )
                    )
        except Exception as e:
            logger.error(f"Failed to list subdirectories in {path}: {e}")
            raise

        return subdirs

    def scan_directory(
        self, root_path: str, recursive: bool = True, extensions: list[str] | None = None
    ) -> list[DirectoryInfo]:
        """Scan directory for subdirectories with image files.

        Args:
            root_path: Root directory to scan
            recursive: Whether to scan recursively (currently always scans one level)
            extensions: List of file extensions to count (defaults to common image formats)

        Returns:
            List of subdirectory information with image counts
        """
        if extensions is None:
            extensions = ["jpg", "jpeg", "png", "webp"]

        # Validate root path
        self.validate_path(root_path)

        # For now, we only scan immediate subdirectories
        # This matches the UI workflow where users select from visible subdirectories
        return self.list_subdirectories(root_path)
