"""Singleton manager for file watching and periodic scanning.

Integrates with application lifecycle.
"""

from image_search_service.core.config import get_settings
from image_search_service.core.logging import get_logger
from image_search_service.queue.auto_detection_jobs import process_new_image
from image_search_service.queue.worker import QUEUE_LOW, get_queue
from image_search_service.services.file_watcher import FileWatcherService

logger = get_logger(__name__)


class WatcherManager:
    """Singleton manager for file watching system."""

    _instance: "WatcherManager | None" = None

    def __init__(self) -> None:
        """Initialize watcher manager."""
        self.settings = get_settings()
        self.file_watcher: FileWatcherService | None = None
        self._started = False

    @classmethod
    def get_instance(cls) -> "WatcherManager":
        """Get singleton instance.

        Returns:
            WatcherManager instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _on_new_image_sync(self, path: str) -> None:
        """Callback for file watcher (sync context).

        Args:
            path: Path to newly detected image
        """
        queue = get_queue(QUEUE_LOW)
        queue.enqueue(
            process_new_image,
            path,
            auto_train=self.settings.watch_auto_train,
            job_timeout=60,
        )
        logger.debug(f"Enqueued new image for processing: {path}")

    def start(self) -> None:
        """Start watching (called from app startup)."""
        if not self.settings.watch_enabled:
            logger.info("File watching disabled")
            return

        if self._started:
            return

        watch_paths = [self.settings.image_root_dir] if self.settings.image_root_dir else []

        if not watch_paths:
            logger.warning("No watch paths configured (IMAGE_ROOT_DIR not set)")
            return

        # Start file watcher
        self.file_watcher = FileWatcherService(
            watch_paths=watch_paths,
            on_new_image=self._on_new_image_sync,
            recursive=True,
            debounce_seconds=self.settings.watch_debounce_seconds,
        )
        self.file_watcher.start()

        self._started = True
        logger.info(f"Watcher manager started, watching: {watch_paths}")

    def stop(self) -> None:
        """Stop watching (called from app shutdown)."""
        if self.file_watcher:
            self.file_watcher.stop()
        self._started = False
        logger.info("Watcher manager stopped")

    def get_status(self) -> dict[str, object]:
        """Get current status.

        Returns:
            Dictionary with watcher status
        """
        return {
            "enabled": self.settings.watch_enabled,
            "running": self._started,
            "watch_paths": [self.settings.image_root_dir] if self.settings.image_root_dir else [],
            "file_watcher_active": self.file_watcher.is_running if self.file_watcher else False,
            "auto_train": self.settings.watch_auto_train,
            "debounce_seconds": self.settings.watch_debounce_seconds,
        }
