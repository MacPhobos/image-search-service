"""File watcher service for detecting new image files."""

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from watchdog.events import FileSystemEvent, FileSystemEventHandler

from image_search_service.core.logging import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}


class ImageFileHandler(FileSystemEventHandler):
    """Handle file system events for images."""

    def __init__(self, callback: Callable[[str], None], debounce_seconds: float = 1.0):
        """Initialize image file handler.

        Args:
            callback: Function to call when new image is detected
            debounce_seconds: Seconds to wait before processing file
        """
        self.callback = callback
        self.debounce_seconds = debounce_seconds
        self._pending: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle new file created.

        Args:
            event: File creation event
        """
        if event.is_directory:
            return
        # event.src_path can be bytes or str, convert to str
        path_str = event.src_path if isinstance(event.src_path, str) else event.src_path.decode()
        path = Path(path_str)
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            self._debounced_callback(str(path))

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modified (sometimes triggered instead of created).

        Args:
            event: File modification event
        """
        if event.is_directory:
            return
        # event.src_path can be bytes or str, convert to str
        path_str = event.src_path if isinstance(event.src_path, str) else event.src_path.decode()
        path = Path(path_str)
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            self._debounced_callback(str(path))

    def _debounced_callback(self, path: str) -> None:
        """Debounce callbacks to handle rapid file operations.

        Args:
            path: File path that triggered event
        """
        with self._lock:
            if path in self._pending:
                self._pending[path].cancel()
            timer = threading.Timer(self.debounce_seconds, self._execute_callback, args=[path])
            self._pending[path] = timer
            timer.start()

    def _execute_callback(self, path: str) -> None:
        """Execute the callback after debounce delay.

        Args:
            path: File path to process
        """
        with self._lock:
            self._pending.pop(path, None)
        try:
            self.callback(path)
        except Exception as e:
            logger.error(f"Error in file watcher callback for {path}: {e}")


class FileWatcherService:
    """Service to watch directories for new images."""

    def __init__(
        self,
        watch_paths: list[str],
        on_new_image: Callable[[str], None],
        recursive: bool = True,
        debounce_seconds: float = 1.0,
    ):
        """Initialize file watcher service.

        Args:
            watch_paths: List of directory paths to watch
            on_new_image: Callback function for new images
            recursive: Watch subdirectories recursively
            debounce_seconds: Seconds to wait before processing file
        """
        self.watch_paths = [Path(p) for p in watch_paths]
        self.on_new_image = on_new_image
        self.recursive = recursive
        self.debounce_seconds = debounce_seconds
        self._observer: Any = None  # Type is watchdog.observers.Observer at runtime
        self._running = False

    def start(self) -> None:
        """Start watching directories."""
        if self._running:
            return

        from watchdog.observers import Observer

        self._observer = Observer()
        handler = ImageFileHandler(self.on_new_image, self.debounce_seconds)

        for path in self.watch_paths:
            if path.exists() and path.is_dir():
                self._observer.schedule(handler, str(path), recursive=self.recursive)
                logger.info(f"Watching directory: {path}")
            else:
                logger.warning(f"Watch path does not exist: {path}")

        self._observer.start()
        self._running = True
        logger.info("File watcher started")

    def stop(self) -> None:
        """Stop watching directories."""
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5.0)
            self._observer = None
        self._running = False
        logger.info("File watcher stopped")

    @property
    def is_running(self) -> bool:
        """Check if watcher is running.

        Returns:
            True if watcher is running
        """
        return self._running
