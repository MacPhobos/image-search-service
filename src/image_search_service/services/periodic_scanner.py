"""Periodic scanner service for fallback image detection."""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path

from image_search_service.core.logging import get_logger

logger = get_logger(__name__)


class PeriodicScannerService:
    """Fallback scanner that periodically scans directories for new images.

    Useful when file watching is not reliable (network mounts, etc.)
    """

    def __init__(
        self,
        scan_paths: list[str],
        interval_seconds: int,
        on_new_image: Callable[[str], Awaitable[None]],
        extensions: list[str] | None = None,
    ):
        """Initialize periodic scanner service.

        Args:
            scan_paths: List of directory paths to scan
            interval_seconds: Seconds between scans
            on_new_image: Async callback function for new images
            extensions: List of file extensions to scan (with dots)
        """
        self.scan_paths = [Path(p) for p in scan_paths]
        self.interval_seconds = interval_seconds
        self.on_new_image = on_new_image
        self.extensions = extensions or [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"]
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._last_scan: dict[str, datetime] = {}  # path -> last modified time

    async def start(self) -> None:
        """Start periodic scanning."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._scan_loop())
        logger.info(f"Periodic scanner started (interval: {self.interval_seconds}s)")

    async def stop(self) -> None:
        """Stop periodic scanning."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Periodic scanner stopped")

    async def _scan_loop(self) -> None:
        """Main scan loop."""
        while self._running:
            try:
                await self._scan_all_paths()
            except Exception as e:
                logger.error(f"Error in periodic scan: {e}")
            await asyncio.sleep(self.interval_seconds)

    async def _scan_all_paths(self) -> None:
        """Scan all configured paths for new images."""
        for path in self.scan_paths:
            if not path.exists():
                continue
            await self._scan_directory(path)

    async def _scan_directory(self, directory: Path) -> None:
        """Scan a directory for new images.

        Args:
            directory: Directory path to scan
        """
        for ext in self.extensions:
            for file_path in directory.rglob(f"*{ext}"):
                if not file_path.is_file():
                    continue

                # Check if file is new or modified
                file_key = str(file_path)
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)

                if file_key not in self._last_scan or mtime > self._last_scan[file_key]:
                    self._last_scan[file_key] = mtime
                    await self.on_new_image(file_key)

    @property
    def is_running(self) -> bool:
        """Check if scanner is running.

        Returns:
            True if scanner is running
        """
        return self._running
