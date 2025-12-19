"""System management endpoints."""

from fastapi import APIRouter

from image_search_service.core.logging import get_logger
from image_search_service.services.watcher_manager import WatcherManager

logger = get_logger(__name__)
router = APIRouter(prefix="/system", tags=["system"])


@router.get("/watcher/status")
async def get_watcher_status() -> dict[str, object]:
    """Get file watcher status.

    Returns:
        Dictionary with watcher configuration and status
    """
    watcher = WatcherManager.get_instance()
    return watcher.get_status()


@router.post("/watcher/start")
async def start_watcher() -> dict[str, object]:
    """Manually start file watcher.

    Returns:
        Dictionary with updated watcher status
    """
    watcher = WatcherManager.get_instance()
    watcher.start()
    return watcher.get_status()


@router.post("/watcher/stop")
async def stop_watcher() -> dict[str, object]:
    """Manually stop file watcher.

    Returns:
        Dictionary with updated watcher status
    """
    watcher = WatcherManager.get_instance()
    watcher.stop()
    return watcher.get_status()
