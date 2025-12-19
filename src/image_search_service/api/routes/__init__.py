"""API routes and endpoints."""

from fastapi import APIRouter

from image_search_service.api.routes.assets import router as assets_router
from image_search_service.api.routes.search import router as search_router

router = APIRouter()


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint that works without external dependencies.

    Returns:
        Status dictionary indicating service health
    """
    return {"status": "ok"}


# API v1 router with all endpoints
api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(assets_router)
api_v1_router.include_router(search_router)
