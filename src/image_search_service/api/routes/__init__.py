"""API routes and endpoints."""

from fastapi import APIRouter

from image_search_service.api.routes.assets import router as assets_router
from image_search_service.api.routes.categories import router as categories_router
from image_search_service.api.routes.evidence import router as evidence_router
from image_search_service.api.routes.images import router as images_router
from image_search_service.api.routes.search import router as search_router
from image_search_service.api.routes.system import router as system_router
from image_search_service.api.routes.training import router as training_router
from image_search_service.api.routes.vectors import router as vectors_router

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
api_v1_router.include_router(categories_router)
api_v1_router.include_router(images_router)
api_v1_router.include_router(search_router)
api_v1_router.include_router(training_router)
api_v1_router.include_router(evidence_router)
api_v1_router.include_router(system_router)
api_v1_router.include_router(vectors_router)
