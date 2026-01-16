"""API routes and endpoints."""

from fastapi import APIRouter

from image_search_service.api.routes.admin import router as admin_router
from image_search_service.api.routes.assets import router as assets_router
from image_search_service.api.routes.categories import router as categories_router
from image_search_service.api.routes.config import router as config_router
from image_search_service.api.routes.evidence import router as evidence_router
from image_search_service.api.routes.face_centroids import (
    router as face_centroids_router,
)
from image_search_service.api.routes.face_sessions import router as face_sessions_router
from image_search_service.api.routes.face_suggestions import (
    router as face_suggestions_router,
)
from image_search_service.api.routes.faces import router as faces_router
from image_search_service.api.routes.images import router as images_router
from image_search_service.api.routes.jobs import (
    job_progress_router,
)
from image_search_service.api.routes.queues import (
    jobs_router,
    workers_router,
)
from image_search_service.api.routes.queues import (
    router as queues_router,
)
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
api_v1_router.include_router(admin_router)
api_v1_router.include_router(assets_router)
api_v1_router.include_router(categories_router)
api_v1_router.include_router(config_router)
api_v1_router.include_router(images_router)
api_v1_router.include_router(search_router)
api_v1_router.include_router(training_router)
api_v1_router.include_router(evidence_router)
api_v1_router.include_router(system_router)
api_v1_router.include_router(vectors_router)
api_v1_router.include_router(faces_router)
api_v1_router.include_router(face_sessions_router)
api_v1_router.include_router(face_suggestions_router)
api_v1_router.include_router(face_centroids_router)
api_v1_router.include_router(queues_router)
api_v1_router.include_router(jobs_router)
api_v1_router.include_router(workers_router)
api_v1_router.include_router(job_progress_router)
