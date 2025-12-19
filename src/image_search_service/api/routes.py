"""API routes and endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint that works without external dependencies.

    Returns:
        Status dictionary indicating service health
    """
    return {"status": "ok"}
