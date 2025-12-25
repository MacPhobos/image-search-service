"""Admin routes for destructive operations."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.api.admin_schemas import (
    DeleteAllDataRequest,
    DeleteAllDataResponse,
)
from image_search_service.core.logging import get_logger
from image_search_service.db.session import get_db
from image_search_service.services import admin_service

logger = get_logger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/data/delete-all", response_model=DeleteAllDataResponse)
async def delete_all_data(
    request: DeleteAllDataRequest,
    db: AsyncSession = Depends(get_db),
) -> DeleteAllDataResponse:
    """Delete all application data from Qdrant and PostgreSQL.

    DANGER: This is irreversible and deletes ALL data except migrations.

    This endpoint removes:
    - All vectors from Qdrant collections (main + faces)
    - All rows from application tables in PostgreSQL
    - Preserves: alembic_version table for migration tracking

    Safety: Requires confirm=true AND confirmation_text="DELETE ALL DATA"

    Args:
        request: Delete all data request with double confirmation
        db: Database session

    Returns:
        Deletion summary with counts and preserved migration version

    Raises:
        HTTPException: 400 if confirmation requirements not met

    Example:
        ```
        POST /api/v1/admin/data/delete-all
        {
            "confirm": true,
            "confirmationText": "DELETE ALL DATA",
            "reason": "Resetting development environment"
        }
        ```
    """
    logger.warning(
        "Admin delete all data endpoint called. "
        f"Confirmation: {request.confirm}, "
        f"Text: '{request.confirmation_text}', "
        f"Reason: {request.reason or 'None'}"
    )

    result = await admin_service.delete_all_data(
        db=db,
        confirm=request.confirm,
        confirmation_text=request.confirmation_text,
        reason=request.reason,
    )

    logger.warning(
        "Admin delete all data completed: "
        f"{sum(result.qdrant_collections_deleted.values())} Qdrant vectors, "
        f"{sum(result.postgres_tables_truncated.values())} PostgreSQL rows deleted"
    )

    return result
