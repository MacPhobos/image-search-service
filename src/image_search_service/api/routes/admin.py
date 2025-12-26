"""Admin routes for destructive operations."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.api.admin_schemas import (
    DeleteAllDataRequest,
    DeleteAllDataResponse,
    ImportRequest,
    ImportResponse,
    PersonMetadataExport,
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


@router.post("/persons/export", response_model=PersonMetadataExport)
async def export_person_metadata(
    max_faces_per_person: int = Query(
        100, ge=1, le=500, description="Maximum face mappings per person"
    ),
    db: AsyncSession = Depends(get_db),
) -> PersonMetadataExport:
    """Export all persons with their face-to-image mappings for backup.

    Returns JSON structure with:
    - Person names and status (active persons only)
    - Up to max_faces_per_person face mappings per person
    - Each face mapping includes image path and bounding box

    Use this export before "Delete All Data" to preserve face labels.
    The export can be imported later to restore face-to-person mappings.

    Args:
        max_faces_per_person: Maximum face mappings per person (1-500)
        db: Database session

    Returns:
        PersonMetadataExport with export metadata and person data

    Example:
        ```
        POST /api/v1/admin/persons/export?max_faces_per_person=100
        ```
    """
    logger.info(
        f"Exporting person metadata with max_faces_per_person={max_faces_per_person}"
    )

    result = await admin_service.export_person_metadata(db, max_faces_per_person)

    logger.info(
        f"Export completed: {result.metadata.total_persons} persons, "
        f"{result.metadata.total_face_mappings} total faces"
    )

    return result


@router.post("/persons/import", response_model=ImportResponse)
async def import_person_metadata(
    request: ImportRequest,
    db: AsyncSession = Depends(get_db),
) -> ImportResponse:
    """Import persons and face mappings from export file.

    Process:
    1. Creates persons (or finds existing by name)
    2. For each face mapping:
       - Checks if image exists at stored path
       - Finds faces in image from database
       - Matches face by bounding box within tolerance
       - Assigns matched face to person

    Options:
    - dry_run: If true, simulates import without making changes
    - tolerance_pixels: Bounding box matching tolerance (default: 10)
    - skip_missing_images: If true, skips faces with missing images

    Use after "Delete All Data" to restore previously exported face labels.

    Args:
        request: Import request with export data and options
        db: Database session

    Returns:
        ImportResponse with import results and statistics

    Example:
        ```
        POST /api/v1/admin/persons/import
        {
            "data": { ... exported data ... },
            "options": {
                "dryRun": false,
                "tolerancePixels": 10,
                "skipMissingImages": true
            }
        }
        ```
    """
    from image_search_service.vector.face_qdrant import FaceQdrantClient

    logger.info(
        f"Importing person metadata: "
        f"dry_run={request.options.dry_run}, "
        f"tolerance={request.options.tolerance_pixels}px, "
        f"skip_missing={request.options.skip_missing_images}"
    )

    face_qdrant = FaceQdrantClient.get_instance()

    result = await admin_service.import_person_metadata(
        db=db,
        import_data=request.data,
        options=request.options,
        face_qdrant=face_qdrant,
    )

    logger.info(
        f"Import completed: "
        f"success={result.success}, "
        f"persons_created={result.persons_created}, "
        f"persons_existing={result.persons_existing}, "
        f"faces_matched={result.total_faces_matched}, "
        f"faces_not_found={result.total_faces_not_found}, "
        f"images_missing={result.total_images_missing}"
    )

    return result
