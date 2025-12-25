"""Admin service for destructive operations."""

from datetime import UTC, datetime

from fastapi import HTTPException, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.api.admin_schemas import DeleteAllDataResponse
from image_search_service.core.logging import get_logger
from image_search_service.vector import qdrant
from image_search_service.vector.face_qdrant import get_face_qdrant_client

logger = get_logger(__name__)


async def delete_all_data(
    db: AsyncSession,
    confirm: bool,
    confirmation_text: str,
    reason: str | None,
) -> DeleteAllDataResponse:
    """Delete all application data from Qdrant and PostgreSQL.

    WARNING: This is highly destructive and cannot be undone. It deletes:
    - All Qdrant vector collections (main + faces)
    - All PostgreSQL table data (except alembic_version for migrations)

    Safety: Requires confirm=true AND confirmation_text="DELETE ALL DATA"

    Args:
        db: Database session
        confirm: Must be True
        confirmation_text: Must exactly match "DELETE ALL DATA"
        reason: Optional reason for deletion (audit trail)

    Returns:
        DeleteAllDataResponse with deletion summary

    Raises:
        HTTPException: 400 if confirmations are invalid
    """
    # Validate confirmations
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must confirm deletion by setting confirm=true",
        )

    if confirmation_text != "DELETE ALL DATA":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="confirmationText must exactly match 'DELETE ALL DATA'",
        )

    logger.warning(
        "ADMIN DELETE ALL DATA initiated. Reason: %s",
        reason or "No reason provided",
    )

    # Step 1: Get current Alembic version before deletion
    alembic_version = await _get_alembic_version(db)
    logger.info(f"Current Alembic version: {alembic_version}")

    # Step 2: Delete Qdrant collections
    qdrant_collections_deleted: dict[str, int] = {}

    # Delete main collection
    try:
        main_deleted = qdrant.reset_collection()
        qdrant_collections_deleted["main"] = main_deleted
        logger.info(f"Deleted {main_deleted} vectors from main collection")
    except Exception as e:
        logger.error(f"Failed to delete main collection: {e}")
        qdrant_collections_deleted["main"] = 0

    # Delete faces collection
    try:
        face_client = get_face_qdrant_client()
        faces_deleted = face_client.reset_collection()
        qdrant_collections_deleted["faces"] = faces_deleted
        logger.info(f"Deleted {faces_deleted} vectors from faces collection")
    except Exception as e:
        logger.error(f"Failed to delete faces collection: {e}")
        qdrant_collections_deleted["faces"] = 0

    # Step 3: Truncate PostgreSQL tables (except alembic_version)
    postgres_tables_truncated = await _truncate_application_tables(db)
    logger.info(f"Truncated {len(postgres_tables_truncated)} PostgreSQL tables")

    # Build response
    total_qdrant = sum(qdrant_collections_deleted.values())
    total_postgres = sum(postgres_tables_truncated.values())

    logger.warning(
        "ADMIN DELETE ALL DATA completed: "
        f"{total_qdrant} Qdrant vectors, "
        f"{total_postgres} PostgreSQL rows deleted. "
        f"Alembic version preserved: {alembic_version}"
    )

    return DeleteAllDataResponse(
        qdrant_collections_deleted=qdrant_collections_deleted,
        postgres_tables_truncated=postgres_tables_truncated,
        alembic_version_preserved=alembic_version,
        message=(
            f"Deleted {total_qdrant} Qdrant vectors "
            f"and {total_postgres} PostgreSQL rows. "
            f"Migration version {alembic_version} preserved."
        ),
        timestamp=datetime.now(UTC),
    )


async def _get_alembic_version(db: AsyncSession) -> str:
    """Get current Alembic migration version.

    Args:
        db: Database session

    Returns:
        Current Alembic version string (e.g., "012_add_faces")
    """
    try:
        result = await db.execute(text("SELECT version_num FROM alembic_version"))
        row = result.fetchone()
        if row:
            return str(row[0])
        return "unknown"
    except Exception as e:
        logger.warning(f"Failed to get Alembic version: {e}")
        return "unknown"


async def _truncate_application_tables(db: AsyncSession) -> dict[str, int]:
    """Truncate all application tables except alembic_version.

    Uses TRUNCATE CASCADE to handle foreign key constraints.

    Args:
        db: Database session

    Returns:
        Dict of table name to row count deleted
    """
    # Define tables in CASCADE order (children before parents where possible)
    # TRUNCATE CASCADE handles dependencies, but we order for clarity
    tables_to_truncate = [
        "face_assignment_events",
        "person_prototypes",
        "face_instances",
        "persons",
        "vector_deletion_logs",
        "training_evidence",
        "training_jobs",
        "training_subdirectories",
        "training_sessions",
        "categories",
        "image_assets",
    ]

    truncated_counts: dict[str, int] = {}

    for table_name in tables_to_truncate:
        try:
            # Get row count before truncation
            count_result = await db.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            row_count = count_result.scalar() or 0

            # Try TRUNCATE CASCADE first (PostgreSQL), fall back to DELETE (SQLite)
            try:
                await db.execute(text(f"TRUNCATE TABLE {table_name} CASCADE"))
            except Exception:
                # SQLite doesn't support TRUNCATE, use DELETE instead
                await db.execute(text(f"DELETE FROM {table_name}"))

            truncated_counts[table_name] = row_count
            logger.info(f"Truncated table {table_name}: {row_count} rows")

        except Exception as e:
            logger.error(f"Failed to truncate table {table_name}: {e}")
            truncated_counts[table_name] = 0

    # Commit truncations
    await db.commit()

    return truncated_counts
