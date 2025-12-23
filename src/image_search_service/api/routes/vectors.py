"""Vector management endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from image_search_service.api.vector_schemas import (
    AssetDeleteResponse,
    CategoryDeleteResponse,
    DeletionLogEntry,
    DeletionLogsResponse,
    DirectoryDeleteRequest,
    DirectoryDeleteResponse,
    DirectoryStats,
    DirectoryStatsResponse,
    OrphanCleanupRequest,
    OrphanCleanupResponse,
    ResetRequest,
    ResetResponse,
    RetrainRequest,
    RetrainResponse,
    SessionDeleteResponse,
)
from image_search_service.core.logging import get_logger
from image_search_service.db.models import (
    Category,
    ImageAsset,
    TrainingJob,
    TrainingSession,
    VectorDeletionLog,
)
from image_search_service.db.session import get_db
from image_search_service.vector import qdrant

logger = get_logger(__name__)
router = APIRouter(prefix="/vectors", tags=["vectors"])


@router.delete("/by-directory", response_model=DirectoryDeleteResponse)
async def delete_by_directory(
    request: DirectoryDeleteRequest,
    db: AsyncSession = Depends(get_db),
) -> DirectoryDeleteResponse:
    """Delete vectors matching directory path prefix.

    Safety: Requires confirm=true

    Args:
        request: Directory deletion request with path prefix and confirmation
        db: Database session

    Returns:
        Deletion result with count and message

    Raises:
        HTTPException: 400 if confirmation not provided
    """
    if not request.confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must confirm deletion by setting confirm=true",
        )

    # Delete vectors from Qdrant
    deleted_count = qdrant.delete_vectors_by_directory(request.path_prefix)

    # Log deletion
    log = VectorDeletionLog(
        deletion_type="DIRECTORY",
        deletion_target=request.path_prefix,
        vector_count=deleted_count,
        deletion_reason=request.deletion_reason,
    )
    db.add(log)
    await db.commit()

    logger.info(
        f"Deleted {deleted_count} vectors for directory '{request.path_prefix}' "
        f"(log_id={log.id})"
    )

    return DirectoryDeleteResponse(
        pathPrefix=request.path_prefix,
        vectorsDeleted=deleted_count,
        message=f"Successfully deleted {deleted_count} vectors",
    )


@router.post("/retrain", response_model=RetrainResponse)
async def retrain_directory(
    request: RetrainRequest,
    db: AsyncSession = Depends(get_db),
) -> RetrainResponse:
    """Delete existing vectors and create new training session.

    Safety: Validates category exists

    Three-step operation:
    1. Delete all vectors for directory
    2. Create new training session with PENDING status
    3. Return new session ID for training

    Args:
        request: Retrain request with path prefix and category ID
        db: Database session

    Returns:
        Retrain result with deletion count and new session ID

    Raises:
        HTTPException: 404 if category not found
    """
    # Verify category exists
    category_query = select(Category).where(Category.id == request.category_id)
    result = await db.execute(category_query)
    category = result.scalar_one_or_none()

    if not category:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Category {request.category_id} not found",
        )

    # Delete vectors from Qdrant
    deleted_count = qdrant.delete_vectors_by_directory(request.path_prefix)

    # Log deletion
    log = VectorDeletionLog(
        deletion_type="DIRECTORY",
        deletion_target=request.path_prefix,
        vector_count=deleted_count,
        deletion_reason=request.deletion_reason or "Retrain operation",
    )
    db.add(log)

    # Create new training session
    new_session = TrainingSession(
        name=f"Retrain: {request.path_prefix}",
        root_path=request.path_prefix,
        category_id=request.category_id,
        status="PENDING",
    )
    db.add(new_session)
    await db.commit()

    # Reload session with category relationship
    query = (
        select(TrainingSession)
        .where(TrainingSession.id == new_session.id)
        .options(selectinload(TrainingSession.category))
    )
    result = await db.execute(query)
    refreshed_session = result.scalar_one()

    logger.info(
        f"Retrain: deleted {deleted_count} vectors, created session {refreshed_session.id} "
        f"for '{request.path_prefix}'"
    )

    return RetrainResponse(
        pathPrefix=request.path_prefix,
        vectorsDeleted=deleted_count,
        newSessionId=refreshed_session.id,
        message=f"Deleted {deleted_count} vectors and created new training session",
    )


@router.get("/directories/stats", response_model=DirectoryStatsResponse)
async def get_directory_stats() -> DirectoryStatsResponse:
    """Get statistics about directories with vectors.

    Returns aggregated view of directory paths and vector counts.

    Returns:
        Directory statistics with vector counts
    """
    # Get directory stats from Qdrant
    stats = qdrant.get_directory_stats()

    # Convert to response format
    directories = [
        DirectoryStats(
            pathPrefix=stat["path_prefix"],
            vectorCount=stat["vector_count"],
            lastIndexed=stat.get("last_indexed"),
        )
        for stat in stats
    ]

    total_vectors = sum(d.vector_count for d in directories)

    logger.info(f"Retrieved stats for {len(directories)} directories ({total_vectors} vectors)")

    return DirectoryStatsResponse(
        directories=directories,
        totalVectors=total_vectors,
    )


@router.delete("/by-asset/{asset_id}", response_model=AssetDeleteResponse)
async def delete_by_asset(
    asset_id: int,
    db: AsyncSession = Depends(get_db),
) -> AssetDeleteResponse:
    """Delete vector for a single asset.

    No confirmation required for single asset deletion.

    Args:
        asset_id: Asset ID to delete
        db: Database session

    Returns:
        Deletion result

    Raises:
        HTTPException: 404 if asset not found
    """
    # Verify asset exists
    asset_query = select(ImageAsset).where(ImageAsset.id == asset_id)
    result = await db.execute(asset_query)
    asset = result.scalar_one_or_none()

    if not asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Asset {asset_id} not found",
        )

    # Delete vector
    deleted_count = qdrant.delete_vectors_by_asset(asset_id)

    # Log deletion
    log = VectorDeletionLog(
        deletion_type="ASSET",
        deletion_target=str(asset_id),
        vector_count=deleted_count,
    )
    db.add(log)
    await db.commit()

    logger.info(f"Deleted vector for asset {asset_id} (log_id={log.id})")

    return AssetDeleteResponse(
        assetId=asset_id,
        vectorsDeleted=deleted_count,
        message=f"Successfully deleted vector for asset {asset_id}",
    )


@router.delete("/by-session/{session_id}", response_model=SessionDeleteResponse)
async def delete_by_session(
    session_id: int,
    db: AsyncSession = Depends(get_db),
) -> SessionDeleteResponse:
    """Delete all vectors from a training session.

    Marks session as reset and deletes all associated vectors.

    Args:
        session_id: Training session ID
        db: Database session

    Returns:
        Deletion result with count

    Raises:
        HTTPException: 404 if session not found
    """
    # Verify session exists
    session_query = select(TrainingSession).where(TrainingSession.id == session_id)
    result = await db.execute(session_query)
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training session {session_id} not found",
        )

    # Get asset IDs from session
    asset_query = (
        select(TrainingJob.asset_id)
        .where(TrainingJob.session_id == session_id)
        .distinct()
    )
    asset_result = await db.execute(asset_query)
    asset_ids = [row[0] for row in asset_result.all()]

    # Delete vectors
    deleted_count = qdrant.delete_vectors_by_session(session_id, asset_ids)

    # Mark session as reset
    session.reset_at = func.now()
    session.reset_reason = "Manual session vector deletion"

    # Log deletion
    log = VectorDeletionLog(
        deletion_type="SESSION",
        deletion_target=str(session_id),
        vector_count=deleted_count,
        metadata_json={"asset_count": len(asset_ids)},
    )
    db.add(log)
    await db.commit()

    logger.info(
        f"Deleted {deleted_count} vectors for session {session_id} "
        f"({len(asset_ids)} assets, log_id={log.id})"
    )

    return SessionDeleteResponse(
        sessionId=session_id,
        vectorsDeleted=deleted_count,
        message=f"Successfully deleted {deleted_count} vectors from session",
    )


@router.delete("/by-category/{category_id}", response_model=CategoryDeleteResponse)
async def delete_by_category(
    category_id: int,
    db: AsyncSession = Depends(get_db),
) -> CategoryDeleteResponse:
    """Delete all vectors in a category.

    Deletes vectors for all training sessions in the category.

    Args:
        category_id: Category ID
        db: Database session

    Returns:
        Deletion result with count

    Raises:
        HTTPException: 404 if category not found
    """
    # Verify category exists
    category_query = select(Category).where(Category.id == category_id)
    result = await db.execute(category_query)
    category = result.scalar_one_or_none()

    if not category:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Category {category_id} not found",
        )

    # Delete vectors from Qdrant
    deleted_count = qdrant.delete_vectors_by_category(category_id)

    # Log deletion
    log = VectorDeletionLog(
        deletion_type="CATEGORY",
        deletion_target=str(category_id),
        vector_count=deleted_count,
        metadata_json={"category_name": category.name},
    )
    db.add(log)
    await db.commit()

    logger.info(
        f"Deleted {deleted_count} vectors for category {category_id} "
        f"({category.name}, log_id={log.id})"
    )

    return CategoryDeleteResponse(
        categoryId=category_id,
        vectorsDeleted=deleted_count,
        message=f"Successfully deleted {deleted_count} vectors from category",
    )


@router.post("/cleanup-orphans", response_model=OrphanCleanupResponse)
async def cleanup_orphans(
    request: OrphanCleanupRequest,
    db: AsyncSession = Depends(get_db),
) -> OrphanCleanupResponse:
    """Remove vectors without corresponding database records.

    Safety: Requires confirm=true

    Identifies vectors in Qdrant where asset_id no longer exists
    in PostgreSQL and deletes them.

    Args:
        request: Cleanup request with confirmation
        db: Database session

    Returns:
        Cleanup result with orphan count

    Raises:
        HTTPException: 400 if confirmation not provided
    """
    if not request.confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must confirm cleanup by setting confirm=true",
        )

    # Get all valid asset IDs from database
    asset_query = select(ImageAsset.id)
    result = await db.execute(asset_query)
    valid_asset_ids = {row[0] for row in result.all()}

    logger.info(f"Found {len(valid_asset_ids)} valid assets in database")

    # Delete orphan vectors
    orphans_deleted = qdrant.delete_orphan_vectors(valid_asset_ids)

    # Log deletion
    log = VectorDeletionLog(
        deletion_type="ORPHAN",
        deletion_target="orphaned_vectors",
        vector_count=orphans_deleted,
        deletion_reason=request.deletion_reason,
        metadata_json={"valid_asset_count": len(valid_asset_ids)},
    )
    db.add(log)
    await db.commit()

    logger.info(
        f"Deleted {orphans_deleted} orphan vectors "
        f"({len(valid_asset_ids)} valid assets, log_id={log.id})"
    )

    return OrphanCleanupResponse(
        orphansDeleted=orphans_deleted,
        message=f"Successfully deleted {orphans_deleted} orphan vectors",
    )


@router.post("/reset", response_model=ResetResponse)
async def reset_collection(
    request: ResetRequest,
    db: AsyncSession = Depends(get_db),
) -> ResetResponse:
    """Delete ALL vectors and recreate collection.

    DANGER: This is irreversible and deletes all vector data.

    Safety: Requires confirm=true AND confirmationText="DELETE ALL VECTORS"

    Args:
        request: Reset request with double confirmation
        db: Database session

    Returns:
        Reset result with total deleted count

    Raises:
        HTTPException: 400 if confirmation requirements not met
    """
    if not request.confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must confirm reset by setting confirm=true",
        )

    if request.confirmation_text != "DELETE ALL VECTORS":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="confirmationText must exactly match 'DELETE ALL VECTORS'",
        )

    # Reset collection (delete all and recreate)
    deleted_count = qdrant.reset_collection()

    # Log deletion
    log = VectorDeletionLog(
        deletion_type="FULL_RESET",
        deletion_target="all_vectors",
        vector_count=deleted_count,
        deletion_reason=request.deletion_reason,
    )
    db.add(log)
    await db.commit()

    logger.warning(
        f"FULL RESET: Deleted {deleted_count} vectors from collection "
        f"(log_id={log.id}, reason='{request.deletion_reason}')"
    )

    return ResetResponse(
        vectorsDeleted=deleted_count,
        message=f"Successfully deleted {deleted_count} vectors and reset collection",
    )


@router.get("/deletion-logs", response_model=DeletionLogsResponse)
async def get_deletion_logs(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> DeletionLogsResponse:
    """Get deletion history with pagination.

    Args:
        page: Page number (1-indexed)
        page_size: Number of items per page (max 100)
        db: Database session

    Returns:
        Paginated deletion logs
    """
    # Get total count
    count_query = select(func.count()).select_from(VectorDeletionLog)
    total_result = await db.execute(count_query)
    total = total_result.scalar_one()

    # Get paginated logs
    offset = (page - 1) * page_size
    logs_query = (
        select(VectorDeletionLog)
        .order_by(VectorDeletionLog.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    result = await db.execute(logs_query)
    logs = result.scalars().all()

    # Convert to response format
    log_entries = [
        DeletionLogEntry(
            id=log.id,
            deletionType=log.deletion_type,
            deletionTarget=log.deletion_target,
            vectorCount=log.vector_count,
            deletionReason=log.deletion_reason,
            createdAt=log.created_at,
        )
        for log in logs
    ]

    return DeletionLogsResponse(
        logs=log_entries,
        total=total,
        page=page,
        pageSize=page_size,
    )
