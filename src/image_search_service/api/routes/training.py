"""Training session management endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.api.schemas import PaginatedResponse
from image_search_service.api.training_schemas import (
    ControlResponse,
    DirectoryInfo,
    DirectoryScanRequest,
    DirectoryScanResponse,
    SubdirectorySelectionUpdate,
    TrainingJobResponse,
    TrainingProgressResponse,
    TrainingSessionCreate,
    TrainingSessionResponse,
    TrainingSessionUpdate,
    TrainingSubdirectoryResponse,
)
from image_search_service.core.logging import get_logger
from image_search_service.db.models import JobStatus, SessionStatus, TrainingSubdirectory
from image_search_service.db.session import get_db
from image_search_service.services.directory_service import DirectoryService
from image_search_service.services.training_service import TrainingService

logger = get_logger(__name__)
router = APIRouter(prefix="/training", tags=["training"])


# ============================================================================
# Session Management Endpoints
# ============================================================================


@router.post("/sessions", response_model=TrainingSessionResponse)
async def create_session(
    data: TrainingSessionCreate, db: AsyncSession = Depends(get_db)
) -> TrainingSessionResponse:
    """Create a new training session.

    Args:
        data: Session creation data
        db: Database session

    Returns:
        Created training session

    Raises:
        HTTPException: If path validation fails
    """
    # Validate root path
    dir_service = DirectoryService()
    try:
        dir_service.validate_path(data.root_path)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    # Create session
    service = TrainingService()
    session = await service.create_session(db, data)

    return TrainingSessionResponse.model_validate(session)


@router.get("/sessions", response_model=PaginatedResponse[TrainingSessionResponse])
async def list_sessions(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: SessionStatus | None = None,
    db: AsyncSession = Depends(get_db),
) -> PaginatedResponse[TrainingSessionResponse]:
    """List training sessions with pagination.

    Args:
        page: Page number (1-indexed)
        page_size: Number of items per page (max 100)
        status: Optional status filter
        db: Database session

    Returns:
        Paginated list of training sessions
    """
    service = TrainingService()
    sessions, total = await service.list_sessions(
        db, page=page, page_size=page_size, status=status.value if status else None
    )

    return PaginatedResponse(
        items=[TrainingSessionResponse.model_validate(s) for s in sessions],
        total=total,
        page=page,
        pageSize=page_size,
        hasMore=(page * page_size) < total,
    )


@router.get("/sessions/{session_id}", response_model=TrainingSessionResponse)
async def get_session(
    session_id: int, db: AsyncSession = Depends(get_db)
) -> TrainingSessionResponse:
    """Get a single training session by ID.

    Args:
        session_id: Session ID
        db: Database session

    Returns:
        Training session details

    Raises:
        HTTPException: If session not found
    """
    service = TrainingService()
    session = await service.get_session(db, session_id)

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training session {session_id} not found",
        )

    return TrainingSessionResponse.model_validate(session)


@router.patch("/sessions/{session_id}", response_model=TrainingSessionResponse)
async def update_session(
    session_id: int,
    data: TrainingSessionUpdate,
    db: AsyncSession = Depends(get_db),
) -> TrainingSessionResponse:
    """Update a training session.

    Args:
        session_id: Session ID
        data: Update data
        db: Database session

    Returns:
        Updated training session

    Raises:
        HTTPException: If session not found
    """
    service = TrainingService()
    try:
        session = await service.update_session(db, session_id, data)
        return TrainingSessionResponse.model_validate(session)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: int, db: AsyncSession = Depends(get_db)
) -> None:
    """Delete a training session.

    Args:
        session_id: Session ID
        db: Database session

    Raises:
        HTTPException: If session not found
    """
    service = TrainingService()
    deleted = await service.delete_session(db, session_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training session {session_id} not found",
        )


# ============================================================================
# Directory Scanning Endpoints
# ============================================================================


@router.post("/directories/scan", response_model=DirectoryScanResponse)
async def scan_directory(data: DirectoryScanRequest) -> DirectoryScanResponse:
    """Scan a directory for subdirectories with image files.

    Args:
        data: Directory scan request

    Returns:
        Scan results with subdirectory information

    Raises:
        HTTPException: If path validation fails
    """
    dir_service = DirectoryService()

    try:
        # Validate and scan
        subdirs = dir_service.scan_directory(
            data.root_path, data.recursive, data.extensions
        )

        # Calculate totals
        total_subdirs = len(subdirs)
        total_images = sum(s.image_count for s in subdirs)

        return DirectoryScanResponse(
            rootPath=data.root_path,
            subdirectories=subdirs,
            totalSubdirectories=total_subdirs,
            totalImages=total_images,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get("/directories", response_model=list[DirectoryInfo])
async def list_directories(
    path: str = Query(..., description="Root directory path"),
    include_training_status: bool = Query(
        False,
        description="Include training status metadata (requires DB lookup)",
    ),
    db: AsyncSession = Depends(get_db),
) -> list[DirectoryInfo]:
    """List subdirectories at a given path.

    Args:
        path: Root directory path
        include_training_status: If True, include training status metadata
        db: Database session

    Returns:
        List of subdirectory information

    Raises:
        HTTPException: If path validation fails
    """
    dir_service = DirectoryService()

    try:
        subdirs = dir_service.list_subdirectories(path)

        # Optionally enrich with training status
        if include_training_status:
            training_service = TrainingService()
            subdirs = await training_service.enrich_with_training_status(db, subdirs, path)

        return subdirs
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


# ============================================================================
# Subdirectory Management Endpoints
# ============================================================================


@router.get(
    "/sessions/{session_id}/subdirectories",
    response_model=list[TrainingSubdirectoryResponse],
)
async def get_subdirectories(
    session_id: int, db: AsyncSession = Depends(get_db)
) -> list[TrainingSubdirectoryResponse]:
    """Get subdirectories for a training session.

    Args:
        session_id: Session ID
        db: Database session

    Returns:
        List of subdirectories for the session

    Raises:
        HTTPException: If session not found
    """
    # Verify session exists
    service = TrainingService()
    session = await service.get_session(db, session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training session {session_id} not found",
        )

    # Return subdirectories from relationship
    return [
        TrainingSubdirectoryResponse.model_validate(s) for s in session.subdirectories
    ]


@router.patch(
    "/sessions/{session_id}/subdirectories",
    response_model=list[TrainingSubdirectoryResponse],
)
async def update_subdirectories(
    session_id: int,
    selections: list[SubdirectorySelectionUpdate],
    db: AsyncSession = Depends(get_db),
) -> list[TrainingSubdirectoryResponse]:
    """Update subdirectory selections for a training session.

    Args:
        session_id: Session ID
        selections: List of subdirectory selection updates
        db: Database session

    Returns:
        Updated list of subdirectories

    Raises:
        HTTPException: If session not found or subdirectory IDs invalid
    """
    # Verify session exists
    service = TrainingService()
    session = await service.get_session(db, session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training session {session_id} not found",
        )

    # Update selections
    for selection in selections:
        query = select(TrainingSubdirectory).where(
            TrainingSubdirectory.id == selection.id,
            TrainingSubdirectory.session_id == session_id,
        )
        result = await db.execute(query)
        subdir = result.scalar_one_or_none()

        if not subdir:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Subdirectory {selection.id} not found in session {session_id}",
            )

        subdir.selected = selection.selected

    await db.commit()

    # Refresh and return updated subdirectories
    await db.refresh(session)
    return [
        TrainingSubdirectoryResponse.model_validate(s) for s in session.subdirectories
    ]


# ============================================================================
# Progress Monitoring Endpoints
# ============================================================================


@router.get("/sessions/{session_id}/progress", response_model=TrainingProgressResponse)
async def get_progress(
    session_id: int, db: AsyncSession = Depends(get_db)
) -> TrainingProgressResponse:
    """Get progress information for a training session.

    Args:
        session_id: Session ID
        db: Database session

    Returns:
        Progress information with stats and job summary

    Raises:
        HTTPException: If session not found
    """
    service = TrainingService()
    try:
        return await service.get_session_progress(db, session_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


# ============================================================================
# Training Control Endpoints
# ============================================================================


@router.post("/sessions/{session_id}/start", response_model=ControlResponse)
async def start_training(
    session_id: int, db: AsyncSession = Depends(get_db)
) -> ControlResponse:
    """Start or resume training for a session.

    Valid state transitions:
    - pending → running (initial start)
    - paused → running (resume)
    - failed → running (retry after failure)

    Args:
        session_id: Session ID
        db: Database session

    Returns:
        Control response with updated session status

    Raises:
        HTTPException: If session not found or invalid state transition
    """
    service = TrainingService()
    try:
        session = await service.start_training(db, session_id)
        return ControlResponse(
            sessionId=session.id,
            status=session.status,
            message=f"Training started for session {session.id}",
        )
    except ValueError as e:
        error_msg = str(e)
        if "not found" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_msg,
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg,
        )


@router.post("/sessions/{session_id}/pause", response_model=ControlResponse)
async def pause_training(
    session_id: int, db: AsyncSession = Depends(get_db)
) -> ControlResponse:
    """Pause a running training session.

    The background worker will detect the pause and stop gracefully.

    Args:
        session_id: Session ID
        db: Database session

    Returns:
        Control response with updated session status

    Raises:
        HTTPException: If session not found or not running
    """
    service = TrainingService()
    try:
        session = await service.pause_training(db, session_id)
        return ControlResponse(
            sessionId=session.id,
            status=session.status,
            message=f"Training paused for session {session.id}",
        )
    except ValueError as e:
        error_msg = str(e)
        if "not found" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_msg,
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg,
        )


@router.post("/sessions/{session_id}/cancel", response_model=ControlResponse)
async def cancel_training(
    session_id: int, db: AsyncSession = Depends(get_db)
) -> ControlResponse:
    """Cancel a training session.

    Cancels all pending jobs and updates session status.

    Args:
        session_id: Session ID
        db: Database session

    Returns:
        Control response with updated session status

    Raises:
        HTTPException: If session not found or invalid state
    """
    service = TrainingService()
    try:
        session = await service.cancel_training(db, session_id)
        return ControlResponse(
            sessionId=session.id,
            status=session.status,
            message=f"Training cancelled for session {session.id}",
        )
    except ValueError as e:
        error_msg = str(e)
        if "not found" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_msg,
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg,
        )


@router.post("/sessions/{session_id}/restart", response_model=ControlResponse)
async def restart_training(
    session_id: int,
    failed_only: bool = Query(True, description="Only restart failed images"),
    db: AsyncSession = Depends(get_db),
) -> ControlResponse:
    """Restart training for failed or all images.

    By default, only restarts failed jobs. Set failed_only=False to restart all jobs.

    Args:
        session_id: Session ID
        failed_only: If True, only restart failed jobs. If False, restart all jobs.
        db: Database session

    Returns:
        Control response with updated session status

    Raises:
        HTTPException: If session not found
    """
    service = TrainingService()
    try:
        session = await service.restart_training(db, session_id, failed_only)
        restart_type = "failed jobs" if failed_only else "all jobs"
        return ControlResponse(
            sessionId=session.id,
            status=session.status,
            message=f"Training restarted for session {session.id} ({restart_type})",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


# ============================================================================
# Job Listing Endpoints
# ============================================================================


@router.get(
    "/sessions/{session_id}/jobs",
    response_model=PaginatedResponse[TrainingJobResponse],
)
async def list_jobs(
    session_id: int,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    job_status: JobStatus | None = Query(None, alias="status"),
    db: AsyncSession = Depends(get_db),
) -> PaginatedResponse[TrainingJobResponse]:
    """List training jobs for a session with optional status filter.

    Args:
        session_id: Session ID
        page: Page number (1-indexed)
        page_size: Number of items per page (max 100)
        job_status: Optional status filter
        db: Database session

    Returns:
        Paginated list of training jobs

    Raises:
        HTTPException: If session not found
    """
    service = TrainingService()

    # Verify session exists
    session = await service.get_session(db, session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training session {session_id} not found",
        )

    jobs, total = await service.list_jobs(db, session_id, page, page_size, job_status)

    return PaginatedResponse(
        items=[TrainingJobResponse.model_validate(j) for j in jobs],
        total=total,
        page=page,
        pageSize=page_size,
        hasMore=(page * page_size) < total,
    )


@router.get("/jobs/{job_id}", response_model=TrainingJobResponse)
async def get_job(
    job_id: int, db: AsyncSession = Depends(get_db)
) -> TrainingJobResponse:
    """Get details of a specific training job.

    Args:
        job_id: Training job ID
        db: Database session

    Returns:
        Training job details

    Raises:
        HTTPException: If job not found
    """
    service = TrainingService()
    job = await service.get_job(db, job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job {job_id} not found",
        )

    return TrainingJobResponse.model_validate(job)


# ============================================================================
# Thumbnail Generation Endpoints
# ============================================================================


@router.post("/sessions/{session_id}/thumbnails")
async def generate_session_thumbnails(
    session_id: int, db: AsyncSession = Depends(get_db)
) -> dict[str, object]:
    """Generate thumbnails for all images in a training session.

    Enqueues background job for thumbnail generation.

    Args:
        session_id: Training session ID
        db: Database session

    Returns:
        Job information with job ID and total images

    Raises:
        HTTPException: If session not found
    """
    from redis import Redis
    from rq import Queue

    from image_search_service.core.config import get_settings

    # Verify session exists
    service = TrainingService()
    session = await service.get_session(db, session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training session {session_id} not found",
        )

    # Get total image count
    total_images = session.total_images

    # Enqueue background job
    settings = get_settings()
    redis_conn = Redis.from_url(settings.redis_url)
    queue = Queue("default", connection=redis_conn)

    from image_search_service.queue.thumbnail_jobs import generate_thumbnails_batch

    job = queue.enqueue(
        generate_thumbnails_batch,
        session_id,
        job_timeout="1h",  # Thumbnails can take time for large sessions
    )

    logger.info(
        f"Enqueued thumbnail generation job {job.id} for session {session_id} "
        f"({total_images} images)"
    )

    return {
        "jobId": job.id,
        "sessionId": session_id,
        "totalImages": total_images,
        "status": "queued",
    }


# ============================================================================
# Incremental Scanning Endpoints
# ============================================================================


@router.post("/scan/incremental")
async def trigger_incremental_scan(
    directory: str = Query(..., description="Directory to scan"),
    auto_train: bool = Query(False, description="Auto-enqueue detected images for training"),
) -> dict[str, object]:
    """Trigger incremental scan of a directory for new images.

    Scans directory for image files and creates ImageAsset records for new images only.
    Optionally enqueues newly detected images for training.

    Args:
        directory: Directory path to scan
        auto_train: If True, auto-enqueue detected images for training

    Returns:
        Dictionary with scan statistics (discovered, created, skipped)
    """
    from pathlib import Path

    from image_search_service.queue.auto_detection_jobs import scan_directory_incremental

    # Validate directory path
    dir_service = DirectoryService()
    try:
        dir_service.validate_path(directory)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    # Check directory exists
    dir_path = Path(directory)
    if not dir_path.exists() or not dir_path.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Directory not found: {directory}",
        )

    # Run incremental scan synchronously (fast operation)
    result = scan_directory_incremental(directory, auto_train=auto_train)

    logger.info(
        f"Incremental scan completed for {directory}: "
        f"discovered={result.get('discovered', 0)}, "
        f"created={result.get('created', 0)}, "
        f"skipped={result.get('skipped', 0)}"
    )

    return result
