"""Training session management endpoints."""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.api.schemas import PaginatedResponse
from image_search_service.api.training_schemas import (
    ClusteringRestartResponse,
    ControlResponse,
    DirectoryImageInfo,
    DirectoryInfo,
    DirectoryPreviewResponse,
    DirectoryScanRequest,
    DirectoryScanResponse,
    FaceDetectionRestartResponse,
    IgnoredDirectoriesResponse,
    IgnoredDirectoryInfo,
    IgnoreDirectoryRequest,
    IgnoreDirectoryResponse,
    SubdirectorySelectionUpdate,
    TrainingJobResponse,
    TrainingProgressResponse,
    TrainingRestartResponse,
    TrainingSessionCreate,
    TrainingSessionResponse,
    TrainingSessionUpdate,
    TrainingSubdirectoryResponse,
    UnifiedProgressResponse,
)
from image_search_service.core.logging import get_logger
from image_search_service.db.models import (
    IgnoredDirectory,
    JobStatus,
    SessionStatus,
    TrainingSubdirectory,
)
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
    include_ignored: bool = Query(
        False,
        description="Include ignored directories in results (default: false)",
    ),
    db: AsyncSession = Depends(get_db),
) -> list[DirectoryInfo]:
    """List subdirectories at a given path.

    Args:
        path: Root directory path
        include_training_status: If True, include training status metadata
        include_ignored: If True, include ignored directories (default: False)
        db: Database session

    Returns:
        List of subdirectory information (filtered by ignore status)

    Raises:
        HTTPException: If path validation fails
    """
    dir_service = DirectoryService()

    try:
        subdirs = dir_service.list_subdirectories(path)

        # Filter out ignored directories unless explicitly requested
        if not include_ignored:
            # Get list of ignored directory paths
            query = select(IgnoredDirectory.path)
            result = await db.execute(query)
            ignored_paths = set(result.scalars().all())

            # Filter subdirectories
            subdirs = [s for s in subdirs if s.path not in ignored_paths]

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


@router.get(
    "/sessions/{session_id}/progress-unified",
    response_model=UnifiedProgressResponse,
    summary="Get unified progress across all training phases",
    description=(
        "Returns combined progress information for training (CLIP embeddings), "
        "face detection (InsightFace), and clustering (HDBSCAN) phases. "
        "Progress is weighted: 30% training, 65% face detection, 5% clustering."
    ),
)
async def get_unified_progress(
    session_id: int,
    db: AsyncSession = Depends(get_db),
) -> UnifiedProgressResponse:
    """Get unified progress across all training phases.

    This endpoint provides a single progress value (0-100%) that accounts for:
    - Phase 1: CLIP embedding generation (30% weight)
    - Phase 2: Face detection with InsightFace (65% weight)
    - Phase 3: HDBSCAN clustering (5% weight)

    Returns detailed progress for each phase along with overall status.

    Args:
        session_id: Training session ID
        db: Database session

    Returns:
        Unified progress response

    Raises:
        HTTPException: If session not found
    """
    from typing import cast

    service = TrainingService()
    try:
        result = await service.get_session_progress_unified(db, session_id)
        return cast(UnifiedProgressResponse, result)
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
# Restart Service Endpoints (Phase 1, 2, 3)
# ============================================================================


@router.post(
    "/sessions/{session_id}/restart-training",
    response_model=TrainingRestartResponse,
    summary="Restart training (Phase 1: CLIP embeddings)",
)
async def restart_training_phase1(
    session_id: int,
    failed_only: bool = Query(
        default=True, description="Reset only failed jobs (true) or all jobs (false)"
    ),
    db: AsyncSession = Depends(get_db),
) -> TrainingRestartResponse:
    """Restart Phase 1 training (CLIP embedding generation).

    Options:
    - failed_only=true: Retry only failed images (default)
    - failed_only=false: Full restart (all images)

    What this does:
    - Resets training jobs to PENDING state
    - Clears error messages and processing times
    - Enqueues training job for background processing
    - Existing Qdrant vectors will be overwritten (upsert)

    Safety notes:
    - Only restarts Phase 1 (CLIP embeddings)
    - Face detection and clustering are unaffected
    - Idempotent: safe to call multiple times
    - Rollback on failure

    State requirements:
    - Training status: COMPLETED, FAILED, or CANCELLED
    - No active training jobs running

    Args:
        session_id: Training session ID
        failed_only: Reset only failed jobs (true) or all jobs (false)
        db: Database session

    Returns:
        Restart response with cleanup statistics

    Raises:
        HTTPException:
            - 400: Invalid state (RUNNING or PENDING)
            - 404: Session not found
            - 409: Active jobs still running
    """
    from image_search_service.services import TrainingRestartService

    service = TrainingRestartService(failed_only=failed_only)

    try:
        stats = await service.restart(db, session_id)

        restart_type = "failed jobs only" if failed_only else "full restart"
        return TrainingRestartResponse(
            sessionId=session_id,
            status="pending",
            message=f"Training restart initiated ({restart_type})",
            cleanupStats=dict(stats),
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.post(
    "/sessions/{session_id}/restart-faces",
    response_model=FaceDetectionRestartResponse,
    summary="Restart face detection (Phase 2: InsightFace)",
)
async def restart_face_detection_phase2(
    session_id: int,
    delete_persons: bool = Query(
        default=False, description="Delete orphaned Person records after face cleanup"
    ),
    db: AsyncSession = Depends(get_db),
) -> FaceDetectionRestartResponse:
    """Restart Phase 2 face detection (InsightFace face detection and embedding).

    Options:
    - delete_persons=false: Preserve Person records (orphan them) - DEFAULT, SAFE
    - delete_persons=true: Delete Person records with no remaining faces - DESTRUCTIVE

    What this does:
    - Deletes all Face records for this session
    - Resets face detection counters to zero
    - Optionally deletes orphaned Person records (use with caution)
    - Enqueues face detection job for background processing

    Safety notes:
    - Only affects Phase 2 (face detection)
    - CLIP embeddings (Phase 1) are preserved
    - Clustering (Phase 3) will need to be rerun after face detection completes
    - Idempotent: safe to call multiple times
    - Rollback on failure

    State requirements:
    - Face detection status: COMPLETED, FAILED, or CANCELLED
    - No active face detection jobs running

    Args:
        session_id: Training session ID
        delete_persons: Delete orphaned Person records (destructive)
        db: Database session

    Returns:
        Restart response with cleanup statistics

    Raises:
        HTTPException:
            - 400: Invalid state (PROCESSING or PENDING)
            - 404: Session not found
            - 409: Active jobs still running
    """
    from image_search_service.services import FaceDetectionRestartService

    service = FaceDetectionRestartService(delete_orphaned_persons=delete_persons)

    try:
        stats = await service.restart(db, session_id)

        return FaceDetectionRestartResponse(
            sessionId=session_id,
            status="pending",
            message=(
                f"Face detection restart initiated "
                f"({'with Person deletion' if delete_persons else 'preserving Persons'})"
            ),
            cleanupStats=dict(stats),
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.post(
    "/sessions/{session_id}/restart-clustering",
    response_model=ClusteringRestartResponse,
    summary="Restart clustering (Phase 3: HDBSCAN)",
)
async def restart_clustering_phase3(
    session_id: int,
    db: AsyncSession = Depends(get_db),
) -> ClusteringRestartResponse:
    """Restart Phase 3 clustering (HDBSCAN face clustering and person assignment).

    What this does:
    - Unassigns all faces (sets person_id to NULL)
    - Resets face cluster assignments
    - Runs clustering synchronously (fast operation, ~1-2 seconds)
    - Returns completed status immediately

    Safety notes:
    - Only affects Phase 3 (clustering)
    - CLIP embeddings (Phase 1) are preserved
    - Face embeddings (Phase 2) are preserved
    - Idempotent: safe to call multiple times
    - No rollback needed (synchronous operation)

    State requirements:
    - Clustering can run at any time after face detection completes
    - No state validation needed (clustering is always safe to rerun)

    Args:
        session_id: Training session ID
        db: Database session

    Returns:
        Restart response with cleanup statistics (status="completed")

    Raises:
        HTTPException:
            - 400: Invalid state or validation error
            - 404: Session not found
            - 409: Operation conflict
    """
    from image_search_service.services import FaceClusteringRestartService

    service = FaceClusteringRestartService()

    try:
        stats = await service.restart(db, session_id)

        return ClusteringRestartResponse(
            sessionId=session_id,
            status="completed",
            message="Clustering restart completed",
            cleanupStats=dict(stats),
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


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


# ============================================================================
# Directory Image Preview Endpoints (for non-ingested images)
# ============================================================================


@router.get("/directories/preview", response_model=DirectoryPreviewResponse)
async def list_directory_preview_images(
    path: str = Query(..., description="Directory path to preview")
) -> DirectoryPreviewResponse:
    """List images in a directory for preview BEFORE ingestion.

    Returns image metadata without creating database records.
    Images are NOT treated as ingested.

    Args:
        path: Directory path (must be under IMAGE_ROOT_DIR)

    Returns:
        List of image files with metadata

    Raises:
        HTTPException: If path validation fails or directory not found
    """
    from datetime import datetime

    # Validate directory path (security)
    dir_service = DirectoryService()
    try:
        dir_service.validate_path(path)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    # List image files
    dir_path = Path(path)
    allowed_extensions = {"jpg", "jpeg", "png", "webp"}
    image_files: list[DirectoryImageInfo] = []

    try:
        for file_path in sorted(dir_path.iterdir()):
            # Skip non-files
            if not file_path.is_file():
                continue

            # Check extension
            ext = file_path.suffix.lower().lstrip(".")
            if ext not in allowed_extensions:
                continue

            # Get file metadata
            stat = file_path.stat()

            # Store full path for thumbnail endpoint
            full_path = str(file_path)

            image_files.append(
                DirectoryImageInfo(
                    filename=file_path.name,
                    fullPath=full_path,
                    sizeBytes=stat.st_size,
                    modifiedAt=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                )
            )

        return DirectoryPreviewResponse(
            directory=path,
            imageCount=len(image_files),
            images=image_files,
        )

    except Exception as e:
        logger.error(f"Failed to list directory preview images: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list images: {str(e)}",
        )


@router.get("/directories/preview/thumbnail")
async def get_directory_preview_thumbnail(
    path: str = Query(..., description="Full path to image file (URL-encoded)")
) -> FileResponse:
    """Serve thumbnail for a non-ingested image.

    Generates thumbnail on-the-fly if needed, caches for reuse.
    Images are NOT treated as ingested (no database records created).

    Args:
        path: Full path to image file (must be under IMAGE_ROOT_DIR)

    Returns:
        Thumbnail image (JPEG, max 400x400, maintains aspect ratio)

    Raises:
        HTTPException: If path invalid, file not found, or unsupported type
    """
    import hashlib
    from urllib.parse import unquote

    from PIL import Image, ImageOps

    from image_search_service.core.config import get_settings

    settings = get_settings()

    # Decode path
    file_path = Path(unquote(path))

    # Validate file exists
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image file not found: {path}",
        )

    # Security: Validate path is within IMAGE_ROOT_DIR
    if settings.image_root_dir:
        allowed_dirs = [Path(settings.image_root_dir)]
        # Use security validation from images.py
        try:
            abs_path = file_path.resolve()
            found_allowed = False
            for allowed_dir in allowed_dirs:
                try:
                    abs_path.relative_to(allowed_dir.resolve())
                    found_allowed = True
                    break
                except ValueError:
                    continue
            if not found_allowed:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access to file path not allowed",
                )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file path: {e}",
            )

    # Validate extension
    allowed_extensions = {"jpg", "jpeg", "png", "webp"}
    ext = file_path.suffix.lower().lstrip(".")
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file_path.suffix}",
        )

    # Generate hash-based thumbnail path
    path_hash = hashlib.md5(str(file_path).encode()).hexdigest()
    shard_prefix = path_hash[:2]
    thumbnail_dir = Path(settings.thumbnail_dir) / "preview" / shard_prefix
    thumbnail_dir.mkdir(parents=True, exist_ok=True)
    thumbnail_path = thumbnail_dir / f"{path_hash}.jpg"

    # Generate thumbnail if doesn't exist
    if not thumbnail_path.exists():
        try:
            logger.info(f"Generating preview thumbnail for {file_path}")

            # Open and process image (similar to ThumbnailService)
            with Image.open(file_path) as img:
                # Apply EXIF orientation
                img = ImageOps.exif_transpose(img) or img

                # Convert to RGB if needed
                if img.mode in ("RGBA", "P", "LA"):
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    if img.mode == "P":
                        img = img.convert("RGBA")
                    if img.mode == "RGBA":
                        background.paste(img, mask=img.split()[-1])
                    else:
                        background.paste(img)
                    img = background
                elif img.mode != "RGB":
                    img = img.convert("RGB")

                # Resize to thumbnail (400x400 max as per requirements)
                max_size = 400
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

                # Save as JPEG
                img.save(thumbnail_path, "JPEG", quality=85, optimize=True)

        except Exception as e:
            logger.error(f"Failed to generate preview thumbnail: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate thumbnail: {str(e)}",
            )

    # Serve thumbnail with cache headers
    return FileResponse(
        path=thumbnail_path,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "public, max-age=86400",  # 24-hour cache
            "ETag": f'"{path_hash}"',
        },
    )


# ============================================================================
# Directory Ignore Endpoints
# ============================================================================


@router.post("/directories/ignore", response_model=IgnoreDirectoryResponse)
async def ignore_directory(
    data: IgnoreDirectoryRequest, db: AsyncSession = Depends(get_db)
) -> IgnoreDirectoryResponse:
    """Mark a directory as ignored (excluded from directory listings).

    Args:
        data: Directory ignore request with path and optional reason
        db: Database session

    Returns:
        Ignore operation response

    Raises:
        HTTPException: If path validation fails or directory already ignored
    """
    # Validate directory path (security check)
    dir_service = DirectoryService()
    try:
        dir_service.validate_path(data.path)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    # Normalize path (resolve to absolute path)
    normalized_path = str(Path(data.path).resolve())

    # Check if already ignored
    query = select(IgnoredDirectory).where(IgnoredDirectory.path == normalized_path)
    result = await db.execute(query)
    existing = result.scalar_one_or_none()

    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Directory already ignored: {normalized_path}",
        )

    # Create ignored directory record
    ignored_dir = IgnoredDirectory(
        path=normalized_path,
        reason=data.reason,
        ignored_by=None,  # TODO: Add user context when auth is implemented
    )
    db.add(ignored_dir)
    await db.commit()
    await db.refresh(ignored_dir)

    logger.info(f"Directory marked as ignored: {normalized_path}")

    return IgnoreDirectoryResponse(
        status="success",
        path=ignored_dir.path,
        ignoredAt=ignored_dir.ignored_at,
    )


@router.delete("/directories/ignore", response_model=IgnoreDirectoryResponse)
async def unignore_directory(
    data: IgnoreDirectoryRequest, db: AsyncSession = Depends(get_db)
) -> IgnoreDirectoryResponse:
    """Remove a directory from the ignore list.

    Args:
        data: Directory unignore request with path
        db: Database session

    Returns:
        Unignore operation response

    Raises:
        HTTPException: If directory not found in ignore list
    """
    # Normalize path
    normalized_path = str(Path(data.path).resolve())

    # Find and delete ignored directory record
    query = select(IgnoredDirectory).where(IgnoredDirectory.path == normalized_path)
    result = await db.execute(query)
    ignored_dir = result.scalar_one_or_none()

    if not ignored_dir:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Directory not in ignore list: {normalized_path}",
        )

    await db.delete(ignored_dir)
    await db.commit()

    logger.info(f"Directory removed from ignore list: {normalized_path}")

    return IgnoreDirectoryResponse(
        status="success",
        path=normalized_path,
        ignoredAt=None,
    )


@router.get("/directories/ignored", response_model=IgnoredDirectoriesResponse)
async def list_ignored_directories(
    db: AsyncSession = Depends(get_db),
) -> IgnoredDirectoriesResponse:
    """List all ignored directories.

    Args:
        db: Database session

    Returns:
        List of ignored directories with metadata
    """
    query = select(IgnoredDirectory).order_by(IgnoredDirectory.path)
    result = await db.execute(query)
    ignored_dirs = result.scalars().all()

    return IgnoredDirectoriesResponse(
        directories=[IgnoredDirectoryInfo.model_validate(d) for d in ignored_dirs]
    )
