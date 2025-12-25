"""Face detection session API routes."""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.api.face_session_schemas import (
    CreateFaceDetectionSessionRequest,
    FaceDetectionSessionListResponse,
    FaceDetectionSessionResponse,
)
from image_search_service.db.models import (
    FaceDetectionSession,
)
from image_search_service.db.models import (
    FaceDetectionSessionStatus as DBFaceDetectionSessionStatus,
)
from image_search_service.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/faces/sessions", tags=["face-sessions"])


# ============ Session Endpoints ============


@router.get("", response_model=FaceDetectionSessionListResponse)
async def list_sessions(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: str | None = Query(
        None, description="Filter by status: pending, processing, completed, failed"
    ),
    db: AsyncSession = Depends(get_db),
) -> FaceDetectionSessionListResponse:
    """List face detection sessions with pagination."""
    base_query = select(FaceDetectionSession).order_by(FaceDetectionSession.created_at.desc())

    # Filter by status if provided
    if status:
        try:
            status_enum = DBFaceDetectionSessionStatus(status)
            base_query = base_query.where(FaceDetectionSession.status == status_enum.value)
        except ValueError:
            valid_statuses = "pending, processing, completed, failed, paused, cancelled"
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status: {status}. Must be one of: {valid_statuses}",
            )

    # Count total
    count_query = select(func.count()).select_from(base_query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # Get paginated results
    query = base_query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    sessions = result.scalars().all()

    items = [_session_to_response(session) for session in sessions]

    return FaceDetectionSessionListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{session_id}", response_model=FaceDetectionSessionResponse)
async def get_session(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> FaceDetectionSessionResponse:
    """Get face detection session by ID."""
    session = await db.get(FaceDetectionSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    return _session_to_response(session)


@router.post("", response_model=FaceDetectionSessionResponse, status_code=201)
async def create_session(
    request: CreateFaceDetectionSessionRequest,
    db: AsyncSession = Depends(get_db),
) -> FaceDetectionSessionResponse:
    """Create and start a new face detection session."""
    from image_search_service.queue.face_jobs import detect_faces_for_session_job
    from image_search_service.queue.worker import QUEUE_HIGH, get_queue

    # Create session record
    session = FaceDetectionSession(
        training_session_id=request.training_session_id,
        status=DBFaceDetectionSessionStatus.PENDING.value,
        min_confidence=request.min_confidence,
        min_face_size=request.min_face_size,
        batch_size=request.batch_size,
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)

    # Enqueue RQ background job for face detection processing
    queue = get_queue(QUEUE_HIGH)
    job = queue.enqueue(
        detect_faces_for_session_job,
        session_id=str(session.id),
        job_timeout="24h",  # Long timeout for large batches
    )
    session.job_id = job.id
    await db.commit()
    await db.refresh(session)

    logger.info(
        f"Created face detection session {session.id} "
        f"(training_session_id={request.training_session_id}, job_id={job.id})"
    )

    return _session_to_response(session)


@router.post("/{session_id}/pause", response_model=FaceDetectionSessionResponse)
async def pause_session(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> FaceDetectionSessionResponse:
    """Pause a running face detection session."""
    session = await db.get(FaceDetectionSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    # Can only pause if status is PROCESSING
    if session.status != DBFaceDetectionSessionStatus.PROCESSING.value:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot pause session with status '{session.status}'. Must be 'processing'.",
        )

    # Update status to PAUSED
    session.status = DBFaceDetectionSessionStatus.PAUSED.value
    await db.commit()
    await db.refresh(session)

    logger.info(f"Paused face detection session {session_id}")

    return _session_to_response(session)


@router.post("/{session_id}/resume", response_model=FaceDetectionSessionResponse)
async def resume_session(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> FaceDetectionSessionResponse:
    """Resume a paused face detection session."""
    session = await db.get(FaceDetectionSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    # Can only resume if status is PAUSED
    if session.status != DBFaceDetectionSessionStatus.PAUSED.value:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot resume session with status '{session.status}'. Must be 'paused'.",
        )

    # Update status back to PROCESSING
    session.status = DBFaceDetectionSessionStatus.PROCESSING.value
    await db.commit()
    await db.refresh(session)

    logger.info(f"Resumed face detection session {session_id}")

    return _session_to_response(session)


@router.delete("/{session_id}")
async def cancel_session(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    """Cancel a face detection session."""
    session = await db.get(FaceDetectionSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    # Update status to CANCELLED
    session.status = DBFaceDetectionSessionStatus.CANCELLED.value
    await db.commit()

    logger.info(f"Cancelled face detection session {session_id}")

    return {"status": "cancelled"}


# ============ Helper Functions ============


def _session_to_response(session: FaceDetectionSession) -> FaceDetectionSessionResponse:
    """Convert FaceDetectionSession model to response schema."""
    return FaceDetectionSessionResponse(
        id=str(session.id),
        training_session_id=session.training_session_id,
        status=session.status,
        total_images=session.total_images,
        processed_images=session.processed_images,
        failed_images=session.failed_images,
        faces_detected=session.faces_detected,
        faces_assigned=session.faces_assigned,
        min_confidence=session.min_confidence,
        min_face_size=session.min_face_size,
        batch_size=session.batch_size,
        last_error=session.last_error,
        created_at=session.created_at,
        started_at=session.started_at,
        completed_at=session.completed_at,
        job_id=session.job_id,
    )
