"""Face suggestion API routes."""

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.api.face_session_schemas import (
    AcceptSuggestionRequest,
    BulkSuggestionActionRequest,
    BulkSuggestionActionResponse,
    FaceSuggestionListResponse,
    FaceSuggestionResponse,
    RejectSuggestionRequest,
)
from image_search_service.db.models import (
    FaceInstance,
    FaceSuggestion,
    FaceSuggestionStatus,
    Person,
)
from image_search_service.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/faces/suggestions", tags=["face-suggestions"])


@router.get("", response_model=FaceSuggestionListResponse)
async def list_suggestions(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: str | None = Query(
        None, description="Filter by status: pending, accepted, rejected"
    ),
    person_id: str | None = Query(None, description="Filter by suggested person ID"),
    db: AsyncSession = Depends(get_db),
) -> FaceSuggestionListResponse:
    """List face suggestions with pagination and filtering."""
    # Build query with eager loading of face and person
    base_query = select(FaceSuggestion).order_by(FaceSuggestion.confidence.desc())

    if status:
        base_query = base_query.where(FaceSuggestion.status == status)

    if person_id:
        base_query = base_query.where(FaceSuggestion.suggested_person_id == UUID(person_id))

    # Count total
    count_query = select(func.count()).select_from(base_query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # Get paginated results
    query = base_query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    suggestions = result.scalars().all()

    # Build response items
    items = []
    for suggestion in suggestions:
        # Get person info
        person = await db.get(Person, suggestion.suggested_person_id)

        # Get face instance for thumbnail URL
        face_instance = await db.get(FaceInstance, suggestion.face_instance_id)
        thumbnail_url = (
            f"/api/v1/images/{face_instance.asset_id}/thumbnail"
            if face_instance
            else None
        )

        items.append(
            FaceSuggestionResponse(
                id=suggestion.id,
                face_instance_id=str(suggestion.face_instance_id),
                suggested_person_id=str(suggestion.suggested_person_id),
                confidence=suggestion.confidence,
                source_face_id=str(suggestion.source_face_id),
                status=suggestion.status,
                created_at=suggestion.created_at,
                reviewed_at=suggestion.reviewed_at,
                face_thumbnail_url=thumbnail_url,
                person_name=person.name if person else None,
            )
        )

    return FaceSuggestionListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/stats")
async def get_suggestion_stats(
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Get suggestion statistics including acceptance rates."""
    # Count by status
    status_query = select(
        FaceSuggestion.status,
        func.count(FaceSuggestion.id).label("count"),
    ).group_by(FaceSuggestion.status)

    status_result = await db.execute(status_query)
    status_counts: dict[str, int] = {status: count for status, count in status_result.all()}

    # Calculate totals
    total = sum(status_counts.values())
    pending = status_counts.get(FaceSuggestionStatus.PENDING.value, 0)
    accepted = status_counts.get(FaceSuggestionStatus.ACCEPTED.value, 0)
    rejected = status_counts.get(FaceSuggestionStatus.REJECTED.value, 0)
    expired = status_counts.get(FaceSuggestionStatus.EXPIRED.value, 0)

    reviewed = accepted + rejected
    acceptance_rate = (accepted / reviewed * 100) if reviewed > 0 else 0.0

    # Get suggestions by person (top 10)
    person_query = (
        select(
            FaceSuggestion.suggested_person_id,
            Person.name,
            func.count(FaceSuggestion.id).label("count"),
        )
        .join(Person, FaceSuggestion.suggested_person_id == Person.id)
        .where(FaceSuggestion.status == FaceSuggestionStatus.PENDING.value)
        .group_by(FaceSuggestion.suggested_person_id, Person.name)
        .order_by(func.count(FaceSuggestion.id).desc())
        .limit(10)
    )

    person_result = await db.execute(person_query)
    top_persons = [
        {"person_id": str(person_id), "name": name, "pending_count": count}
        for person_id, name, count in person_result.all()
    ]

    return {
        "total": total,
        "pending": pending,
        "accepted": accepted,
        "rejected": rejected,
        "expired": expired,
        "reviewed": reviewed,
        "acceptance_rate": round(acceptance_rate, 1),
        "top_persons_with_pending": top_persons,
    }


@router.get("/{suggestion_id}", response_model=FaceSuggestionResponse)
async def get_suggestion(
    suggestion_id: int,
    db: AsyncSession = Depends(get_db),
) -> FaceSuggestionResponse:
    """Get a face suggestion by ID."""
    suggestion = await db.get(FaceSuggestion, suggestion_id)
    if not suggestion:
        raise HTTPException(
            status_code=404, detail=f"Suggestion {suggestion_id} not found"
        )

    person = await db.get(Person, suggestion.suggested_person_id)

    # Get face instance for thumbnail URL
    face_instance = await db.get(FaceInstance, suggestion.face_instance_id)
    thumbnail_url = (
        f"/api/v1/images/{face_instance.asset_id}/thumbnail"
        if face_instance
        else None
    )

    return FaceSuggestionResponse(
        id=suggestion.id,
        face_instance_id=str(suggestion.face_instance_id),
        suggested_person_id=str(suggestion.suggested_person_id),
        confidence=suggestion.confidence,
        source_face_id=str(suggestion.source_face_id),
        status=suggestion.status,
        created_at=suggestion.created_at,
        reviewed_at=suggestion.reviewed_at,
        face_thumbnail_url=thumbnail_url,
        person_name=person.name if person else None,
    )


@router.post("/{suggestion_id}/accept", response_model=FaceSuggestionResponse)
async def accept_suggestion(
    suggestion_id: int,
    request: AcceptSuggestionRequest,
    db: AsyncSession = Depends(get_db),
) -> FaceSuggestionResponse:
    """Accept a face suggestion and assign the face to the person."""
    suggestion = await db.get(FaceSuggestion, suggestion_id)
    if not suggestion:
        raise HTTPException(
            status_code=404, detail=f"Suggestion {suggestion_id} not found"
        )

    if suggestion.status != FaceSuggestionStatus.PENDING.value:
        raise HTTPException(
            status_code=400,
            detail=f"Suggestion is already {suggestion.status}",
        )

    # Get the face instance
    face = await db.get(FaceInstance, suggestion.face_instance_id)
    if not face:
        raise HTTPException(status_code=404, detail="Face instance not found")

    # Assign face to person
    face.person_id = suggestion.suggested_person_id

    # Update suggestion status
    suggestion.status = FaceSuggestionStatus.ACCEPTED.value
    suggestion.reviewed_at = datetime.now(UTC)

    await db.commit()
    await db.refresh(suggestion)

    # Get person for response
    person = await db.get(Person, suggestion.suggested_person_id)

    # Get thumbnail URL from face instance (already loaded)
    thumbnail_url = f"/api/v1/images/{face.asset_id}/thumbnail"

    logger.info(
        f"Accepted suggestion {suggestion_id}: "
        f"face {suggestion.face_instance_id} → person {suggestion.suggested_person_id}"
    )

    return FaceSuggestionResponse(
        id=suggestion.id,
        face_instance_id=str(suggestion.face_instance_id),
        suggested_person_id=str(suggestion.suggested_person_id),
        confidence=suggestion.confidence,
        source_face_id=str(suggestion.source_face_id),
        status=suggestion.status,
        created_at=suggestion.created_at,
        reviewed_at=suggestion.reviewed_at,
        face_thumbnail_url=thumbnail_url,
        person_name=person.name if person else None,
    )


@router.post("/{suggestion_id}/reject", response_model=FaceSuggestionResponse)
async def reject_suggestion(
    suggestion_id: int,
    request: RejectSuggestionRequest,
    db: AsyncSession = Depends(get_db),
) -> FaceSuggestionResponse:
    """Reject a face suggestion."""
    suggestion = await db.get(FaceSuggestion, suggestion_id)
    if not suggestion:
        raise HTTPException(
            status_code=404, detail=f"Suggestion {suggestion_id} not found"
        )

    if suggestion.status != FaceSuggestionStatus.PENDING.value:
        raise HTTPException(
            status_code=400,
            detail=f"Suggestion is already {suggestion.status}",
        )

    # Update suggestion status
    suggestion.status = FaceSuggestionStatus.REJECTED.value
    suggestion.reviewed_at = datetime.now(UTC)

    await db.commit()
    await db.refresh(suggestion)

    person = await db.get(Person, suggestion.suggested_person_id)

    # Get face instance for thumbnail URL
    face_instance = await db.get(FaceInstance, suggestion.face_instance_id)
    thumbnail_url = (
        f"/api/v1/images/{face_instance.asset_id}/thumbnail"
        if face_instance
        else None
    )

    logger.info(f"Rejected suggestion {suggestion_id}")

    return FaceSuggestionResponse(
        id=suggestion.id,
        face_instance_id=str(suggestion.face_instance_id),
        suggested_person_id=str(suggestion.suggested_person_id),
        confidence=suggestion.confidence,
        source_face_id=str(suggestion.source_face_id),
        status=suggestion.status,
        created_at=suggestion.created_at,
        reviewed_at=suggestion.reviewed_at,
        face_thumbnail_url=thumbnail_url,
        person_name=person.name if person else None,
    )


@router.post("/bulk-action", response_model=BulkSuggestionActionResponse)
async def bulk_suggestion_action(
    request: BulkSuggestionActionRequest,
    db: AsyncSession = Depends(get_db),
) -> BulkSuggestionActionResponse:
    """Accept or reject multiple suggestions at once."""
    processed = 0
    failed = 0
    errors: list[str] = []

    for suggestion_id in request.suggestion_ids:
        suggestion = await db.get(FaceSuggestion, suggestion_id)
        if not suggestion:
            failed += 1
            errors.append(f"Suggestion {suggestion_id} not found")
            continue

        if suggestion.status != FaceSuggestionStatus.PENDING.value:
            failed += 1
            errors.append(
                f"Suggestion {suggestion_id} is already {suggestion.status}"
            )
            continue

        try:
            if request.action == "accept":
                # Get and update face
                face = await db.get(FaceInstance, suggestion.face_instance_id)
                if face:
                    face.person_id = suggestion.suggested_person_id
                suggestion.status = FaceSuggestionStatus.ACCEPTED.value
            else:  # reject
                suggestion.status = FaceSuggestionStatus.REJECTED.value

            suggestion.reviewed_at = datetime.now(UTC)
            processed += 1

        except Exception as e:
            failed += 1
            errors.append(f"Error processing suggestion {suggestion_id}: {str(e)}")

    await db.commit()

    logger.info(f"Bulk {request.action}: processed {processed}, failed {failed}")

    return BulkSuggestionActionResponse(
        processed=processed,
        failed=failed,
        errors=errors[:10],  # Limit errors to first 10
    )
