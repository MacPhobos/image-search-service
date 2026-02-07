"""Face suggestion API routes."""

import json
import logging
import uuid as uuid_lib
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from redis import Redis
from rq import Queue
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.api.face_session_schemas import (
    AcceptSuggestionRequest,
    BulkSuggestionActionRequest,
    BulkSuggestionActionResponse,
    FaceSuggestionListResponse,
    FaceSuggestionResponse,
    FaceSuggestionsGroupedResponse,
    FindMoreCentroidRequest,
    FindMoreJobInfo,
    FindMoreJobResponse,
    FindMoreSuggestionsRequest,
    RejectSuggestionRequest,
    SuggestionGroup,
)
from image_search_service.core.config import get_settings
from image_search_service.db.models import (
    FaceInstance,
    FaceSuggestion,
    FaceSuggestionStatus,
    ImageAsset,
    Person,
)
from image_search_service.db.session import get_db
from image_search_service.queue.face_jobs import find_more_suggestions_job
from image_search_service.services.config_service import ConfigService
from image_search_service.vector.face_qdrant import get_face_qdrant_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/faces/suggestions", tags=["face-suggestions"])


# Helper functions for building suggestion responses
async def _build_suggestion_response(
    suggestion: FaceSuggestion,
    db: AsyncSession,
    person: Person | None = None,
    face_instance: FaceInstance | None = None,
) -> FaceSuggestionResponse:
    """Build a FaceSuggestionResponse from database models."""
    # Lazy load if not provided
    if person is None:
        person = await db.get(Person, suggestion.suggested_person_id)
    if face_instance is None:
        face_instance = await db.get(FaceInstance, suggestion.face_instance_id)

    # Get ImageAsset for path
    image_asset = None
    if face_instance:
        image_asset = await db.get(ImageAsset, face_instance.asset_id)

    thumbnail_url = (
        f"/api/v1/images/{face_instance.asset_id}/thumbnail"
        if face_instance
        else None
    )
    full_image_url = (
        f"/api/v1/images/{face_instance.asset_id}/full"
        if face_instance
        else None
    )
    path = image_asset.path if image_asset else ""

    return FaceSuggestionResponse(
        id=suggestion.id,
        face_instance_id=str(suggestion.face_instance_id),
        suggested_person_id=str(suggestion.suggested_person_id),
        confidence=suggestion.confidence,
        source_face_id=str(suggestion.source_face_id) if suggestion.source_face_id else None,
        status=suggestion.status,
        created_at=suggestion.created_at,
        reviewed_at=suggestion.reviewed_at,
        face_thumbnail_url=thumbnail_url,
        person_name=person.name if person else None,
        full_image_url=full_image_url,
        path=path,
        bbox_x=face_instance.bbox_x if face_instance else None,
        bbox_y=face_instance.bbox_y if face_instance else None,
        bbox_w=face_instance.bbox_w if face_instance else None,
        bbox_h=face_instance.bbox_h if face_instance else None,
        detection_confidence=face_instance.detection_confidence if face_instance else None,
        quality_score=face_instance.quality_score if face_instance else None,
        # Multi-prototype scoring fields
        matching_prototype_ids=suggestion.matching_prototype_ids,
        prototype_scores=suggestion.prototype_scores,
        aggregate_confidence=suggestion.aggregate_confidence,
        prototype_match_count=suggestion.prototype_match_count,
    )


async def _list_suggestions_flat(
    db: AsyncSession,
    page: int,
    page_size: int,
    status: str | None,
    person_id: str | None,
) -> FaceSuggestionListResponse:
    """Legacy flat pagination mode."""
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
        items.append(await _build_suggestion_response(suggestion, db))

    return FaceSuggestionListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
    )


async def _list_suggestions_grouped(
    db: AsyncSession,
    page: int,
    groups_per_page: int,
    suggestions_per_group: int,
    status: str | None,
    person_id: str | None,
) -> FaceSuggestionsGroupedResponse:
    """Group-based pagination mode by person."""
    # Build base WHERE clause
    where_clauses = []
    if status:
        where_clauses.append(FaceSuggestion.status == status)
    if person_id:
        where_clauses.append(FaceSuggestion.suggested_person_id == UUID(person_id))

    # Query 1: Get person groups with aggregates, ordered by max confidence
    group_query = (
        select(
            FaceSuggestion.suggested_person_id,
            func.count(FaceSuggestion.id).label("count"),
            func.max(FaceSuggestion.confidence).label("max_conf"),
        )
        .group_by(FaceSuggestion.suggested_person_id)
        .order_by(func.max(FaceSuggestion.confidence).desc())
    )

    if where_clauses:
        for clause in where_clauses:
            group_query = group_query.where(clause)

    # Count total groups
    count_query = select(func.count()).select_from(group_query.subquery())
    total_groups_result = await db.execute(count_query)
    total_groups = total_groups_result.scalar() or 0

    # Get paginated person groups
    paginated_groups_query = group_query.offset((page - 1) * groups_per_page).limit(groups_per_page)
    groups_result = await db.execute(paginated_groups_query)
    person_groups = groups_result.all()

    if not person_groups:
        return FaceSuggestionsGroupedResponse(
            groups=[],
            total_groups=total_groups,
            total_suggestions=0,
            page=page,
            groups_per_page=groups_per_page,
            suggestions_per_group=suggestions_per_group,
        )

    # Extract person IDs
    person_ids = [row.suggested_person_id for row in person_groups]

    # Query 2: Get suggestions for these persons with row number limiting
    # Use SQLAlchemy ORM with window function for cross-database compatibility
    from sqlalchemy import func as sql_func

    # Create subquery with row numbers
    row_num = (
        sql_func.row_number()
        .over(
            partition_by=FaceSuggestion.suggested_person_id,
            order_by=FaceSuggestion.confidence.desc(),
        )
        .label("rn")
    )

    # Build query for suggestions with filters
    suggestions_query = select(FaceSuggestion, row_num).where(
        FaceSuggestion.suggested_person_id.in_(person_ids)
    )

    # Apply the same WHERE clauses used for grouping (includes status filter)
    if where_clauses:
        for clause in where_clauses:
            suggestions_query = suggestions_query.where(clause)

    ranked_subquery = suggestions_query.subquery()

    # Select from subquery where row number <= suggestions_per_group
    suggestions_query = (
        select(ranked_subquery)
        .where(ranked_subquery.c.rn <= suggestions_per_group)
        .order_by(
            ranked_subquery.c.suggested_person_id,
            ranked_subquery.c.confidence.desc(),
        )
    )

    suggestions_result = await db.execute(suggestions_query)
    rows = suggestions_result.fetchall()

    # Reconstruct FaceSuggestion objects from rows
    # Rows contain columns from FaceSuggestion plus the 'rn' column
    suggestions_by_person: dict[UUID, list[FaceSuggestion]] = {}
    for row in rows:
        # Access columns by their table column names
        suggestion = FaceSuggestion(
            id=row.id,
            face_instance_id=row.face_instance_id,
            suggested_person_id=row.suggested_person_id,
            confidence=row.confidence,
            source_face_id=row.source_face_id,
            status=row.status,
            created_at=row.created_at,
            reviewed_at=row.reviewed_at,
        )
        if suggestion.suggested_person_id not in suggestions_by_person:
            suggestions_by_person[suggestion.suggested_person_id] = []
        suggestions_by_person[suggestion.suggested_person_id].append(suggestion)

    # Load all persons in one query
    persons_query = select(Person).where(Person.id.in_(person_ids))
    persons_result = await db.execute(persons_query)
    persons_map = {p.id: p for p in persons_result.scalars().all()}

    # Load all face instances in one query
    all_face_instance_ids = [
        s.face_instance_id
        for suggestions in suggestions_by_person.values()
        for s in suggestions
    ]
    face_instances_query = select(FaceInstance).where(
        FaceInstance.id.in_(all_face_instance_ids)
    )
    face_instances_result = await db.execute(face_instances_query)
    face_instances_map = {f.id: f for f in face_instances_result.scalars().all()}

    # Build response groups
    groups = []
    total_suggestions = 0
    for row in person_groups:
        # Extract row attributes using column indexing to avoid type confusion
        suggested_person_id: UUID = row[0]
        row_count: int = row[1]
        row_max_conf: float = row[2]

        person = persons_map.get(suggested_person_id)
        group_suggestions = suggestions_by_person.get(suggested_person_id, [])

        # Build suggestion responses
        suggestion_responses = []
        for suggestion in group_suggestions:
            face_instance = face_instances_map.get(suggestion.face_instance_id)
            suggestion_responses.append(
                await _build_suggestion_response(suggestion, db, person, face_instance)
            )

        groups.append(
            SuggestionGroup(
                person_id=str(suggested_person_id),
                person_name=person.name if person else None,
                suggestion_count=row_count,
                max_confidence=row_max_conf,
                suggestions=suggestion_responses,
            )
        )
        total_suggestions += row_count

    return FaceSuggestionsGroupedResponse(
        groups=groups,
        total_groups=total_groups,
        total_suggestions=total_suggestions,
        page=page,
        groups_per_page=groups_per_page,
        suggestions_per_group=suggestions_per_group,
    )


@router.get("", response_model=FaceSuggestionListResponse | FaceSuggestionsGroupedResponse)
async def list_suggestions(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100, alias="pageSize"),
    status: str | None = Query(
        None, description="Filter by status: pending, accepted, rejected"
    ),
    person_id: str | None = Query(
        None, description="Filter by suggested person ID", alias="personId"
    ),
    grouped: bool = Query(
        True, description="Use group-based pagination (default: true)"
    ),
    groups_per_page: int | None = Query(
        None,
        description="Groups per page (uses config if not provided)",
        alias="groupsPerPage",
    ),
    suggestions_per_group: int | None = Query(
        None,
        description="Suggestions per group (uses config if not provided)",
        alias="suggestionsPerGroup",
    ),
    db: AsyncSession = Depends(get_db),
) -> FaceSuggestionListResponse | FaceSuggestionsGroupedResponse:
    """List face suggestions with pagination and filtering.

    Supports two pagination modes:
    - grouped=true (default): Group-based pagination by person
    - grouped=false: Legacy flat pagination
    """
    # Get config defaults if not provided
    config_service = ConfigService(db)
    if groups_per_page is None:
        groups_per_page = await config_service.get_int("face_suggestion_groups_per_page")
    if suggestions_per_group is None:
        suggestions_per_group = await config_service.get_int("face_suggestion_items_per_group")

    # Validate after loading config defaults
    if groups_per_page < 1 or groups_per_page > 50:
        raise HTTPException(
            status_code=422,
            detail="groupsPerPage must be between 1 and 50"
        )
    if suggestions_per_group < 1 or suggestions_per_group > 50:
        raise HTTPException(
            status_code=422,
            detail="suggestionsPerGroup must be between 1 and 50"
        )

    if grouped:
        # Group-based pagination
        return await _list_suggestions_grouped(
            db=db,
            page=page,
            groups_per_page=groups_per_page,
            suggestions_per_group=suggestions_per_group,
            status=status,
            person_id=person_id,
        )
    else:
        # Legacy flat pagination
        return await _list_suggestions_flat(
            db=db,
            page=page,
            page_size=page_size,
            status=status,
            person_id=person_id,
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

    # Get face instance for thumbnail URL and bounding box data
    face_instance = await db.get(FaceInstance, suggestion.face_instance_id)

    # Get ImageAsset for path
    image_asset = None
    if face_instance:
        image_asset = await db.get(ImageAsset, face_instance.asset_id)

    thumbnail_url = (
        f"/api/v1/images/{face_instance.asset_id}/thumbnail"
        if face_instance
        else None
    )
    full_image_url = (
        f"/api/v1/images/{face_instance.asset_id}/full"
        if face_instance
        else None
    )
    path = image_asset.path if image_asset else ""

    return FaceSuggestionResponse(
        id=suggestion.id,
        face_instance_id=str(suggestion.face_instance_id),
        suggested_person_id=str(suggestion.suggested_person_id),
        confidence=suggestion.confidence,
        source_face_id=str(suggestion.source_face_id) if suggestion.source_face_id else None,
        status=suggestion.status,
        created_at=suggestion.created_at,
        reviewed_at=suggestion.reviewed_at,
        face_thumbnail_url=thumbnail_url,
        person_name=person.name if person else None,
        full_image_url=full_image_url,
        path=path,
        bbox_x=face_instance.bbox_x if face_instance else None,
        bbox_y=face_instance.bbox_y if face_instance else None,
        bbox_w=face_instance.bbox_w if face_instance else None,
        bbox_h=face_instance.bbox_h if face_instance else None,
        detection_confidence=face_instance.detection_confidence if face_instance else None,
        quality_score=face_instance.quality_score if face_instance else None,
        # Multi-prototype scoring fields
        matching_prototype_ids=suggestion.matching_prototype_ids,
        prototype_scores=suggestion.prototype_scores,
        aggregate_confidence=suggestion.aggregate_confidence,
        prototype_match_count=suggestion.prototype_match_count,
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

    # Sync person_id to Qdrant (must happen after DB commit succeeds)
    try:
        qdrant = get_face_qdrant_client()
        qdrant.update_person_ids(
            [face.qdrant_point_id],
            suggestion.suggested_person_id,
        )
    except Exception as e:
        logger.error(
            f"Failed to sync person_id to Qdrant for face {face.id}: {e}. "
            f"DB is updated but Qdrant is out of sync."
        )
        # Do not re-raise -- DB commit succeeded, Qdrant can be repaired later

    # Get person for response
    person = await db.get(Person, suggestion.suggested_person_id)

    # Get ImageAsset for path
    image_asset = await db.get(ImageAsset, face.asset_id)

    # Get thumbnail URL and full image URL from face instance (already loaded)
    thumbnail_url = f"/api/v1/images/{face.asset_id}/thumbnail"
    full_image_url = f"/api/v1/images/{face.asset_id}/full"
    path = image_asset.path if image_asset else ""

    logger.info(
        f"Accepted suggestion {suggestion_id}: "
        f"face {suggestion.face_instance_id} → person {suggestion.suggested_person_id}"
    )

    return FaceSuggestionResponse(
        id=suggestion.id,
        face_instance_id=str(suggestion.face_instance_id),
        suggested_person_id=str(suggestion.suggested_person_id),
        confidence=suggestion.confidence,
        source_face_id=str(suggestion.source_face_id) if suggestion.source_face_id else None,
        status=suggestion.status,
        created_at=suggestion.created_at,
        reviewed_at=suggestion.reviewed_at,
        face_thumbnail_url=thumbnail_url,
        person_name=person.name if person else None,
        full_image_url=full_image_url,
        path=path,
        bbox_x=face.bbox_x,
        bbox_y=face.bbox_y,
        bbox_w=face.bbox_w,
        bbox_h=face.bbox_h,
        detection_confidence=face.detection_confidence,
        quality_score=face.quality_score,
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

    # Get face instance for thumbnail URL and bounding box data
    face_instance = await db.get(FaceInstance, suggestion.face_instance_id)

    # Get ImageAsset for path
    image_asset = None
    if face_instance:
        image_asset = await db.get(ImageAsset, face_instance.asset_id)

    thumbnail_url = (
        f"/api/v1/images/{face_instance.asset_id}/thumbnail"
        if face_instance
        else None
    )
    full_image_url = (
        f"/api/v1/images/{face_instance.asset_id}/full"
        if face_instance
        else None
    )
    path = image_asset.path if image_asset else ""

    logger.info(f"Rejected suggestion {suggestion_id}")

    return FaceSuggestionResponse(
        id=suggestion.id,
        face_instance_id=str(suggestion.face_instance_id),
        suggested_person_id=str(suggestion.suggested_person_id),
        confidence=suggestion.confidence,
        source_face_id=str(suggestion.source_face_id) if suggestion.source_face_id else None,
        status=suggestion.status,
        created_at=suggestion.created_at,
        reviewed_at=suggestion.reviewed_at,
        face_thumbnail_url=thumbnail_url,
        person_name=person.name if person else None,
        full_image_url=full_image_url,
        path=path,
        bbox_x=face_instance.bbox_x if face_instance else None,
        bbox_y=face_instance.bbox_y if face_instance else None,
        bbox_w=face_instance.bbox_w if face_instance else None,
        bbox_h=face_instance.bbox_h if face_instance else None,
        detection_confidence=face_instance.detection_confidence if face_instance else None,
        quality_score=face_instance.quality_score if face_instance else None,
        # Multi-prototype scoring fields
        matching_prototype_ids=suggestion.matching_prototype_ids,
        prototype_scores=suggestion.prototype_scores,
        aggregate_confidence=suggestion.aggregate_confidence,
        prototype_match_count=suggestion.prototype_match_count,
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
    affected_person_ids: set[UUID] = set()
    qdrant_sync_batch: dict[UUID, list[UUID]] = {}

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
                    # Collect for Qdrant batch sync
                    if suggestion.suggested_person_id not in qdrant_sync_batch:
                        qdrant_sync_batch[suggestion.suggested_person_id] = []
                    qdrant_sync_batch[suggestion.suggested_person_id].append(
                        face.qdrant_point_id
                    )
                suggestion.status = FaceSuggestionStatus.ACCEPTED.value
                # Track affected person IDs for auto-find-more
                affected_person_ids.add(suggestion.suggested_person_id)
            else:  # reject
                suggestion.status = FaceSuggestionStatus.REJECTED.value

            suggestion.reviewed_at = datetime.now(UTC)
            processed += 1

        except Exception as e:
            failed += 1
            errors.append(f"Error processing suggestion {suggestion_id}: {str(e)}")

    await db.commit()

    # Batch sync person_ids to Qdrant (after DB commit succeeds)
    if qdrant_sync_batch:
        try:
            qdrant = get_face_qdrant_client()
            for person_id, point_ids in qdrant_sync_batch.items():
                qdrant.update_person_ids(point_ids, person_id)
            logger.info(
                f"Synced {sum(len(v) for v in qdrant_sync_batch.values())} "
                f"face person_ids to Qdrant for {len(qdrant_sync_batch)} persons"
            )
        except Exception as e:
            logger.error(
                f"Failed to sync person_ids to Qdrant during bulk action: {e}. "
                f"DB is updated but Qdrant is out of sync."
            )

    logger.info(f"Bulk {request.action}: processed {processed}, failed {failed}")

    # Auto-trigger find-more jobs if requested
    find_more_jobs: list[FindMoreJobInfo] | None = None

    if request.auto_find_more and request.action == "accept" and affected_person_ids:
        find_more_jobs = []
        settings = get_settings()
        redis_conn = Redis.from_url(settings.redis_url)
        queue = Queue(connection=redis_conn)

        for person_id in affected_person_ids:
            job_uuid = str(uuid_lib.uuid4())
            progress_key = f"find_more:progress:{person_id}:{job_uuid}"

            # Initialize progress key so SSE subscription doesn't 404
            initial_progress = {
                "phase": "queued",
                "current": 0,
                "total": 100,
                "message": "Job queued, waiting for worker...",
                "person_id": str(person_id),
                "timestamp": datetime.now(UTC).isoformat(),
            }
            redis_conn.set(progress_key, json.dumps(initial_progress), ex=3600)  # 1 hour TTL

            job = queue.enqueue(
                find_more_suggestions_job,
                str(person_id),
                request.find_more_prototype_count,
                None,  # min_confidence
                100,  # max_suggestions
                progress_key,
                job_id=job_uuid,
            )

            find_more_jobs.append(
                FindMoreJobInfo(
                    person_id=str(person_id),
                    job_id=job.id,
                    progress_key=progress_key,
                )
            )

        logger.info(
            f"Auto-triggered {len(find_more_jobs)} find-more jobs for "
            f"{len(affected_person_ids)} persons"
        )

    return BulkSuggestionActionResponse(
        processed=processed,
        failed=failed,
        errors=errors[:10],  # Limit errors to first 10
        find_more_jobs=find_more_jobs,
    )


@router.post(
    "/persons/{person_id}/find-more",
    response_model=FindMoreJobResponse,
    status_code=201,
    summary="Start job to find more suggestions using dynamic prototypes",
    description="""
    Samples random labeled faces (weighted by quality and diversity)
    as temporary prototypes to search for additional similar faces.

    Does NOT modify the person's configured prototypes.
    Uses the same similarity threshold as normal suggestion generation.
    """,
)
async def start_find_more_suggestions(
    person_id: UUID,
    request: FindMoreSuggestionsRequest,
    db: AsyncSession = Depends(get_db),
) -> FindMoreJobResponse:
    """Start a background job to find more suggestions for a person."""

    # Validate person exists
    person = await db.get(Person, person_id)
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")

    # Count available labeled faces
    labeled_count_result = await db.execute(
        select(func.count()).where(FaceInstance.person_id == person_id)
    )
    labeled_count = labeled_count_result.scalar() or 0

    if labeled_count < 10:
        raise HTTPException(
            status_code=400,
            detail=f"Person has only {labeled_count} labeled faces. Minimum 10 required.",
        )

    # Adjust prototype count if exceeds available
    actual_count = min(request.prototype_count, labeled_count)

    # Generate full UUID for job_id
    job_uuid = str(uuid_lib.uuid4())
    progress_key = f"find_more:progress:{person_id}:{job_uuid}"

    # Get Redis connection
    settings = get_settings()
    redis_conn = Redis.from_url(settings.redis_url)

    # Initialize progress key so SSE subscription doesn't 404
    initial_progress = {
        "phase": "queued",
        "current": 0,
        "total": 100,
        "message": "Job queued, waiting for worker...",
        "person_id": str(person_id),
        "timestamp": datetime.now(UTC).isoformat(),
    }
    redis_conn.set(progress_key, json.dumps(initial_progress), ex=3600)  # 1 hour TTL

    # Enqueue job
    queue = Queue(connection=redis_conn)
    job = queue.enqueue(
        find_more_suggestions_job,
        str(person_id),
        actual_count,
        request.min_confidence,  # Pass from request
        request.max_suggestions,
        progress_key,
        job_id=job_uuid,
    )

    logger.info(
        f"Started find-more job {job.id} for person {person.name} "
        f"with {actual_count} prototypes"
    )

    return FindMoreJobResponse(
        job_id=job.id,
        person_id=str(person_id),
        person_name=person.name,
        prototype_count=actual_count,
        labeled_face_count=labeled_count,
        status="queued",
        progress_key=progress_key,
    )


@router.post(
    "/persons/{person_id}/find-more-centroid",
    response_model=FindMoreJobResponse,
    status_code=201,
    summary="Start job to find more suggestions using person centroid",
    description="""
    Uses the person's centroid embedding for faster, more consistent matching.
    Centroids are computed from all labeled faces for the person and provide
    a robust average representation.

    If no centroid exists, one will be computed on-demand.
    Requires at least 5 labeled faces for centroid computation.
    """,
)
async def start_find_more_centroid_suggestions(
    person_id: UUID,
    request: FindMoreCentroidRequest,
    db: AsyncSession = Depends(get_db),
) -> FindMoreJobResponse:
    """Start a background job to find more suggestions using centroid matching."""
    from image_search_service.queue.face_jobs import find_more_centroid_suggestions_job

    # Validate person exists
    person = await db.get(Person, person_id)
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")

    # Count available labeled faces
    labeled_count_result = await db.execute(
        select(func.count()).where(FaceInstance.person_id == person_id)
    )
    labeled_count = labeled_count_result.scalar() or 0

    if labeled_count < 5:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Person has only {labeled_count} labeled faces. "
                "Minimum 5 required for centroid."
            ),
        )

    # Generate job UUID
    job_uuid = str(uuid_lib.uuid4())
    progress_key = f"find_more_centroid:progress:{person_id}:{job_uuid}"

    # Get Redis connection
    settings = get_settings()
    redis_conn = Redis.from_url(settings.redis_url)

    # Initialize progress key so SSE subscription doesn't 404
    initial_progress = {
        "phase": "queued",
        "current": 0,
        "total": 100,
        "message": "Job queued, waiting for worker...",
        "person_id": str(person_id),
        "timestamp": datetime.now(UTC).isoformat(),
    }
    redis_conn.set(progress_key, json.dumps(initial_progress), ex=3600)  # 1 hour TTL

    # Enqueue job
    queue = Queue(connection=redis_conn)
    job = queue.enqueue(
        find_more_centroid_suggestions_job,
        str(person_id),
        request.min_similarity,
        request.max_results,
        request.unassigned_only,
        progress_key,
        job_id=job_uuid,
    )

    logger.info(
        f"Started centroid find-more job {job.id} for person {person.name} "
        f"(similarity={request.min_similarity}, max={request.max_results})"
    )

    return FindMoreJobResponse(
        job_id=job.id,
        person_id=str(person_id),
        person_name=person.name,
        prototype_count=1,  # Using single centroid
        labeled_face_count=labeled_count,
        status="queued",
        progress_key=progress_key,
    )
