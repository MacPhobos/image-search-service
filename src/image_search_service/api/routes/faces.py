"""Face detection and recognition API routes."""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import Integer, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.api.face_schemas import (
    AssignFaceRequest,
    AssignFaceResponse,
    BoundingBox,
    BulkMoveRequest,
    BulkMoveResponse,
    BulkRemoveRequest,
    BulkRemoveResponse,
    ClusterDetailResponse,
    ClusterDualRequest,
    ClusterDualResponse,
    ClusteringResultResponse,
    ClusterListResponse,
    ClusterSummary,
    CreatePersonRequest,
    CreatePersonResponse,
    DetectFacesRequest,
    DetectFacesResponse,
    FaceInPhoto,
    FaceInstanceListResponse,
    FaceInstanceResponse,
    FaceSuggestionItem,
    FaceSuggestionsResponse,
    LabelClusterRequest,
    LabelClusterResponse,
    MergePersonsRequest,
    MergePersonsResponse,
    PersonListResponse,
    PersonPhotoGroup,
    PersonPhotosResponse,
    PersonResponse,
    PinPrototypeRequest,
    PinPrototypeResponse,
    PrototypeListItem,
    PrototypeListResponse,
    RecomputePrototypesRequest,
    RecomputePrototypesResponse,
    SplitClusterRequest,
    SplitClusterResponse,
    TemporalCoverage,
    TrainMatchingRequest,
    TrainMatchingResponse,
    TriggerClusteringRequest,
    UnassignFaceResponse,
)
from image_search_service.db.models import (
    FaceAssignmentEvent,
    FaceInstance,
    ImageAsset,
    Person,
    PersonPrototype,
    PersonStatus,
    PrototypeRole,
)
from image_search_service.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/faces", tags=["Faces"])


# ============ Cluster Endpoints ============


@router.get("/clusters", response_model=ClusterListResponse)
async def list_clusters(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    include_labeled: bool = Query(
        False, description="Include clusters already assigned to persons"
    ),
    db: AsyncSession = Depends(get_db),
) -> ClusterListResponse:
    """List face clusters with pagination."""
    # Build subquery for cluster aggregation
    # Note: array_agg is PostgreSQL-specific, group_concat for SQLite
    # In production we use Postgres; for tests we detect SQLite
    is_sqlite = db.bind and "sqlite" in str(db.bind.dialect.name)

    if is_sqlite:
        # SQLite: use group_concat (returns string, we'll split it)
        face_ids_expr = func.group_concat(FaceInstance.id).label("face_ids")
    else:
        # PostgreSQL: use array_agg (returns array)
        face_ids_expr = func.array_agg(FaceInstance.id).label("face_ids")

    # Use bool_or to check if any face in cluster has person_id
    # This properly handles NULL checking without UUID comparison issues
    if is_sqlite:
        # SQLite doesn't have bool_or, use MAX with CAST to integer
        # CAST(person_id IS NOT NULL AS INTEGER) gives 1 or 0
        has_person_expr = func.max(
            func.cast(FaceInstance.person_id.isnot(None), Integer)
        ).label("has_person")
        # SQLite: get any person_id using array aggregation
        # group_concat returns string, we'll handle conversion later
        person_id_expr = func.group_concat(FaceInstance.person_id).label("person_id_concat")
    else:
        # PostgreSQL: use bool_or to check if any face is labeled
        has_person_expr = func.bool_or(FaceInstance.person_id.isnot(None)).label("has_person")
        # Get any person_id from the cluster (for display)
        # array_agg returns array, we'll extract first element later
        # Filter out NULLs to only get actual person_ids
        person_id_expr = func.array_agg(
            FaceInstance.person_id
        ).label("person_ids_array")

    cluster_query = (
        select(
            FaceInstance.cluster_id,
            person_id_expr,
            has_person_expr,
            func.count(FaceInstance.id).label("face_count"),
            func.avg(FaceInstance.quality_score).label("avg_quality"),
            face_ids_expr,
        )
        .where(FaceInstance.cluster_id.isnot(None))
        .group_by(FaceInstance.cluster_id)
    )

    if not include_labeled:
        # Filter for clusters where NO face has person_id (unlabeled clusters)
        # has_person will be FALSE or NULL for fully unlabeled clusters
        if is_sqlite:
            # SQLite: has_person is 0 or NULL for unlabeled
            cluster_query = cluster_query.having(
                func.coalesce(has_person_expr, False).is_(False)
            )
        else:
            # PostgreSQL: bool_or returns NULL if all inputs are NULL
            # We want clusters where has_person IS NOT TRUE (NULL or FALSE)
            cluster_query = cluster_query.having(has_person_expr.isnot(True))

    # Count total clusters
    count_subquery = cluster_query.subquery()
    count_query = select(func.count()).select_from(count_subquery)
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # Get paginated results
    paginated_query = cluster_query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(paginated_query)
    rows = result.all()

    items = []
    for row in rows:
        # Extract person_id from aggregated result
        person_id = None
        if is_sqlite:
            # SQLite: person_id_concat is comma-separated string or None
            if hasattr(row, 'person_id_concat') and row.person_id_concat:
                # Get first person_id from concatenated string
                first_id_str = row.person_id_concat.split(',')[0].strip()
                if first_id_str:
                    try:
                        person_id = UUID(first_id_str)
                    except ValueError:
                        pass
        else:
            # PostgreSQL: person_ids_array is array of UUIDs
            if hasattr(row, 'person_ids_array') and row.person_ids_array:
                # Filter out None values and get first
                non_null_ids = [pid for pid in row.person_ids_array if pid is not None]
                if non_null_ids:
                    person_id = non_null_ids[0]

        # Get person name if assigned
        person_name = None
        if person_id:
            person = await db.get(Person, person_id)
            person_name = person.name if person else None

        # Handle face_ids based on database type
        if is_sqlite and isinstance(row.face_ids, str):
            # SQLite returns comma-separated string of UUIDs
            face_ids_list = (
                [UUID(x.strip()) for x in row.face_ids.split(",")] if row.face_ids else []
            )
        else:
            # PostgreSQL returns array of UUIDs
            face_ids_list = list(row.face_ids) if row.face_ids else []

        items.append(
            ClusterSummary(
                cluster_id=row.cluster_id,
                face_count=row.face_count,
                sample_face_ids=face_ids_list[:5],
                avg_quality=row.avg_quality,
                person_id=person_id,
                person_name=person_name,
            )
        )

    return ClusterListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/clusters/{cluster_id}", response_model=ClusterDetailResponse)
async def get_cluster(
    cluster_id: str,
    db: AsyncSession = Depends(get_db),
) -> ClusterDetailResponse:
    """Get detailed info for a specific cluster."""
    # Get all faces in cluster with person names
    query = (
        select(FaceInstance, Person.name)
        .outerjoin(Person, FaceInstance.person_id == Person.id)
        .where(FaceInstance.cluster_id == cluster_id)
    )
    result = await db.execute(query)
    faces_data = result.all()

    if not faces_data:
        raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")

    # Get person info from first face (clusters typically have one person)
    first_face, first_person_name = faces_data[0]
    person_id = first_face.person_id
    person_name = first_person_name

    return ClusterDetailResponse(
        cluster_id=cluster_id,
        faces=[_face_to_response(face, name) for face, name in faces_data],
        person_id=person_id,
        person_name=person_name,
    )


@router.post("/clusters/{cluster_id}/label", response_model=LabelClusterResponse)
async def label_cluster(
    cluster_id: str,
    request: LabelClusterRequest,
    db: AsyncSession = Depends(get_db),
) -> LabelClusterResponse:
    """Label a cluster with a person name, creating the person if needed."""
    from redis import Redis
    from rq import Queue

    from image_search_service.core.config import get_settings
    from image_search_service.queue.face_jobs import propagate_person_label_job
    from image_search_service.vector.face_qdrant import get_face_qdrant_client

    # Get faces in cluster
    query = select(FaceInstance).where(FaceInstance.cluster_id == cluster_id)
    result = await db.execute(query)
    faces = result.scalars().all()

    if not faces:
        raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")

    # Find or create person by name (case-insensitive)
    person_query = select(Person).where(func.lower(Person.name) == request.name.lower())
    person_result = await db.execute(person_query)
    person = person_result.scalar_one_or_none()

    if not person:
        person = Person(name=request.name)
        db.add(person)
        await db.flush()
        logger.info(f"Created new person: {person.name} ({person.id})")

    # Assign person to all faces in cluster
    face_ids = []
    qdrant_point_ids = []

    for face in faces:
        face.person_id = person.id
        face_ids.append(face.id)
        qdrant_point_ids.append(face.qdrant_point_id)

    # Update Qdrant payloads
    qdrant = get_face_qdrant_client()
    qdrant.update_person_ids(qdrant_point_ids, person.id)

    # Create prototypes (top 3 quality faces as exemplars)
    sorted_faces = sorted(faces, key=lambda f: f.quality_score or 0, reverse=True)
    prototypes_created = 0

    for face in sorted_faces[:3]:
        prototype = PersonPrototype(
            person_id=person.id,
            face_instance_id=face.id,
            qdrant_point_id=face.qdrant_point_id,
            role=PrototypeRole.EXEMPLAR,
        )
        db.add(prototype)
        prototypes_created += 1

        # Mark as prototype in Qdrant
        qdrant.update_payload(face.qdrant_point_id, {"is_prototype": True})

    await db.commit()

    logger.info(f"Labeled cluster {cluster_id} as person {person.name} ({len(faces)} faces)")

    # Trigger propagation job using the best quality face as source
    if sorted_faces:
        best_face = sorted_faces[0]
        try:
            settings = get_settings()
            redis_conn = Redis.from_url(settings.redis_url)
            queue = Queue("default", connection=redis_conn)

            queue.enqueue(
                propagate_person_label_job,
                source_face_id=str(best_face.id),
                person_id=str(person.id),
                min_confidence=0.7,
                max_suggestions=50,
                job_timeout="10m",
            )
            logger.info(
                f"Queued propagation job for face {best_face.id} → person {person.id} "
                f"(cluster {cluster_id})"
            )
        except Exception as e:
            logger.warning(f"Failed to enqueue propagation job: {e}")
            # Don't fail the request if job queueing fails

    return LabelClusterResponse(
        person_id=person.id,
        person_name=person.name,
        faces_labeled=len(faces),
        prototypes_created=prototypes_created,
    )


@router.post("/clusters/{cluster_id}/split", response_model=SplitClusterResponse)
async def split_cluster(
    cluster_id: str,
    request: SplitClusterRequest,
    db: AsyncSession = Depends(get_db),
) -> SplitClusterResponse:
    """Split a cluster into smaller sub-clusters using tighter HDBSCAN params."""
    from image_search_service.db.sync_operations import get_sync_session
    from image_search_service.faces.clusterer import get_face_clusterer

    # Use sync session for clusterer
    with get_sync_session() as sync_db:
        clusterer = get_face_clusterer(
            sync_db,
            min_cluster_size=request.min_cluster_size,
        )
        result = clusterer.recluster_within_cluster(cluster_id)

    if result.get("status") == "too_small":
        raise HTTPException(
            status_code=400,
            detail=f"Cluster too small to split ({result.get('count')} faces)",
        )

    return SplitClusterResponse(
        original_cluster_id=cluster_id,
        new_clusters=result.get("new_cluster_ids", []),
        status=result.get("status", "unknown"),
    )


# ============ Person Endpoints ============


@router.get("/persons", response_model=PersonListResponse)
async def list_persons(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: str | None = Query(None, description="Filter by status: active, merged, hidden"),
    db: AsyncSession = Depends(get_db),
) -> PersonListResponse:
    """List persons with pagination."""
    base_query = select(Person)

    if status:
        base_query = base_query.where(Person.status == PersonStatus(status))

    # Count total
    count_query = select(func.count()).select_from(base_query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # Get paginated results
    query = base_query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    persons = result.scalars().all()

    items = []
    for person in persons:
        # Count faces and prototypes
        face_count_query = select(func.count()).where(FaceInstance.person_id == person.id)
        face_count = (await db.execute(face_count_query)).scalar() or 0

        proto_count_query = select(func.count()).where(PersonPrototype.person_id == person.id)
        proto_count = (await db.execute(proto_count_query)).scalar() or 0

        items.append(
            PersonResponse(
                id=person.id,
                name=person.name,
                status=person.status.value,
                face_count=face_count,
                prototype_count=proto_count,
                created_at=person.created_at,
                updated_at=person.updated_at,
            )
        )

    return PersonListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post("/persons", response_model=CreatePersonResponse, status_code=201)
async def create_person(
    request: CreatePersonRequest,
    db: AsyncSession = Depends(get_db),
) -> CreatePersonResponse:
    """Create a new person entity."""
    # Check for existing person with same name (case-insensitive)
    person_query = select(Person).where(func.lower(Person.name) == request.name.lower())
    person_result = await db.execute(person_query)
    existing_person = person_result.scalar_one_or_none()

    if existing_person:
        raise HTTPException(
            status_code=409,
            detail=f"Person with name '{request.name}' already exists",
        )

    # Create new person
    person = Person(name=request.name)
    db.add(person)
    await db.commit()
    await db.refresh(person)

    logger.info(f"Created new person: {person.name} ({person.id})")

    return CreatePersonResponse(
        id=person.id,
        name=person.name,
        status=person.status.value,
        created_at=person.created_at,
    )


@router.get("/persons/{person_id}/photos", response_model=PersonPhotosResponse)
async def get_person_photos(
    person_id: UUID,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> PersonPhotosResponse:
    """
    Get photos containing faces assigned to this person, grouped by photo.

    Returns photos with all faces in each photo, including faces belonging
    to other persons or no person. This enables photo-level review workflow.
    """
    # Step 1: Verify person exists
    person = await db.get(Person, person_id)
    if not person:
        raise HTTPException(status_code=404, detail=f"Person {person_id} not found")

    # Step 2: Get distinct asset_ids for photos containing this person's faces
    asset_subquery = (
        select(FaceInstance.asset_id)
        .where(FaceInstance.person_id == person_id)
        .distinct()
        .subquery()
    )

    # Step 3: Count total photos
    count_query = select(func.count()).select_from(asset_subquery)
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # Step 4: Get paginated asset IDs ordered by asset_id DESC
    # (ideally we'd order by taken_at but that's not in ImageAsset model yet)
    paginated_assets_query = (
        select(ImageAsset.id)
        .where(ImageAsset.id.in_(select(asset_subquery.c.asset_id)))
        .order_by(ImageAsset.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    assets_result = await db.execute(paginated_assets_query)
    asset_ids = [row[0] for row in assets_result.all()]

    # Step 5: For each photo, get ALL faces (not just this person's)
    # Build a dict mapping asset_id -> list of faces
    if asset_ids:
        faces_query = (
            select(FaceInstance, Person.name)
            .outerjoin(Person, FaceInstance.person_id == Person.id)
            .where(FaceInstance.asset_id.in_(asset_ids))
            .order_by(FaceInstance.asset_id, FaceInstance.bbox_x)
        )
        faces_result = await db.execute(faces_query)
        faces_data = faces_result.all()

        # Group faces by asset_id
        faces_by_asset: dict[int, list[tuple[FaceInstance, str | None]]] = {}
        for face, person_name in faces_data:
            if face.asset_id not in faces_by_asset:
                faces_by_asset[face.asset_id] = []
            faces_by_asset[face.asset_id].append((face, person_name))
    else:
        faces_by_asset = {}

    # Step 6: Build PersonPhotoGroup for each photo
    items = []
    for asset_id in asset_ids:
        faces_in_photo = faces_by_asset.get(asset_id, [])

        # Convert to FaceInPhoto schema
        face_schemas = []
        has_non_person_faces = False

        for face, face_person_name in faces_in_photo:
            face_schemas.append(
                FaceInPhoto(
                    face_instance_id=face.id,
                    bbox_x=face.bbox_x,
                    bbox_y=face.bbox_y,
                    bbox_w=face.bbox_w,
                    bbox_h=face.bbox_h,
                    detection_confidence=face.detection_confidence,
                    quality_score=face.quality_score,
                    person_id=face.person_id,
                    person_name=face_person_name,
                    cluster_id=face.cluster_id,
                )
            )

            # Check if this face belongs to a different person or no person
            if face.person_id != person_id:
                has_non_person_faces = True

        items.append(
            PersonPhotoGroup(
                photo_id=asset_id,
                taken_at=None,  # TODO: Add EXIF taken_at to ImageAsset model
                thumbnail_url=f"/api/v1/images/{asset_id}/thumbnail",
                full_url=f"/api/v1/images/{asset_id}/full",
                faces=face_schemas,
                face_count=len(face_schemas),
                has_non_person_faces=has_non_person_faces,
            )
        )

    return PersonPhotosResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        person_id=person_id,
        person_name=person.name,
    )


@router.post("/persons/{person_id}/merge", response_model=MergePersonsResponse)
async def merge_persons(
    person_id: UUID,
    request: MergePersonsRequest,
    db: AsyncSession = Depends(get_db),
) -> MergePersonsResponse:
    """Merge one person into another."""
    from image_search_service.vector.face_qdrant import get_face_qdrant_client

    if person_id == request.into_person_id:
        raise HTTPException(status_code=400, detail="Cannot merge person into itself")

    source = await db.get(Person, person_id)
    target = await db.get(Person, request.into_person_id)

    if not source:
        raise HTTPException(status_code=404, detail=f"Source person {person_id} not found")
    if not target:
        raise HTTPException(
            status_code=404, detail=f"Target person {request.into_person_id} not found"
        )

    # Get all faces for source person
    query = select(FaceInstance).where(FaceInstance.person_id == person_id)
    result = await db.execute(query)
    faces = result.scalars().all()

    # Move faces to target person
    qdrant = get_face_qdrant_client()
    qdrant_point_ids = []

    for face in faces:
        face.person_id = target.id
        qdrant_point_ids.append(face.qdrant_point_id)

    # Update Qdrant
    if qdrant_point_ids:
        qdrant.update_person_ids(qdrant_point_ids, target.id)

    # Mark source person as merged
    source.status = PersonStatus.MERGED
    source.merged_into_id = target.id

    await db.commit()

    logger.info(f"Merged person {source.name} into {target.name} ({len(faces)} faces)")

    # Trigger background jobs to update person_ids for all affected assets
    # Get unique asset IDs from the moved faces
    affected_asset_ids = {face.asset_id for face in faces}
    if affected_asset_ids:
        try:
            from redis import Redis
            from rq import Queue

            from image_search_service.core.config import get_settings
            from image_search_service.queue.jobs import update_asset_person_ids_job

            settings = get_settings()
            redis_conn = Redis.from_url(settings.redis_url)
            queue = Queue("default", connection=redis_conn)

            for asset_id in affected_asset_ids:
                queue.enqueue(
                    update_asset_person_ids_job,
                    asset_id=asset_id,
                    job_timeout="5m",
                )

            logger.info(
                f"Queued {len(affected_asset_ids)} person_ids update jobs for merge "
                f"{source.name} → {target.name}"
            )
        except Exception as e:
            logger.warning(f"Failed to enqueue person_ids update jobs: {e}")
            # Don't fail the request if job queueing fails

    return MergePersonsResponse(
        source_person_id=person_id,
        target_person_id=target.id,
        faces_moved=len(faces),
    )


@router.post("/persons/{person_id}/photos/bulk-remove", response_model=BulkRemoveResponse)
async def bulk_remove_from_person(
    person_id: UUID,
    request: BulkRemoveRequest,
    db: AsyncSession = Depends(get_db),
) -> BulkRemoveResponse:
    """
    Remove person assignment from all faces in selected photos.

    Only affects faces currently assigned to the specified person.
    Other faces in the same photos are not modified.
    """
    from image_search_service.vector.face_qdrant import get_face_qdrant_client

    # Step 1: Verify person exists
    person = await db.get(Person, person_id)
    if not person:
        raise HTTPException(status_code=404, detail=f"Person {person_id} not found")

    # Handle empty photo_ids list
    if not request.photo_ids:
        return BulkRemoveResponse(
            updated_faces=0,
            updated_photos=0,
            skipped_faces=0,
        )

    # Step 2: Find faces in selected photos that belong to this person
    query = select(FaceInstance).where(
        FaceInstance.asset_id.in_(request.photo_ids),
        FaceInstance.person_id == person_id,
    )
    result = await db.execute(query)
    faces = result.scalars().all()

    if not faces:
        # No faces matched - all were already unassigned or belong to other persons
        return BulkRemoveResponse(
            updated_faces=0,
            updated_photos=0,
            skipped_faces=0,
        )

    # Step 3: Update database - set person_id to None
    face_ids = []
    qdrant_point_ids = []
    affected_photo_ids = set()

    for face in faces:
        face.person_id = None
        face_ids.append(face.id)
        qdrant_point_ids.append(face.qdrant_point_id)
        affected_photo_ids.add(face.asset_id)

    try:
        # Step 4: Update Qdrant payloads (remove person_id)
        qdrant = get_face_qdrant_client()
        qdrant.update_person_ids(qdrant_point_ids, None)

        # Step 5: Create audit event
        event = FaceAssignmentEvent(
            operation="REMOVE_FROM_PERSON",
            from_person_id=person_id,
            to_person_id=None,
            affected_photo_ids=list(affected_photo_ids),
            affected_face_instance_ids=[str(fid) for fid in face_ids],
            face_count=len(faces),
            photo_count=len(affected_photo_ids),
            actor=None,
            note=f"Bulk remove from {len(request.photo_ids)} selected photos",
        )
        db.add(event)

        await db.commit()

        logger.info(
            f"Bulk removed {len(faces)} faces from person {person.name} "
            f"in {len(affected_photo_ids)} photos"
        )

        # Trigger background jobs to update person_ids for all affected assets
        if affected_photo_ids:
            try:
                from redis import Redis
                from rq import Queue

                from image_search_service.core.config import get_settings
                from image_search_service.queue.jobs import update_asset_person_ids_job

                settings = get_settings()
                redis_conn = Redis.from_url(settings.redis_url)
                queue = Queue("default", connection=redis_conn)

                for asset_id in affected_photo_ids:
                    queue.enqueue(
                        update_asset_person_ids_job,
                        asset_id=asset_id,
                        job_timeout="5m",
                    )

                logger.info(
                    f"Queued {len(affected_photo_ids)} person_ids update jobs for bulk remove "
                    f"from person {person.name}"
                )
            except Exception as e:
                logger.warning(f"Failed to enqueue person_ids update jobs: {e}")
                # Don't fail the request if job queueing fails

        return BulkRemoveResponse(
            updated_faces=len(faces),
            updated_photos=len(affected_photo_ids),
            skipped_faces=0,
        )

    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to bulk remove faces from person {person_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to remove faces: {str(e)}")


@router.post("/persons/{person_id}/photos/bulk-move", response_model=BulkMoveResponse)
async def bulk_move_to_person(
    person_id: UUID,
    request: BulkMoveRequest,
    db: AsyncSession = Depends(get_db),
) -> BulkMoveResponse:
    """
    Move faces from one person to another.

    Only affects faces currently assigned to the specified person.
    Creates a new person if to_person_name is provided.
    """
    from redis import Redis
    from rq import Queue

    from image_search_service.core.config import get_settings
    from image_search_service.queue.face_jobs import propagate_person_label_job
    from image_search_service.vector.face_qdrant import get_face_qdrant_client

    # Step 1: Verify source person exists
    source_person = await db.get(Person, person_id)
    if not source_person:
        raise HTTPException(status_code=404, detail=f"Person {person_id} not found")

    # Handle empty photo_ids list
    if not request.photo_ids:
        # Still need to determine target person for response
        if request.to_person_id:
            target_person = await db.get(Person, request.to_person_id)
            if not target_person:
                raise HTTPException(
                    status_code=404, detail=f"Target person {request.to_person_id} not found"
                )
            return BulkMoveResponse(
                to_person_id=target_person.id,
                to_person_name=target_person.name,
                updated_faces=0,
                updated_photos=0,
                skipped_faces=0,
                person_created=False,
            )
        else:
            # Create new person even with no faces to move
            target_person = Person(name=request.to_person_name)
            db.add(target_person)
            await db.flush()
            return BulkMoveResponse(
                to_person_id=target_person.id,
                to_person_name=target_person.name,
                updated_faces=0,
                updated_photos=0,
                skipped_faces=0,
                person_created=True,
            )

    # Step 2: Resolve destination person
    person_created = False
    if request.to_person_id:
        # Use existing person
        target_person = await db.get(Person, request.to_person_id)
        if not target_person:
            raise HTTPException(
                status_code=404, detail=f"Target person {request.to_person_id} not found"
            )
    else:
        # Create new person (case-insensitive check for existing name)
        # Validator ensures to_person_name is not None here, but mypy needs explicit check
        if not request.to_person_name:
            raise HTTPException(
                status_code=422,
                detail="to_person_name must be provided when to_person_id is not set",
            )

        person_query = select(Person).where(
            func.lower(Person.name) == request.to_person_name.lower()
        )
        person_result = await db.execute(person_query)
        target_person = person_result.scalar_one_or_none()

        if not target_person:
            target_person = Person(name=request.to_person_name)
            db.add(target_person)
            await db.flush()
            person_created = True
            logger.info(f"Created new person: {target_person.name} ({target_person.id})")

    # Step 3: Find faces in selected photos that belong to source person
    query = select(FaceInstance).where(
        FaceInstance.asset_id.in_(request.photo_ids),
        FaceInstance.person_id == person_id,
    )
    result = await db.execute(query)
    faces = result.scalars().all()

    if not faces:
        # No faces matched
        return BulkMoveResponse(
            to_person_id=target_person.id,
            to_person_name=target_person.name,
            updated_faces=0,
            updated_photos=0,
            skipped_faces=0,
            person_created=person_created,
        )

    # Step 4: Update database - move faces to target person
    face_ids = []
    qdrant_point_ids = []
    affected_photo_ids = set()

    for face in faces:
        face.person_id = target_person.id
        face_ids.append(face.id)
        qdrant_point_ids.append(face.qdrant_point_id)
        affected_photo_ids.add(face.asset_id)

    try:
        # Step 5: Update Qdrant payloads
        qdrant = get_face_qdrant_client()
        qdrant.update_person_ids(qdrant_point_ids, target_person.id)

        # Step 6: Create audit event
        event = FaceAssignmentEvent(
            operation="MOVE_TO_PERSON",
            from_person_id=person_id,
            to_person_id=target_person.id,
            affected_photo_ids=list(affected_photo_ids),
            affected_face_instance_ids=[str(fid) for fid in face_ids],
            face_count=len(faces),
            photo_count=len(affected_photo_ids),
            actor=None,
            note=f"Bulk move from {len(request.photo_ids)} selected photos",
        )
        db.add(event)

        await db.commit()

        # Create prototypes from verified labels
        from image_search_service.services.prototype_service import create_or_update_prototypes

        settings = get_settings()
        for face in faces:
            try:
                await create_or_update_prototypes(
                    db=db,
                    qdrant=qdrant,
                    person_id=target_person.id,
                    newly_labeled_face_id=face.id,
                    max_exemplars=settings.face_prototype_max_exemplars,
                    min_quality_threshold=settings.face_prototype_min_quality,
                )
            except Exception as e:
                logger.warning(f"Failed to create prototype for face {face.id}: {e}")
                # Continue with other faces

        await db.commit()

        logger.info(
            f"Bulk moved {len(faces)} faces from {source_person.name} to {target_person.name} "
            f"in {len(affected_photo_ids)} photos"
        )

        # Trigger background jobs to update person_ids for all affected assets
        if affected_photo_ids:
            try:
                from redis import Redis
                from rq import Queue

                from image_search_service.core.config import get_settings
                from image_search_service.queue.jobs import update_asset_person_ids_job

                settings = get_settings()
                redis_conn = Redis.from_url(settings.redis_url)
                queue = Queue("default", connection=redis_conn)

                for asset_id in affected_photo_ids:
                    queue.enqueue(
                        update_asset_person_ids_job,
                        asset_id=asset_id,
                        job_timeout="5m",
                    )

                logger.info(
                    f"Queued {len(affected_photo_ids)} person_ids update jobs for bulk move "
                    f"{source_person.name} → {target_person.name}"
                )
            except Exception as e:
                logger.warning(f"Failed to enqueue person_ids update jobs: {e}")
                # Don't fail the request if job queueing fails

        # Trigger propagation job using the first face as source
        if faces:
            # Sort by quality to use best face
            sorted_faces = sorted(faces, key=lambda f: f.quality_score or 0, reverse=True)
            best_face = sorted_faces[0]
            try:
                settings = get_settings()
                redis_conn = Redis.from_url(settings.redis_url)
                queue = Queue("default", connection=redis_conn)

                queue.enqueue(
                    propagate_person_label_job,
                    source_face_id=str(best_face.id),
                    person_id=str(target_person.id),
                    min_confidence=0.7,
                    max_suggestions=50,
                    job_timeout="10m",
                )
                logger.info(
                    f"Queued propagation job for face {best_face.id} → person {target_person.id} "
                    f"(bulk move from person {person_id})"
                )
            except Exception as e:
                logger.warning(f"Failed to enqueue propagation job: {e}")
                # Don't fail the request if job queueing fails

        return BulkMoveResponse(
            to_person_id=target_person.id,
            to_person_name=target_person.name,
            updated_faces=len(faces),
            updated_photos=len(affected_photo_ids),
            skipped_faces=0,
            person_created=person_created,
        )

    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to bulk move faces from person {person_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to move faces: {str(e)}")


# ============ Face Detection Endpoints ============


@router.post("/detect/{asset_id}", response_model=DetectFacesResponse)
async def detect_faces_in_asset(
    asset_id: int,
    request: DetectFacesRequest,
    db: AsyncSession = Depends(get_db),
) -> DetectFacesResponse:
    """Detect faces in a specific asset."""
    from image_search_service.db.sync_operations import get_sync_session
    from image_search_service.faces.service import get_face_service

    # Verify asset exists
    asset = await db.get(ImageAsset, asset_id)
    if not asset:
        raise HTTPException(status_code=404, detail=f"Asset {asset_id} not found")

    # Use sync session for face detection
    with get_sync_session() as sync_db:
        service = get_face_service(sync_db)
        sync_asset = sync_db.get(ImageAsset, asset_id)
        if not sync_asset:
            raise HTTPException(status_code=404, detail=f"Asset {asset_id} not found")

        faces = service.process_asset(
            sync_asset,
            min_confidence=request.min_confidence,
            min_face_size=request.min_face_size,
        )

    return DetectFacesResponse(
        asset_id=asset_id,
        faces_detected=len(faces),
        face_ids=[f.id for f in faces],
    )


@router.post("/cluster", response_model=ClusteringResultResponse)
async def trigger_clustering(
    request: TriggerClusteringRequest,
    db: AsyncSession = Depends(get_db),
) -> ClusteringResultResponse:
    """Trigger face clustering on unlabeled faces."""
    from image_search_service.db.sync_operations import get_sync_session
    from image_search_service.faces.clusterer import get_face_clusterer

    with get_sync_session() as sync_db:
        clusterer = get_face_clusterer(
            sync_db,
            min_cluster_size=request.min_cluster_size,
        )
        result = clusterer.cluster_unlabeled_faces(
            quality_threshold=request.quality_threshold,
            max_faces=request.max_faces,
        )

    return ClusteringResultResponse(
        total_faces=result["total_faces"],
        clusters_found=result["clusters_found"],
        noise_count=result["noise_count"],
    )


@router.get("/assets/{asset_id}", response_model=FaceInstanceListResponse)
async def get_faces_for_asset(
    asset_id: int,
    db: AsyncSession = Depends(get_db),
) -> FaceInstanceListResponse:
    """Get all detected faces for a specific asset."""
    # Join with Person table to get person names
    query = (
        select(FaceInstance, Person.name)
        .outerjoin(Person, FaceInstance.person_id == Person.id)
        .where(FaceInstance.asset_id == asset_id)
    )
    result = await db.execute(query)
    faces_data = result.all()

    return FaceInstanceListResponse(
        items=[_face_to_response(face, person_name) for face, person_name in faces_data],
        total=len(faces_data),
        page=1,
        page_size=len(faces_data),
    )


@router.post("/faces/{face_id}/assign", response_model=AssignFaceResponse)
async def assign_face_to_person(
    face_id: UUID,
    request: AssignFaceRequest,
    db: AsyncSession = Depends(get_db),
) -> AssignFaceResponse:
    """Assign a single face instance to a person."""
    from redis import Redis
    from rq import Queue

    from image_search_service.core.config import get_settings
    from image_search_service.queue.face_jobs import propagate_person_label_job
    from image_search_service.vector.face_qdrant import get_face_qdrant_client

    # Get face instance from DB
    face = await db.get(FaceInstance, face_id)
    if not face:
        raise HTTPException(status_code=404, detail=f"Face {face_id} not found")

    # Get target person from DB
    person = await db.get(Person, request.person_id)
    if not person:
        raise HTTPException(status_code=404, detail=f"Person {request.person_id} not found")

    # Track previous assignment for audit log
    previous_person_id = face.person_id

    # Update face.person_id to new person
    face.person_id = person.id

    try:
        # Check if Qdrant point exists before trying to update
        qdrant = get_face_qdrant_client()
        if not qdrant.point_exists(face.qdrant_point_id):
            await db.rollback()
            raise HTTPException(
                status_code=404,
                detail=(
                    "Face embedding not found in vector database. "
                    "The face may need to be re-detected."
                ),
            )

        # Update Qdrant payload with new person_id
        qdrant.update_person_ids([face.qdrant_point_id], person.id)

        # Create audit event
        event = FaceAssignmentEvent(
            operation="ASSIGN_TO_PERSON",
            from_person_id=previous_person_id,
            to_person_id=person.id,
            affected_photo_ids=[face.asset_id],
            affected_face_instance_ids=[str(face.id)],
            face_count=1,
            photo_count=1,
            actor=None,
            note=f"Single face assignment to {person.name}",
        )
        db.add(event)

        await db.commit()

        # Create prototype from this verified label
        from image_search_service.services.prototype_service import create_or_update_prototypes

        settings = get_settings()
        try:
            await create_or_update_prototypes(
                db=db,
                qdrant=qdrant,
                person_id=person.id,
                newly_labeled_face_id=face.id,
                max_exemplars=settings.face_prototype_max_exemplars,
                min_quality_threshold=settings.face_prototype_min_quality,
            )
            await db.commit()
        except Exception as e:
            logger.warning(f"Failed to create prototype for face {face.id}: {e}")
            # Don't fail the request if prototype creation fails

        logger.info(f"Assigned face {face_id} to person {person.name} ({person.id})")

        # Trigger background job to update asset's person_ids in Qdrant
        try:
            from redis import Redis
            from rq import Queue

            from image_search_service.core.config import get_settings
            from image_search_service.queue.jobs import update_asset_person_ids_job

            settings = get_settings()
            redis_conn = Redis.from_url(settings.redis_url)
            queue = Queue("default", connection=redis_conn)

            queue.enqueue(
                update_asset_person_ids_job,
                asset_id=face.asset_id,
                job_timeout="5m",
            )
            logger.info(f"Queued person_ids update job for asset {face.asset_id}")
        except Exception as e:
            logger.warning(f"Failed to enqueue person_ids update job: {e}")
            # Don't fail the request if job queueing fails

        # Trigger propagation job using the assigned face as source
        try:
            settings = get_settings()
            redis_conn = Redis.from_url(settings.redis_url)
            queue = Queue("default", connection=redis_conn)

            queue.enqueue(
                propagate_person_label_job,
                source_face_id=str(face.id),
                person_id=str(person.id),
                min_confidence=0.7,
                max_suggestions=50,
                job_timeout="10m",
            )
            logger.info(
                f"Queued propagation job for face {face.id} → person {person.id} "
                f"(single assignment)"
            )
        except Exception as e:
            logger.warning(f"Failed to enqueue propagation job: {e}")
            # Don't fail the request if job queueing fails

        return AssignFaceResponse(
            face_id=face.id,
            person_id=person.id,
            person_name=person.name,
        )

    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to assign face {face_id} to person {request.person_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to assign face: {str(e)}")


@router.delete("/faces/{face_id}/person", response_model=UnassignFaceResponse)
async def unassign_face_from_person(
    face_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> UnassignFaceResponse:
    """Unassign a face instance from its currently assigned person."""
    from image_search_service.vector.face_qdrant import get_face_qdrant_client

    # Get face instance from DB
    face = await db.get(FaceInstance, face_id)
    if not face:
        raise HTTPException(status_code=404, detail=f"Face {face_id} not found")

    # Check if face is assigned to any person
    if face.person_id is None:
        raise HTTPException(
            status_code=400,
            detail="Face is not assigned to any person",
        )

    # Store previous person info before clearing
    previous_person_id = face.person_id
    person = await db.get(Person, previous_person_id)
    if not person:
        # Edge case: person was deleted but face still references it
        # Proceed with unassignment but log warning
        logger.warning(f"Face {face_id} references non-existent person {previous_person_id}")
        raise HTTPException(
            status_code=404,
            detail=f"Person {previous_person_id} not found",
        )

    previous_person_name = person.name

    # Clear person assignment
    face.person_id = None

    try:
        # Update Qdrant payload (remove person_id)
        qdrant = get_face_qdrant_client()
        qdrant.update_person_ids([face.qdrant_point_id], None)

        # Create audit event
        event = FaceAssignmentEvent(
            operation="UNASSIGN_FROM_PERSON",
            from_person_id=previous_person_id,
            to_person_id=None,
            affected_photo_ids=[face.asset_id],
            affected_face_instance_ids=[str(face.id)],
            face_count=1,
            photo_count=1,
            actor=None,
            note=f"Unassigned face from {previous_person_name}",
        )
        db.add(event)

        await db.commit()

        logger.info(
            f"Unassigned face {face_id} from person {previous_person_name} ({previous_person_id})"
        )

        # Trigger background job to update asset's person_ids in Qdrant
        try:
            from redis import Redis
            from rq import Queue

            from image_search_service.core.config import get_settings
            from image_search_service.queue.jobs import update_asset_person_ids_job

            settings = get_settings()
            redis_conn = Redis.from_url(settings.redis_url)
            queue = Queue("default", connection=redis_conn)

            queue.enqueue(
                update_asset_person_ids_job,
                asset_id=face.asset_id,
                job_timeout="5m",
            )
            logger.info(f"Queued person_ids update job for asset {face.asset_id}")
        except Exception as e:
            logger.warning(f"Failed to enqueue person_ids update job: {e}")
            # Don't fail the request if job queueing fails

        return UnassignFaceResponse(
            face_id=face.id,
            previous_person_id=previous_person_id,
            previous_person_name=previous_person_name,
        )

    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to unassign face {face_id} from person {previous_person_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to unassign face: {str(e)}")


@router.get("/faces/{face_id}/suggestions", response_model=FaceSuggestionsResponse)
async def get_face_suggestions(
    face_id: UUID,
    min_confidence: float = Query(0.7, ge=0.0, le=1.0, description="Minimum confidence threshold"),
    limit: int = Query(5, ge=1, le=10, description="Maximum number of suggestions"),
    db: AsyncSession = Depends(get_db),
) -> FaceSuggestionsResponse:
    """Get person suggestions for a face based on similarity to person prototypes.

    This endpoint compares the face's embedding against all person prototype embeddings
    in Qdrant and returns the most similar persons above the confidence threshold.

    Args:
        face_id: UUID of the face instance
        min_confidence: Minimum cosine similarity score (0.0-1.0, default 0.7)
        limit: Maximum number of suggestions to return (1-10, default 5)
        db: Database session

    Returns:
        List of person suggestions with confidence scores
    """
    from image_search_service.vector.face_qdrant import get_face_qdrant_client

    # Step 1: Get face instance from database
    face = await db.get(FaceInstance, face_id)
    if not face:
        raise HTTPException(status_code=404, detail=f"Face {face_id} not found")

    # Step 2: Get face embedding from Qdrant
    qdrant = get_face_qdrant_client()
    embedding = qdrant.get_embedding_by_point_id(face.qdrant_point_id)

    if embedding is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Face embedding not found in vector database. "
                "The face may need to be re-detected."
            ),
        )

    # Step 3: Search for similar person prototypes
    try:
        similar_prototypes = qdrant.search_against_prototypes(
            query_embedding=embedding,
            limit=limit * 3,  # Get more candidates to deduplicate by person
            score_threshold=min_confidence,
        )
    except Exception as e:
        logger.error(f"Failed to search for similar prototypes for face {face_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to search for similar faces")

    # Step 4: Group by person_id and get person details
    # Use dict to deduplicate by person and keep highest confidence
    person_suggestions: dict[UUID, tuple[str, float]] = {}

    for proto in similar_prototypes:
        if proto.payload is None:
            continue

        person_id_str = proto.payload.get("person_id")
        if not person_id_str:
            # Skip prototypes without person assignment (shouldn't happen)
            continue

        try:
            person_id = UUID(person_id_str)
        except ValueError:
            logger.warning(f"Invalid person_id in prototype payload: {person_id_str}")
            continue

        # Get person from database
        person = await db.get(Person, person_id)
        if not person:
            logger.warning(f"Person {person_id} not found in database")
            continue

        # Only include active persons
        if person.status != PersonStatus.ACTIVE:
            continue

        # Keep highest confidence for each person
        confidence = proto.score
        if person_id not in person_suggestions or confidence > person_suggestions[person_id][1]:
            person_suggestions[person_id] = (person.name, confidence)

    # Step 5: Build response sorted by confidence (highest first)
    suggestions = [
        FaceSuggestionItem(
            person_id=person_id,
            person_name=name,
            confidence=confidence,
        )
        for person_id, (name, confidence) in person_suggestions.items()
    ]

    # Sort by confidence descending
    suggestions.sort(key=lambda x: x.confidence, reverse=True)

    # Limit to requested number
    suggestions = suggestions[:limit]

    logger.info(
        f"Found {len(suggestions)} person suggestions for face {face_id} "
        f"(threshold={min_confidence})"
    )

    return FaceSuggestionsResponse(
        face_id=face_id,
        suggestions=suggestions,
        threshold_used=min_confidence,
    )


# ============ Dual-Mode Clustering Endpoints ============


@router.post("/cluster/dual", response_model=ClusterDualResponse)
async def cluster_faces_dual(
    request: ClusterDualRequest,
) -> ClusterDualResponse:
    """Run dual-mode face clustering (supervised + unsupervised).

    - Phase 1: Assigns unlabeled faces to known people (supervised matching)
    - Phase 2: Clusters remaining unknown faces (unsupervised clustering)

    Can run as background job (default) or synchronously.
    """
    from redis import Redis
    from rq import Queue

    from image_search_service.core.config import get_settings
    from image_search_service.queue.face_jobs import cluster_dual_job

    if request.queue:
        # Queue as background job
        settings = get_settings()
        redis = Redis.from_url(settings.redis_url)
        q = Queue("default", connection=redis)

        job = q.enqueue(
            cluster_dual_job,
            person_threshold=request.person_threshold,
            unknown_method=request.unknown_method,
            unknown_min_size=request.unknown_min_size,
            unknown_eps=request.unknown_eps,
            max_faces=request.max_faces,
        )
        logger.info(f"Queued dual-mode clustering job: {job.id}")
        return ClusterDualResponse(job_id=job.id, status="queued")
    else:
        # Run synchronously (not recommended for large datasets)
        logger.warning("Running dual-mode clustering synchronously (may be slow)")
        result = cluster_dual_job(
            person_threshold=request.person_threshold,
            unknown_method=request.unknown_method,
            unknown_min_size=request.unknown_min_size,
            unknown_eps=request.unknown_eps,
            max_faces=request.max_faces,
        )
        return ClusterDualResponse(status="completed", result=result)


@router.post("/train", response_model=TrainMatchingResponse)
async def train_face_matching(
    request: TrainMatchingRequest,
) -> TrainMatchingResponse:
    """Train face matching model using triplet loss.

    Improves person separation for better clustering accuracy by fine-tuning
    embeddings using labeled face data.

    Can run as background job (default) or synchronously.
    """
    from redis import Redis
    from rq import Queue

    from image_search_service.core.config import get_settings
    from image_search_service.queue.face_jobs import train_person_matching_job

    if request.queue:
        # Queue as background job
        settings = get_settings()
        redis = Redis.from_url(settings.redis_url)
        q = Queue("default", connection=redis)

        job = q.enqueue(
            train_person_matching_job,
            epochs=request.epochs,
            margin=request.margin,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            min_faces_per_person=request.min_faces,
            checkpoint_path=request.checkpoint_path,
        )
        logger.info(f"Queued face training job: {job.id}")
        return TrainMatchingResponse(job_id=job.id, status="queued")
    else:
        # Run synchronously (not recommended - training can be slow)
        logger.warning("Running face training synchronously (may be very slow)")
        result = train_person_matching_job(
            epochs=request.epochs,
            margin=request.margin,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            min_faces_per_person=request.min_faces,
            checkpoint_path=request.checkpoint_path,
        )
        return TrainMatchingResponse(status="completed", result=result)


# ============ Prototype Management Endpoints ============


@router.post(
    "/persons/{person_id}/prototypes/pin",
    response_model=PinPrototypeResponse,
    summary="Pin face as prototype",
)
async def pin_prototype_endpoint(
    person_id: UUID,
    request: PinPrototypeRequest,
    db: AsyncSession = Depends(get_db),
) -> PinPrototypeResponse:
    """Pin a face as prototype with optional era assignment.

    Quotas:
    - Max 3 PRIMARY pins per person
    - Max 1 TEMPORAL pin per era bucket
    """
    from image_search_service.services import prototype_service
    from image_search_service.vector.face_qdrant import get_face_qdrant_client

    qdrant = get_face_qdrant_client()

    prototype = await prototype_service.pin_prototype(
        db=db,
        qdrant=qdrant,
        person_id=person_id,
        face_instance_id=request.face_instance_id,
        age_era_bucket=request.age_era_bucket,
        role=request.role,
        pinned_by=None,  # TODO: Add authentication
    )

    await db.commit()

    return PinPrototypeResponse(
        prototype_id=prototype.id,
        role=prototype.role.value,
        age_era_bucket=prototype.age_era_bucket,
        is_pinned=prototype.is_pinned,
        created_at=prototype.created_at,
    )


@router.delete(
    "/persons/{person_id}/prototypes/{prototype_id}/pin",
    summary="Unpin prototype",
)
async def unpin_prototype_endpoint(
    person_id: UUID,
    prototype_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    """Unpin a prototype. The slot may be filled automatically."""
    from image_search_service.services import prototype_service
    from image_search_service.vector.face_qdrant import get_face_qdrant_client

    qdrant = get_face_qdrant_client()

    await prototype_service.unpin_prototype(
        db=db,
        qdrant=qdrant,
        person_id=person_id,
        prototype_id=prototype_id,
    )

    await db.commit()

    return {"status": "unpinned"}


@router.get(
    "/persons/{person_id}/prototypes",
    response_model=PrototypeListResponse,
    summary="List prototypes",
)
async def list_prototypes_endpoint(
    person_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> PrototypeListResponse:
    """List all prototypes with temporal breakdown and coverage stats."""
    from image_search_service.services import prototype_service

    # Get prototypes
    prototypes = await prototype_service.get_prototypes_for_person(db, person_id)

    # Get coverage stats
    coverage = await prototype_service.get_temporal_coverage(db, person_id)

    # Load face instances for quality scores
    face_ids = [p.face_instance_id for p in prototypes if p.face_instance_id]
    faces_map = {}
    if face_ids:
        face_query = select(FaceInstance).where(FaceInstance.id.in_(face_ids))
        face_result = await db.execute(face_query)
        faces_map = {f.id: f for f in face_result.scalars().all()}

    # Build response items
    items = []
    for proto in prototypes:
        face = faces_map.get(proto.face_instance_id) if proto.face_instance_id else None
        quality_score = face.quality_score if face else None
        # Construct thumbnail URL if face and asset exist
        thumbnail_url = (
            f"/api/v1/images/{face.asset_id}/thumbnail" if face and face.asset_id else None
        )

        items.append(
            PrototypeListItem(
                id=proto.id,
                face_instance_id=proto.face_instance_id,
                role=proto.role.value,
                age_era_bucket=proto.age_era_bucket,
                decade_bucket=proto.decade_bucket,
                is_pinned=proto.is_pinned,
                quality_score=quality_score,
                created_at=proto.created_at,
                thumbnail_url=thumbnail_url,
            )
        )

    # Cast dict values to expected types
    covered_eras = coverage["covered_eras"]
    missing_eras = coverage["missing_eras"]
    coverage_percentage = coverage["coverage_percentage"]
    total_prototypes = coverage["total_prototypes"]

    assert isinstance(covered_eras, list)
    assert isinstance(missing_eras, list)
    assert isinstance(coverage_percentage, float)
    assert isinstance(total_prototypes, int)

    return PrototypeListResponse(
        items=items,
        coverage=TemporalCoverage(
            covered_eras=covered_eras,
            missing_eras=missing_eras,
            coverage_percentage=coverage_percentage,
            total_prototypes=total_prototypes,
        ),
    )


@router.get(
    "/persons/{person_id}/temporal-coverage",
    summary="Get temporal coverage",
)
async def get_temporal_coverage_endpoint(
    person_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> dict[str, list[str] | float | int]:
    """Get detailed temporal coverage report for a person."""
    from image_search_service.services import prototype_service

    return await prototype_service.get_temporal_coverage(db, person_id)


@router.post(
    "/persons/{person_id}/prototypes/recompute",
    response_model=RecomputePrototypesResponse,
    summary="Recompute prototypes",
)
async def recompute_prototypes_endpoint(
    person_id: UUID,
    request: RecomputePrototypesRequest,
    db: AsyncSession = Depends(get_db),
) -> RecomputePrototypesResponse:
    """Trigger temporal re-diversification of prototypes.

    This endpoint recomputes prototypes with temporal diversity:
    - Ensures coverage across age eras
    - Respects pinned prototypes (if preserve_pins=True)
    - Prunes excess prototypes while maintaining quality
    """
    from image_search_service.services import prototype_service
    from image_search_service.vector.face_qdrant import get_face_qdrant_client

    # Get Qdrant client
    qdrant = get_face_qdrant_client()

    # Execute recomputation
    result = await prototype_service.recompute_prototypes_for_person(
        db=db,
        qdrant=qdrant,
        person_id=person_id,
        preserve_pins=request.preserve_pins,
    )

    # Extract results
    prototypes_created = result["prototypes_created"]
    prototypes_removed = result["prototypes_removed"]
    coverage = result["coverage"]

    # Cast dict values to expected types
    assert isinstance(prototypes_created, int)
    assert isinstance(prototypes_removed, int)
    assert isinstance(coverage, dict)

    covered_eras = coverage["covered_eras"]
    missing_eras = coverage["missing_eras"]
    coverage_percentage = coverage["coverage_percentage"]
    total_prototypes = coverage["total_prototypes"]

    assert isinstance(covered_eras, list)
    assert isinstance(missing_eras, list)
    assert isinstance(coverage_percentage, float)
    assert isinstance(total_prototypes, int)

    return RecomputePrototypesResponse(
        prototypes_created=prototypes_created,
        prototypes_removed=prototypes_removed,
        coverage=TemporalCoverage(
            covered_eras=covered_eras,
            missing_eras=missing_eras,
            coverage_percentage=coverage_percentage,
            total_prototypes=total_prototypes,
        ),
    )


# ============ Helper Functions ============


def _face_to_response(face: FaceInstance, person_name: str | None = None) -> FaceInstanceResponse:
    """Convert FaceInstance model to response schema."""
    return FaceInstanceResponse(
        id=face.id,
        asset_id=face.asset_id,
        bbox=BoundingBox(x=face.bbox_x, y=face.bbox_y, width=face.bbox_w, height=face.bbox_h),
        detection_confidence=face.detection_confidence,
        quality_score=face.quality_score,
        cluster_id=face.cluster_id,
        person_id=face.person_id,
        person_name=person_name,
        created_at=face.created_at,
    )
