"""Face detection and recognition API routes."""

import logging
from datetime import date, datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import Integer, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.api.face_schemas import (
    AssignFaceRequest,
    AssignFaceResponse,
    AssignmentEventResponse,
    AssignmentHistoryResponse,
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
    PersonDetailResponse,
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
    SuggestionRegenerationResponse,
    TemporalCoverage,
    TrainMatchingRequest,
    TrainMatchingResponse,
    TriggerClusteringRequest,
    UnassignFaceResponse,
    UnifiedPeopleListResponse,
    UpdatePersonRequest,
    UpdatePersonResponse,
)
from image_search_service.db.models import (
    FaceAssignmentEvent,
    FaceInstance,
    FaceSuggestion,
    FaceSuggestionStatus,
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
    min_confidence: float | None = Query(
        None, ge=0.0, le=1.0, description="Minimum intra-cluster confidence threshold"
    ),
    min_cluster_size: int | None = Query(
        None, ge=1, description="Minimum number of faces per cluster"
    ),
    db: AsyncSession = Depends(get_db),
) -> ClusterListResponse:
    """List face clusters with pagination and optional filtering.

    Supports filtering by:
    - include_labeled: Whether to include clusters assigned to persons
    - min_confidence: Minimum average pairwise similarity within cluster
    - min_cluster_size: Minimum number of faces in cluster

    When min_confidence is specified, cluster confidence is calculated on-the-fly
    by comparing face embeddings in Qdrant. This may add latency for large clusters.
    """
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

    # Filter by minimum cluster size (SQL-level filtering)
    if min_cluster_size is not None:
        cluster_query = cluster_query.having(
            func.count(FaceInstance.id) >= min_cluster_size
        )

    # Note: min_confidence filtering is done post-query since it requires
    # fetching embeddings from Qdrant for similarity calculation

    # Count total clusters BEFORE min_confidence filtering
    # (confidence filtering happens post-query)
    count_subquery = cluster_query.subquery()
    count_query = select(func.count()).select_from(count_subquery)
    total_result = await db.execute(count_query)
    pre_filter_total = total_result.scalar() or 0

    # Get paginated results (we'll fetch more and filter if needed)
    # For confidence filtering, we need to over-fetch since some may be filtered out
    fetch_limit = page_size
    if min_confidence is not None:
        # Fetch 3x requested to account for confidence filtering
        fetch_limit = min(page_size * 3, 100)

    paginated_query = cluster_query.offset((page - 1) * page_size).limit(fetch_limit)
    result = await db.execute(paginated_query)
    rows = result.all()

    # Initialize clustering service for confidence calculation
    clustering_service = None
    if min_confidence is not None:
        from image_search_service.services.face_clustering_service import (
            FaceClusteringService,
        )
        from image_search_service.vector.face_qdrant import get_face_qdrant_client

        qdrant = get_face_qdrant_client()
        clustering_service = FaceClusteringService(db, qdrant)

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

        # Calculate cluster confidence if filtering requested
        cluster_confidence = None
        if clustering_service is not None:
            try:
                # Get Qdrant point IDs for this cluster's faces
                qdrant_point_ids_query = select(FaceInstance.qdrant_point_id).where(
                    FaceInstance.cluster_id == row.cluster_id
                )
                qdrant_result = await db.execute(qdrant_point_ids_query)
                qdrant_point_ids = [r[0] for r in qdrant_result.all()]

                if qdrant_point_ids:
                    cluster_confidence = await clustering_service.calculate_cluster_confidence(
                        cluster_id=row.cluster_id,
                        qdrant_point_ids=qdrant_point_ids,
                    )

                    # Filter out clusters below confidence threshold
                    if min_confidence is not None and cluster_confidence < min_confidence:
                        logger.debug(
                            f"Filtered out cluster {row.cluster_id} "
                            f"(confidence {cluster_confidence:.3f} < {min_confidence})"
                        )
                        continue
            except Exception as e:
                logger.warning(
                    f"Failed to calculate confidence for cluster {row.cluster_id}: {e}"
                )
                # Don't filter out clusters with calculation errors
                cluster_confidence = None

        # Select representative face (highest quality)
        representative_face_id = None
        if face_ids_list:
            # Simple approach: sort by quality and pick first
            # For more sophisticated selection, use clustering_service
            representative_face_id = face_ids_list[0]  # Placeholder for now

        items.append(
            ClusterSummary(
                cluster_id=row.cluster_id,
                face_count=row.face_count,
                sample_face_ids=face_ids_list[:5],
                avg_quality=row.avg_quality,
                cluster_confidence=cluster_confidence,
                representative_face_id=representative_face_id,
                person_id=person_id,
                person_name=person_name,
            )
        )

        # Stop if we have enough items after filtering
        if len(items) >= page_size:
            break

    # Adjust total count if confidence filtering was applied
    total = len(items) if min_confidence is not None else pre_filter_total

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


# ============ Unified People Endpoint ============


@router.get("/people", response_model=UnifiedPeopleListResponse)
async def list_unified_people(
    include_identified: bool = Query(True, description="Include identified persons"),
    include_unidentified: bool = Query(True, description="Include unidentified clusters"),
    include_noise: bool = Query(False, description="Include noise/unknown faces"),
    sort_by: str = Query(
        "face_count",
        regex="^(face_count|name)$",
        description="Sort by: face_count, name",
    ),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="Sort order"),
    db: AsyncSession = Depends(get_db),
) -> UnifiedPeopleListResponse:
    """
    List all people (identified and unidentified) in a unified view.

    This endpoint combines:
    - Identified people (with names from Person table)
    - Unidentified clusters (face groups without names)
    - Optionally noise faces (ungrouped faces)

    The unified view eliminates the "Clusters" vs "People" distinction,
    treating both as "people" with different types (identified/unidentified/noise).

    Args:
        include_identified: Include persons with assigned names
        include_unidentified: Include face clusters without person assignment
        include_noise: Include noise/outlier faces (cluster_id = '-1' or NULL)
        sort_by: Field to sort by (face_count or name)
        sort_order: Sort order (asc or desc)

    Returns:
        Unified list of people with counts broken down by type
    """
    from image_search_service.services.person_service import PersonService

    service = PersonService(db)
    return await service.get_all_people(
        include_identified=include_identified,
        include_unidentified=include_unidentified,
        include_noise=include_noise,
        sort_by=sort_by,
        sort_order=sort_order,
    )


# ============ Person Endpoints ============


@router.get("/persons", response_model=PersonListResponse)
async def list_persons(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=1000),
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


@router.get("/persons/{person_id}", response_model=PersonDetailResponse)
async def get_person(
    person_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> PersonDetailResponse:
    """Get a single person by ID with detailed information.

    Returns:
        PersonDetailResponse with face_count, photo_count, and thumbnail_url

    Raises:
        HTTPException: 404 if person not found
    """
    # Get person from database
    person = await db.get(Person, person_id)
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")

    # Count faces assigned to this person
    face_count_query = select(func.count()).where(FaceInstance.person_id == person_id)
    face_count = (await db.execute(face_count_query)).scalar() or 0

    # Count distinct photos (assets) containing this person's faces
    photo_count_query = (
        select(func.count(func.distinct(FaceInstance.asset_id)))
        .where(FaceInstance.person_id == person_id)
    )
    photo_count = (await db.execute(photo_count_query)).scalar() or 0

    # Get thumbnail URL from the highest quality face
    thumbnail_url = None
    if face_count > 0:
        # Get highest quality face for this person
        best_face_query = (
            select(FaceInstance)
            .where(FaceInstance.person_id == person_id)
            .order_by(FaceInstance.quality_score.desc())
            .limit(1)
        )
        best_face_result = await db.execute(best_face_query)
        best_face = best_face_result.scalar_one_or_none()

        if best_face:
            thumbnail_url = f"/api/v1/images/{best_face.asset_id}/thumbnail"

    return PersonDetailResponse(
        id=person.id,
        name=person.name,
        status=person.status.value,
        face_count=face_count,
        photo_count=photo_count,
        thumbnail_url=thumbnail_url,
        created_at=person.created_at,
        updated_at=person.updated_at,
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


@router.patch("/persons/{person_id}", response_model=UpdatePersonResponse)
async def update_person(
    person_id: UUID,
    request: UpdatePersonRequest,
    db: AsyncSession = Depends(get_db),
) -> UpdatePersonResponse:
    """Update person's name and/or birth_date.

    Args:
        person_id: UUID of the person to update
        request: UpdatePersonRequest with optional name and/or birth_date
        db: Database session dependency

    Returns:
        UpdatePersonResponse with updated person details

    Raises:
        HTTPException: 404 if person not found, 409 if name already exists
    """
    # Get person from database
    person = await db.get(Person, person_id)
    if not person:
        raise HTTPException(status_code=404, detail=f"Person {person_id} not found")

    # Update name if provided
    if request.name is not None:
        # Check for existing person with same name (case-insensitive), excluding current person
        person_query = select(Person).where(
            func.lower(Person.name) == request.name.lower(),
            Person.id != person_id,
        )
        person_result = await db.execute(person_query)
        existing_person = person_result.scalar_one_or_none()

        if existing_person:
            raise HTTPException(
                status_code=409,
                detail=f"Person with name '{request.name}' already exists",
            )

        person.name = request.name

    # Update birth_date if provided (can be set to None to clear)
    if request.birth_date is not None:
        person.birth_date = request.birth_date

    # Commit changes
    await db.commit()
    await db.refresh(person)

    logger.info(
        f"Updated person {person_id}: name={person.name}, birth_date={person.birth_date}"
    )

    return UpdatePersonResponse(
        id=person.id,
        name=person.name,
        birth_date=person.birth_date,
        status=person.status.value,
        created_at=person.created_at,
        updated_at=person.updated_at,
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

    # Step 4: Get paginated assets with taken_at and path
    paginated_assets_query = (
        select(ImageAsset.id, ImageAsset.taken_at, ImageAsset.path)
        .where(ImageAsset.id.in_(select(asset_subquery.c.asset_id)))
        .order_by(ImageAsset.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    assets_result = await db.execute(paginated_assets_query)
    assets_data = assets_result.all()
    asset_ids = [row[0] for row in assets_data]

    # Build maps for asset metadata
    taken_at_by_asset = {row[0]: row[1] for row in assets_data}
    path_by_asset = {row[0]: row[2] for row in assets_data}

    # Step 5: For each photo, get ALL faces with Person info (for birth_date)
    if asset_ids:
        faces_query = (
            select(FaceInstance, Person.name, Person.birth_date)
            .outerjoin(Person, FaceInstance.person_id == Person.id)
            .where(FaceInstance.asset_id.in_(asset_ids))
            .order_by(FaceInstance.asset_id, FaceInstance.bbox_x)
        )
        faces_result = await db.execute(faces_query)
        faces_data = faces_result.all()

        # Group faces by asset_id
        faces_by_asset: dict[int, list[tuple[FaceInstance, str | None, date | None]]] = {}
        for face, person_name, birth_date in faces_data:
            if face.asset_id not in faces_by_asset:
                faces_by_asset[face.asset_id] = []
            faces_by_asset[face.asset_id].append((face, person_name, birth_date))
    else:
        faces_by_asset = {}

    # Step 6: Build PersonPhotoGroup for each photo
    items = []
    for asset_id in asset_ids:
        faces_in_photo = faces_by_asset.get(asset_id, [])
        photo_taken_at = taken_at_by_asset.get(asset_id)

        # Convert to FaceInPhoto schema
        face_schemas = []
        has_non_person_faces = False

        for face, face_person_name, face_birth_date in faces_in_photo:
            # Calculate age at time of photo
            person_age_at_photo = calculate_age_at_date(face_birth_date, photo_taken_at)

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
                    person_age_at_photo=person_age_at_photo,
                )
            )

            # Check if this face belongs to a different person or no person
            if face.person_id != person_id:
                has_non_person_faces = True

        items.append(
            PersonPhotoGroup(
                photo_id=asset_id,
                taken_at=photo_taken_at,
                thumbnail_url=f"/api/v1/images/{asset_id}/thumbnail",
                full_url=f"/api/v1/images/{asset_id}/full",
                path=path_by_asset.get(asset_id, ""),
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


@router.post(
    "/persons/{person_id}/suggestions/regenerate",
    response_model=SuggestionRegenerationResponse,
)
async def regenerate_suggestions_for_person(
    person_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> SuggestionRegenerationResponse:
    """Trigger background job to regenerate suggestions using current prototypes.

    This expires existing pending suggestions and queues a new search
    using the highest quality prototype.

    Args:
        person_id: UUID of the person to regenerate suggestions for
        db: Database session dependency

    Returns:
        SuggestionRegenerationResponse with status and message

    Raises:
        HTTPException: 404 if person not found, 400 if no prototypes exist
    """
    from redis import Redis
    from rq import Queue

    from image_search_service.core.config import get_settings
    from image_search_service.queue.face_jobs import propagate_person_label_multiproto_job
    from image_search_service.services.prototype_service import get_prototypes_for_person

    # Step 1: Verify person exists
    person = await db.get(Person, person_id)
    if not person:
        raise HTTPException(status_code=404, detail=f"Person {person_id} not found")

    # Step 2: Get all prototypes for this person
    prototypes = await get_prototypes_for_person(db, person_id)

    if not prototypes:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot regenerate suggestions: person {person.name} has no prototypes",
        )

    # Step 3: Queue multi-prototype propagation job
    # Note: Expiration of old suggestions is handled inside the job
    try:
        settings = get_settings()
        redis_conn = Redis.from_url(settings.redis_url)
        queue = Queue("default", connection=redis_conn)

        queue.enqueue(
            propagate_person_label_multiproto_job,
            person_id=str(person_id),
            min_confidence=0.7,
            max_suggestions=50,
            job_timeout="10m",
        )
        logger.info(
            f"Queued multi-prototype suggestion regeneration job for person {person.name} "
            f"using {len(prototypes)} prototypes"
        )
    except Exception as e:
        logger.error(f"Failed to queue suggestion regeneration job: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to queue suggestion regeneration job",
        )

    return SuggestionRegenerationResponse(
        status="queued",
        message=f"Multi-prototype suggestion regeneration queued for person {person.name}",
        expired_count=0,  # Will be reported in job result
    )


@router.get(
    "/persons/{person_id}/assignment-history",
    response_model=AssignmentHistoryResponse,
)
async def get_person_assignment_history(
    person_id: UUID,
    limit: int = Query(20, ge=1, le=100, description="Maximum events to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    operation: str | None = Query(None, description="Filter by operation type"),
    since: datetime | None = Query(None, description="Only events after this time"),
    db: AsyncSession = Depends(get_db),
) -> AssignmentHistoryResponse:
    """Get assignment history for a specific person.

    Returns events where this person is either:
    - The source (from_person_id) - faces were removed from this person
    - The destination (to_person_id) - faces were assigned to this person

    Args:
        person_id: UUID of the person to get history for
        limit: Maximum number of events to return (1-100)
        offset: Offset for pagination
        operation: Optional filter by operation type (REMOVE_FROM_PERSON, MOVE_TO_PERSON)
        since: Optional filter to only show events after this timestamp
        db: Database session dependency

    Returns:
        AssignmentHistoryResponse with paginated events

    Raises:
        HTTPException: 404 if person not found
    """
    # Verify person exists
    person = await db.get(Person, person_id)
    if not person:
        raise HTTPException(status_code=404, detail=f"Person {person_id} not found")

    # Build query for events involving this person
    query = select(FaceAssignmentEvent).where(
        or_(
            FaceAssignmentEvent.from_person_id == person_id,
            FaceAssignmentEvent.to_person_id == person_id,
        )
    )

    # Apply filters
    if operation:
        query = query.where(FaceAssignmentEvent.operation == operation)
    if since:
        query = query.where(FaceAssignmentEvent.created_at >= since)

    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total = await db.scalar(count_query) or 0

    # Apply pagination and ordering
    query = (
        query.order_by(FaceAssignmentEvent.created_at.desc())
        .offset(offset)
        .limit(limit)
    )

    result = await db.execute(query)
    events = result.scalars().all()

    # Collect unique person IDs for enrichment
    person_ids: set[UUID] = set()
    for event in events:
        if event.from_person_id:
            person_ids.add(event.from_person_id)
        if event.to_person_id:
            person_ids.add(event.to_person_id)

    # Fetch person names for enrichment
    persons_map: dict[UUID, str] = {}
    if person_ids:
        persons_query = select(Person).where(Person.id.in_(person_ids))
        persons_result = await db.execute(persons_query)
        persons_map = {p.id: p.name for p in persons_result.scalars().all()}

    # Build response
    event_responses = []
    for event in events:
        # Convert affected_face_instance_ids from list[str] to list[UUID]
        face_instance_ids = []
        if event.affected_face_instance_ids:
            face_instance_ids = [UUID(fid) for fid in event.affected_face_instance_ids]

        # Get asset_ids (already integers)
        asset_ids = event.affected_photo_ids or []

        event_responses.append(
            AssignmentEventResponse(
                id=event.id,
                operation=event.operation,
                created_at=event.created_at,
                face_count=event.face_count,
                photo_count=event.photo_count,
                face_instance_ids=face_instance_ids,
                asset_ids=asset_ids,
                from_person_id=event.from_person_id,
                to_person_id=event.to_person_id,
                from_person_name=persons_map.get(event.from_person_id)
                if event.from_person_id
                else None,
                to_person_name=persons_map.get(event.to_person_id)
                if event.to_person_id
                else None,
                note=event.note,
            )
        )

    return AssignmentHistoryResponse(
        events=event_responses,
        total=total,
        offset=offset,
        limit=limit,
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

        # Step 4.5: Expire all pending suggestions where these faces were the source
        from datetime import UTC, datetime

        expire_result = await db.execute(
            update(FaceSuggestion)
            .where(
                FaceSuggestion.source_face_id.in_(face_ids),
                FaceSuggestion.status == FaceSuggestionStatus.PENDING.value,
            )
            .values(
                status=FaceSuggestionStatus.EXPIRED.value,
                reviewed_at=datetime.now(UTC),
            )
        )
        expired_count = expire_result.rowcount  # type: ignore[attr-defined]

        if expired_count > 0:
            logger.info(
                f"Expired {expired_count} pending suggestions based on "
                f"{len(face_ids)} unassigned faces"
            )

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

        # Step 5.5: Expire pending suggestions based on old person assignment
        # When faces move to a different person, old suggestions are no longer valid
        from datetime import UTC, datetime

        expire_result = await db.execute(
            update(FaceSuggestion)
            .where(
                FaceSuggestion.source_face_id.in_(face_ids),
                FaceSuggestion.status == FaceSuggestionStatus.PENDING.value,
            )
            .values(
                status=FaceSuggestionStatus.EXPIRED.value,
                reviewed_at=datetime.now(UTC),
            )
        )
        expired_count = expire_result.rowcount  # type: ignore[attr-defined]

        if expired_count > 0:
            logger.info(
                f"Expired {expired_count} pending suggestions based on {len(face_ids)} moved faces"
            )

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

        # Expire all pending suggestions where this face was the source
        # When a face is unassigned, suggestions based on it are no longer valid
        from datetime import UTC, datetime

        expire_result = await db.execute(
            update(FaceSuggestion)
            .where(
                FaceSuggestion.source_face_id == face_id,
                FaceSuggestion.status == FaceSuggestionStatus.PENDING.value,
            )
            .values(
                status=FaceSuggestionStatus.EXPIRED.value,
                reviewed_at=datetime.now(UTC),
            )
        )
        expired_count = expire_result.rowcount  # type: ignore[attr-defined]

        if expired_count > 0:
            logger.info(
                f"Expired {expired_count} pending suggestions based on face {face_id}"
            )

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


@router.delete(
    "/persons/{person_id}/prototypes/{prototype_id}",
    summary="Delete prototype",
)
async def delete_prototype_endpoint(
    person_id: UUID,
    prototype_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    """Delete a prototype assignment entirely."""
    from image_search_service.services import prototype_service
    from image_search_service.vector.face_qdrant import get_face_qdrant_client

    qdrant = get_face_qdrant_client()

    await prototype_service.delete_prototype(
        db=db,
        qdrant=qdrant,
        person_id=person_id,
        prototype_id=prototype_id,
    )

    await db.commit()

    return {"status": "deleted"}


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
    - Optionally triggers suggestion rescan after recomputation
    """
    from datetime import UTC, datetime

    from redis import Redis
    from rq import Queue

    from image_search_service.core.config import get_settings
    from image_search_service.queue.face_jobs import propagate_person_label_multiproto_job
    from image_search_service.services import prototype_service
    from image_search_service.vector.face_qdrant import get_face_qdrant_client

    # Get settings to determine if auto-rescan is enabled
    settings = get_settings()

    # Determine if we should trigger rescan
    trigger_rescan = request.trigger_rescan
    if trigger_rescan is None:
        trigger_rescan = settings.face_suggestions_auto_rescan_on_recompute

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

    # Initialize rescan status
    rescan_triggered = False
    rescan_message = None

    # Optional auto-rescan after recomputation
    if trigger_rescan:
        # Get person for logging
        person = await db.get(Person, person_id)
        if person:
            # Get prototypes for this person
            prototypes = await prototype_service.get_prototypes_for_person(db, person_id)

            if prototypes:
                # Find highest quality prototype by loading face instances
                face_ids = [p.face_instance_id for p in prototypes if p.face_instance_id]
                if face_ids:
                    face_query = select(FaceInstance).where(FaceInstance.id.in_(face_ids))
                    face_result = await db.execute(face_query)
                    faces = list(face_result.scalars().all())

                    if faces:
                        # Find face with highest quality score
                        best_face = max(faces, key=lambda f: f.quality_score or 0.0)

                        # Expire existing pending suggestions for this person
                        expire_result = await db.execute(
                            update(FaceSuggestion)
                            .where(
                                FaceSuggestion.suggested_person_id == person_id,
                                FaceSuggestion.status == FaceSuggestionStatus.PENDING.value,
                            )
                            .values(
                                status=FaceSuggestionStatus.EXPIRED.value,
                                reviewed_at=datetime.now(UTC),
                            )
                        )
                        expired_count = expire_result.rowcount  # type: ignore[attr-defined]
                        await db.commit()

                        # Queue multi-prototype propagation job
                        try:
                            redis_conn = Redis.from_url(settings.redis_url)
                            queue = Queue("default", connection=redis_conn)

                            queue.enqueue(
                                propagate_person_label_multiproto_job,
                                person_id=str(person_id),
                                min_confidence=0.7,
                                max_suggestions=50,
                                job_timeout="10m",
                            )

                            rescan_triggered = True
                            rescan_message = (
                                f"Multi-prototype suggestion rescan queued for {person.name}. "
                                f"{expired_count} old suggestions expired. "
                                f"Using {len(prototypes)} prototypes."
                            )
                            logger.info(
                                f"Auto-triggered multi-prototype suggestion rescan "
                                f"for {person.name} using {len(prototypes)} prototypes"
                            )
                        except Exception as e:
                            logger.error(f"Failed to queue auto-rescan job: {e}")
                            rescan_message = f"Failed to queue rescan job: {str(e)}"

    return RecomputePrototypesResponse(
        prototypes_created=prototypes_created,
        prototypes_removed=prototypes_removed,
        coverage=TemporalCoverage(
            covered_eras=covered_eras,
            missing_eras=missing_eras,
            coverage_percentage=coverage_percentage,
            total_prototypes=total_prototypes,
        ),
        rescan_triggered=rescan_triggered,
        rescan_message=rescan_message,
    )


# ============ Helper Functions ============


def calculate_age_at_date(birth_date: date | None, photo_date: datetime | None) -> int | None:
    """Calculate age at the time a photo was taken.

    Args:
        birth_date: Person's birth date
        photo_date: Date the photo was taken (from EXIF)

    Returns:
        Age in years at the time of the photo, or None if either date is missing

    Examples:
        >>> from datetime import date, datetime
        >>> calculate_age_at_date(date(1990, 6, 15), datetime(2020, 8, 1))
        30
        >>> calculate_age_at_date(date(1990, 6, 15), datetime(2020, 3, 1))
        29
        >>> calculate_age_at_date(None, datetime(2020, 1, 1))
        None
    """
    if not birth_date or not photo_date:
        return None

    photo_d = photo_date.date()
    age = photo_d.year - birth_date.year

    # Adjust if birthday hasn't occurred yet in the photo year
    if (photo_d.month, photo_d.day) < (birth_date.month, birth_date.day):
        age -= 1

    return max(0, age)  # Never return negative age


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
