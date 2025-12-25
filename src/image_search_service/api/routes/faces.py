"""Face detection and recognition API routes."""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
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
    LabelClusterRequest,
    LabelClusterResponse,
    MergePersonsRequest,
    MergePersonsResponse,
    PersonListResponse,
    PersonPhotoGroup,
    PersonPhotosResponse,
    PersonResponse,
    SplitClusterRequest,
    SplitClusterResponse,
    TrainMatchingRequest,
    TrainMatchingResponse,
    TriggerClusteringRequest,
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
    cluster_query = (
        select(
            FaceInstance.cluster_id,
            FaceInstance.person_id,
            func.count(FaceInstance.id).label("face_count"),
            func.avg(FaceInstance.quality_score).label("avg_quality"),
            func.array_agg(FaceInstance.id).label("face_ids"),
        )
        .where(FaceInstance.cluster_id.isnot(None))
        .group_by(FaceInstance.cluster_id, FaceInstance.person_id)
    )

    if not include_labeled:
        cluster_query = cluster_query.where(FaceInstance.person_id.is_(None))

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
        # Get person name if assigned
        person_name = None
        if row.person_id:
            person = await db.get(Person, row.person_id)
            person_name = person.name if person else None

        items.append(
            ClusterSummary(
                cluster_id=row.cluster_id,
                face_count=row.face_count,
                sample_face_ids=row.face_ids[:5] if row.face_ids else [],
                avg_quality=row.avg_quality,
                person_id=row.person_id,
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
    # Get all faces in cluster
    query = select(FaceInstance).where(FaceInstance.cluster_id == cluster_id)
    result = await db.execute(query)
    faces = result.scalars().all()

    if not faces:
        raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")

    # Get person info if assigned
    person_id = faces[0].person_id if faces else None
    person_name = None
    if person_id:
        person = await db.get(Person, person_id)
        person_name = person.name if person else None

    return ClusterDetailResponse(
        cluster_id=cluster_id,
        faces=[_face_to_response(f) for f in faces],
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
    status: str | None = Query(
        None, description="Filter by status: active, merged, hidden"
    ),
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

        proto_count_query = select(func.count()).where(
            PersonPrototype.person_id == person.id
        )
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

        logger.info(
            f"Bulk moved {len(faces)} faces from {source_person.name} to {target_person.name} "
            f"in {len(affected_photo_ids)} photos"
        )

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
    query = select(FaceInstance).where(FaceInstance.asset_id == asset_id)
    result = await db.execute(query)
    faces = result.scalars().all()

    return FaceInstanceListResponse(
        items=[_face_to_response(f) for f in faces],
        total=len(faces),
        page=1,
        page_size=len(faces),
    )


@router.post("/faces/{face_id}/assign", response_model=AssignFaceResponse)
async def assign_face_to_person(
    face_id: UUID,
    request: AssignFaceRequest,
    db: AsyncSession = Depends(get_db),
) -> AssignFaceResponse:
    """Assign a single face instance to a person."""
    from image_search_service.vector.face_qdrant import get_face_qdrant_client

    # Get face instance from DB
    face = await db.get(FaceInstance, face_id)
    if not face:
        raise HTTPException(status_code=404, detail=f"Face {face_id} not found")

    # Get target person from DB
    person = await db.get(Person, request.person_id)
    if not person:
        raise HTTPException(
            status_code=404, detail=f"Person {request.person_id} not found"
        )

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
                detail=f"Face embedding not found in vector database. The face may need to be re-detected.",
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

        logger.info(f"Assigned face {face_id} to person {person.name} ({person.id})")

        return AssignFaceResponse(
            face_id=face.id,
            person_id=person.id,
            person_name=person.name,
        )

    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to assign face {face_id} to person {request.person_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to assign face: {str(e)}")


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


# ============ Helper Functions ============


def _face_to_response(face: FaceInstance) -> FaceInstanceResponse:
    """Convert FaceInstance model to response schema."""
    return FaceInstanceResponse(
        id=face.id,
        asset_id=face.asset_id,
        bbox=BoundingBox(
            x=face.bbox_x, y=face.bbox_y, width=face.bbox_w, height=face.bbox_h
        ),
        detection_confidence=face.detection_confidence,
        quality_score=face.quality_score,
        cluster_id=face.cluster_id,
        person_id=face.person_id,
        person_name=None,  # Would need join to get this
        created_at=face.created_at,
    )
