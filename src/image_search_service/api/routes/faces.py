"""Face detection and recognition API routes."""

import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.api.face_schemas import (
    BoundingBox,
    ClusterDetailResponse,
    ClusteringResultResponse,
    ClusterListResponse,
    ClusterSummary,
    DetectFacesRequest,
    DetectFacesResponse,
    FaceInstanceListResponse,
    FaceInstanceResponse,
    LabelClusterRequest,
    LabelClusterResponse,
    MergePersonsRequest,
    MergePersonsResponse,
    PersonListResponse,
    PersonResponse,
    SplitClusterRequest,
    SplitClusterResponse,
    TriggerClusteringRequest,
)
from image_search_service.db.models import (
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
    status: Optional[str] = Query(
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
