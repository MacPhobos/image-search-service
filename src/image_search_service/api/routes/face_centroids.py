"""Face centroid API routes for person-based face recognition."""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.api.centroid_schemas import (
    CentroidInfo,
    CentroidSuggestion,
    CentroidSuggestionRequest,
    CentroidSuggestionResponse,
    ComputeCentroidsRequest,
    ComputeCentroidsResponse,
    DeleteCentroidsResponse,
    GetCentroidsResponse,
)
from image_search_service.core.config import get_settings
from image_search_service.db.models import (
    CentroidStatus,
    CentroidType,
    FaceInstance,
    ImageAsset,
    Person,
    PersonCentroid,
)
from image_search_service.db.session import get_db
from image_search_service.services.centroid_service import (
    compute_centroids_for_person,
    deprecate_centroids,
    get_active_centroid,
    get_person_face_embeddings,
    is_centroid_stale,
)
from image_search_service.vector.centroid_qdrant import CentroidQdrantClient
from image_search_service.vector.face_qdrant import FaceQdrantClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/faces/centroids", tags=["face-centroids"])


def _get_face_qdrant() -> FaceQdrantClient:
    """Dependency to get FaceQdrantClient singleton."""
    return FaceQdrantClient.get_instance()


def _get_centroid_qdrant() -> CentroidQdrantClient:
    """Dependency to get CentroidQdrantClient singleton."""
    return CentroidQdrantClient.get_instance()


def _build_centroid_info(
    centroid: PersonCentroid,
    is_stale: bool = False,
) -> CentroidInfo:
    """Build CentroidInfo response from database model.

    Args:
        centroid: PersonCentroid database model
        is_stale: Whether centroid is stale and needs rebuild

    Returns:
        CentroidInfo response schema
    """
    return CentroidInfo(
        centroid_id=centroid.centroid_id,
        centroid_type=centroid.centroid_type.value,
        cluster_label=centroid.cluster_label or "global",
        n_faces=centroid.n_faces,
        model_version=centroid.model_version,
        centroid_version=centroid.centroid_version,
        created_at=centroid.created_at,
        is_stale=is_stale,
    )


@router.post("/persons/{person_id}/compute", response_model=ComputeCentroidsResponse)
async def compute_centroids(
    person_id: UUID,
    request: ComputeCentroidsRequest,
    db: AsyncSession = Depends(get_db),
    face_qdrant: FaceQdrantClient = Depends(_get_face_qdrant),
    centroid_qdrant: CentroidQdrantClient = Depends(_get_centroid_qdrant),
) -> ComputeCentroidsResponse:
    """Compute centroids for a person.

    Rebuilds if stale or force_rebuild=True.

    Args:
        person_id: UUID of person to compute centroids for
        request: Computation parameters (force_rebuild, min_faces, etc.)
        db: Database session
        face_qdrant: Face Qdrant client
        centroid_qdrant: Centroid Qdrant client

    Returns:
        ComputeCentroidsResponse with computed centroids

    Raises:
        404: Person not found
        422: Person has fewer than min_faces faces
    """
    settings = get_settings()

    # Verify person exists
    person = await db.get(Person, person_id)
    if not person:
        raise HTTPException(status_code=404, detail=f"Person {person_id} not found")

    # Check if person has enough faces
    face_ids, _ = await get_person_face_embeddings(db, face_qdrant, person_id)
    if len(face_ids) < request.min_faces:
        raise HTTPException(
            status_code=422,
            detail=f"Person has only {len(face_ids)} faces. "
            f"Minimum {request.min_faces} required.",
        )

    # Check if rebuild needed (unless force_rebuild)
    rebuilt = False
    stale_reason = None

    if not request.force_rebuild:
        existing = await get_active_centroid(db, person_id, CentroidType.GLOBAL)
        if existing:
            stale = is_centroid_stale(
                existing,
                face_ids,
                settings.centroid_model_version,
                settings.centroid_algorithm_version,
            )
            if not stale:
                # Existing centroid is fresh, return it
                logger.debug(f"Centroid {existing.centroid_id} is fresh, skipping rebuild")
                return ComputeCentroidsResponse(
                    person_id=person_id,
                    centroids=[_build_centroid_info(existing, is_stale=False)],
                    rebuilt=False,
                    stale_reason=None,
                )
            else:
                stale_reason = "Face assignments or model version changed"
                rebuilt = True
    else:
        stale_reason = "Force rebuild requested"
        rebuilt = True

    # Compute centroid
    centroid = await compute_centroids_for_person(
        db=db,
        face_qdrant=face_qdrant,
        centroid_qdrant=centroid_qdrant,
        person_id=person_id,
        force_rebuild=request.force_rebuild,
    )

    if not centroid:
        raise HTTPException(
            status_code=422,
            detail="Failed to compute centroid (insufficient faces or error)",
        )

    await db.commit()

    logger.info(
        f"Computed centroid {centroid.centroid_id} for person {person.name} "
        f"({len(face_ids)} faces)"
    )

    return ComputeCentroidsResponse(
        person_id=person_id,
        centroids=[_build_centroid_info(centroid, is_stale=False)],
        rebuilt=rebuilt,
        stale_reason=stale_reason,
    )


@router.get("/persons/{person_id}", response_model=GetCentroidsResponse)
async def get_centroids(
    person_id: UUID,
    include_stale: bool = False,
    db: AsyncSession = Depends(get_db),
    face_qdrant: FaceQdrantClient = Depends(_get_face_qdrant),
) -> GetCentroidsResponse:
    """Get current centroids for a person with staleness check.

    Args:
        person_id: UUID of person
        include_stale: Include deprecated/failed centroids (default: False)
        db: Database session
        face_qdrant: Face Qdrant client

    Returns:
        GetCentroidsResponse with centroids and staleness info

    Raises:
        404: Person not found
    """
    settings = get_settings()

    # Verify person exists
    person = await db.get(Person, person_id)
    if not person:
        raise HTTPException(status_code=404, detail=f"Person {person_id} not found")

    # Query active centroids
    query = select(PersonCentroid).where(
        PersonCentroid.person_id == person_id,
    )

    if not include_stale:
        query = query.where(PersonCentroid.status == CentroidStatus.ACTIVE)

    query = query.order_by(PersonCentroid.created_at.desc())

    result = await db.execute(query)
    centroids = result.scalars().all()

    if not centroids:
        return GetCentroidsResponse(
            person_id=person_id,
            centroids=[],
            is_stale=True,
            stale_reason="No centroids found",
        )

    # Check staleness for active centroids
    face_ids, _ = await get_person_face_embeddings(db, face_qdrant, person_id)

    centroid_infos = []
    any_stale = False
    stale_reason = None

    for centroid in centroids:
        if centroid.status == CentroidStatus.ACTIVE:
            stale = is_centroid_stale(
                centroid,
                face_ids,
                settings.centroid_model_version,
                settings.centroid_algorithm_version,
            )
            if stale:
                any_stale = True
                stale_reason = "Face assignments or model version changed"
            centroid_infos.append(_build_centroid_info(centroid, is_stale=stale))
        else:
            # Deprecated/failed centroids are always marked stale
            centroid_infos.append(_build_centroid_info(centroid, is_stale=True))

    return GetCentroidsResponse(
        person_id=person_id,
        centroids=centroid_infos,
        is_stale=any_stale,
        stale_reason=stale_reason,
    )


@router.post(
    "/persons/{person_id}/suggestions", response_model=CentroidSuggestionResponse
)
async def get_centroid_suggestions(
    person_id: UUID,
    request: CentroidSuggestionRequest,
    db: AsyncSession = Depends(get_db),
    face_qdrant: FaceQdrantClient = Depends(_get_face_qdrant),
    centroid_qdrant: CentroidQdrantClient = Depends(_get_centroid_qdrant),
) -> CentroidSuggestionResponse:
    """Get face suggestions using person centroids.

    Auto-rebuilds centroids if stale and auto_rebuild=True.

    Args:
        person_id: UUID of person
        request: Search parameters (min_similarity, max_results, etc.)
        db: Database session
        face_qdrant: Face Qdrant client
        centroid_qdrant: Centroid Qdrant client

    Returns:
        CentroidSuggestionResponse with face suggestions

    Raises:
        404: Person not found
        422: Person has no centroids and insufficient faces to build
    """
    settings = get_settings()

    # Verify person exists
    person = await db.get(Person, person_id)
    if not person:
        raise HTTPException(status_code=404, detail=f"Person {person_id} not found")

    # Get or compute centroid
    centroid = await get_active_centroid(db, person_id, CentroidType.GLOBAL)
    rebuilt_centroids = False

    if not centroid or (
        request.auto_rebuild
        and is_centroid_stale(
            centroid,
            (await get_person_face_embeddings(db, face_qdrant, person_id))[0],
            settings.centroid_model_version,
            settings.centroid_algorithm_version,
        )
    ):
        # Compute new centroid
        centroid = await compute_centroids_for_person(
            db=db,
            face_qdrant=face_qdrant,
            centroid_qdrant=centroid_qdrant,
            person_id=person_id,
            force_rebuild=True,
        )

        if not centroid:
            raise HTTPException(
                status_code=422,
                detail="Person has no centroids and insufficient faces to build",
            )

        rebuilt_centroids = True
        await db.commit()
        logger.info(f"Rebuilt centroid {centroid.centroid_id} for person {person.name}")

    # Get centroid vector from Qdrant
    centroid_vector = centroid_qdrant.get_centroid_vector(centroid.centroid_id)
    if not centroid_vector:
        raise HTTPException(
            status_code=500,
            detail=f"Centroid vector not found in Qdrant for {centroid.centroid_id}",
        )

    # Search faces collection with centroid
    scored_points = centroid_qdrant.search_faces_with_centroid(
        centroid_vector=centroid_vector,
        limit=request.max_results,
        score_threshold=request.min_similarity,
        exclude_person_id=person_id if request.unassigned_only else None,
    )

    # Build suggestions
    suggestions: list[CentroidSuggestion] = []
    face_instance_ids = [
        UUID(str(point.payload.get("face_instance_id")))
        for point in scored_points
        if point.payload
    ]

    # Bulk load face instances and their assets for deduplication
    if face_instance_ids:
        face_query = select(FaceInstance).where(FaceInstance.id.in_(face_instance_ids))
        face_result = await db.execute(face_query)
        faces_map = {f.id: f for f in face_result.scalars().all()}

        # Bulk load assets to get perceptual_hash
        asset_ids = list({f.asset_id for f in faces_map.values()})
        asset_query = select(ImageAsset).where(ImageAsset.id.in_(asset_ids))
        asset_result = await db.execute(asset_query)
        assets_map = {a.id: a for a in asset_result.scalars().all()}

        # Deduplicate by perceptual_hash - keep first (highest scoring) occurrence
        seen_hashes: set[str] = set()

        for point in scored_points:
            if not point.payload:
                continue

            face_id = UUID(str(point.payload.get("face_instance_id")))
            face = faces_map.get(face_id)

            if not face:
                continue

            # Apply filters
            if request.unassigned_only and face.person_id is not None:
                continue

            # Note: is_prototype field doesn't exist in FaceInstance model
            # Skipping exclude_prototypes filter for now

            # Deduplicate by perceptual_hash
            asset = assets_map.get(face.asset_id)
            if asset and asset.perceptual_hash:
                if asset.perceptual_hash in seen_hashes:
                    continue
                seen_hashes.add(asset.perceptual_hash)

            suggestions.append(
                CentroidSuggestion(
                    face_instance_id=face_id,
                    asset_id=str(face.asset_id),
                    score=point.score,
                    matched_centroid=str(centroid.centroid_id),
                    thumbnail_url=f"/api/v1/images/{face.asset_id}/thumbnail",
                )
            )

    # Sort by score descending and limit
    suggestions.sort(key=lambda s: s.score, reverse=True)
    suggestions = suggestions[: request.max_results]

    # Log deduplication statistics
    total_before_dedup = len(scored_points)
    total_after_dedup = len(suggestions)
    duplicates_removed = total_before_dedup - total_after_dedup

    logger.debug(
        f"Deduplication: {total_before_dedup} → {total_after_dedup} results "
        f"({duplicates_removed} duplicates removed)"
    )

    logger.info(
        f"Found {len(suggestions)} face suggestions for person {person.name} "
        f"using centroid {centroid.centroid_id}"
    )

    return CentroidSuggestionResponse(
        person_id=person_id,
        centroids_used=[centroid.centroid_id],
        suggestions=suggestions,
        total_found=len(suggestions),
        rebuilt_centroids=rebuilt_centroids,
    )


@router.delete("/persons/{person_id}", response_model=DeleteCentroidsResponse)
async def delete_centroids(
    person_id: UUID,
    db: AsyncSession = Depends(get_db),
    centroid_qdrant: CentroidQdrantClient = Depends(_get_centroid_qdrant),
) -> DeleteCentroidsResponse:
    """Delete all centroids for a person (triggers rebuild on next request).

    Marks all centroids as deprecated in DB and deletes from Qdrant.

    Args:
        person_id: UUID of person
        db: Database session
        centroid_qdrant: Centroid Qdrant client

    Returns:
        DeleteCentroidsResponse with deletion count

    Raises:
        404: Person not found
    """
    # Verify person exists
    person = await db.get(Person, person_id)
    if not person:
        raise HTTPException(status_code=404, detail=f"Person {person_id} not found")

    # Deprecate centroids in DB
    deleted_count = await deprecate_centroids(db, centroid_qdrant, person_id)

    # Delete from Qdrant
    if deleted_count > 0:
        # Get all centroid IDs to delete from Qdrant
        query = select(PersonCentroid).where(
            PersonCentroid.person_id == person_id,
            PersonCentroid.status == CentroidStatus.DEPRECATED,
        )
        result = await db.execute(query)
        centroids = result.scalars().all()

        for centroid in centroids:
            try:
                centroid_qdrant.delete_centroid(centroid.qdrant_point_id)
            except Exception as e:
                logger.warning(
                    f"Failed to delete centroid {centroid.centroid_id} from Qdrant: {e}"
                )

    await db.commit()

    logger.info(f"Deleted {deleted_count} centroids for person {person.name}")

    return DeleteCentroidsResponse(
        person_id=person_id,
        deleted_count=deleted_count,
    )
