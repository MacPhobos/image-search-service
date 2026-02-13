"""Unknown person discovery and management routes."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import and_, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.api.unknown_person_schemas import (
    AcceptUnknownPersonRequest,
    AcceptUnknownPersonResponse,
    DiscoverJobResponse,
    DiscoverUnknownPersonsRequest,
    DismissUnknownPersonRequest,
    DismissUnknownPersonResponse,
    FaceInGroupResponse,
    MergeGroupsRequest,
    MergeGroupsResponse,
    MergeSuggestion,
    MergeSuggestionsResponse,
    UnknownPersonCandidateDetail,
    UnknownPersonCandidateGroup,
    UnknownPersonCandidatesResponse,
    UnknownPersonsStatsResponse,
)
from image_search_service.core.config import get_settings
from image_search_service.db.models import DismissedUnknownPersonGroup, FaceInstance
from image_search_service.db.session import get_db
from image_search_service.queue.worker import get_queue, get_redis
from image_search_service.services.cluster_labeling_service import ClusterLabelingService
from image_search_service.services.unknown_person_service import (
    compute_membership_hash,
    dismiss_group,
    get_dismissed_hashes,
)
from image_search_service.vector.face_qdrant import get_face_qdrant_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/faces/unknown-persons", tags=["unknown-persons"])


# ============ Endpoint 1: GET /stats ============


@router.get("/stats", response_model=UnknownPersonsStatsResponse)
async def get_unknown_persons_stats(
    db: AsyncSession = Depends(get_db),
) -> UnknownPersonsStatsResponse:
    """Get statistics about unknown persons and unassigned faces.

    Returns:
        Statistics including counts of unassigned, clustered, noise faces,
        candidate groups, and dismissal information.
    """
    # Count total unassigned faces (no person_id)
    unassigned_result = await db.execute(
        select(func.count(FaceInstance.id)).where(FaceInstance.person_id.is_(None))
    )
    total_unassigned = unassigned_result.scalar_one()

    # Count clustered unassigned faces (has cluster_id, not noise, starts with 'unknown_')
    clustered_result = await db.execute(
        select(func.count(FaceInstance.id)).where(
            and_(
                FaceInstance.person_id.is_(None),
                FaceInstance.cluster_id.isnot(None),
                FaceInstance.cluster_id != "-1",
                FaceInstance.cluster_id.like("unknown_%"),
            )
        )
    )
    total_clustered = clustered_result.scalar_one()

    # Count noise faces (cluster_id = '-1')
    noise_result = await db.execute(
        select(func.count(FaceInstance.id)).where(
            and_(
                FaceInstance.person_id.is_(None),
                FaceInstance.cluster_id == "-1",
            )
        )
    )
    total_noise = noise_result.scalar_one()

    # Count unclustered faces (no cluster_id or non-unknown clusters)
    total_unclustered = total_unassigned - total_clustered - total_noise

    # Count candidate groups (distinct cluster_ids)
    groups_result = await db.execute(
        select(func.count(func.distinct(FaceInstance.cluster_id))).where(
            and_(
                FaceInstance.person_id.is_(None),
                FaceInstance.cluster_id.isnot(None),
                FaceInstance.cluster_id != "-1",
                FaceInstance.cluster_id.like("unknown_%"),
            )
        )
    )
    candidate_groups = groups_result.scalar_one()

    # Count dismissed groups
    dismissed_result = await db.execute(select(func.count(DismissedUnknownPersonGroup.id)))
    total_dismissed = dismissed_result.scalar_one()

    # Average group size
    if candidate_groups > 0:
        avg_group_size = total_clustered / candidate_groups
    else:
        avg_group_size = 0.0

    # Get last discovery timestamp from Redis
    last_discovery_at = None
    try:
        redis_conn = get_redis()
        last_discovery_ts = redis_conn.get("unknown_persons:last_discovery")
        if last_discovery_ts and isinstance(last_discovery_ts, bytes):
            # Worker stores JSON object with timestamp field
            parsed_data = json.loads(last_discovery_ts.decode("utf-8"))
            if isinstance(parsed_data, dict) and "timestamp" in parsed_data:
                last_discovery_at = datetime.fromisoformat(parsed_data["timestamp"])
            elif isinstance(parsed_data, str):
                # Backward compat: plain ISO string
                last_discovery_at = datetime.fromisoformat(parsed_data)
    except Exception as e:
        logger.warning(f"Failed to get last discovery timestamp from Redis: {e}")

    # Average group confidence (from Redis cache, fallback to 0.0)
    avg_group_confidence = 0.0
    if candidate_groups > 0:
        try:
            redis_conn = get_redis()
            # Try to get cached avg confidence
            avg_conf_cached = redis_conn.get("unknown_persons:avg_confidence")
            if avg_conf_cached and isinstance(avg_conf_cached, bytes):
                avg_group_confidence = float(avg_conf_cached.decode("utf-8"))
        except Exception as e:
            logger.warning(f"Failed to get avg confidence from Redis: {e}")

    return UnknownPersonsStatsResponse(
        total_unassigned_faces=total_unassigned,
        total_clustered_faces=total_clustered,
        total_noise_faces=total_noise,
        total_unclustered_faces=total_unclustered,
        candidate_groups=candidate_groups,
        avg_group_size=avg_group_size,
        avg_group_confidence=avg_group_confidence,
        total_dismissed_groups=total_dismissed,
        last_discovery_at=last_discovery_at,
    )


# ============ Endpoint 2: POST /discover ============


@router.post("/discover", response_model=DiscoverJobResponse)
async def discover_unknown_persons(
    request: DiscoverUnknownPersonsRequest,
) -> DiscoverJobResponse:
    """Trigger unknown person discovery via background clustering job.

    Enqueues a job that will:
    1. Fetch unassigned faces with embeddings
    2. Run clustering algorithm (HDBSCAN/DBSCAN/Agglomerative)
    3. Update cluster_ids with 'unknown_' prefix
    4. Cache cluster confidence scores in Redis

    Args:
        request: Discovery parameters (clustering method, thresholds, limits)

    Returns:
        Job ID and progress tracking key for monitoring
    """
    try:
        # Import job function
        from image_search_service.queue.face_jobs import discover_unknown_persons_job

        # Pre-generate job UUID BEFORE enqueuing (ensures progress_key consistency)
        job_uuid = str(uuid.uuid4())
        progress_key = f"job:{job_uuid}:progress"

        # Pre-seed Redis progress key so SSE subscription doesn't 404
        redis_conn = get_redis()
        initial_progress = {
            "phase": "queued",
            "current": 0,
            "total": 100,
            "message": "Job queued, waiting for worker...",
        }
        redis_conn.set(progress_key, json.dumps(initial_progress), ex=3600)  # 1 hour TTL

        # Get queue
        queue = get_queue("default")

        # Enqueue job with pre-generated job_id and progress_key
        queue.enqueue(
            discover_unknown_persons_job,
            clustering_method=request.clustering_method,
            min_cluster_size=request.min_cluster_size,
            min_quality=request.min_quality,
            max_faces=request.max_faces,
            min_cluster_confidence=request.min_cluster_confidence,
            eps=request.eps,
            progress_key=progress_key,
            job_id=job_uuid,
            job_timeout="30m",
        )

        logger.info(f"Enqueued unknown persons discovery job: {job_uuid}")

        return DiscoverJobResponse(
            job_id=job_uuid,
            status="queued",
            progress_key=progress_key,
            params=request.model_dump(by_alias=True),  # Return camelCase keys
        )

    except Exception as e:
        logger.error(f"Failed to enqueue discovery job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to enqueue job: {str(e)}")


# ============ Endpoint 3: GET /candidates ============


@router.get("/candidates", response_model=UnknownPersonCandidatesResponse)
async def get_unknown_person_candidates(
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    groups_per_page: int = Query(
        default=50, ge=1, le=200, description="Groups per page", alias="groupsPerPage"
    ),
    faces_per_group: int = Query(
        default=6, ge=1, le=50, description="Sample faces per group", alias="facesPerGroup"
    ),
    min_confidence: float | None = Query(
        default=None,
        ge=0.0,
        le=1.0,
        description="Override min confidence filter",
        alias="minConfidence",
    ),
    min_group_size: int | None = Query(
        default=None, ge=1, le=100, description="Override min size", alias="minGroupSize"
    ),
    sort_by: str = Query(
        default="face_count",
        pattern="^(face_count|confidence|quality)$",
        description="Sort field",
        alias="sortBy",
    ),
    sort_order: str = Query(
        default="desc", pattern="^(asc|desc)$", description="Sort order", alias="sortOrder"
    ),
    include_dismissed: bool = Query(
        default=False, description="Include dismissed groups", alias="includeDismissed"
    ),
    db: AsyncSession = Depends(get_db),
) -> UnknownPersonCandidatesResponse:
    """Get paginated list of unknown person candidate groups.

    Groups are clusters of unassigned faces that may represent distinct individuals.
    Supports filtering by confidence, size, and dismissal status.

    Args:
        page: Page number (1-indexed)
        groups_per_page: Number of groups per page
        faces_per_group: Number of sample faces to return per group
        min_confidence: Minimum cluster confidence (overrides admin default)
        min_group_size: Minimum faces per group (overrides admin default)
        sort_by: Sort by 'face_count', 'confidence', or 'quality'
        sort_order: 'asc' or 'desc'
        include_dismissed: Whether to include dismissed groups
        db: Database session

    Returns:
        Paginated candidate groups with metadata and sample faces
    """
    settings = get_settings()

    # Read discovery params from Redis to get the threshold used during discovery
    redis_conn = get_redis()
    discovery_params_raw = redis_conn.get("unknown_persons:discovery_params")
    discovery_min_confidence = 0.50  # safe fallback
    if discovery_params_raw and isinstance(discovery_params_raw, bytes):
        try:
            discovery_params = json.loads(discovery_params_raw.decode("utf-8"))
            discovery_min_confidence = float(discovery_params.get("min_cluster_confidence", 0.50))
        except Exception:
            pass

    # Effective thresholds (request params override admin defaults)
    effective_min_conf = (
        min_confidence if min_confidence is not None else settings.unknown_person_default_threshold
    )
    effective_min_size = (
        min_group_size
        if min_group_size is not None
        else settings.unknown_person_min_display_count
    )

    # Get dismissed hashes (used for filtering AND flagging)
    dismissed_hashes = await get_dismissed_hashes(db)

    # Query all candidate groups (cluster_id, face_count, avg_quality)
    group_query = (
        select(
            FaceInstance.cluster_id,
            func.count(FaceInstance.id).label("face_count"),
            func.avg(FaceInstance.quality_score).label("avg_quality"),
        )
        .where(
            and_(
                FaceInstance.person_id.is_(None),
                FaceInstance.cluster_id.isnot(None),
                FaceInstance.cluster_id != "-1",
                FaceInstance.cluster_id.like("unknown_%"),
            )
        )
        .group_by(FaceInstance.cluster_id)
    )

    group_result = await db.execute(group_query)
    raw_groups = group_result.all()

    # Build groups with confidence and dismissal filtering
    # Track filter transparency counters
    filtered_by_confidence = 0
    filtered_by_size = 0
    filtered_by_dismissed = 0
    total_before_filtering = len(raw_groups)

    candidate_groups_data: list[dict[str, Any]] = []

    for row in raw_groups:
        cluster_id = row.cluster_id
        face_count = row.face_count
        avg_quality = row.avg_quality or 0.0

        # Get confidence from Redis cache
        try:
            cached_conf = redis_conn.get(f"unknown_persons:cluster:{cluster_id}")
            if cached_conf and isinstance(cached_conf, bytes):
                # Worker stores JSON object with confidence field
                parsed_data = json.loads(cached_conf.decode("utf-8"))
                if isinstance(parsed_data, dict):
                    cluster_confidence = float(parsed_data.get("confidence", 0.0))
                else:
                    # Backward compat: plain float value
                    cluster_confidence = float(parsed_data)
            else:
                cluster_confidence = 0.0
        except Exception:
            cluster_confidence = 0.0

        # Apply size and confidence filters
        if face_count < effective_min_size:
            filtered_by_size += 1
            continue
        if cluster_confidence < effective_min_conf:
            filtered_by_confidence += 1
            continue

        # Check dismissal status BEFORE adding to candidates
        # Fetch ALL faces to compute membership hash
        all_faces_query = (
            select(FaceInstance.id)
            .where(
                and_(
                    FaceInstance.cluster_id == cluster_id,
                    FaceInstance.person_id.is_(None),
                )
            )
        )
        all_faces_result = await db.execute(all_faces_query)
        all_face_ids = [row_id[0] for row_id in all_faces_result.all()]

        # Compute membership hash from ALL faces (not just samples)
        membership_hash = compute_membership_hash(all_face_ids)

        # Check if dismissed
        is_dismissed_flag = membership_hash in dismissed_hashes

        # Skip if dismissed and not including dismissed
        if is_dismissed_flag and not include_dismissed:
            filtered_by_dismissed += 1
            continue

        candidate_groups_data.append(
            {
                "cluster_id": cluster_id,
                "face_count": face_count,
                "avg_quality": avg_quality,
                "cluster_confidence": cluster_confidence,
                "membership_hash": membership_hash,  # Cache for later use
                "is_dismissed": is_dismissed_flag,
            }
        )

    # Sort groups
    reverse = sort_order == "desc"
    if sort_by == "face_count":
        candidate_groups_data.sort(key=lambda g: g["face_count"], reverse=reverse)
    elif sort_by == "confidence":
        candidate_groups_data.sort(key=lambda g: g["cluster_confidence"], reverse=reverse)
    elif sort_by == "quality":
        candidate_groups_data.sort(key=lambda g: g["avg_quality"], reverse=reverse)

    # Paginate AFTER filtering
    total_groups_count = len(candidate_groups_data)
    start_idx = (page - 1) * groups_per_page
    end_idx = start_idx + groups_per_page
    paginated_groups = candidate_groups_data[start_idx:end_idx]

    # Fetch sample faces for each group
    groups: list[UnknownPersonCandidateGroup] = []
    for group_data in paginated_groups:
        cluster_id = group_data["cluster_id"]
        membership_hash = group_data["membership_hash"]
        is_dismissed_flag = group_data["is_dismissed"]

        # Now fetch top faces by quality for display
        faces_query = (
            select(FaceInstance)
            .where(
                and_(
                    FaceInstance.cluster_id == cluster_id,
                    FaceInstance.person_id.is_(None),
                )
            )
            .order_by(FaceInstance.quality_score.desc())
            .limit(faces_per_group + 1)  # +1 for representative face
        )
        faces_result = await db.execute(faces_query)
        faces = list(faces_result.scalars().all())

        if not faces:
            continue

        # Representative face (highest quality)
        representative_face = _build_face_response(faces[0])

        # Sample faces (next N faces)
        sample_faces = [_build_face_response(f) for f in faces[1 : faces_per_group + 1]]

        # Check dismissal timestamp
        dismissed_at = None
        if is_dismissed_flag:
            dismissed_record = await db.execute(
                select(DismissedUnknownPersonGroup.dismissed_at).where(
                    DismissedUnknownPersonGroup.membership_hash == membership_hash
                )
            )
            dismissed_at = dismissed_record.scalar_one_or_none()

        groups.append(
            UnknownPersonCandidateGroup(
                group_id=cluster_id,
                membership_hash=membership_hash,
                face_count=group_data["face_count"],
                cluster_confidence=group_data["cluster_confidence"],
                avg_quality=group_data["avg_quality"],
                representative_face=representative_face,
                sample_faces=sample_faces,
                is_dismissed=is_dismissed_flag,
                dismissed_at=dismissed_at,
            )
        )

    # Global stats
    unassigned_result = await db.execute(
        select(func.count(FaceInstance.id)).where(FaceInstance.person_id.is_(None))
    )
    total_unassigned = unassigned_result.scalar_one()

    noise_result = await db.execute(
        select(func.count(FaceInstance.id)).where(
            and_(
                FaceInstance.person_id.is_(None),
                FaceInstance.cluster_id == "-1",
            )
        )
    )
    total_noise = noise_result.scalar_one()

    dismissed_count_result = await db.execute(select(func.count(DismissedUnknownPersonGroup.id)))
    total_dismissed = dismissed_count_result.scalar_one()

    # Last discovery timestamp
    last_discovery_at = None
    try:
        last_discovery_ts = redis_conn.get("unknown_persons:last_discovery")
        if last_discovery_ts and isinstance(last_discovery_ts, bytes):
            # Worker stores JSON object with timestamp field
            parsed_data = json.loads(last_discovery_ts.decode("utf-8"))
            if isinstance(parsed_data, dict) and "timestamp" in parsed_data:
                last_discovery_at = datetime.fromisoformat(parsed_data["timestamp"])
            elif isinstance(parsed_data, str):
                # Backward compat: plain ISO string
                last_discovery_at = datetime.fromisoformat(parsed_data)
    except Exception as e:
        logger.warning(f"Failed to get last discovery timestamp: {e}")

    return UnknownPersonCandidatesResponse(
        groups=groups,
        total_groups=total_groups_count,
        total_unassigned_faces=total_unassigned,
        total_noise_faces=total_noise,
        total_dismissed_groups=total_dismissed,
        page=page,
        groups_per_page=groups_per_page,
        faces_per_group=faces_per_group,
        last_discovery_at=last_discovery_at,
        min_group_size_setting=effective_min_size,
        min_confidence_setting=effective_min_conf,
        discovery_min_confidence=discovery_min_confidence,
        filtered_by_confidence=filtered_by_confidence,
        filtered_by_size=filtered_by_size,
        filtered_by_dismissed=filtered_by_dismissed,
        total_before_filtering=total_before_filtering,
    )


# ============ Endpoint 4: GET /candidates/merge-suggestions ============
# NOTE: This MUST come BEFORE /candidates/{group_id} to avoid route conflicts


@router.get("/candidates/merge-suggestions", response_model=MergeSuggestionsResponse)
async def get_merge_suggestions(
    max_suggestions: int = Query(10, ge=1, le=50, description="Max merge suggestions to return"),
    min_similarity: float = Query(
        0.60, ge=0.0, le=1.0, description="Min centroid similarity threshold"
    ),
    db: AsyncSession = Depends(get_db),
) -> MergeSuggestionsResponse:
    """Get suggestions for merging similar candidate groups.

    Computes pairwise centroid similarity between all active candidate groups
    and returns pairs with similarity above threshold.

    Algorithm:
    1. Query all active (non-dismissed) candidate groups
    2. For each group, compute centroid (average embedding) from Qdrant
    3. Compute pairwise cosine similarities between all centroids
    4. Return top N pairs above similarity threshold, sorted by similarity desc

    Args:
        max_suggestions: Maximum number of suggestions to return
        min_similarity: Minimum centroid similarity to suggest merge
        db: Database session

    Returns:
        List of merge suggestions with similarity scores
    """
    # Get all active candidate groups (cluster_id, face_count)
    groups_query = (
        select(
            FaceInstance.cluster_id,
            func.count(FaceInstance.id).label("face_count"),
        )
        .where(
            and_(
                FaceInstance.person_id.is_(None),
                FaceInstance.cluster_id.isnot(None),
                FaceInstance.cluster_id != "-1",
                FaceInstance.cluster_id.like("unknown_%"),
            )
        )
        .group_by(FaceInstance.cluster_id)
    )
    groups_result = await db.execute(groups_query)
    groups = groups_result.all()

    if len(groups) < 2:
        return MergeSuggestionsResponse(suggestions=[], total_groups_compared=len(groups))

    # Initialize Qdrant client
    qdrant_client = get_face_qdrant_client()

    # Compute centroid for each group
    import numpy as np
    from numpy.typing import NDArray

    group_centroids: dict[str, tuple[NDArray[np.float64], int]] = {}  # cluster_id -> (centroid, face_count)  # noqa: E501

    for row in groups:
        cluster_id = row.cluster_id
        face_count = row.face_count

        # Fetch Qdrant point IDs for this cluster
        point_ids_query = select(FaceInstance.qdrant_point_id).where(
            and_(
                FaceInstance.cluster_id == cluster_id,
                FaceInstance.person_id.is_(None),
            )
        )
        point_ids_result = await db.execute(point_ids_query)
        point_ids = [r[0] for r in point_ids_result.all()]

        if not point_ids:
            logger.warning(f"No Qdrant points found for cluster {cluster_id}")
            continue

        # Retrieve embeddings from Qdrant
        embeddings = []
        for point_id in point_ids:
            try:
                embedding = qdrant_client.get_embedding_by_point_id(point_id)
                if embedding is not None:
                    embeddings.append(np.array(embedding))
            except Exception as e:
                logger.warning(f"Failed to retrieve embedding for point {point_id}: {e}")

        if not embeddings:
            logger.warning(f"No embeddings retrieved for cluster {cluster_id}")
            continue

        # Compute centroid (mean of all embeddings)
        centroid = np.mean(embeddings, axis=0)
        group_centroids[cluster_id] = (centroid, face_count)

    # Compute pairwise similarities
    from sklearn.metrics.pairwise import cosine_similarity

    cluster_ids = list(group_centroids.keys())
    suggestions: list[MergeSuggestion] = []

    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            cluster_a = cluster_ids[i]
            cluster_b = cluster_ids[j]

            centroid_a, count_a = group_centroids[cluster_a]
            centroid_b, count_b = group_centroids[cluster_b]

            # Compute cosine similarity
            # Clamp to [0.0, 1.0] to handle floating-point precision issues
            similarity = float(
                cosine_similarity(centroid_a.reshape(1, -1), centroid_b.reshape(1, -1))[0][0]
            )
            similarity = max(0.0, min(1.0, similarity))

            if similarity >= min_similarity:
                suggestions.append(
                    MergeSuggestion(
                        group_a_id=cluster_a,
                        group_b_id=cluster_b,
                        similarity=similarity,
                        group_a_face_count=count_a,
                        group_b_face_count=count_b,
                    )
                )

    # Sort by similarity descending
    suggestions.sort(key=lambda s: s.similarity, reverse=True)

    # Limit to max_suggestions
    suggestions = suggestions[:max_suggestions]

    logger.info(
        f"Found {len(suggestions)} merge suggestions from {len(group_centroids)} groups "
        f"(threshold: {min_similarity})"
    )

    return MergeSuggestionsResponse(
        suggestions=suggestions,
        total_groups_compared=len(group_centroids),
    )


# ============ Endpoint 5: POST /candidates/merge ============
# NOTE: This MUST come BEFORE /candidates/{group_id} to avoid route conflicts


@router.post("/candidates/merge", response_model=MergeGroupsResponse)
async def merge_candidate_groups(
    request: MergeGroupsRequest,
    db: AsyncSession = Depends(get_db),
) -> MergeGroupsResponse:
    """Merge two candidate groups by moving all faces from group_b to group_a.

    Workflow:
    1. Validate both groups exist and have unassigned faces
    2. Get cluster_ids for both groups (groups are identified by cluster_id)
    3. Move all faces from group_b's cluster to group_a's cluster
    4. Return merged group metadata

    Args:
        request: Group IDs to merge (group_a is target, group_b is source)
        db: Database session

    Returns:
        Merged group metadata (merged_group_id, total_faces, faces_moved)

    Raises:
        HTTPException: 404 if either group not found, 400 if groups are the same
    """
    group_a_id = request.group_a_id
    group_b_id = request.group_b_id

    # Validate groups are different
    if group_a_id == group_b_id:
        raise HTTPException(status_code=400, detail="Cannot merge a group with itself")

    # Verify group_a exists and has faces
    group_a_query = (
        select(func.count(FaceInstance.id))
        .where(
            and_(
                FaceInstance.cluster_id == group_a_id,
                FaceInstance.person_id.is_(None),
            )
        )
    )
    group_a_result = await db.execute(group_a_query)
    group_a_count = group_a_result.scalar_one()

    if group_a_count == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Group {group_a_id} not found or has no unassigned faces"
        )

    # Verify group_b exists and has faces
    group_b_query = (
        select(func.count(FaceInstance.id))
        .where(
            and_(
                FaceInstance.cluster_id == group_b_id,
                FaceInstance.person_id.is_(None),
            )
        )
    )
    group_b_result = await db.execute(group_b_query)
    group_b_count = group_b_result.scalar_one()

    if group_b_count == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Group {group_b_id} not found or has no unassigned faces"
        )

    # Move all faces from group_b to group_a
    await db.execute(
        update(FaceInstance)
        .where(
            and_(
                FaceInstance.cluster_id == group_b_id,
                FaceInstance.person_id.is_(None),
            )
        )
        .values(cluster_id=group_a_id)
    )

    await db.commit()

    total_faces = group_a_count + group_b_count

    logger.info(
        f"Merged groups: {group_b_id} ({group_b_count} faces) -> {group_a_id} "
        f"(total: {total_faces} faces)"
    )

    return MergeGroupsResponse(
        merged_group_id=group_a_id,
        total_faces=total_faces,
        faces_moved=group_b_count,
    )


# ============ Endpoint 6: GET /candidates/{group_id} ============


@router.get("/candidates/{group_id}", response_model=UnknownPersonCandidateDetail)
async def get_unknown_person_candidate_detail(
    group_id: str,
    db: AsyncSession = Depends(get_db),
) -> UnknownPersonCandidateDetail:
    """Get detailed view of a single unknown person group with all faces.

    Args:
        group_id: Cluster ID (e.g., 'unknown_1')
        db: Database session

    Returns:
        Detailed group information with all faces

    Raises:
        HTTPException: 404 if group not found or has no unassigned faces
    """
    # Fetch all faces in group
    faces_query = (
        select(FaceInstance)
        .where(
            and_(
                FaceInstance.cluster_id == group_id,
                FaceInstance.person_id.is_(None),
            )
        )
        .order_by(FaceInstance.quality_score.desc())
    )
    faces_result = await db.execute(faces_query)
    faces = list(faces_result.scalars().all())

    if not faces:
        raise HTTPException(status_code=404, detail=f"Group {group_id} not found or has no faces")

    # Compute stats
    face_count = len(faces)
    avg_quality = sum(f.quality_score or 0 for f in faces) / face_count

    # Get confidence from Redis cache
    redis_conn = get_redis()
    cluster_confidence = 0.0
    try:
        cached_conf = redis_conn.get(f"unknown_persons:cluster:{group_id}")
        if cached_conf and isinstance(cached_conf, bytes):
            # Worker stores JSON object with confidence field
            parsed_data = json.loads(cached_conf.decode("utf-8"))
            if isinstance(parsed_data, dict):
                cluster_confidence = float(parsed_data.get("confidence", 0.0))
            else:
                # Backward compat: plain float value
                cluster_confidence = float(parsed_data)
    except Exception as e:
        logger.warning(f"Failed to get cluster confidence for {group_id}: {e}")

    # Compute membership hash
    face_ids = [f.id for f in faces]
    membership_hash = compute_membership_hash(face_ids)

    # Build face responses
    face_responses = [_build_face_response(f) for f in faces]

    return UnknownPersonCandidateDetail(
        group_id=group_id,
        membership_hash=membership_hash,
        face_count=face_count,
        cluster_confidence=cluster_confidence,
        avg_quality=avg_quality,
        faces=face_responses,
    )


# ============ Endpoint 7: POST /candidates/{group_id}/accept ============


@router.post("/candidates/{group_id}/accept", response_model=AcceptUnknownPersonResponse)
async def accept_unknown_person_candidate(
    group_id: str,
    request: AcceptUnknownPersonRequest,
    db: AsyncSession = Depends(get_db),
) -> AcceptUnknownPersonResponse:
    """Accept an unknown person group and create a labeled person.

    Workflow:
    1. Query all faces in group (person_id IS NULL, cluster_id = group_id)
    2. Validate at least one face exists
    3. Apply exclusions (partial acceptance support)
    4. Delegate to ClusterLabelingService.label_cluster_as_person()
    5. ALWAYS trigger find-more for propagation
    6. Optionally trigger re-clustering to update remaining unknowns

    Args:
        group_id: Cluster ID to accept
        request: Person name and optional exclusions
        db: Database session

    Returns:
        Person creation details and job IDs

    Raises:
        HTTPException: 404 if no faces found, 409 if all already assigned
    """
    # Fetch all faces in group
    faces_query = select(FaceInstance).where(
        and_(
            FaceInstance.cluster_id == group_id,
            FaceInstance.person_id.is_(None),
        )
    )
    faces_result = await db.execute(faces_query)
    faces = list(faces_result.scalars().all())

    if not faces:
        raise HTTPException(
            status_code=404, detail=f"No unassigned faces found for group {group_id}"
        )

    # Check if all faces are already assigned (should not happen but defensive)
    assigned_count = sum(1 for f in faces if f.person_id is not None)
    if assigned_count == len(faces):
        raise HTTPException(
            status_code=409,
            detail=f"All faces in group {group_id} are already assigned to persons",
        )

    # Extract face IDs
    face_ids = [f.id for f in faces]

    # Initialize ClusterLabelingService
    qdrant_client = get_face_qdrant_client()
    labeling_service = ClusterLabelingService(db=db, qdrant=qdrant_client)

    # Label cluster as person (handles exclusions, prototypes, find-more)
    try:
        result = await labeling_service.label_cluster_as_person(
            face_ids=face_ids,
            person_name=request.name,
            exclude_face_ids=request.face_ids_to_exclude or [],
            trigger_find_more=True,  # ALWAYS trigger find-more
            trigger_reclustering=False,  # We'll handle this separately
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Trigger re-clustering if requested
    reclustering_job_id = None
    if request.trigger_reclustering:
        try:
            from image_search_service.queue.face_jobs import discover_unknown_persons_job

            queue = get_queue("default")
            settings = get_settings()

            job = queue.enqueue(
                discover_unknown_persons_job,
                clustering_method="hdbscan",
                min_cluster_size=settings.unknown_person_min_display_count,
                min_quality=0.3,
                max_faces=settings.unknown_person_max_faces,
                min_cluster_confidence=settings.unknown_person_default_threshold,
                eps=0.5,
                job_timeout="30m",
            )
            reclustering_job_id = str(job.id) if job.id is not None else None
            logger.info(f"Triggered re-clustering job: {reclustering_job_id}")

        except Exception as e:
            logger.warning(f"Failed to trigger re-clustering: {e}", exc_info=True)

    return AcceptUnknownPersonResponse(
        person_id=result["person_id"],
        person_name=result["person_name"],
        faces_assigned=result["faces_assigned"],
        faces_excluded=result["faces_excluded"],
        prototypes_created=result["prototypes_created"],
        find_more_job_id=result.get("find_more_job_id"),
        reclustering_job_id=reclustering_job_id,
    )


# ============ Endpoint 8: POST /candidates/{group_id}/dismiss ============


@router.post("/candidates/{group_id}/dismiss", response_model=DismissUnknownPersonResponse)
async def dismiss_unknown_person_candidate(
    group_id: str,
    request: DismissUnknownPersonRequest,
    db: AsyncSession = Depends(get_db),
) -> DismissUnknownPersonResponse:
    """Dismiss an unknown person group (mark as not a real person).

    Records dismissal by membership hash so the group stays dismissed
    even if cluster_id changes across re-clustering runs.

    Optionally marks faces as noise (cluster_id = '-1') to exclude from
    future clustering.

    Args:
        group_id: Cluster ID to dismiss
        request: Dismissal reason and noise flag
        db: Database session

    Returns:
        Dismissal confirmation with affected face count

    Raises:
        HTTPException: 404 if group not found
    """
    # Fetch all faces in group
    faces_query = select(FaceInstance).where(
        and_(
            FaceInstance.cluster_id == group_id,
            FaceInstance.person_id.is_(None),
        )
    )
    faces_result = await db.execute(faces_query)
    faces = list(faces_result.scalars().all())

    if not faces:
        raise HTTPException(status_code=404, detail=f"Group {group_id} not found")

    face_ids = [f.id for f in faces]
    face_count = len(faces)

    # Compute membership hash
    membership_hash = compute_membership_hash(face_ids)

    # Check if already dismissed (idempotent operation)
    existing_dismissal = await db.execute(
        select(DismissedUnknownPersonGroup).where(
            DismissedUnknownPersonGroup.membership_hash == membership_hash
        )
    )
    already_dismissed = existing_dismissal.scalar_one_or_none()

    if not already_dismissed:
        # Record dismissal only if not already dismissed
        await dismiss_group(
            db=db,
            membership_hash=membership_hash,
            cluster_id=group_id,
            face_count=face_count,
            face_instance_ids=face_ids,
            reason=request.reason,
            marked_as_noise=request.mark_as_noise,
        )

    # Mark faces as noise if requested (idempotent update)
    if request.mark_as_noise:
        await db.execute(
            update(FaceInstance)
            .where(FaceInstance.id.in_(face_ids))
            .values(cluster_id="-1")
        )

    await db.commit()

    logger.info(
        f"Dismissed group {group_id} with {face_count} faces "
        f"(marked_as_noise={request.mark_as_noise})"
    )

    return DismissUnknownPersonResponse(
        group_id=group_id,
        membership_hash=membership_hash,
        faces_affected=face_count,
        marked_as_noise=request.mark_as_noise,
    )


# ============ Helper Functions ============


def _build_face_response(face: FaceInstance) -> FaceInGroupResponse:
    """Build FaceInGroupResponse from FaceInstance model.

    Args:
        face: FaceInstance database model

    Returns:
        FaceInGroupResponse schema for API
    """
    # Handle both FaceInstance.asset_id (int) and face.asset.id (int)
    # asset_id is the foreign key field
    asset_id = face.asset_id if hasattr(face, "asset_id") else face.asset.id

    # Generate thumbnail URL using the same pattern as other components
    thumbnail_url = f"/api/v1/images/{asset_id}/thumbnail" if asset_id else None

    return FaceInGroupResponse(
        face_instance_id=face.id,
        asset_id=asset_id,
        quality_score=face.quality_score or 0.0,
        detection_confidence=face.detection_confidence,
        bbox_x=face.bbox_x,
        bbox_y=face.bbox_y,
        bbox_w=face.bbox_w,
        bbox_h=face.bbox_h,
        thumbnail_url=thumbnail_url,
    )


