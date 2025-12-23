"""RQ background jobs for face detection, clustering, and assignment."""

import logging
import uuid
from datetime import datetime
from typing import Optional

from rq import get_current_job

from image_search_service.db.models import FaceInstance, ImageAsset
from image_search_service.db.sync_operations import get_sync_session

logger = logging.getLogger(__name__)


def detect_faces_job(
    asset_ids: list[str],
    min_confidence: float = 0.5,
    min_face_size: int = 20,
) -> dict:
    """RQ job to detect and embed faces for a batch of assets.

    Args:
        asset_ids: List of asset UUID strings to process
        min_confidence: Minimum detection confidence
        min_face_size: Minimum face size in pixels

    Returns:
        Summary dict with processing statistics
    """
    job = get_current_job()
    job_id = job.id if job else "no-job"

    # Convert string UUIDs to UUID objects
    uuids = [uuid.UUID(aid) for aid in asset_ids]

    logger.info(f"[{job_id}] Starting face detection for {len(uuids)} assets")

    from image_search_service.faces.service import get_face_service

    db_session = get_sync_session()
    try:
        service = get_face_service(db_session)
        result = service.process_assets_batch(
            asset_ids=uuids,
            min_confidence=min_confidence,
            min_face_size=min_face_size,
        )

        logger.info(
            f"[{job_id}] Face detection complete: "
            f"{result['processed']} assets, {result['total_faces']} faces, "
            f"{result['errors']} errors"
        )

        return result
    finally:
        db_session.close()


def cluster_faces_job(
    quality_threshold: float = 0.5,
    max_faces: int = 50000,
    min_cluster_size: int = 5,
    min_samples: int = 3,
    time_bucket: Optional[str] = None,
) -> dict:
    """RQ job to cluster unlabeled faces using HDBSCAN.

    Args:
        quality_threshold: Minimum quality score for faces
        max_faces: Maximum faces to cluster at once
        min_cluster_size: HDBSCAN min_cluster_size
        min_samples: HDBSCAN min_samples
        time_bucket: Optional YYYY-MM filter

    Returns:
        Clustering statistics
    """
    job = get_current_job()
    job_id = job.id if job else "no-job"

    logger.info(f"[{job_id}] Starting face clustering (max={max_faces})")

    from image_search_service.faces.clusterer import get_face_clusterer

    db_session = get_sync_session()
    try:
        clusterer = get_face_clusterer(
            db_session=db_session,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        )
        result = clusterer.cluster_unlabeled_faces(
            quality_threshold=quality_threshold,
            max_faces=max_faces,
            time_bucket=time_bucket,
        )

        logger.info(
            f"[{job_id}] Clustering complete: "
            f"{result['clusters_found']} clusters, {result['noise_count']} noise"
        )

        return result
    finally:
        db_session.close()


def assign_faces_job(
    since: Optional[str] = None,
    max_faces: int = 1000,
    similarity_threshold: float = 0.6,
) -> dict:
    """RQ job to assign new faces to known persons via prototype matching.

    Args:
        since: Only process faces created after this ISO datetime string
        max_faces: Maximum faces to process
        similarity_threshold: Minimum similarity for auto-assignment

    Returns:
        Assignment statistics
    """
    job = get_current_job()
    job_id = job.id if job else "no-job"

    # Convert ISO string to datetime
    since_dt = None
    if since:
        since_dt = datetime.fromisoformat(since)

    logger.info(
        f"[{job_id}] Starting face assignment "
        f"(max={max_faces}, threshold={similarity_threshold})"
    )

    from image_search_service.faces.assigner import get_face_assigner

    db_session = get_sync_session()
    try:
        assigner = get_face_assigner(
            db_session=db_session,
            similarity_threshold=similarity_threshold,
        )
        result = assigner.assign_new_faces(
            since=since_dt,
            max_faces=max_faces,
        )

        logger.info(
            f"[{job_id}] Assignment complete: "
            f"{result['assigned']} assigned, {result['unassigned']} unassigned"
        )

        return result
    finally:
        db_session.close()


def compute_centroids_job() -> dict:
    """RQ job to compute/update person centroids.

    Returns:
        Centroid computation statistics
    """
    job = get_current_job()
    job_id = job.id if job else "no-job"

    logger.info(f"[{job_id}] Starting centroid computation")

    from image_search_service.faces.assigner import get_face_assigner

    db_session = get_sync_session()
    try:
        assigner = get_face_assigner(db_session=db_session)
        result = assigner.compute_person_centroids()

        logger.info(
            f"[{job_id}] Centroid computation complete: "
            f"{result['centroids_computed']} centroids"
        )

        return result
    finally:
        db_session.close()


def backfill_faces_job(
    limit: int = 1000,
    offset: int = 0,
    min_confidence: float = 0.5,
) -> dict:
    """RQ job to backfill face detection for existing assets without faces.

    Args:
        limit: Number of assets to process
        offset: Starting offset for pagination
        min_confidence: Detection confidence threshold

    Returns:
        Backfill statistics
    """
    job = get_current_job()
    job_id = job.id if job else "no-job"

    logger.info(f"[{job_id}] Starting face backfill (limit={limit}, offset={offset})")

    from sqlalchemy import select

    from image_search_service.faces.service import get_face_service

    db_session = get_sync_session()
    try:
        # Get assets without any face detections
        subquery = select(FaceInstance.asset_id).distinct()
        query = (
            select(ImageAsset)
            .where(~ImageAsset.id.in_(subquery))
            .offset(offset)
            .limit(limit)
        )
        assets = db_session.execute(query).scalars().all()

        if not assets:
            return {"processed": 0, "total_faces": 0, "status": "no_assets_to_process"}

        service = get_face_service(db_session)
        result = service.process_assets_batch(
            asset_ids=[a.id for a in assets],
            min_confidence=min_confidence,
        )

        logger.info(
            f"[{job_id}] Backfill complete: "
            f"{result['processed']} assets, {result['total_faces']} faces"
        )

        return result
    finally:
        db_session.close()
