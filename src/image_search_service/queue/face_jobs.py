"""RQ background jobs for face detection, clustering, and assignment."""

import logging
from datetime import datetime
from typing import Any

from rq import get_current_job

from image_search_service.db.models import FaceInstance, ImageAsset
from image_search_service.db.sync_operations import get_sync_session

logger = logging.getLogger(__name__)


def detect_faces_job(
    asset_ids: list[str],
    min_confidence: float = 0.5,
    min_face_size: int = 20,
) -> dict[str, Any]:
    """RQ job to detect and embed faces for a batch of assets.

    Args:
        asset_ids: List of asset ID strings to process
        min_confidence: Minimum detection confidence
        min_face_size: Minimum face size in pixels

    Returns:
        Summary dict with processing statistics
    """
    job = get_current_job()
    job_id = job.id if job else "no-job"

    # Convert string asset IDs to integers
    asset_id_ints = [int(aid) for aid in asset_ids]

    logger.info(f"[{job_id}] Starting face detection for {len(asset_id_ints)} assets")

    from image_search_service.faces.service import get_face_service

    db_session = get_sync_session()
    try:
        service = get_face_service(db_session)
        result = service.process_assets_batch(
            asset_ids=asset_id_ints,
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
    time_bucket: str | None = None,
) -> dict[str, Any]:
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
    since: str | None = None,
    max_faces: int = 1000,
    similarity_threshold: float = 0.6,
) -> dict[str, Any]:
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


def compute_centroids_job() -> dict[str, Any]:
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
    batch_size: int = 8,
) -> dict[str, Any]:
    """RQ job to backfill face detection for existing assets without faces.

    Args:
        limit: Number of assets to process
        offset: Starting offset for pagination
        min_confidence: Detection confidence threshold
        batch_size: Number of images to pre-load in parallel

    Returns:
        Backfill statistics
    """
    job = get_current_job()
    job_id = job.id if job else "no-job"

    logger.info(
        f"[{job_id}] Starting face backfill "
        f"(limit={limit}, offset={offset}, batch_size={batch_size})"
    )

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
            batch_size=batch_size,
        )

        logger.info(
            f"[{job_id}] Backfill complete: "
            f"{result['processed']} assets, {result['total_faces']} faces, "
            f"throughput={result.get('throughput', 0):.2f} img/s"
        )

        return result
    finally:
        db_session.close()


# ============ Dual-Mode Clustering Jobs ============


def cluster_dual_job(
    person_threshold: float = 0.7,
    unknown_method: str = "hdbscan",
    unknown_min_size: int = 3,
    unknown_eps: float = 0.5,
    max_faces: int | None = None,
) -> dict[str, Any]:
    """RQ job for dual-mode face clustering (supervised + unsupervised).

    Args:
        person_threshold: Minimum similarity for assignment to person (0-1)
        unknown_method: Clustering method for unknown faces (hdbscan, dbscan, agglomerative)
        unknown_min_size: Minimum cluster size for unknown faces
        unknown_eps: Distance threshold for DBSCAN/Agglomerative
        max_faces: Optional limit on number of faces to process

    Returns:
        Result dict with clustering statistics
    """
    job = get_current_job()
    job_id = job.id if job else "no-job"

    logger.info(
        f"[{job_id}] Starting dual-mode clustering: "
        f"threshold={person_threshold}, method={unknown_method}"
    )

    from image_search_service.faces.dual_clusterer import get_dual_mode_clusterer

    db_session = get_sync_session()
    try:
        clusterer = get_dual_mode_clusterer(
            db_session=db_session,
            person_match_threshold=person_threshold,
            unknown_min_cluster_size=unknown_min_size,
            unknown_method=unknown_method,
            unknown_eps=unknown_eps,
        )
        result = clusterer.cluster_all_faces(max_faces=max_faces)

        logger.info(
            f"[{job_id}] Dual-mode clustering complete: "
            f"{result.get('total_processed', 0)} faces processed, "
            f"{result.get('assigned_to_people', 0)} assigned, "
            f"{result.get('unknown_clusters', 0)} unknown clusters"
        )

        return result
    finally:
        db_session.close()


def train_person_matching_job(
    epochs: int = 20,
    margin: float = 0.2,
    batch_size: int = 32,
    learning_rate: float = 0.0001,
    min_faces_per_person: int = 5,
    checkpoint_path: str | None = None,
) -> dict[str, Any]:
    """RQ job for training face matching model using triplet loss.

    Args:
        epochs: Number of training epochs
        margin: Triplet loss margin
        batch_size: Batch size for training
        learning_rate: Learning rate
        min_faces_per_person: Minimum faces required per person for training
        checkpoint_path: Optional path to save checkpoint

    Returns:
        Result dict with training statistics
    """
    job = get_current_job()
    job_id = job.id if job else "no-job"

    logger.info(
        f"[{job_id}] Starting face matching training: "
        f"epochs={epochs}, margin={margin}, batch_size={batch_size}"
    )

    from image_search_service.faces.trainer import get_face_trainer

    db_session = get_sync_session()
    try:
        trainer = get_face_trainer(
            db_session=db_session,
            margin=margin,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        result = trainer.fine_tune_for_person_clustering(
            min_faces_per_person=min_faces_per_person,
            checkpoint_path=checkpoint_path,
        )

        logger.info(
            f"[{job_id}] Training complete: "
            f"{result.get('persons_used', 0)} persons, "
            f"{result.get('total_triplets', 0)} triplets, "
            f"final_loss={result.get('final_loss', 0):.4f}"
        )

        return result
    finally:
        db_session.close()


def recluster_after_training_job(
    person_threshold: float = 0.7,
    unknown_method: str = "hdbscan",
) -> dict[str, Any]:
    """RQ job to recluster faces after training.

    Runs clustering with potentially improved embeddings.

    Args:
        person_threshold: Minimum similarity for assignment to person
        unknown_method: Clustering method for unknown faces

    Returns:
        Result dict with clustering statistics
    """
    job = get_current_job()
    job_id = job.id if job else "no-job"

    logger.info(f"[{job_id}] Starting post-training reclustering")

    return cluster_dual_job(
        person_threshold=person_threshold,
        unknown_method=unknown_method,
    )
