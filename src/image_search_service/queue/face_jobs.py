"""RQ background jobs for face detection, clustering, and assignment.

FORK-SAFETY REQUIREMENTS FOR MACOS
==================================
All job functions in this module are executed by RQ work-horse subprocesses.
On macOS, these subprocesses are created via spawn() (not fork()), requiring:

1. SYNCHRONOUS database operations ONLY
   ✓ Use: get_sync_session() for all database access
   ✗ Don't use: async/await, asyncio, greenlets

2. No fork-unsafe libraries:
   ✓ All jobs use sync_operations (PostgreSQL sync layer)
   ✗ GSS authentication disabled on macOS (fork-unsafe)
   ✗ No gRPC async operations

3. Verified job functions:
   ✓ cluster_faces_job() - Sync operations
   ✓ assign_faces_job() - Sync operations
   ✓ compute_centroids_job() - Sync operations
   ✓ detect_faces_for_session_job() - Sync operations
   ✓ propagate_person_label_job() - Sync operations
   ✓ expire_old_suggestions_job() - Sync operations
   ✓ discover_unknown_persons_job() - Sync operations

See worker.py for complete macOS fork-safety architecture.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
from rq import get_current_job
from sqlalchemy import select

from image_search_service.db.models import (
    FaceInstance,
    FaceSuggestion,
    FaceSuggestionStatus,
    ImageAsset,
)
from image_search_service.db.sync_operations import get_sync_session

logger = logging.getLogger(__name__)


def cluster_faces_job(
    quality_threshold: float = 0.5,
    max_faces: int = 50000,
    min_cluster_size: int = 5,
    min_samples: int = 3,
    time_bucket: str | None = None,
) -> dict[str, Any]:
    """RQ job to cluster unlabeled faces using HDBSCAN.

    FORK-SAFETY (macOS): Disable proxy detection in work-horse subprocess.

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
    from image_search_service.vector.face_qdrant import get_face_qdrant_client

    db_session = get_sync_session()
    try:
        qdrant_client = get_face_qdrant_client()
        clusterer = get_face_clusterer(
            db_session=db_session,
            qdrant_client=qdrant_client,
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

    FORK-SAFETY (macOS): Disable proxy detection in work-horse subprocess.

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
        f"[{job_id}] Starting face assignment (max={max_faces}, threshold={similarity_threshold})"
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
            f"[{job_id}] Centroid computation complete: {result['centroids_computed']} centroids"
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
        query = select(ImageAsset).where(~ImageAsset.id.in_(subquery)).offset(offset).limit(limit)
        assets = db_session.execute(query).scalars().all()

        if not assets:
            return {"processed": 0, "total_faces": 0, "status": "no_assets_to_process"}

        service = get_face_service(db_session)
        result = service.process_assets_batch(
            asset_ids=[a.id for a in assets],
            min_confidence=min_confidence,
            prefetch_batch_size=batch_size,
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


def detect_faces_for_session_job(
    session_id: str,
) -> dict[str, Any]:
    """Process face detection for an entire session.

    This job:
    1. Updates session status to PROCESSING
    2. Gets assets to process (either from linked training session or all unprocessed)
    3. Processes images in batches using the existing FaceProcessingService
    4. Updates progress after each batch
    5. Auto-assigns faces to known persons using the existing assigner
    6. Marks session as COMPLETED or FAILED

    FORK-SAFETY (macOS): Disable proxy detection in work-horse subprocess.

    Args:
        session_id: UUID string of the FaceDetectionSession

    Returns:
        dict with session results
    """
    import uuid as uuid_lib

    from sqlalchemy import select

    from image_search_service.db.models import (
        FaceDetectionSession,
        FaceDetectionSessionStatus,
        FaceInstance,
        TrainingEvidence,
    )
    from image_search_service.faces.assigner import get_face_assigner
    from image_search_service.faces.service import get_face_service

    job = get_current_job()
    job_id = job.id if job else "no-job"

    session_uuid = uuid_lib.UUID(session_id)

    logger.info(f"[{job_id}] Starting face detection for session {session_id}")

    db_session = get_sync_session()
    try:
        # Get session and update status to PROCESSING
        session = db_session.get(FaceDetectionSession, session_uuid)
        if not session:
            logger.error(f"[{job_id}] Session {session_id} not found")
            return {
                "session_id": session_id,
                "status": "failed",
                "error": "Session not found",
            }

        # Check if this is first run or resume
        is_first_run = session.asset_ids_json is None

        if is_first_run:
            # First run: query assets and store IDs
            logger.info(f"[{job_id}] First run: querying assets to process")

            # Get asset IDs that already have faces
            processed_subquery = select(FaceInstance.asset_id).distinct()

            # Get assets without faces
            query = select(ImageAsset).where(~ImageAsset.id.in_(processed_subquery))

            # If linked to training session, filter further
            if session.training_session_id:
                logger.info(
                    f"[{job_id}] Filtering to training session {session.training_session_id}"
                )
                # Get assets from this training session via TrainingEvidence
                training_assets = select(TrainingEvidence.asset_id).where(
                    TrainingEvidence.session_id == session.training_session_id
                )
                query = query.where(ImageAsset.id.in_(training_assets))

            assets = db_session.execute(query).scalars().all()

            # Store asset IDs as JSON string
            asset_ids = [a.id for a in assets]
            session.asset_ids_json = json.dumps(asset_ids)
            session.current_asset_index = 0
            session.status = FaceDetectionSessionStatus.PROCESSING.value
            session.started_at = datetime.now()
            db_session.commit()

            logger.info(f"[{job_id}] Stored {len(asset_ids)} asset IDs for processing")
        else:
            # Resume: load stored asset IDs and continue from current position
            # asset_ids_json is guaranteed to be non-None here due to is_first_run check
            assert session.asset_ids_json is not None
            asset_ids = json.loads(session.asset_ids_json)
            start_index = session.current_asset_index

            logger.info(
                f"[{job_id}] Resuming from index {start_index} of {len(asset_ids)} total assets"
            )

            # Load only the remaining assets
            remaining_ids = asset_ids[start_index:]
            if remaining_ids:
                query = select(ImageAsset).where(ImageAsset.id.in_(remaining_ids))
                assets = db_session.execute(query).scalars().all()

                # Sort assets to match the order in asset_ids
                id_to_asset = {a.id: a for a in assets}
                assets = [id_to_asset[aid] for aid in remaining_ids if aid in id_to_asset]
            else:
                assets = []

            # Update status to PROCESSING
            session.status = FaceDetectionSessionStatus.PROCESSING.value
            if session.started_at is None:
                session.started_at = datetime.now()
            db_session.commit()

        if not assets:
            logger.info(f"[{job_id}] No assets to process")
            session.status = FaceDetectionSessionStatus.COMPLETED.value
            session.completed_at = datetime.now()
            session.total_images = 0
            db_session.commit()
            return {
                "session_id": session_id,
                "status": "completed",
                "total_images": 0,
                "processed_images": 0,
                "failed_images": 0,
                "faces_detected": 0,
                "faces_assigned": 0,
            }

        total_assets = len(assets)
        session.total_images = total_assets
        db_session.commit()

        logger.info(f"[{job_id}] Processing {total_assets} assets")

        # Process in batches
        face_service = get_face_service(db_session)
        batch_size = session.batch_size
        total_faces_detected = 0
        total_failed = 0
        last_error = None

        # Calculate total batches for progress tracking
        import math

        total_batches = math.ceil(len(assets) / batch_size) if len(assets) > 0 else 0
        session.total_batches = total_batches
        db_session.commit()

        for i in range(0, len(assets), batch_size):
            # Check if session status changed to PAUSED or CANCELLED
            db_session.refresh(session)

            if session.status == FaceDetectionSessionStatus.PAUSED.value:
                logger.info(
                    f"[{job_id}] Session {session_id} paused at batch "
                    f"{session.current_batch}/{total_batches}"
                )
                return {
                    "session_id": session_id,
                    "status": "paused",
                    "total_images": session.total_images,
                    "processed_images": session.processed_images,
                    "failed_images": session.failed_images,
                    "faces_detected": session.faces_detected,
                    "faces_assigned": session.faces_assigned,
                    "message": "Session paused by user",
                }

            if session.status == FaceDetectionSessionStatus.CANCELLED.value:
                logger.info(
                    f"[{job_id}] Session {session_id} cancelled at batch "
                    f"{session.current_batch}/{total_batches}"
                )
                session.completed_at = datetime.now()
                db_session.commit()
                return {
                    "session_id": session_id,
                    "status": "cancelled",
                    "total_images": session.total_images,
                    "processed_images": session.processed_images,
                    "failed_images": session.failed_images,
                    "faces_detected": session.faces_detected,
                    "faces_assigned": session.faces_assigned,
                    "message": "Session cancelled by user",
                }

            batch = assets[i : i + batch_size]
            asset_ids = [a.id for a in batch]

            current_batch_num = (i // batch_size) + 1
            session.current_batch = current_batch_num

            logger.info(
                f"[{job_id}] Processing batch {current_batch_num}/{total_batches} "
                f"({len(asset_ids)} assets)"
            )

            try:
                result = face_service.process_assets_batch(
                    asset_ids=asset_ids,
                    min_confidence=session.min_confidence,
                    min_face_size=session.min_face_size,
                    prefetch_batch_size=session.batch_size,
                )

                # Update session progress
                total_faces_detected += result["total_faces"]
                total_failed += result["errors"]

                session.processed_images += result["processed"]
                session.faces_detected = total_faces_detected
                session.failed_images = total_failed

                # Update current position for resume support
                session.current_asset_index += len(batch)

                if result.get("error_details"):
                    last_error = f"Batch errors: {result['error_details'][0]['error']}"
                    session.last_error = last_error

                db_session.commit()

                # Update Redis cache for real-time progress (fast, non-blocking)
                try:
                    from image_search_service.queue.worker import get_redis

                    redis_client = get_redis()
                    cache_key = f"face_detection:{session.id}:progress"

                    # Store progress as JSON with metadata
                    progress_data = {
                        "processed_images": session.processed_images,
                        "total_images": session.total_images,
                        "faces_detected": session.faces_detected,
                        "failed_images": session.failed_images,
                        "current_batch": session.current_batch,
                        "total_batches": session.total_batches,
                        "updated_at": datetime.now(UTC).isoformat(),
                    }
                    redis_client.set(cache_key, json.dumps(progress_data), ex=3600)

                    logger.debug(
                        f"[{job_id}] Updated Redis cache: "
                        f"{session.processed_images}/{session.total_images} images"
                    )
                except Exception as e:
                    # Graceful degradation - log but don't fail the job
                    logger.warning(
                        f"[{job_id}] Failed to update Redis progress cache: {e} "
                        "(continuing with database-only progress)"
                    )

                logger.info(
                    f"[{job_id}] Batch complete: "
                    f"{result['processed']} processed, {result['total_faces']} faces, "
                    f"{result['errors']} errors"
                )

            except Exception as e:
                logger.error(f"[{job_id}] Batch processing error: {e}")
                total_failed += len(batch)
                last_error = str(e)
                session.failed_images = total_failed
                session.last_error = last_error
                # Still update position even on error to skip problematic batch
                session.current_asset_index += len(batch)
                db_session.commit()

                # Update Redis cache even on error (for consistent progress reporting)
                try:
                    from image_search_service.queue.worker import get_redis

                    redis_client = get_redis()
                    cache_key = f"face_detection:{session.id}:progress"

                    progress_data = {
                        "processed_images": session.processed_images,
                        "total_images": session.total_images,
                        "faces_detected": session.faces_detected,
                        "failed_images": session.failed_images,
                        "current_batch": session.current_batch,
                        "total_batches": session.total_batches,
                        "updated_at": datetime.now(UTC).isoformat(),
                    }
                    redis_client.set(cache_key, json.dumps(progress_data), ex=3600)
                except Exception as redis_error:
                    logger.warning(
                        f"[{job_id}] Failed to update Redis cache on error: {redis_error}"
                    )

        # Auto-assign faces to known persons (uses config-based thresholds)
        logger.info(f"[{job_id}] Auto-assigning faces to known persons")
        faces_assigned_to_persons = 0
        suggestions_created = 0

        try:
            # FaceAssigner now uses config service internally for thresholds
            assigner = get_face_assigner(db_session=db_session)

            # Assign faces created during this session
            assignment_result = assigner.assign_new_faces(
                since=session.started_at,
                max_faces=10000,  # Process all new faces
            )

            faces_assigned_to_persons = assignment_result.get("auto_assigned", 0)
            suggestions_created = assignment_result.get("suggestions_created", 0)

            # Update detailed tracking fields
            session.faces_assigned_to_persons = faces_assigned_to_persons
            session.suggestions_created = suggestions_created
            # Update legacy field for backward compatibility
            session.faces_assigned = faces_assigned_to_persons

            logger.info(
                f"[{job_id}] Auto-assignment complete: "
                f"{faces_assigned_to_persons} faces auto-assigned, "
                f"{suggestions_created} suggestions created"
            )

        except Exception as e:
            logger.error(f"[{job_id}] Auto-assignment error: {e}")
            # Don't fail the whole session for assignment errors
            session.last_error = f"Assignment error: {e}"

        # Run clustering for unlabeled faces
        if total_faces_detected > 0:
            logger.info(f"[{job_id}] Running clustering for unlabeled faces")
            try:
                from image_search_service.faces.clusterer import get_face_clusterer
                from image_search_service.vector.face_qdrant import get_face_qdrant_client

                qdrant_client = get_face_qdrant_client()
                clusterer = get_face_clusterer(
                    db_session,
                    qdrant_client=qdrant_client,
                    min_cluster_size=3,  # Minimum faces per cluster
                )

                # Cluster faces that don't have a person_id (unlabeled)
                cluster_result = clusterer.cluster_unlabeled_faces(
                    quality_threshold=0.3,  # Include most detected faces
                    max_faces=10000,  # Process up to 10k faces
                )

                clusters_found = cluster_result.get("clusters_found", 0)

                logger.info(
                    f"[{job_id}] Clustering complete: "
                    f"{clusters_found} clusters, "
                    f"{cluster_result.get('noise_count', 0)} noise"
                )

                # Update session with clustering stats
                session.clusters_created = clusters_found
                # Update legacy field for backward compatibility (sum of persons + clusters)
                session.faces_assigned = (session.faces_assigned_to_persons or 0) + clusters_found
                db_session.commit()

            except Exception as e:
                logger.error(f"[{job_id}] Clustering error (non-fatal): {e}")
                # Don't fail the whole job if clustering fails

        # ============================================================
        # Post-Training Suggestion Generation
        # ============================================================
        suggestions_jobs_queued = 0
        suggestions_mode = "all"
        centroid_jobs_queued = 0
        prototype_jobs_queued = 0

        try:
            from sqlalchemy import func

            from image_search_service.db.models import Person
            from image_search_service.queue.worker import get_redis
            from image_search_service.services.config_service import SyncConfigService

            # Get configuration
            sync_config = SyncConfigService(db_session)
            suggestions_mode = sync_config.get_string("post_training_suggestions_mode")
            top_n_count = sync_config.get_int("post_training_suggestions_top_n_count")
            use_centroids = sync_config.get_bool("post_training_use_centroids")
            min_faces_for_centroid = sync_config.get_int("centroid_min_faces_for_suggestions")

            logger.info(
                f"[{job_id}] Queuing post-training suggestions",
                extra={
                    "session_id": session_id,
                    "mode": suggestions_mode,
                    "use_centroids": use_centroids,
                    "min_faces_for_centroid": min_faces_for_centroid,
                    "top_n_count": top_n_count if suggestions_mode == "top_n" else None,
                }
            )

            # Get persons to generate suggestions for
            if suggestions_mode == "all":
                # Get ALL persons with at least 1 labeled face
                persons_query = (
                    db_session.query(Person.id, func.count(FaceInstance.id).label("face_count"))
                    .join(FaceInstance, FaceInstance.person_id == Person.id)
                    .group_by(Person.id)
                    .having(func.count(FaceInstance.id) > 0)
                    .order_by(func.count(FaceInstance.id).desc())
                )
            else:  # top_n
                # Get TOP N persons by labeled face count
                persons_query = (
                    db_session.query(Person.id, func.count(FaceInstance.id).label("face_count"))
                    .join(FaceInstance, FaceInstance.person_id == Person.id)
                    .group_by(Person.id)
                    .having(func.count(FaceInstance.id) > 0)
                    .order_by(func.count(FaceInstance.id).desc())
                    .limit(top_n_count)
                )

            persons = persons_query.all()

            if persons:
                # Queue suggestion jobs (centroid or prototype based on config)
                from rq import Queue

                redis_client = get_redis()
                queue = Queue("default", connection=redis_client)

                for person in persons:
                    # Decide which job to use based on face count and config
                    if use_centroids and person.face_count >= min_faces_for_centroid:
                        # Use faster centroid-based search
                        job = queue.enqueue(
                            "image_search_service.queue.face_jobs.find_more_centroid_suggestions_job",
                            person_id=str(person.id),
                            min_similarity=0.70,
                            max_results=50,
                            unassigned_only=True,
                            job_timeout="10m",
                        )
                        centroid_jobs_queued += 1
                        logger.debug(
                            f"[{job_id}] Queued centroid suggestion job",
                            extra={
                                "person_id": str(person.id),
                                "face_count": person.face_count,
                                "job_id": job.id,
                                "job_type": "centroid",
                            }
                        )
                    else:
                        # Fall back to prototype-based (for persons with few faces or when disabled)
                        job = queue.enqueue(
                            "image_search_service.queue.face_jobs.propagate_person_label_multiproto_job",
                            person_id=str(person.id),
                            max_suggestions=50,  # Standard limit
                            min_confidence=0.7,  # Standard threshold
                            job_timeout="10m",
                        )
                        prototype_jobs_queued += 1
                        logger.debug(
                            f"[{job_id}] Queued multi-proto suggestion job",
                            extra={
                                "person_id": str(person.id),
                                "face_count": person.face_count,
                                "job_id": job.id,
                                "job_type": "prototype",
                                "reason": "insufficient_faces" if use_centroids else "centroids_disabled",  # noqa: E501
                            }
                        )

                    suggestions_jobs_queued += 1

                logger.info(
                    f"[{job_id}] Post-training suggestion jobs queued",
                    extra={
                        "session_id": session_id,
                        "jobs_queued": suggestions_jobs_queued,
                        "centroid_jobs": centroid_jobs_queued,
                        "prototype_jobs": prototype_jobs_queued,
                        "mode": suggestions_mode,
                    }
                )
            else:
                logger.info(
                    f"[{job_id}] No persons with labeled faces found, "
                    "skipping post-training suggestions",
                    extra={"session_id": session_id},
                )

        except Exception as e:
            logger.error(f"[{job_id}] Post-training suggestion queuing error (non-fatal): {e}")
            # Don't fail the whole job if suggestion queuing fails

        # Mark session as complete
        session.status = FaceDetectionSessionStatus.COMPLETED.value
        session.completed_at = datetime.now()
        db_session.commit()

        logger.info(
            f"[{job_id}] Session {session_id} complete: "
            f"{session.processed_images} processed, {total_faces_detected} faces, "
            f"{session.faces_assigned_to_persons} assigned to persons, "
            f"{session.clusters_created} clusters, {total_failed} failed"
        )

        return {
            "session_id": session_id,
            "status": "completed",
            "total_images": session.total_images,
            "processed_images": session.processed_images,
            "failed_images": session.failed_images,
            "faces_detected": session.faces_detected,
            "faces_assigned": session.faces_assigned,
            "last_error": session.last_error,
            "suggestions_jobs_queued": suggestions_jobs_queued,
            "suggestions_mode": suggestions_mode,
        }

    except Exception as e:
        logger.exception(f"[{job_id}] Fatal error processing session {session_id}: {e}")

        # Mark session as failed
        try:
            session = db_session.get(FaceDetectionSession, session_uuid)
            if session:
                session.status = FaceDetectionSessionStatus.FAILED.value
                session.last_error = str(e)
                session.completed_at = datetime.now()
                db_session.commit()
        except Exception as commit_error:
            logger.error(f"[{job_id}] Failed to update session status: {commit_error}")

        return {
            "session_id": session_id,
            "status": "failed",
            "error": str(e),
        }

    finally:
        db_session.close()


def propagate_person_label_job(
    source_face_id: str,
    person_id: str,
    min_confidence: float | None = None,
    max_suggestions: int | None = None,
) -> dict[str, Any]:
    """Find similar faces and create suggestions when a face is labeled to a person.

    This job is triggered when a user assigns a face to a person. It searches for
    similar unassigned faces and creates FaceSuggestion records for user review.

    FORK-SAFETY (macOS): Disable proxy detection in work-horse subprocess.

    Args:
        source_face_id: UUID string of the face that was just labeled
        person_id: UUID string of the person the face was assigned to
        min_confidence: Minimum cosine similarity
            (defaults to config: face_suggestion_threshold)
        max_suggestions: Maximum suggestions to create
            (defaults to config: face_suggestion_max_results)

    Returns:
        Dictionary with job results
    """
    import uuid as uuid_lib

    from image_search_service.services.config_service import SyncConfigService
    from image_search_service.vector.face_qdrant import get_face_qdrant_client

    job = get_current_job()
    job_id = job.id if job else "no-job"

    # Convert string UUIDs to UUID objects
    source_face_uuid = uuid_lib.UUID(source_face_id)
    person_uuid = uuid_lib.UUID(person_id)

    logger.info(f"[{job_id}] Starting propagation for face {source_face_id} → person {person_id}")

    db_session = get_sync_session()

    # Get config values if not provided
    config_service = SyncConfigService(db_session)
    if min_confidence is None:
        min_confidence = config_service.get_float("face_suggestion_threshold")
    if max_suggestions is None:
        max_suggestions = config_service.get_int("face_suggestion_max_results")

    logger.info(
        f"[{job_id}] Using min_confidence={min_confidence}, max_suggestions={max_suggestions}"
    )

    try:
        # 1. Get the source face instance
        source_face = db_session.get(FaceInstance, source_face_uuid)
        if not source_face:
            logger.error(f"[{job_id}] Face {source_face_id} not found")
            return {"status": "error", "message": f"Face {source_face_id} not found"}

        # 2. Get face vector from Qdrant
        face_client = get_face_qdrant_client()

        # Retrieve the source face vector
        points = face_client.client.retrieve(
            collection_name="faces",
            ids=[str(source_face.qdrant_point_id)],
            with_vectors=True,
        )

        if not points or not points[0].vector:
            logger.error(f"[{job_id}] No vector found for face {source_face_id}")
            return {
                "status": "error",
                "message": f"No vector found for face {source_face_id}",
            }

        # Extract vector - handle both dict and list formats
        raw_vector = points[0].vector
        if isinstance(raw_vector, dict):
            source_vector: list[float] = list(raw_vector.values())[0]  # type: ignore[assignment]
        else:
            source_vector = raw_vector  # type: ignore[assignment]

        # 3. Search for similar faces (unassigned ones)
        search_results = face_client.client.query_points(
            collection_name="faces",
            query=source_vector,
            limit=max_suggestions + 10,  # Get extra to filter assigned ones
            score_threshold=min_confidence,
            with_payload=True,
        )

        # 4. Filter to unassigned faces and create suggestions
        suggestions_created = 0
        faces_checked = 0

        for result in search_results.points:
            qdrant_point_id = uuid_lib.UUID(str(result.id))
            confidence = result.score

            faces_checked += 1

            # Get the face instance by qdrant_point_id (NOT by primary key)
            face = db_session.execute(
                select(FaceInstance).where(FaceInstance.qdrant_point_id == qdrant_point_id)
            ).scalar_one_or_none()

            if not face:
                continue

            # Skip the source face itself
            if face.id == source_face_uuid:
                continue

            # Skip if already assigned to a person
            if face.person_id is not None:
                continue

            # Skip if suggestion already exists
            existing = db_session.execute(
                select(FaceSuggestion).where(
                    FaceSuggestion.face_instance_id == face.id,
                    FaceSuggestion.suggested_person_id == person_uuid,
                    FaceSuggestion.status == FaceSuggestionStatus.PENDING.value,
                )
            ).scalar_one_or_none()

            if existing:
                continue

            # Create suggestion
            suggestion = FaceSuggestion(
                face_instance_id=face.id,
                suggested_person_id=person_uuid,
                confidence=confidence,
                source_face_id=source_face_uuid,
                status=FaceSuggestionStatus.PENDING.value,
            )
            db_session.add(suggestion)
            suggestions_created += 1

            if suggestions_created >= max_suggestions:
                break

        db_session.commit()

        logger.info(
            f"[{job_id}] Propagation complete for face {source_face_id} → person {person_id}: "
            f"checked {faces_checked} faces, created {suggestions_created} suggestions"
        )

        return {
            "status": "completed",
            "source_face_id": source_face_id,
            "person_id": person_id,
            "faces_checked": faces_checked,
            "suggestions_created": suggestions_created,
        }

    except Exception as e:
        logger.exception(f"[{job_id}] Error in propagation job: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        db_session.close()


def expire_old_suggestions_job(
    days_threshold: int | None = None,
) -> dict[str, Any]:
    """Expire pending suggestions older than the threshold.

    This job should be run periodically (e.g., daily) to clean up
    stale suggestions that were never reviewed.

    Args:
        days_threshold: Number of days after which to expire suggestions
                       (defaults to config: face_suggestion_expiry_days)

    Returns:
        Dictionary with job results
    """
    from image_search_service.services.config_service import SyncConfigService

    job = get_current_job()
    job_id = job.id if job else "no-job"

    db_session = get_sync_session()

    # Get config value if not provided
    if days_threshold is None:
        config_service = SyncConfigService(db_session)
        days_threshold = config_service.get_int("face_suggestion_expiry_days")

    logger.info(f"[{job_id}] Expiring suggestions older than {days_threshold} days")

    try:
        cutoff_date = datetime.now(UTC) - timedelta(days=days_threshold)

        # Find pending suggestions older than cutoff
        query = select(FaceSuggestion).where(
            FaceSuggestion.status == FaceSuggestionStatus.PENDING.value,
            FaceSuggestion.created_at < cutoff_date,
        )
        result = db_session.execute(query)
        old_suggestions = result.scalars().all()

        expired_count = 0
        for suggestion in old_suggestions:
            suggestion.status = FaceSuggestionStatus.EXPIRED.value
            suggestion.reviewed_at = datetime.now(UTC)
            expired_count += 1

        db_session.commit()

        logger.info(f"[{job_id}] Expired {expired_count} old suggestions")

        return {
            "status": "completed",
            "expired_count": expired_count,
            "threshold_days": days_threshold,
        }

    except Exception as e:
        logger.exception(f"[{job_id}] Error expiring suggestions: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        db_session.close()


def find_more_suggestions_job(
    person_id: str,
    prototype_count: int = 50,
    min_confidence: float | None = None,
    max_suggestions: int = 100,
    progress_key: str | None = None,
) -> dict[str, Any]:
    """Find more face suggestions using random sampling of labeled faces.

    Uses Quality + Diversity weighted selection to pick labeled faces as temp prototypes.
    This enables discovery of faces that don't match the fixed prototype set but may
    still belong to the same person.

    Selection Strategy (Quality + Diversity):
    1. Get all labeled faces for the person (excluding existing prototypes)
    2. Score each face: quality_score * 0.7 + diversity_bonus * 0.3
       - diversity_bonus based on unique asset_id distribution
    3. Weighted random selection of top N faces
    4. For each selected face, query Qdrant for similar unknown faces
    5. Aggregate results using MAX score across all matches
    6. Create FaceSuggestion records (skip duplicates with existing pending)
    7. Update progress in Redis after each prototype

    Args:
        person_id: UUID string of the person
        prototype_count: Number of faces to sample as prototypes (default: 50)
        min_confidence: Similarity threshold (default: from SystemConfig)
        max_suggestions: Maximum new suggestions to create
        progress_key: Redis key for progress updates (optional)

    Returns:
        dict with: status, suggestions_created, prototypes_used,
        candidates_found, duplicates_skipped
    """
    import random
    import uuid as uuid_lib

    from image_search_service.db.models import Person, PersonPrototype
    from image_search_service.services.config_service import SyncConfigService
    from image_search_service.vector.face_qdrant import get_face_qdrant_client

    job = get_current_job()
    job_id = job.id if job else "no-job"

    person_uuid = uuid_lib.UUID(person_id)

    logger.info(
        f"[{job_id}] Starting find-more for person {person_id} with {prototype_count} prototypes"
    )

    db_session = get_sync_session()

    # Get config values if not provided
    config_service = SyncConfigService(db_session)
    if min_confidence is None:
        min_confidence = config_service.get_float("face_suggestion_threshold")

    # Helper to update progress
    def update_progress(phase: str, current: int, total: int, message: str) -> None:
        if not progress_key:
            return
        try:
            from redis import Redis

            from image_search_service.core.config import get_settings

            settings = get_settings()
            redis_client = Redis.from_url(settings.redis_url)

            progress_data = {
                "phase": phase,
                "current": current,
                "total": total,
                "message": message,
                "timestamp": datetime.now(UTC).isoformat(),
            }
            redis_client.set(progress_key, json.dumps(progress_data), ex=3600)
        except Exception as e:
            logger.warning(f"[{job_id}] Failed to update progress: {e}")

    try:
        update_progress("selecting", 0, prototype_count, "Selecting prototypes...")

        # 1. Get person
        person = db_session.get(Person, person_uuid)
        if not person:
            logger.warning(f"[{job_id}] Person {person_id} not found")
            return {"status": "error", "message": "Person not found"}

        # 2. Get existing prototypes to exclude
        existing_proto_query = select(PersonPrototype.face_instance_id).where(
            PersonPrototype.person_id == person_uuid
        )
        existing_proto_ids = set(db_session.execute(existing_proto_query).scalars().all())

        # 3. Get all labeled faces for this person (excluding prototypes)
        labeled_faces_query = select(FaceInstance).where(FaceInstance.person_id == person_uuid)
        if existing_proto_ids:
            labeled_faces_query = labeled_faces_query.where(
                ~FaceInstance.id.in_(existing_proto_ids)
            )
        labeled_faces = list(db_session.execute(labeled_faces_query).scalars().all())

        if len(labeled_faces) < 10:
            logger.warning(
                f"[{job_id}] Only {len(labeled_faces)} labeled faces available "
                f"(minimum 10 required)"
            )
            return {
                "status": "error",
                "message": f"Only {len(labeled_faces)} labeled faces (need 10+)",
            }

        # 4. Score faces using Quality + Diversity
        asset_usage_count: dict[int, int] = {}
        face_scores: list[tuple[FaceInstance, float]] = []

        for face in labeled_faces:
            quality = face.quality_score or 0.5

            # Diversity bonus (penalize repeated use of same asset)
            asset_id = face.asset_id
            usage = asset_usage_count.get(asset_id, 0)
            diversity_penalty = min(0.3, usage * 0.1)
            diversity_bonus = 0.3 - diversity_penalty

            score = quality * 0.7 + diversity_bonus
            face_scores.append((face, score))

            # Track usage for next iteration
            asset_usage_count[asset_id] = usage + 1

        # 5. Weighted random selection
        face_scores.sort(key=lambda x: x[1], reverse=True)
        actual_count = min(prototype_count, len(face_scores))

        # Use top faces with weighted randomness
        selected_faces: list[FaceInstance] = []
        weights = [score for _, score in face_scores[: actual_count * 2]]
        candidates = [face for face, _ in face_scores[: actual_count * 2]]

        if len(candidates) > 0:
            selected_faces = random.choices(
                candidates,
                weights=weights,
                k=min(actual_count, len(candidates)),
            )

        logger.info(
            f"[{job_id}] Selected {len(selected_faces)} prototypes "
            f"from {len(labeled_faces)} labeled faces"
        )

        update_progress("searching", 0, len(selected_faces), "Searching for similar faces...")

        # 6. Search using each selected face
        qdrant = get_face_qdrant_client()
        candidate_faces: dict[str, dict[str, Any]] = {}  # face_id -> {scores, max_score}

        for idx, face in enumerate(selected_faces):
            if not face.qdrant_point_id:
                continue

            # Get embedding
            embedding = qdrant.get_embedding_by_point_id(face.qdrant_point_id)
            if not embedding:
                continue

            # Search for similar faces
            results = qdrant.search_similar_faces(
                query_embedding=embedding,
                limit=max_suggestions * 3,
                score_threshold=min_confidence,
            )

            for result in results:
                if result.payload is None:
                    continue

                face_id_str = result.payload.get("face_instance_id")
                if not face_id_str:
                    continue

                try:
                    face_id_uuid = uuid_lib.UUID(face_id_str)
                except ValueError:
                    continue

                # Check if face is already assigned
                candidate_face = db_session.get(FaceInstance, face_id_uuid)
                if not candidate_face or candidate_face.person_id is not None:
                    continue

                # Aggregate scores
                face_id = str(face_id_uuid)
                if face_id not in candidate_faces:
                    candidate_faces[face_id] = {
                        "scores": {},
                        "max_score": 0.0,
                        "face_instance": candidate_face,
                    }

                proto_id = str(face.id)
                candidate_faces[face_id]["scores"][proto_id] = result.score
                candidate_faces[face_id]["max_score"] = max(
                    candidate_faces[face_id]["max_score"],
                    result.score,
                )

            update_progress(
                "searching",
                idx + 1,
                len(selected_faces),
                f"Processed {idx + 1}/{len(selected_faces)} prototypes",
            )

        logger.info(
            f"[{job_id}] Found {len(candidate_faces)} candidate faces "
            f"from {len(selected_faces)} prototypes"
        )

        update_progress("creating", 0, len(candidate_faces), "Creating suggestions...")

        # 7. Get existing pending suggestions to avoid duplicates
        existing_pending_query = select(FaceSuggestion.face_instance_id).where(
            FaceSuggestion.suggested_person_id == person_uuid,
            FaceSuggestion.status == FaceSuggestionStatus.PENDING.value,
        )
        existing_pending_ids = set(
            str(fid) for fid in db_session.execute(existing_pending_query).scalars().all()
        )

        # 8. Sort by confidence and create suggestions
        sorted_candidates = sorted(
            candidate_faces.items(),
            key=lambda x: x[1]["max_score"],
            reverse=True,
        )[:max_suggestions]

        suggestions_created = 0
        duplicates_skipped = 0

        for face_id, data in sorted_candidates:
            # Skip if already has pending suggestion
            if face_id in existing_pending_ids:
                duplicates_skipped += 1
                continue

            # Find best prototype (highest quality among matches)
            def get_quality(pid: str) -> float:
                proto_face = db_session.get(FaceInstance, uuid_lib.UUID(pid))
                if proto_face and proto_face.quality_score is not None:
                    return proto_face.quality_score
                return 0.0

            best_proto_id = max(data["scores"].keys(), key=get_quality)

            suggestion = FaceSuggestion(
                face_instance_id=uuid_lib.UUID(face_id),
                suggested_person_id=person_uuid,
                source_face_id=uuid_lib.UUID(best_proto_id),
                confidence=data["scores"][best_proto_id],
                aggregate_confidence=data["max_score"],
                matching_prototype_ids=list(data["scores"].keys()),
                prototype_scores=data["scores"],
                prototype_match_count=len(data["scores"]),
                status=FaceSuggestionStatus.PENDING.value,
            )
            db_session.add(suggestion)
            suggestions_created += 1

        db_session.commit()

        logger.info(
            f"[{job_id}] Find-more complete for {person.name}: "
            f"{suggestions_created} suggestions created, "
            f"{duplicates_skipped} duplicates skipped"
        )

        # Update final progress
        update_progress(
            "completed",
            len(selected_faces),
            len(selected_faces),
            f"Created {suggestions_created} new suggestions",
        )

        return {
            "status": "completed",
            "suggestions_created": suggestions_created,
            "prototypes_used": len(selected_faces),
            "candidates_found": len(candidate_faces),
            "duplicates_skipped": duplicates_skipped,
        }

    except Exception as e:
        logger.exception(f"[{job_id}] Error in find-more job: {e}")
        if progress_key:
            update_progress("failed", 0, 0, str(e))
        return {"status": "error", "message": str(e)}
    finally:
        db_session.close()


def propagate_person_label_multiproto_job(
    person_id: str,
    min_confidence: float = 0.7,
    max_suggestions: int = 50,
    preserve_existing: bool = True,
) -> dict[str, Any]:
    """Generate face suggestions using ALL prototypes for a person.

    Instead of using a single source face, this searches with each prototype
    and aggregates the results for better matching quality.

    For each candidate face:
    - Searches against ALL prototypes for the person
    - Records which prototypes matched and their scores
    - Uses MAX score as aggregate_confidence
    - Stores prototype match count

    Args:
        person_id: UUID string of the person
        min_confidence: Minimum cosine similarity (default: 0.7)
        max_suggestions: Maximum suggestions to create (default: 50)
        preserve_existing: If True, keep existing pending suggestions and only add new ones
            (default: True)

    Returns:
        dict with counts: suggestions_created, prototypes_used, candidates_evaluated,
        expired_count, preserved_count
    """
    import uuid as uuid_lib

    from image_search_service.db.models import Person, PersonPrototype
    from image_search_service.vector.face_qdrant import get_face_qdrant_client

    job = get_current_job()
    job_id = job.id if job else "no-job"

    logger.info(f"[{job_id}] Starting multi-prototype propagation for person {person_id}")

    db_session = get_sync_session()

    try:
        # 1. Get person
        person_uuid = uuid_lib.UUID(person_id)
        person = db_session.get(Person, person_uuid)

        if not person:
            logger.warning(f"[{job_id}] Person {person_id} not found")
            return {"status": "error", "message": "Person not found"}

        # 2. Get all prototypes for this person
        prototypes_query = select(PersonPrototype).where(PersonPrototype.person_id == person_uuid)
        prototypes = list(db_session.execute(prototypes_query).scalars().all())

        if not prototypes:
            logger.warning(f"[{job_id}] No prototypes found for person {person_id}")
            return {"status": "error", "message": "No prototypes"}

        # 3. Get prototype embeddings from Qdrant
        qdrant = get_face_qdrant_client()

        # First, collect all face instances and point IDs
        face_instances = {}  # proto.face_instance_id -> face
        point_ids = []

        for proto in prototypes:
            face = db_session.get(FaceInstance, proto.face_instance_id)
            if face and face.qdrant_point_id:
                face_instances[proto.face_instance_id] = face
                point_ids.append(face.qdrant_point_id)

        # Batch retrieve embeddings
        embeddings_map = qdrant.get_embeddings_batch(point_ids)

        # Build prototype embeddings dict
        prototype_embeddings: dict[str, dict[str, Any]] = {}

        for proto in prototypes:
            face = face_instances.get(proto.face_instance_id)
            if face and face.qdrant_point_id:
                embedding = embeddings_map.get(face.qdrant_point_id)
                if embedding:
                    prototype_embeddings[str(proto.face_instance_id)] = {
                        "embedding": embedding,
                        "face_id": str(face.id),
                        "quality": face.quality_score or 0.0,
                    }

        if not prototype_embeddings:
            logger.warning(f"[{job_id}] No prototype embeddings found for person {person_id}")
            return {"status": "error", "message": "No prototype embeddings found"}

        logger.info(
            f"[{job_id}] Found {len(prototype_embeddings)} prototypes for person {person.name}"
        )

        # 4. Search using each prototype and aggregate results
        candidate_faces: dict[str, dict[str, Any]] = {}  # face_id -> {scores, max_score, ...}

        for proto_face_id, proto_data in prototype_embeddings.items():
            # Search for similar faces
            results = qdrant.search_similar_faces(
                query_embedding=proto_data["embedding"],
                limit=max_suggestions * 3,  # Get more candidates for aggregation
                score_threshold=min_confidence,
            )

            logger.debug(
                f"[{job_id}] Prototype {proto_face_id}: found {len(results)} similar faces"
            )

            for result in results:
                # Extract face_id from payload
                if result.payload is None:
                    continue

                face_id_str = result.payload.get("face_instance_id")
                if not face_id_str:
                    continue

                try:
                    face_id_uuid = uuid_lib.UUID(face_id_str)
                except ValueError:
                    logger.warning(f"[{job_id}] Invalid face_id in payload: {face_id_str}")
                    continue

                score = result.score

                # Get the face instance to check person assignment
                face = db_session.get(FaceInstance, face_id_uuid)

                if not face:
                    continue

                # Skip if this face belongs to any person already
                if face.person_id is not None:
                    continue

                # Aggregate results
                face_id = str(face_id_uuid)
                if face_id not in candidate_faces:
                    candidate_faces[face_id] = {
                        "scores": {},
                        "max_score": 0.0,
                        "face_instance": face,
                    }

                candidate_faces[face_id]["scores"][proto_face_id] = score
                candidate_faces[face_id]["max_score"] = max(
                    candidate_faces[face_id]["max_score"],
                    score,
                )

        logger.info(
            f"[{job_id}] Aggregated {len(candidate_faces)} candidate faces from all prototypes"
        )

        # 5. Sort by aggregate confidence (max score) and take top N
        sorted_candidates = sorted(
            candidate_faces.items(),
            key=lambda x: x[1]["max_score"],
            reverse=True,
        )[:max_suggestions]

        # 6. Conditionally expire old pending suggestions for this person
        now = datetime.now(UTC)
        expired_count = 0

        if not preserve_existing:
            # Only expire when explicitly requested (preserve_existing=False)
            expire_result = db_session.execute(
                select(FaceSuggestion).where(
                    FaceSuggestion.suggested_person_id == person_uuid,
                    FaceSuggestion.status == FaceSuggestionStatus.PENDING.value,
                )
            )
            old_suggestions = expire_result.scalars().all()

            for old_suggestion in old_suggestions:
                old_suggestion.status = FaceSuggestionStatus.EXPIRED.value
                old_suggestion.reviewed_at = now

            expired_count = len(old_suggestions)
            if expired_count > 0:
                logger.info(f"[{job_id}] Expired {expired_count} old pending suggestions")

        # 6.5. Get existing pending suggestions to avoid duplicates (when preserving)
        existing_face_ids: set[str] = set()
        if preserve_existing:
            existing_pending_result = db_session.execute(
                select(FaceSuggestion.face_instance_id).where(
                    FaceSuggestion.suggested_person_id == person_uuid,
                    FaceSuggestion.status == FaceSuggestionStatus.PENDING.value,
                )
            )
            existing_face_ids = set(str(fid) for fid in existing_pending_result.scalars().all())
            if existing_face_ids:
                logger.debug(
                    f"[{job_id}] Found {len(existing_face_ids)} existing pending suggestions "
                    "to preserve"
                )

        # 7. Create new suggestions with multi-prototype data
        suggestions_created = 0
        skipped_duplicates = 0

        for face_id, data in sorted_candidates:
            # Skip if already has pending suggestion (preserve mode)
            if face_id in existing_face_ids:
                logger.debug(f"[{job_id}] Skipping face {face_id} - already has pending suggestion")
                skipped_duplicates += 1
                continue

            # Find best prototype for source_face_id (highest quality among matches)
            best_proto_id = max(
                data["scores"].keys(),
                key=lambda pid: prototype_embeddings[pid]["quality"],
            )

            suggestion = FaceSuggestion(
                face_instance_id=uuid_lib.UUID(face_id),
                suggested_person_id=person_uuid,
                source_face_id=uuid_lib.UUID(best_proto_id),  # Best quality matching prototype
                confidence=data["scores"][best_proto_id],  # Score from that prototype
                aggregate_confidence=data["max_score"],  # MAX score across all
                matching_prototype_ids=list(data["scores"].keys()),
                prototype_scores=data["scores"],
                prototype_match_count=len(data["scores"]),
                status=FaceSuggestionStatus.PENDING.value,
            )
            db_session.add(suggestion)
            suggestions_created += 1

        db_session.commit()

        preserved_count = len(existing_face_ids) if preserve_existing else 0

        logger.info(
            f"[{job_id}] Multi-prototype propagation complete for {person.name}: "
            f"{suggestions_created} suggestions created, "
            f"{len(prototype_embeddings)} prototypes used, "
            f"{preserved_count} preserved, {skipped_duplicates} duplicates skipped"
        )

        return {
            "status": "completed",
            "suggestions_created": suggestions_created,
            "prototypes_used": len(prototype_embeddings),
            "candidates_evaluated": len(candidate_faces),
            "expired_count": expired_count,
            "preserved_count": preserved_count,
            "skipped_duplicates": skipped_duplicates,
        }

    except Exception as e:
        logger.exception(f"[{job_id}] Error in multi-prototype propagation job: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        db_session.close()


def find_more_centroid_suggestions_job(
    person_id: str,
    min_similarity: float = 0.65,
    max_results: int = 200,
    unassigned_only: bool = True,
    progress_key: str | None = None,
) -> dict[str, Any]:
    """Find more suggestions using person centroid (faster than dynamic prototypes).

    Flow:
    1. Get or compute centroid for person (use CentroidService)
    2. Search Qdrant faces collection using centroid embedding
    3. Filter results (exclude already-assigned faces if unassigned_only)
    4. Create FaceSuggestion records for matches (status=PENDING)
    5. Return job result with counts

    Args:
        person_id: UUID string of the person
        min_similarity: Minimum cosine similarity threshold (0.5-0.95)
        max_results: Maximum number of suggestions to create (1-500)
        unassigned_only: If True, only suggest unassigned faces

    Returns:
        dict with: suggestions_created, centroids_used, candidates_found,
        duplicates_skipped
    """
    import uuid as uuid_lib

    from image_search_service.db.models import CentroidType, FaceInstance, Person, PersonCentroid
    from image_search_service.vector.centroid_qdrant import get_centroid_qdrant_client
    from image_search_service.vector.face_qdrant import get_face_qdrant_client

    job = get_current_job()
    job_id = job.id if job else "no-job"

    person_uuid = uuid_lib.UUID(person_id)

    logger.info(
        f"[{job_id}] Starting centroid-based find-more for person {person_id} "
        f"(min_similarity={min_similarity}, max_results={max_results})"
    )

    db_session = get_sync_session()

    # Helper to update progress
    def update_progress(phase: str, current: int, total: int, message: str) -> None:
        if not progress_key:
            return
        try:
            from redis import Redis

            from image_search_service.core.config import get_settings

            settings = get_settings()
            redis_client = Redis.from_url(settings.redis_url)

            progress_data = {
                "phase": phase,
                "current": current,
                "total": total,
                "message": message,
                "person_id": person_id,
                "timestamp": datetime.now(UTC).isoformat(),
            }
            redis_client.set(progress_key, json.dumps(progress_data), ex=3600)
        except Exception as e:
            logger.warning(f"[{job_id}] Failed to update progress: {e}")

    try:
        # Initial progress update
        update_progress("starting", 0, 4, "Initializing centroid search")
        # 1. Get person
        person = db_session.get(Person, person_uuid)
        if not person:
            logger.warning(f"[{job_id}] Person {person_id} not found")
            return {"status": "error", "message": "Person not found"}

        # 2. Get or compute active centroid
        centroid_query = (
            select(PersonCentroid)
            .where(
                PersonCentroid.person_id == person_uuid,
                PersonCentroid.status == "active",
                PersonCentroid.centroid_type == CentroidType.GLOBAL,
            )
            .order_by(PersonCentroid.created_at.desc())
            .limit(1)
        )
        centroid_result = db_session.execute(centroid_query)
        centroid = centroid_result.scalar_one_or_none()

        if not centroid:
            # Compute centroid on-demand (sync operation)
            logger.info(f"[{job_id}] No active centroid found, computing...")

            # Get all face instances for this person
            faces_query = select(FaceInstance).where(FaceInstance.person_id == person_uuid)
            faces = list(db_session.execute(faces_query).scalars().all())

            if len(faces) < 5:
                logger.warning(
                    f"[{job_id}] Only {len(faces)} faces available "
                    f"(minimum 5 required for centroid)"
                )
                return {
                    "status": "error",
                    "message": f"Only {len(faces)} faces (need 5+)",
                }

            # Retrieve embeddings from Qdrant
            face_qdrant = get_face_qdrant_client()

            # Collect all Qdrant point IDs
            point_ids = [face.qdrant_point_id for face in faces if face.qdrant_point_id]

            # Batch retrieve embeddings
            embeddings_map = face_qdrant.get_embeddings_batch(point_ids)

            # Build result lists, maintaining order
            embeddings = []
            face_ids = []

            for face in faces:
                if face.qdrant_point_id:
                    embedding = embeddings_map.get(face.qdrant_point_id)
                    if embedding:
                        embeddings.append(embedding)
                        face_ids.append(face.id)

            if len(embeddings) < 5:
                logger.warning(
                    f"[{job_id}] Only {len(embeddings)} embeddings found in Qdrant "
                    f"(minimum 5 required)"
                )
                return {
                    "status": "error",
                    "message": f"Only {len(embeddings)} embeddings (need 5+)",
                }

            # Compute centroid vector
            import hashlib

            import numpy as np

            from image_search_service.core.config import get_settings

            embeddings_array = np.array(embeddings, dtype=np.float32)

            # Compute mean and normalize
            centroid_mean = np.mean(embeddings_array, axis=0)
            norm = np.linalg.norm(centroid_mean)
            centroid_vector = (centroid_mean / norm).astype(np.float32)

            # Create centroid record
            centroid_id = uuid_lib.uuid4()
            settings = get_settings()

            # Compute source hash
            sorted_ids = sorted(str(fid) for fid in face_ids)
            hash_input = ":".join(sorted_ids)
            source_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

            centroid = PersonCentroid(
                centroid_id=centroid_id,
                person_id=person_uuid,
                qdrant_point_id=centroid_id,
                model_version=settings.centroid_model_version,
                centroid_version=settings.centroid_algorithm_version,
                centroid_type=CentroidType.GLOBAL,
                cluster_label="global",
                n_faces=len(face_ids),
                status="active",
                source_face_ids_hash=source_hash,
                build_params={
                    "trim_outliers": False,
                    "n_faces_used": len(face_ids),
                },
            )
            db_session.add(centroid)
            db_session.flush()

            # Store in Qdrant
            centroid_qdrant = get_centroid_qdrant_client()
            centroid_qdrant.upsert_centroid(
                centroid_id=centroid_id,
                vector=centroid_vector.tolist(),
                payload={
                    "person_id": person_uuid,
                    "centroid_id": centroid_id,
                    "model_version": settings.centroid_model_version,
                    "centroid_version": settings.centroid_algorithm_version,
                    "centroid_type": "global",
                    "cluster_label": "global",
                    "n_faces": len(face_ids),
                    "created_at": centroid.created_at.isoformat(),
                    "source_hash": source_hash,
                    "build_params": centroid.build_params,
                },
            )

            db_session.commit()
            logger.info(
                f"[{job_id}] Created centroid {centroid_id} from {len(face_ids)} faces"
            )

        # 3. Get centroid vector from Qdrant
        update_progress("retrieving", 1, 4, "Retrieved centroid embedding")
        centroid_qdrant = get_centroid_qdrant_client()
        centroid_vector = centroid_qdrant.get_centroid_vector(centroid.centroid_id)

        if not centroid_vector:
            logger.error(
                f"[{job_id}] Centroid vector not found in Qdrant for {centroid.centroid_id}"
            )
            update_progress("failed", 0, 0, "Centroid vector not found")
            return {"status": "error", "message": "Centroid vector not found"}

        # 4. Search faces collection using centroid
        update_progress(
            "searching", 2, 4, f"Searching for similar faces (threshold={min_similarity})"
        )
        search_results = centroid_qdrant.search_faces_with_centroid(
            centroid_vector=centroid_vector,
            limit=max_results * 2,  # Get extra to account for filtering
            score_threshold=min_similarity,
            exclude_person_id=person_uuid if unassigned_only else None,
        )

        logger.info(
            f"[{job_id}] Found {len(search_results)} candidate faces using centroid"
        )

        # 5. Get existing pending suggestions to avoid duplicates
        update_progress(
            "creating", 3, 4, f"Creating suggestions from {len(search_results)} matches"
        )
        existing_pending_query = select(FaceSuggestion.face_instance_id).where(
            FaceSuggestion.suggested_person_id == person_uuid,
            FaceSuggestion.status == FaceSuggestionStatus.PENDING.value,
        )
        existing_pending_ids = set(
            str(fid) for fid in db_session.execute(existing_pending_query).scalars().all()
        )

        # 6. Create FaceSuggestion records
        suggestions_created = 0
        duplicates_skipped = 0
        face_qdrant = get_face_qdrant_client()

        for result in search_results:
            if suggestions_created >= max_results:
                break

            # Get face_id from payload
            if result.payload is None:
                continue

            face_id_str = result.payload.get("face_instance_id")
            if not face_id_str:
                continue

            try:
                face_id_uuid = uuid_lib.UUID(face_id_str)
            except ValueError:
                logger.warning(f"[{job_id}] Invalid face_id in payload: {face_id_str}")
                continue

            # Check if already has pending suggestion
            if str(face_id_uuid) in existing_pending_ids:
                duplicates_skipped += 1
                continue

            # Get face instance to verify it's still unassigned
            face_inst = db_session.get(FaceInstance, face_id_uuid)
            if not face_inst:
                continue

            # Skip if face is assigned (double-check)
            if unassigned_only and face_inst.person_id is not None:
                continue

            # Create suggestion
            suggestion = FaceSuggestion(
                face_instance_id=face_id_uuid,
                suggested_person_id=person_uuid,
                source_face_id=None,  # Centroid-based suggestions don't have a single source face
                confidence=result.score,
                status=FaceSuggestionStatus.PENDING.value,
            )
            db_session.add(suggestion)
            suggestions_created += 1

        db_session.commit()

        logger.info(
            f"[{job_id}] Centroid-based find-more complete for {person.name}: "
            f"{suggestions_created} suggestions created, "
            f"{duplicates_skipped} duplicates skipped"
        )

        # Update final progress
        update_progress(
            "completed",
            4,
            4,
            f"Created {suggestions_created} new suggestions",
        )

        return {
            "status": "completed",
            "suggestions_created": suggestions_created,
            "centroids_used": 1,
            "candidates_found": len(search_results),
            "duplicates_skipped": duplicates_skipped,
        }

    except Exception as e:
        logger.exception(f"[{job_id}] Error in centroid-based find-more job: {e}")
        update_progress("failed", 0, 0, str(e))
        return {"status": "error", "message": str(e)}
    finally:
        db_session.close()


def discover_unknown_persons_job(
    clustering_method: str = "hdbscan",
    min_cluster_size: int = 5,
    min_quality: float = 0.3,
    max_faces: int = 50000,
    min_cluster_confidence: float = 0.70,
    eps: float = 0.5,
    progress_key: str | None = None,
) -> dict[str, Any]:
    """Discover unknown person groups by clustering unassigned faces.

    This job:
    1. Retrieves all unassigned face embeddings from Qdrant
    2. Runs HDBSCAN clustering to find natural groupings
    3. Computes confidence scores for each cluster
    4. Filters clusters by confidence threshold
    5. Updates FaceInstance.cluster_id in database
    6. Caches cluster metadata in Redis

    FORK-SAFETY (macOS): Uses sync database and sync Qdrant client.

    Args:
        clustering_method: Algorithm to use (currently only "hdbscan" supported)
        min_cluster_size: Minimum faces per cluster (HDBSCAN parameter)
        min_quality: Minimum quality_score to include faces (0.0-1.0)
        max_faces: Maximum faces to cluster (memory ceiling)
        min_cluster_confidence: Minimum average pairwise similarity to keep cluster
        eps: Not used for HDBSCAN (reserved for DBSCAN)

    Returns:
        dict with: status, total_faces, clusters_found, noise_count,
        qualifying_groups, filtered_low_confidence

    Raises:
        ValueError: If max_faces exceeded or invalid parameters
    """
    job = get_current_job()
    job_id = job.id if job else "no-job"

    logger.info(
        f"[{job_id}] Starting unknown person discovery: "
        f"method={clustering_method}, min_size={min_cluster_size}, "
        f"min_quality={min_quality}, max_faces={max_faces}, "
        f"min_confidence={min_cluster_confidence}"
    )

    # Validate parameters
    if clustering_method != "hdbscan":
        # Write error to progress if key provided (before worker setup)
        if progress_key:
            try:
                from image_search_service.queue.worker import get_redis
                redis_client = get_redis()
                progress_data = {
                    "phase": "failed",
                    "current": 0,
                    "total": 100,
                    "message": f"Unsupported clustering method: {clustering_method}",
                    "timestamp": datetime.now(UTC).isoformat(),
                }
                redis_client.set(progress_key, json.dumps(progress_data), ex=3600)
            except Exception as e:
                logger.warning(f"[{job_id}] Failed to update progress on validation error: {e}")
        return {
            "status": "error",
            "message": f"Unsupported clustering method: {clustering_method}",
        }

    # Import dependencies
    import uuid as uuid_lib

    try:
        import hdbscan
    except ImportError:
        logger.error("hdbscan not installed, cannot perform clustering")
        # Write error to progress if key provided (before worker setup)
        if progress_key:
            try:
                from image_search_service.queue.worker import get_redis
                redis_client = get_redis()
                progress_data = {
                    "phase": "failed",
                    "current": 0,
                    "total": 100,
                    "message": "hdbscan package not installed",
                    "timestamp": datetime.now(UTC).isoformat(),
                }
                redis_client.set(progress_key, json.dumps(progress_data), ex=3600)
            except Exception as e:
                logger.warning(f"[{job_id}] Failed to update progress on import error: {e}")
        return {
            "status": "error",
            "message": "hdbscan package not installed",
        }

    from image_search_service.queue.worker import get_redis
    from image_search_service.services.face_clustering_service import (
        compute_cluster_confidence_from_embeddings,
    )
    from image_search_service.services.unknown_person_service import compute_membership_hash
    from image_search_service.vector.face_qdrant import get_face_qdrant_client

    db_session = get_sync_session()

    try:
        # Helper function to update progress
        def update_progress(
            phase: str, current: int = 0, total: int = 0, message: str = ""
        ) -> None:
            """Update job progress in Redis for real-time UI updates."""
            if not progress_key:
                logger.warning(f"[{job_id}] No progress_key provided, skipping progress update")
                return
            try:
                redis_client = get_redis()

                progress_data = {
                    "phase": phase,
                    "current": current,
                    "total": total,
                    "message": message,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
                redis_client.set(progress_key, json.dumps(progress_data), ex=3600)
                logger.debug(f"[{job_id}] Progress: {phase} - {message}")
            except Exception as e:
                logger.warning(f"[{job_id}] Failed to update progress: {e}")

        # PHASE 1: Retrieve unassigned face embeddings
        update_progress("retrieval", 0, 100, "Retrieving unassigned face embeddings...")

        qdrant_client = get_face_qdrant_client()
        faces_with_embeddings = qdrant_client.get_unlabeled_faces_with_embeddings(
            quality_threshold=min_quality,
            limit=max_faces,
        )

        total_faces = len(faces_with_embeddings)
        logger.info(f"[{job_id}] Retrieved {total_faces} unassigned faces for clustering")

        if total_faces < min_cluster_size:
            logger.warning(
                f"[{job_id}] Only {total_faces} faces available, "
                f"less than min_cluster_size={min_cluster_size}"
            )
            update_progress(
                "completed",
                100,
                100,
                f"Insufficient faces ({total_faces} < {min_cluster_size})",
            )
            return {
                "status": "completed",
                "total_faces": total_faces,
                "clusters_found": 0,
                "noise_count": 0,
                "qualifying_groups": 0,
                "message": f"Insufficient faces ({total_faces} < {min_cluster_size})",
            }

        update_progress("retrieval", 100, 100, f"Retrieved {total_faces} faces")

        # PHASE 2: Memory ceiling check
        update_progress("memory_check", 0, 100, "Checking memory requirements...")

        # HDBSCAN needs O(N²) memory for distance matrix
        max_clustering_memory_gb = 4
        estimated_memory_gb = (total_faces**2 * 8) / (1024**3)

        if estimated_memory_gb > max_clustering_memory_gb:
            error_msg = (
                f"Memory ceiling exceeded: {estimated_memory_gb:.2f} GB required "
                f"(max: {max_clustering_memory_gb} GB). "
                f"Reduce max_faces (currently {total_faces}) or use sampling."
            )
            logger.error(f"[{job_id}] {error_msg}")
            update_progress("failed", 0, 0, error_msg)
            return {"status": "error", "message": error_msg}

        logger.info(
            f"[{job_id}] Memory check passed: {estimated_memory_gb:.2f} GB "
            f"(max: {max_clustering_memory_gb} GB)"
        )
        update_progress("memory_check", 100, 100, "Memory check passed")

        # PHASE 3: Build embedding matrix
        update_progress("embedding", 0, 100, "Building embedding matrix...")

        face_ids = [face_id for face_id, _ in faces_with_embeddings]
        embeddings = np.array(
            [embedding for _, embedding in faces_with_embeddings], dtype=np.float32
        )

        logger.info(f"[{job_id}] Built embedding matrix: {embeddings.shape}")
        update_progress("embedding", 100, 100, f"Matrix shape: {embeddings.shape}")

        # PHASE 4: Run HDBSCAN clustering
        update_progress("clustering", 0, 100, "Running HDBSCAN clustering...")

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=3,  # More conservative than min_cluster_size
            metric="euclidean",  # Standard for face embeddings
            cluster_selection_method="eom",  # Excess of mass (more stable)
            core_dist_n_jobs=1,  # Single thread (RQ worker context)
        )

        logger.info(f"[{job_id}] Starting HDBSCAN clustering...")
        labels = clusterer.fit_predict(embeddings)
        logger.info(f"[{job_id}] HDBSCAN clustering complete")

        update_progress("clustering", 100, 100, "Clustering complete")

        # PHASE 5: Compute cluster metadata
        update_progress("metadata", 0, 100, "Computing cluster metadata...")

        unique_labels = set(labels)
        noise_count = int(np.sum(labels == -1))  # -1 = noise in HDBSCAN

        logger.info(
            f"[{job_id}] Found {len(unique_labels) - 1} clusters "
            f"(excluding noise: {noise_count} faces)"
        )

        # Compute confidence and membership hash for each cluster
        cluster_metadata: dict[str, dict[str, Any]] = {}
        qualifying_groups = 0
        filtered_low_confidence = 0
        actual_clusters_found = len(unique_labels) - (1 if -1 in unique_labels else 0)

        for label in unique_labels:
            if label == -1:
                continue  # Skip noise

            # Get face indices for this cluster
            cluster_mask = labels == label
            cluster_indices = np.where(cluster_mask)[0]
            cluster_embeddings = embeddings[cluster_indices]

            # Compute confidence
            confidence = compute_cluster_confidence_from_embeddings(
                cluster_embeddings,
                sample_size=20,
            )

            # Get face IDs for this cluster
            cluster_face_ids = [face_ids[i] for i in cluster_indices]
            face_count = len(cluster_face_ids)

            # Namespace cluster_id to avoid collision with dual_clusterer
            cluster_id = f"unknown_{label}"

            # Compute membership hash
            membership_hash = compute_membership_hash(cluster_face_ids)

            # Filter by confidence and size
            if confidence >= min_cluster_confidence and face_count >= min_cluster_size:
                cluster_metadata[cluster_id] = {
                    "cluster_id": cluster_id,
                    "face_count": face_count,
                    "confidence": confidence,
                    "membership_hash": membership_hash,
                    "face_ids": [str(fid) for fid in cluster_face_ids],
                }
                qualifying_groups += 1
                logger.info(
                    f"[{job_id}] Cluster {cluster_id}: "
                    f"{face_count} faces, confidence={confidence:.3f} ✓"
                )
            else:
                filtered_low_confidence += 1
                logger.debug(
                    f"[{job_id}] Cluster {cluster_id}: "
                    f"{face_count} faces, confidence={confidence:.3f} "
                    f"(filtered: confidence < {min_cluster_confidence})"
                )

        update_progress(
            "metadata",
            100,
            100,
            f"Found {qualifying_groups} qualifying groups (filtered {filtered_low_confidence})",
        )

        # PHASE 6: Update FaceInstance.cluster_id in database
        update_progress("persistence", 0, 100, "Updating database...")

        # Build mapping: face_id -> cluster_id
        face_to_cluster: dict[str, str] = {}

        for label, face_id in zip(labels, face_ids):
            if label == -1:
                # Noise faces get special marker
                face_to_cluster[str(face_id)] = "-1"
            else:
                cluster_id = f"unknown_{label}"
                # Only update if cluster qualified (passed confidence filter)
                if cluster_id in cluster_metadata:
                    face_to_cluster[str(face_id)] = cluster_id

        # Batch update FaceInstance records
        updated_count = 0
        for face_id_str, cluster_id in face_to_cluster.items():
            try:
                face_uuid = uuid_lib.UUID(face_id_str)
                stmt = (
                    select(FaceInstance)
                    .where(FaceInstance.id == face_uuid)
                )
                face = db_session.execute(stmt).scalar_one_or_none()

                if face:
                    face.cluster_id = cluster_id
                    updated_count += 1

            except Exception as e:
                logger.warning(f"[{job_id}] Failed to update face {face_id_str}: {e}")
                continue

        db_session.commit()
        logger.info(f"[{job_id}] Updated cluster_id for {updated_count} faces")

        update_progress("persistence", 100, 100, f"Updated {updated_count} faces")

        # PHASE 7: Cache metadata in Redis
        update_progress("caching", 0, 100, "Caching cluster metadata...")

        redis_client = get_redis()

        # Cache last discovery metadata
        discovery_metadata = {
            "timestamp": datetime.now(UTC).isoformat(),
            "total_faces": total_faces,
            "clusters_found": actual_clusters_found,
            "qualifying_groups": qualifying_groups,
            "noise_count": noise_count,
            "params": {
                "clustering_method": clustering_method,
                "min_cluster_size": min_cluster_size,
                "min_quality": min_quality,
                "min_cluster_confidence": min_cluster_confidence,
            },
        }
        redis_client.set(
            "unknown_persons:last_discovery",
            json.dumps(discovery_metadata),
            ex=86400,  # 24h TTL
        )

        # Store discovery params so the read path knows what threshold was used
        redis_client.set(
            "unknown_persons:discovery_params",
            json.dumps({
                "min_cluster_confidence": min_cluster_confidence,
                "timestamp": datetime.now(UTC).isoformat(),
                "total_clusters": actual_clusters_found,
                "total_faces": total_faces,
            }),
            ex=86400,  # 24h TTL, same as cluster metadata
        )

        # Cache individual cluster metadata
        for cluster_id, metadata in cluster_metadata.items():
            redis_client.set(
                f"unknown_persons:cluster:{cluster_id}",
                json.dumps(metadata),
                ex=86400,  # 24h TTL
            )

        logger.info(
            f"[{job_id}] Cached metadata for {len(cluster_metadata)} clusters in Redis"
        )

        update_progress("caching", 100, 100, "Metadata cached")

        # PHASE 8: Report completion
        update_progress(
            "completed",
            100,
            100,
            f"Discovery complete: {qualifying_groups} groups found",
        )

        result = {
            "status": "completed",
            "total_faces": total_faces,
            "clusters_found": actual_clusters_found,
            "noise_count": noise_count,
            "qualifying_groups": qualifying_groups,
            "filtered_low_confidence": filtered_low_confidence,
            "updated_faces": updated_count,
        }

        logger.info(
            f"[{job_id}] Unknown person discovery complete: "
            f"{total_faces} faces, {qualifying_groups} qualifying groups, "
            f"{noise_count} noise"
        )

        return result

    except Exception as e:
        logger.exception(f"[{job_id}] Fatal error in unknown person discovery: {e}")
        return {
            "status": "error",
            "message": str(e),
        }
    finally:
        db_session.close()
