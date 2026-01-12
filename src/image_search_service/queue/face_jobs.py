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
   ✓ detect_faces_job() - Sync operations
   ✓ cluster_faces_job() - Sync operations
   ✓ assign_faces_job() - Sync operations
   ✓ compute_centroids_job() - Sync operations
   ✓ detect_faces_for_session_job() - Sync operations
   ✓ propagate_person_label_job() - Sync operations
   ✓ expire_old_suggestions_job() - Sync operations

See worker.py for complete macOS fork-safety architecture.
"""

import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

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


def detect_faces_job(
    asset_ids: list[str],
    min_confidence: float = 0.5,
    min_face_size: int = 20,
) -> dict[str, Any]:
    """RQ job to detect and embed faces for a batch of assets.

    FORK-SAFETY (macOS): Disable proxy detection immediately to prevent
    urllib from forking in multi-threaded work-horse context.

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

                clusterer = get_face_clusterer(
                    db_session,
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


def cleanup_orphaned_suggestions_job() -> dict[str, Any]:
    """Clean up orphaned FaceSuggestion records where source face assignments changed.

    Finds and expires all pending suggestions where:
    - source_face.person_id is NULL (face was unassigned), OR
    - source_face.person_id != suggestion.suggested_person_id (face moved to different person)

    This job should be run periodically to fix any orphaned suggestions that
    weren't cleaned up during face assignment changes.

    Returns:
        Dictionary with job results
    """
    from sqlalchemy import and_, or_

    job = get_current_job()
    job_id = job.id if job else "no-job"

    logger.info(f"[{job_id}] Starting cleanup of orphaned face suggestions")

    db_session = get_sync_session()

    try:
        # Find pending suggestions where source face assignment is invalid
        # Join with FaceInstance to check current person assignment
        query = (
            select(FaceSuggestion)
            .join(FaceInstance, FaceSuggestion.source_face_id == FaceInstance.id)
            .where(
                FaceSuggestion.status == FaceSuggestionStatus.PENDING.value,
                or_(
                    # Source face is no longer assigned to any person
                    FaceInstance.person_id.is_(None),
                    # Source face moved to a different person
                    and_(
                        FaceInstance.person_id.isnot(None),
                        FaceInstance.person_id != FaceSuggestion.suggested_person_id,
                    ),
                ),
            )
        )

        result = db_session.execute(query)
        orphaned_suggestions = result.scalars().all()

        expired_count = 0
        for suggestion in orphaned_suggestions:
            suggestion.status = FaceSuggestionStatus.EXPIRED.value
            suggestion.reviewed_at = datetime.now(UTC)
            expired_count += 1

        db_session.commit()

        logger.info(f"[{job_id}] Expired {expired_count} orphaned suggestions")

        return {
            "status": "completed",
            "expired_count": expired_count,
        }

    except Exception as e:
        logger.exception(f"[{job_id}] Error cleaning up orphaned suggestions: {e}")
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

                face_id_str = result.payload.get("face_id")
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
        prototype_embeddings: dict[str, dict[str, Any]] = {}

        for proto in prototypes:
            # Get face instance for quality score
            face = db_session.get(FaceInstance, proto.face_instance_id)

            if face and face.qdrant_point_id:
                embedding = qdrant.get_embedding_by_point_id(face.qdrant_point_id)
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

                face_id_str = result.payload.get("face_id")
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
