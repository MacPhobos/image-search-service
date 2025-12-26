"""RQ background jobs for face detection, clustering, and assignment."""

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

        session.status = FaceDetectionSessionStatus.PROCESSING.value
        session.started_at = datetime.now()
        db_session.commit()

        # Get assets to process
        logger.info(f"[{job_id}] Getting assets to process")

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

        for i in range(0, len(assets), batch_size):
            batch = assets[i : i + batch_size]
            asset_ids = [a.id for a in batch]

            logger.info(
                f"[{job_id}] Processing batch {i // batch_size + 1} "
                f"({len(asset_ids)} assets)"
            )

            try:
                result = face_service.process_assets_batch(
                    asset_ids=asset_ids,
                    min_confidence=session.min_confidence,
                    min_face_size=session.min_face_size,
                    batch_size=session.batch_size,
                )

                # Update session progress
                total_faces_detected += result["total_faces"]
                total_failed += result["errors"]

                session.processed_images += result["processed"]
                session.faces_detected = total_faces_detected
                session.failed_images = total_failed

                if result.get("error_details"):
                    last_error = f"Batch errors: {result['error_details'][0]['error']}"
                    session.last_error = last_error

                db_session.commit()

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
                db_session.commit()

        # Auto-assign faces to known persons
        logger.info(f"[{job_id}] Auto-assigning faces to known persons")
        faces_assigned = 0

        try:
            assigner = get_face_assigner(
                db_session=db_session,
                similarity_threshold=0.6,  # Use default threshold
            )

            # Assign faces created during this session
            assignment_result = assigner.assign_new_faces(
                since=session.started_at,
                max_faces=10000,  # Process all new faces
            )

            faces_assigned = assignment_result.get("assigned", 0)
            session.faces_assigned = faces_assigned

            logger.info(
                f"[{job_id}] Auto-assignment complete: {faces_assigned} faces assigned"
            )

        except Exception as e:
            logger.error(f"[{job_id}] Auto-assignment error: {e}")
            # Don't fail the whole session for assignment errors
            session.last_error = f"Assignment error: {e}"

        # Mark session as complete
        session.status = FaceDetectionSessionStatus.COMPLETED.value
        session.completed_at = datetime.now()
        db_session.commit()

        logger.info(
            f"[{job_id}] Session {session_id} complete: "
            f"{session.processed_images} processed, {total_faces_detected} faces, "
            f"{faces_assigned} assigned, {total_failed} failed"
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
    min_confidence: float = 0.7,
    max_suggestions: int = 50,
) -> dict[str, Any]:
    """Find similar faces and create suggestions when a face is labeled to a person.

    This job is triggered when a user assigns a face to a person. It searches for
    similar unassigned faces and creates FaceSuggestion records for user review.

    Args:
        source_face_id: UUID string of the face that was just labeled
        person_id: UUID string of the person the face was assigned to
        min_confidence: Minimum cosine similarity to create suggestion (default 0.7)
        max_suggestions: Maximum number of suggestions to create (default 50)

    Returns:
        Dictionary with job results
    """
    import uuid as uuid_lib

    from image_search_service.vector.face_qdrant import get_face_qdrant_client

    job = get_current_job()
    job_id = job.id if job else "no-job"

    # Convert string UUIDs to UUID objects
    source_face_uuid = uuid_lib.UUID(source_face_id)
    person_uuid = uuid_lib.UUID(person_id)

    logger.info(f"[{job_id}] Starting propagation for face {source_face_id} → person {person_id}")

    db_session = get_sync_session()

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
            face_id = uuid_lib.UUID(str(result.id))
            confidence = result.score

            # Skip the source face itself
            if face_id == source_face_uuid:
                continue

            faces_checked += 1

            # Get the face instance
            face = db_session.get(FaceInstance, face_id)
            if not face:
                continue

            # Skip if already assigned to a person
            if face.person_id is not None:
                continue

            # Skip if suggestion already exists
            existing = db_session.execute(
                select(FaceSuggestion).where(
                    FaceSuggestion.face_instance_id == face_id,
                    FaceSuggestion.suggested_person_id == person_uuid,
                    FaceSuggestion.status == FaceSuggestionStatus.PENDING.value,
                )
            ).scalar_one_or_none()

            if existing:
                continue

            # Create suggestion
            suggestion = FaceSuggestion(
                face_instance_id=face_id,
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
    days_threshold: int = 30,
) -> dict[str, Any]:
    """Expire pending suggestions older than the threshold.

    This job should be run periodically (e.g., daily) to clean up
    stale suggestions that were never reviewed.

    Args:
        days_threshold: Number of days after which to expire suggestions

    Returns:
        Dictionary with job results
    """
    job = get_current_job()
    job_id = job.id if job else "no-job"

    logger.info(f"[{job_id}] Expiring suggestions older than {days_threshold} days")

    db_session = get_sync_session()

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
