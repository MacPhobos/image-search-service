"""Training job functions for RQ background processing."""

import hashlib
import time
from datetime import UTC, datetime

from sqlalchemy import select

from image_search_service.core.config import get_settings
from image_search_service.core.device import get_device_info
from image_search_service.core.logging import get_logger
from image_search_service.db.models import JobStatus, TrainingJob
from image_search_service.db.sync_operations import (
    create_evidence_sync,
    get_asset_by_id_sync,
    get_session_by_id_sync,
    get_sync_session,
    update_asset_indexed_at_sync,
    update_job_progress_sync,
    update_training_job_sync,
)
from image_search_service.queue.progress import ProgressTracker
from image_search_service.services.embedding import get_embedding_service
from image_search_service.vector.qdrant import ensure_collection, upsert_vector

logger = get_logger(__name__)


def train_session(session_id: int) -> dict[str, object]:
    """Main training job that orchestrates batch processing.

    This is the entry point job enqueued by the API. It discovers all assets,
    creates TrainingJob records, and processes them in batches.

    Args:
        session_id: Training session ID

    Returns:
        Dictionary with summary statistics
    """
    logger.info(f"Starting training session {session_id}")
    start_time = datetime.now(UTC)

    db_session = get_sync_session()
    tracker = ProgressTracker(session_id)
    settings = get_settings()

    try:
        # Get all pending jobs for this session
        query = (
            select(TrainingJob)
            .where(TrainingJob.session_id == session_id)
            .where(TrainingJob.status == JobStatus.PENDING.value)
        )
        result = db_session.execute(query)
        pending_jobs = list(result.scalars().all())

        if not pending_jobs:
            logger.warning(f"No pending jobs found for session {session_id}")
            return {
                "status": "completed",
                "session_id": session_id,
                "processed": 0,
                "failed": 0,
                "message": "No pending jobs",
            }

        logger.info(f"Found {len(pending_jobs)} pending jobs for session {session_id}")

        # Process jobs in batches
        batch_size = settings.training_batch_size
        total_jobs = len(pending_jobs)
        processed_count = 0
        failed_count = 0

        for batch_num in range(0, total_jobs, batch_size):
            # Check if session was cancelled or paused
            if tracker.should_stop(db_session):
                status = "cancelled" if tracker.check_cancelled(db_session) else "paused"
                logger.info(f"Session {session_id} was {status}, stopping processing")
                return {
                    "status": status,
                    "session_id": session_id,
                    "processed": processed_count,
                    "failed": failed_count,
                    "message": f"Processing {status}",
                }

            # Get batch of jobs
            batch_end = min(batch_num + batch_size, total_jobs)
            batch_jobs = pending_jobs[batch_num:batch_end]
            batch_asset_ids = [job.asset_id for job in batch_jobs]

            logger.info(
                f"Processing batch {batch_num // batch_size + 1}: "
                f"jobs {batch_num + 1}-{batch_end} of {total_jobs}"
            )

            # Process batch
            batch_result = train_batch(
                session_id, batch_asset_ids, batch_num // batch_size + 1
            )

            processed_count += batch_result.get("processed", 0)
            failed_count += batch_result.get("failed", 0)

            # Update progress
            tracker.update_progress(db_session, processed_count, failed_count)

            # Log progress
            progress = tracker.get_current_progress(db_session)
            rate = tracker.calculate_rate(start_time, processed_count)
            logger.info(
                f"Session {session_id} progress: {progress['percentage']}% "
                f"({processed_count}/{total_jobs}), rate: {rate} img/min"
            )

        # Calculate final statistics
        elapsed = (datetime.now(UTC) - start_time).total_seconds()
        rate = tracker.calculate_rate(start_time, processed_count)

        logger.info(
            f"Completed training session {session_id}: "
            f"{processed_count} processed, {failed_count} failed, "
            f"elapsed: {elapsed:.2f}s, rate: {rate} img/min"
        )

        # Mark training session as completed
        from image_search_service.db.models import SessionStatus
        from image_search_service.db.sync_operations import get_session_by_id_sync

        training_session = get_session_by_id_sync(db_session, session_id)
        if training_session and processed_count > 0:
            training_session.status = SessionStatus.COMPLETED.value
            training_session.completed_at = datetime.now(UTC)
            db_session.commit()
            logger.info(f"Marked training session {session_id} as completed")

            # Auto-trigger face detection if we successfully processed images
            try:
                # Check if face detection session already exists for this training session
                from image_search_service.db.models import (
                    FaceDetectionSession,
                    FaceDetectionSessionStatus,
                )
                from image_search_service.queue.worker import get_queue

                existing_query = select(FaceDetectionSession).where(
                    FaceDetectionSession.training_session_id == session_id
                )
                existing_result = db_session.execute(existing_query)
                existing_face_session = existing_result.scalar_one_or_none()

                if existing_face_session:
                    logger.info(
                        f"Face detection session {existing_face_session.id} already exists "
                        f"for training session {session_id}, skipping auto-trigger"
                    )
                else:
                    # Create face detection session
                    face_session = FaceDetectionSession(
                        training_session_id=session_id,
                        status=FaceDetectionSessionStatus.PENDING.value,
                        min_confidence=0.5,
                        min_face_size=20,
                        batch_size=16,
                    )
                    db_session.add(face_session)
                    db_session.commit()
                    db_session.refresh(face_session)

                    # Enqueue face detection job
                    from image_search_service.queue.face_jobs import (
                        detect_faces_for_session_job,
                    )

                    queue = get_queue("default")
                    face_job = queue.enqueue(
                        detect_faces_for_session_job,
                        str(face_session.id),
                        job_timeout=86400,  # 24 hours
                    )

                    face_session.job_id = face_job.id
                    db_session.commit()

                    logger.info(
                        f"Auto-triggered face detection session {face_session.id} "
                        f"(job {face_job.id}) for completed training session {session_id}"
                    )

            except Exception as e:
                # Don't fail the training completion if face detection trigger fails
                logger.error(
                    f"Failed to auto-trigger face detection for training session {session_id}: {e}"
                )

        return {
            "status": "completed",
            "session_id": session_id,
            "processed": processed_count,
            "failed": failed_count,
            "elapsed_seconds": round(elapsed, 2),
            "rate_per_minute": rate,
        }

    except Exception as e:
        logger.exception(f"Error in training session {session_id}: {e}")
        return {
            "status": "failed",
            "session_id": session_id,
            "error": str(e),
        }
    finally:
        db_session.close()


def train_batch(session_id: int, asset_ids: list[int], batch_num: int) -> dict[str, int]:
    """Process a batch of images.

    Args:
        session_id: Training session ID
        asset_ids: List of asset IDs to process
        batch_num: Batch number for logging

    Returns:
        Dictionary with batch results
    """
    logger.debug(f"Training batch {batch_num} with {len(asset_ids)} assets")

    db_session = get_sync_session()
    tracker = ProgressTracker(session_id)

    processed = 0
    failed = 0

    try:
        # Get all jobs for these assets
        query = (
            select(TrainingJob)
            .where(TrainingJob.session_id == session_id)
            .where(TrainingJob.asset_id.in_(asset_ids))
        )
        result = db_session.execute(query)
        jobs = list(result.scalars().all())

        # Process each job
        for job in jobs:
            # Check cancellation before each asset
            if tracker.should_stop(db_session):
                logger.info(f"Batch {batch_num} stopped due to session cancellation/pause")
                break

            try:
                # Train single asset
                result_dict = train_single_asset(job.id, job.asset_id, session_id)

                if result_dict.get("status") == "success":
                    processed += 1
                else:
                    failed += 1

            except Exception as e:
                logger.error(f"Error training asset {job.asset_id}: {e}")
                failed += 1

                # Update job with error
                update_training_job_sync(
                    db_session, job.id, JobStatus.FAILED.value, str(e)
                )

        logger.debug(f"Batch {batch_num} completed: {processed} success, {failed} failed")

        return {"processed": processed, "failed": failed}

    finally:
        db_session.close()


def train_single_asset(job_id: int, asset_id: int, session_id: int) -> dict[str, object]:
    """Train single image with full evidence capture.

    Args:
        job_id: Training job ID
        asset_id: Image asset ID
        session_id: Training session ID

    Returns:
        Dictionary with training result
    """
    start_time = time.time()
    db_session = get_sync_session()
    embedding_service = get_embedding_service()

    try:
        # Update job status to running
        update_training_job_sync(db_session, job_id, JobStatus.RUNNING.value)

        # Get asset from database
        asset = get_asset_by_id_sync(db_session, asset_id)
        if not asset:
            error_msg = f"Asset {asset_id} not found"
            logger.error(error_msg)
            update_training_job_sync(db_session, job_id, JobStatus.FAILED.value, error_msg)
            return {"status": "error", "message": error_msg}

        # Get training session to access category_id
        training_session = get_session_by_id_sync(db_session, session_id)
        category_id = training_session.category_id if training_session else None

        # Ensure Qdrant collection exists
        ensure_collection(embedding_service.embedding_dim)

        # Generate embedding
        logger.debug(f"Generating embedding for {asset.path}")
        embedding_start = time.time()

        try:
            vector = embedding_service.embed_image(asset.path)
        except Exception as e:
            error_msg = f"Failed to embed image {asset.path}: {e}"
            logger.error(error_msg)
            update_training_job_sync(db_session, job_id, JobStatus.FAILED.value, error_msg)

            # Create evidence for failure with metadata
            failure_metadata = _build_evidence_metadata(
                asset=asset,
                vector=None,
                embedding_time_ms=int((time.time() - embedding_start) * 1000),
                total_time_ms=int((time.time() - start_time) * 1000),
                embedding_service=embedding_service,
            )

            create_evidence_sync(
                db_session,
                {
                    "asset_id": asset_id,
                    "session_id": session_id,
                    "model_name": "OpenCLIP",
                    "model_version": get_settings().clip_model_name,
                    "device": embedding_service.device,
                    "processing_time_ms": int((time.time() - embedding_start) * 1000),
                    "error_message": error_msg,
                    "metadata_json": failure_metadata,
                },
            )

            return {"status": "error", "message": error_msg}

        embedding_time_ms = int((time.time() - embedding_start) * 1000)

        # Calculate embedding checksum
        embedding_bytes = "".join(str(v) for v in vector).encode()
        checksum = hashlib.sha256(embedding_bytes).hexdigest()

        # Upsert to Qdrant
        payload: dict[str, str | int] = {"path": asset.path}
        if asset.created_at:
            payload["created_at"] = asset.created_at.isoformat()
        if category_id is not None:
            payload["category_id"] = category_id

        upsert_vector(
            asset_id=asset.id,
            vector=vector,
            payload=payload,
        )

        # Update asset indexed_at timestamp
        update_asset_indexed_at_sync(db_session, asset_id)

        # Calculate total processing time
        total_time_ms = int((time.time() - start_time) * 1000)

        # Update job status and timing
        update_job_progress_sync(db_session, job_id, 100, total_time_ms)
        update_training_job_sync(db_session, job_id, JobStatus.COMPLETED.value)

        # Build comprehensive metadata
        metadata = _build_evidence_metadata(
            asset=asset,
            vector=vector,
            embedding_time_ms=embedding_time_ms,
            total_time_ms=total_time_ms,
            embedding_service=embedding_service,
        )

        # Create evidence record
        create_evidence_sync(
            db_session,
            {
                "asset_id": asset_id,
                "session_id": session_id,
                "model_name": "OpenCLIP",
                "model_version": get_settings().clip_model_name,
                "embedding_checksum": checksum,
                "device": embedding_service.device,
                "processing_time_ms": embedding_time_ms,
                "metadata_json": metadata,
            },
        )

        logger.debug(
            f"Successfully trained asset {asset_id} in {total_time_ms}ms "
            f"(embedding: {embedding_time_ms}ms)"
        )

        return {
            "status": "success",
            "asset_id": asset_id,
            "processing_time_ms": total_time_ms,
        }

    except Exception as e:
        error_msg = f"Unexpected error training asset {asset_id}: {e}"
        logger.exception(error_msg)

        # Update job with error
        update_training_job_sync(db_session, job_id, JobStatus.FAILED.value, error_msg)

        return {"status": "error", "message": error_msg}

    finally:
        db_session.close()


def _build_evidence_metadata(
    asset: object,
    vector: list[float] | None,
    embedding_time_ms: int,
    total_time_ms: int,
    embedding_service: object,
) -> dict[str, object]:
    """Build comprehensive metadata for evidence record.

    Args:
        asset: ImageAsset object
        vector: Embedding vector (None if failed)
        embedding_time_ms: Time to generate embedding
        total_time_ms: Total processing time
        embedding_service: EmbeddingService instance

    Returns:
        Dictionary with structured metadata
    """
    import math
    import sys

    # Image metadata
    image_meta: dict[str, object] = {}
    if hasattr(asset, "width") and asset.width:
        image_meta["width"] = asset.width
    if hasattr(asset, "height") and asset.height:
        image_meta["height"] = asset.height
    if hasattr(asset, "file_size") and asset.file_size:
        image_meta["file_size"] = asset.file_size
    if hasattr(asset, "mime_type") and asset.mime_type:
        image_meta["mime_type"] = asset.mime_type

    # Embedding metadata
    embedding_meta: dict[str, object] = {}
    if vector is not None:
        embedding_meta["dimension"] = len(vector)
        # Calculate L2 norm
        norm = math.sqrt(sum(v * v for v in vector))
        embedding_meta["norm"] = round(norm, 6)
        embedding_meta["generation_time_ms"] = embedding_time_ms

    # Environment metadata
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    py_version += f".{sys.version_info.micro}"
    environment_meta: dict[str, object] = {
        "python_version": py_version,
    }

    try:
        device_info = get_device_info()
        environment_meta.update({
            "device": device_info["selected_device"],
            "cuda_available": device_info["cuda_available"],
            "mps_available": device_info.get("mps_available", False),
            "platform": device_info["platform"],
            "machine": device_info["machine"],
            "pytorch_version": device_info["pytorch_version"],
        })
        if device_info["cuda_available"]:
            environment_meta["gpu_name"] = device_info.get("cuda_device_name")
            environment_meta["cuda_version"] = device_info.get("cuda_version")
    except Exception as e:
        logger.warning(f"Failed to capture device info: {e}")
        environment_meta["device"] = "unknown"

    # Combine all metadata
    metadata: dict[str, object] = {
        "image": image_meta,
        "embedding": embedding_meta,
        "environment": environment_meta,
        "timing": {
            "embedding_time_ms": embedding_time_ms,
            "total_time_ms": total_time_ms,
            "overhead_ms": total_time_ms - embedding_time_ms,
        },
    }

    return metadata
