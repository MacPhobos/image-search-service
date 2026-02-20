"""Training job functions for RQ background processing.

FORK-SAFETY REQUIREMENTS FOR MACOS
==================================
All job functions in this module are executed by RQ work-horse subprocesses.
On macOS, these subprocesses are created via spawn() (not fork()), meaning:

1. Job functions MUST use SYNCHRONOUS operations only
   ✓ Use: get_sync_session() for database operations
   ✗ Don't use: async/await, asyncio, greenlets, or async connection pools

2. Why sync-only:
   - Async operations use greenlet state that doesn't survive subprocess spawn
   - Thread state from connection pools can corrupt in child processes
   - GPU state needs clean initialization in each work-horse
   - Single worker + sync ops = predictable, safe execution

3. Current job functions (all sync-compliant):
   ✓ train_session() - Uses sync database operations
   ✓ train_batch() - Uses ThreadPoolExecutor for I/O (not multiprocessing)
   ✓ train_single_asset() - Direct sync operations

4. Memory management (critical for MPS):
   - Explicit gc.collect() every N images
   - Explicit tensor deletion (del + gc.collect())
   - PIL operations protected by module-level lock
   - No EXIF parsing (disabled entirely)

5. GPU model initialization:
   - EmbeddingPreloadWorker preloads in work-horse subprocess
   - Ensures clean GPU state in child process
   - PyTorch MPS high watermark disabled
"""

import gc
import hashlib
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from PIL import Image, ImageFile
from sqlalchemy import select

from image_search_service.core.config import get_settings
from image_search_service.core.device import get_device_info
from image_search_service.core.logging import get_logger
from image_search_service.db.models import JobStatus, SessionStatus, TrainingJob
from image_search_service.db.sync_operations import (
    create_evidence_sync,
    create_training_jobs_sync,
    discover_assets_sync,
    get_asset_by_id_sync,
    get_session_by_id_sync,
    get_sync_session,
    increment_subdirectory_trained_count_sync,
    update_asset_indexed_at_sync,
    update_job_progress_sync,
    update_training_job_sync,
)
from image_search_service.queue.progress import ProgressTracker
from image_search_service.services.embedding import get_embedding_service
from image_search_service.vector.qdrant import (
    ensure_collection,
    upsert_vector,
    upsert_vectors_batch,
)

logger = get_logger(__name__)

# PIL configuration to prevent EXIF parsing crashes
# PIL's TIFF/EXIF parser is not thread-safe and crashes on malformed metadata
# when called from multiple threads simultaneously
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Disable EXIF data loading to prevent segfaults on corrupted EXIF
# PIL's native C EXIF parser crashes with signal 11 when parsing malformed
# EXIF data, especially in multi-threaded contexts. Since EXIF data is not
# needed for embedding generation, we disable it entirely at the class level.
Image.Image._getexif = lambda self: None

# Lock for thread-safe PIL image loading
# PIL's C code is not thread-safe. Even with EXIF disabled, use a lock
# to serialize image operations and prevent race conditions.
_image_load_lock = threading.Lock()

# Sentinel for producer completion
_SENTINEL = object()


@dataclass
class LoadedImageBatch:
    """A batch of images loaded from disk, ready for GPU processing."""

    # List of (job, asset, pil_image) tuples
    items: list[tuple[Any, Any, Image.Image]] = field(default_factory=list)
    # Errors encountered during loading
    errors: list[dict[str, Any]] = field(default_factory=list)
    # Time spent loading this batch
    io_time: float = 0.0
    # Batch index
    batch_index: int = 0


def _load_image_pil(path: str) -> tuple[str, Image.Image | None]:
    """Load image from disk as PIL Image (I/O bound operation).

    Protected against EXIF parsing crashes via serialized PIL operations.
    PIL's TIFF/EXIF parser (native C code) is not thread-safe and crashes
    with segfaults on malformed EXIF data when called from multiple threads
    simultaneously. Uses a module-level lock to serialize image loading,
    preventing race conditions in PIL's C extensions.

    Returns (path, image) where image is None if loading failed.
    """
    try:
        # Serialize PIL operations to prevent thread-safety issues in C code
        # when parsing corrupted EXIF data
        with _image_load_lock:
            img = Image.open(path)
            img.load()  # Load image data into memory
        return (path, img)
    except Exception as e:
        logger.warning(f"Failed to load image {path}: {e}")
        return (path, None)


def train_session(session_id: int) -> dict[str, object]:
    """Main training job that orchestrates batch processing.

    This is the entry point job enqueued by the API.  When the session is in
    DISCOVERING state the worker performs asset discovery and job creation
    synchronously before starting GPU processing.  This keeps the HTTP
    response fast (milliseconds) while all heavy work runs in the background.

    FORK-SAFETY (macOS): This function runs in RQ work-horse subprocess.
    We disable proxy detection immediately to prevent urllib from forking
    in multi-threaded context (which causes Signal 11 crashes).

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
        # ------------------------------------------------------------------
        # Phase 0: Asset discovery (only when session is in DISCOVERING state)
        # ------------------------------------------------------------------
        training_session = get_session_by_id_sync(db_session, session_id)
        if not training_session:
            logger.error(f"Training session {session_id} not found in worker")
            return {
                "status": "failed",
                "session_id": session_id,
                "error": f"Training session {session_id} not found",
            }

        if training_session.status == SessionStatus.DISCOVERING.value:
            logger.info(
                f"Session {session_id} is DISCOVERING — running asset discovery in worker"
            )

            # Check for early cancellation before doing expensive I/O
            if tracker.should_stop(db_session):
                logger.info(
                    f"Session {session_id} was cancelled before discovery started"
                )
                return {
                    "status": "cancelled",
                    "session_id": session_id,
                    "processed": 0,
                    "failed": 0,
                    "message": "Cancelled before discovery",
                }

            # Discover all image assets (filesystem scan + DB inserts).
            # Pass a cancellation callback so the scan can be interrupted
            # every 500 files when the user cancels mid-discovery.
            try:
                assets = discover_assets_sync(
                    db_session,
                    session_id,
                    cancellation_check=lambda: tracker.should_stop(db_session),
                )
            except ValueError as exc:
                logger.error(f"Asset discovery failed for session {session_id}: {exc}")
                training_session = get_session_by_id_sync(db_session, session_id)
                if training_session:
                    training_session.status = SessionStatus.FAILED.value
                    db_session.commit()
                return {
                    "status": "failed",
                    "session_id": session_id,
                    "error": str(exc),
                }

            if not assets:
                logger.warning(f"No assets discovered for session {session_id}")
                training_session = get_session_by_id_sync(db_session, session_id)
                if training_session:
                    training_session.status = SessionStatus.FAILED.value
                    db_session.commit()
                return {
                    "status": "failed",
                    "session_id": session_id,
                    "error": "No assets found for session",
                }

            logger.info(
                f"Discovered {len(assets)} assets for session {session_id}; "
                "creating training jobs"
            )

            # Create TrainingJob records with perceptual-hash deduplication
            job_counts = create_training_jobs_sync(db_session, session_id, assets)

            logger.info(
                f"Created {job_counts['jobs_created']} training jobs for session "
                f"{session_id}: {job_counts['unique']} unique, "
                f"{job_counts['skipped']} skipped"
            )

            # Transition DISCOVERING → RUNNING and set totals / timestamps
            training_session = get_session_by_id_sync(db_session, session_id)
            if training_session:
                training_session.total_images = len(assets)
                training_session.processed_images = 0
                training_session.failed_images = 0
                training_session.status = SessionStatus.RUNNING.value
                training_session.started_at = datetime.now(UTC)
                db_session.commit()

            logger.info(
                f"Session {session_id} transitioned DISCOVERING → RUNNING "
                f"({len(assets)} total images)"
            )

        # ------------------------------------------------------------------
        # Phase 1: Process pending training jobs
        # ------------------------------------------------------------------

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
        # (SessionStatus and get_session_by_id_sync are imported at module level)
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
        # Best-effort DB update: mark session as FAILED so it does not stay
        # stuck in DISCOVERING or RUNNING permanently.
        try:
            recovery_session = get_session_by_id_sync(db_session, session_id)
            if recovery_session and recovery_session.status in (
                SessionStatus.DISCOVERING.value,
                SessionStatus.RUNNING.value,
            ):
                recovery_session.status = SessionStatus.FAILED.value
                db_session.commit()
        except Exception:
            logger.error(
                f"Failed to update session {session_id} status to FAILED during error recovery"
            )
        return {
            "status": "failed",
            "session_id": session_id,
            "error": str(e),
        }
    finally:
        db_session.close()


def train_batch(
    session_id: int,
    asset_ids: list[int],
    batch_num: int,
    gpu_batch_size: int | None = None,
    io_workers: int = 4,
    pipeline_queue_size: int = 4,
) -> dict[str, int | float]:
    """Process a batch of images with pipelined I/O and batched GPU inference.

    Uses producer-consumer pattern:
    - Background thread: loads images in parallel using ThreadPoolExecutor
    - Main thread: batched GPU embedding + Qdrant upserts + DB operations

    Memory Management:
    - Explicit tensor cleanup in embed_images_batch() (embedding.py)
    - Periodic garbage collection during batch processing (configurable interval)
    - Both critical for MPS on macOS to prevent GPU memory accumulation
    - Safe on CUDA (minimal overhead) and CPU (no-op)

    FORK-SAFETY (macOS): Can be called directly as job or from train_session.
    If called directly as job, ensure proxy detection is disabled.

    Args:
        session_id: Training session ID
        asset_ids: List of asset IDs to process
        batch_num: Batch number for logging
        gpu_batch_size: Number of images per GPU batch (from config if None)
        io_workers: Number of I/O threads for parallel image loading (default: 4)
        pipeline_queue_size: Number of batches to buffer (default: 2)

    Returns:
        Dictionary with batch results including timing metrics
    """
    logger.debug(f"Training batch {batch_num} with {len(asset_ids)} assets (pipelined)")
    start_time = time.time()

    db_session = get_sync_session()
    tracker = ProgressTracker(session_id)
    embedding_service = get_embedding_service()
    settings = get_settings()

    # Use config-provided batch size if not specified
    if gpu_batch_size is None:
        gpu_batch_size = settings.gpu_batch_size
        logger.debug(f"Using GPU batch size from config: {gpu_batch_size}")

    # Ensure collection exists (cached - only hits API once per process)
    ensure_collection(embedding_service.embedding_dim)

    # Get training session for category_id
    training_session = get_session_by_id_sync(db_session, session_id)
    category_id = training_session.category_id if training_session else None

    processed = 0
    failed = 0
    total_io_time = 0.0
    total_gpu_time = 0.0

    try:
        # Get all jobs for these assets
        query = (
            select(TrainingJob)
            .where(TrainingJob.session_id == session_id)
            .where(TrainingJob.asset_id.in_(asset_ids))
        )
        result = db_session.execute(query)
        jobs = list(result.scalars().all())

        if not jobs:
            return {"processed": 0, "failed": 0, "io_time": 0.0, "gpu_time": 0.0}

        # Build job lookup and get assets
        job_by_asset_id: dict[int, Any] = {}
        asset_by_id: dict[int, Any] = {}
        job_items: list[tuple[Any, Any, str]] = []  # (job, asset, path)

        for job in jobs:
            asset = get_asset_by_id_sync(db_session, job.asset_id)
            if not asset:
                update_training_job_sync(
                    db_session, job.id, JobStatus.FAILED.value, "Asset not found"
                )
                failed += 1
                continue

            job_by_asset_id[job.asset_id] = job
            asset_by_id[job.asset_id] = asset
            job_items.append((job, asset, asset.path))

        if not job_items:
            return {"processed": 0, "failed": failed, "io_time": 0.0, "gpu_time": 0.0}

        # Create GPU-sized sub-batches
        gpu_batches: list[list[tuple[Any, Any, str]]] = []
        for i in range(0, len(job_items), gpu_batch_size):
            gpu_batches.append(job_items[i : i + gpu_batch_size])

        # Bounded queue for loaded batches
        batch_queue: queue.Queue[LoadedImageBatch | object] = queue.Queue(
            maxsize=pipeline_queue_size
        )

        shutdown_event = threading.Event()
        io_time_accumulator: list[float] = []

        def io_producer() -> None:
            """Background thread: loads images from disk."""
            try:
                with ThreadPoolExecutor(max_workers=io_workers) as executor:
                    for batch_idx, batch in enumerate(gpu_batches):
                        if shutdown_event.is_set():
                            break

                        io_start = time.time()
                        loaded_batch = LoadedImageBatch(batch_index=batch_idx)

                        # Map paths to jobs/assets
                        path_to_item = {path: (job, asset) for job, asset, path in batch}

                        # Load images in parallel
                        futures = {
                            executor.submit(_load_image_pil, path): path
                            for path in path_to_item.keys()
                        }

                        for future in as_completed(futures):
                            if shutdown_event.is_set():
                                break
                            path = futures[future]
                            job, asset = path_to_item[path]
                            try:
                                _, img = future.result()
                                if img is not None:
                                    loaded_batch.items.append((job, asset, img))
                                else:
                                    loaded_batch.errors.append({
                                        "job_id": job.id,
                                        "asset_id": asset.id,
                                        "error": f"Failed to load: {path}",
                                    })
                            except Exception as e:
                                loaded_batch.errors.append({
                                    "job_id": job.id,
                                    "asset_id": asset.id,
                                    "error": str(e),
                                })

                        loaded_batch.io_time = time.time() - io_start
                        io_time_accumulator.append(loaded_batch.io_time)

                        if not shutdown_event.is_set():
                            batch_queue.put(loaded_batch)

            except Exception as e:
                logger.error(f"I/O producer error: {e}")
                shutdown_event.set()
            finally:
                batch_queue.put(_SENTINEL)

        # Start I/O producer in background
        producer_thread = threading.Thread(target=io_producer, name="io-producer")
        producer_thread.start()

        # Process batches in main thread (GPU + DB)
        qdrant_buffer: list[dict[str, Any]] = []
        qdrant_batch_size = 100

        while True:
            # Check cancellation
            if tracker.should_stop(db_session):
                logger.info(f"Batch {batch_num} stopping due to cancellation/pause")
                shutdown_event.set()
                break

            try:
                item = batch_queue.get(timeout=0.1)
            except queue.Empty:
                if shutdown_event.is_set() and batch_queue.empty():
                    break
                continue

            if item is _SENTINEL:
                break

            loaded_batch: LoadedImageBatch = item  # type: ignore[assignment]

            # Handle loading errors
            for err in loaded_batch.errors:
                update_training_job_sync(
                    db_session, err["job_id"], JobStatus.FAILED.value, err["error"]
                )
                failed += 1

            if not loaded_batch.items:
                continue

            # Batch GPU embedding
            gpu_start = time.time()
            images = [img for _, _, img in loaded_batch.items]

            try:
                embeddings = embedding_service.embed_images_batch(images)
            except Exception as e:
                logger.error(f"Batch embedding failed: {e}")
                for job, asset, _ in loaded_batch.items:
                    update_training_job_sync(
                        db_session, job.id, JobStatus.FAILED.value, str(e)
                    )
                    failed += 1
                continue

            gpu_elapsed = time.time() - gpu_start
            total_gpu_time += gpu_elapsed

            # Process each result
            for (job, asset, _), embedding in zip(loaded_batch.items, embeddings):
                try:
                    # Calculate checksum
                    embedding_bytes = "".join(str(v) for v in embedding).encode()
                    checksum = hashlib.sha256(embedding_bytes).hexdigest()

                    # Prepare Qdrant point
                    payload: dict[str, str | int] = {"path": asset.path}
                    if asset.created_at:
                        payload["created_at"] = asset.created_at.isoformat()
                    if category_id is not None:
                        payload["category_id"] = category_id

                    qdrant_buffer.append({
                        "asset_id": asset.id,
                        "vector": embedding,
                        "payload": payload,
                    })

                    # Update job status
                    update_training_job_sync(db_session, job.id, JobStatus.COMPLETED.value)
                    update_asset_indexed_at_sync(db_session, asset.id)

                    # Create evidence (simplified - skip detailed metadata for speed)
                    create_evidence_sync(
                        db_session,
                        {
                            "asset_id": asset.id,
                            "session_id": session_id,
                            "model_name": "OpenCLIP",
                            "model_version": settings.clip_model_name,
                            "embedding_checksum": checksum,
                            "device": embedding_service.device,
                            "processing_time_ms": int(gpu_elapsed * 1000 / len(images)),
                        },
                    )

                    # Increment subdirectory trained count
                    increment_subdirectory_trained_count_sync(
                        db_session, session_id, asset.path
                    )

                    processed += 1

                    # Periodic GPU memory cleanup (critical for MPS on macOS)
                    # Forces garbage collection every N images to free accumulated tensors
                    if (
                        settings.gpu_memory_cleanup_enabled
                        and processed % settings.gpu_memory_cleanup_interval == 0
                    ):
                        gc.collect()
                        logger.debug(
                            f"Periodic garbage collection after {processed} images processed"
                        )

                    # Flush Qdrant buffer when full
                    if len(qdrant_buffer) >= qdrant_batch_size:
                        upsert_vectors_batch(qdrant_buffer)
                        logger.debug(f"Flushed {len(qdrant_buffer)} vectors to Qdrant")
                        qdrant_buffer.clear()

                except Exception as e:
                    logger.error(f"Error processing asset {asset.id}: {e}")
                    update_training_job_sync(
                        db_session, job.id, JobStatus.FAILED.value, str(e)
                    )
                    failed += 1

        # Wait for producer
        producer_thread.join()

        # Flush remaining Qdrant buffer
        if qdrant_buffer:
            upsert_vectors_batch(qdrant_buffer)
            logger.debug(f"Flushed final {len(qdrant_buffer)} vectors to Qdrant")
            qdrant_buffer.clear()

        total_io_time = sum(io_time_accumulator)
        elapsed_time = time.time() - start_time

        # Calculate pipeline efficiency
        sequential_time = total_io_time + total_gpu_time
        if sequential_time > 0 and elapsed_time > 0:
            time_saved = sequential_time - elapsed_time
            overlap_potential = min(total_io_time, total_gpu_time)
            if overlap_potential > 0:
                efficiency = min(1.0, max(0.0, time_saved / overlap_potential))
            else:
                efficiency = 0.0
        else:
            efficiency = 0.0

        throughput = processed / elapsed_time if elapsed_time > 0 else 0.0

        logger.info(
            f"Batch {batch_num} pipeline stats: io_time={total_io_time:.2f}s, "
            f"gpu_time={total_gpu_time:.2f}s, elapsed={elapsed_time:.2f}s, "
            f"efficiency={efficiency:.1%}, throughput={throughput:.1f} img/s"
        )

        return {
            "processed": processed,
            "failed": failed,
            "io_time": total_io_time,
            "gpu_time": total_gpu_time,
            "elapsed_time": elapsed_time,
            "efficiency": efficiency,
        }

    except Exception as e:
        logger.exception(f"Error in train_batch: {e}")
        return {"processed": processed, "failed": failed + len(asset_ids) - processed}

    finally:
        db_session.close()


def train_single_asset(job_id: int, asset_id: int, session_id: int) -> dict[str, object]:
    """Train single image with full evidence capture.

    FORK-SAFETY (macOS): Disable proxy detection immediately to prevent
    urllib from forking in multi-threaded work-horse context.

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

        # Increment subdirectory trained count
        increment_subdirectory_trained_count_sync(db_session, session_id, asset.path)

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
