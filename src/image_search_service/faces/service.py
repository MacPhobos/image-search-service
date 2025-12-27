"""High-level face processing service."""

import logging
import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session as SyncSession

from image_search_service.db.models import FaceInstance, ImageAsset
from image_search_service.faces.detector import detect_faces, detect_faces_from_path
from image_search_service.vector.face_qdrant import get_face_qdrant_client

logger = logging.getLogger(__name__)


# Sentinel value to signal producer completion
_SENTINEL = object()


@dataclass
class LoadedBatch:
    """A batch of images loaded from disk, ready for GPU processing."""

    # Map of image_path -> (asset, loaded_image)
    images: dict[str, tuple["ImageAsset", "np.ndarray[Any, Any]"]] = field(
        default_factory=dict
    )
    # Errors encountered during loading
    errors: list[dict[str, Any]] = field(default_factory=list)
    # Time spent loading this batch (I/O time)
    io_time: float = 0.0
    # Batch index for ordering
    batch_index: int = 0


@dataclass
class ProcessedBatch:
    """Results from GPU processing a batch."""

    # Number of assets successfully processed
    processed: int = 0
    # Total faces detected
    total_faces: int = 0
    # Errors encountered during processing
    errors: list[dict[str, Any]] = field(default_factory=list)
    # GPU processing time for this batch
    gpu_time: float = 0.0
    # Qdrant points to buffer
    qdrant_points: list[dict[str, Any]] = field(default_factory=list)
    # FaceInstance records created (for DB commit)
    face_instances: list["FaceInstance"] = field(default_factory=list)


def _load_image(image_path: str) -> tuple[str, "np.ndarray[Any, Any] | None"]:
    """Load image from disk (I/O bound operation).

    This is a module-level function to work with ThreadPoolExecutor.
    Returns (path, image_array) where image_array is None if loading failed.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"cv2.imread returned None for: {image_path}")
            return (image_path, None)
        return (image_path, image)
    except Exception as e:
        logger.error(f"Failed to load image {image_path}: {e}")
        return (image_path, None)


class FaceProcessingService:
    """Orchestrates face detection, embedding, and storage."""

    def __init__(self, db_session: SyncSession):
        self.db = db_session
        self.qdrant = get_face_qdrant_client()

    def process_asset(
        self,
        asset: ImageAsset,
        min_confidence: float = 0.5,
        min_face_size: int = 20,
    ) -> list[FaceInstance]:
        """Process a single asset: detect faces, store in DB and Qdrant.

        Returns list of created/existing FaceInstance records.
        Idempotent: re-running won't create duplicates.
        """
        image_path = self._resolve_asset_path(asset)
        if not image_path or not Path(image_path).exists():
            logger.warning(f"Asset {asset.id} has no valid image path")
            return []

        # Detect faces
        detected = detect_faces_from_path(
            str(image_path),
            min_confidence=min_confidence,
            min_face_size=min_face_size,
        )

        if not detected:
            logger.debug(f"No faces detected in asset {asset.id}")
            return []

        face_instances = []
        qdrant_points = []

        for face in detected:
            # Check if this face already exists (idempotency)
            existing = self._find_existing_face(asset.id, face.bbox)
            if existing:
                face_instances.append(existing)
                continue

            # Create new FaceInstance
            point_id = uuid.uuid4()
            quality_score = face.compute_quality_score()

            face_instance = FaceInstance(
                id=uuid.uuid4(),
                asset_id=asset.id,
                bbox_x=face.bbox[0],
                bbox_y=face.bbox[1],
                bbox_w=face.bbox[2],
                bbox_h=face.bbox[3],
                landmarks=face.landmarks_as_dict(),
                detection_confidence=face.confidence,
                quality_score=quality_score,
                qdrant_point_id=point_id,
            )

            self.db.add(face_instance)
            face_instances.append(face_instance)

            # Prepare Qdrant point
            qdrant_point: dict[str, Any] = {
                "point_id": point_id,
                "embedding": face.embedding.tolist(),
                "asset_id": asset.id,
                "face_instance_id": face_instance.id,
                "detection_confidence": face.confidence,
                "quality_score": quality_score,
                "bbox": {
                    "x": face.bbox[0],
                    "y": face.bbox[1],
                    "w": face.bbox[2],
                    "h": face.bbox[3],
                },
            }

            # Add optional taken_at if available
            if hasattr(asset, "file_modified_at") and asset.file_modified_at:
                qdrant_point["taken_at"] = asset.file_modified_at

            qdrant_points.append(qdrant_point)

        # Batch upsert to Qdrant FIRST (before commit)
        if qdrant_points:
            try:
                self.qdrant.upsert_faces_batch(qdrant_points)
                logger.info(f"Stored {len(qdrant_points)} new faces for asset {asset.id}")
            except Exception as e:
                # Rollback DB if Qdrant fails
                self.db.rollback()
                logger.error(f"Failed to upsert faces to Qdrant for asset {asset.id}: {e}")
                raise RuntimeError(f"Face detection failed: Qdrant upsert error: {e}") from e

        # Commit DB changes ONLY if Qdrant succeeded
        self.db.commit()

        return face_instances

    def process_assets_batch(
        self,
        asset_ids: list[int],
        min_confidence: float = 0.5,
        min_face_size: int = 20,
        prefetch_batch_size: int = 8,
        io_workers: int = 4,
        qdrant_batch_size: int = 100,
        progress_callback: Any = None,
        pipeline_queue_size: int = 2,
    ) -> dict[str, Any]:
        """Process multiple assets with pipelined I/O and GPU processing.

        Uses a producer-consumer pattern where I/O runs in background thread:
        - Background thread: loads images in parallel using ThreadPoolExecutor
        - Main thread: processes loaded batches through GPU inference + DB operations
        - Bounded queue ensures GPU never starves while limiting memory usage

        This achieves true overlap: while GPU processes batch N, I/O loads batch N+1.
        DB operations stay in main thread for SQLAlchemy thread-safety.

        Args:
            asset_ids: List of asset IDs to process
            min_confidence: Minimum detection confidence threshold
            min_face_size: Minimum face width/height in pixels
            prefetch_batch_size: Number of images per batch (default: 8)
            io_workers: Number of I/O threads for parallel image loading (default: 4)
            qdrant_batch_size: Buffer size for cross-asset Qdrant upserts (default: 100)
            progress_callback: Optional callable(step: int) to report progress
            pipeline_queue_size: Number of batches to buffer between I/O and GPU (default: 2)

        Returns:
            Summary dict with counts, throughput, and timing metrics:
            - processed: Number of assets processed
            - total_faces: Total faces detected
            - errors: Number of errors
            - throughput: Assets per second
            - elapsed_time: Total wall clock time
            - io_time: Total I/O time (loading images from disk)
            - gpu_time: Total GPU time (face detection)
            - pipeline_efficiency: Ratio of overlapped time (higher is better)
        """
        if not asset_ids:
            return {
                "processed": 0,
                "total_faces": 0,
                "errors": 0,
                "throughput": 0.0,
                "io_time": 0.0,
                "gpu_time": 0.0,
                "elapsed_time": 0.0,
                "pipeline_efficiency": 0.0,
            }

        # Special case: prefetch_batch_size=1 means sequential processing (backward compatible)
        if prefetch_batch_size == 1:
            result = self._process_assets_sequential(
                asset_ids, min_confidence, min_face_size, progress_callback
            )
            result["pipeline_efficiency"] = 0.0
            return result

        start_time = time.time()

        # Pre-fetch all assets from database (relatively fast, not the bottleneck)
        assets: list[ImageAsset] = []
        db_errors: list[dict[str, Any]] = []
        for asset_id in asset_ids:
            asset = self.db.get(ImageAsset, asset_id)
            if asset:
                assets.append(asset)
            else:
                db_errors.append({"asset_id": asset_id, "error": "Asset not found"})
                logger.warning(f"Asset {asset_id} not found")

        if not assets:
            elapsed_time = time.time() - start_time
            return {
                "processed": 0,
                "total_faces": 0,
                "errors": len(db_errors),
                "error_details": db_errors,
                "throughput": 0.0,
                "elapsed_time": elapsed_time,
                "io_time": 0.0,
                "gpu_time": 0.0,
                "pipeline_efficiency": 0.0,
            }

        # Create batches of assets with their paths
        batches: list[list[tuple[ImageAsset, str]]] = []
        current_batch: list[tuple[ImageAsset, str]] = []

        for asset in assets:
            try:
                image_path = self._resolve_asset_path(asset)
                if not image_path or not Path(image_path).exists():
                    db_errors.append({"asset_id": asset.id, "error": "Invalid image path"})
                    logger.warning(f"Asset {asset.id} has no valid image path")
                    continue
                current_batch.append((asset, str(image_path)))
                if len(current_batch) >= prefetch_batch_size:
                    batches.append(current_batch)
                    current_batch = []
            except Exception as e:
                db_errors.append({"asset_id": asset.id, "error": str(e)})
                logger.error(f"Cannot resolve path for asset {asset.id}: {e}")

        if current_batch:
            batches.append(current_batch)

        if not batches:
            elapsed_time = time.time() - start_time
            return {
                "processed": 0,
                "total_faces": 0,
                "errors": len(db_errors),
                "error_details": db_errors,
                "throughput": 0.0,
                "elapsed_time": elapsed_time,
                "io_time": 0.0,
                "gpu_time": 0.0,
                "pipeline_efficiency": 0.0,
            }

        # Bounded queue for loaded batches (controls memory usage)
        batch_queue: queue.Queue[LoadedBatch | object] = queue.Queue(
            maxsize=pipeline_queue_size
        )

        # Shared state for error handling and shutdown
        shutdown_event = threading.Event()
        producer_error: list[Exception] = []  # List to store exception from producer
        io_time_accumulator: list[float] = []  # Thread-safe accumulator for I/O times

        def io_producer() -> None:
            """Background thread: loads batches of images from disk."""
            try:
                with ThreadPoolExecutor(max_workers=io_workers) as executor:
                    for batch_idx, batch in enumerate(batches):
                        if shutdown_event.is_set():
                            break

                        io_start = time.time()
                        loaded_batch = LoadedBatch(batch_index=batch_idx)

                        # Submit all image loads for this batch
                        path_to_asset = {path: asset for asset, path in batch}
                        futures = {
                            executor.submit(_load_image, path): path
                            for path in path_to_asset.keys()
                        }

                        # Collect results as they complete
                        for future in as_completed(futures):
                            if shutdown_event.is_set():
                                break
                            path = futures[future]
                            asset = path_to_asset[path]
                            try:
                                _, image = future.result()
                                if image is not None:
                                    loaded_batch.images[path] = (asset, image)
                                else:
                                    loaded_batch.errors.append(
                                        {
                                            "asset_id": asset.id,
                                            "error": f"Failed to load image: {path}",
                                        }
                                    )
                            except Exception as e:
                                loaded_batch.errors.append(
                                    {"asset_id": asset.id, "error": str(e)}
                                )

                        loaded_batch.io_time = time.time() - io_start
                        io_time_accumulator.append(loaded_batch.io_time)

                        # Put batch in queue (blocks if queue is full - backpressure)
                        if not shutdown_event.is_set():
                            batch_queue.put(loaded_batch)

            except Exception as e:
                logger.error(f"Producer thread error: {e}")
                producer_error.append(e)
                shutdown_event.set()
            finally:
                # Signal completion to consumer
                batch_queue.put(_SENTINEL)

        # Start I/O producer in background
        producer_thread = threading.Thread(target=io_producer, name="io-producer")
        producer_thread.start()

        # Process batches in main thread (GPU + DB operations)
        total_gpu_time = 0.0
        total_processed = 0
        total_faces = 0
        all_errors: list[dict[str, Any]] = list(db_errors)
        qdrant_buffer: list[dict[str, Any]] = []

        while True:
            try:
                # Get next batch (blocks until available, with timeout to check shutdown)
                item = batch_queue.get(timeout=0.1)
            except queue.Empty:
                if shutdown_event.is_set() and batch_queue.empty():
                    break
                continue

            # Check for sentinel (producer done)
            if item is _SENTINEL:
                break

            if shutdown_event.is_set():
                break

            loaded_batch: LoadedBatch = item  # type: ignore[assignment]

            # Collect errors from loading phase
            all_errors.extend(loaded_batch.errors)

            # Process each loaded image through GPU (main thread - thread-safe for DB)
            for path, (asset, image) in loaded_batch.images.items():
                if shutdown_event.is_set():
                    break

                try:
                    gpu_start = time.time()
                    detected = detect_faces(image, min_confidence, min_face_size)
                    gpu_elapsed = time.time() - gpu_start
                    total_gpu_time += gpu_elapsed

                    if not detected:
                        logger.debug(f"No faces detected in asset {asset.id}")
                        total_processed += 1
                        if progress_callback:
                            progress_callback(1)
                        continue

                    # Create FaceInstance records and buffer Qdrant points
                    for face in detected:
                        # Check if this face already exists (idempotency)
                        existing = self._find_existing_face(asset.id, face.bbox)
                        if existing:
                            continue

                        # Create new FaceInstance
                        point_id = uuid.uuid4()
                        quality_score = face.compute_quality_score()

                        face_instance = FaceInstance(
                            id=uuid.uuid4(),
                            asset_id=asset.id,
                            bbox_x=face.bbox[0],
                            bbox_y=face.bbox[1],
                            bbox_w=face.bbox[2],
                            bbox_h=face.bbox[3],
                            landmarks=face.landmarks_as_dict(),
                            detection_confidence=face.confidence,
                            quality_score=quality_score,
                            qdrant_point_id=point_id,
                        )

                        self.db.add(face_instance)
                        total_faces += 1

                        # Prepare Qdrant point
                        qdrant_point: dict[str, Any] = {
                            "point_id": point_id,
                            "embedding": face.embedding.tolist(),
                            "asset_id": asset.id,
                            "face_instance_id": face_instance.id,
                            "detection_confidence": face.confidence,
                            "quality_score": quality_score,
                            "bbox": {
                                "x": face.bbox[0],
                                "y": face.bbox[1],
                                "w": face.bbox[2],
                                "h": face.bbox[3],
                            },
                        }

                        if hasattr(asset, "file_modified_at") and asset.file_modified_at:
                            qdrant_point["taken_at"] = asset.file_modified_at

                        qdrant_buffer.append(qdrant_point)

                        # Flush when buffer is full
                        if len(qdrant_buffer) >= qdrant_batch_size:
                            try:
                                self.qdrant.upsert_faces_batch(qdrant_buffer)
                                logger.info(f"Flushed {len(qdrant_buffer)} faces to Qdrant")
                                self.db.commit()
                                qdrant_buffer.clear()
                            except Exception as e:
                                self.db.rollback()
                                logger.error(f"Failed to flush Qdrant buffer: {e}")
                                shutdown_event.set()
                                raise RuntimeError(
                                    f"Qdrant batch upsert failed: {e}"
                                ) from e

                    total_processed += 1
                    if progress_callback:
                        progress_callback(1)

                except Exception as e:
                    logger.error(f"Error processing asset {asset.id}: {e}")
                    all_errors.append({"asset_id": asset.id, "error": str(e)})
                    if progress_callback:
                        progress_callback(1)
                    # Don't shutdown on single asset error, continue processing

        # Wait for producer to finish
        producer_thread.join()

        # Check for producer errors
        if producer_error:
            logger.error(f"Pipeline failed due to producer error: {producer_error[0]}")

        # Calculate total I/O time from accumulator
        total_io_time = sum(io_time_accumulator)

        # Flush remaining Qdrant buffer at end
        if qdrant_buffer and not shutdown_event.is_set():
            try:
                self.qdrant.upsert_faces_batch(qdrant_buffer)
                logger.info(f"Flushed final {len(qdrant_buffer)} faces to Qdrant")
                self.db.commit()
                qdrant_buffer.clear()
            except Exception as e:
                self.db.rollback()
                logger.error(f"Failed to flush final Qdrant buffer: {e}")
                raise RuntimeError(f"Final Qdrant batch upsert failed: {e}") from e

        elapsed_time = time.time() - start_time
        throughput = total_processed / elapsed_time if elapsed_time > 0 else 0.0

        # Calculate pipeline efficiency
        # Efficiency measures how much I/O was hidden behind GPU processing
        # Perfect efficiency (1.0) = elapsed_time equals max(io_time, gpu_time)
        sequential_time = total_io_time + total_gpu_time
        if sequential_time > 0 and elapsed_time > 0:
            # Time saved compared to sequential execution
            time_saved = sequential_time - elapsed_time
            # Normalize by the time that could potentially be overlapped
            overlap_potential = min(total_io_time, total_gpu_time)
            if overlap_potential > 0:
                pipeline_efficiency = min(1.0, max(0.0, time_saved / overlap_potential))
            else:
                pipeline_efficiency = 0.0
        else:
            pipeline_efficiency = 0.0

        # Log pipeline performance metrics
        logger.info(
            f"Pipeline stats: io_time={total_io_time:.2f}s, gpu_time={total_gpu_time:.2f}s, "
            f"elapsed={elapsed_time:.2f}s, efficiency={pipeline_efficiency:.1%}, "
            f"throughput={throughput:.2f} img/s"
        )

        return {
            "processed": total_processed,
            "total_faces": total_faces,
            "errors": len(all_errors),
            "error_details": all_errors,
            "throughput": throughput,
            "elapsed_time": elapsed_time,
            "io_time": total_io_time,
            "gpu_time": total_gpu_time,
            "pipeline_efficiency": pipeline_efficiency,
        }

    def _process_assets_sequential(
        self,
        asset_ids: list[int],
        min_confidence: float = 0.5,
        min_face_size: int = 20,
        progress_callback: Any = None,
    ) -> dict[str, Any]:
        """Sequential processing for batch_size=1 (backward compatible)."""
        total_faces = 0
        processed = 0
        errors = 0
        error_details: list[dict[str, Any]] = []
        start_time = time.time()

        # Timing tracking
        total_io_time = 0.0
        total_gpu_time = 0.0

        for asset_id in asset_ids:
            io_start = time.time()
            asset = self.db.get(ImageAsset, asset_id)
            io_elapsed = time.time() - io_start
            total_io_time += io_elapsed

            if not asset:
                logger.warning(f"Asset not found: {asset_id}")
                errors += 1
                error_details.append({"asset_id": asset_id, "error": "Asset not found"})
                if progress_callback:
                    progress_callback(1)
                continue

            try:
                gpu_start = time.time()
                faces = self.process_asset(asset, min_confidence, min_face_size)
                gpu_elapsed = time.time() - gpu_start
                total_gpu_time += gpu_elapsed

                total_faces += len(faces)
                processed += 1
                if progress_callback:
                    progress_callback(1)
            except Exception as e:
                logger.error(f"Error processing asset {asset_id}: {e}")
                errors += 1
                error_details.append({"asset_id": asset_id, "error": str(e)})
                if progress_callback:
                    progress_callback(1)

        elapsed_time = time.time() - start_time
        throughput = processed / elapsed_time if elapsed_time > 0 else 0.0

        return {
            "processed": processed,
            "total_faces": total_faces,
            "errors": errors,
            "error_details": error_details,
            "throughput": throughput,
            "elapsed_time": elapsed_time,
            "io_time": total_io_time,
            "gpu_time": total_gpu_time,
        }

    def _resolve_asset_path(self, asset: ImageAsset) -> str | None:
        """Get the file path for an asset."""
        # Check common path attributes
        if hasattr(asset, "path") and asset.path:
            return str(asset.path)
        if hasattr(asset, "file_path") and asset.file_path:
            return str(asset.file_path)
        if hasattr(asset, "source_path") and asset.source_path:
            return str(asset.source_path)
        return None

    def _process_loaded_image(
        self,
        asset: ImageAsset,
        image: "np.ndarray[Any, Any]",
        min_confidence: float = 0.5,
        min_face_size: int = 20,
    ) -> list[FaceInstance]:
        """Process a pre-loaded image through face detection.

        This is separated from image loading to allow parallel I/O.

        Args:
            asset: The ImageAsset record
            image: Pre-loaded BGR image array from cv2.imread
            min_confidence: Minimum detection confidence threshold
            min_face_size: Minimum face width/height in pixels

        Returns:
            List of created/existing FaceInstance records
        """
        # Detect faces using pre-loaded image
        detected = detect_faces(
            image,
            min_confidence=min_confidence,
            min_face_size=min_face_size,
        )

        if not detected:
            logger.debug(f"No faces detected in asset {asset.id}")
            return []

        face_instances = []
        qdrant_points = []

        for face in detected:
            # Check if this face already exists (idempotency)
            existing = self._find_existing_face(asset.id, face.bbox)
            if existing:
                face_instances.append(existing)
                continue

            # Create new FaceInstance
            point_id = uuid.uuid4()
            quality_score = face.compute_quality_score()

            face_instance = FaceInstance(
                id=uuid.uuid4(),
                asset_id=asset.id,
                bbox_x=face.bbox[0],
                bbox_y=face.bbox[1],
                bbox_w=face.bbox[2],
                bbox_h=face.bbox[3],
                landmarks=face.landmarks_as_dict(),
                detection_confidence=face.confidence,
                quality_score=quality_score,
                qdrant_point_id=point_id,
            )

            self.db.add(face_instance)
            face_instances.append(face_instance)

            # Prepare Qdrant point
            qdrant_point: dict[str, Any] = {
                "point_id": point_id,
                "embedding": face.embedding.tolist(),
                "asset_id": asset.id,
                "face_instance_id": face_instance.id,
                "detection_confidence": face.confidence,
                "quality_score": quality_score,
                "bbox": {
                    "x": face.bbox[0],
                    "y": face.bbox[1],
                    "w": face.bbox[2],
                    "h": face.bbox[3],
                },
            }

            # Add optional taken_at if available
            if hasattr(asset, "file_modified_at") and asset.file_modified_at:
                qdrant_point["taken_at"] = asset.file_modified_at

            qdrant_points.append(qdrant_point)

        # Batch upsert to Qdrant FIRST (before commit)
        if qdrant_points:
            try:
                self.qdrant.upsert_faces_batch(qdrant_points)
                logger.info(f"Stored {len(qdrant_points)} new faces for asset {asset.id}")
            except Exception as e:
                # Rollback DB if Qdrant fails
                self.db.rollback()
                logger.error(f"Failed to upsert faces to Qdrant for asset {asset.id}: {e}")
                raise RuntimeError(f"Face detection failed: Qdrant upsert error: {e}") from e

        # Commit DB changes ONLY if Qdrant succeeded
        self.db.commit()

        return face_instances

    def _find_existing_face(
        self,
        asset_id: int,
        bbox: tuple[int, int, int, int],
    ) -> FaceInstance | None:
        """Check if a face with this bbox already exists for the asset."""
        stmt = select(FaceInstance).where(
            FaceInstance.asset_id == asset_id,
            FaceInstance.bbox_x == bbox[0],
            FaceInstance.bbox_y == bbox[1],
            FaceInstance.bbox_w == bbox[2],
            FaceInstance.bbox_h == bbox[3],
        )
        return self.db.execute(stmt).scalar_one_or_none()


def get_face_service(db_session: SyncSession) -> FaceProcessingService:
    """Factory function for FaceProcessingService."""
    return FaceProcessingService(db_session)
