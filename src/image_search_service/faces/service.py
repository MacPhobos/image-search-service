"""High-level face processing service."""

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session as SyncSession

from image_search_service.db.models import FaceInstance, ImageAsset
from image_search_service.faces.detector import detect_faces_from_path
from image_search_service.vector.face_qdrant import get_face_qdrant_client

logger = logging.getLogger(__name__)


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

        # Commit DB changes
        self.db.commit()

        # Batch upsert to Qdrant
        if qdrant_points:
            self.qdrant.upsert_faces_batch(qdrant_points)
            logger.info(f"Stored {len(qdrant_points)} new faces for asset {asset.id}")

        return face_instances

    def process_assets_batch(
        self,
        asset_ids: list[int],
        min_confidence: float = 0.5,
        min_face_size: int = 20,
        batch_size: int = 8,
        progress_callback: Any = None,
    ) -> dict[str, Any]:
        """Process multiple assets in batch with parallel I/O loading.

        Uses ThreadPoolExecutor to pre-load images while GPU processes current image.
        This overlaps disk I/O with GPU inference for better throughput.

        Args:
            asset_ids: List of asset IDs to process
            min_confidence: Minimum detection confidence threshold
            min_face_size: Minimum face width/height in pixels
            batch_size: Number of images to pre-load in parallel (default: 8)
            progress_callback: Optional callable(step: int) to report progress

        Returns:
            Summary dict with counts and throughput metrics.
        """
        if not asset_ids:
            return {
                "processed": 0,
                "total_faces": 0,
                "errors": 0,
                "throughput": 0.0,
            }

        total_faces = 0
        processed = 0
        errors = 0
        error_details: list[dict[str, Any]] = []
        start_time = time.time()

        # Special case: batch_size=1 means sequential processing (backward compatible)
        if batch_size == 1:
            return self._process_assets_sequential(
                asset_ids, min_confidence, min_face_size, progress_callback
            )

        # Producer-consumer pattern: ThreadPoolExecutor pre-loads images
        # while GPU processes them sequentially
        def load_asset_data(asset_id: int) -> tuple[int, ImageAsset | None, str | None]:
            """Load asset from DB and return (id, asset, error)."""
            try:
                asset = self.db.get(ImageAsset, asset_id)
                if not asset:
                    return (asset_id, None, f"Asset not found: {asset_id}")
                return (asset_id, asset, None)
            except Exception as e:
                return (asset_id, None, f"Error loading asset: {e}")

        # Use ThreadPoolExecutor to parallelize DB lookups
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            # Submit all asset loading tasks
            future_to_id = {
                executor.submit(load_asset_data, asset_id): asset_id
                for asset_id in asset_ids
            }

            # Process results as they complete
            for future in as_completed(future_to_id):
                asset_id, asset, error = future.result()

                if error:
                    logger.warning(error)
                    errors += 1
                    error_details.append({"asset_id": asset_id, "error": error})
                    continue

                if not asset:
                    logger.warning(f"Asset not found: {asset_id}")
                    errors += 1
                    error_details.append(
                        {"asset_id": asset_id, "error": "Asset not found"}
                    )
                    continue

                # Process face detection (GPU-bound, runs sequentially)
                try:
                    faces = self.process_asset(asset, min_confidence, min_face_size)
                    total_faces += len(faces)
                    processed += 1
                    if progress_callback:
                        progress_callback(1)
                except Exception as e:
                    logger.error(f"Error processing asset {asset_id}: {e}")
                    errors += 1
                    error_details.append(
                        {"asset_id": asset_id, "error": str(e)}
                    )
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

        for asset_id in asset_ids:
            asset = self.db.get(ImageAsset, asset_id)
            if not asset:
                logger.warning(f"Asset not found: {asset_id}")
                errors += 1
                error_details.append({"asset_id": asset_id, "error": "Asset not found"})
                if progress_callback:
                    progress_callback(1)
                continue

            try:
                faces = self.process_asset(asset, min_confidence, min_face_size)
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
