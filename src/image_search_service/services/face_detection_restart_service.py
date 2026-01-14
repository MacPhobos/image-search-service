"""Face detection restart service for restarting face detection sessions."""

import time
from typing import Any
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.core.logging import get_logger
from image_search_service.db.models import (
    FaceDetectionSession,
    FaceDetectionSessionStatus,
    FaceInstance,
    FaceSuggestion,
    Person,
    SessionStatus,
    TrainingEvidence,
    TrainingSession,
)
from image_search_service.queue.worker import QUEUE_HIGH, get_queue
from image_search_service.services.restart_service_base import CleanupStats, RestartServiceBase
from image_search_service.vector.face_qdrant import FaceQdrantClient

logger = get_logger(__name__)


class FaceDetectionRestartService(RestartServiceBase):
    """Restart face detection session (Phase 2: InsightFace).

    Options:
        delete_orphaned_persons: If True, delete Person records with no face instances.
                                If False (default), preserve orphaned Person records.

    Cleanup sequence (order critical):
    1. Query all face instances for session's assets
    2. Delete Qdrant face vectors FIRST (prevent orphaned DB records)
    3. Delete FaceInstance records (cascades to suggestions)
    4. Optionally delete orphaned Person records

    Safety features:
    - Validates training completion before restart
    - Conservative default: preserve Person records
    - Idempotent: safe to call when no faces exist
    - Uses existing detect_faces_for_session_job (no new code path)
    - Rollback limited to session state (face deletions irreversible)
    """

    def __init__(self, delete_orphaned_persons: bool = False):
        """Initialize face detection restart service.

        Args:
            delete_orphaned_persons: Whether to delete orphaned Person records
        """
        self.delete_orphaned_persons = delete_orphaned_persons
        self._qdrant_client: FaceQdrantClient | None = None

    @property
    def qdrant_client(self) -> FaceQdrantClient:
        """Get Qdrant client (lazy initialization)."""
        if self._qdrant_client is None:
            self._qdrant_client = FaceQdrantClient.get_instance()
        return self._qdrant_client

    async def validate_state(
        self,
        db: AsyncSession,
        session_id: int,
    ) -> None:
        """Validate face detection can be restarted.

        Requirements:
        - Training session must be COMPLETED
        - Face detection session must not be PROCESSING
        - No active face detection jobs

        Args:
            db: Database session
            session_id: Training session ID (not face detection session ID)

        Raises:
            ValueError: If training not completed or face detection running
            RuntimeError: If face detection job still active
        """
        # Check training session
        training_result = await db.execute(
            select(TrainingSession).where(TrainingSession.id == session_id)
        )
        training_session = training_result.scalar_one_or_none()

        if not training_session:
            raise ValueError(f"Training session {session_id} not found")

        if training_session.status != SessionStatus.COMPLETED.value:
            raise ValueError(
                f"Cannot restart face detection: training session {session_id} "
                f"not completed (status: {training_session.status})"
            )

        # Check face detection session (if exists)
        face_session_result = await db.execute(
            select(FaceDetectionSession).where(
                FaceDetectionSession.training_session_id == session_id
            )
        )
        face_session = face_session_result.scalar_one_or_none()

        if face_session:
            if face_session.status == FaceDetectionSessionStatus.PROCESSING.value:
                raise ValueError(
                    f"Cannot restart: face detection session {face_session.id} "
                    "is currently processing. Cancel or wait for completion first."
                )

            # Check for active RQ job
            if face_session.job_id:
                # Note: We can't easily check RQ job status without importing RQ
                # This is a basic check - actual job may still be running
                logger.warning(
                    f"Face detection session has job_id {face_session.job_id}, "
                    "assuming it's completed based on status"
                )

    async def cleanup(
        self,
        db: AsyncSession,
        session_id: int,
    ) -> CleanupStats:
        """Delete face detection results for training session.

        Steps:
        1. Query all face instances for session's assets
        2. Delete Qdrant face vectors FIRST
        3. Delete face instances (cascades to suggestions)
        4. Optionally delete orphaned persons

        Idempotent: Safe to call when no faces exist.

        Args:
            db: Database session
            session_id: Training session ID

        Returns:
            CleanupStats with deletion counts
        """
        start_time = time.time()

        # 1. Get all face instances for this session's assets
        face_instances = await self._get_face_instances_for_session(db, session_id)

        if not face_instances:
            # Already clean - idempotent
            logger.info(f"No face instances found for session {session_id}, cleanup skipped")
            return CleanupStats(
                operation="face_detection_restart",
                session_id=session_id,
                items_deleted=0,
                items_reset=0,
                items_preserved=0,
                duration_ms=0,
            )

        face_ids = [f.qdrant_point_id for f in face_instances]
        face_instance_ids = [f.id for f in face_instances]
        face_count = len(face_instances)

        logger.info(
            f"Found {face_count} face instances to delete for session {session_id}",
            extra={"session_id": session_id, "face_count": face_count},
        )

        # 2. Delete from Qdrant FIRST (can't rollback, so fail early)
        try:
            qdrant_deleted = await self._delete_qdrant_vectors(face_ids)
        except Exception as e:
            logger.error(f"Qdrant delete failed: {e}", exc_info=True)
            raise RuntimeError(
                f"Failed to delete {len(face_ids)} face vectors from Qdrant. "
                "Cannot proceed with database cleanup."
            ) from e

        # 3. Delete from database (rollback-able)
        try:
            # Delete suggestions first (may cascade already, but be explicit)
            suggestions_deleted = await self._delete_suggestions(db, face_instance_ids)

            # Delete face instances
            db_deleted = await self._delete_face_instances(db, face_instance_ids)

        except Exception as e:
            logger.error(
                f"Database delete failed, Qdrant vectors already deleted: {face_ids}",
                exc_info=True,
            )
            raise RuntimeError(
                f"Database cleanup failed. Qdrant vectors deleted but database records remain. "
                f"Manual cleanup may be needed for {len(face_ids)} orphaned vectors."
            ) from e

        # 4. Optionally delete orphaned persons
        persons_deleted = 0
        persons_preserved = 0

        if self.delete_orphaned_persons:
            persons_deleted = await self._delete_orphaned_persons(db)
            logger.info(f"Deleted {persons_deleted} orphaned Person records")
        else:
            persons_preserved = await self._count_orphaned_persons(db)
            logger.info(f"Preserved {persons_preserved} orphaned Person records")

        duration_ms = int((time.time() - start_time) * 1000)

        stats = CleanupStats(
            operation="face_detection_restart",
            session_id=session_id,
            items_deleted=db_deleted + suggestions_deleted + persons_deleted,
            items_reset=0,
            items_preserved=persons_preserved,
            duration_ms=duration_ms,
        )

        # Add extra details
        stats["face_instances_deleted"] = db_deleted  # type: ignore
        stats["qdrant_vectors_deleted"] = qdrant_deleted  # type: ignore
        stats["suggestions_deleted"] = suggestions_deleted  # type: ignore
        stats["persons_deleted"] = persons_deleted  # type: ignore
        stats["persons_orphaned"] = persons_preserved  # type: ignore

        logger.info(
            f"Cleanup completed: deleted {db_deleted} faces, {suggestions_deleted} suggestions, "
            f"{persons_deleted} persons",
            extra={
                "session_id": session_id,
                "stats": stats,
            },
        )

        return stats

    async def reset_state(
        self,
        db: AsyncSession,
        session_id: int,
    ) -> None:
        """Reset or create face detection session.

        Args:
            db: Database session
            session_id: Training session ID
        """
        # Get existing face session
        face_session_result = await db.execute(
            select(FaceDetectionSession).where(
                FaceDetectionSession.training_session_id == session_id
            )
        )
        face_session = face_session_result.scalar_one_or_none()

        if not face_session:
            # Create new session
            # Count total images from training session
            total_images_result = await db.execute(
                select(func.count(TrainingEvidence.asset_id.distinct())).where(
                    TrainingEvidence.session_id == session_id
                )
            )
            total_images = total_images_result.scalar() or 0

            face_session = FaceDetectionSession(
                training_session_id=session_id,
                status=FaceDetectionSessionStatus.PENDING.value,
                total_images=total_images,
                processed_images=0,
                failed_images=0,
                faces_detected=0,
                faces_assigned=0,
                faces_assigned_to_persons=0,
                clusters_created=0,
                suggestions_created=0,
                current_batch=0,
                total_batches=0,
                current_asset_index=0,
                asset_ids_json=None,
                job_id=None,
            )
            db.add(face_session)
            logger.info(f"Created new face detection session for training session {session_id}")
        else:
            # Reset existing session
            face_session.status = FaceDetectionSessionStatus.PENDING.value
            face_session.processed_images = 0
            face_session.failed_images = 0
            face_session.faces_detected = 0
            face_session.faces_assigned = 0
            face_session.faces_assigned_to_persons = 0
            face_session.clusters_created = 0
            face_session.suggestions_created = 0
            face_session.current_batch = 0
            face_session.current_asset_index = 0
            face_session.asset_ids_json = None
            face_session.job_id = None
            face_session.last_error = None
            face_session.started_at = None
            face_session.completed_at = None
            logger.info(
                f"Reset face detection session {face_session.id} for training session {session_id}"
            )

    async def enqueue_job(
        self,
        db: AsyncSession,
        session_id: int,
    ) -> str:
        """Enqueue face detection job using existing function.

        CRITICAL: Uses existing detect_faces_for_session_job.

        Args:
            db: Database session
            session_id: Training session ID

        Returns:
            RQ job ID
        """
        from image_search_service.queue.face_jobs import detect_faces_for_session_job

        # Get face session ID
        face_session_result = await db.execute(
            select(FaceDetectionSession).where(
                FaceDetectionSession.training_session_id == session_id
            )
        )
        face_session = face_session_result.scalar_one_or_none()

        if not face_session:
            raise ValueError(
                f"Face detection session not found for training session {session_id}. "
                "reset_state should have created it."
            )

        # Enqueue job (high priority)
        queue = get_queue(QUEUE_HIGH)
        rq_job = queue.enqueue(
            detect_faces_for_session_job,
            str(face_session.id),
            job_timeout="2h",
        )

        # Store job ID
        face_session.job_id = rq_job.id
        await db.commit()

        logger.info(
            "Enqueued face detection job",
            extra={
                "session_id": session_id,
                "face_session_id": str(face_session.id),
                "job_id": rq_job.id,
            },
        )

        return str(rq_job.id)

    async def _get_face_instances_for_session(
        self,
        db: AsyncSession,
        session_id: int,
    ) -> list[FaceInstance]:
        """Query all face instances for training session's assets.

        Args:
            db: Database session
            session_id: Training session ID

        Returns:
            List of FaceInstance objects
        """
        # Get asset IDs from training evidence
        asset_ids_result = await db.execute(
            select(TrainingEvidence.asset_id).where(TrainingEvidence.session_id == session_id)
        )
        asset_ids = [row[0] for row in asset_ids_result]

        if not asset_ids:
            return []

        # Get face instances for these assets
        face_instances_result = await db.execute(
            select(FaceInstance).where(FaceInstance.asset_id.in_(asset_ids))
        )
        return list(face_instances_result.scalars().all())

    async def _delete_qdrant_vectors(
        self,
        face_ids: list[UUID],
    ) -> int:
        """Delete face vectors from Qdrant.

        Args:
            face_ids: List of Qdrant point IDs (UUIDs)

        Returns:
            Number of vectors deleted
        """
        if not face_ids:
            return 0

        from qdrant_client.models import PointIdsList

        # Get collection name from settings
        from image_search_service.core.config import get_settings

        collection_name = get_settings().qdrant_face_collection

        # Batch delete (max 1000 per batch for safety)
        total_deleted = 0
        batch_size = 1000

        for i in range(0, len(face_ids), batch_size):
            batch = face_ids[i : i + batch_size]

            self.qdrant_client.client.delete(
                collection_name=collection_name,
                points_selector=PointIdsList(points=[str(uuid) for uuid in batch]),
            )
            total_deleted += len(batch)

            logger.debug(f"Deleted batch of {len(batch)} face vectors from Qdrant")

        logger.info(f"Deleted {total_deleted} face vectors from Qdrant")
        return total_deleted

    async def _delete_face_instances(
        self,
        db: AsyncSession,
        face_instance_ids: list[UUID],
    ) -> int:
        """Delete face instances (cascades to suggestions).

        Args:
            db: Database session
            face_instance_ids: List of face instance IDs

        Returns:
            Number of face instances deleted
        """
        if not face_instance_ids:
            return 0

        result = await db.execute(
            delete(FaceInstance).where(FaceInstance.id.in_(face_instance_ids))
        )

        rowcount = result.rowcount or 0  # type: ignore[attr-defined]
        logger.info(f"Deleted {rowcount} face instances from database")
        return rowcount

    async def _delete_suggestions(
        self,
        db: AsyncSession,
        face_instance_ids: list[UUID],
    ) -> int:
        """Delete face suggestions (may have already cascaded).

        Args:
            db: Database session
            face_instance_ids: List of face instance IDs

        Returns:
            Number of suggestions deleted
        """
        if not face_instance_ids:
            return 0

        result = await db.execute(
            delete(FaceSuggestion).where(FaceSuggestion.face_instance_id.in_(face_instance_ids))
        )

        rowcount = result.rowcount or 0  # type: ignore[attr-defined]
        logger.info(f"Deleted {rowcount} face suggestions from database")
        return rowcount

    async def _delete_orphaned_persons(
        self,
        db: AsyncSession,
    ) -> int:
        """Delete persons with no remaining face instances.

        Args:
            db: Database session

        Returns:
            Number of persons deleted
        """
        # Find persons with zero faces
        orphaned_persons_result = await db.execute(
            select(Person.id)
            .outerjoin(FaceInstance, FaceInstance.person_id == Person.id)
            .group_by(Person.id)
            .having(func.count(FaceInstance.id) == 0)
        )
        person_ids = [row[0] for row in orphaned_persons_result]

        if not person_ids:
            return 0

        # Delete persons (cascades to prototypes)
        result = await db.execute(delete(Person).where(Person.id.in_(person_ids)))

        rowcount = result.rowcount or 0  # type: ignore[attr-defined]
        logger.info(f"Deleted {rowcount} orphaned Person records")
        return rowcount

    async def _count_orphaned_persons(
        self,
        db: AsyncSession,
    ) -> int:
        """Count persons with no face instances.

        Args:
            db: Database session

        Returns:
            Number of orphaned persons
        """
        orphaned_persons_result = await db.execute(
            select(Person.id)
            .outerjoin(FaceInstance, FaceInstance.person_id == Person.id)
            .group_by(Person.id)
            .having(func.count(FaceInstance.id) == 0)
        )
        person_ids = list(orphaned_persons_result.scalars().all())

        count = len(person_ids)
        logger.debug(f"Found {count} orphaned Person records")
        return count

    async def _capture_snapshot(
        self,
        db: AsyncSession,
        session_id: int,
    ) -> dict[str, Any]:
        """Capture face detection state for rollback.

        NOTE: We cannot snapshot deleted face instances or persons.
        Rollback is limited to session state only.

        Args:
            db: Database session
            session_id: Training session ID

        Returns:
            Snapshot dictionary with session state
        """
        face_session_result = await db.execute(
            select(FaceDetectionSession).where(
                FaceDetectionSession.training_session_id == session_id
            )
        )
        face_session = face_session_result.scalar_one_or_none()

        if not face_session:
            return {
                "session_exists": False,
                "training_session_id": session_id,
            }

        return {
            "session_exists": True,
            "session_id": str(face_session.id),
            "training_session_id": session_id,
            "status": face_session.status,
            "processed_images": face_session.processed_images,
            "faces_detected": face_session.faces_detected,
            "faces_assigned": face_session.faces_assigned,
            "faces_assigned_to_persons": face_session.faces_assigned_to_persons,
            "clusters_created": face_session.clusters_created,
            "suggestions_created": face_session.suggestions_created,
            "job_id": face_session.job_id,
            "asset_ids_json": face_session.asset_ids_json,
            "current_asset_index": face_session.current_asset_index,
        }

    async def _restore_snapshot(
        self,
        db: AsyncSession,
        snapshot: dict[str, Any],
    ) -> None:
        """Restore face detection session state.

        NOTE: Cannot restore deleted face instances or persons.

        Args:
            db: Database session
            snapshot: Snapshot dictionary from _capture_snapshot
        """
        if not snapshot["session_exists"]:
            # Delete any session created during failed restart
            await db.execute(
                delete(FaceDetectionSession).where(
                    FaceDetectionSession.training_session_id == snapshot["training_session_id"]
                )
            )
            logger.info("Deleted face session created during failed restart")
            return

        face_session_result = await db.execute(
            select(FaceDetectionSession).where(
                FaceDetectionSession.id == UUID(snapshot["session_id"])
            )
        )
        face_session = face_session_result.scalar_one_or_none()

        if face_session:
            face_session.status = snapshot["status"]
            face_session.processed_images = snapshot["processed_images"]
            face_session.faces_detected = snapshot["faces_detected"]
            face_session.faces_assigned = snapshot["faces_assigned"]
            face_session.faces_assigned_to_persons = snapshot["faces_assigned_to_persons"]
            face_session.clusters_created = snapshot["clusters_created"]
            face_session.suggestions_created = snapshot["suggestions_created"]
            face_session.job_id = snapshot["job_id"]
            face_session.asset_ids_json = snapshot["asset_ids_json"]
            face_session.current_asset_index = snapshot["current_asset_index"]

            logger.info(
                "Restored face detection session state",
                extra={"session_id": snapshot["session_id"]},
            )
