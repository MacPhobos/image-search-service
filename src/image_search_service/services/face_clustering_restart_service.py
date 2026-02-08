"""Face clustering restart service for re-running clustering without re-detecting faces."""

import time
from typing import Any
from uuid import UUID

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.core.logging import get_logger
from image_search_service.db.models import (
    FaceDetectionSession,
    FaceDetectionSessionStatus,
    FaceInstance,
    FaceSuggestion,
    Person,
    TrainingEvidence,
)
from image_search_service.services.restart_service_base import CleanupStats, RestartServiceBase

logger = get_logger(__name__)


class FaceClusteringRestartService(RestartServiceBase):
    """Restart face clustering (Phase 3: HDBSCAN).

    Cleanup operations:
    - Delete Person records where name LIKE 'Unknown Person%'
    - Delete auto-accepted FaceSuggestion records
    - Reset FaceInstance.person_id and cluster_id to NULL

    Preserved data:
    - Face detection results (FaceInstance records)
    - Manually assigned faces
    - Qdrant vectors (metadata-only operation)
    - Manually created Person records

    Safety features:
    - Validates face detection completion
    - Uses naming convention to identify auto-created persons
    - Preserves manually assigned faces
    - Qdrant vectors untouched (metadata-only operation)
    - Uses existing cluster_unlabeled_faces function
    - Idempotent: safe to call when no clusters exist
    """

    async def validate_state(
        self,
        db: AsyncSession,
        session_id: int,
    ) -> None:
        """Validate clustering can be restarted.

        Requirements:
        - Face detection must be COMPLETED
        - At least 1 face detected

        Args:
            db: Database session
            session_id: Training session ID

        Raises:
            ValueError: If face detection not completed or no faces
        """
        # Get face detection session
        face_session_result = await db.execute(
            select(FaceDetectionSession).where(
                FaceDetectionSession.training_session_id == session_id
            )
        )
        face_session = face_session_result.scalar_one_or_none()

        if not face_session:
            raise ValueError(
                f"Cannot restart clustering: no face detection session "
                f"for training session {session_id}"
            )

        if face_session.status != FaceDetectionSessionStatus.COMPLETED.value:
            raise ValueError(
                f"Cannot restart clustering: face detection not completed "
                f"(status: {face_session.status})"
            )

        if face_session.faces_detected == 0:
            raise ValueError("Cannot restart clustering: no faces detected")

    async def cleanup(
        self,
        db: AsyncSession,
        session_id: int,
    ) -> CleanupStats:
        """Reset clustering state.

        Deletes:
        - Person records where name LIKE 'Unknown Person%'
        - PersonPrototype records (FK cascade)
        - FaceSuggestion records where status = 'accepted'

        Resets:
        - FaceInstance.person_id to NULL (where person was deleted or auto-assigned)
        - FaceInstance.cluster_id to NULL

        Preserves:
        - Manually created Person records
        - Manually assigned faces (person_id kept for non-auto persons)
        - Face detection results (FaceInstance records)
        - Qdrant vectors (untouched)

        Idempotent: Safe to call when no clusters exist.

        Args:
            db: Database session
            session_id: Training session ID

        Returns:
            CleanupStats with deletion/reset counts
        """
        start_time = time.time()

        # 1. Find "Unknown Person" records (created by clustering)
        unknown_persons = await self._get_unknown_persons(db)
        unknown_person_ids = [p.id for p in unknown_persons]

        if not unknown_persons:
            # Already clean - idempotent
            logger.info(f"No unknown persons found for session {session_id}, cleanup skipped")
            return CleanupStats(
                operation="clustering_restart",
                session_id=session_id,
                items_deleted=0,
                items_reset=0,
                items_preserved=0,
                duration_ms=0,
            )

        logger.info(
            f"Found {len(unknown_persons)} unknown persons to delete",
            extra={"session_id": session_id, "person_count": len(unknown_persons)},
        )

        # 2. Delete unknown persons (cascades to prototypes)
        persons_deleted = await self._delete_unknown_persons(db, unknown_person_ids)

        # 3. Delete auto-accepted suggestions (from clustering)
        suggestions_deleted = await self._delete_clustering_suggestions(db, session_id)

        # 4. Reset face instance assignments
        # Reset faces that were assigned to unknown persons or have cluster_id
        faces_reset = await self._reset_face_assignments(db, session_id)

        # 5. Count preserved (manually assigned) faces
        faces_preserved = await self._count_manually_assigned_faces(db, session_id)

        duration_ms = int((time.time() - start_time) * 1000)

        stats = CleanupStats(
            operation="clustering_restart",
            session_id=session_id,
            items_deleted=persons_deleted + suggestions_deleted,
            items_reset=faces_reset,
            items_preserved=faces_preserved,
            duration_ms=duration_ms,
        )

        # Add extra details
        stats["unknown_persons_deleted"] = persons_deleted  # type: ignore
        stats["suggestions_deleted"] = suggestions_deleted  # type: ignore
        stats["faces_reset"] = faces_reset  # type: ignore
        stats["manually_assigned_faces_preserved"] = faces_preserved  # type: ignore

        logger.info(
            f"Cleanup completed: deleted {persons_deleted} persons, "
            f"{suggestions_deleted} suggestions, reset {faces_reset} faces, "
            f"preserved {faces_preserved} manually assigned faces",
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
        """Reset face detection session clustering counters.

        Args:
            db: Database session
            session_id: Training session ID
        """
        face_session_result = await db.execute(
            select(FaceDetectionSession).where(
                FaceDetectionSession.training_session_id == session_id
            )
        )
        face_session = face_session_result.scalar_one_or_none()

        if not face_session:
            raise ValueError(
                f"Face detection session not found for training session {session_id}"
            )

        # Reset clustering counters
        face_session.faces_assigned_to_persons = 0
        face_session.clusters_created = 0
        face_session.suggestions_created = 0
        # Reset legacy field (sum of assigned + clustered)
        face_session.faces_assigned = 0

        logger.info(
            f"Reset clustering counters for face detection session {face_session.id}",
            extra={"session_id": session_id, "face_session_id": str(face_session.id)},
        )

    async def enqueue_job(
        self,
        db: AsyncSession,
        session_id: int,
    ) -> str:
        """Run clustering synchronously (not a background job).

        CRITICAL: Uses existing cluster_unlabeled_faces function.

        Note: Clustering is typically fast enough to run synchronously.
        The actual clustering is done via the FaceClusterer service.

        Args:
            db: Database session
            session_id: Training session ID

        Returns:
            Job status string (not an RQ job ID)
        """
        from image_search_service.db.sync_operations import get_sync_session
        from image_search_service.faces.clusterer import get_face_clusterer

        # Get face detection session
        face_session_result = await db.execute(
            select(FaceDetectionSession).where(
                FaceDetectionSession.training_session_id == session_id
            )
        )
        face_session = face_session_result.scalar_one_or_none()

        if not face_session:
            raise ValueError(
                f"Face detection session not found for training session {session_id}"
            )

        # Commit async session before using sync session
        await db.commit()

        # Run clustering synchronously using existing function
        # NOTE: We need to use sync session for the clusterer
        from image_search_service.vector.face_qdrant import get_face_qdrant_client

        sync_db = get_sync_session()
        try:
            qdrant_client = get_face_qdrant_client()
            clusterer = get_face_clusterer(
                db_session=sync_db,
                qdrant_client=qdrant_client,
                min_cluster_size=3,  # Minimum faces per cluster
            )

            logger.info(
                f"Starting clustering for session {session_id}",
                extra={"session_id": session_id, "face_session_id": str(face_session.id)},
            )

            # Cluster unlabeled faces
            cluster_result = clusterer.cluster_unlabeled_faces(
                quality_threshold=0.3,  # Include most detected faces
                max_faces=10000,  # Process up to 10k faces
            )

            clusters_found = cluster_result.get("clusters_found", 0)
            noise_count = cluster_result.get("noise_count", 0)

            logger.info(
                f"Clustering complete: {clusters_found} clusters, {noise_count} noise",
                extra={
                    "session_id": session_id,
                    "clusters_found": clusters_found,
                    "noise_count": noise_count,
                },
            )

            # Update session with clustering stats (sync session)
            sync_face_session = sync_db.get(FaceDetectionSession, face_session.id)
            if sync_face_session:
                sync_face_session.clusters_created = clusters_found
                # Update legacy field for backward compatibility
                sync_face_session.faces_assigned = (
                    sync_face_session.faces_assigned_to_persons or 0
                ) + clusters_found
                sync_db.commit()

        finally:
            sync_db.close()

        # Refresh async session
        await db.refresh(face_session)

        return "clustering_completed_synchronously"

    async def _get_unknown_persons(
        self,
        db: AsyncSession,
    ) -> list[Person]:
        """Find persons created by clustering (naming convention).

        Args:
            db: Database session

        Returns:
            List of Person objects with names starting with "Unknown Person"
        """
        result = await db.execute(select(Person).where(Person.name.like("Unknown Person%")))
        return list(result.scalars().all())

    async def _delete_unknown_persons(
        self,
        db: AsyncSession,
        person_ids: list[UUID],
    ) -> int:
        """Delete unknown persons (cascades to prototypes).

        Args:
            db: Database session
            person_ids: List of person IDs to delete

        Returns:
            Number of persons deleted
        """
        if not person_ids:
            return 0

        result = await db.execute(delete(Person).where(Person.id.in_(person_ids)))

        rowcount = result.rowcount or 0  # type: ignore[attr-defined]
        logger.info(f"Deleted {rowcount} unknown Person records")
        return rowcount

    async def _delete_clustering_suggestions(
        self,
        db: AsyncSession,
        session_id: int,
    ) -> int:
        """Delete auto-accepted suggestions from clustering.

        Deletes suggestions where:
        - Status is 'accepted'
        - Confidence is >= 0.7 (auto-accept threshold)
        - Face instance belongs to this training session

        Args:
            db: Database session
            session_id: Training session ID

        Returns:
            Number of suggestions deleted
        """
        # Get face instance IDs for this session
        face_ids = await self._get_face_instance_ids(db, session_id)

        if not face_ids:
            return 0

        # Delete accepted suggestions (auto-accepted by clustering)
        # High confidence suggestions (>= 0.7) are typically auto-accepted
        result = await db.execute(
            delete(FaceSuggestion).where(
                FaceSuggestion.face_instance_id.in_(face_ids),
                FaceSuggestion.status == "accepted",
                FaceSuggestion.confidence >= 0.7,  # Auto-accepted threshold
            )
        )

        rowcount = result.rowcount or 0  # type: ignore[attr-defined]
        logger.info(f"Deleted {rowcount} auto-accepted suggestions")
        return rowcount

    async def _reset_face_assignments(
        self,
        db: AsyncSession,
        session_id: int,
    ) -> int:
        """Reset face assignments to NULL for clustered faces.

        Only resets faces where cluster_id is set (i.e., auto-clustered faces).
        Preserves manually assigned faces (no cluster_id).

        Args:
            db: Database session
            session_id: Training session ID

        Returns:
            Number of faces reset
        """
        face_ids = await self._get_face_instance_ids(db, session_id)

        if not face_ids:
            return 0

        # Reset only faces with cluster_id (auto-clustered)
        # This preserves manually assigned faces
        result = await db.execute(
            update(FaceInstance)
            .where(
                FaceInstance.id.in_(face_ids),
                FaceInstance.cluster_id.isnot(None),  # Only reset clustered faces
            )
            .values(
                person_id=None,
                cluster_id=None,
            )
        )

        rowcount = result.rowcount or 0  # type: ignore[attr-defined]
        logger.info(f"Reset {rowcount} face assignments")
        return rowcount

    async def _count_manually_assigned_faces(
        self,
        db: AsyncSession,
        session_id: int,
    ) -> int:
        """Count faces with manual person assignments.

        Manual assignments have person_id but no cluster_id.

        Args:
            db: Database session
            session_id: Training session ID

        Returns:
            Number of manually assigned faces
        """
        face_ids = await self._get_face_instance_ids(db, session_id)

        if not face_ids:
            return 0

        result = await db.execute(
            select(func.count(FaceInstance.id)).where(
                FaceInstance.id.in_(face_ids),
                FaceInstance.person_id.isnot(None),
                FaceInstance.cluster_id.is_(None),  # Manual assignments have no cluster_id
            )
        )

        count = result.scalar() or 0
        logger.debug(f"Found {count} manually assigned faces")
        return count

    async def _get_face_instance_ids(
        self,
        db: AsyncSession,
        session_id: int,
    ) -> list[UUID]:
        """Get all face instance IDs for training session.

        Args:
            db: Database session
            session_id: Training session ID

        Returns:
            List of face instance IDs
        """
        # Get asset IDs from training evidence
        asset_ids_result = await db.execute(
            select(TrainingEvidence.asset_id).where(TrainingEvidence.session_id == session_id)
        )
        asset_ids = [row[0] for row in asset_ids_result]

        if not asset_ids:
            return []

        # Get face instance IDs for these assets
        face_ids_result = await db.execute(
            select(FaceInstance.id).where(FaceInstance.asset_id.in_(asset_ids))
        )
        return [row[0] for row in face_ids_result]

    async def _capture_snapshot(
        self,
        db: AsyncSession,
        session_id: int,
    ) -> dict[str, Any]:
        """Capture clustering state for rollback.

        NOTE: Cannot snapshot deleted persons/assignments.
        Rollback limited to session counters only.

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
            raise ValueError(
                f"Face detection session not found for training session {session_id}"
            )

        return {
            "session_id": str(face_session.id),
            "training_session_id": session_id,
            "faces_assigned": face_session.faces_assigned,
            "faces_assigned_to_persons": face_session.faces_assigned_to_persons,
            "clusters_created": face_session.clusters_created,
            "suggestions_created": face_session.suggestions_created,
        }

    async def _restore_snapshot(
        self,
        db: AsyncSession,
        snapshot: dict[str, Any],
    ) -> None:
        """Restore clustering counters.

        NOTE: Cannot restore deleted persons/assignments.

        Args:
            db: Database session
            snapshot: Snapshot dictionary from _capture_snapshot
        """
        face_session_result = await db.execute(
            select(FaceDetectionSession).where(
                FaceDetectionSession.id == UUID(snapshot["session_id"])
            )
        )
        face_session = face_session_result.scalar_one_or_none()

        if face_session:
            face_session.faces_assigned = snapshot["faces_assigned"]
            face_session.faces_assigned_to_persons = snapshot["faces_assigned_to_persons"]
            face_session.clusters_created = snapshot["clusters_created"]
            face_session.suggestions_created = snapshot["suggestions_created"]

            logger.info(
                "Restored clustering counters",
                extra={"session_id": snapshot["session_id"]},
            )
