"""Service for labeling face clusters as persons.

Extracted from faces.py route handler for reuse by both the Clusters
endpoint and the Unknown Persons endpoint. Handles:
- Person creation (or finding existing by name)
- Face assignment (full or partial with exclusions)
- Prototype creation (top 3 quality faces as exemplars)
- Qdrant sync (person_id + is_assigned + is_prototype)
- Find-more job trigger
"""

import logging
import uuid
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.db.models import (
    FaceInstance,
    Person,
    PersonPrototype,
    PrototypeRole,
)
from image_search_service.vector.face_qdrant import FaceQdrantClient

logger = logging.getLogger(__name__)


class ClusterLabelingService:
    """Shared service for creating a Person from a cluster of faces."""

    def __init__(
        self, db: AsyncSession, qdrant: FaceQdrantClient
    ) -> None:
        """Initialize cluster labeling service.

        Args:
            db: Async database session
            qdrant: Face Qdrant client for vector updates
        """
        self.db = db
        self.qdrant = qdrant

    async def label_cluster_as_person(
        self,
        face_ids: list[uuid.UUID],
        person_name: str,
        exclude_face_ids: list[uuid.UUID] | None = None,
        trigger_find_more: bool = True,
        trigger_reclustering: bool = False,
    ) -> dict[str, Any]:
        """Create a Person from a set of face IDs.

        This is the core logic extracted from the POST /faces/clusters/{cluster_id}/label
        endpoint. It handles the entire workflow of converting a cluster (or partial cluster)
        into a labeled Person entity.

        Workflow:
        1. Validate that all provided face IDs exist
        2. Apply exclusions (partial acceptance support)
        3. Find or create Person by name (case-insensitive)
        4. Assign person_id to accepted faces
        5. Create prototypes (top 3 quality faces as EXEMPLAR role)
        6. Sync to Qdrant (person_id, is_assigned=True, is_prototype)
        7. Optionally enqueue find-more job for propagation

        Args:
            face_ids: All face IDs in the group to potentially label.
            person_name: Name for the person (new or existing).
            exclude_face_ids: Face IDs to exclude from labeling (partial acceptance).
            trigger_find_more: Whether to enqueue find-more job for propagation.
            trigger_reclustering: Whether to enqueue re-clustering job (not used yet).

        Returns:
            Dict with:
                - person_id: UUID of the labeled person
                - person_name: Name of the person
                - faces_assigned: Count of faces actually assigned
                - faces_excluded: Count of faces excluded
                - prototypes_created: Count of prototype records created
                - find_more_job_id: Optional job ID if triggered

        Raises:
            ValueError: If no valid faces remain after exclusions
        """
        # 1. Fetch face instances
        query = select(FaceInstance).where(FaceInstance.id.in_(face_ids))
        result = await self.db.execute(query)
        all_faces = list(result.scalars().all())

        if not all_faces:
            raise ValueError(f"No faces found for provided IDs: {face_ids}")

        # 2. Apply exclusions (partial acceptance)
        exclude_set = set(exclude_face_ids or [])
        faces_to_assign = [f for f in all_faces if f.id not in exclude_set]
        faces_excluded = [f for f in all_faces if f.id in exclude_set]

        if not faces_to_assign:
            raise ValueError(
                "All faces excluded - at least one face must be assigned to create a person"
            )

        logger.info(
            f"Labeling {len(faces_to_assign)} faces as '{person_name}' "
            f"(excluding {len(faces_excluded)} faces)"
        )

        # 3. Find or create person by name (case-insensitive)
        person_query = select(Person).where(func.lower(Person.name) == person_name.lower())
        person_result = await self.db.execute(person_query)
        person = person_result.scalar_one_or_none()

        if not person:
            person = Person(name=person_name)
            self.db.add(person)
            await self.db.flush()
            logger.info(f"Created new person: {person.name} ({person.id})")
        else:
            logger.info(f"Using existing person: {person.name} ({person.id})")

        # 4. Assign person_id to faces
        face_ids_assigned = []
        qdrant_point_ids_assigned = []

        for face in faces_to_assign:
            face.person_id = person.id
            face_ids_assigned.append(face.id)
            qdrant_point_ids_assigned.append(face.qdrant_point_id)

        # 5. Update Qdrant payloads (person_id + is_assigned=True)
        # NOTE: is_assigned is automatically set in update_person_ids via upsert logic
        self.qdrant.update_person_ids(qdrant_point_ids_assigned, person.id)

        # 6. Create prototypes (top 3 quality faces as exemplars)
        sorted_faces = sorted(faces_to_assign, key=lambda f: f.quality_score or 0, reverse=True)
        prototypes_created = 0

        for face in sorted_faces[:3]:
            prototype = PersonPrototype(
                person_id=person.id,
                face_instance_id=face.id,
                qdrant_point_id=face.qdrant_point_id,
                role=PrototypeRole.EXEMPLAR,
            )
            self.db.add(prototype)
            prototypes_created += 1

            # Mark as prototype in Qdrant
            self.qdrant.update_payload(face.qdrant_point_id, {"is_prototype": True})

        await self.db.commit()

        logger.info(
            f"Labeled {len(faces_to_assign)} faces as person {person.name} "
            f"(created {prototypes_created} prototypes)"
        )

        # 7. Trigger find-more job (optional)
        find_more_job_id = None
        if trigger_find_more and sorted_faces:
            find_more_job_id = await self._enqueue_find_more_job(
                source_face_id=sorted_faces[0].id,
                person_id=person.id,
            )

        return {
            "person_id": person.id,
            "person_name": person.name,
            "faces_assigned": len(faces_to_assign),
            "faces_excluded": len(faces_excluded),
            "prototypes_created": prototypes_created,
            "find_more_job_id": find_more_job_id,
        }

    async def _enqueue_find_more_job(
        self,
        source_face_id: uuid.UUID,
        person_id: uuid.UUID,
    ) -> str | None:
        """Enqueue find-more propagation job for newly labeled person.

        Args:
            source_face_id: Best quality face from the labeled cluster
            person_id: Person ID to propagate

        Returns:
            Job ID if successfully enqueued, None if failed
        """
        try:
            from redis import Redis
            from rq import Queue

            from image_search_service.core.config import get_settings
            from image_search_service.queue.face_jobs import propagate_person_label_job

            settings = get_settings()
            redis_conn = Redis.from_url(settings.redis_url)
            queue = Queue("default", connection=redis_conn)

            job = queue.enqueue(
                propagate_person_label_job,
                source_face_id=str(source_face_id),
                person_id=str(person_id),
                min_confidence=0.7,
                max_suggestions=50,
                job_timeout="10m",
            )
            job_id = str(job.id) if job.id is not None else None
            logger.info(
                f"Queued find-more job {job_id} for face {source_face_id} → person {person_id}"
            )
            return job_id

        except Exception as e:
            logger.warning(f"Failed to enqueue find-more job: {e}", exc_info=True)
            return None
