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
    FaceAssignmentEvent,
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
        person_name: str | None = None,
        person_id: uuid.UUID | None = None,
        exclude_face_ids: list[uuid.UUID] | None = None,
        trigger_find_more: bool = True,
        trigger_reclustering: bool = False,
    ) -> dict[str, Any]:
        """Create a Person from a set of face IDs or assign to existing person.

        This is the core logic extracted from the POST /faces/clusters/{cluster_id}/label
        endpoint. It handles the entire workflow of converting a cluster (or partial cluster)
        into a labeled Person entity.

        Workflow:
        1. Validate that all provided face IDs exist
        2. Apply exclusions (partial acceptance support)
        3. Find/create Person by name OR look up existing Person by ID
        4. Assign person_id to accepted faces
        5. Create prototypes (top 3 quality faces as EXEMPLAR role, if person has none)
        6. Sync to Qdrant (person_id, is_assigned=True, is_prototype)
        7. Create FaceAssignmentEvent audit record
        8. Optionally enqueue find-more job for propagation

        Args:
            face_ids: All face IDs in the group to potentially label.
            person_name: Name for the person (new or existing).
                Mutually exclusive with person_id.
            person_id: ID of existing person to assign faces to.
                Mutually exclusive with person_name.
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
                - assignment_event_id: UUID of the audit event created

        Raises:
            ValueError: If no valid faces remain after exclusions, or if person_id not found,
                       or if person status is not 'active'
        """
        # 0. Validate: exactly one of person_name or person_id must be provided
        if not person_name and not person_id:
            raise ValueError("Either person_name or person_id must be provided")
        if person_name and person_id:
            raise ValueError("Provide either person_name or person_id, not both")

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

        # 3. Find/create person by name OR look up existing person by ID
        if person_id:
            # Look up existing person by ID
            person = await self.db.get(Person, person_id)
            if person is None:
                raise ValueError(f"Person with id {person_id} not found")
            # Validate person status
            if person.status != "active":
                raise ValueError(
                    f"Cannot assign faces to person '{person.name}' (id={person_id}): "
                    f"status is '{person.status}', must be 'active'. "
                    f"Merged or hidden persons cannot receive new face assignments."
                )
            logger.info(
                f"Assigning {len(faces_to_assign)} faces to existing person: "
                f"{person.name} (id={person.id})"
            )
        else:
            # Find-or-create by name (existing behavior)
            assert person_name is not None  # Guaranteed by validation above
            logger.info(
                f"Labeling {len(faces_to_assign)} faces as '{person_name}' "
                f"(excluding {len(faces_excluded)} faces)"
            )
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

        # 6. Create prototypes (top 3 quality faces as exemplars, if person has none)
        sorted_faces = sorted(faces_to_assign, key=lambda f: f.quality_score or 0, reverse=True)
        prototypes_created = 0

        # Check if person already has prototypes
        existing_proto_query = select(func.count()).where(
            PersonPrototype.person_id == person.id
        )
        existing_proto_count = await self.db.scalar(existing_proto_query) or 0

        if existing_proto_count == 0:
            # New person (or person with no prototypes) -- create EXEMPLAR prototypes
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

            logger.info(
                f"Created {prototypes_created} exemplar prototypes for {person.name}"
            )
        else:
            logger.info(
                f"Person {person.name} already has {existing_proto_count} prototypes, "
                f"skipping prototype creation"
            )

        # 7. Create audit event for undo capability
        event = FaceAssignmentEvent(
            operation="ASSIGN_TO_PERSON",
            from_person_id=None,  # Faces were unassigned
            to_person_id=person.id,
            affected_photo_ids=list(set(f.asset_id for f in faces_to_assign)),
            affected_face_instance_ids=[str(fid) for fid in face_ids_assigned],
            face_count=len(face_ids_assigned),
            photo_count=len(set(f.asset_id for f in faces_to_assign)),
            note=f"Cluster labeling: assigned {len(face_ids_assigned)} faces to {person.name}",
        )
        self.db.add(event)

        await self.db.commit()

        logger.info(
            f"Labeled {len(faces_to_assign)} faces as person {person.name} "
            f"(created {prototypes_created} prototypes)"
        )

        # 8. Trigger find-more job (optional)
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
            "assignment_event_id": event.id,
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
