"""Incremental face assignment - matching new faces to known persons via prototypes."""

import logging
import uuid
from datetime import datetime

import numpy as np
from sqlalchemy import select, update
from sqlalchemy.orm import Session as SyncSession

logger = logging.getLogger(__name__)


class FaceAssigner:
    """Assigns new faces to known persons using prototype matching."""

    def __init__(
        self,
        db_session: SyncSession,
        similarity_threshold: float = 0.6,  # Cosine similarity threshold
        max_matches_per_face: int = 3,  # Consider top N matches
    ):
        """Initialize face assigner.

        Args:
            db_session: Synchronous SQLAlchemy session
            similarity_threshold: Minimum similarity for auto-assignment (0-1)
            max_matches_per_face: Number of prototype matches to consider
        """
        self.db = db_session
        self.similarity_threshold = similarity_threshold
        self.max_matches_per_face = max_matches_per_face

    def assign_new_faces(
        self,
        since: datetime | None = None,
        max_faces: int = 1000,
    ) -> dict:
        """Assign new/unlabeled faces using two-tier threshold system.

        Process:
        1. Get faces without person_id (optionally filtered by created_at)
        2. For each face, search against all prototypes
        3. If best match >= auto_assign_threshold -> auto-assign to person
        4. If best match >= suggestion_threshold -> create FaceSuggestion for review
        5. Otherwise, leave face unassigned

        Args:
            since: Only process faces created after this datetime
            max_faces: Maximum faces to process in this batch

        Returns:
            Summary dict with assignment statistics including auto_assigned
            and suggestions_created counts
        """
        from image_search_service.db.models import (
            FaceInstance,
            FaceSuggestion,
            PersonPrototype,
        )
        from image_search_service.services.config_service import SyncConfigService
        from image_search_service.vector.face_qdrant import get_face_qdrant_client

        # Get thresholds from config service
        config_service = SyncConfigService(self.db)
        auto_assign_threshold = config_service.get_float("face_auto_assign_threshold")
        suggestion_threshold = config_service.get_float("face_suggestion_threshold")

        logger.info(
            f"Using thresholds: auto_assign={auto_assign_threshold}, "
            f"suggestion={suggestion_threshold}"
        )

        qdrant = get_face_qdrant_client()

        # Check if we have any prototypes
        prototype_count_query = select(PersonPrototype)
        prototype_result = self.db.execute(prototype_count_query)
        prototypes = prototype_result.scalars().all()

        if not prototypes:
            logger.info("No prototypes available for assignment")
            return {
                "processed": 0,
                "auto_assigned": 0,
                "suggestions_created": 0,
                "unassigned": 0,
                "status": "no_prototypes",
            }

        logger.info(
            f"Found {len(prototypes)} prototypes for "
            f"{len(set(p.person_id for p in prototypes))} persons"
        )

        # Get unassigned faces
        query = select(FaceInstance).where(
            FaceInstance.person_id.is_(None),
            FaceInstance.cluster_id.is_(None),  # Also skip already-clustered faces
        )

        if since:
            query = query.where(FaceInstance.created_at >= since)

        query = query.limit(max_faces)
        result = self.db.execute(query)
        faces = result.scalars().all()

        if not faces:
            logger.info("No unassigned faces to process")
            return {
                "processed": 0,
                "auto_assigned": 0,
                "suggestions_created": 0,
                "unassigned": 0,
                "status": "no_new_faces",
            }

        logger.info(f"Processing {len(faces)} unassigned faces")

        assigned_count = 0
        suggestion_count = 0
        unassigned_count = 0
        assignments = {}  # person_id -> list of (face_id, qdrant_point_id)

        for face in faces:
            # Get face embedding from Qdrant
            embedding = self._get_face_embedding(face.qdrant_point_id)

            if embedding is None:
                logger.warning(f"Could not get embedding for face {face.id}")
                unassigned_count += 1
                continue

            # Search against prototypes using the lower threshold to catch suggestions
            matches = qdrant.search_against_prototypes(
                query_embedding=embedding,
                limit=self.max_matches_per_face,
                score_threshold=suggestion_threshold,
            )

            if not matches:
                unassigned_count += 1
                continue

            # Get best match
            best_match = matches[0]

            # Get person_id from prototype's payload
            person_id_str = best_match.payload.get("person_id")
            if not person_id_str:
                # Need to look up prototype to get person
                prototype_point_id = uuid.UUID(best_match.id)
                prototype = self.db.execute(
                    select(PersonPrototype).where(
                        PersonPrototype.qdrant_point_id == prototype_point_id
                    )
                ).scalar_one_or_none()

                if prototype:
                    person_id = prototype.person_id
                else:
                    unassigned_count += 1
                    continue
            else:
                person_id = uuid.UUID(person_id_str)

            # Apply two-tier threshold logic
            if best_match.score >= auto_assign_threshold:
                # AUTO-ASSIGN: High confidence match
                if person_id not in assignments:
                    assignments[person_id] = []
                assignments[person_id].append((face.id, face.qdrant_point_id))
                assigned_count += 1

                logger.debug(
                    f"Auto-assigned face {face.id} to person {person_id} "
                    f"(score: {best_match.score:.3f})"
                )

            elif best_match.score >= suggestion_threshold:
                # SUGGESTION: Medium confidence match - create suggestion for review
                # Check if suggestion already exists
                existing = self.db.execute(
                    select(FaceSuggestion).where(
                        FaceSuggestion.face_instance_id == face.id,
                        FaceSuggestion.suggested_person_id == person_id,
                        FaceSuggestion.status == "pending",
                    )
                ).scalar_one_or_none()

                if not existing:
                    # Get source face from prototype if available
                    source_face_id = face.id  # Default to self
                    prototype_point_id_str = best_match.id
                    if prototype_point_id_str:
                        prototype = self.db.execute(
                            select(PersonPrototype).where(
                                PersonPrototype.qdrant_point_id
                                == uuid.UUID(prototype_point_id_str)
                            )
                        ).scalar_one_or_none()
                        if prototype and prototype.face_instance_id:
                            source_face_id = prototype.face_instance_id

                    suggestion = FaceSuggestion(
                        face_instance_id=face.id,
                        suggested_person_id=person_id,
                        confidence=best_match.score,
                        source_face_id=source_face_id,
                        status="pending",
                    )
                    self.db.add(suggestion)
                    suggestion_count += 1

                    logger.debug(
                        f"Created suggestion for face {face.id} -> person {person_id} "
                        f"(score: {best_match.score:.3f})"
                    )
            else:
                unassigned_count += 1

        # Batch update database and Qdrant for auto-assignments
        for person_id, face_data in assignments.items():
            face_ids = [f[0] for f in face_data]
            qdrant_point_ids = [f[1] for f in face_data]

            # Update database
            stmt = (
                update(FaceInstance)
                .where(FaceInstance.id.in_(face_ids))
                .values(person_id=person_id)
            )
            self.db.execute(stmt)

            # Update Qdrant
            qdrant.update_person_ids(qdrant_point_ids, person_id)

        self.db.commit()

        persons_matched = len(assignments)

        logger.info(
            f"Assignment complete: {assigned_count} auto-assigned to "
            f"{persons_matched} persons, {suggestion_count} suggestions created, "
            f"{unassigned_count} unassigned"
        )

        return {
            "processed": len(faces),
            "auto_assigned": assigned_count,
            "suggestions_created": suggestion_count,
            "unassigned": unassigned_count,
            "persons_matched": persons_matched,
            "status": "completed",
        }

    def _get_face_embedding(self, qdrant_point_id: uuid.UUID) -> list[float] | None:
        """Get face embedding from Qdrant by point ID.

        Args:
            qdrant_point_id: UUID of the Qdrant point

        Returns:
            512-dim embedding vector or None if not found
        """
        from image_search_service.vector.face_qdrant import get_face_qdrant_client

        qdrant = get_face_qdrant_client()

        try:
            # Use retrieve to get specific point with vector
            points = qdrant.client.retrieve(
                collection_name="faces",
                ids=[str(qdrant_point_id)],
                with_vectors=True,
            )

            if points and points[0].vector:
                return points[0].vector
            return None
        except Exception as e:
            logger.error(f"Error retrieving embedding for {qdrant_point_id}: {e}")
            return None

    def compute_person_centroids(self) -> dict:
        """Compute centroid embeddings for all persons and update prototypes.

        This creates/updates CENTROID-type prototypes for each person,
        which can improve matching accuracy.

        Returns:
            Summary of centroids computed
        """
        from image_search_service.db.models import (
            FaceInstance,
            Person,
            PersonPrototype,
            PrototypeRole,
        )
        from image_search_service.vector.face_qdrant import get_face_qdrant_client

        qdrant = get_face_qdrant_client()

        # Get all active persons
        persons_query = select(Person).where(Person.status == "active")
        persons = self.db.execute(persons_query).scalars().all()

        centroids_computed = 0

        for person in persons:
            # Get all face embeddings for this person
            faces_query = select(FaceInstance).where(FaceInstance.person_id == person.id)
            faces = self.db.execute(faces_query).scalars().all()

            if len(faces) < 2:
                continue  # Need at least 2 faces for meaningful centroid

            # Collect embeddings
            embeddings = []
            for face in faces:
                embedding = self._get_face_embedding(face.qdrant_point_id)
                if embedding:
                    embeddings.append(embedding)

            if len(embeddings) < 2:
                continue

            # Compute centroid (mean of normalized vectors, then re-normalize)
            centroid = np.mean(embeddings, axis=0)
            centroid = centroid / np.linalg.norm(centroid)  # Re-normalize

            # Check if centroid prototype already exists
            existing_centroid = self.db.execute(
                select(PersonPrototype).where(
                    PersonPrototype.person_id == person.id,
                    PersonPrototype.role == PrototypeRole.CENTROID,
                )
            ).scalar_one_or_none()

            if existing_centroid:
                # Update existing centroid in Qdrant
                qdrant.client.set_payload(
                    collection_name="faces",
                    payload={"is_centroid": True},
                    points=[str(existing_centroid.qdrant_point_id)],
                )
                # Re-upload vector
                qdrant.upsert_face(
                    point_id=existing_centroid.qdrant_point_id,
                    embedding=centroid.tolist(),
                    asset_id=faces[0].asset_id,  # Use first face's asset
                    face_instance_id=faces[0].id,
                    detection_confidence=1.0,
                    person_id=person.id,
                    is_prototype=True,
                )
            else:
                # Create new centroid prototype
                centroid_point_id = uuid.uuid4()

                prototype = PersonPrototype(
                    person_id=person.id,
                    face_instance_id=None,  # Centroid has no specific face
                    qdrant_point_id=centroid_point_id,
                    role=PrototypeRole.CENTROID,
                )
                self.db.add(prototype)

                # Add centroid to Qdrant
                qdrant.upsert_face(
                    point_id=centroid_point_id,
                    embedding=centroid.tolist(),
                    asset_id=faces[0].asset_id,
                    face_instance_id=faces[0].id,
                    detection_confidence=1.0,
                    person_id=person.id,
                    is_prototype=True,
                )

            centroids_computed += 1

        self.db.commit()

        logger.info(f"Computed centroids for {centroids_computed} persons")

        return {
            "persons_processed": len(persons),
            "centroids_computed": centroids_computed,
        }


def get_face_assigner(
    db_session: SyncSession,
    similarity_threshold: float = 0.6,
) -> FaceAssigner:
    """Factory function for FaceAssigner.

    Args:
        db_session: Synchronous SQLAlchemy session
        similarity_threshold: Minimum similarity for auto-assignment (0-1)

    Returns:
        Configured FaceAssigner instance
    """
    return FaceAssigner(
        db_session=db_session,
        similarity_threshold=similarity_threshold,
    )
