"""Prototype management service for person face recognition.

This service handles creation and pruning of person prototypes from individual
face→person assignments. Prototypes are high-quality exemplar faces used for
generating face suggestions.

Key concepts:
- Prototypes are created from user-verified face assignments (not clusters)
- Each prototype has role=EXEMPLAR and links to a specific face instance
- System maintains max N prototypes per person, prioritizing quality and diversity
- Prototypes are marked in Qdrant with is_prototype=True for efficient search
"""

import logging
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.db.models import FaceInstance, PersonPrototype, PrototypeRole
from image_search_service.vector.face_qdrant import FaceQdrantClient

logger = logging.getLogger(__name__)


async def create_or_update_prototypes(
    db: AsyncSession,
    qdrant: FaceQdrantClient,
    person_id: UUID,
    newly_labeled_face_id: UUID,
    max_exemplars: int = 5,
    min_quality_threshold: float = 0.5,
) -> PersonPrototype | None:
    """Create prototype from newly labeled face, then prune if needed.

    This function is called whenever a user assigns a face to a person. It:
    1. Checks if the face meets quality requirements
    2. Creates a new prototype if it doesn't already exist
    3. Prunes excess prototypes if the person exceeds max_exemplars

    Args:
        db: Async database session
        qdrant: Qdrant client for vector operations
        person_id: UUID of the person to create prototype for
        newly_labeled_face_id: UUID of the face instance that was just labeled
        max_exemplars: Maximum number of exemplar prototypes per person
        min_quality_threshold: Minimum quality score (0.0-1.0) for prototype creation

    Returns:
        PersonPrototype if created, None if skipped (quality too low or already exists)
    """
    # Step 1: Get face and check quality
    face = await db.get(FaceInstance, newly_labeled_face_id)
    if not face:
        logger.warning(f"Face {newly_labeled_face_id} not found, cannot create prototype")
        return None

    # Check quality threshold
    if face.quality_score is None or face.quality_score < min_quality_threshold:
        logger.debug(
            f"Skipping prototype creation for face {newly_labeled_face_id}: "
            f"quality {face.quality_score} < threshold {min_quality_threshold}"
        )
        return None

    # Step 2: Check if prototype already exists for this face
    existing_proto_query = select(PersonPrototype).where(
        PersonPrototype.face_instance_id == newly_labeled_face_id
    )
    existing_proto_result = await db.execute(existing_proto_query)
    existing_proto = existing_proto_result.scalar_one_or_none()

    if existing_proto:
        logger.debug(
            f"Prototype already exists for face {newly_labeled_face_id}, skipping creation"
        )
        return existing_proto

    # Step 3: Create new PersonPrototype with role=EXEMPLAR
    prototype = PersonPrototype(
        person_id=person_id,
        face_instance_id=face.id,
        qdrant_point_id=face.qdrant_point_id,
        role=PrototypeRole.EXEMPLAR,
    )
    db.add(prototype)
    await db.flush()

    # Step 4: Mark face in Qdrant with is_prototype=True
    try:
        qdrant.update_payload(face.qdrant_point_id, {"is_prototype": True})
        logger.info(
            f"Created prototype {prototype.id} for person {person_id} "
            f"from face {newly_labeled_face_id} (quality={face.quality_score:.2f})"
        )
    except Exception as e:
        logger.error(f"Failed to update Qdrant payload for prototype {prototype.id}: {e}")
        # Continue - prototype is in DB, Qdrant update can be retried
        # Don't rollback the prototype creation

    # Step 5: Get all EXEMPLAR prototypes for this person
    all_exemplars_query = (
        select(PersonPrototype)
        .where(
            PersonPrototype.person_id == person_id,
            PersonPrototype.role == PrototypeRole.EXEMPLAR,
        )
        .order_by(PersonPrototype.created_at)
    )
    all_exemplars_result = await db.execute(all_exemplars_query)
    all_exemplars = list(all_exemplars_result.scalars().all())

    # Step 6: Prune if exceeding max_exemplars
    if len(all_exemplars) > max_exemplars:
        logger.info(
            f"Person {person_id} has {len(all_exemplars)} prototypes, "
            f"pruning to {max_exemplars}"
        )
        await prune_prototypes(db, qdrant, person_id, all_exemplars, max_exemplars)

    return prototype


async def prune_prototypes(
    db: AsyncSession,
    qdrant: FaceQdrantClient,
    person_id: UUID,
    exemplars: list[PersonPrototype],
    max_exemplars: int,
) -> list[UUID]:
    """Keep top quality prototypes while ensuring photo diversity.

    Pruning strategy:
    1. Sort prototypes by quality score (descending)
    2. Select top N, preferring different asset_ids for diversity
    3. Delete non-selected prototypes from database
    4. Update Qdrant to remove is_prototype=True flag

    Args:
        db: Async database session
        qdrant: Qdrant client for vector operations
        person_id: UUID of the person being pruned
        exemplars: List of all current exemplar prototypes for this person
        max_exemplars: Maximum number to keep

    Returns:
        List of deleted prototype UUIDs
    """
    if len(exemplars) <= max_exemplars:
        # No pruning needed
        return []

    # Load face instances for all prototypes to access quality scores
    face_ids = [proto.face_instance_id for proto in exemplars if proto.face_instance_id]
    if not face_ids:
        logger.warning(f"No face instances found for prototypes of person {person_id}")
        return []

    face_query = select(FaceInstance).where(FaceInstance.id.in_(face_ids))
    face_result = await db.execute(face_query)
    faces_by_id = {face.id: face for face in face_result.scalars().all()}

    # Build list of (prototype, quality_score, asset_id) tuples
    proto_data: list[tuple[PersonPrototype, float, int | None]] = []
    for proto in exemplars:
        if proto.face_instance_id and proto.face_instance_id in faces_by_id:
            face = faces_by_id[proto.face_instance_id]
            quality = face.quality_score or 0.0
            proto_data.append((proto, quality, face.asset_id))
        else:
            # Prototype without face instance (orphaned) - add with low quality
            proto_data.append((proto, 0.0, None))

    # Sort by quality descending
    proto_data.sort(key=lambda x: x[1], reverse=True)

    # Select top N with diversity preference (prefer different photos)
    selected_protos: list[PersonPrototype] = []
    seen_asset_ids: set[int] = set()

    # First pass: select prototypes from unique photos
    for proto, quality, asset_id in proto_data:
        if len(selected_protos) >= max_exemplars:
            break
        if asset_id is not None and asset_id not in seen_asset_ids:
            selected_protos.append(proto)
            seen_asset_ids.add(asset_id)

    # Second pass: fill remaining slots with highest quality regardless of diversity
    if len(selected_protos) < max_exemplars:
        for proto, quality, asset_id in proto_data:
            if len(selected_protos) >= max_exemplars:
                break
            if proto not in selected_protos:
                selected_protos.append(proto)

    # Identify prototypes to delete
    selected_ids = {proto.id for proto in selected_protos}
    to_delete = [proto for proto in exemplars if proto.id not in selected_ids]

    if not to_delete:
        return []

    # Delete from database
    deleted_ids = []
    qdrant_point_ids = []

    for proto in to_delete:
        deleted_ids.append(proto.id)
        qdrant_point_ids.append(proto.qdrant_point_id)
        await db.delete(proto)

    await db.flush()

    # Update Qdrant: set is_prototype=False for deleted prototypes
    try:
        if qdrant_point_ids:
            for point_id in qdrant_point_ids:
                qdrant.update_payload(point_id, {"is_prototype": False})
            logger.info(
                f"Pruned {len(deleted_ids)} prototypes for person {person_id}, "
                f"kept {len(selected_protos)}"
            )
    except Exception as e:
        logger.error(f"Failed to update Qdrant for pruned prototypes: {e}")
        # Continue - prototypes are deleted from DB, Qdrant cleanup can be retried

    return deleted_ids


async def get_prototype_count(db: AsyncSession, person_id: UUID) -> int:
    """Get count of active prototypes for a person.

    Args:
        db: Async database session
        person_id: UUID of the person

    Returns:
        Count of EXEMPLAR prototypes
    """
    from sqlalchemy import func

    count_query = select(func.count()).where(
        PersonPrototype.person_id == person_id,
        PersonPrototype.role == PrototypeRole.EXEMPLAR,
    )
    result = await db.execute(count_query)
    return result.scalar() or 0
