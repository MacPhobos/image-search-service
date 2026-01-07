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

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.db.models import FaceInstance, PersonPrototype, PrototypeRole
from image_search_service.vector.face_qdrant import FaceQdrantClient

logger = logging.getLogger(__name__)


async def create_or_update_prototypes_legacy(
    db: AsyncSession,
    qdrant: FaceQdrantClient,
    person_id: UUID,
    newly_labeled_face_id: UUID,
    max_exemplars: int = 5,
    min_quality_threshold: float = 0.5,
) -> PersonPrototype | None:
    """Legacy prototype creation (pre-temporal mode).

    This is the original implementation used when temporal mode is disabled.
    It creates EXEMPLAR prototypes and prunes based on quality only.

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


async def create_or_update_prototypes_temporal(
    db: AsyncSession,
    qdrant: FaceQdrantClient,
    person_id: UUID,
    newly_labeled_face_id: UUID,
) -> PersonPrototype | None:
    """Temporal-aware prototype creation (Phase 4).

    When temporal mode is enabled, this function:
    1. Checks if the face can be a prototype (quality threshold)
    2. Classifies the face by age era
    3. Creates/updates prototype with TEMPORAL role if era needs coverage
    4. Otherwise creates EXEMPLAR if slots available
    5. Triggers pruning if exceeding max_total

    Args:
        db: Async database session
        qdrant: Qdrant client for vector operations
        person_id: UUID of the person
        newly_labeled_face_id: UUID of the newly labeled face

    Returns:
        PersonPrototype if created/updated, None if skipped
    """
    from sqlalchemy.orm import selectinload

    from image_search_service.core.config import get_settings
    from image_search_service.services import temporal_service

    settings = get_settings()

    # Step 1: Get face with asset for temporal metadata
    face_query = (
        select(FaceInstance)
        .options(selectinload(FaceInstance.asset))
        .where(FaceInstance.id == newly_labeled_face_id)
    )
    result = await db.execute(face_query)
    face = result.scalar_one_or_none()

    if not face:
        logger.warning(f"Face {newly_labeled_face_id} not found, cannot create prototype")
        return None

    # Step 2: Check quality threshold
    if face.quality_score is None or face.quality_score < settings.face_prototype_min_quality:
        logger.debug(
            f"Skipping prototype creation for face {newly_labeled_face_id}: "
            f"quality {face.quality_score} < threshold {settings.face_prototype_min_quality}"
        )
        return None

    # Step 3: Classify face by age era
    temporal_meta = temporal_service.extract_temporal_metadata(face.landmarks)
    age_era = temporal_service.classify_age_era(temporal_meta.get("age_estimate"))

    # Step 4: Check if prototype already exists for this face
    existing_proto_query = select(PersonPrototype).where(
        PersonPrototype.face_instance_id == newly_labeled_face_id
    )
    existing_proto_result = await db.execute(existing_proto_query)
    existing_proto = existing_proto_result.scalar_one_or_none()

    # Step 5: Determine if this era needs coverage
    needs_era_coverage = False
    if age_era:
        era_value = age_era.value
        # Check if era already has a prototype
        era_proto_query = select(func.count()).where(
            PersonPrototype.person_id == person_id,
            PersonPrototype.age_era_bucket == era_value,
        )
        result = await db.execute(era_proto_query)
        era_count = result.scalar() or 0
        needs_era_coverage = era_count == 0

    # Step 6: Assign role based on era coverage need
    era_bucket: str | None
    if needs_era_coverage and age_era:
        role = PrototypeRole.TEMPORAL
        era_bucket = age_era.value
        logger.info(
            f"Face {newly_labeled_face_id} fills gap for era {era_bucket}, "
            f"assigning TEMPORAL role"
        )
    else:
        role = PrototypeRole.EXEMPLAR
        era_bucket = age_era.value if age_era else None

    # Step 7: Create or update prototype
    if existing_proto:
        # Update existing prototype
        existing_proto.role = role
        existing_proto.age_era_bucket = era_bucket
        prototype = existing_proto
        logger.debug(f"Updated existing prototype {prototype.id} with role {role.value}")
    else:
        # Create new prototype
        prototype = PersonPrototype(
            person_id=person_id,
            face_instance_id=face.id,
            qdrant_point_id=face.qdrant_point_id,
            role=role,
            age_era_bucket=era_bucket,
        )
        db.add(prototype)
        logger.info(
            f"Created {role.value} prototype for person {person_id} "
            f"(face {newly_labeled_face_id}, era={era_bucket})"
        )

    await db.flush()

    # Step 8: Update Qdrant
    try:
        qdrant.update_payload(face.qdrant_point_id, {"is_prototype": True})
    except Exception as e:
        logger.error(f"Failed to update Qdrant payload for prototype {prototype.id}: {e}")

    # Step 9: No automatic pruning - only prune based on prototype_max_exemplars
    # in legacy mode or when explicitly requested via recompute_prototypes_for_person
    return prototype


async def create_or_update_prototypes(
    db: AsyncSession,
    qdrant: FaceQdrantClient,
    person_id: UUID,
    newly_labeled_face_id: UUID,
    max_exemplars: int = 5,
    min_quality_threshold: float = 0.5,
) -> PersonPrototype | None:
    """Create prototype from newly labeled face - temporal-aware if enabled.

    This is the main entry point for prototype creation. It delegates to either:
    - create_or_update_prototypes_temporal (when FACE_PROTOTYPE_TEMPORAL_MODE=True)
    - create_or_update_prototypes_legacy (when FACE_PROTOTYPE_TEMPORAL_MODE=False)

    Args:
        db: Async database session
        qdrant: Qdrant client for vector operations
        person_id: UUID of the person to create prototype for
        newly_labeled_face_id: UUID of the face instance that was just labeled
        max_exemplars: Maximum exemplars (ignored in temporal mode)
        min_quality_threshold: Minimum quality (ignored in temporal mode, uses config)

    Returns:
        PersonPrototype if created, None if skipped
    """
    from image_search_service.core.config import get_settings

    settings = get_settings()

    if settings.face_prototype_temporal_mode:
        # Use temporal-aware implementation
        return await create_or_update_prototypes_temporal(
            db=db,
            qdrant=qdrant,
            person_id=person_id,
            newly_labeled_face_id=newly_labeled_face_id,
        )
    else:
        # Use legacy implementation for backward compatibility
        return await create_or_update_prototypes_legacy(
            db=db,
            qdrant=qdrant,
            person_id=person_id,
            newly_labeled_face_id=newly_labeled_face_id,
            max_exemplars=max_exemplars,
            min_quality_threshold=min_quality_threshold,
        )


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


# ============ Phase 3: Manual Pinning API Functions ============


async def validate_pin_request(
    db: AsyncSession,
    person_id: UUID,
    face_instance_id: UUID,
    age_era_bucket: str | None,
    role: str,
) -> None:
    """Validate pin request. Raises HTTPException if invalid.

    Validates:
    - Face belongs to person
    - Pin quota not exceeded (max 3 PRIMARY, max 1 per era TEMPORAL)

    Args:
        db: Async database session
        person_id: UUID of the person
        face_instance_id: UUID of the face to pin
        age_era_bucket: Optional era bucket (required for TEMPORAL role)
        role: Prototype role ("primary" or "temporal")

    Raises:
        HTTPException: 400 for validation errors, 404 if face not found
    """
    from fastapi import HTTPException

    from image_search_service.core.config import get_settings

    settings = get_settings()

    # Verify face exists and belongs to person
    face = await db.get(FaceInstance, face_instance_id)
    if not face:
        raise HTTPException(status_code=404, detail="Face instance not found")

    if face.person_id != person_id:
        raise HTTPException(
            status_code=400,
            detail=f"Face {face_instance_id} does not belong to person {person_id}",
        )

    # Validate role
    role_upper = role.upper()
    if role_upper not in ["PRIMARY", "TEMPORAL"]:
        raise HTTPException(
            status_code=400, detail=f"Invalid role '{role}'. Must be 'primary' or 'temporal'"
        )

    # For TEMPORAL, require era bucket
    if role_upper == "TEMPORAL" and not age_era_bucket:
        raise HTTPException(
            status_code=400, detail="age_era_bucket is required for temporal prototypes"
        )

    # Check PRIMARY quota - limited by max_exemplars
    if role_upper == "PRIMARY":
        primary_count_query = select(func.count()).where(
            PersonPrototype.person_id == person_id,
            PersonPrototype.role == PrototypeRole.PRIMARY,
            PersonPrototype.is_pinned == True,  # noqa: E712
        )
        result = await db.execute(primary_count_query)
        primary_count = result.scalar() or 0

        # PRIMARY prototypes count toward max_exemplars limit
        max_primaries = settings.face_prototype_max_exemplars // 2  # Reserve half for primaries
        if primary_count >= max_primaries:
            max_ex = settings.face_prototype_max_exemplars
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Maximum {max_primaries} PRIMARY prototypes already pinned "
                    f"(based on max_exemplars={max_ex})"
                ),
            )

    # Check TEMPORAL era quota
    if role_upper == "TEMPORAL" and age_era_bucket:
        era_count_query = select(func.count()).where(
            PersonPrototype.person_id == person_id,
            PersonPrototype.role == PrototypeRole.TEMPORAL,
            PersonPrototype.age_era_bucket == age_era_bucket,
            PersonPrototype.is_pinned == True,  # noqa: E712
        )
        result = await db.execute(era_count_query)
        era_count = result.scalar() or 0

        if era_count >= 1:
            raise HTTPException(
                status_code=400,
                detail=f"Era '{age_era_bucket}' already has a pinned temporal prototype",
            )


async def pin_prototype(
    db: AsyncSession,
    qdrant: FaceQdrantClient,
    person_id: UUID,
    face_instance_id: UUID,
    age_era_bucket: str | None = None,
    role: str = "temporal",
    pinned_by: str | None = None,
) -> PersonPrototype:
    """Pin a face as prototype with optional era assignment.

    Validates:
    - Face belongs to person
    - Pin quota not exceeded (max 3 PRIMARY, max 1 per era TEMPORAL)
    - Creates or updates prototype with is_pinned=True

    Args:
        db: Async database session
        qdrant: Qdrant client for vector operations
        person_id: UUID of the person
        face_instance_id: UUID of the face to pin
        age_era_bucket: Optional era bucket (required for TEMPORAL)
        role: "primary" or "temporal"
        pinned_by: User identifier who pinned

    Returns:
        Created or updated PersonPrototype

    Raises:
        HTTPException: 400 for quota/validation errors, 404 if not found
    """
    from datetime import datetime as dt

    # Validate request
    await validate_pin_request(db, person_id, face_instance_id, age_era_bucket, role)

    # Get face instance
    face = await db.get(FaceInstance, face_instance_id)
    if not face:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Face instance not found")

    # Check if prototype already exists for this face
    existing_proto_query = select(PersonPrototype).where(
        PersonPrototype.face_instance_id == face_instance_id
    )
    existing_proto_result = await db.execute(existing_proto_query)
    existing_proto = existing_proto_result.scalar_one_or_none()

    role_enum = PrototypeRole.PRIMARY if role.upper() == "PRIMARY" else PrototypeRole.TEMPORAL

    if existing_proto:
        # Update existing prototype
        existing_proto.role = role_enum
        existing_proto.age_era_bucket = age_era_bucket
        existing_proto.is_pinned = True
        existing_proto.pinned_by = pinned_by
        existing_proto.pinned_at = dt.now()
        prototype = existing_proto
        logger.info(
            f"Updated prototype {prototype.id} for person {person_id} "
            f"(face {face_instance_id}, role={role})"
        )
    else:
        # Create new prototype
        prototype = PersonPrototype(
            person_id=person_id,
            face_instance_id=face.id,
            qdrant_point_id=face.qdrant_point_id,
            role=role_enum,
            age_era_bucket=age_era_bucket,
            is_pinned=True,
            pinned_by=pinned_by,
            pinned_at=dt.now(),
        )
        db.add(prototype)
        logger.info(
            f"Created pinned prototype {prototype.id} for person {person_id} "
            f"(face {face_instance_id}, role={role})"
        )

    await db.flush()

    # Update Qdrant payload
    try:
        qdrant.update_payload(face.qdrant_point_id, {"is_prototype": True})
    except Exception as e:
        logger.error(f"Failed to update Qdrant payload for prototype {prototype.id}: {e}")

    return prototype


async def unpin_prototype(
    db: AsyncSession,
    qdrant: FaceQdrantClient,
    person_id: UUID,
    prototype_id: UUID,
) -> bool:
    """Unpin a prototype. May trigger automatic replacement.

    Args:
        db: Async database session
        qdrant: Qdrant client for vector operations
        person_id: UUID of the person
        prototype_id: UUID of prototype to unpin

    Returns:
        True if unpinned successfully

    Raises:
        HTTPException: 404 if prototype not found
    """
    from fastapi import HTTPException

    # Get prototype
    prototype = await db.get(PersonPrototype, prototype_id)
    if not prototype:
        raise HTTPException(status_code=404, detail="Prototype not found")

    if prototype.person_id != person_id:
        raise HTTPException(
            status_code=400,
            detail=f"Prototype {prototype_id} does not belong to person {person_id}",
        )

    # Unpin the prototype
    prototype.is_pinned = False
    prototype.pinned_by = None
    prototype.pinned_at = None

    await db.flush()

    logger.info(f"Unpinned prototype {prototype_id} for person {person_id}")

    # Note: Automatic replacement logic can be added in Phase 4
    # For now, just unpin and don't delete

    return True


async def delete_prototype(
    db: AsyncSession,
    qdrant: FaceQdrantClient,
    person_id: UUID,
    prototype_id: UUID,
) -> bool:
    """Delete a prototype assignment entirely.

    This removes the prototype from the database and updates Qdrant to mark
    the face as no longer a prototype.

    Args:
        db: Async database session
        qdrant: Qdrant client for vector operations
        person_id: UUID of the person
        prototype_id: UUID of prototype to delete

    Returns:
        True if deleted successfully

    Raises:
        HTTPException: 404 if prototype not found, 400 if belongs to wrong person
    """
    from fastapi import HTTPException

    # Get prototype
    prototype = await db.get(PersonPrototype, prototype_id)
    if not prototype:
        raise HTTPException(status_code=404, detail="Prototype not found")

    if prototype.person_id != person_id:
        raise HTTPException(
            status_code=400,
            detail=f"Prototype {prototype_id} does not belong to person {person_id}",
        )

    # Store qdrant_point_id before deletion
    qdrant_point_id = prototype.qdrant_point_id

    # Delete the prototype
    await db.delete(prototype)
    await db.flush()

    # Update Qdrant: set is_prototype=False
    try:
        qdrant.update_payload(qdrant_point_id, {"is_prototype": False})
        logger.info(f"Deleted prototype {prototype_id} for person {person_id}")
    except Exception as e:
        logger.error(
            f"Failed to update Qdrant payload for deleted prototype {prototype_id}: {e}"
        )
        # Continue - prototype is deleted from DB, Qdrant update can be retried

    return True


async def get_prototypes_for_person(
    db: AsyncSession,
    person_id: UUID,
) -> list[PersonPrototype]:
    """Get all prototypes for a person with temporal metadata.

    Args:
        db: Async database session
        person_id: UUID of the person

    Returns:
        List of PersonPrototype objects
    """
    query = (
        select(PersonPrototype)
        .where(PersonPrototype.person_id == person_id)
        .order_by(
            PersonPrototype.is_pinned.desc(),
            PersonPrototype.role,
            PersonPrototype.age_era_bucket,
            PersonPrototype.created_at.desc(),
        )
    )
    result = await db.execute(query)
    return list(result.scalars().all())


async def get_temporal_coverage(
    db: AsyncSession,
    person_id: UUID,
) -> dict[str, list[str] | float | int]:
    """Calculate temporal coverage stats for a person.

    Args:
        db: Async database session
        person_id: UUID of the person

    Returns:
        Dict with coverage statistics:
        - covered_eras: list[str]
        - missing_eras: list[str]
        - coverage_percentage: float
        - total_prototypes: int
    """
    from image_search_service.db.models import AgeEraBucket

    # Get all prototypes
    prototypes = await get_prototypes_for_person(db, person_id)

    # Get covered eras (only TEMPORAL prototypes count)
    covered_eras = {
        proto.age_era_bucket
        for proto in prototypes
        if proto.role == PrototypeRole.TEMPORAL and proto.age_era_bucket
    }

    # All possible eras
    all_eras = {era.value for era in AgeEraBucket}

    # Missing eras
    missing_eras = all_eras - covered_eras

    # Coverage percentage
    coverage_percentage = (len(covered_eras) / len(all_eras)) * 100 if all_eras else 0.0

    return {
        "covered_eras": sorted(list(covered_eras)),
        "missing_eras": sorted(list(missing_eras)),
        "coverage_percentage": coverage_percentage,
        "total_prototypes": len(prototypes),
    }


# ============ Phase 4: Smart Selection Service ============


async def get_faces_with_temporal_metadata(
    db: AsyncSession,
    person_id: UUID,
) -> list[FaceInstance]:
    """Get all verified faces for person with temporal metadata extracted.

    Returns faces joined with their asset metadata for temporal classification.

    Args:
        db: Async database session
        person_id: UUID of the person

    Returns:
        List of FaceInstance objects (asset relationship eager-loaded)
    """
    from sqlalchemy.orm import selectinload

    # Query faces with asset relationship for photo timestamp
    query = (
        select(FaceInstance)
        .options(selectinload(FaceInstance.asset))
        .where(FaceInstance.person_id == person_id)
        .order_by(FaceInstance.quality_score.desc().nulls_last())
    )

    result = await db.execute(query)
    faces = list(result.scalars().all())

    return faces


async def select_temporal_prototypes(
    db: AsyncSession,
    qdrant: FaceQdrantClient,
    person_id: UUID,
    preserve_pins: bool = True,
) -> list[PersonPrototype]:
    """Select prototypes ensuring temporal diversity across age eras.

    Strategy:
    1. Keep all pinned prototypes (if preserve_pins=True)
    2. For each era without a pinned prototype:
       - Select highest quality face from that era
       - Assign TEMPORAL role
    3. Fill remaining slots with EXEMPLAR (best quality overall)
    4. Use FALLBACK for low-quality era representatives

    Args:
        db: Async database session
        qdrant: Qdrant client for vector operations
        person_id: UUID of the person
        preserve_pins: If True, preserve pinned prototypes

    Returns:
        List of newly created/updated prototypes
    """
    from image_search_service.core.config import get_settings
    from image_search_service.db.models import AgeEraBucket
    from image_search_service.services import temporal_service

    settings = get_settings()

    # Get all faces with temporal metadata
    faces = await get_faces_with_temporal_metadata(db, person_id)

    if not faces:
        logger.info(f"No faces found for person {person_id}, skipping prototype selection")
        return []

    # Classify faces by age era
    face_by_era: dict[str, list[tuple[FaceInstance, float]]] = {}
    for face in faces:
        # Extract temporal metadata from landmarks
        temporal_meta = temporal_service.extract_temporal_metadata(face.landmarks)
        age_era = temporal_service.classify_age_era(temporal_meta.get("age_estimate"))

        if age_era:
            era_bucket = age_era.value

            # Compute temporal quality score
            base_quality = face.quality_score or 0.0
            temporal_quality = temporal_service.compute_temporal_quality_score(
                base_quality=base_quality,
                pose=temporal_meta.get("pose"),
                bbox_area=temporal_meta.get("bbox_area"),
                age_confidence=temporal_meta.get("age_confidence"),
            )

            if era_bucket not in face_by_era:
                face_by_era[era_bucket] = []
            face_by_era[era_bucket].append((face, temporal_quality))

    # Sort faces within each era by quality (descending)
    for era in face_by_era:
        face_by_era[era].sort(key=lambda x: x[1], reverse=True)

    # Get existing prototypes to identify pinned ones
    existing_protos = await get_prototypes_for_person(db, person_id)
    pinned_protos = [p for p in existing_protos if p.is_pinned] if preserve_pins else []
    pinned_eras = {p.age_era_bucket for p in pinned_protos if p.age_era_bucket}
    pinned_face_ids = {p.face_instance_id for p in pinned_protos if p.face_instance_id}

    # Select prototypes: one per era (unless already pinned)
    selected_protos: list[PersonPrototype] = []
    used_face_ids = pinned_face_ids.copy()

    # Phase 1: Select TEMPORAL prototypes for uncovered eras
    for era in AgeEraBucket:
        era_value = era.value

        # Skip if era already has pinned prototype
        if era_value in pinned_eras:
            continue

        # Skip if no faces in this era
        if era_value not in face_by_era:
            continue

        # Select highest quality face from this era
        for face, quality in face_by_era[era_value]:
            if face.id not in used_face_ids:
                # Determine role based on quality threshold
                role = (
                    PrototypeRole.TEMPORAL
                    if quality >= settings.face_prototype_min_quality
                    else PrototypeRole.FALLBACK
                )

                # Check if prototype already exists for this face
                existing_proto_query = select(PersonPrototype).where(
                    PersonPrototype.face_instance_id == face.id
                )
                result = await db.execute(existing_proto_query)
                existing_proto = result.scalar_one_or_none()

                if existing_proto:
                    # Update existing prototype
                    existing_proto.role = role
                    existing_proto.age_era_bucket = era_value
                    prototype = existing_proto
                else:
                    # Create new prototype
                    prototype = PersonPrototype(
                        person_id=person_id,
                        face_instance_id=face.id,
                        qdrant_point_id=face.qdrant_point_id,
                        role=role,
                        age_era_bucket=era_value,
                    )
                    db.add(prototype)

                await db.flush()

                # Update Qdrant
                try:
                    qdrant.update_payload(face.qdrant_point_id, {"is_prototype": True})
                except Exception as e:
                    logger.error(f"Failed to update Qdrant for prototype {prototype.id}: {e}")

                selected_protos.append(prototype)
                used_face_ids.add(face.id)
                logger.info(
                    f"Selected {role.value} prototype for era {era_value} "
                    f"(face {face.id}, quality={quality:.2f})"
                )
                break

    # Phase 2: Fill remaining slots with EXEMPLAR (highest quality overall)
    max_exemplars = settings.face_prototype_max_exemplars
    current_count = len(pinned_protos) + len(selected_protos)

    if current_count < max_exemplars:
        # Get all faces sorted by quality
        all_faces_sorted = sorted(
            [(f, f.quality_score or 0.0) for f in faces], key=lambda x: x[1], reverse=True
        )

        for face, quality in all_faces_sorted:
            if current_count >= max_exemplars:
                break

            if face.id in used_face_ids:
                continue

            if quality < settings.face_prototype_min_quality:
                continue

            # Check if prototype already exists
            existing_proto_query = select(PersonPrototype).where(
                PersonPrototype.face_instance_id == face.id
            )
            result = await db.execute(existing_proto_query)
            existing_proto = result.scalar_one_or_none()

            if existing_proto:
                existing_proto.role = PrototypeRole.EXEMPLAR
                prototype = existing_proto
            else:
                prototype = PersonPrototype(
                    person_id=person_id,
                    face_instance_id=face.id,
                    qdrant_point_id=face.qdrant_point_id,
                    role=PrototypeRole.EXEMPLAR,
                )
                db.add(prototype)

            await db.flush()

            try:
                qdrant.update_payload(face.qdrant_point_id, {"is_prototype": True})
            except Exception as e:
                logger.error(f"Failed to update Qdrant for prototype {prototype.id}: {e}")

            selected_protos.append(prototype)
            used_face_ids.add(face.id)
            current_count += 1
            logger.info(
                f"Selected EXEMPLAR prototype (face {face.id}, quality={quality:.2f})"
            )

    return selected_protos


async def prune_temporal_prototypes(
    db: AsyncSession,
    qdrant: FaceQdrantClient,
    person_id: UUID,
    max_exemplars: int,
    preserve_pins: bool = True,
) -> list[UUID]:
    """Prune prototypes to max_exemplars while respecting temporal coverage.

    Priority (highest to lowest):
    1. Pinned PRIMARY prototypes (never pruned if preserve_pins=True)
    2. Pinned TEMPORAL prototypes per era
    3. Auto-selected TEMPORAL prototypes (one per era)
    4. EXEMPLAR prototypes (fill remaining)
    5. FALLBACK prototypes (only when no better option)

    Args:
        db: Async database session
        qdrant: Qdrant client for vector operations
        person_id: UUID of the person
        max_exemplars: Maximum total prototypes allowed
        preserve_pins: If True, never prune pinned prototypes

    Returns:
        List of deleted prototype IDs
    """
    # Get all prototypes
    all_protos = await get_prototypes_for_person(db, person_id)

    if len(all_protos) <= max_exemplars:
        return []

    # Separate pinned and unpinned
    pinned = [p for p in all_protos if p.is_pinned and preserve_pins]
    unpinned = [p for p in all_protos if not (p.is_pinned and preserve_pins)]

    # If pinned prototypes exceed limit, we have a problem
    if len(pinned) > max_exemplars:
        logger.warning(
            f"Person {person_id} has {len(pinned)} pinned prototypes "
            f"exceeding max_exemplars={max_exemplars}"
        )
        # Don't prune pinned if preserve_pins=True
        return []

    # Calculate how many unpinned we can keep
    slots_available = max_exemplars - len(pinned)

    if slots_available <= 0:
        # Must prune all unpinned
        to_delete = unpinned
    else:
        # Priority sort unpinned prototypes
        # Priority: TEMPORAL > EXEMPLAR > FALLBACK
        # Within priority, keep highest quality

        # Load face instances for quality scores
        face_ids = [p.face_instance_id for p in unpinned if p.face_instance_id]
        if face_ids:
            face_query = select(FaceInstance).where(FaceInstance.id.in_(face_ids))
            result = await db.execute(face_query)
            faces_by_id = {f.id: f for f in result.scalars().all()}
        else:
            faces_by_id = {}

        # Build priority list
        proto_priority: list[tuple[PersonPrototype, int, float]] = []
        for proto in unpinned:
            # Assign priority score (higher = keep)
            if proto.role == PrototypeRole.TEMPORAL:
                priority = 3
            elif proto.role == PrototypeRole.EXEMPLAR:
                priority = 2
            elif proto.role == PrototypeRole.FALLBACK:
                priority = 1
            else:
                priority = 0  # CENTROID, etc.

            # Get quality score
            quality = 0.0
            if proto.face_instance_id and proto.face_instance_id in faces_by_id:
                face = faces_by_id[proto.face_instance_id]
                quality = face.quality_score or 0.0

            proto_priority.append((proto, priority, quality))

        # Sort by priority (desc), then quality (desc)
        proto_priority.sort(key=lambda x: (x[1], x[2]), reverse=True)

        # Keep top slots_available, delete rest
        to_delete = [p[0] for p in proto_priority[slots_available:]]

    # Delete prototypes
    deleted_ids: list[UUID] = []
    for proto in to_delete:
        deleted_ids.append(proto.id)

        # Update Qdrant
        try:
            qdrant.update_payload(proto.qdrant_point_id, {"is_prototype": False})
        except Exception as e:
            logger.error(f"Failed to update Qdrant for deleted prototype {proto.id}: {e}")

        await db.delete(proto)

    await db.flush()

    logger.info(
        f"Pruned {len(deleted_ids)} prototypes for person {person_id}, "
        f"kept {len(all_protos) - len(deleted_ids)}"
    )

    return deleted_ids


async def recompute_prototypes_for_person(
    db: AsyncSession,
    qdrant: FaceQdrantClient,
    person_id: UUID,
    preserve_pins: bool = True,
) -> dict[str, int | dict[str, list[str] | float | int]]:
    """Full recomputation of prototypes with temporal diversity.

    This orchestrator function:
    1. Selects temporal prototypes ensuring era coverage
    2. Prunes excess prototypes while respecting pins and temporal coverage
    3. Recalculates coverage statistics

    Args:
        db: Async database session
        qdrant: Qdrant client for vector operations
        person_id: UUID of the person
        preserve_pins: If True, preserve pinned prototypes

    Returns:
        Dict with:
            - prototypes_created: int
            - prototypes_removed: int
            - coverage: TemporalCoverage dict
    """
    from image_search_service.core.config import get_settings

    settings = get_settings()

    # Count prototypes before
    initial_protos = await get_prototypes_for_person(db, person_id)
    initial_count = len(initial_protos)

    # Phase 1: Select temporal prototypes
    await select_temporal_prototypes(
        db=db,
        qdrant=qdrant,
        person_id=person_id,
        preserve_pins=preserve_pins,
    )

    # Phase 2: Prune if needed
    deleted_ids = await prune_temporal_prototypes(
        db=db,
        qdrant=qdrant,
        person_id=person_id,
        max_exemplars=settings.face_prototype_max_exemplars,
        preserve_pins=preserve_pins,
    )

    # Commit changes
    await db.commit()

    # Phase 3: Calculate coverage
    coverage = await get_temporal_coverage(db, person_id)

    # Count final prototypes
    final_protos = await get_prototypes_for_person(db, person_id)
    final_count = len(final_protos)

    prototypes_created = max(0, final_count - initial_count + len(deleted_ids))
    prototypes_removed = len(deleted_ids)

    logger.info(
        f"Recomputed prototypes for person {person_id}: "
        f"created={prototypes_created}, removed={prototypes_removed}, "
        f"coverage={coverage['coverage_percentage']:.1f}%"
    )

    return {
        "prototypes_created": prototypes_created,
        "prototypes_removed": prototypes_removed,
        "coverage": coverage,
    }
