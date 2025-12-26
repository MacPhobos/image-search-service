"""Admin service for destructive operations."""

import os
from datetime import UTC, datetime
from math import sqrt

from fastapi import HTTPException, status
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.api.admin_schemas import (
    BoundingBoxExport,
    DeleteAllDataResponse,
    ExportMetadata,
    FaceMappingExport,
    FaceMappingResult,
    ImportOptions,
    ImportResponse,
    PersonExport,
    PersonImportResult,
    PersonMetadataExport,
)
from image_search_service.core.logging import get_logger
from image_search_service.db.models import FaceInstance, ImageAsset, Person, PersonStatus
from image_search_service.vector import qdrant
from image_search_service.vector.face_qdrant import FaceQdrantClient, get_face_qdrant_client

logger = get_logger(__name__)


async def delete_all_data(
    db: AsyncSession,
    confirm: bool,
    confirmation_text: str,
    reason: str | None,
) -> DeleteAllDataResponse:
    """Delete all application data from Qdrant and PostgreSQL.

    WARNING: This is highly destructive and cannot be undone. It deletes:
    - All Qdrant vector collections (main + faces)
    - All PostgreSQL table data (except alembic_version for migrations)

    Safety: Requires confirm=true AND confirmation_text="DELETE ALL DATA"

    Args:
        db: Database session
        confirm: Must be True
        confirmation_text: Must exactly match "DELETE ALL DATA"
        reason: Optional reason for deletion (audit trail)

    Returns:
        DeleteAllDataResponse with deletion summary

    Raises:
        HTTPException: 400 if confirmations are invalid
    """
    # Validate confirmations
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must confirm deletion by setting confirm=true",
        )

    if confirmation_text != "DELETE ALL DATA":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="confirmationText must exactly match 'DELETE ALL DATA'",
        )

    logger.warning(
        "ADMIN DELETE ALL DATA initiated. Reason: %s",
        reason or "No reason provided",
    )

    # Step 1: Get current Alembic version before deletion
    alembic_version = await _get_alembic_version(db)
    logger.info(f"Current Alembic version: {alembic_version}")

    # Step 2: Delete Qdrant collections
    qdrant_collections_deleted: dict[str, int] = {}

    # Delete main collection
    try:
        main_deleted = qdrant.reset_collection()
        qdrant_collections_deleted["main"] = main_deleted
        logger.info(f"Deleted {main_deleted} vectors from main collection")
    except Exception as e:
        logger.error(f"Failed to delete main collection: {e}")
        qdrant_collections_deleted["main"] = 0

    # Delete faces collection
    try:
        face_client = get_face_qdrant_client()
        faces_deleted = face_client.reset_collection()
        qdrant_collections_deleted["faces"] = faces_deleted
        logger.info(f"Deleted {faces_deleted} vectors from faces collection")
    except Exception as e:
        logger.error(f"Failed to delete faces collection: {e}")
        qdrant_collections_deleted["faces"] = 0

    # Step 3: Truncate PostgreSQL tables (except alembic_version)
    postgres_tables_truncated = await _truncate_application_tables(db)
    logger.info(f"Truncated {len(postgres_tables_truncated)} PostgreSQL tables")

    # Build response
    total_qdrant = sum(qdrant_collections_deleted.values())
    total_postgres = sum(postgres_tables_truncated.values())

    logger.warning(
        "ADMIN DELETE ALL DATA completed: "
        f"{total_qdrant} Qdrant vectors, "
        f"{total_postgres} PostgreSQL rows deleted. "
        f"Alembic version preserved: {alembic_version}"
    )

    return DeleteAllDataResponse(
        qdrant_collections_deleted=qdrant_collections_deleted,
        postgres_tables_truncated=postgres_tables_truncated,
        alembic_version_preserved=alembic_version,
        message=(
            f"Deleted {total_qdrant} Qdrant vectors "
            f"and {total_postgres} PostgreSQL rows. "
            f"Migration version {alembic_version} preserved."
        ),
        timestamp=datetime.now(UTC),
    )


async def _get_alembic_version(db: AsyncSession) -> str:
    """Get current Alembic migration version.

    Args:
        db: Database session

    Returns:
        Current Alembic version string (e.g., "012_add_faces")
    """
    try:
        result = await db.execute(text("SELECT version_num FROM alembic_version"))
        row = result.fetchone()
        if row:
            return str(row[0])
        return "unknown"
    except Exception as e:
        logger.warning(f"Failed to get Alembic version: {e}")
        return "unknown"


async def _truncate_application_tables(db: AsyncSession) -> dict[str, int]:
    """Truncate all application tables except alembic_version.

    Uses TRUNCATE CASCADE to handle foreign key constraints.

    Args:
        db: Database session

    Returns:
        Dict of table name to row count deleted
    """
    # Define tables in CASCADE order (children before parents where possible)
    # TRUNCATE CASCADE handles dependencies, but we order for clarity
    tables_to_truncate = [
        "face_assignment_events",
        "person_prototypes",
        "face_instances",
        "persons",
        "vector_deletion_logs",
        "training_evidence",
        "training_jobs",
        "training_subdirectories",
        "training_sessions",
        "categories",
        "image_assets",
    ]

    truncated_counts: dict[str, int] = {}

    for table_name in tables_to_truncate:
        try:
            # Get row count before truncation
            count_result = await db.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            row_count = count_result.scalar() or 0

            # Try TRUNCATE CASCADE first (PostgreSQL), fall back to DELETE (SQLite)
            try:
                await db.execute(text(f"TRUNCATE TABLE {table_name} CASCADE"))
            except Exception:
                # SQLite doesn't support TRUNCATE, use DELETE instead
                await db.execute(text(f"DELETE FROM {table_name}"))

            truncated_counts[table_name] = row_count
            logger.info(f"Truncated table {table_name}: {row_count} rows")

        except Exception as e:
            logger.error(f"Failed to truncate table {table_name}: {e}")
            truncated_counts[table_name] = 0

    # Commit truncations
    await db.commit()

    return truncated_counts


async def export_person_metadata(
    db: AsyncSession,
    max_faces_per_person: int = 100,
) -> PersonMetadataExport:
    """Export all active persons with their face-to-image mappings.

    For each person, selects up to max_faces_per_person faces,
    ordered by quality_score (descending), then detection_confidence.

    Returns JSON-serializable export structure including:
    - Person names and status
    - Face mappings with image paths and bounding boxes

    Args:
        db: Database session
        max_faces_per_person: Maximum number of faces to export per person (default 100)

    Returns:
        PersonMetadataExport with all active persons and their face mappings
    """
    from sqlalchemy import select

    logger.info("Starting person metadata export (max_faces_per_person=%d)", max_faces_per_person)

    # Step 1: Get all active persons ordered by name
    stmt = select(Person).where(Person.status == PersonStatus.ACTIVE).order_by(Person.name)
    result = await db.execute(stmt)
    persons = result.scalars().all()

    logger.info("Found %d active persons to export", len(persons))

    # Step 2: For each person, get face mappings with image paths
    person_exports: list[PersonExport] = []
    total_face_mappings = 0

    for person in persons:
        # Query faces with image paths, ordered by quality
        face_stmt = (
            select(FaceInstance, ImageAsset.path)
            .join(ImageAsset, FaceInstance.asset_id == ImageAsset.id)
            .where(FaceInstance.person_id == person.id)
            .order_by(
                FaceInstance.quality_score.desc().nullslast(),
                FaceInstance.detection_confidence.desc(),
            )
            .limit(max_faces_per_person)
        )

        face_result = await db.execute(face_stmt)
        face_rows = face_result.all()

        # Build face mappings for this person
        face_mappings: list[FaceMappingExport] = []
        for face_instance, image_path in face_rows:
            face_mapping = FaceMappingExport(
                image_path=image_path,
                bounding_box=BoundingBoxExport(
                    x=face_instance.bbox_x,
                    y=face_instance.bbox_y,
                    width=face_instance.bbox_w,
                    height=face_instance.bbox_h,
                ),
                detection_confidence=face_instance.detection_confidence,
                quality_score=face_instance.quality_score,
            )
            face_mappings.append(face_mapping)

        # Only include persons with at least one face mapping
        if face_mappings:
            person_export = PersonExport(
                name=person.name,
                status=person.status.value,
                face_mappings=face_mappings,
            )
            person_exports.append(person_export)
            total_face_mappings += len(face_mappings)

    logger.info(
        "Export completed: %d persons with %d total face mappings",
        len(person_exports),
        total_face_mappings,
    )

    # Step 3: Build export response
    return PersonMetadataExport(
        version="1.0",
        exported_at=datetime.now(UTC),
        metadata=ExportMetadata(
            total_persons=len(person_exports),
            total_face_mappings=total_face_mappings,
            export_format="seed",
        ),
        persons=person_exports,
    )


def _match_face_by_bbox(
    detected_faces: list[FaceInstance],
    target_x: int,
    target_y: int,
    target_w: int,
    target_h: int,
    tolerance: int,
) -> FaceInstance | None:
    """Find detected face matching target bounding box within tolerance.

    Returns the closest matching face if within tolerance, else None.

    Args:
        detected_faces: List of FaceInstance objects from database
        target_x: Target bounding box x coordinate
        target_y: Target bounding box y coordinate
        target_w: Target bounding box width
        target_h: Target bounding box height
        tolerance: Maximum pixel difference allowed for each dimension

    Returns:
        Closest matching FaceInstance or None if no match within tolerance
    """
    candidates = []
    for face in detected_faces:
        if (
            abs(face.bbox_x - target_x) <= tolerance
            and abs(face.bbox_y - target_y) <= tolerance
            and abs(face.bbox_w - target_w) <= tolerance
            and abs(face.bbox_h - target_h) <= tolerance
        ):
            # Calculate Euclidean distance for all four dimensions
            distance = sqrt(
                (face.bbox_x - target_x) ** 2
                + (face.bbox_y - target_y) ** 2
                + (face.bbox_w - target_w) ** 2
                + (face.bbox_h - target_h) ** 2
            )
            candidates.append((face, distance))

    if candidates:
        # Return closest match
        return min(candidates, key=lambda x: x[1])[0]
    return None


async def import_person_metadata(
    db: AsyncSession,
    import_data: PersonMetadataExport,
    options: ImportOptions,
    face_qdrant: FaceQdrantClient,
) -> ImportResponse:
    """Import persons and face mappings from export file.

    For each person:
    1. Create person (or find existing by name)
    2. For each face mapping:
       - Check if image file exists
       - Find faces already in database for image
       - Match face by bounding box
       - Assign face to person (if not dry_run)

    Args:
        db: Database session
        import_data: PersonMetadataExport with persons and face mappings
        options: Import options (dry_run, tolerance, skip_missing_images)
        face_qdrant: FaceQdrantClient for updating Qdrant vectors

    Returns:
        ImportResponse with import results and statistics
    """
    logger.info(
        "Starting person metadata import: %d persons, dry_run=%s, tolerance=%d",
        len(import_data.persons),
        options.dry_run,
        options.tolerance_pixels,
    )

    person_results: list[PersonImportResult] = []
    errors: list[str] = []
    persons_created = 0
    persons_existing = 0
    total_faces_matched = 0
    total_faces_not_found = 0
    total_images_missing = 0

    # Process each person
    for person_data in import_data.persons:
        try:
            # Step 1: Find or create person (case-insensitive name lookup)
            stmt = select(Person).where(func.lower(Person.name) == person_data.name.lower())
            result = await db.execute(stmt)
            existing_person = result.scalar_one_or_none()

            person: Person | None = None
            person_id_str: str | None = None

            if existing_person:
                person = existing_person
                person_status = "existing"
                persons_existing += 1
                person_id_str = str(person.id)
                logger.info("Found existing person: %s (id=%s)", person.name, person.id)
            else:
                if options.dry_run:
                    # In dry run, simulate person creation
                    person_status = "created"
                    persons_created += 1
                    person_id_str = None
                    logger.info("Dry run: would create person '%s'", person_data.name)
                else:
                    # Create new person
                    person = Person(
                        name=person_data.name,
                        status=PersonStatus.ACTIVE,
                    )
                    db.add(person)
                    await db.flush()  # Get person.id without committing
                    person_status = "created"
                    persons_created += 1
                    person_id_str = str(person.id)
                    logger.info("Created new person: %s (id=%s)", person.name, person.id)

            # Track results for this person
            faces_matched = 0
            faces_not_found = 0
            images_missing = 0
            face_mapping_results: list[FaceMappingResult] = []

            # Step 2: Process each face mapping for this person
            for face_mapping in person_data.face_mappings:
                image_path = face_mapping.image_path
                bbox = face_mapping.bounding_box

                # Check if image file exists (if skip_missing_images is enabled)
                if options.skip_missing_images and not os.path.exists(image_path):
                    images_missing += 1
                    face_mapping_results.append(
                        FaceMappingResult(
                            image_path=image_path,
                            status="image_missing",
                            error="Image file not found on filesystem",
                        )
                    )
                    logger.debug("Image not found: %s", image_path)
                    continue

                # Step 3: Find image asset in database
                asset_stmt = select(ImageAsset).where(ImageAsset.path == image_path)
                asset_result = await db.execute(asset_stmt)
                asset = asset_result.scalar_one_or_none()

                if not asset:
                    faces_not_found += 1
                    face_mapping_results.append(
                        FaceMappingResult(
                            image_path=image_path,
                            status="not_found",
                            error="Image asset not found in database",
                        )
                    )
                    logger.debug("Image asset not in database: %s", image_path)
                    continue

                # Step 4: Get faces for this image
                faces_stmt = select(FaceInstance).where(FaceInstance.asset_id == asset.id)
                faces_result = await db.execute(faces_stmt)
                detected_faces = list(faces_result.scalars().all())

                if not detected_faces:
                    faces_not_found += 1
                    face_mapping_results.append(
                        FaceMappingResult(
                            image_path=image_path,
                            status="not_found",
                            error="No faces detected in image",
                        )
                    )
                    logger.debug("No faces found for image: %s", image_path)
                    continue

                # Step 5: Match face by bounding box
                matched_face = _match_face_by_bbox(
                    detected_faces=detected_faces,
                    target_x=bbox.x,
                    target_y=bbox.y,
                    target_w=bbox.width,
                    target_h=bbox.height,
                    tolerance=options.tolerance_pixels,
                )

                if not matched_face:
                    faces_not_found += 1
                    error_msg = (
                        f"No face matched bounding box within "
                        f"{options.tolerance_pixels}px tolerance"
                    )
                    face_mapping_results.append(
                        FaceMappingResult(
                            image_path=image_path,
                            status="not_found",
                            error=error_msg,
                        )
                    )
                    logger.debug(
                        "No face matched bbox (%d,%d,%d,%d) in %s",
                        bbox.x,
                        bbox.y,
                        bbox.width,
                        bbox.height,
                        image_path,
                    )
                    continue

                # Step 6: Assign face to person (if not dry_run and person exists)
                if not options.dry_run and person is not None:
                    # Update database
                    matched_face.person_id = person.id

                    # Update Qdrant
                    face_qdrant.update_person_ids(
                        point_ids=[matched_face.qdrant_point_id],
                        person_id=person.id,
                    )

                    logger.debug(
                        "Assigned face %s to person %s (%s)",
                        matched_face.id,
                        person.name,
                        person.id,
                    )

                faces_matched += 1
                face_mapping_results.append(
                    FaceMappingResult(
                        image_path=image_path,
                        status="matched",
                        matched_face_id=str(matched_face.id),
                    )
                )

            # Build result for this person
            person_result = PersonImportResult(
                name=person_data.name,
                status=person_status,
                person_id=person_id_str,
                faces_matched=faces_matched,
                faces_not_found=faces_not_found,
                images_missing=images_missing,
                details=face_mapping_results,
            )
            person_results.append(person_result)

            # Update totals
            total_faces_matched += faces_matched
            total_faces_not_found += faces_not_found
            total_images_missing += images_missing

        except Exception as e:
            error_msg = f"Error processing person '{person_data.name}': {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
            # Continue processing other persons
            continue

    # Commit all changes (if not dry_run)
    if not options.dry_run:
        await db.commit()
        logger.info("Import committed to database")
    else:
        logger.info("Dry run completed, no changes committed")

    logger.info(
        "Import completed: %d persons created, %d existing, "
        "%d faces matched, %d not found, %d images missing",
        persons_created,
        persons_existing,
        total_faces_matched,
        total_faces_not_found,
        total_images_missing,
    )

    return ImportResponse(
        success=len(errors) == 0,
        dry_run=options.dry_run,
        persons_created=persons_created,
        persons_existing=persons_existing,
        total_faces_matched=total_faces_matched,
        total_faces_not_found=total_faces_not_found,
        total_images_missing=total_images_missing,
        person_results=person_results,
        errors=errors,
        timestamp=datetime.now(UTC),
    )
