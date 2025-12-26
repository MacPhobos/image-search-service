"""Unit tests for admin export/import functionality."""

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.api.admin_schemas import (
    BoundingBoxExport,
    ExportMetadata,
    FaceMappingExport,
    ImportOptions,
    PersonExport,
    PersonMetadataExport,
)
from image_search_service.db.models import FaceInstance, ImageAsset, Person, PersonStatus
from image_search_service.services.admin_service import (
    _match_face_by_bbox,
    export_person_metadata,
    import_person_metadata,
)


# ============ Export Tests ============


@pytest.mark.asyncio
async def test_export_person_metadata_empty_database(db_session: AsyncSession) -> None:
    """Test export with no persons returns empty export structure."""
    result = await export_person_metadata(db_session, max_faces_per_person=100)

    assert result.version == "1.0"
    assert result.metadata.total_persons == 0
    assert result.metadata.total_face_mappings == 0
    assert result.metadata.export_format == "seed"
    assert result.persons == []
    assert isinstance(result.exported_at, datetime)


@pytest.mark.asyncio
async def test_export_person_metadata_persons_without_faces(
    db_session: AsyncSession,
) -> None:
    """Test export with persons but no faces returns empty export (persons not included)."""
    # Create persons with no face assignments
    person1 = Person(name="Alice", status=PersonStatus.ACTIVE)
    person2 = Person(name="Bob", status=PersonStatus.ACTIVE)
    db_session.add(person1)
    db_session.add(person2)
    await db_session.commit()

    result = await export_person_metadata(db_session, max_faces_per_person=100)

    # Persons without faces should not be included in export
    assert result.metadata.total_persons == 0
    assert result.metadata.total_face_mappings == 0
    assert result.persons == []


@pytest.mark.asyncio
async def test_export_person_metadata_with_faces(db_session: AsyncSession) -> None:
    """Test export with persons and faces includes all data."""
    # Create image assets
    asset1 = ImageAsset(path="/photos/img1.jpg", training_status="trained")
    asset2 = ImageAsset(path="/photos/img2.jpg", training_status="trained")
    db_session.add(asset1)
    db_session.add(asset2)
    await db_session.flush()

    # Create person
    person = Person(name="Alice", status=PersonStatus.ACTIVE)
    db_session.add(person)
    await db_session.flush()

    # Create face instances
    face1 = FaceInstance(
        asset_id=asset1.id,
        person_id=person.id,
        bbox_x=100,
        bbox_y=200,
        bbox_w=50,
        bbox_h=60,
        detection_confidence=0.95,
        quality_score=0.85,
        qdrant_point_id=uuid.uuid4(),
    )
    face2 = FaceInstance(
        asset_id=asset2.id,
        person_id=person.id,
        bbox_x=150,
        bbox_y=250,
        bbox_w=55,
        bbox_h=65,
        detection_confidence=0.92,
        quality_score=0.90,  # Higher quality
        qdrant_point_id=uuid.uuid4(),
    )
    db_session.add(face1)
    db_session.add(face2)
    await db_session.commit()

    result = await export_person_metadata(db_session, max_faces_per_person=100)

    # Verify export structure
    assert result.metadata.total_persons == 1
    assert result.metadata.total_face_mappings == 2
    assert len(result.persons) == 1

    # Verify person data
    exported_person = result.persons[0]
    assert exported_person.name == "Alice"
    assert exported_person.status == "active"
    assert len(exported_person.face_mappings) == 2

    # Verify faces are ordered by quality_score descending (face2 first)
    face_mapping_1 = exported_person.face_mappings[0]
    assert face_mapping_1.image_path == "/photos/img2.jpg"
    assert face_mapping_1.quality_score == 0.90
    assert face_mapping_1.bounding_box.x == 150
    assert face_mapping_1.bounding_box.y == 250
    assert face_mapping_1.bounding_box.width == 55
    assert face_mapping_1.bounding_box.height == 65

    face_mapping_2 = exported_person.face_mappings[1]
    assert face_mapping_2.image_path == "/photos/img1.jpg"
    assert face_mapping_2.quality_score == 0.85


@pytest.mark.asyncio
async def test_export_person_metadata_respects_max_faces_limit(
    db_session: AsyncSession,
) -> None:
    """Test max_faces_per_person limit is enforced."""
    # Create image assets
    assets = []
    for i in range(5):
        asset = ImageAsset(path=f"/photos/img{i}.jpg", training_status="trained")
        db_session.add(asset)
        assets.append(asset)
    await db_session.flush()

    # Create person
    person = Person(name="Alice", status=PersonStatus.ACTIVE)
    db_session.add(person)
    await db_session.flush()

    # Create 5 face instances with different quality scores
    for i, asset in enumerate(assets):
        face = FaceInstance(
            asset_id=asset.id,
            person_id=person.id,
            bbox_x=100 + i * 10,
            bbox_y=200,
            bbox_w=50,
            bbox_h=60,
            detection_confidence=0.9,
            quality_score=0.5 + i * 0.1,  # 0.5, 0.6, 0.7, 0.8, 0.9
            qdrant_point_id=uuid.uuid4(),
        )
        db_session.add(face)
    await db_session.commit()

    # Export with max_faces_per_person=3
    result = await export_person_metadata(db_session, max_faces_per_person=3)

    # Should only include 3 faces (highest quality)
    assert result.metadata.total_persons == 1
    assert result.metadata.total_face_mappings == 3
    assert len(result.persons[0].face_mappings) == 3

    # Verify highest quality faces are included (0.9, 0.8, 0.7)
    quality_scores = [fm.quality_score for fm in result.persons[0].face_mappings]
    assert quality_scores == [0.9, 0.8, 0.7]


@pytest.mark.asyncio
async def test_export_person_metadata_orders_by_quality_then_confidence(
    db_session: AsyncSession,
) -> None:
    """Test faces are ordered by quality_score desc (nulls last), then detection_confidence desc."""
    # Create image assets
    asset1 = ImageAsset(path="/photos/img1.jpg", training_status="trained")
    asset2 = ImageAsset(path="/photos/img2.jpg", training_status="trained")
    asset3 = ImageAsset(path="/photos/img3.jpg", training_status="trained")
    db_session.add_all([asset1, asset2, asset3])
    await db_session.flush()

    # Create person
    person = Person(name="Alice", status=PersonStatus.ACTIVE)
    db_session.add(person)
    await db_session.flush()

    # Create faces with different quality scores (including None)
    face1 = FaceInstance(
        asset_id=asset1.id,
        person_id=person.id,
        bbox_x=100,
        bbox_y=200,
        bbox_w=50,
        bbox_h=60,
        detection_confidence=0.95,
        quality_score=None,  # Null quality
        qdrant_point_id=uuid.uuid4(),
    )
    face2 = FaceInstance(
        asset_id=asset2.id,
        person_id=person.id,
        bbox_x=150,
        bbox_y=250,
        bbox_w=55,
        bbox_h=65,
        detection_confidence=0.80,
        quality_score=0.70,
        qdrant_point_id=uuid.uuid4(),
    )
    face3 = FaceInstance(
        asset_id=asset3.id,
        person_id=person.id,
        bbox_x=200,
        bbox_y=300,
        bbox_w=60,
        bbox_h=70,
        detection_confidence=0.90,
        quality_score=None,  # Null quality but higher confidence than face1
        qdrant_point_id=uuid.uuid4(),
    )
    db_session.add_all([face1, face2, face3])
    await db_session.commit()

    result = await export_person_metadata(db_session, max_faces_per_person=100)

    # Order should be: face2 (quality 0.7), then nulls sorted by confidence (face3 0.9, face1 0.95)
    exported_faces = result.persons[0].face_mappings
    assert len(exported_faces) == 3
    assert exported_faces[0].image_path == "/photos/img2.jpg"  # quality 0.7
    # Next two have None quality, ordered by confidence desc


@pytest.mark.asyncio
async def test_export_person_metadata_excludes_inactive_persons(
    db_session: AsyncSession,
) -> None:
    """Test only active persons are exported."""
    # Create active and hidden persons
    active_person = Person(name="Alice", status=PersonStatus.ACTIVE)
    hidden_person = Person(name="Bob", status=PersonStatus.HIDDEN)
    db_session.add_all([active_person, hidden_person])
    await db_session.flush()

    # Create image assets
    asset1 = ImageAsset(path="/photos/alice.jpg", training_status="trained")
    asset2 = ImageAsset(path="/photos/bob.jpg", training_status="trained")
    db_session.add_all([asset1, asset2])
    await db_session.flush()

    # Create faces for both persons
    face1 = FaceInstance(
        asset_id=asset1.id,
        person_id=active_person.id,
        bbox_x=100,
        bbox_y=200,
        bbox_w=50,
        bbox_h=60,
        detection_confidence=0.95,
        quality_score=0.85,
        qdrant_point_id=uuid.uuid4(),
    )
    face2 = FaceInstance(
        asset_id=asset2.id,
        person_id=hidden_person.id,
        bbox_x=150,
        bbox_y=250,
        bbox_w=55,
        bbox_h=65,
        detection_confidence=0.92,
        quality_score=0.90,
        qdrant_point_id=uuid.uuid4(),
    )
    db_session.add_all([face1, face2])
    await db_session.commit()

    result = await export_person_metadata(db_session, max_faces_per_person=100)

    # Only active person should be exported
    assert result.metadata.total_persons == 1
    assert result.persons[0].name == "Alice"


@pytest.mark.asyncio
async def test_export_person_metadata_multiple_persons_ordered_by_name(
    db_session: AsyncSession,
) -> None:
    """Test multiple persons are ordered alphabetically by name."""
    # Create persons in random order
    person_charlie = Person(name="Charlie", status=PersonStatus.ACTIVE)
    person_alice = Person(name="Alice", status=PersonStatus.ACTIVE)
    person_bob = Person(name="Bob", status=PersonStatus.ACTIVE)
    db_session.add_all([person_charlie, person_alice, person_bob])
    await db_session.flush()

    # Create image assets and faces for each person
    for idx, person in enumerate([person_alice, person_bob, person_charlie]):
        asset = ImageAsset(path=f"/photos/{person.name}.jpg", training_status="trained")
        db_session.add(asset)
        await db_session.flush()

        face = FaceInstance(
            asset_id=asset.id,
            person_id=person.id,
            bbox_x=100,
            bbox_y=200,
            bbox_w=50,
            bbox_h=60,
            detection_confidence=0.95,
            quality_score=0.85,
            qdrant_point_id=uuid.uuid4(),
        )
        db_session.add(face)
    await db_session.commit()

    result = await export_person_metadata(db_session, max_faces_per_person=100)

    # Persons should be alphabetically ordered
    assert result.metadata.total_persons == 3
    assert [p.name for p in result.persons] == ["Alice", "Bob", "Charlie"]


# ============ Bounding Box Matching Tests ============


def test_match_face_by_bbox_exact_match() -> None:
    """Test bounding box matching with exact coordinates."""
    face = MagicMock()
    face.bbox_x, face.bbox_y, face.bbox_w, face.bbox_h = 100, 200, 50, 60

    result = _match_face_by_bbox(
        detected_faces=[face],
        target_x=100,
        target_y=200,
        target_w=50,
        target_h=60,
        tolerance=10,
    )

    assert result is face


def test_match_face_by_bbox_within_tolerance() -> None:
    """Test bounding box matching within tolerance."""
    face = MagicMock()
    face.bbox_x, face.bbox_y, face.bbox_w, face.bbox_h = 105, 195, 52, 58

    result = _match_face_by_bbox(
        detected_faces=[face],
        target_x=100,
        target_y=200,
        target_w=50,
        target_h=60,
        tolerance=10,
    )

    assert result is face


def test_match_face_by_bbox_outside_tolerance() -> None:
    """Test no match when outside tolerance."""
    face = MagicMock()
    face.bbox_x, face.bbox_y, face.bbox_w, face.bbox_h = 120, 200, 50, 60  # x is 20 off

    result = _match_face_by_bbox(
        detected_faces=[face],
        target_x=100,
        target_y=200,
        target_w=50,
        target_h=60,
        tolerance=10,
    )

    assert result is None


def test_match_face_by_bbox_closest_match_selected() -> None:
    """Test that closest matching face is selected when multiple candidates exist."""
    # Create three faces with different distances from target
    face1 = MagicMock()
    face1.bbox_x, face1.bbox_y, face1.bbox_w, face1.bbox_h = 110, 200, 50, 60  # Distance ~10

    face2 = MagicMock()
    face2.bbox_x, face2.bbox_y, face2.bbox_w, face2.bbox_h = 105, 202, 51, 59  # Distance ~5

    face3 = MagicMock()
    face3.bbox_x, face3.bbox_y, face3.bbox_w, face3.bbox_h = 108, 205, 50, 60  # Distance ~9

    result = _match_face_by_bbox(
        detected_faces=[face1, face2, face3],
        target_x=100,
        target_y=200,
        target_w=50,
        target_h=60,
        tolerance=15,
    )

    # face2 should be selected (closest match)
    assert result is face2


def test_match_face_by_bbox_empty_list() -> None:
    """Test no match when no faces provided."""
    result = _match_face_by_bbox(
        detected_faces=[],
        target_x=100,
        target_y=200,
        target_w=50,
        target_h=60,
        tolerance=10,
    )

    assert result is None


def test_match_face_by_bbox_all_dimensions_must_match() -> None:
    """Test that all four dimensions must be within tolerance."""
    # Face matches in x, y, w but not h
    face = MagicMock()
    face.bbox_x, face.bbox_y, face.bbox_w, face.bbox_h = 105, 195, 52, 100  # h is way off

    result = _match_face_by_bbox(
        detected_faces=[face],
        target_x=100,
        target_y=200,
        target_w=50,
        target_h=60,
        tolerance=10,
    )

    assert result is None


# ============ Import Tests ============


@pytest.mark.asyncio
async def test_import_person_metadata_dry_run_no_changes(
    db_session: AsyncSession,
) -> None:
    """Test dry_run mode makes no database changes."""
    # Create import data
    import_data = PersonMetadataExport(
        version="1.0",
        exported_at=datetime.now(UTC),
        metadata=ExportMetadata(total_persons=1, total_face_mappings=1),
        persons=[
            PersonExport(
                name="Alice",
                status="active",
                face_mappings=[
                    FaceMappingExport(
                        image_path="/photos/alice.jpg",
                        bounding_box=BoundingBoxExport(x=100, y=200, width=50, height=60),
                        detection_confidence=0.95,
                        quality_score=0.85,
                    )
                ],
            )
        ],
    )

    # Create mock face qdrant client
    mock_face_qdrant = MagicMock()
    mock_face_qdrant.update_person_ids = MagicMock()

    # Import with dry_run=True
    options = ImportOptions(dry_run=True, tolerance_pixels=10, skip_missing_images=True)
    result = await import_person_metadata(db_session, import_data, options, mock_face_qdrant)

    # Verify response
    assert result.dry_run is True
    assert result.persons_created == 1  # Would create
    assert result.persons_existing == 0

    # Verify no actual database changes
    stmt = select(Person)
    db_result = await db_session.execute(stmt)
    persons = db_result.scalars().all()
    assert len(persons) == 0  # No person created in dry run


@pytest.mark.asyncio
async def test_import_person_metadata_creates_new_persons(
    db_session: AsyncSession,
) -> None:
    """Test creating new persons during import."""
    # Create image asset and face instance
    asset = ImageAsset(path="/photos/alice.jpg", training_status="trained")
    db_session.add(asset)
    await db_session.flush()

    face = FaceInstance(
        asset_id=asset.id,
        person_id=None,  # Not assigned yet
        bbox_x=100,
        bbox_y=200,
        bbox_w=50,
        bbox_h=60,
        detection_confidence=0.95,
        quality_score=0.85,
        qdrant_point_id=uuid.uuid4(),
    )
    db_session.add(face)
    await db_session.commit()

    # Create import data
    import_data = PersonMetadataExport(
        version="1.0",
        exported_at=datetime.now(UTC),
        metadata=ExportMetadata(total_persons=1, total_face_mappings=1),
        persons=[
            PersonExport(
                name="Alice",
                status="active",
                face_mappings=[
                    FaceMappingExport(
                        image_path="/photos/alice.jpg",
                        bounding_box=BoundingBoxExport(x=100, y=200, width=50, height=60),
                        detection_confidence=0.95,
                        quality_score=0.85,
                    )
                ],
            )
        ],
    )

    # Create mock face qdrant client
    mock_face_qdrant = MagicMock()
    mock_face_qdrant.update_person_ids = MagicMock()

    # Import (skip_missing_images=False since files don't exist in test)
    options = ImportOptions(dry_run=False, tolerance_pixels=10, skip_missing_images=False)
    result = await import_person_metadata(db_session, import_data, options, mock_face_qdrant)

    # Verify response
    assert result.success is True
    assert result.dry_run is False
    assert result.persons_created == 1
    assert result.persons_existing == 0
    assert result.total_faces_matched == 1
    assert result.total_faces_not_found == 0

    # Verify person was created
    stmt = select(Person).where(Person.name == "Alice")
    db_result = await db_session.execute(stmt)
    person = db_result.scalar_one()
    assert person.name == "Alice"
    assert person.status == PersonStatus.ACTIVE

    # Verify face was assigned
    await db_session.refresh(face)
    assert face.person_id == person.id


@pytest.mark.asyncio
async def test_import_person_metadata_finds_existing_persons(
    db_session: AsyncSession,
) -> None:
    """Test finding existing persons by name (case-insensitive)."""
    # Create existing person
    existing_person = Person(name="Alice", status=PersonStatus.ACTIVE)
    db_session.add(existing_person)
    await db_session.flush()

    # Create image asset and face instance
    asset = ImageAsset(path="/photos/alice.jpg", training_status="trained")
    db_session.add(asset)
    await db_session.flush()

    face = FaceInstance(
        asset_id=asset.id,
        person_id=None,
        bbox_x=100,
        bbox_y=200,
        bbox_w=50,
        bbox_h=60,
        detection_confidence=0.95,
        quality_score=0.85,
        qdrant_point_id=uuid.uuid4(),
    )
    db_session.add(face)
    await db_session.commit()

    # Create import data with different case
    import_data = PersonMetadataExport(
        version="1.0",
        exported_at=datetime.now(UTC),
        metadata=ExportMetadata(total_persons=1, total_face_mappings=1),
        persons=[
            PersonExport(
                name="ALICE",  # Different case
                status="active",
                face_mappings=[
                    FaceMappingExport(
                        image_path="/photos/alice.jpg",
                        bounding_box=BoundingBoxExport(x=100, y=200, width=50, height=60),
                        detection_confidence=0.95,
                        quality_score=0.85,
                    )
                ],
            )
        ],
    )

    # Create mock face qdrant client
    mock_face_qdrant = MagicMock()
    mock_face_qdrant.update_person_ids = MagicMock()

    # Import (skip_missing_images=False since files don't exist in test)
    options = ImportOptions(dry_run=False, tolerance_pixels=10, skip_missing_images=False)
    result = await import_person_metadata(db_session, import_data, options, mock_face_qdrant)

    # Verify response
    assert result.success is True
    assert result.persons_created == 0
    assert result.persons_existing == 1  # Found existing
    assert result.total_faces_matched == 1

    # Verify no duplicate person created
    stmt = select(Person)
    db_result = await db_session.execute(stmt)
    persons = db_result.scalars().all()
    assert len(persons) == 1  # Still only one person


@pytest.mark.asyncio
async def test_import_person_metadata_face_matching_success(
    db_session: AsyncSession,
) -> None:
    """Test successful face matching by bounding box."""
    # Create image asset with multiple faces
    asset = ImageAsset(path="/photos/group.jpg", training_status="trained")
    db_session.add(asset)
    await db_session.flush()

    # Create two faces in the image
    face1 = FaceInstance(
        asset_id=asset.id,
        person_id=None,
        bbox_x=100,
        bbox_y=200,
        bbox_w=50,
        bbox_h=60,
        detection_confidence=0.95,
        quality_score=0.85,
        qdrant_point_id=uuid.uuid4(),
    )
    face2 = FaceInstance(
        asset_id=asset.id,
        person_id=None,
        bbox_x=300,
        bbox_y=200,
        bbox_w=50,
        bbox_h=60,
        detection_confidence=0.92,
        quality_score=0.80,
        qdrant_point_id=uuid.uuid4(),
    )
    db_session.add_all([face1, face2])
    await db_session.commit()

    # Create import data targeting face1
    import_data = PersonMetadataExport(
        version="1.0",
        exported_at=datetime.now(UTC),
        metadata=ExportMetadata(total_persons=1, total_face_mappings=1),
        persons=[
            PersonExport(
                name="Alice",
                status="active",
                face_mappings=[
                    FaceMappingExport(
                        image_path="/photos/group.jpg",
                        bounding_box=BoundingBoxExport(x=105, y=205, width=48, height=62),  # Close to face1
                        detection_confidence=0.95,
                        quality_score=0.85,
                    )
                ],
            )
        ],
    )

    # Create mock face qdrant client
    mock_face_qdrant = MagicMock()
    mock_face_qdrant.update_person_ids = MagicMock()

    # Import (skip_missing_images=False since files don't exist in test)
    options = ImportOptions(dry_run=False, tolerance_pixels=10, skip_missing_images=False)
    result = await import_person_metadata(db_session, import_data, options, mock_face_qdrant)

    # Verify face1 was matched
    await db_session.refresh(face1)
    await db_session.refresh(face2)

    stmt = select(Person).where(Person.name == "Alice")
    db_result = await db_session.execute(stmt)
    person = db_result.scalar_one()

    assert face1.person_id == person.id  # face1 matched
    assert face2.person_id is None  # face2 not matched


@pytest.mark.asyncio
async def test_import_person_metadata_missing_image_handling(
    db_session: AsyncSession,
) -> None:
    """Test handling of missing image files (when skip_missing_images=False)."""
    # Create import data with non-existent image
    import_data = PersonMetadataExport(
        version="1.0",
        exported_at=datetime.now(UTC),
        metadata=ExportMetadata(total_persons=1, total_face_mappings=1),
        persons=[
            PersonExport(
                name="Alice",
                status="active",
                face_mappings=[
                    FaceMappingExport(
                        image_path="/nonexistent/path.jpg",
                        bounding_box=BoundingBoxExport(x=100, y=200, width=50, height=60),
                        detection_confidence=0.95,
                        quality_score=0.85,
                    )
                ],
            )
        ],
    )

    # Create mock face qdrant client
    mock_face_qdrant = MagicMock()
    mock_face_qdrant.update_person_ids = MagicMock()

    # Import with skip_missing_images=True
    options = ImportOptions(dry_run=False, tolerance_pixels=10, skip_missing_images=True)
    result = await import_person_metadata(db_session, import_data, options, mock_face_qdrant)

    # Should skip missing image gracefully
    assert result.success is True
    assert result.persons_created == 1  # Person still created
    assert result.total_faces_matched == 0
    assert result.total_images_missing == 1

    # Verify person result details
    person_result = result.person_results[0]
    assert person_result.images_missing == 1
    assert person_result.details[0].status == "image_missing"
    assert "not found" in person_result.details[0].error.lower()


@pytest.mark.asyncio
async def test_import_person_metadata_tolerance_parameter(
    db_session: AsyncSession,
) -> None:
    """Test tolerance parameter affects face matching."""
    # Create image asset and face
    asset = ImageAsset(path="/photos/alice.jpg", training_status="trained")
    db_session.add(asset)
    await db_session.flush()

    face = FaceInstance(
        asset_id=asset.id,
        person_id=None,
        bbox_x=100,
        bbox_y=200,
        bbox_w=50,
        bbox_h=60,
        detection_confidence=0.95,
        quality_score=0.85,
        qdrant_point_id=uuid.uuid4(),
    )
    db_session.add(face)
    await db_session.commit()

    # Create import data with bbox slightly different (8px off)
    import_data = PersonMetadataExport(
        version="1.0",
        exported_at=datetime.now(UTC),
        metadata=ExportMetadata(total_persons=1, total_face_mappings=1),
        persons=[
            PersonExport(
                name="Alice",
                status="active",
                face_mappings=[
                    FaceMappingExport(
                        image_path="/photos/alice.jpg",
                        bounding_box=BoundingBoxExport(x=108, y=200, width=50, height=60),  # 8px off in x
                        detection_confidence=0.95,
                        quality_score=0.85,
                    )
                ],
            )
        ],
    )

    # Create mock face qdrant client
    mock_face_qdrant = MagicMock()
    mock_face_qdrant.update_person_ids = MagicMock()

    # Test with tolerance=5 (should NOT match)
    options_strict = ImportOptions(dry_run=False, tolerance_pixels=5, skip_missing_images=False)
    result_strict = await import_person_metadata(db_session, import_data, options_strict, mock_face_qdrant)

    assert result_strict.total_faces_matched == 0
    assert result_strict.total_faces_not_found == 1

    # Rollback and retry with tolerance=10 (should match)
    await db_session.rollback()

    # Recreate data (rolled back)
    db_session.add(asset)
    await db_session.flush()
    db_session.add(face)
    await db_session.commit()

    options_loose = ImportOptions(dry_run=False, tolerance_pixels=10, skip_missing_images=False)
    result_loose = await import_person_metadata(db_session, import_data, options_loose, mock_face_qdrant)

    assert result_loose.total_faces_matched == 1
    assert result_loose.total_faces_not_found == 0


@pytest.mark.asyncio
async def test_import_person_metadata_image_not_in_database(
    db_session: AsyncSession,
) -> None:
    """Test handling when image exists on filesystem but not in database."""
    # Create import data (no corresponding ImageAsset in DB)
    import_data = PersonMetadataExport(
        version="1.0",
        exported_at=datetime.now(UTC),
        metadata=ExportMetadata(total_persons=1, total_face_mappings=1),
        persons=[
            PersonExport(
                name="Alice",
                status="active",
                face_mappings=[
                    FaceMappingExport(
                        image_path="/photos/not_in_db.jpg",
                        bounding_box=BoundingBoxExport(x=100, y=200, width=50, height=60),
                        detection_confidence=0.95,
                        quality_score=0.85,
                    )
                ],
            )
        ],
    )

    # Create mock face qdrant client
    mock_face_qdrant = MagicMock()
    mock_face_qdrant.update_person_ids = MagicMock()

    # Import (skip_missing_images=False to check database)
    options = ImportOptions(dry_run=False, tolerance_pixels=10, skip_missing_images=False)
    result = await import_person_metadata(db_session, import_data, options, mock_face_qdrant)

    # Should report not_found
    assert result.total_faces_not_found == 1
    assert result.person_results[0].details[0].status == "not_found"
    assert "not found in database" in result.person_results[0].details[0].error.lower()


@pytest.mark.asyncio
async def test_import_person_metadata_no_faces_detected_in_image(
    db_session: AsyncSession,
) -> None:
    """Test handling when image has no detected faces."""
    # Create image asset without faces
    asset = ImageAsset(path="/photos/no_faces.jpg", training_status="trained")
    db_session.add(asset)
    await db_session.commit()

    # Create import data
    import_data = PersonMetadataExport(
        version="1.0",
        exported_at=datetime.now(UTC),
        metadata=ExportMetadata(total_persons=1, total_face_mappings=1),
        persons=[
            PersonExport(
                name="Alice",
                status="active",
                face_mappings=[
                    FaceMappingExport(
                        image_path="/photos/no_faces.jpg",
                        bounding_box=BoundingBoxExport(x=100, y=200, width=50, height=60),
                        detection_confidence=0.95,
                        quality_score=0.85,
                    )
                ],
            )
        ],
    )

    # Create mock face qdrant client
    mock_face_qdrant = MagicMock()
    mock_face_qdrant.update_person_ids = MagicMock()

    # Import
    options = ImportOptions(dry_run=False, tolerance_pixels=10, skip_missing_images=False)
    result = await import_person_metadata(db_session, import_data, options, mock_face_qdrant)

    # Should report not_found
    assert result.total_faces_not_found == 1
    assert result.person_results[0].details[0].status == "not_found"
    assert "no faces detected" in result.person_results[0].details[0].error.lower()


@pytest.mark.asyncio
async def test_import_person_metadata_updates_qdrant(
    db_session: AsyncSession,
) -> None:
    """Test that Qdrant is updated when faces are assigned."""
    # Create image asset and face
    asset = ImageAsset(path="/photos/alice.jpg", training_status="trained")
    db_session.add(asset)
    await db_session.flush()

    qdrant_point_id = uuid.uuid4()
    face = FaceInstance(
        asset_id=asset.id,
        person_id=None,
        bbox_x=100,
        bbox_y=200,
        bbox_w=50,
        bbox_h=60,
        detection_confidence=0.95,
        quality_score=0.85,
        qdrant_point_id=qdrant_point_id,
    )
    db_session.add(face)
    await db_session.commit()

    # Create import data
    import_data = PersonMetadataExport(
        version="1.0",
        exported_at=datetime.now(UTC),
        metadata=ExportMetadata(total_persons=1, total_face_mappings=1),
        persons=[
            PersonExport(
                name="Alice",
                status="active",
                face_mappings=[
                    FaceMappingExport(
                        image_path="/photos/alice.jpg",
                        bounding_box=BoundingBoxExport(x=100, y=200, width=50, height=60),
                        detection_confidence=0.95,
                        quality_score=0.85,
                    )
                ],
            )
        ],
    )

    # Create mock face qdrant client
    mock_face_qdrant = MagicMock()
    mock_face_qdrant.update_person_ids = MagicMock()

    # Import (skip_missing_images=False since files don't exist in test)
    options = ImportOptions(dry_run=False, tolerance_pixels=10, skip_missing_images=False)
    result = await import_person_metadata(db_session, import_data, options, mock_face_qdrant)

    # Verify Qdrant was updated
    assert mock_face_qdrant.update_person_ids.called
    call_args = mock_face_qdrant.update_person_ids.call_args
    assert call_args[1]["point_ids"] == [qdrant_point_id]

    # Verify person_id was passed
    stmt = select(Person).where(Person.name == "Alice")
    db_result = await db_session.execute(stmt)
    person = db_result.scalar_one()
    assert call_args[1]["person_id"] == person.id


@pytest.mark.asyncio
async def test_import_person_metadata_error_handling_continues_processing(
    db_session: AsyncSession,
) -> None:
    """Test that errors on one person don't stop processing others."""
    # Create valid setup for second person
    asset2 = ImageAsset(path="/photos/bob.jpg", training_status="trained")
    db_session.add(asset2)
    await db_session.flush()

    face2 = FaceInstance(
        asset_id=asset2.id,
        person_id=None,
        bbox_x=100,
        bbox_y=200,
        bbox_w=50,
        bbox_h=60,
        detection_confidence=0.95,
        quality_score=0.85,
        qdrant_point_id=uuid.uuid4(),
    )
    db_session.add(face2)
    await db_session.commit()

    # Create import data with two persons (first will fail, second should succeed)
    import_data = PersonMetadataExport(
        version="1.0",
        exported_at=datetime.now(UTC),
        metadata=ExportMetadata(total_persons=2, total_face_mappings=2),
        persons=[
            PersonExport(
                name="Alice",
                status="active",
                face_mappings=[
                    FaceMappingExport(
                        image_path="/nonexistent/alice.jpg",  # Will fail
                        bounding_box=BoundingBoxExport(x=100, y=200, width=50, height=60),
                        detection_confidence=0.95,
                        quality_score=0.85,
                    )
                ],
            ),
            PersonExport(
                name="Bob",
                status="active",
                face_mappings=[
                    FaceMappingExport(
                        image_path="/photos/bob.jpg",  # Will succeed
                        bounding_box=BoundingBoxExport(x=100, y=200, width=50, height=60),
                        detection_confidence=0.95,
                        quality_score=0.85,
                    )
                ],
            ),
        ],
    )

    # Create mock face qdrant client
    mock_face_qdrant = MagicMock()
    mock_face_qdrant.update_person_ids = MagicMock()

    # Import with skip_missing_images=False to trigger database lookup
    options = ImportOptions(dry_run=False, tolerance_pixels=10, skip_missing_images=False)
    result = await import_person_metadata(db_session, import_data, options, mock_face_qdrant)

    # Verify both persons were processed
    assert len(result.person_results) == 2

    # Bob should have succeeded
    bob_result = next(r for r in result.person_results if r.name == "Bob")
    assert bob_result.faces_matched == 1
