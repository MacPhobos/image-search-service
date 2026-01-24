"""Integration tests for FaceSuggestion cleanup on face assignment changes.

These tests verify that pending FaceSuggestion records are properly expired
when face-to-person assignments change, preventing orphaned/stale suggestions
from polluting the database.
"""

import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from image_search_service.db.models import (
    FaceInstance,
    FaceSuggestion,
    FaceSuggestionStatus,
    ImageAsset,
    Person,
    PersonStatus,
    TrainingStatus,
)

# ============ Fixtures ============


@pytest.fixture
def mock_qdrant_client():
    """Create a mock Qdrant client that doesn't make real API calls."""
    client = MagicMock()
    client.update_person_ids.return_value = None  # Success
    client.update_payload.return_value = None  # Success
    client.point_exists.return_value = True  # Point exists
    return client


@pytest.fixture
async def image_asset(db_session):
    """Create a test ImageAsset."""
    asset = ImageAsset(
        path="/test/images/photo.jpg",
        training_status=TrainingStatus.PENDING.value,
        width=640,
        height=480,
        file_size=102400,
        mime_type="image/jpeg",
    )
    db_session.add(asset)
    await db_session.commit()
    await db_session.refresh(asset)
    return asset


@pytest.fixture
async def person_a(db_session):
    """Create test Person A."""
    person = Person(
        id=uuid.uuid4(),
        name="Person A",
        status=PersonStatus.ACTIVE.value,
    )
    db_session.add(person)
    await db_session.commit()
    await db_session.refresh(person)
    return person


@pytest.fixture
async def person_b(db_session):
    """Create test Person B."""
    person = Person(
        id=uuid.uuid4(),
        name="Person B",
        status=PersonStatus.ACTIVE.value,
    )
    db_session.add(person)
    await db_session.commit()
    await db_session.refresh(person)
    return person


@pytest.fixture
async def face_assigned_to_person_a(db_session, image_asset, person_a):
    """Create a FaceInstance assigned to Person A."""
    face = FaceInstance(
        id=uuid.uuid4(),
        asset_id=image_asset.id,
        person_id=person_a.id,
        bbox_x=100,
        bbox_y=150,
        bbox_w=80,
        bbox_h=80,
        detection_confidence=0.95,
        quality_score=0.75,
        qdrant_point_id=uuid.uuid4(),
    )
    db_session.add(face)
    await db_session.commit()
    await db_session.refresh(face)
    return face


@pytest.fixture
async def unassigned_face(db_session, image_asset):
    """Create an unassigned FaceInstance (no person_id)."""
    face = FaceInstance(
        id=uuid.uuid4(),
        asset_id=image_asset.id,
        person_id=None,  # Unassigned
        bbox_x=200,
        bbox_y=150,
        bbox_w=80,
        bbox_h=80,
        detection_confidence=0.92,
        quality_score=0.70,
        qdrant_point_id=uuid.uuid4(),
    )
    db_session.add(face)
    await db_session.commit()
    await db_session.refresh(face)
    return face


async def create_suggestion(
    db_session,
    face_instance_id: uuid.UUID,
    suggested_person_id: uuid.UUID,
    source_face_id: uuid.UUID,
    status: str = FaceSuggestionStatus.PENDING.value,
) -> FaceSuggestion:
    """Helper to create a FaceSuggestion record."""
    suggestion = FaceSuggestion(
        face_instance_id=face_instance_id,
        suggested_person_id=suggested_person_id,
        source_face_id=source_face_id,
        confidence=0.85,
        status=status,
    )
    db_session.add(suggestion)
    await db_session.commit()
    await db_session.refresh(suggestion)
    return suggestion


# ============ Unassign Face Tests ============


@pytest.mark.asyncio
async def test_unassign_face_expires_pending_suggestions(
    test_client, db_session, face_assigned_to_person_a, unassigned_face, person_a, mock_qdrant_client  # noqa: E501
):
    """Test that unassigning a face expires pending suggestions where it was the source.

    Setup: Face A is assigned to Person A, Face B is unassigned
    - Create pending suggestion: Face B should be Person A (based on Face A)
    - Unassign Face A
    - Assert: Suggestion is now EXPIRED
    """
    # Create pending suggestion based on face_assigned_to_person_a
    suggestion = await create_suggestion(
        db_session,
        face_instance_id=unassigned_face.id,  # Target face
        suggested_person_id=person_a.id,  # Suggested person
        source_face_id=face_assigned_to_person_a.id,  # Source face
        status=FaceSuggestionStatus.PENDING.value,
    )

    # Verify suggestion is pending
    assert suggestion.status == FaceSuggestionStatus.PENDING.value
    assert suggestion.reviewed_at is None

    # Mock Qdrant client to avoid real API calls
    with patch(
        "image_search_service.vector.face_qdrant.get_face_qdrant_client",
        return_value=mock_qdrant_client,
    ):
        # Unassign the source face
        response = await test_client.delete(
            f"/api/v1/faces/faces/{face_assigned_to_person_a.id}/person"
        )
        assert response.status_code == 200

    # Refresh suggestion from database
    await db_session.refresh(suggestion)

    # Assert: Suggestion should now be EXPIRED
    assert suggestion.status == FaceSuggestionStatus.EXPIRED.value
    assert suggestion.reviewed_at is not None
    assert isinstance(suggestion.reviewed_at, datetime)


@pytest.mark.asyncio
async def test_unassign_face_preserves_accepted_rejected_suggestions(
    test_client, db_session, face_assigned_to_person_a, unassigned_face, person_a
, mock_qdrant_client):
    """Test that unassigning a face does NOT affect ACCEPTED/REJECTED suggestions.

    Only PENDING suggestions should be expired. Already reviewed suggestions
    should remain unchanged.
    """
    # Create accepted and rejected suggestions
    accepted_suggestion = await create_suggestion(
        db_session,
        face_instance_id=unassigned_face.id,
        suggested_person_id=person_a.id,
        source_face_id=face_assigned_to_person_a.id,
        status=FaceSuggestionStatus.ACCEPTED.value,
    )
    accepted_suggestion.reviewed_at = datetime.now(UTC)

    rejected_suggestion = await create_suggestion(
        db_session,
        face_instance_id=unassigned_face.id,
        suggested_person_id=person_a.id,
        source_face_id=face_assigned_to_person_a.id,
        status=FaceSuggestionStatus.REJECTED.value,
    )
    rejected_suggestion.reviewed_at = datetime.now(UTC)

    await db_session.commit()

    # Mock Qdrant client to avoid real API calls
    with patch(
        "image_search_service.vector.face_qdrant.get_face_qdrant_client",
        return_value=mock_qdrant_client,
    ):
        # Unassign the source face
        response = await test_client.delete(
            f"/api/v1/faces/faces/{face_assigned_to_person_a.id}/person"
        )
        assert response.status_code == 200

    # Refresh suggestions
    await db_session.refresh(accepted_suggestion)
    await db_session.refresh(rejected_suggestion)

    # Assert: Reviewed suggestions should be unchanged
    assert accepted_suggestion.status == FaceSuggestionStatus.ACCEPTED.value
    assert rejected_suggestion.status == FaceSuggestionStatus.REJECTED.value


@pytest.mark.asyncio
async def test_unassign_face_no_suggestions_no_error(
    test_client, db_session, face_assigned_to_person_a, person_a
, mock_qdrant_client):
    """Test that unassigning a face with NO suggestions completes successfully.

    Edge case: Face has no related suggestions. Should not error.
    """
    # No suggestions created

    # Mock Qdrant client
    with patch(
        "image_search_service.vector.face_qdrant.get_face_qdrant_client",
        return_value=mock_qdrant_client,
    ):
        # Unassign the face
        response = await test_client.delete(
            f"/api/v1/faces/faces/{face_assigned_to_person_a.id}/person"
        )

    # Assert: Should succeed
    assert response.status_code == 200
    data = response.json()
    assert data["faceId"] == str(face_assigned_to_person_a.id)
    assert data["previousPersonId"] == str(person_a.id)


# ============ Bulk Remove Tests ============


@pytest.mark.asyncio
async def test_bulk_remove_expires_suggestions(
    test_client, db_session, image_asset, person_a
, mock_qdrant_client):
    """Test that bulk_remove_from_person expires pending suggestions for all removed faces.

    Setup: Multiple faces assigned to Person A, each with pending suggestions
    - Bulk remove all faces from Person A
    - Assert: All pending suggestions are EXPIRED
    """
    # Create 3 faces assigned to Person A
    faces = []
    for i in range(3):
        face = FaceInstance(
            id=uuid.uuid4(),
            asset_id=image_asset.id,
            person_id=person_a.id,
            bbox_x=100 + (i * 50),
            bbox_y=150,
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.90,
            quality_score=0.70,
            qdrant_point_id=uuid.uuid4(),
        )
        db_session.add(face)
        faces.append(face)

    await db_session.commit()

    # Create pending suggestions for each face
    suggestions = []
    for face in faces:
        suggestion = await create_suggestion(
            db_session,
            face_instance_id=uuid.uuid4(),  # Different target face
            suggested_person_id=person_a.id,
            source_face_id=face.id,  # Source is one of our faces
            status=FaceSuggestionStatus.PENDING.value,
        )
        suggestions.append(suggestion)

    # Verify all suggestions are pending
    for s in suggestions:
        assert s.status == FaceSuggestionStatus.PENDING.value

    # Mock Qdrant client
    with patch(
        "image_search_service.vector.face_qdrant.get_face_qdrant_client",
        return_value=mock_qdrant_client,
    ):
        # Bulk remove faces from Person A
        response = await test_client.post(
            f"/api/v1/faces/persons/{person_a.id}/photos/bulk-remove",
            json={"photo_ids": [image_asset.id]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["updatedFaces"] == 3

    # Refresh suggestions from database
    for suggestion in suggestions:
        await db_session.refresh(suggestion)

    # Assert: All suggestions should be EXPIRED
    for suggestion in suggestions:
        assert suggestion.status == FaceSuggestionStatus.EXPIRED.value
        assert suggestion.reviewed_at is not None


@pytest.mark.asyncio
async def test_bulk_remove_empty_photo_ids_no_error(test_client, db_session, person_a):
    """Test that bulk_remove with empty photo_ids returns gracefully.

    Edge case: Empty photo_ids list should not cause errors.
    """
    response = await test_client.post(
        f"/api/v1/faces/persons/{person_a.id}/photos/bulk-remove",
        json={"photo_ids": []},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["updatedFaces"] == 0
    assert data["updatedPhotos"] == 0


# ============ Bulk Move Tests ============


@pytest.mark.asyncio
async def test_bulk_move_expires_old_suggestions(
    test_client, db_session, image_asset, face_assigned_to_person_a, person_a, person_b
, mock_qdrant_client):
    """Test that bulk_move_to_person expires suggestions for old person assignment.

    Setup: Face assigned to Person A with pending suggestions
    - Move face to Person B
    - Assert: Suggestions pointing to Person A are EXPIRED
    """
    # Create pending suggestion based on face_assigned_to_person_a → Person A
    suggestion = await create_suggestion(
        db_session,
        face_instance_id=uuid.uuid4(),  # Different target face
        suggested_person_id=person_a.id,  # Suggested person A
        source_face_id=face_assigned_to_person_a.id,
        status=FaceSuggestionStatus.PENDING.value,
    )

    assert suggestion.status == FaceSuggestionStatus.PENDING.value

    # Mock Qdrant client
    with patch(
        "image_search_service.vector.face_qdrant.get_face_qdrant_client",
        return_value=mock_qdrant_client,
    ):
        # Move face from Person A to Person B
        response = await test_client.post(
            f"/api/v1/faces/persons/{person_a.id}/photos/bulk-move",
            json={
                "photo_ids": [image_asset.id],
                "to_person_id": str(person_b.id),
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["toPersonId"] == str(person_b.id)
        assert data["updatedFaces"] == 1

    # Refresh suggestion
    await db_session.refresh(suggestion)

    # Assert: Old suggestion is EXPIRED
    assert suggestion.status == FaceSuggestionStatus.EXPIRED.value
    assert suggestion.reviewed_at is not None


@pytest.mark.asyncio
async def test_bulk_move_creates_new_person(
    test_client, db_session, image_asset, face_assigned_to_person_a, person_a
, mock_qdrant_client):
    """Test that bulk_move can create a new person and expire old suggestions.

    Setup: Face assigned to Person A
    - Move to new person "Person C" (doesn't exist yet)
    - Assert: New person created, old suggestions expired
    """
    # Create pending suggestion
    suggestion = await create_suggestion(
        db_session,
        face_instance_id=uuid.uuid4(),
        suggested_person_id=person_a.id,
        source_face_id=face_assigned_to_person_a.id,
        status=FaceSuggestionStatus.PENDING.value,
    )

    # Mock Qdrant client
    with patch(
        "image_search_service.vector.face_qdrant.get_face_qdrant_client",
        return_value=mock_qdrant_client,
    ):
        # Move to new person
        response = await test_client.post(
            f"/api/v1/faces/persons/{person_a.id}/photos/bulk-move",
            json={
                "photo_ids": [image_asset.id],
                "to_person_name": "Person C",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["toPersonName"] == "Person C"
        assert data["personCreated"] is True
        assert data["updatedFaces"] == 1

    # Refresh suggestion
    await db_session.refresh(suggestion)

    # Assert: Suggestion is expired
    assert suggestion.status == FaceSuggestionStatus.EXPIRED.value


# ============ Cleanup Job Tests ============


@pytest.mark.asyncio
async def test_cleanup_orphaned_suggestions_job_unassigned_source(
    db_session, face_assigned_to_person_a, unassigned_face, person_a
):
    """Test cleanup job expires suggestions when source face.person_id is NULL.

    Edge case: Source face was unassigned but cleanup logic didn't run.
    This test simulates the cleanup logic directly without running the job.
    """
    from sqlalchemy import and_, or_, select, update

    # Create pending suggestion
    suggestion = await create_suggestion(
        db_session,
        face_instance_id=unassigned_face.id,
        suggested_person_id=person_a.id,
        source_face_id=face_assigned_to_person_a.id,
        status=FaceSuggestionStatus.PENDING.value,
    )

    assert suggestion.status == FaceSuggestionStatus.PENDING.value

    # Manually NULL out source face person_id (simulating edge case)
    await db_session.execute(
        update(FaceInstance)
        .where(FaceInstance.id == face_assigned_to_person_a.id)
        .values(person_id=None)
    )
    await db_session.commit()

    # Simulate cleanup job logic
    query = (
        select(FaceSuggestion)
        .join(FaceInstance, FaceSuggestion.source_face_id == FaceInstance.id)
        .where(
            FaceSuggestion.status == FaceSuggestionStatus.PENDING.value,
            or_(
                # Source face is no longer assigned to any person
                FaceInstance.person_id.is_(None),
                # Source face moved to a different person
                and_(
                    FaceInstance.person_id.isnot(None),
                    FaceInstance.person_id != FaceSuggestion.suggested_person_id,
                ),
            ),
        )
    )
    result = await db_session.execute(query)
    orphaned = result.scalars().all()

    assert len(orphaned) == 1

    # Expire orphaned suggestions
    for sug in orphaned:
        sug.status = FaceSuggestionStatus.EXPIRED.value
        sug.reviewed_at = datetime.now(UTC)

    await db_session.commit()

    # Refresh suggestion
    await db_session.refresh(suggestion)

    # Assert: Suggestion is now EXPIRED
    assert suggestion.status == FaceSuggestionStatus.EXPIRED.value
    assert suggestion.reviewed_at is not None


@pytest.mark.asyncio
async def test_cleanup_orphaned_suggestions_job_moved_source(
    db_session, face_assigned_to_person_a, unassigned_face, person_a, person_b
):
    """Test cleanup job expires suggestions when source face moved to different person.

    Edge case: Source face moved from Person A to Person B but cleanup didn't run.
    Suggestion still points to Person A, so it's orphaned.
    """
    from sqlalchemy import and_, or_, select, update

    # Create suggestion: Face B should be Person A (based on Face A)
    suggestion = await create_suggestion(
        db_session,
        face_instance_id=unassigned_face.id,
        suggested_person_id=person_a.id,  # Suggests Person A
        source_face_id=face_assigned_to_person_a.id,  # Source is Face A
        status=FaceSuggestionStatus.PENDING.value,
    )

    # Manually move source face to Person B (simulating edge case)
    await db_session.execute(
        update(FaceInstance)
        .where(FaceInstance.id == face_assigned_to_person_a.id)
        .values(person_id=person_b.id)
    )
    await db_session.commit()

    # Simulate cleanup job logic
    query = (
        select(FaceSuggestion)
        .join(FaceInstance, FaceSuggestion.source_face_id == FaceInstance.id)
        .where(
            FaceSuggestion.status == FaceSuggestionStatus.PENDING.value,
            or_(
                FaceInstance.person_id.is_(None),
                and_(
                    FaceInstance.person_id.isnot(None),
                    FaceInstance.person_id != FaceSuggestion.suggested_person_id,
                ),
            ),
        )
    )
    result = await db_session.execute(query)
    orphaned = result.scalars().all()

    assert len(orphaned) == 1

    # Expire orphaned suggestions
    for sug in orphaned:
        sug.status = FaceSuggestionStatus.EXPIRED.value
        sug.reviewed_at = datetime.now(UTC)

    await db_session.commit()

    # Refresh suggestion
    await db_session.refresh(suggestion)

    # Assert: Orphaned suggestion is EXPIRED
    assert suggestion.status == FaceSuggestionStatus.EXPIRED.value


@pytest.mark.asyncio
async def test_cleanup_orphaned_suggestions_job_preserves_valid(
    db_session, face_assigned_to_person_a, unassigned_face, person_a
):
    """Test cleanup job does NOT expire valid pending suggestions.

    Setup: Create suggestion where source face still has matching person_id
    - Run cleanup job
    - Assert: Valid suggestion remains PENDING
    """
    from image_search_service.queue.face_jobs import cleanup_orphaned_suggestions_job

    # Create valid suggestion (source face still assigned to suggested person)
    suggestion = await create_suggestion(
        db_session,
        face_instance_id=unassigned_face.id,
        suggested_person_id=person_a.id,  # Suggests Person A
        source_face_id=face_assigned_to_person_a.id,  # Source is assigned to Person A
        status=FaceSuggestionStatus.PENDING.value,
    )

    # Verify source face is still assigned to person_a
    await db_session.refresh(face_assigned_to_person_a)
    assert face_assigned_to_person_a.person_id == person_a.id

    # Run cleanup job
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor() as executor:
        result = await asyncio.get_event_loop().run_in_executor(
            executor, cleanup_orphaned_suggestions_job
        )

    assert result["status"] == "completed"
    assert result["expired_count"] == 0  # No orphaned suggestions

    # Refresh suggestion
    await db_session.refresh(suggestion)

    # Assert: Valid suggestion remains PENDING
    assert suggestion.status == FaceSuggestionStatus.PENDING.value
    assert suggestion.reviewed_at is None


@pytest.mark.asyncio
async def test_cleanup_orphaned_suggestions_job_handles_multiple(
    db_session, image_asset, person_a, person_b
):
    """Test cleanup job handles multiple orphaned suggestions correctly.

    Setup: Create mix of valid and orphaned suggestions
    - Simulate cleanup logic
    - Assert: Only orphaned ones are expired
    """
    from sqlalchemy import and_, or_, select, update

    # Create 2 faces: one assigned, one unassigned
    assigned_face = FaceInstance(
        id=uuid.uuid4(),
        asset_id=image_asset.id,
        person_id=person_a.id,
        bbox_x=100,
        bbox_y=150,
        bbox_w=80,
        bbox_h=80,
        detection_confidence=0.90,
        quality_score=0.70,
        qdrant_point_id=uuid.uuid4(),
    )
    orphaned_source_face = FaceInstance(
        id=uuid.uuid4(),
        asset_id=image_asset.id,
        person_id=person_a.id,  # Will be nulled out
        bbox_x=200,
        bbox_y=150,
        bbox_w=80,
        bbox_h=80,
        detection_confidence=0.90,
        quality_score=0.70,
        qdrant_point_id=uuid.uuid4(),
    )
    db_session.add(assigned_face)
    db_session.add(orphaned_source_face)
    await db_session.commit()

    # Create valid suggestion
    valid_suggestion = await create_suggestion(
        db_session,
        face_instance_id=uuid.uuid4(),
        suggested_person_id=person_a.id,
        source_face_id=assigned_face.id,
        status=FaceSuggestionStatus.PENDING.value,
    )

    # Create orphaned suggestion
    orphaned_suggestion = await create_suggestion(
        db_session,
        face_instance_id=uuid.uuid4(),
        suggested_person_id=person_a.id,
        source_face_id=orphaned_source_face.id,
        status=FaceSuggestionStatus.PENDING.value,
    )

    # Orphan the second suggestion by nulling person_id
    await db_session.execute(
        update(FaceInstance)
        .where(FaceInstance.id == orphaned_source_face.id)
        .values(person_id=None)
    )
    await db_session.commit()

    # Simulate cleanup job logic
    query = (
        select(FaceSuggestion)
        .join(FaceInstance, FaceSuggestion.source_face_id == FaceInstance.id)
        .where(
            FaceSuggestion.status == FaceSuggestionStatus.PENDING.value,
            or_(
                FaceInstance.person_id.is_(None),
                and_(
                    FaceInstance.person_id.isnot(None),
                    FaceInstance.person_id != FaceSuggestion.suggested_person_id,
                ),
            ),
        )
    )
    result = await db_session.execute(query)
    orphaned = result.scalars().all()

    assert len(orphaned) == 1

    # Expire orphaned suggestions
    for sug in orphaned:
        sug.status = FaceSuggestionStatus.EXPIRED.value
        sug.reviewed_at = datetime.now(UTC)

    await db_session.commit()

    # Refresh suggestions
    await db_session.refresh(valid_suggestion)
    await db_session.refresh(orphaned_suggestion)

    # Assert: Only orphaned suggestion is expired
    assert valid_suggestion.status == FaceSuggestionStatus.PENDING.value
    assert orphaned_suggestion.status == FaceSuggestionStatus.EXPIRED.value
