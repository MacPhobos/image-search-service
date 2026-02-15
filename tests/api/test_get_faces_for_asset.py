"""Test for /api/v1/faces/assets/{asset_id} endpoint to verify personName is populated."""

import uuid

import pytest

# mock_image_asset from root conftest.py


@pytest.fixture
async def mock_person(db_session):
    """Create a mock Person in the database."""
    from image_search_service.db.models import Person, PersonStatus

    person = Person(
        id=uuid.uuid4(),
        name="Chantal",
        status=PersonStatus.ACTIVE.value,
    )
    db_session.add(person)
    await db_session.commit()
    await db_session.refresh(person)
    return person


@pytest.mark.asyncio
async def test_get_faces_for_asset_with_person_name(
    test_client, db_session, mock_image_asset, mock_person
):
    """Test that personName is populated when a face is assigned to a person.

    This is a regression test for the bug where personName was always null
    even when personId was set.
    """
    from image_search_service.db.models import FaceInstance

    # Create a face assigned to a person
    face = FaceInstance(
        id=uuid.uuid4(),
        asset_id=mock_image_asset.id,
        bbox_x=100,
        bbox_y=100,
        bbox_w=80,
        bbox_h=80,
        detection_confidence=0.95,
        quality_score=0.85,
        qdrant_point_id=uuid.uuid4(),
        person_id=mock_person.id,  # Assigned to person
    )
    db_session.add(face)
    await db_session.commit()

    # Call the endpoint
    response = await test_client.get(f"/api/v1/faces/assets/{mock_image_asset.id}")

    assert response.status_code == 200
    data = response.json()

    assert data["total"] == 1
    assert len(data["items"]) == 1

    face_response = data["items"][0]
    assert face_response["id"] == str(face.id)
    assert face_response["personId"] == str(mock_person.id)

    # BUG: personName should be "Chantal" but was null
    assert face_response["personName"] == "Chantal", (
        f"Expected personName='Chantal' but got {face_response['personName']!r}. "
        "The query should join with the persons table to populate person names."
    )


@pytest.mark.asyncio
async def test_get_faces_for_asset_without_person(
    test_client, db_session, mock_image_asset
):
    """Test that personName is null when a face has no person assigned."""
    from image_search_service.db.models import FaceInstance

    # Create a face NOT assigned to any person
    face = FaceInstance(
        id=uuid.uuid4(),
        asset_id=mock_image_asset.id,
        bbox_x=100,
        bbox_y=100,
        bbox_w=80,
        bbox_h=80,
        detection_confidence=0.95,
        quality_score=0.85,
        qdrant_point_id=uuid.uuid4(),
        person_id=None,  # No person assigned
    )
    db_session.add(face)
    await db_session.commit()

    # Call the endpoint
    response = await test_client.get(f"/api/v1/faces/assets/{mock_image_asset.id}")

    assert response.status_code == 200
    data = response.json()

    assert data["total"] == 1
    assert len(data["items"]) == 1

    face_response = data["items"][0]
    assert face_response["id"] == str(face.id)
    assert face_response["personId"] is None
    assert face_response["personName"] is None
