"""Test for edge case where face references a non-existent person."""

import uuid

import pytest

# mock_image_asset from root conftest.py


@pytest.mark.asyncio
async def test_get_faces_with_orphaned_person_id(test_client, db_session, mock_image_asset):
    """Test that personName is null when face references a deleted person.

    This tests the edge case where person_id is set but the person
    doesn't exist (e.g., was deleted without CASCADE).
    """
    from image_search_service.db.models import FaceInstance

    # Create a face with a person_id that doesn't exist in the database
    orphaned_person_id = uuid.uuid4()

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
        person_id=orphaned_person_id,  # Person doesn't exist!
    )
    db_session.add(face)
    await db_session.commit()

    # Call the endpoint
    response = await test_client.get(f"/api/v1/faces/assets/{mock_image_asset.id}")

    assert response.status_code == 200
    data = response.json()

    assert data["total"] == 1
    face_response = data["items"][0]

    # The face has a person_id but the person doesn't exist
    assert face_response["personId"] == str(orphaned_person_id)

    # personName should be null because the join returns no match
    assert face_response["personName"] is None, (
        "When person doesn't exist, personName should be null (LEFT JOIN behavior)"
    )
