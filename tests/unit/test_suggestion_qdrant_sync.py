"""Unit tests for suggestion acceptance Qdrant sync (C2 bug fix).

Tests that accepting face suggestions properly syncs person_id to Qdrant,
preventing drift between PostgreSQL and Qdrant.
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
async def mock_person(db_session):
    """Create a mock Person in the database (async)."""
    person = Person(
        id=uuid.uuid4(),
        name="Test Person",
        status=PersonStatus.ACTIVE.value,
    )
    db_session.add(person)
    await db_session.commit()
    await db_session.refresh(person)
    return person


@pytest.fixture
async def mock_person2(db_session):
    """Create a second mock Person in the database (async)."""
    person = Person(
        id=uuid.uuid4(),
        name="Second Person",
        status=PersonStatus.ACTIVE.value,
    )
    db_session.add(person)
    await db_session.commit()
    await db_session.refresh(person)
    return person


@pytest.fixture
async def mock_image_asset(db_session):
    """Create a mock ImageAsset in the database (async)."""
    asset = ImageAsset(
        path=f"/test/images/photo_{uuid.uuid4().hex[:8]}.jpg",
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
async def mock_face_instance(db_session, mock_image_asset):
    """Create a mock FaceInstance in the database (async)."""
    face = FaceInstance(
        id=uuid.uuid4(),
        asset_id=mock_image_asset.id,
        person_id=None,  # Unassigned
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
async def mock_face_suggestion(db_session, mock_face_instance, mock_person):
    """Create a mock FaceSuggestion in the database (async)."""
    suggestion = FaceSuggestion(
        face_instance_id=mock_face_instance.id,
        suggested_person_id=mock_person.id,
        confidence=0.9,
        source_face_id=mock_face_instance.id,
        status=FaceSuggestionStatus.PENDING.value,
        created_at=datetime.now(UTC),
    )
    db_session.add(suggestion)
    await db_session.commit()
    await db_session.refresh(suggestion)
    return suggestion


# ============ Tests ============


@pytest.mark.asyncio
async def test_accept_suggestion_syncs_to_qdrant(
    test_client,
    db_session,
    mock_face_instance,
    mock_person,
    mock_face_suggestion,
):
    """Test that accepting a single suggestion calls update_person_ids on Qdrant."""
    # Mock Qdrant client
    with patch(
        "image_search_service.api.routes.face_suggestions.get_face_qdrant_client"
    ) as mock_get_qdrant:
        mock_qdrant = MagicMock()
        mock_get_qdrant.return_value = mock_qdrant

        # Execute: Accept the suggestion
        response = await test_client.post(
            f"/api/v1/faces/suggestions/{mock_face_suggestion.id}/accept",
            json={},
        )

        # Assert: HTTP response successful
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == FaceSuggestionStatus.ACCEPTED.value

        # Assert: Qdrant update_person_ids was called with correct args
        mock_qdrant.update_person_ids.assert_called_once_with(
            [mock_face_instance.qdrant_point_id],
            mock_person.id,
        )


@pytest.mark.asyncio
async def test_accept_suggestion_continues_on_qdrant_failure(
    test_client,
    db_session,
    mock_face_instance,
    mock_person,
    mock_face_suggestion,
):
    """Test that Qdrant failure doesn't break suggestion acceptance (error logged, not raised)."""
    # Mock Qdrant client to raise exception
    with patch(
        "image_search_service.api.routes.face_suggestions.get_face_qdrant_client"
    ) as mock_get_qdrant:
        mock_qdrant = MagicMock()
        mock_qdrant.update_person_ids.side_effect = Exception("Qdrant connection failed")
        mock_get_qdrant.return_value = mock_qdrant

        # Execute: Accept the suggestion
        response = await test_client.post(
            f"/api/v1/faces/suggestions/{mock_face_suggestion.id}/accept",
            json={},
        )

        # Assert: HTTP response still successful (DB commit succeeded)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == FaceSuggestionStatus.ACCEPTED.value

        # Assert: Qdrant was attempted but failure didn't propagate
        mock_qdrant.update_person_ids.assert_called_once()


@pytest.mark.asyncio
async def test_bulk_accept_syncs_to_qdrant_batched(
    test_client,
    db_session,
    mock_image_asset,
    mock_person,
    mock_person2,
):
    """Test that bulk accepting suggestions calls update_person_ids with batched point IDs."""
    # Create multiple faces and suggestions
    face1 = FaceInstance(
        id=uuid.uuid4(),
        asset_id=mock_image_asset.id,
        person_id=None,
        bbox_x=100,
        bbox_y=150,
        bbox_w=80,
        bbox_h=80,
        detection_confidence=0.95,
        quality_score=0.75,
        qdrant_point_id=uuid.uuid4(),
    )
    face2 = FaceInstance(
        id=uuid.uuid4(),
        asset_id=mock_image_asset.id,
        person_id=None,
        bbox_x=200,
        bbox_y=250,
        bbox_w=80,
        bbox_h=80,
        detection_confidence=0.95,
        quality_score=0.75,
        qdrant_point_id=uuid.uuid4(),
    )
    face3 = FaceInstance(
        id=uuid.uuid4(),
        asset_id=mock_image_asset.id,
        person_id=None,
        bbox_x=300,
        bbox_y=350,
        bbox_w=80,
        bbox_h=80,
        detection_confidence=0.95,
        quality_score=0.75,
        qdrant_point_id=uuid.uuid4(),
    )

    db_session.add_all([face1, face2, face3])
    await db_session.commit()
    await db_session.refresh(face1)
    await db_session.refresh(face2)
    await db_session.refresh(face3)

    suggestion1 = FaceSuggestion(
        face_instance_id=face1.id,
        suggested_person_id=mock_person.id,
        confidence=0.9,
        source_face_id=face1.id,
        status=FaceSuggestionStatus.PENDING.value,
        created_at=datetime.now(UTC),
    )
    suggestion2 = FaceSuggestion(
        face_instance_id=face2.id,
        suggested_person_id=mock_person.id,
        confidence=0.85,
        source_face_id=face2.id,
        status=FaceSuggestionStatus.PENDING.value,
        created_at=datetime.now(UTC),
    )
    suggestion3 = FaceSuggestion(
        face_instance_id=face3.id,
        suggested_person_id=mock_person2.id,
        confidence=0.8,
        source_face_id=face3.id,
        status=FaceSuggestionStatus.PENDING.value,
        created_at=datetime.now(UTC),
    )

    db_session.add_all([suggestion1, suggestion2, suggestion3])
    await db_session.commit()
    await db_session.refresh(suggestion1)
    await db_session.refresh(suggestion2)
    await db_session.refresh(suggestion3)

    # Mock Qdrant client
    with patch(
        "image_search_service.api.routes.face_suggestions.get_face_qdrant_client"
    ) as mock_get_qdrant:
        mock_qdrant = MagicMock()
        mock_get_qdrant.return_value = mock_qdrant

        # Execute: Bulk accept all suggestions
        response = await test_client.post(
            "/api/v1/faces/suggestions/bulk-action",
            json={
                "action": "accept",
                "suggestion_ids": [suggestion1.id, suggestion2.id, suggestion3.id],
                "auto_find_more": False,
            },
        )

        # Assert: HTTP response successful
        assert response.status_code == 200
        data = response.json()
        assert data["processed"] == 3
        assert data["failed"] == 0

        # Assert: Qdrant update_person_ids called for each person with batched point IDs
        assert mock_qdrant.update_person_ids.call_count == 2

        # Check calls (order may vary, so use set comparison)
        calls = mock_qdrant.update_person_ids.call_args_list
        call_map = {call[0][1]: call[0][0] for call in calls}  # person_id -> point_ids

        # Person 1 should have face1 and face2
        assert mock_person.id in call_map
        assert set(call_map[mock_person.id]) == {face1.qdrant_point_id, face2.qdrant_point_id}

        # Person 2 should have face3
        assert mock_person2.id in call_map
        assert set(call_map[mock_person2.id]) == {face3.qdrant_point_id}


@pytest.mark.asyncio
async def test_bulk_reject_does_not_sync_to_qdrant(
    test_client,
    db_session,
    mock_face_instance,
    mock_person,
    mock_face_suggestion,
):
    """Test that rejecting suggestions does NOT sync to Qdrant (no person assignment)."""
    # Mock Qdrant client
    with patch(
        "image_search_service.api.routes.face_suggestions.get_face_qdrant_client"
    ) as mock_get_qdrant:
        mock_qdrant = MagicMock()
        mock_get_qdrant.return_value = mock_qdrant

        # Execute: Reject the suggestion
        response = await test_client.post(
            "/api/v1/faces/suggestions/bulk-action",
            json={
                "action": "reject",
                "suggestion_ids": [mock_face_suggestion.id],
                "auto_find_more": False,
            },
        )

        # Assert: HTTP response successful
        assert response.status_code == 200

        # Assert: Qdrant was NOT called (rejection doesn't assign person)
        mock_qdrant.update_person_ids.assert_not_called()


@pytest.mark.asyncio
async def test_bulk_accept_continues_on_partial_qdrant_failure(
    test_client,
    db_session,
    mock_image_asset,
    mock_person,
    mock_person2,
):
    """Test that partial Qdrant failures don't break bulk acceptance."""
    # Create two faces and suggestions for different persons
    face1 = FaceInstance(
        id=uuid.uuid4(),
        asset_id=mock_image_asset.id,
        person_id=None,
        bbox_x=100,
        bbox_y=150,
        bbox_w=80,
        bbox_h=80,
        detection_confidence=0.95,
        quality_score=0.75,
        qdrant_point_id=uuid.uuid4(),
    )
    face2 = FaceInstance(
        id=uuid.uuid4(),
        asset_id=mock_image_asset.id,
        person_id=None,
        bbox_x=200,
        bbox_y=250,
        bbox_w=80,
        bbox_h=80,
        detection_confidence=0.95,
        quality_score=0.75,
        qdrant_point_id=uuid.uuid4(),
    )

    db_session.add_all([face1, face2])
    await db_session.commit()
    await db_session.refresh(face1)
    await db_session.refresh(face2)

    suggestion1 = FaceSuggestion(
        face_instance_id=face1.id,
        suggested_person_id=mock_person.id,
        confidence=0.9,
        source_face_id=face1.id,
        status=FaceSuggestionStatus.PENDING.value,
        created_at=datetime.now(UTC),
    )
    suggestion2 = FaceSuggestion(
        face_instance_id=face2.id,
        suggested_person_id=mock_person2.id,
        confidence=0.85,
        source_face_id=face2.id,
        status=FaceSuggestionStatus.PENDING.value,
        created_at=datetime.now(UTC),
    )

    db_session.add_all([suggestion1, suggestion2])
    await db_session.commit()
    await db_session.refresh(suggestion1)
    await db_session.refresh(suggestion2)

    # Mock Qdrant client to fail on second call
    with patch(
        "image_search_service.api.routes.face_suggestions.get_face_qdrant_client"
    ) as mock_get_qdrant:
        mock_qdrant = MagicMock()
        # First call succeeds, subsequent calls fail
        mock_qdrant.update_person_ids.side_effect = [
            None,  # Success
            Exception("Qdrant partial failure"),
        ]
        mock_get_qdrant.return_value = mock_qdrant

        # Execute: Bulk accept both suggestions
        response = await test_client.post(
            "/api/v1/faces/suggestions/bulk-action",
            json={
                "action": "accept",
                "suggestion_ids": [suggestion1.id, suggestion2.id],
                "auto_find_more": False,
            },
        )

        # Assert: HTTP response still successful (DB commits succeeded)
        assert response.status_code == 200
        data = response.json()
        assert data["processed"] == 2
        assert data["failed"] == 0

        # Assert: Qdrant was attempted for both persons
        assert mock_qdrant.update_person_ids.call_count == 2
