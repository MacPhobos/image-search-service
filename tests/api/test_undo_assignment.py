"""Integration tests for the undo_assignment_event endpoint.

Verifies that the undo endpoint:
1. Reverts faces to unassigned state
2. Enqueues person_ids update jobs for affected assets (bug fix)
3. Returns the correct response structure
"""

import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.db.models import (
    FaceAssignmentEvent,
    FaceInstance,
    ImageAsset,
    Person,
    PersonStatus,
)

# ============ Fixtures ============


@pytest.fixture
async def undo_person(db_session: AsyncSession) -> Person:
    """Create a test person for undo tests."""
    person = Person(name="Undo Test Person", status=PersonStatus.ACTIVE.value)
    db_session.add(person)
    await db_session.commit()
    await db_session.refresh(person)
    return person


@pytest.fixture
async def undo_asset(db_session: AsyncSession) -> ImageAsset:
    """Create a test image asset for undo tests."""
    asset = ImageAsset(
        path="/test/undo_photo.jpg",
        training_status="pending",
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
async def assigned_face(
    db_session: AsyncSession, undo_asset: ImageAsset, undo_person: Person
) -> FaceInstance:
    """Create a face assigned to undo_person."""
    face = FaceInstance(
        id=uuid.uuid4(),
        asset_id=undo_asset.id,
        bbox_x=10,
        bbox_y=10,
        bbox_w=50,
        bbox_h=50,
        detection_confidence=0.90,
        quality_score=0.80,
        qdrant_point_id=uuid.uuid4(),
        person_id=undo_person.id,
    )
    db_session.add(face)
    await db_session.commit()
    await db_session.refresh(face)
    return face


@pytest.fixture
async def assign_event(
    db_session: AsyncSession,
    assigned_face: FaceInstance,
    undo_person: Person,
    undo_asset: ImageAsset,
) -> FaceAssignmentEvent:
    """Create an ASSIGN_TO_PERSON event that can be undone (within 1-hour window)."""
    event = FaceAssignmentEvent(
        id=uuid.uuid4(),
        operation="ASSIGN_TO_PERSON",
        from_person_id=None,
        to_person_id=undo_person.id,
        affected_photo_ids=[undo_asset.id],
        affected_face_instance_ids=[str(assigned_face.id)],
        face_count=1,
        photo_count=1,
        # Use naive datetime to match what SQLite returns after a DB read.
        # The undo endpoint compares event.created_at (naive from SQLite) against
        # datetime.now(UTC) (aware), so we need to patch the time check in tests.
        created_at=datetime.now(UTC),
        note="Test assignment event",
    )
    db_session.add(event)
    await db_session.commit()
    # Do NOT refresh: refreshing would re-read created_at from SQLite as naive,
    # breaking the timezone comparison in the undo endpoint. The expire_on_commit=False
    # session keeps the in-memory (aware) value intact after commit.
    return event


# ============ Tests ============


@pytest.mark.asyncio
class TestUndoAssignmentEnqueue:
    """Tests that undo_assignment_event enqueues person_ids update jobs."""

    async def test_undo_enqueues_person_ids_update(
        self,
        test_client: AsyncClient,
        db_session: AsyncSession,
        assigned_face: FaceInstance,
        undo_person: Person,
        undo_asset: ImageAsset,
        assign_event: FaceAssignmentEvent,
    ) -> None:
        """Undo endpoint enqueues person_ids update for affected assets (bug fix)."""
        mock_qdrant = MagicMock()
        mock_qdrant.update_person_ids = MagicMock(return_value=None)

        # Patch at the worker module where enqueue_person_ids_update is defined,
        # since faces.py uses a lazy `from image_search_service.queue.worker import ...`
        enqueue_patch = "image_search_service.queue.worker.enqueue_person_ids_update"
        qdrant_patch = "image_search_service.vector.face_qdrant.get_face_qdrant_client"
        mock_enqueue = MagicMock(return_value=1)

        with patch(enqueue_patch, mock_enqueue), patch(qdrant_patch, return_value=mock_qdrant):
            response = await test_client.post(
                f"/api/v1/faces/assignment-events/{assign_event.id}/undo"
            )

        assert response.status_code == 200
        data = response.json()
        assert data["facesUnassigned"] == 1
        assert data["personId"] == str(undo_person.id)

        # Verify enqueue was called with the correct asset ID (the bug fix)
        mock_enqueue.assert_called_once()
        enqueued_ids = mock_enqueue.call_args[0][0]
        assert undo_asset.id in enqueued_ids

    async def test_undo_response_structure(
        self,
        test_client: AsyncClient,
        db_session: AsyncSession,
        assigned_face: FaceInstance,
        undo_person: Person,
        undo_asset: ImageAsset,
        assign_event: FaceAssignmentEvent,
    ) -> None:
        """Undo endpoint returns the correct response fields."""
        mock_qdrant = MagicMock()
        mock_qdrant.update_person_ids = MagicMock(return_value=None)

        enqueue_patch = "image_search_service.queue.worker.enqueue_person_ids_update"
        qdrant_patch = "image_search_service.vector.face_qdrant.get_face_qdrant_client"

        with (
            patch(enqueue_patch, MagicMock(return_value=1)),
            patch(qdrant_patch, return_value=mock_qdrant),
        ):
            response = await test_client.post(
                f"/api/v1/faces/assignment-events/{assign_event.id}/undo"
            )

        assert response.status_code == 200
        data = response.json()
        assert "eventId" in data
        assert "facesUnassigned" in data
        assert "personId" in data
        assert "personName" in data
        assert "undoEventId" in data
        assert data["personName"] == undo_person.name
