"""Race condition tests for HTTP-level concurrent operations.

These tests document race conditions in the face suggestion acceptance flow
by demonstrating that concurrent operations succeed when they should conflict.

BUG DOCUMENTATION TESTS (not fixes):
- These tests PASS showing the bug exists (both accepts succeed)
- When locks are added, these tests should be updated to verify conflicts
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
from httpx import AsyncClient

from image_search_service.db.models import (
    FaceInstance,
    FaceSuggestion,
    FaceSuggestionStatus,
)


@pytest.mark.asyncio
async def test_accept_suggestion_when_concurrent_accepts_then_both_succeed_no_locking(
    test_client: AsyncClient,
    db_session: AsyncMock,
) -> None:
    """Two concurrent accepts on the same PENDING suggestion.

    Expected current behavior (BUG): Both succeed with 200.
    Expected correct behavior: One succeeds, one fails with 409 Conflict.

    The race window exists because there is no SELECT ... FOR UPDATE:
    1. Request A reads suggestion (status=PENDING)
    2. Request B reads suggestion (status=PENDING)  <-- stale read
    3. Both pass status check
    4. Both commit (second writer wins silently)
    """
    suggestion_id = 42
    face_id = UUID("00000000-0000-0000-0000-000000000064")  # face_id = 100
    person_a_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    person_b_id = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")

    # Create mock suggestion that starts as PENDING
    mock_suggestion = MagicMock(spec=FaceSuggestion)
    mock_suggestion.id = suggestion_id
    mock_suggestion.status = FaceSuggestionStatus.PENDING.value
    mock_suggestion.face_instance_id = face_id
    mock_suggestion.suggested_person_id = person_a_id
    mock_suggestion.source_face_id = UUID("11111111-1111-1111-1111-111111111111")
    mock_suggestion.confidence = 0.95

    mock_face = MagicMock(spec=FaceInstance)
    mock_face.id = face_id
    mock_face.person_id = None
    mock_face.asset_id = 1
    mock_face.qdrant_point_id = UUID("22222222-2222-2222-2222-222222222222")
    mock_face.bbox_x = 100
    mock_face.bbox_y = 100
    mock_face.bbox_w = 50
    mock_face.bbox_h = 50
    mock_face.detection_confidence = 0.9
    mock_face.quality_score = 0.8

    # Track operation interleaving
    operation_log: list[str] = []

    # Mock db.get to return the same PENDING suggestion for both requests
    async def tracked_get(model: type, pk: Any, **kwargs: Any) -> Any:
        operation_log.append(f"get_{model.__name__}_{pk}")
        # Add small delay to widen race window
        await asyncio.sleep(0.01)
        if model == FaceSuggestion:
            return mock_suggestion
        if model == FaceInstance:
            return mock_face
        # Return None for Person, ImageAsset (will be handled by route)
        return None

    db_session.get = tracked_get

    commit_count = 0

    async def counting_commit() -> None:
        nonlocal commit_count
        commit_count += 1
        operation_log.append(f"commit_{commit_count}")

    db_session.commit = counting_commit
    db_session.refresh = AsyncMock()

    # Mock Qdrant client to avoid errors
    with patch(
        "image_search_service.api.routes.face_suggestions.get_face_qdrant_client"
    ) as mock_qdrant_getter:
        mock_qdrant = MagicMock()
        mock_qdrant.update_person_ids = MagicMock()
        mock_qdrant_getter.return_value = mock_qdrant

        # Fire both accepts concurrently
        results = await asyncio.gather(
            test_client.post(
                f"/api/v1/faces/suggestions/{suggestion_id}/accept",
                json={"person_id": str(person_a_id)},
            ),
            test_client.post(
                f"/api/v1/faces/suggestions/{suggestion_id}/accept",
                json={"person_id": str(person_b_id)},
            ),
            return_exceptions=True,
        )

    # BUG DOCUMENTATION: Both requests succeed (no locking prevents double-accept)
    success_count = sum(
        1
        for r in results
        if not isinstance(r, Exception) and hasattr(r, "status_code") and r.status_code == 200
    )
    assert success_count == 2, (
        f"Expected both accepts to succeed (demonstrating race condition), "
        f"got {[r.status_code if hasattr(r, 'status_code') else str(r) for r in results]}"
    )

    # Both transactions committed without conflict
    assert commit_count == 2, "Both transactions committed without conflict"

    # Both requests read the suggestion as PENDING
    suggestion_reads = [log for log in operation_log if "FaceSuggestion_42" in log]
    assert len(suggestion_reads) == 2, "Both requests read the suggestion"


@pytest.mark.asyncio
async def test_bulk_action_when_overlapping_sets_then_no_conflict_detection(
    test_client: AsyncClient,
    db_session: AsyncMock,
) -> None:
    """Two bulk-accept requests with overlapping suggestion sets.

    Bulk A accepts [1, 2, 3], Bulk B accepts [2, 3, 4].
    Suggestions 2 and 3 are in both sets.

    Expected current behavior (BUG): Both succeed, suggestions 2 and 3
    get processed twice without conflict detection.

    Expected correct behavior: Conflict detection or row-level locking
    prevents duplicate processing.
    """
    # Setup mock suggestions (all PENDING)
    mock_suggestions: dict[int, MagicMock] = {}
    mock_faces: dict[UUID, MagicMock] = {}

    for sid in [1, 2, 3, 4]:
        suggestion = MagicMock(spec=FaceSuggestion)
        suggestion.id = sid
        suggestion.status = FaceSuggestionStatus.PENDING.value
        face_uuid = UUID(f"00000000-0000-0000-0000-0000000000{sid:02x}")
        suggestion.face_instance_id = face_uuid
        suggestion.suggested_person_id = UUID(
            f"{sid:08x}-0000-0000-0000-000000000000"
        )
        mock_suggestions[sid] = suggestion

        face = MagicMock(spec=FaceInstance)
        face.id = face_uuid
        face.person_id = None
        face.qdrant_point_id = UUID(f"11111111-1111-1111-1111-1111111111{sid:02x}")
        face.asset_id = sid
        mock_faces[face_uuid] = face

    status_changes: list[str] = []

    async def tracked_get(model: type, pk: Any, **kwargs: Any) -> Any:
        if model == FaceSuggestion and pk in mock_suggestions:
            status_changes.append(
                f"read_suggestion_{pk}_status={mock_suggestions[pk].status}"
            )
            return mock_suggestions[pk]
        if model == FaceInstance and pk in mock_faces:
            return mock_faces[pk]
        return None

    db_session.get = tracked_get
    db_session.commit = AsyncMock()
    db_session.refresh = AsyncMock()

    with patch(
        "image_search_service.api.routes.face_suggestions.get_face_qdrant_client"
    ) as mock_qdrant_getter:
        mock_qdrant = MagicMock()
        mock_qdrant.update_person_ids = MagicMock()
        mock_qdrant_getter.return_value = mock_qdrant

        results = await asyncio.gather(
            test_client.post(
                "/api/v1/faces/suggestions/bulk-action",
                json={"action": "accept", "suggestion_ids": [1, 2, 3]},
            ),
            test_client.post(
                "/api/v1/faces/suggestions/bulk-action",
                json={"action": "accept", "suggestion_ids": [2, 3, 4]},
            ),
            return_exceptions=True,
        )

    # Count how many times overlapping suggestions (2, 3) were read as PENDING
    pending_reads_for_2 = [
        e for e in status_changes if "read_suggestion_2_status=pending" in e
    ]
    pending_reads_for_3 = [
        e for e in status_changes if "read_suggestion_3_status=pending" in e
    ]

    # BUG DOCUMENTATION: Both requests read overlapping suggestions as PENDING
    # because there is no row-level locking
    assert (
        len(pending_reads_for_2) >= 1
    ), "Suggestion 2 was read at least once as PENDING"
    assert (
        len(pending_reads_for_3) >= 1
    ), "Suggestion 3 was read at least once as PENDING"

    # Both requests succeeded
    success_count = sum(
        1
        for r in results
        if not isinstance(r, Exception) and hasattr(r, "status_code") and r.status_code == 200
    )
    assert success_count == 2, "Both bulk actions succeeded"


@pytest.mark.asyncio
async def test_accept_suggestion_when_qdrant_fails_then_db_not_rolled_back(
    test_client: AsyncClient,
    db_session: AsyncMock,
) -> None:
    """Accept suggestion: DB commits, then Qdrant fails.

    Expected current behavior (BUG): API returns 200 (success), but DB and Qdrant
    are out of sync. The face has person_id in PostgreSQL but NOT in Qdrant.
    Error is only logged, not returned to user.

    Expected correct behavior: Either rollback DB on Qdrant failure, or return
    partial success status (207 Multi-Status), or enqueue reconciliation job.
    """
    suggestion_id = 42
    person_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    face_id = UUID("00000000-0000-0000-0000-000000000064")

    mock_suggestion = MagicMock(spec=FaceSuggestion)
    mock_suggestion.id = suggestion_id
    mock_suggestion.status = FaceSuggestionStatus.PENDING.value
    mock_suggestion.face_instance_id = face_id
    mock_suggestion.suggested_person_id = person_id
    mock_suggestion.confidence = 0.95
    mock_suggestion.source_face_id = UUID("11111111-1111-1111-1111-111111111111")

    mock_face = MagicMock(spec=FaceInstance)
    mock_face.id = face_id
    mock_face.person_id = None
    mock_face.qdrant_point_id = UUID("22222222-2222-2222-2222-222222222222")
    mock_face.asset_id = 1
    mock_face.bbox_x = 100
    mock_face.bbox_y = 100
    mock_face.bbox_w = 50
    mock_face.bbox_h = 50
    mock_face.detection_confidence = 0.9
    mock_face.quality_score = 0.8

    async def mock_get(model: type, pk: Any, **kwargs: Any) -> Any:
        if model == FaceSuggestion:
            return mock_suggestion
        if model == FaceInstance:
            return mock_face
        return None

    db_session.get = mock_get
    db_session.commit = AsyncMock()  # DB commit succeeds
    db_session.refresh = AsyncMock()

    # Mock Qdrant to fail
    mock_qdrant = MagicMock()
    mock_qdrant.update_person_ids.side_effect = ConnectionError(
        "Qdrant cluster unavailable"
    )

    with patch(
        "image_search_service.api.routes.face_suggestions.get_face_qdrant_client",
        return_value=mock_qdrant,
    ):
        response = await test_client.post(
            f"/api/v1/faces/suggestions/{suggestion_id}/accept",
            json={"person_id": str(person_id)},
        )

    # BUG DOCUMENTATION: API returns 200 despite Qdrant failure
    assert response.status_code == 200, (
        "Accept returns success even though Qdrant is out of sync"
    )

    # DB was committed (face.person_id changed)
    db_session.commit.assert_called_once()

    # Qdrant was attempted but failed
    mock_qdrant.update_person_ids.assert_called_once()

    # No rollback happened -- this IS the bug
    # Correct behavior options:
    # 1. DB commit should be rolled back on Qdrant failure, OR
    # 2. Response should indicate partial success (207 Multi-Status), OR
    # 3. A reconciliation job should be enqueued


@pytest.mark.asyncio
async def test_bulk_action_when_partial_qdrant_sync_failure_then_desync_accepted(
    test_client: AsyncClient,
    db_session: AsyncMock,
) -> None:
    """Bulk accept with partial Qdrant sync failure.

    DB commits all 3 face assignments, then Qdrant sync fails on person-b.
    Result: 2 faces synced in Qdrant, 1 face desynced.

    Expected current behavior (BUG): API returns 200 with processed=3,
    no indication of partial Qdrant sync failure.

    Expected correct behavior: Return 207 Multi-Status or enqueue repair job.
    """
    # Setup 3 suggestions for 2 different people
    person_a = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    person_b = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")

    mock_suggestions: dict[int, MagicMock] = {}
    mock_faces: dict[UUID, MagicMock] = {}

    for sid, pid in [(1, person_a), (2, person_b), (3, person_a)]:
        suggestion = MagicMock(spec=FaceSuggestion)
        suggestion.id = sid
        suggestion.status = FaceSuggestionStatus.PENDING.value
        face_uuid = UUID(f"00000000-0000-0000-0000-0000000000{sid:02x}")
        suggestion.face_instance_id = face_uuid
        suggestion.suggested_person_id = pid
        mock_suggestions[sid] = suggestion

        face = MagicMock(spec=FaceInstance)
        face.id = face_uuid
        face.person_id = None
        face.qdrant_point_id = UUID(f"11111111-1111-1111-1111-1111111111{sid:02x}")
        face.asset_id = sid
        mock_faces[face_uuid] = face

    async def mock_get(model: type, pk: Any, **kwargs: Any) -> Any:
        if model == FaceSuggestion and pk in mock_suggestions:
            return mock_suggestions[pk]
        if model == FaceInstance and pk in mock_faces:
            return mock_faces[pk]
        return None

    db_session.get = mock_get
    db_session.commit = AsyncMock()
    db_session.refresh = AsyncMock()

    # Qdrant fails for person-b, succeeds for person-a
    call_count = 0

    def qdrant_update_with_failure(
        point_ids: list[UUID], person_id: UUID
    ) -> None:
        nonlocal call_count
        call_count += 1
        if person_id == person_b:
            raise ConnectionError("Qdrant timeout on person-b sync")

    mock_qdrant = MagicMock()
    mock_qdrant.update_person_ids.side_effect = qdrant_update_with_failure

    with patch(
        "image_search_service.api.routes.face_suggestions.get_face_qdrant_client",
        return_value=mock_qdrant,
    ):
        response = await test_client.post(
            "/api/v1/faces/suggestions/bulk-action",
            json={"action": "accept", "suggestion_ids": [1, 2, 3]},
        )

    # BUG DOCUMENTATION: API returns 200 with processed=3 -- no indication of Qdrant desync
    assert response.status_code == 200
    data = response.json()
    assert data["processed"] == 3, "All 3 processed in DB despite Qdrant failure"

    # DB commit happened once (single batch)
    db_session.commit.assert_called_once()

    # Qdrant: person-a synced, person-b failed
    # Result: faces for person-a are in sync, face for person-b is desynced
    assert call_count >= 1, "At least one Qdrant sync was attempted"
