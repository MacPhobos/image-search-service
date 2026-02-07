"""Race condition tests for centroid computation service.

These tests document race conditions in concurrent centroid computation
for the same person, demonstrating that multiple computations can proceed
simultaneously without coordination.

BUG DOCUMENTATION TESTS (not fixes):
- Tests PASS showing bugs exist (double deprecation, both compute)
- When locks are added, update tests to verify proper serialization
"""

import asyncio
import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest


@pytest.mark.asyncio
async def test_compute_centroids_when_concurrent_same_person_then_double_deprecation() -> None:
    """Two concurrent centroid computations for the same person.

    Expected current behavior (BUG):
    1. Both read existing centroid as "stale" (or no centroid exists)
    2. Both deprecate existing centroids (double-deprecation)
    3. Both compute new centroids
    4. Second INSERT may violate unique partial index, OR
    5. Both create centroids, one overwrites the other in Qdrant

    Expected correct behavior: Only one computation proceeds at a time
    via SELECT ... FOR UPDATE on person_id or centroid record locking.
    """
    from image_search_service.services.centroid_service import (
        compute_centroids_for_person,
    )

    person_id = uuid.uuid4()

    # Mock database session
    db_session = AsyncMock()

    # Mock Qdrant clients
    mock_face_qdrant = MagicMock()
    mock_centroid_qdrant = MagicMock()

    # Mock embeddings retrieval (5 faces, sufficient for centroid)
    fake_embeddings = np.random.randn(5, 512).astype(np.float32).tolist()
    fake_face_ids = [uuid.uuid4() for _ in range(5)]

    # Track deprecation calls
    deprecation_count = 0

    async def mock_deprecate_centroids(
        db: AsyncMock, centroid_qdrant: MagicMock, pid: uuid.UUID
    ) -> int:
        nonlocal deprecation_count
        deprecation_count += 1
        # Simulate deprecating 1 existing centroid
        return 1

    with patch(
        "image_search_service.services.centroid_service.get_person_face_embeddings",
        return_value=(fake_face_ids, fake_embeddings),
    ), patch(
        "image_search_service.services.centroid_service.deprecate_centroids",
        side_effect=mock_deprecate_centroids,
    ), patch(
        "image_search_service.services.centroid_service.get_settings",
    ) as mock_settings:
        # Mock settings
        settings_mock = MagicMock()
        settings_mock.centroid_min_faces = 3
        settings_mock.centroid_model_version = "v1"
        settings_mock.centroid_algorithm_version = 1
        settings_mock.centroid_trim_threshold_small = 0.05
        settings_mock.centroid_trim_threshold_large = 0.1
        mock_settings.return_value = settings_mock

        # Mock database operations
        db_session.execute = AsyncMock(
            return_value=MagicMock(
                scalar_one_or_none=MagicMock(return_value=None)
            )
        )
        db_session.flush = AsyncMock()

        # Mock db.add to set created_at timestamp (server_default in real DB)
        def mock_add(obj: Any) -> None:
            if not hasattr(obj, "created_at") or obj.created_at is None:
                obj.created_at = datetime.now(UTC)

        db_session.add = MagicMock(side_effect=mock_add)

        # Mock Qdrant upsert
        mock_centroid_qdrant.upsert_centroid = MagicMock()

        # Run two concurrent computations
        results = await asyncio.gather(
            compute_centroids_for_person(
                db_session,
                mock_face_qdrant,
                mock_centroid_qdrant,
                person_id,
                force_rebuild=True,  # Force rebuild to bypass staleness check
            ),
            compute_centroids_for_person(
                db_session,
                mock_face_qdrant,
                mock_centroid_qdrant,
                person_id,
                force_rebuild=True,
            ),
            return_exceptions=True,
        )

    # BUG DOCUMENTATION: Both calls deprecated centroids (double-deprecation)
    assert deprecation_count == 2, (
        f"Expected 2 deprecation calls (race condition), got {deprecation_count}"
    )

    # Both calls attempted to create new centroids
    assert db_session.add.call_count == 2, (
        f"Expected 2 centroid additions (race condition), got {db_session.add.call_count}"
    )

    # Both calls attempted Qdrant upserts
    upsert_count = mock_centroid_qdrant.upsert_centroid.call_count
    assert upsert_count == 2, (
        f"Expected 2 Qdrant upserts (race condition), got {upsert_count}"
    )

    # Both results should be PersonCentroid objects (not exceptions)
    assert not isinstance(
        results[0], Exception
    ), f"First computation failed: {results[0]}"
    assert not isinstance(
        results[1], Exception
    ), f"Second computation failed: {results[1]}"


@pytest.mark.asyncio
async def test_centroid_computation_when_qdrant_fails_then_no_active_centroid() -> None:
    """Centroid computation: DB flush succeeds, Qdrant upsert fails.

    Expected behavior: centroid.status set to FAILED, exception re-raised.
    BUT: The deprecation of the OLD centroid already happened (line 325 in centroid_service.py),
    so the person now has NO active centroid (old deprecated, new failed).

    This is a transactional gap -- the person is left without an active centroid
    until the next successful computation.

    Expected correct behavior: Either use a transaction rollback on Qdrant failure,
    or implement a two-phase commit pattern, or accept eventual consistency with
    a background repair job.
    """
    from image_search_service.services.centroid_service import (
        compute_centroids_for_person,
    )

    person_id = uuid.uuid4()

    # Mock database session
    db_session = AsyncMock()

    # Mock Qdrant clients
    mock_face_qdrant = MagicMock()
    mock_centroid_qdrant = MagicMock()
    mock_centroid_qdrant.upsert_centroid.side_effect = ConnectionError(
        "Qdrant down"
    )

    # Mock embeddings retrieval
    fake_embeddings = np.random.randn(5, 512).astype(np.float32).tolist()
    fake_face_ids = [uuid.uuid4() for _ in range(5)]

    added_objects: list[MagicMock] = []

    with patch(
        "image_search_service.services.centroid_service.get_person_face_embeddings",
        return_value=(fake_face_ids, fake_embeddings),
    ), patch(
        "image_search_service.services.centroid_service.deprecate_centroids",
    ) as mock_deprecate, patch(
        "image_search_service.services.centroid_service.get_settings",
    ) as mock_settings:
        # Mock settings
        settings_mock = MagicMock()
        settings_mock.centroid_min_faces = 3
        settings_mock.centroid_model_version = "v1"
        settings_mock.centroid_algorithm_version = 1
        settings_mock.centroid_trim_threshold_small = 0.05
        settings_mock.centroid_trim_threshold_large = 0.1
        mock_settings.return_value = settings_mock

        # Mock database operations
        db_session.execute = AsyncMock(
            return_value=MagicMock(
                scalar_one_or_none=MagicMock(return_value=None)
            )
        )
        db_session.flush = AsyncMock()

        # Mock db.add to set created_at and track objects
        def mock_add_with_timestamp(obj: Any) -> None:
            if not hasattr(obj, "created_at") or obj.created_at is None:
                obj.created_at = datetime.now(UTC)
            added_objects.append(obj)

        db_session.add = MagicMock(side_effect=mock_add_with_timestamp)

        # Execute and expect exception
        with pytest.raises(ConnectionError, match="Qdrant down"):
            await compute_centroids_for_person(
                db_session,
                mock_face_qdrant,
                mock_centroid_qdrant,
                person_id,
                force_rebuild=True,
            )

    # Old centroid was deprecated BEFORE the failure
    mock_deprecate.assert_called_once_with(
        db_session, mock_centroid_qdrant, person_id
    )

    # New centroid was added to DB
    assert len(added_objects) == 1

    # New centroid was marked as FAILED (code sets centroid.status = CentroidStatus.FAILED)
    new_centroid = added_objects[0]
    from image_search_service.db.models import CentroidStatus

    assert new_centroid.status == CentroidStatus.FAILED

    # BUG: Old centroid deprecated + new centroid FAILED = no active centroid
    # This is a transactional gap -- the caller must handle this edge case
    # The person now has NO active centroid until next successful computation
