"""Unit tests for unlabeled face filtering with is_assigned sentinel."""

import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

from qdrant_client.models import (
    FieldCondition,
    Filter,
    IsEmptyCondition,
    Record,
)

from image_search_service.vector.face_qdrant import FaceQdrantClient


class TestGetUnlabeledFacesWithEmbeddings:
    """Test server-side filtering for unlabeled faces."""

    def test_uses_is_empty_condition_for_person_id(self) -> None:
        """Test that get_unlabeled_faces_with_embeddings uses IsEmptyCondition for person_id."""
        # Setup
        face_client = FaceQdrantClient()
        face_client._client = MagicMock()

        # Mock scroll to return no results (we just want to verify the filter)
        face_client._client.scroll.return_value = ([], None)

        # Execute
        face_client.get_unlabeled_faces_with_embeddings(quality_threshold=0.0, limit=100)

        # Verify
        assert face_client._client.scroll.called
        call_kwargs = face_client._client.scroll.call_args.kwargs

        # Check that scroll_filter contains IsEmptyCondition
        scroll_filter = call_kwargs.get("scroll_filter")
        assert scroll_filter is not None
        assert isinstance(scroll_filter, Filter)
        assert len(scroll_filter.must) >= 1

        # Check for IsEmptyCondition on person_id
        has_is_empty = any(
            isinstance(cond, IsEmptyCondition) and cond.is_empty.key == "person_id"
            for cond in scroll_filter.must
        )
        assert has_is_empty, "Expected IsEmptyCondition for person_id in filter"

    def test_adds_quality_filter_when_threshold_set(self) -> None:
        """Test that quality_threshold adds FieldCondition with Range filter."""
        # Setup
        face_client = FaceQdrantClient()
        face_client._client = MagicMock()
        face_client._client.scroll.return_value = ([], None)

        # Execute with quality threshold
        face_client.get_unlabeled_faces_with_embeddings(quality_threshold=0.7, limit=100)

        # Verify
        call_kwargs = face_client._client.scroll.call_args.kwargs
        scroll_filter = call_kwargs.get("scroll_filter")
        assert scroll_filter is not None

        # Check for quality_score FieldCondition
        quality_conditions = [
            cond
            for cond in scroll_filter.must
            if isinstance(cond, FieldCondition) and cond.key == "quality_score"
        ]
        assert len(quality_conditions) == 1
        quality_cond = quality_conditions[0]
        assert quality_cond.range is not None
        assert quality_cond.range.gte == 0.7

    def test_no_quality_filter_when_threshold_zero(self) -> None:
        """Test that quality filter is not added when threshold is 0."""
        # Setup
        face_client = FaceQdrantClient()
        face_client._client = MagicMock()
        face_client._client.scroll.return_value = ([], None)

        # Execute with zero threshold
        face_client.get_unlabeled_faces_with_embeddings(quality_threshold=0.0, limit=100)

        # Verify
        call_kwargs = face_client._client.scroll.call_args.kwargs
        scroll_filter = call_kwargs.get("scroll_filter")
        assert scroll_filter is not None

        # Should only have IsEmptyCondition, no quality filter
        quality_conditions = [
            cond
            for cond in scroll_filter.must
            if isinstance(cond, FieldCondition) and cond.key == "quality_score"
        ]
        assert len(quality_conditions) == 0

    def test_respects_limit_parameter(self) -> None:
        """Test that limit parameter caps the number of results returned."""
        # Setup
        face_client = FaceQdrantClient()
        face_client._client = MagicMock()

        # Create mock records with embeddings
        mock_records = []
        for i in range(150):  # More than limit
            record = MagicMock(spec=Record)
            record.payload = {
                "face_instance_id": str(uuid.uuid4()),
            }
            record.vector = [0.1] * 512  # Mock embedding
            mock_records.append(record)

        # Return records in batches
        face_client._client.scroll.side_effect = [
            (mock_records[:100], "offset1"),  # First batch
            (mock_records[100:], None),  # Second batch, no more
        ]

        # Execute with limit
        result = face_client.get_unlabeled_faces_with_embeddings(quality_threshold=0.0, limit=120)

        # Verify
        assert len(result) == 120, "Should return exactly 120 results despite 150 available"

    def test_extracts_face_instance_id_and_embedding(self) -> None:
        """Test that face_instance_id and embedding are correctly extracted."""
        # Setup
        face_client = FaceQdrantClient()
        face_client._client = MagicMock()

        face_id = uuid.uuid4()
        embedding = [0.5] * 512

        # Create mock record
        record = MagicMock(spec=Record)
        record.payload = {"face_instance_id": str(face_id)}
        record.vector = embedding

        face_client._client.scroll.return_value = ([record], None)

        # Execute
        result = face_client.get_unlabeled_faces_with_embeddings(quality_threshold=0.0, limit=10)

        # Verify
        assert len(result) == 1
        returned_id, returned_embedding = result[0]
        assert returned_id == face_id
        assert returned_embedding == embedding


class TestEnsureIsAssignedIndex:
    """Test is_assigned index creation."""

    def test_creates_payload_index(self) -> None:
        """Test that ensure_is_assigned_index creates BOOL payload index."""
        # Setup
        face_client = FaceQdrantClient()
        face_client._client = MagicMock()

        # Execute
        face_client.ensure_is_assigned_index()

        # Verify
        assert face_client._client.create_payload_index.called
        call_kwargs = face_client._client.create_payload_index.call_args.kwargs
        assert call_kwargs["field_name"] == "is_assigned"
        from qdrant_client.models import PayloadSchemaType

        assert call_kwargs["field_schema"] == PayloadSchemaType.BOOL

    def test_handles_existing_index_gracefully(self) -> None:
        """Test that existing index doesn't cause failure."""
        # Setup
        face_client = FaceQdrantClient()
        face_client._client = MagicMock()
        face_client._client.create_payload_index.side_effect = Exception("Index exists")

        # Execute - should not raise
        face_client.ensure_is_assigned_index()

        # Verify - should have attempted to create
        assert face_client._client.create_payload_index.called


class TestBackfillIsAssigned:
    """Test is_assigned backfill logic."""

    def test_categorizes_assigned_and_unassigned_faces(self) -> None:
        """Test that backfill correctly categorizes faces by person_id presence."""
        # Setup
        face_client = FaceQdrantClient()
        mock_client = MagicMock()
        face_client._client = mock_client

        # Create mock records - mix of assigned and unassigned
        # Use SimpleNamespace to create objects with attribute access
        assigned_id = "assigned-1"
        unassigned_id = "unassigned-1"

        assigned_record = SimpleNamespace(
            id=assigned_id,
            payload={"person_id": str(uuid.uuid4())}
        )

        unassigned_record = SimpleNamespace(
            id=unassigned_id,
            payload={}  # No person_id
        )

        # Return records in one batch
        mock_client.scroll.return_value = (
            [assigned_record, unassigned_record],
            None,
        )

        # Execute
        updated = face_client.backfill_is_assigned(batch_size=100)

        # Verify
        assert updated == 2

        # Check set_payload was called twice (once for assigned, once for unassigned)
        assert mock_client.set_payload.call_count == 2

        # Verify assigned faces got is_assigned=True
        assigned_call = [
            call for call in mock_client.set_payload.call_args_list if assigned_id in call.kwargs["points"]
        ]
        assert len(assigned_call) == 1
        assert assigned_call[0].kwargs["payload"] == {"is_assigned": True}

        # Verify unassigned faces got is_assigned=False
        unassigned_call = [
            call
            for call in mock_client.set_payload.call_args_list
            if unassigned_id in call.kwargs["points"]
        ]
        assert len(unassigned_call) == 1
        assert unassigned_call[0].kwargs["payload"] == {"is_assigned": False}

    def test_processes_in_batches(self) -> None:
        """Test that backfill respects batch_size parameter."""
        # Setup
        face_client = FaceQdrantClient()
        mock_client = MagicMock()
        face_client._client = mock_client

        # Create multiple batches of records with SimpleNamespace
        batch1 = [
            SimpleNamespace(id=f"id-{i}", payload={})
            for i in range(10)
        ]

        batch2 = [
            SimpleNamespace(id=f"id-{i + 10}", payload={})
            for i in range(10)
        ]

        mock_client.scroll.side_effect = [
            (batch1, "offset1"),
            (batch2, None),
        ]

        # Execute
        updated = face_client.backfill_is_assigned(batch_size=10)

        # Verify
        assert updated == 20
        # scroll should be called twice (once per batch)
        assert mock_client.scroll.call_count == 2

    def test_returns_count_of_updated_points(self) -> None:
        """Test that backfill returns correct count of updated points."""
        # Setup
        face_client = FaceQdrantClient()
        mock_client = MagicMock()
        face_client._client = mock_client

        # Create records with SimpleNamespace
        records = [
            SimpleNamespace(id=f"id-{i}", payload={})
            for i in range(5)
        ]

        mock_client.scroll.return_value = (records, None)

        # Execute
        updated = face_client.backfill_is_assigned(batch_size=100)

        # Verify
        assert updated == 5


class TestUpdatePersonIdsWithIsAssigned:
    """Test that update_person_ids maintains is_assigned sentinel."""

    def test_sets_is_assigned_true_when_assigning_person(self) -> None:
        """Test that assigning a person_id also sets is_assigned=True."""
        # Setup
        face_client = FaceQdrantClient()
        face_client._client = MagicMock()

        point_ids = [uuid.uuid4()]
        person_id = uuid.uuid4()

        # Execute
        face_client.update_person_ids(point_ids, person_id)

        # Verify
        assert face_client._client.set_payload.called
        call_kwargs = face_client._client.set_payload.call_args.kwargs
        payload = call_kwargs["payload"]
        assert payload["person_id"] == str(person_id)
        assert payload["is_assigned"] is True

    def test_sets_is_assigned_false_when_clearing_person(self) -> None:
        """Test that clearing person_id also sets is_assigned=False."""
        # Setup
        face_client = FaceQdrantClient()
        face_client._client = MagicMock()

        point_ids = [uuid.uuid4()]

        # Execute
        face_client.update_person_ids(point_ids, None)

        # Verify
        # Should call delete_payload for person_id
        assert face_client._client.delete_payload.called
        delete_kwargs = face_client._client.delete_payload.call_args.kwargs
        assert "person_id" in delete_kwargs["keys"]

        # Should also call set_payload for is_assigned=False
        assert face_client._client.set_payload.called
        set_kwargs = face_client._client.set_payload.call_args.kwargs
        assert set_kwargs["payload"] == {"is_assigned": False}


class TestUpsertFaceWithIsAssigned:
    """Test that upsert_face sets is_assigned based on person_id."""

    def test_sets_is_assigned_true_when_person_id_present(self) -> None:
        """Test that upsert_face sets is_assigned=True when person_id is provided."""
        # Setup
        face_client = FaceQdrantClient()
        face_client._client = MagicMock()

        # Mock ensure_collection to avoid actual collection operations
        face_client.ensure_collection = MagicMock()

        # Execute
        face_client.upsert_face(
            point_id=uuid.uuid4(),
            embedding=[0.1] * 512,
            asset_id=uuid.uuid4(),
            face_instance_id=uuid.uuid4(),
            detection_confidence=0.9,
            person_id=uuid.uuid4(),  # person_id provided
        )

        # Verify
        assert face_client._client.upsert.called
        points = face_client._client.upsert.call_args.kwargs["points"]
        assert len(points) == 1
        payload = points[0].payload
        assert payload["is_assigned"] is True

    def test_sets_is_assigned_false_when_person_id_absent(self) -> None:
        """Test that upsert_face sets is_assigned=False when person_id is None."""
        # Setup
        face_client = FaceQdrantClient()
        face_client._client = MagicMock()
        face_client.ensure_collection = MagicMock()

        # Execute
        face_client.upsert_face(
            point_id=uuid.uuid4(),
            embedding=[0.1] * 512,
            asset_id=uuid.uuid4(),
            face_instance_id=uuid.uuid4(),
            detection_confidence=0.9,
            person_id=None,  # No person_id
        )

        # Verify
        points = face_client._client.upsert.call_args.kwargs["points"]
        payload = points[0].payload
        assert payload["is_assigned"] is False


class TestUpsertFacesBatchWithIsAssigned:
    """Test that upsert_faces_batch sets is_assigned based on person_id."""

    def test_sets_is_assigned_correctly_in_batch(self) -> None:
        """Test that batch upsert sets is_assigned based on person_id presence."""
        # Setup
        face_client = FaceQdrantClient()
        face_client._client = MagicMock()
        face_client.ensure_collection = MagicMock()

        # Create batch with mixed assigned/unassigned
        faces = [
            {
                "point_id": uuid.uuid4(),
                "embedding": [0.1] * 512,
                "asset_id": uuid.uuid4(),
                "face_instance_id": uuid.uuid4(),
                "detection_confidence": 0.9,
                "person_id": uuid.uuid4(),  # Assigned
            },
            {
                "point_id": uuid.uuid4(),
                "embedding": [0.1] * 512,
                "asset_id": uuid.uuid4(),
                "face_instance_id": uuid.uuid4(),
                "detection_confidence": 0.9,
                "person_id": None,  # Unassigned
            },
        ]

        # Execute
        face_client.upsert_faces_batch(faces)

        # Verify
        points = face_client._client.upsert.call_args.kwargs["points"]
        assert len(points) == 2
        assert points[0].payload["is_assigned"] is True
        assert points[1].payload["is_assigned"] is False
