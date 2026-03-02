"""Unit tests for queue/jobs.py — the core data pipeline jobs.

Tests cover the two main job functions:
- index_asset(): Generates image embeddings and upserts vectors to Qdrant
- update_asset_person_ids_job(): Syncs person_ids from DB to Qdrant payload

These are synchronous RQ worker jobs. Tests use sync_db_session (in-memory
SQLite) with real DB records and mock only external services (embedding
service, Qdrant client, sync engine).

Pattern follows tests/unit/queue/test_face_jobs_coverage.py (the gold standard).
"""

import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from image_search_service.db.models import (
    FaceInstance,
    ImageAsset,
    Person,
    PersonStatus,
    TrainingStatus,
)
from image_search_service.queue.jobs import (
    index_asset,
    update_asset_person_ids_job,
)
from tests.constants import CLIP_EMBEDDING_DIM

# ============ Fixtures ============


@pytest.fixture
def create_image_asset(sync_db_session):
    """Factory for creating ImageAsset records in the sync test DB."""

    def _create(
        path: str | None = None,
        training_status: str = TrainingStatus.PENDING.value,
        created_at: datetime | None = None,
    ) -> ImageAsset:
        if path is None:
            path = f"/test/images/photo_{uuid.uuid4().hex[:8]}.jpg"

        asset = ImageAsset(
            path=path,
            training_status=training_status,
            width=640,
            height=480,
            file_size=102400,
            mime_type="image/jpeg",
        )
        if created_at is not None:
            asset.created_at = created_at

        sync_db_session.add(asset)
        sync_db_session.commit()
        sync_db_session.refresh(asset)
        return asset

    return _create


@pytest.fixture
def create_person(sync_db_session):
    """Factory for creating Person records."""
    _counter = [0]

    def _create(name: str | None = None) -> Person:
        if name is None:
            _counter[0] += 1
            name = f"Test Person {_counter[0]}"

        person = Person(
            id=uuid.uuid4(),
            name=name,
            status=PersonStatus.ACTIVE.value,
        )
        sync_db_session.add(person)
        sync_db_session.commit()
        sync_db_session.refresh(person)
        return person

    return _create


@pytest.fixture
def create_face_instance(sync_db_session):
    """Factory for creating FaceInstance records.

    Each invocation generates unique bbox coordinates to satisfy the
    (asset_id, bbox_x, bbox_y, bbox_w, bbox_h) unique constraint.
    """
    _counter = [0]

    def _create(
        asset_id: int,
        person_id: uuid.UUID | None = None,
    ) -> FaceInstance:
        _counter[0] += 1
        face = FaceInstance(
            id=uuid.uuid4(),
            asset_id=asset_id,
            person_id=person_id,
            bbox_x=100 + _counter[0],
            bbox_y=150 + _counter[0],
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.95,
            quality_score=0.75,
            qdrant_point_id=uuid.uuid4(),
        )
        sync_db_session.add(face)
        sync_db_session.commit()
        sync_db_session.refresh(face)
        return face

    return _create


# ============ index_asset Tests ============


class TestIndexAsset:
    """Tests for the index_asset job function."""

    def test_index_asset_when_asset_not_found_then_returns_error(self, sync_db_session):
        """Asset not in DB should return error status without crashing."""
        mock_embedding_service = MagicMock()
        mock_embedding_service.embedding_dim = CLIP_EMBEDDING_DIM

        with (
            patch(
                "image_search_service.queue.jobs.get_embedding_service",
                return_value=mock_embedding_service,
            ),
            patch(
                "image_search_service.queue.jobs.ensure_collection",
            ),
            patch(
                "image_search_service.queue.jobs.get_sync_engine",
                return_value=sync_db_session.get_bind(),
            ),
        ):
            result = index_asset("999999")

        assert result["status"] == "error"
        assert "not found" in result["message"].lower()

    def test_index_asset_when_successful_then_upserts_vector_and_sets_indexed_at(
        self, sync_db_session, create_image_asset
    ):
        """Successful indexing should call upsert_vector with correct args
        and set indexed_at timestamp on the asset."""
        asset = create_image_asset(path="/test/images/sunset_beach.jpg")
        assert asset.indexed_at is None

        mock_embedding_service = MagicMock()
        mock_embedding_service.embedding_dim = CLIP_EMBEDDING_DIM
        mock_embedding_service.embed_image.return_value = [0.1] * CLIP_EMBEDDING_DIM

        mock_upsert = MagicMock()

        with (
            patch(
                "image_search_service.queue.jobs.get_embedding_service",
                return_value=mock_embedding_service,
            ),
            patch(
                "image_search_service.queue.jobs.ensure_collection",
            ),
            patch(
                "image_search_service.queue.jobs.get_sync_engine",
                return_value=sync_db_session.get_bind(),
            ),
            patch(
                "image_search_service.queue.jobs.upsert_vector",
                mock_upsert,
            ),
        ):
            result = index_asset(str(asset.id))

        # Verify success status
        assert result["status"] == "success"
        assert result["asset_id"] == str(asset.id)

        # Verify embed_image was called with the asset path
        mock_embedding_service.embed_image.assert_called_once_with("/test/images/sunset_beach.jpg")

        # Verify upsert_vector was called with correct asset_id, vector, and payload
        mock_upsert.assert_called_once()
        call_kwargs = mock_upsert.call_args[1]
        assert call_kwargs["asset_id"] == asset.id
        assert call_kwargs["vector"] == [0.1] * CLIP_EMBEDDING_DIM
        assert call_kwargs["payload"]["path"] == "/test/images/sunset_beach.jpg"

        # Verify indexed_at was set on the asset (DB state change)
        sync_db_session.refresh(asset)
        assert asset.indexed_at is not None

    def test_index_asset_when_embedding_fails_then_returns_error(
        self, sync_db_session, create_image_asset
    ):
        """Embedding failure should be caught and return error status."""
        asset = create_image_asset()

        mock_embedding_service = MagicMock()
        mock_embedding_service.embedding_dim = CLIP_EMBEDDING_DIM
        mock_embedding_service.embed_image.side_effect = RuntimeError("Model load failed")

        with (
            patch(
                "image_search_service.queue.jobs.get_embedding_service",
                return_value=mock_embedding_service,
            ),
            patch(
                "image_search_service.queue.jobs.ensure_collection",
            ),
            patch(
                "image_search_service.queue.jobs.get_sync_engine",
                return_value=sync_db_session.get_bind(),
            ),
        ):
            result = index_asset(str(asset.id))

        assert result["status"] == "error"
        assert "Model load failed" in result["message"]

        # indexed_at should NOT be set on failure
        sync_db_session.refresh(asset)
        assert asset.indexed_at is None

    def test_index_asset_when_asset_has_created_at_then_payload_includes_it(
        self, sync_db_session, create_image_asset
    ):
        """When asset has created_at, the upsert payload should include it."""
        known_time = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
        asset = create_image_asset(
            path="/test/images/timestamped.jpg",
            created_at=known_time,
        )

        mock_embedding_service = MagicMock()
        mock_embedding_service.embedding_dim = CLIP_EMBEDDING_DIM
        mock_embedding_service.embed_image.return_value = [0.5] * CLIP_EMBEDDING_DIM

        mock_upsert = MagicMock()

        with (
            patch(
                "image_search_service.queue.jobs.get_embedding_service",
                return_value=mock_embedding_service,
            ),
            patch(
                "image_search_service.queue.jobs.ensure_collection",
            ),
            patch(
                "image_search_service.queue.jobs.get_sync_engine",
                return_value=sync_db_session.get_bind(),
            ),
            patch(
                "image_search_service.queue.jobs.upsert_vector",
                mock_upsert,
            ),
        ):
            result = index_asset(str(asset.id))

        assert result["status"] == "success"

        # Verify payload contains created_at
        call_kwargs = mock_upsert.call_args[1]
        assert "created_at" in call_kwargs["payload"]
        # The value should be an ISO format string
        assert "2024-06-15" in call_kwargs["payload"]["created_at"]

    def test_index_asset_when_ensure_collection_called_then_uses_embedding_dim(
        self, sync_db_session, create_image_asset
    ):
        """ensure_collection should be called with the embedding service dimension."""
        asset = create_image_asset()

        mock_embedding_service = MagicMock()
        mock_embedding_service.embedding_dim = CLIP_EMBEDDING_DIM
        mock_embedding_service.embed_image.return_value = [0.1] * CLIP_EMBEDDING_DIM

        mock_ensure = MagicMock()

        with (
            patch(
                "image_search_service.queue.jobs.get_embedding_service",
                return_value=mock_embedding_service,
            ),
            patch(
                "image_search_service.queue.jobs.ensure_collection",
                mock_ensure,
            ),
            patch(
                "image_search_service.queue.jobs.get_sync_engine",
                return_value=sync_db_session.get_bind(),
            ),
            patch(
                "image_search_service.queue.jobs.upsert_vector",
            ),
        ):
            index_asset(str(asset.id))

        mock_ensure.assert_called_once_with(CLIP_EMBEDDING_DIM)


# ============ update_asset_person_ids_job Tests ============


class TestUpdateAssetPersonIdsJob:
    """Tests for the update_asset_person_ids_job function."""

    def test_update_person_ids_when_asset_not_found_then_returns_skipped(self, sync_db_session):
        """Missing asset should return skipped status (deleted asset guard from commit 92f45d8)."""
        with patch(
            "image_search_service.queue.jobs.get_sync_engine",
            return_value=sync_db_session.get_bind(),
        ):
            result = update_asset_person_ids_job(999999)

        assert result["status"] == "skipped"
        assert "not found" in result["message"].lower()

    def test_update_person_ids_when_asset_has_faces_then_updates_payload(
        self,
        sync_db_session,
        create_image_asset,
        create_person,
        create_face_instance,
    ):
        """Asset with assigned faces should produce correct person_ids list in Qdrant."""
        asset = create_image_asset()
        person_a = create_person(name="Alice Jobs Test")
        person_b = create_person(name="Bob Jobs Test")

        # Create face instances with person assignments
        create_face_instance(asset_id=asset.id, person_id=person_a.id)
        create_face_instance(asset_id=asset.id, person_id=person_b.id)

        mock_update_payload = MagicMock()

        with (
            patch(
                "image_search_service.queue.jobs.get_sync_engine",
                return_value=sync_db_session.get_bind(),
            ),
            patch(
                "image_search_service.queue.jobs.update_vector_payload",
                mock_update_payload,
            ),
        ):
            result = update_asset_person_ids_job(asset.id)

        assert result["status"] == "success"
        assert result["person_count"] == 2

        # Verify update_vector_payload was called with correct args
        mock_update_payload.assert_called_once()
        call_args = mock_update_payload.call_args[0]
        assert call_args[0] == asset.id

        # The person_ids payload should contain both person IDs as strings
        payload = call_args[1]
        assert "person_ids" in payload
        person_id_strs = set(payload["person_ids"])
        assert str(person_a.id) in person_id_strs
        assert str(person_b.id) in person_id_strs

    def test_update_person_ids_when_no_faces_then_sends_empty_list(
        self,
        sync_db_session,
        create_image_asset,
    ):
        """Asset with no face instances should update payload with empty person_ids."""
        asset = create_image_asset()

        mock_update_payload = MagicMock()

        with (
            patch(
                "image_search_service.queue.jobs.get_sync_engine",
                return_value=sync_db_session.get_bind(),
            ),
            patch(
                "image_search_service.queue.jobs.update_vector_payload",
                mock_update_payload,
            ),
        ):
            result = update_asset_person_ids_job(asset.id)

        assert result["status"] == "success"
        assert result["person_count"] == 0

        # Verify empty person_ids sent to Qdrant
        call_args = mock_update_payload.call_args[0]
        assert call_args[1] == {"person_ids": []}

    def test_update_person_ids_when_faces_have_null_person_id_then_excluded(
        self,
        sync_db_session,
        create_image_asset,
        create_person,
        create_face_instance,
    ):
        """Faces with NULL person_id should be excluded from the person_ids list."""
        asset = create_image_asset()
        person = create_person(name="Charlie Jobs Test")

        # One assigned face, one unassigned
        create_face_instance(asset_id=asset.id, person_id=person.id)
        create_face_instance(asset_id=asset.id, person_id=None)

        mock_update_payload = MagicMock()

        with (
            patch(
                "image_search_service.queue.jobs.get_sync_engine",
                return_value=sync_db_session.get_bind(),
            ),
            patch(
                "image_search_service.queue.jobs.update_vector_payload",
                mock_update_payload,
            ),
        ):
            result = update_asset_person_ids_job(asset.id)

        assert result["status"] == "success"
        assert result["person_count"] == 1

        call_args = mock_update_payload.call_args[0]
        person_ids = call_args[1]["person_ids"]
        assert len(person_ids) == 1
        assert str(person.id) in person_ids

    def test_update_person_ids_when_duplicate_person_then_deduplicated(
        self,
        sync_db_session,
        create_image_asset,
        create_person,
        create_face_instance,
    ):
        """Multiple faces assigned to the same person should produce a single entry
        (DISTINCT in the query)."""
        asset = create_image_asset()
        person = create_person(name="Diana Jobs Test")

        # Two faces, both assigned to same person
        create_face_instance(asset_id=asset.id, person_id=person.id)
        create_face_instance(asset_id=asset.id, person_id=person.id)

        mock_update_payload = MagicMock()

        with (
            patch(
                "image_search_service.queue.jobs.get_sync_engine",
                return_value=sync_db_session.get_bind(),
            ),
            patch(
                "image_search_service.queue.jobs.update_vector_payload",
                mock_update_payload,
            ),
        ):
            result = update_asset_person_ids_job(asset.id)

        assert result["status"] == "success"
        assert result["person_count"] == 1

        call_args = mock_update_payload.call_args[0]
        person_ids = call_args[1]["person_ids"]
        assert len(person_ids) == 1

    def test_update_person_ids_when_qdrant_fails_then_returns_skipped(
        self,
        sync_db_session,
        create_image_asset,
    ):
        """Qdrant failure should be caught and return skipped status (asset may not be indexed)."""
        asset = create_image_asset()

        with (
            patch(
                "image_search_service.queue.jobs.get_sync_engine",
                return_value=sync_db_session.get_bind(),
            ),
            patch(
                "image_search_service.queue.jobs.update_vector_payload",
                side_effect=RuntimeError("Qdrant unavailable"),
            ),
        ):
            result = update_asset_person_ids_job(asset.id)

        assert result["status"] == "skipped"
        assert "not be indexed yet" in result["message"].lower()

    def test_update_person_ids_returns_asset_id_as_string(
        self,
        sync_db_session,
        create_image_asset,
    ):
        """The returned asset_id should always be a string representation."""
        asset = create_image_asset()

        mock_update_payload = MagicMock()

        with (
            patch(
                "image_search_service.queue.jobs.get_sync_engine",
                return_value=sync_db_session.get_bind(),
            ),
            patch(
                "image_search_service.queue.jobs.update_vector_payload",
                mock_update_payload,
            ),
        ):
            result = update_asset_person_ids_job(asset.id)

        assert isinstance(result["asset_id"], str)
        assert result["asset_id"] == str(asset.id)
