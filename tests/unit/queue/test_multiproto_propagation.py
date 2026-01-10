"""Unit tests for multi-prototype propagation job.

Tests the propagate_person_label_multiproto_job that generates suggestions
using ALL prototypes for a person instead of a single source face.
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
    PersonPrototype,
    PersonStatus,
    PrototypeRole,
    TrainingStatus,
)
from image_search_service.queue.face_jobs import propagate_person_label_multiproto_job

# ============ Fixtures ============


@pytest.fixture
def create_person(sync_db_session):
    """Factory fixture for creating Person records (synchronous)."""

    def _create(name: str = "Test Person", status: PersonStatus = PersonStatus.ACTIVE):
        person = Person(
            id=uuid.uuid4(),
            name=name,
            status=status.value,
        )
        sync_db_session.add(person)
        sync_db_session.commit()
        sync_db_session.refresh(person)
        return person

    return _create


@pytest.fixture
def create_image_asset(sync_db_session):
    """Factory fixture for creating ImageAsset records (synchronous)."""

    def _create(path: str | None = None):
        if path is None:
            path = f"/test/images/photo_{uuid.uuid4().hex[:8]}.jpg"

        asset = ImageAsset(
            path=path,
            training_status=TrainingStatus.PENDING.value,
            width=640,
            height=480,
            file_size=102400,
            mime_type="image/jpeg",
        )
        sync_db_session.add(asset)
        sync_db_session.commit()
        sync_db_session.refresh(asset)
        return asset

    return _create


@pytest.fixture
def create_face_instance(sync_db_session, create_image_asset):
    """Factory fixture for creating FaceInstance records (synchronous)."""

    def _create(
        person_id: uuid.UUID | None = None,
        quality_score: float = 0.75,
        asset_id: uuid.UUID | None = None,
        qdrant_point_id: uuid.UUID | None = None,
    ):
        # Create asset if not provided
        if asset_id is None:
            asset = create_image_asset()
            asset_id = asset.id

        # Generate qdrant_point_id if not provided
        if qdrant_point_id is None:
            qdrant_point_id = uuid.uuid4()

        face = FaceInstance(
            id=uuid.uuid4(),
            asset_id=asset_id,
            person_id=person_id,
            bbox_x=100,
            bbox_y=150,
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.95,
            quality_score=quality_score,
            qdrant_point_id=qdrant_point_id,
        )
        sync_db_session.add(face)
        sync_db_session.commit()
        sync_db_session.refresh(face)
        # Make object available for inspection by adding to session
        sync_db_session.expunge(face)
        sync_db_session.add(face)
        return face

    return _create


@pytest.fixture
def create_prototype(sync_db_session):
    """Factory fixture for creating PersonPrototype records (synchronous)."""

    def _create(
        person_id: uuid.UUID,
        face_instance_id: uuid.UUID,
        role: PrototypeRole = PrototypeRole.EXEMPLAR,
        qdrant_point_id: uuid.UUID | None = None,
    ):
        if qdrant_point_id is None:
            qdrant_point_id = uuid.uuid4()

        prototype = PersonPrototype(
            id=uuid.uuid4(),
            person_id=person_id,
            face_instance_id=face_instance_id,
            qdrant_point_id=qdrant_point_id,
            role=role.value,
            created_at=datetime.now(UTC),
        )
        sync_db_session.add(prototype)
        sync_db_session.commit()
        sync_db_session.refresh(prototype)
        return prototype

    return _create


@pytest.fixture
def create_suggestion(sync_db_session, create_face_instance):
    """Factory fixture for creating FaceSuggestion records (synchronous)."""

    def _create(
        suggested_person_id: uuid.UUID,
        status: str = "pending",
        face_instance_id: uuid.UUID | None = None,
        source_face_id: uuid.UUID | None = None,
        confidence: float = 0.85,
    ):
        # Create face if not provided
        if face_instance_id is None:
            face = create_face_instance()
            face_instance_id = face.id

        # Create source face if not provided
        if source_face_id is None:
            source_face = create_face_instance()
            source_face_id = source_face.id

        suggestion = FaceSuggestion(
            face_instance_id=face_instance_id,
            suggested_person_id=suggested_person_id,
            source_face_id=source_face_id,
            confidence=confidence,
            status=status,
            created_at=datetime.now(UTC),
        )
        sync_db_session.add(suggestion)
        sync_db_session.commit()
        sync_db_session.refresh(suggestion)
        return suggestion

    return _create


@pytest.fixture
def mock_qdrant():
    """Mock Qdrant client for testing."""
    mock = MagicMock()
    mock.get_embedding_by_point_id = MagicMock(return_value=[0.1] * 512)
    mock.search_similar_faces = MagicMock(return_value=[])
    return mock


# ============ Tests ============


class TestMultiPrototypePropagation:
    """Tests for propagate_person_label_multiproto_job."""

    def test_returns_error_when_person_not_found(self, sync_db_session):
        """Returns error when person doesn't exist."""
        # Mock dependencies
        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=sync_db_session,
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=None),
        ):
            result = propagate_person_label_multiproto_job(str(uuid.uuid4()))

        assert result["status"] == "error"
        assert "not found" in result["message"].lower()

    def test_returns_error_when_no_prototypes(
        self, sync_db_session, create_person, mock_qdrant
    ):
        """Returns error when person has no prototypes."""
        person = create_person(name="Alice")

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=sync_db_session,
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=None),
            patch(
                "image_search_service.vector.face_qdrant.get_face_qdrant_client",
                return_value=mock_qdrant,
            ),
        ):
            result = propagate_person_label_multiproto_job(str(person.id))

        assert result["status"] == "error"
        assert "no prototypes" in result["message"].lower()


    def test_creates_suggestions_with_single_prototype(
        self, sync_db_session, create_person, create_face_instance, create_prototype, mock_qdrant
    ):
        """Creates suggestions using a single prototype."""
        # Setup: Person with one prototype
        person = create_person(name="Bob")
        proto_face = create_face_instance(person_id=person.id, quality_score=0.9)
        create_prototype(person_id=person.id, face_instance_id=proto_face.id)

        # Create unassigned face that will be returned by Qdrant
        candidate_face = create_face_instance()

        # Mock Qdrant to return the candidate face
        mock_result = MagicMock()
        mock_result.payload = {"face_id": str(candidate_face.id)}
        mock_result.score = 0.85
        mock_qdrant.search_similar_faces = MagicMock(return_value=[mock_result])

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session", return_value=sync_db_session
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=None),
            patch(
                "image_search_service.vector.face_qdrant.get_face_qdrant_client",
                return_value=mock_qdrant,
            ),
        ):
            result = propagate_person_label_multiproto_job(str(person.id))

        assert result["status"] == "completed"
        assert result["suggestions_created"] == 1
        assert result["prototypes_used"] == 1
        assert result["candidates_evaluated"] == 1


    def test_aggregates_scores_from_multiple_prototypes(
        self, sync_db_session, create_person, create_face_instance, create_prototype, mock_qdrant
    ):
        """Aggregates scores when multiple prototypes match same face."""
        # Setup: Person with 3 prototypes of different quality
        person = create_person(name="Carol")

        proto1 = create_face_instance(person_id=person.id, quality_score=0.7)
        proto2 = create_face_instance(person_id=person.id, quality_score=0.9)
        proto3 = create_face_instance(person_id=person.id, quality_score=0.8)

        create_prototype(person_id=person.id, face_instance_id=proto1.id)
        create_prototype(person_id=person.id, face_instance_id=proto2.id)
        create_prototype(person_id=person.id, face_instance_id=proto3.id)

        # Unassigned face that matches all prototypes
        candidate = create_face_instance()
        candidate_id = candidate.id  # Capture ID immediately

        # Mock: all 3 prototypes find the same face with different scores
        mock_results = [
            [MagicMock(payload={"face_id": str(candidate_id)}, score=0.75)],  # proto1
            [MagicMock(payload={"face_id": str(candidate_id)}, score=0.92)],  # proto2 (highest)
            [MagicMock(payload={"face_id": str(candidate_id)}, score=0.80)],  # proto3
        ]
        mock_qdrant.search_similar_faces = MagicMock(side_effect=mock_results)

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session", return_value=sync_db_session
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=None),
            patch(
                "image_search_service.vector.face_qdrant.get_face_qdrant_client",
                return_value=mock_qdrant,
            ),
        ):
            result = propagate_person_label_multiproto_job(str(person.id))

        # Verify job results
        assert result["status"] == "completed"
        assert result["suggestions_created"] == 1
        assert result["prototypes_used"] == 3

        # Verify suggestion created with aggregated data
        from sqlalchemy import select

        suggestion_query = select(FaceSuggestion).where(
            FaceSuggestion.face_instance_id == candidate_id
        )
        suggestion_result = sync_db_session.execute(suggestion_query)
        suggestion = suggestion_result.scalar_one_or_none()

        assert suggestion is not None
        assert suggestion.aggregate_confidence == 0.92  # MAX score
        assert suggestion.prototype_match_count == 3
        assert len(suggestion.matching_prototype_ids) == 3


    def test_uses_max_score_as_aggregate_confidence(
        self, sync_db_session, create_person, create_face_instance, create_prototype, mock_qdrant
    ):
        """Uses MAX score across prototypes as aggregate_confidence."""
        person = create_person(name="Dave")

        # Create 2 prototypes
        proto1 = create_face_instance(person_id=person.id, quality_score=0.6)
        proto2 = create_face_instance(person_id=person.id, quality_score=0.8)

        create_prototype(person_id=person.id, face_instance_id=proto1.id)
        create_prototype(person_id=person.id, face_instance_id=proto2.id)

        # Candidate face
        candidate = create_face_instance()
        candidate_id = candidate.id  # Capture ID immediately

        # Mock: proto1 has higher score (0.88) than proto2 (0.72)
        mock_results = [
            [MagicMock(payload={"face_id": str(candidate_id)}, score=0.88)],  # proto1 (MAX)
            [MagicMock(payload={"face_id": str(candidate_id)}, score=0.72)],  # proto2
        ]
        mock_qdrant.search_similar_faces = MagicMock(side_effect=mock_results)

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session", return_value=sync_db_session
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=None),
            patch(
                "image_search_service.vector.face_qdrant.get_face_qdrant_client",
                return_value=mock_qdrant,
            ),
        ):
            result = propagate_person_label_multiproto_job(str(person.id))

        # Verify aggregate_confidence is MAX
        from sqlalchemy import select

        suggestion_query = select(FaceSuggestion).where(
            FaceSuggestion.face_instance_id == candidate_id
        )
        suggestion_result = sync_db_session.execute(suggestion_query)
        suggestion = suggestion_result.scalar_one_or_none()

        assert suggestion is not None
        assert suggestion.aggregate_confidence == 0.88  # MAX of 0.88 and 0.72


    def test_selects_best_quality_prototype_as_source(
        self, sync_db_session, create_person, create_face_instance, create_prototype, mock_qdrant
    ):
        """Selects highest quality prototype as source_face_id."""
        person = create_person(name="Eve")

        # Create 3 prototypes with different quality
        proto_low = create_face_instance(person_id=person.id, quality_score=0.5)
        proto_high = create_face_instance(person_id=person.id, quality_score=0.95)
        proto_mid = create_face_instance(person_id=person.id, quality_score=0.7)

        create_prototype(person_id=person.id, face_instance_id=proto_low.id)
        create_prototype(person_id=person.id, face_instance_id=proto_high.id)
        create_prototype(person_id=person.id, face_instance_id=proto_mid.id)

        # Candidate that matches all
        candidate = create_face_instance()
        candidate_id = candidate.id  # Capture ID immediately
        proto_high_id = proto_high.id  # Capture ID

        # All prototypes find the candidate (order shouldn't matter)
        mock_results = [
            [MagicMock(payload={"face_id": str(candidate_id)}, score=0.75)],  # low
            [MagicMock(payload={"face_id": str(candidate_id)}, score=0.82)],  # high
            [MagicMock(payload={"face_id": str(candidate_id)}, score=0.78)],  # mid
        ]
        mock_qdrant.search_similar_faces = MagicMock(side_effect=mock_results)

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session", return_value=sync_db_session
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=None),
            patch(
                "image_search_service.vector.face_qdrant.get_face_qdrant_client",
                return_value=mock_qdrant,
            ),
        ):
            result = propagate_person_label_multiproto_job(str(person.id))

        # Verify source_face_id is the highest quality prototype
        from sqlalchemy import select

        suggestion_query = select(FaceSuggestion).where(
            FaceSuggestion.face_instance_id == candidate_id
        )
        suggestion_result = sync_db_session.execute(suggestion_query)
        suggestion = suggestion_result.scalar_one_or_none()

        assert suggestion is not None
        assert suggestion.source_face_id == proto_high_id  # Highest quality (0.95)


    def test_expires_old_pending_suggestions(
        self,
        sync_db_session,
        create_person,
        create_face_instance,
        create_prototype,
        create_suggestion,
        mock_qdrant,
    ):
        """Expires old pending suggestions before creating new ones."""
        person = create_person(name="Frank")

        # Create prototype
        proto_face = create_face_instance(person_id=person.id, quality_score=0.85)
        create_prototype(person_id=person.id, face_instance_id=proto_face.id)

        # Create existing pending suggestions
        old_suggestion1 = create_suggestion(
            suggested_person_id=person.id, status=FaceSuggestionStatus.PENDING.value
        )
        old_suggestion2 = create_suggestion(
            suggested_person_id=person.id, status=FaceSuggestionStatus.PENDING.value
        )
        # Capture IDs before job execution
        old_suggestion1_id = old_suggestion1.id
        old_suggestion2_id = old_suggestion2.id

        # Mock Qdrant to return empty (no new suggestions)
        mock_qdrant.search_similar_faces = MagicMock(return_value=[])

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session", return_value=sync_db_session
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=None),
            patch(
                "image_search_service.vector.face_qdrant.get_face_qdrant_client",
                return_value=mock_qdrant,
            ),
        ):
            result = propagate_person_label_multiproto_job(str(person.id))

        # Verify old suggestions were expired
        assert result["expired_count"] == 2

        # Query fresh from DB

        refreshed1 = sync_db_session.get(FaceSuggestion, old_suggestion1_id)
        refreshed2 = sync_db_session.get(FaceSuggestion, old_suggestion2_id)

        assert refreshed1 is not None
        assert refreshed2 is not None
        assert refreshed1.status == FaceSuggestionStatus.EXPIRED.value
        assert refreshed2.status == FaceSuggestionStatus.EXPIRED.value
        assert refreshed1.reviewed_at is not None
        assert refreshed2.reviewed_at is not None


    def test_skips_faces_already_assigned_to_person(
        self, sync_db_session, create_person, create_face_instance, create_prototype, mock_qdrant
    ):
        """Skips faces that are already assigned to any person."""
        person = create_person(name="Grace")

        # Create prototype
        proto_face = create_face_instance(person_id=person.id, quality_score=0.8)
        create_prototype(person_id=person.id, face_instance_id=proto_face.id)

        # Create face already assigned to a person
        other_person = create_person(name="Other")
        assigned_face = create_face_instance(person_id=other_person.id)

        # Create unassigned face
        unassigned_face = create_face_instance()
        unassigned_face_id = unassigned_face.id  # Capture ID immediately

        # Mock Qdrant to return both faces
        mock_results = [
            MagicMock(payload={"face_id": str(assigned_face.id)}, score=0.90),
            MagicMock(payload={"face_id": str(unassigned_face_id)}, score=0.85),
        ]
        mock_qdrant.search_similar_faces = MagicMock(return_value=mock_results)

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session", return_value=sync_db_session
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=None),
            patch(
                "image_search_service.vector.face_qdrant.get_face_qdrant_client",
                return_value=mock_qdrant,
            ),
        ):
            result = propagate_person_label_multiproto_job(str(person.id))

        # Should only create suggestion for unassigned face
        assert result["suggestions_created"] == 1

        # Verify suggestion is for unassigned face only
        from sqlalchemy import select

        all_suggestions = sync_db_session.execute(select(FaceSuggestion))
        suggestions = all_suggestions.scalars().all()

        assert len(suggestions) == 1
        assert suggestions[0].face_instance_id == unassigned_face_id


    def test_respects_max_suggestions_limit(
        self, sync_db_session, create_person, create_face_instance, create_prototype, mock_qdrant
    ):
        """Creates at most max_suggestions suggestions."""
        person = create_person(name="Hank")

        # Create prototype
        proto_face = create_face_instance(person_id=person.id, quality_score=0.85)
        create_prototype(person_id=person.id, face_instance_id=proto_face.id)

        # Create many unassigned faces
        candidate_faces = [create_face_instance() for _ in range(10)]

        # Mock Qdrant to return all 10 faces
        mock_results = [
            MagicMock(payload={"face_id": str(face.id)}, score=0.8 + i * 0.01)
            for i, face in enumerate(candidate_faces)
        ]
        mock_qdrant.search_similar_faces = MagicMock(return_value=mock_results)

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session", return_value=sync_db_session
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=None),
            patch(
                "image_search_service.vector.face_qdrant.get_face_qdrant_client",
                return_value=mock_qdrant,
            ),
        ):
            result = propagate_person_label_multiproto_job(
                str(person.id), max_suggestions=5  # Limit to 5
            )

        # Should respect max_suggestions limit
        assert result["suggestions_created"] == 5
        assert result["candidates_evaluated"] == 10  # All were evaluated


    def test_respects_min_confidence_threshold(
        self, sync_db_session, create_person, create_face_instance, create_prototype, mock_qdrant
    ):
        """Only includes suggestions above min_confidence threshold."""
        person = create_person(name="Ivy")

        # Create prototype
        proto_face = create_face_instance(person_id=person.id, quality_score=0.8)
        create_prototype(person_id=person.id, face_instance_id=proto_face.id)

        # Create candidate faces
        high_conf_face = create_face_instance()
        low_conf_face = create_face_instance()

        # Mock Qdrant to return both faces with different scores
        mock_results = [
            MagicMock(payload={"face_id": str(high_conf_face.id)}, score=0.85),  # Above threshold
            MagicMock(payload={"face_id": str(low_conf_face.id)}, score=0.65),  # Below threshold
        ]
        mock_qdrant.search_similar_faces = MagicMock(return_value=mock_results)

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session", return_value=sync_db_session
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=None),
            patch(
                "image_search_service.vector.face_qdrant.get_face_qdrant_client",
                return_value=mock_qdrant,
            ),
        ):
            result = propagate_person_label_multiproto_job(
                str(person.id), min_confidence=0.7  # Threshold
            )

        # Verify threshold was passed to Qdrant search
        mock_qdrant.search_similar_faces.assert_called_once()
        call_kwargs = mock_qdrant.search_similar_faces.call_args[1]
        assert call_kwargs["score_threshold"] == 0.7


    def test_handles_invalid_face_id_in_payload(
        self, sync_db_session, create_person, create_face_instance, create_prototype, mock_qdrant
    ):
        """Gracefully handles invalid face_id in Qdrant payload."""
        person = create_person(name="Jack")

        # Create prototype
        proto_face = create_face_instance(person_id=person.id, quality_score=0.8)
        create_prototype(person_id=person.id, face_instance_id=proto_face.id)

        # Mock Qdrant to return invalid payload
        mock_results = [
            MagicMock(payload={"face_id": "not-a-uuid"}, score=0.85),
            MagicMock(payload=None, score=0.80),  # No payload
            MagicMock(payload={"other_field": "value"}, score=0.75),  # Missing face_id
        ]
        mock_qdrant.search_similar_faces = MagicMock(return_value=mock_results)

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session", return_value=sync_db_session
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=None),
            patch(
                "image_search_service.vector.face_qdrant.get_face_qdrant_client",
                return_value=mock_qdrant,
            ),
        ):
            result = propagate_person_label_multiproto_job(str(person.id))

        # Should handle gracefully without creating suggestions
        assert result["status"] == "completed"
        assert result["suggestions_created"] == 0
        assert result["candidates_evaluated"] == 0


    def test_returns_detailed_metadata(
        self, sync_db_session, create_person, create_face_instance, create_prototype, mock_qdrant
    ):
        """Returns detailed metadata about the operation."""
        person = create_person(name="Kelly")

        # Create 2 prototypes
        proto1 = create_face_instance(person_id=person.id, quality_score=0.8)
        proto2 = create_face_instance(person_id=person.id, quality_score=0.9)

        create_prototype(person_id=person.id, face_instance_id=proto1.id)
        create_prototype(person_id=person.id, face_instance_id=proto2.id)

        # Create candidates
        candidate1 = create_face_instance()
        candidate2 = create_face_instance()

        # Mock: proto1 finds both, proto2 finds only candidate1
        mock_results = [
            [
                MagicMock(payload={"face_id": str(candidate1.id)}, score=0.85),
                MagicMock(payload={"face_id": str(candidate2.id)}, score=0.80),
            ],
            [MagicMock(payload={"face_id": str(candidate1.id)}, score=0.88)],
        ]
        mock_qdrant.search_similar_faces = MagicMock(side_effect=mock_results)

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session", return_value=sync_db_session
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=None),
            patch(
                "image_search_service.vector.face_qdrant.get_face_qdrant_client",
                return_value=mock_qdrant,
            ),
        ):
            result = propagate_person_label_multiproto_job(str(person.id))

        # Verify detailed metadata
        assert result["status"] == "completed"
        assert result["suggestions_created"] == 2  # Both candidates
        assert result["prototypes_used"] == 2
        assert result["candidates_evaluated"] == 2
        assert result["expired_count"] == 0
