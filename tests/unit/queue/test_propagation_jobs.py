"""Additional tests for propagate_person_label_multiproto_job (P3 priority)."""

import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from image_search_service.db.models import FaceSuggestion, FaceSuggestionStatus
from image_search_service.queue.face_jobs import propagate_person_label_multiproto_job


class TestPropagatePersonLabelMultiprotoJobAdditional:
    """Additional tests for propagate_person_label_multiproto_job (P3 priority)."""

    def test_propagate_skip_already_assigned(
        self,
        sync_db_session,
        mock_job,
        mock_get_sync_session,
        create_person,
        create_face_instance,
        mock_face_qdrant,
    ):
        """Test that faces already assigned to a person are skipped."""
        person = create_person()

        # Create prototype
        from image_search_service.db.models import PersonPrototype, PrototypeRole

        proto_face = create_face_instance(person_id=person.id)
        prototype = PersonPrototype(
            id=uuid.uuid4(),
            person_id=person.id,
            face_instance_id=proto_face.id,
            qdrant_point_id=proto_face.qdrant_point_id,
            role=PrototypeRole.EXEMPLAR.value,
            created_at=datetime.now(UTC),
        )
        sync_db_session.add(prototype)
        sync_db_session.commit()

        # Create face already assigned to another person
        other_person = create_person(name="Other")
        assigned_face = create_face_instance(person_id=other_person.id)

        # Mock Qdrant to return the assigned face
        mock_result = MagicMock()
        mock_result.payload = {"face_instance_id": str(assigned_face.id)}
        mock_result.score = 0.85
        mock_face_qdrant.search_similar_faces = MagicMock(return_value=[mock_result])

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
            patch(
                "image_search_service.vector.face_qdrant.get_face_qdrant_client",
                return_value=mock_face_qdrant,
            ),
        ):
            result = propagate_person_label_multiproto_job(str(person.id))

        # Should skip assigned face, create no suggestions
        assert result["status"] == "completed"
        assert result["suggestions_created"] == 0

    def test_propagate_handles_no_prototypes(
        self, sync_db_session, mock_job, mock_get_sync_session, create_person
    ):
        """Test handling when person has no prototypes."""
        person = create_person()

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
        ):
            result = propagate_person_label_multiproto_job(str(person.id))

        assert result["status"] == "error"
        assert "no prototypes" in result["message"].lower()

    def test_propagate_preserves_existing(
        self,
        sync_db_session,
        mock_job,
        mock_get_sync_session,
        create_person,
        create_face_instance,
        create_suggestion,
        mock_face_qdrant,
    ):
        """Test preserve_existing parameter."""
        person = create_person()

        # Create prototype
        from image_search_service.db.models import PersonPrototype, PrototypeRole

        proto_face = create_face_instance(person_id=person.id)
        prototype = PersonPrototype(
            id=uuid.uuid4(),
            person_id=person.id,
            face_instance_id=proto_face.id,
            qdrant_point_id=proto_face.qdrant_point_id,
            role=PrototypeRole.EXEMPLAR.value,
            created_at=datetime.now(UTC),
        )
        sync_db_session.add(prototype)
        sync_db_session.commit()

        # Create existing pending suggestion
        existing_suggestion = create_suggestion(
            suggested_person_id=person.id, status=FaceSuggestionStatus.PENDING.value
        )
        existing_id = existing_suggestion.id

        # Mock Qdrant (empty results)
        mock_face_qdrant.search_similar_faces = MagicMock(return_value=[])

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
            patch(
                "image_search_service.vector.face_qdrant.get_face_qdrant_client",
                return_value=mock_face_qdrant,
            ),
        ):
            result = propagate_person_label_multiproto_job(str(person.id), preserve_existing=True)

        # With preserve_existing=True, old suggestions should NOT be expired
        assert result["expired_count"] == 0

        # Verify suggestion still pending (query fresh from DB)
        updated_suggestion = sync_db_session.get(FaceSuggestion, existing_id)
        assert updated_suggestion is not None
        assert updated_suggestion.status == FaceSuggestionStatus.PENDING.value
