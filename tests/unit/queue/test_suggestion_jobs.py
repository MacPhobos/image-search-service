"""Tests for expire_old_suggestions_job function (P1 priority)."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

from image_search_service.db.models import FaceSuggestion, FaceSuggestionStatus
from image_search_service.queue.face_jobs import expire_old_suggestions_job


class TestExpireOldSuggestionsJob:
    """Tests for expire_old_suggestions_job function (P1 priority)."""

    def test_expire_old_suggestions(
        self, sync_db_session, mock_job, mock_get_sync_session, create_suggestion, create_person
    ):
        """Test expiring suggestions older than threshold."""
        # Create persons to avoid unique constraint violations
        person1 = create_person(name="Person 1")
        person2 = create_person(name="Person 2")
        person3 = create_person(name="Person 3")

        # Create old pending suggestions (31 days old)
        old_date = datetime.now(UTC) - timedelta(days=31)
        old_suggestion1 = create_suggestion(
            suggested_person_id=person1.id, status="pending", created_at=old_date
        )
        old_suggestion2 = create_suggestion(
            suggested_person_id=person2.id, status="pending", created_at=old_date
        )

        # Create recent suggestion (should not be expired)
        recent_suggestion = create_suggestion(
            suggested_person_id=person3.id, status="pending", created_at=datetime.now(UTC)
        )

        # Store IDs before job execution
        old_suggestion1_id = old_suggestion1.id
        old_suggestion2_id = old_suggestion2.id
        recent_suggestion_id = recent_suggestion.id

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
        ):
            result = expire_old_suggestions_job(days_threshold=30)

        assert result["status"] == "completed"
        assert result["expired_count"] == 2
        assert result["threshold_days"] == 30

        # Verify old suggestions were expired (query fresh from DB)
        updated_old1 = sync_db_session.get(FaceSuggestion, old_suggestion1_id)
        updated_old2 = sync_db_session.get(FaceSuggestion, old_suggestion2_id)
        updated_recent = sync_db_session.get(FaceSuggestion, recent_suggestion_id)

        assert updated_old1 is not None
        assert updated_old2 is not None
        assert updated_old1.status == FaceSuggestionStatus.EXPIRED.value
        assert updated_old2.status == FaceSuggestionStatus.EXPIRED.value
        assert updated_old1.reviewed_at is not None
        assert updated_old2.reviewed_at is not None

        # Recent suggestion should remain pending
        assert updated_recent is not None
        assert updated_recent.status == FaceSuggestionStatus.PENDING.value

    def test_expire_no_old_suggestions(self, sync_db_session, mock_job, mock_get_sync_session):
        """Test expiring when no old suggestions exist."""
        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
        ):
            result = expire_old_suggestions_job(days_threshold=30)

        assert result["status"] == "completed"
        assert result["expired_count"] == 0

    def test_expire_uses_config_default(
        self, sync_db_session, mock_job, mock_get_sync_session, create_suggestion
    ):
        """Test that function uses config value when days_threshold not provided."""
        # Create old suggestion
        old_date = datetime.now(UTC) - timedelta(days=100)
        create_suggestion(status="pending", created_at=old_date)

        # Mock SyncConfigService
        mock_config = MagicMock()
        mock_config.get_int.return_value = 90  # Config says 90 days

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
            patch(
                "image_search_service.services.config_service.SyncConfigService",
                return_value=mock_config,
            ),
        ):
            result = expire_old_suggestions_job()  # No days_threshold provided

        assert result["status"] == "completed"
        assert result["expired_count"] == 1
        mock_config.get_int.assert_called_once_with("face_suggestion_expiry_days")
