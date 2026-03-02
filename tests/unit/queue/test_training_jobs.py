"""Tests for train_person_matching_job function (P2 priority)."""

from unittest.mock import MagicMock, patch

from image_search_service.queue.face_jobs import train_person_matching_job


class TestTrainPersonMatchingJob:
    """Tests for train_person_matching_job function (P2 priority)."""

    def test_train_person_matching_basic(self, sync_db_session, mock_job, mock_get_sync_session):
        """Test basic training job execution."""
        mock_trainer = MagicMock()
        mock_trainer.fine_tune_for_person_clustering.return_value = {
            "status": "completed",
            "persons_used": 5,
            "total_triplets": 150,
            "final_loss": 0.15,
        }

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
            patch("image_search_service.faces.trainer.get_face_trainer", return_value=mock_trainer),
        ):
            result = train_person_matching_job()

        assert result["status"] == "completed"
        assert result["persons_used"] == 5
        assert result["final_loss"] == 0.15
        mock_trainer.fine_tune_for_person_clustering.assert_called_once()

    def test_train_person_matching_custom_params(
        self, sync_db_session, mock_job, mock_get_sync_session
    ):
        """Test training with custom parameters."""
        mock_trainer = MagicMock()
        mock_trainer.fine_tune_for_person_clustering.return_value = {
            "status": "completed",
            "persons_used": 0,
            "total_triplets": 0,
            "final_loss": 0.0,
        }

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
            patch("image_search_service.faces.trainer.get_face_trainer", return_value=mock_trainer),
        ):
            result = train_person_matching_job(
                epochs=50, margin=0.3, batch_size=64, learning_rate=0.001, min_faces_per_person=10
            )

        assert result["status"] == "completed"
        # Verify custom params passed through to fine_tune_for_person_clustering
        call_kwargs = mock_trainer.fine_tune_for_person_clustering.call_args[1]
        assert call_kwargs["min_faces_per_person"] == 10
