"""Tests for cluster_dual_job function (P1 priority)."""

from unittest.mock import MagicMock, patch

from image_search_service.queue.face_jobs import cluster_dual_job


class TestClusterDualJob:
    """Tests for cluster_dual_job function (P1 priority)."""

    def test_cluster_dual_job_basic(
        self, sync_db_session, mock_job, mock_get_sync_session, create_face_instance
    ):
        """Test basic dual-mode clustering execution."""
        # Create some unassigned faces
        for _ in range(5):
            create_face_instance(person_id=None)

        # Mock dependencies
        mock_clusterer = MagicMock()
        mock_clusterer.cluster_all_faces.return_value = {
            "status": "completed",
            "assigned_to_people": 2,
            "unknown_clusters": 1,
            "still_unlabeled": 2,
            "total_processed": 5,
        }

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
            patch(
                "image_search_service.faces.dual_clusterer.get_dual_mode_clusterer",
                return_value=mock_clusterer,
            ),
        ):
            result = cluster_dual_job()

        assert result["total_processed"] == 5
        assert result["assigned_to_people"] == 2
        assert result["unknown_clusters"] == 1
        mock_clusterer.cluster_all_faces.assert_called_once()

    def test_cluster_dual_job_custom_params(self, sync_db_session, mock_job, mock_get_sync_session):
        """Test clustering with custom parameters."""
        mock_clusterer = MagicMock()
        mock_clusterer.cluster_all_faces.return_value = {
            "total_processed": 0,
            "assigned_to_people": 0,
            "unknown_clusters": 0,
        }

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
            patch(
                "image_search_service.faces.dual_clusterer.get_dual_mode_clusterer",
                return_value=mock_clusterer,
            ),
        ):
            result = cluster_dual_job(
                person_threshold=0.8,
                unknown_method="kmeans",
                unknown_min_size=5,
                unknown_eps=0.3,
                max_faces=1000,
            )

        assert result["total_processed"] == 0
        # Verify custom params passed through to cluster_all_faces
        call_kwargs = mock_clusterer.cluster_all_faces.call_args[1]
        assert call_kwargs["max_faces"] == 1000

    def test_cluster_dual_job_empty_db(self, sync_db_session, mock_job, mock_get_sync_session):
        """Test clustering with no faces in database."""
        mock_clusterer = MagicMock()
        mock_clusterer.cluster_all_faces.return_value = {
            "total_processed": 0,
            "assigned_to_people": 0,
            "unknown_clusters": 0,
            "still_unlabeled": 0,
        }

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
            patch(
                "image_search_service.faces.dual_clusterer.get_dual_mode_clusterer",
                return_value=mock_clusterer,
            ),
        ):
            result = cluster_dual_job()

        assert result["total_processed"] == 0
        assert result["assigned_to_people"] == 0
        assert result["unknown_clusters"] == 0
