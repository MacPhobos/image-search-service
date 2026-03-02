"""Tests for compute_centroids_job function (P2 priority)."""

from unittest.mock import MagicMock, patch

from image_search_service.queue.face_jobs import compute_centroids_job


class TestComputeCentroidsJob:
    """Tests for compute_centroids_job function (P2 priority)."""

    def test_compute_centroids_basic(
        self,
        sync_db_session,
        mock_job,
        mock_get_sync_session,
        create_person,
        create_face_instance,
    ):
        """Test basic centroid computation."""
        person = create_person()
        # Create faces for person
        for _ in range(5):
            create_face_instance(person_id=person.id)

        mock_assigner = MagicMock()
        mock_assigner.compute_person_centroids.return_value = {
            "status": "completed",
            "centroids_computed": 1,
            "persons_processed": 1,
        }

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
            patch(
                "image_search_service.faces.assigner.get_face_assigner",
                return_value=mock_assigner,
            ),
        ):
            result = compute_centroids_job()

        assert result["status"] == "completed"
        assert result["centroids_computed"] == 1
        mock_assigner.compute_person_centroids.assert_called_once()

    def test_compute_centroids_insufficient_faces(
        self, sync_db_session, mock_job, mock_get_sync_session, create_person, create_face_instance
    ):
        """Test centroid computation when person has insufficient faces."""
        person = create_person()
        # Only 1 face (below typical min_faces threshold)
        create_face_instance(person_id=person.id)

        mock_assigner = MagicMock()
        mock_assigner.compute_person_centroids.return_value = {
            "status": "completed",
            "centroids_computed": 0,  # Skipped due to insufficient faces
            "persons_processed": 1,
        }

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
            patch(
                "image_search_service.faces.assigner.get_face_assigner",
                return_value=mock_assigner,
            ),
        ):
            result = compute_centroids_job()

        assert result["status"] == "completed"
        assert result["centroids_computed"] == 0

    def test_compute_centroids_updates_existing(
        self,
        sync_db_session,
        mock_job,
        mock_get_sync_session,
        create_person,
        create_face_instance,
    ):
        """Test re-computation updates existing centroid version."""
        person = create_person()
        for _ in range(5):
            create_face_instance(person_id=person.id)

        # Create existing centroid (just for simulating a re-computation scenario)
        # Note: We're just testing that the job calls the assigner correctly
        # The actual centroid creation is mocked

        mock_assigner = MagicMock()
        mock_assigner.compute_person_centroids.return_value = {
            "status": "completed",
            "centroids_computed": 1,  # Re-computed
            "persons_processed": 1,
        }

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
            patch(
                "image_search_service.faces.assigner.get_face_assigner",
                return_value=mock_assigner,
            ),
        ):
            result = compute_centroids_job()

        assert result["status"] == "completed"
        assert result["centroids_computed"] == 1
