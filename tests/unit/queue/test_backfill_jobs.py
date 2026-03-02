"""Tests for backfill_faces_job function (P2 priority)."""

from unittest.mock import MagicMock, patch

from image_search_service.queue.face_jobs import backfill_faces_job


class TestBackfillFacesJob:
    """Tests for backfill_faces_job function (P2 priority)."""

    def test_backfill_faces_basic(
        self, sync_db_session, mock_job, mock_get_sync_session, create_image_asset
    ):
        """Test basic backfill processing."""
        # Create assets without faces
        for _ in range(3):
            create_image_asset()

        mock_service = MagicMock()
        mock_service.process_assets_batch.return_value = {
            "processed": 3,
            "total_faces": 5,
            "errors": 0,
            "throughput": 1.5,
        }

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
            patch("image_search_service.faces.service.get_face_service", return_value=mock_service),
        ):
            result = backfill_faces_job(limit=3, batch_size=2)

        assert result["processed"] == 3
        assert result["total_faces"] == 5
        mock_service.process_assets_batch.assert_called_once()

    def test_backfill_faces_empty(self, sync_db_session, mock_job, mock_get_sync_session):
        """Test backfill when no images need processing."""
        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
        ):
            result = backfill_faces_job()

        assert result["processed"] == 0
        assert result["status"] == "no_assets_to_process"
