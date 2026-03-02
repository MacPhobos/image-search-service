"""Tests for detect_faces_for_session_job function (P3 priority - partial coverage)."""

import json
import uuid
from unittest.mock import MagicMock, patch

from image_search_service.db.models import FaceDetectionSessionStatus
from image_search_service.queue.face_jobs import detect_faces_for_session_job


class TestDetectFacesForSessionJob:
    """Tests for detect_faces_for_session_job function (P3 priority - partial coverage)."""

    def test_detect_faces_session_not_found(self, sync_db_session, mock_job, mock_get_sync_session):
        """Test error handling when session doesn't exist."""
        invalid_session_id = str(uuid.uuid4())

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
        ):
            result = detect_faces_for_session_job(invalid_session_id)

        assert result["status"] == "failed"
        assert "not found" in result["error"].lower()

    def test_detect_faces_basic_flow(
        self,
        sync_db_session,
        mock_job,
        mock_get_sync_session,
        create_detection_session,
        create_image_asset,
    ):
        """Test basic face detection flow for session."""
        session = create_detection_session()
        # Create assets to process
        for _ in range(3):
            create_image_asset()

        mock_service = MagicMock()
        mock_service.process_assets_batch.return_value = {
            "processed": 3,
            "total_faces": 3,
            "errors": 0,
        }

        mock_assigner = MagicMock()
        mock_assigner.assign_new_faces.return_value = {
            "assigned": 2,
            "unassigned": 1,
        }

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
            patch("image_search_service.faces.service.get_face_service", return_value=mock_service),
            patch(
                "image_search_service.faces.assigner.get_face_assigner",
                return_value=mock_assigner,
            ),
            patch("redis.Redis") as mock_redis_class,
        ):
            # Mock Redis for progress tracking
            mock_redis = MagicMock()
            mock_redis_class.from_url.return_value = mock_redis

            result = detect_faces_for_session_job(str(session.id))

        assert result["status"] == "completed"
        assert result["total_images"] > 0
        assert result["faces_detected"] > 0

    def test_detect_faces_no_faces_detected(
        self,
        sync_db_session,
        mock_job,
        mock_get_sync_session,
        create_detection_session,
        create_image_asset,
    ):
        """Test when detection runs but finds no faces."""
        session = create_detection_session()
        create_image_asset()

        mock_service = MagicMock()
        mock_service.process_assets_batch.return_value = {
            "processed": 1,
            "total_faces": 0,  # No faces found
            "errors": 0,
        }

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
            patch("image_search_service.faces.service.get_face_service", return_value=mock_service),
            patch("redis.Redis") as mock_redis_class,
        ):
            mock_redis = MagicMock()
            mock_redis_class.from_url.return_value = mock_redis

            result = detect_faces_for_session_job(str(session.id))

        assert result["status"] == "completed"
        assert result["faces_detected"] == 0

    def test_detect_faces_error_handling(
        self,
        sync_db_session,
        mock_job,
        mock_get_sync_session,
        create_detection_session,
        create_image_asset,
    ):
        """Test error handling during batch processing."""
        session = create_detection_session()
        create_image_asset()

        mock_service = MagicMock()
        mock_service.process_assets_batch.return_value = {
            "processed": 1,
            "total_faces": 0,
            "errors": 1,
            "error_details": [{"asset_id": 1, "error": "Detection failed"}],
        }

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
            patch("image_search_service.faces.service.get_face_service", return_value=mock_service),
            patch("redis.Redis") as mock_redis_class,
        ):
            mock_redis = MagicMock()
            mock_redis_class.from_url.return_value = mock_redis

            result = detect_faces_for_session_job(str(session.id))

        assert result["status"] == "completed"
        assert result["failed_images"] == 1

    def test_detect_faces_empty_asset_list(
        self, sync_db_session, mock_job, mock_get_sync_session, create_detection_session
    ):
        """Test when session has no assets to process."""
        session = create_detection_session()

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
        ):
            result = detect_faces_for_session_job(str(session.id))

        assert result["status"] == "completed"
        assert result["total_images"] == 0

    def test_detect_faces_resume_session(
        self,
        sync_db_session,
        mock_job,
        mock_get_sync_session,
        create_detection_session,
        create_image_asset,
    ):
        """Test resuming a paused session."""
        # Create session with stored asset IDs (simulating previous run)
        session = create_detection_session()
        assets = [create_image_asset() for _ in range(5)]
        asset_ids = [a.id for a in assets]

        # Set session as if it was paused mid-processing
        session.asset_ids_json = json.dumps(asset_ids)
        session.current_asset_index = 2  # Processed first 2
        session.status = FaceDetectionSessionStatus.PENDING.value
        sync_db_session.commit()

        mock_service = MagicMock()
        mock_service.process_assets_batch.return_value = {
            "processed": 3,  # Remaining 3
            "total_faces": 3,
            "errors": 0,
        }

        with (
            patch(
                "image_search_service.queue.face_jobs.get_sync_session",
                return_value=mock_get_sync_session(),
            ),
            patch("image_search_service.queue.face_jobs.get_current_job", return_value=mock_job),
            patch("image_search_service.faces.service.get_face_service", return_value=mock_service),
            patch("redis.Redis") as mock_redis_class,
        ):
            mock_redis = MagicMock()
            mock_redis_class.from_url.return_value = mock_redis

            result = detect_faces_for_session_job(str(session.id))

        assert result["status"] == "completed"
