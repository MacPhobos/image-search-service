"""Tests for face processing service."""

import uuid
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def sync_db_session(db_session):
    """Convert async session to sync-compatible mock for face service tests.

    Note: FaceProcessingService currently uses synchronous sessions.
    This fixture provides a mock that works with the service's sync interface.
    """
    # Create a mock that looks like a sync session but delegates to async
    mock_session = MagicMock()

    # Store the async session for manual async operations
    mock_session._async_session = db_session

    return mock_session


class TestFaceProcessingService:
    """Tests for FaceProcessingService."""

    @pytest.mark.asyncio
    async def test_process_asset_no_image(self, db_session, mock_image_asset, mock_qdrant_client):
        """Test processing asset with missing image."""
        from image_search_service.faces.service import FaceProcessingService

        # Set invalid path
        mock_image_asset.path = "/nonexistent/path.jpg"
        await db_session.commit()

        # Create service with mocked session
        with patch("image_search_service.faces.service.get_face_qdrant_client") as mock_get_qdrant:
            mock_get_qdrant.return_value = mock_qdrant_client

            # Mock the DB session operations
            with patch.object(FaceProcessingService, "_resolve_asset_path") as mock_resolve:
                mock_resolve.return_value = "/nonexistent/path.jpg"

                # Create a mock session
                mock_session = MagicMock()
                service = FaceProcessingService(mock_session)

                with patch("image_search_service.faces.detector.detect_faces_from_path") as mock_detect:
                    mock_detect.return_value = []
                    faces = service.process_asset(mock_image_asset)

        assert len(faces) == 0

    @pytest.mark.asyncio
    async def test_process_asset_creates_face_instance(
        self, db_session, mock_image_asset, mock_qdrant_client, mock_detected_face
    ):
        """Test that processing creates FaceInstance records."""
        from image_search_service.db.models import FaceInstance
        from image_search_service.faces.service import FaceProcessingService

        mock_image_asset.path = "/valid/path.jpg"
        await db_session.commit()

        with patch("image_search_service.faces.service.get_face_qdrant_client") as mock_get_qdrant:
            mock_get_qdrant.return_value = mock_qdrant_client

            # Create a minimal mock session that tracks added instances
            mock_session = MagicMock()
            added_faces = []

            def mock_add(obj):
                if isinstance(obj, FaceInstance):
                    added_faces.append(obj)

            mock_session.add = mock_add
            mock_session.commit = MagicMock()
            mock_session.execute = MagicMock(return_value=MagicMock(scalar_one_or_none=lambda: None))

            service = FaceProcessingService(mock_session)

            with patch.object(service, "_resolve_asset_path") as mock_resolve:
                mock_resolve.return_value = "/valid/path.jpg"

                with patch("image_search_service.faces.detector.detect_faces_from_path") as mock_detect:
                    mock_detect.return_value = [mock_detected_face]
                    faces = service.process_asset(mock_image_asset)

        assert len(faces) == 1
        assert isinstance(faces[0], FaceInstance)
        assert faces[0].asset_id == mock_image_asset.id
        assert faces[0].bbox_x == 100
        assert faces[0].bbox_y == 150
        assert faces[0].bbox_w == 80
        assert faces[0].bbox_h == 80

    @pytest.mark.asyncio
    async def test_process_asset_stores_in_qdrant(
        self, db_session, mock_image_asset, mock_qdrant_client, mock_detected_face
    ):
        """Test that processing stores embeddings in Qdrant."""
        from image_search_service.faces.service import FaceProcessingService

        mock_image_asset.path = "/valid/path.jpg"
        await db_session.commit()

        with patch("image_search_service.faces.service.get_face_qdrant_client") as mock_get_qdrant:
            mock_get_qdrant.return_value = mock_qdrant_client

            mock_session = MagicMock()
            mock_session.add = MagicMock()
            mock_session.commit = MagicMock()
            mock_session.execute = MagicMock(return_value=MagicMock(scalar_one_or_none=lambda: None))

            service = FaceProcessingService(mock_session)

            with patch.object(service, "_resolve_asset_path") as mock_resolve:
                mock_resolve.return_value = "/valid/path.jpg"

                with patch("image_search_service.faces.detector.detect_faces_from_path") as mock_detect:
                    mock_detect.return_value = [mock_detected_face]
                    service.process_asset(mock_image_asset)

        # Verify Qdrant upsert was called
        mock_qdrant_client.upsert_faces_batch.assert_called_once()
        call_args = mock_qdrant_client.upsert_faces_batch.call_args[0][0]
        assert len(call_args) == 1
        assert "embedding" in call_args[0]
        assert "point_id" in call_args[0]

    @pytest.mark.asyncio
    async def test_process_asset_idempotent(
        self, db_session, mock_image_asset, mock_qdrant_client, mock_detected_face
    ):
        """Test that reprocessing same asset doesn't duplicate faces."""
        from image_search_service.db.models import FaceInstance
        from image_search_service.faces.service import FaceProcessingService

        # Create an existing face instance
        existing_face = FaceInstance(
            id=uuid.uuid4(),
            asset_id=mock_image_asset.id,
            bbox_x=100,
            bbox_y=150,
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.95,
            quality_score=0.75,
            qdrant_point_id=uuid.uuid4(),
        )

        with patch("image_search_service.faces.service.get_face_qdrant_client") as mock_get_qdrant:
            mock_get_qdrant.return_value = mock_qdrant_client

            mock_session = MagicMock()
            mock_session.add = MagicMock()
            mock_session.commit = MagicMock()

            # Mock _find_existing_face to return the existing face
            mock_session.execute = MagicMock(
                return_value=MagicMock(scalar_one_or_none=lambda: existing_face)
            )

            service = FaceProcessingService(mock_session)

            with patch.object(service, "_resolve_asset_path") as mock_resolve:
                mock_resolve.return_value = "/valid/path.jpg"

                with patch("image_search_service.faces.detector.detect_faces_from_path") as mock_detect:
                    mock_detect.return_value = [mock_detected_face]
                    faces = service.process_asset(mock_image_asset)

        # Should return existing face, not create new one
        assert len(faces) == 1
        assert faces[0] == existing_face
        # Verify no new faces were added
        mock_session.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_assets_batch(
        self, db_session, mock_image_asset, mock_qdrant_client, mock_detected_face
    ):
        """Test batch processing multiple assets."""
        from image_search_service.faces.service import FaceProcessingService

        mock_image_asset.path = "/valid/path.jpg"
        await db_session.commit()

        with patch("image_search_service.faces.service.get_face_qdrant_client") as mock_get_qdrant:
            mock_get_qdrant.return_value = mock_qdrant_client

            mock_session = MagicMock()
            mock_session.get = MagicMock(return_value=mock_image_asset)
            mock_session.add = MagicMock()
            mock_session.commit = MagicMock()
            mock_session.execute = MagicMock(return_value=MagicMock(scalar_one_or_none=lambda: None))

            service = FaceProcessingService(mock_session)

            with patch.object(service, "_resolve_asset_path") as mock_resolve:
                mock_resolve.return_value = "/valid/path.jpg"

                with patch("image_search_service.faces.detector.detect_faces_from_path") as mock_detect:
                    mock_detect.return_value = [mock_detected_face]
                    result = service.process_assets_batch([mock_image_asset.id])

        assert result["processed"] == 1
        assert result["total_faces"] == 1
        assert result["errors"] == 0

    @pytest.mark.asyncio
    async def test_process_assets_batch_handles_errors(
        self, db_session, mock_qdrant_client
    ):
        """Test batch processing handles errors gracefully."""
        from image_search_service.faces.service import FaceProcessingService

        with patch("image_search_service.faces.service.get_face_qdrant_client") as mock_get_qdrant:
            mock_get_qdrant.return_value = mock_qdrant_client

            mock_session = MagicMock()
            # Simulate asset not found
            mock_session.get = MagicMock(return_value=None)

            service = FaceProcessingService(mock_session)
            result = service.process_assets_batch([999])

        assert result["processed"] == 0
        assert result["errors"] == 1

    def test_resolve_asset_path_from_path_attribute(self):
        """Test path resolution from 'path' attribute."""
        from image_search_service.db.models import ImageAsset
        from image_search_service.faces.service import FaceProcessingService

        mock_asset = MagicMock(spec=ImageAsset)
        mock_asset.path = "/test/image.jpg"

        mock_session = MagicMock()
        service = FaceProcessingService(mock_session)

        path = service._resolve_asset_path(mock_asset)
        assert path == "/test/image.jpg"

    def test_resolve_asset_path_no_path(self):
        """Test path resolution when no path attribute exists."""
        from image_search_service.faces.service import FaceProcessingService

        mock_asset = MagicMock()
        # Remove all path attributes
        del mock_asset.path
        del mock_asset.file_path
        del mock_asset.source_path

        mock_session = MagicMock()
        service = FaceProcessingService(mock_session)

        path = service._resolve_asset_path(mock_asset)
        assert path is None
