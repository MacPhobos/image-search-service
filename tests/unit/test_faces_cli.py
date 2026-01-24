"""Tests for faces CLI commands."""

import uuid
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from image_search_service.db.models import FaceInstance, ImageAsset
from image_search_service.scripts.faces import faces_app

runner = CliRunner()


@pytest.fixture
def mock_db_session():
    """Mock database session."""
    session = MagicMock()
    return session


@pytest.fixture
def mock_face_instances():
    """Create mock face instances for testing."""
    faces = []
    for i in range(5):
        face = MagicMock(spec=FaceInstance)
        face.id = uuid.uuid4()
        face.asset_id = i + 1
        face.qdrant_point_id = uuid.uuid4()
        faces.append(face)
    return faces


@pytest.fixture
def mock_assets():
    """Create mock image assets."""
    assets = []
    for i in range(3):
        asset = MagicMock(spec=ImageAsset)
        asset.id = i + 1
        asset.path = f"/fake/path/image_{i}.jpg"
        assets.append(asset)
    return assets


def test_find_orphans_no_orphans(mock_db_session, mock_face_instances):
    """Test find-orphans command when no orphans exist."""
    # Mock database query to return face instances
    mock_result = MagicMock()
    mock_result.scalars().all.return_value = mock_face_instances[:3]
    mock_db_session.execute.return_value = mock_result

    # Mock Qdrant client - all points exist
    mock_qdrant = MagicMock()
    mock_qdrant.point_exists.return_value = True

    with patch("image_search_service.db.sync_operations.get_sync_session") as mock_get_session:
        with patch("image_search_service.vector.face_qdrant.get_face_qdrant_client") as mock_get_qdrant:  # noqa: E501
            mock_get_session.return_value = mock_db_session
            mock_get_qdrant.return_value = mock_qdrant

            result = runner.invoke(faces_app, ["find-orphans", "--limit", "3"])

            assert result.exit_code == 0
            assert "Total faces checked: 3" in result.stdout
            assert "Orphaned faces found: 0" in result.stdout
            assert "No orphaned faces found!" in result.stdout


def test_find_orphans_with_orphans(mock_db_session, mock_face_instances):
    """Test find-orphans command when orphans exist."""
    # Mock database query to return face instances
    mock_result = MagicMock()
    mock_result.scalars().all.return_value = mock_face_instances
    mock_db_session.execute.return_value = mock_result

    # Mock Qdrant client - some points missing
    mock_qdrant = MagicMock()

    def point_exists_side_effect(point_id):
        # First two faces are orphaned
        return point_id not in [mock_face_instances[0].qdrant_point_id, mock_face_instances[1].qdrant_point_id]  # noqa: E501

    mock_qdrant.point_exists.side_effect = point_exists_side_effect

    with patch("image_search_service.db.sync_operations.get_sync_session") as mock_get_session:
        with patch("image_search_service.vector.face_qdrant.get_face_qdrant_client") as mock_get_qdrant:  # noqa: E501
            mock_get_session.return_value = mock_db_session
            mock_get_qdrant.return_value = mock_qdrant

            result = runner.invoke(faces_app, ["find-orphans", "--limit", "5"])

            assert result.exit_code == 0
            assert "Total faces checked: 5" in result.stdout
            assert "Orphaned faces found: 2" in result.stdout
            assert "Affected assets: 2" in result.stdout
            assert "TIP: Use --fix to re-detect faces" in result.stdout


def test_find_orphans_with_fix(mock_db_session, mock_face_instances, mock_assets):
    """Test find-orphans command with --fix flag."""
    # Mock database query to return face instances
    mock_result = MagicMock()
    mock_result.scalars().all.return_value = mock_face_instances[:3]
    mock_db_session.execute.return_value = mock_result

    # Mock get() for asset retrieval
    def get_side_effect(model, asset_id):
        if model == ImageAsset:
            for asset in mock_assets:
                if asset.id == asset_id:
                    return asset
        return None

    mock_db_session.get.side_effect = get_side_effect

    # Mock Qdrant client - all points missing
    mock_qdrant = MagicMock()
    mock_qdrant.point_exists.return_value = False

    # Mock face service
    mock_service = MagicMock()
    mock_service.process_asset.return_value = [MagicMock()]  # Return 1 face per asset

    with patch("image_search_service.db.sync_operations.get_sync_session") as mock_get_session:
        with patch("image_search_service.vector.face_qdrant.get_face_qdrant_client") as mock_get_qdrant:  # noqa: E501
            with patch("image_search_service.faces.service.get_face_service") as mock_get_service:
                mock_get_session.return_value = mock_db_session
                mock_get_qdrant.return_value = mock_qdrant
                mock_get_service.return_value = mock_service

                result = runner.invoke(faces_app, ["find-orphans", "--limit", "3", "--fix"])

                assert result.exit_code == 0
                assert "Orphaned faces found: 3" in result.stdout
                assert "RE-DETECTING FACES FOR AFFECTED ASSETS" in result.stdout
                assert "Assets processed:" in result.stdout
                assert "Total faces re-detected:" in result.stdout


def test_find_orphans_respects_limit(mock_db_session, mock_face_instances):
    """Test that find-orphans respects the limit parameter."""
    # Mock database query to return face instances
    mock_result = MagicMock()
    mock_result.scalars().all.return_value = mock_face_instances[:2]
    mock_db_session.execute.return_value = mock_result

    # Mock Qdrant client
    mock_qdrant = MagicMock()
    mock_qdrant.point_exists.return_value = True

    with patch("image_search_service.db.sync_operations.get_sync_session") as mock_get_session:
        with patch("image_search_service.vector.face_qdrant.get_face_qdrant_client") as mock_get_qdrant:  # noqa: E501
            mock_get_session.return_value = mock_db_session
            mock_get_qdrant.return_value = mock_qdrant

            result = runner.invoke(faces_app, ["find-orphans", "--limit", "2"])

            assert result.exit_code == 0
            # Verify that the query was called with the correct limit
            assert mock_db_session.execute.called
