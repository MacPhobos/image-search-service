"""Tests for Qdrant bootstrap script."""

from unittest.mock import MagicMock, Mock, patch

import pytest
from qdrant_client.models import (
    CollectionDescription,
    CollectionInfo,
    CollectionsResponse,
    Distance,
)
from typer.testing import CliRunner

from image_search_service.scripts.bootstrap_qdrant import (
    app,
    ensure_faces_collection,
    ensure_image_assets_collection,
)


@pytest.fixture
def mock_qdrant_client() -> MagicMock:
    """Create a mock Qdrant client."""
    client = MagicMock()
    client.get_collections.return_value = CollectionsResponse(collections=[])
    return client


@pytest.fixture
def mock_settings() -> Mock:
    """Create mock settings."""
    settings = Mock()
    settings.qdrant_collection = "image_assets"
    settings.embedding_dim = 768  # Image search uses 768-dim (CLIP/SigLIP)
    settings.qdrant_url = "http://localhost:6333"
    settings.qdrant_api_key = None
    return settings


def test_ensure_image_assets_collection_creates_new(mock_qdrant_client: MagicMock) -> None:
    """Test creating image_assets collection when it doesn't exist."""
    with patch("image_search_service.scripts.bootstrap_qdrant.get_settings") as mock_get_settings:
        mock_settings = Mock()
        mock_settings.qdrant_collection = "image_assets"
        mock_settings.embedding_dim = 768  # Image search uses 768-dim
        mock_get_settings.return_value = mock_settings

        # Collection doesn't exist
        mock_qdrant_client.get_collections.return_value = CollectionsResponse(collections=[])

        result = ensure_image_assets_collection(mock_qdrant_client)

        assert result is True
        mock_qdrant_client.create_collection.assert_called_once()
        call_args = mock_qdrant_client.create_collection.call_args
        assert call_args.kwargs["collection_name"] == "image_assets"
        assert call_args.kwargs["vectors_config"].size == 768
        assert call_args.kwargs["vectors_config"].distance == Distance.COSINE


def test_ensure_image_assets_collection_already_exists(mock_qdrant_client: MagicMock) -> None:
    """Test that existing collection is not recreated."""
    with patch("image_search_service.scripts.bootstrap_qdrant.get_settings") as mock_get_settings:
        mock_settings = Mock()
        mock_settings.qdrant_collection = "image_assets"
        mock_settings.embedding_dim = 768  # Image search uses 768-dim
        mock_get_settings.return_value = mock_settings

        # Collection already exists
        existing_collection = CollectionDescription(name="image_assets")
        mock_qdrant_client.get_collections.return_value = CollectionsResponse(
            collections=[existing_collection]
        )

        result = ensure_image_assets_collection(mock_qdrant_client)

        assert result is False
        mock_qdrant_client.create_collection.assert_not_called()


def test_ensure_faces_collection_creates_new(mock_qdrant_client: MagicMock) -> None:
    """Test creating faces collection with payload indexes."""
    with patch("image_search_service.scripts.bootstrap_qdrant.get_settings") as mock_get_settings:
        mock_settings = Mock()
        mock_settings.qdrant_face_collection = "test_faces"
        mock_get_settings.return_value = mock_settings

        # Collection doesn't exist
        mock_qdrant_client.get_collections.return_value = CollectionsResponse(collections=[])

        result = ensure_faces_collection(mock_qdrant_client)

        assert result is True
        mock_qdrant_client.create_collection.assert_called_once()
        call_args = mock_qdrant_client.create_collection.call_args
        assert call_args.kwargs["collection_name"] == "test_faces"
        assert call_args.kwargs["vectors_config"].size == 512
        assert call_args.kwargs["vectors_config"].distance == Distance.COSINE

        # Should create 6 payload indexes (person_id, cluster_id, is_prototype, is_assigned, asset_id, face_instance_id)
        assert mock_qdrant_client.create_payload_index.call_count == 6


def test_ensure_faces_collection_already_exists(mock_qdrant_client: MagicMock) -> None:
    """Test that existing faces collection is not recreated."""
    with patch("image_search_service.scripts.bootstrap_qdrant.get_settings") as mock_get_settings:
        mock_settings = Mock()
        mock_settings.qdrant_face_collection = "test_faces"
        mock_get_settings.return_value = mock_settings

        # Collection already exists
        existing_collection = CollectionDescription(name="test_faces")
        mock_qdrant_client.get_collections.return_value = CollectionsResponse(
            collections=[existing_collection]
        )

        result = ensure_faces_collection(mock_qdrant_client)

        assert result is False
        mock_qdrant_client.create_collection.assert_not_called()
        mock_qdrant_client.create_payload_index.assert_not_called()


def test_init_command_success() -> None:
    """Test init command creates all collections."""
    runner = CliRunner()

    with (
        patch("image_search_service.scripts.bootstrap_qdrant.get_qdrant_client") as mock_get_client,
        patch("image_search_service.scripts.bootstrap_qdrant.get_settings") as mock_get_settings,
    ):
        mock_client = MagicMock()
        mock_client.get_collections.return_value = CollectionsResponse(collections=[])
        mock_get_client.return_value = mock_client

        mock_settings = Mock()
        mock_settings.qdrant_collection = "image_assets"
        mock_settings.embedding_dim = 768  # Image search uses 768-dim
        mock_settings.qdrant_face_collection = "test_faces"
        mock_settings.qdrant_centroid_collection = "test_person_centroids"
        mock_settings.siglip_embedding_dim = 768
        mock_settings.use_siglip = False
        mock_settings.siglip_rollout_percentage = 0
        mock_settings.siglip_collection = "test_image_assets_siglip"
        mock_get_settings.return_value = mock_settings

        result = runner.invoke(app, ["init"])

        assert result.exit_code == 0
        assert "Bootstrap complete!" in result.stdout
        # Should create image_assets, faces, and person_centroids (3 collections)
        # SigLIP is skipped because use_siglip=False and rollout=0
        assert mock_client.create_collection.call_count == 3


def test_init_command_connection_failure() -> None:
    """Test init command handles connection failure."""
    runner = CliRunner()

    with patch(
        "image_search_service.scripts.bootstrap_qdrant.get_qdrant_client"
    ) as mock_get_client:
        mock_client = MagicMock()
        mock_client.get_collections.side_effect = Exception("Connection failed")
        mock_get_client.return_value = mock_client

        result = runner.invoke(app, ["init"])

        assert result.exit_code == 1
        assert "Bootstrap failed" in result.stdout


def test_verify_command_success() -> None:
    """Test verify command passes when collections are configured correctly."""
    runner = CliRunner()

    with (
        patch("image_search_service.scripts.bootstrap_qdrant.get_qdrant_client") as mock_get_client,
        patch("image_search_service.scripts.bootstrap_qdrant.get_settings") as mock_get_settings,
    ):
        mock_client = MagicMock()

        # Mock existing collections (include person_centroids)
        image_collection = CollectionDescription(name="image_assets")
        faces_collection = CollectionDescription(name="test_faces")
        centroids_collection = CollectionDescription(name="test_person_centroids")
        mock_client.get_collections.return_value = CollectionsResponse(
            collections=[image_collection, faces_collection, centroids_collection]
        )

        # Mock collection info for verification
        mock_image_info = Mock(spec=CollectionInfo)
        mock_image_info.points_count = 100
        mock_image_info.config = Mock()
        mock_image_info.config.params = Mock()
        mock_image_info.config.params.vectors = Mock()
        mock_image_info.config.params.vectors.size = 768  # Image collection uses 768-dim
        mock_image_info.config.params.vectors.distance = Distance.COSINE

        mock_faces_info = Mock(spec=CollectionInfo)
        mock_faces_info.points_count = 50
        mock_faces_info.config = Mock()
        mock_faces_info.config.params = Mock()
        mock_faces_info.config.params.vectors = Mock()
        mock_faces_info.config.params.vectors.size = 512  # Face collection stays at 512-dim
        mock_faces_info.config.params.vectors.distance = Distance.COSINE

        mock_centroids_info = Mock(spec=CollectionInfo)
        mock_centroids_info.points_count = 10
        mock_centroids_info.config = Mock()
        mock_centroids_info.config.params = Mock()
        mock_centroids_info.config.params.vectors = Mock()
        mock_centroids_info.config.params.vectors.size = 512  # Centroid collection uses 512-dim
        mock_centroids_info.config.params.vectors.distance = Distance.COSINE

        mock_client.get_collection.side_effect = [
            mock_image_info,
            mock_faces_info,
            mock_centroids_info,
        ]
        mock_get_client.return_value = mock_client

        mock_settings = Mock()
        mock_settings.qdrant_collection = "image_assets"
        mock_settings.embedding_dim = 768  # Image search uses 768-dim
        mock_settings.qdrant_face_collection = "test_faces"
        mock_settings.qdrant_centroid_collection = "test_person_centroids"
        mock_settings.use_siglip = False
        mock_settings.siglip_rollout_percentage = 0
        mock_get_settings.return_value = mock_settings

        result = runner.invoke(app, ["verify"])

        assert result.exit_code == 0
        assert "All verifications passed!" in result.stdout


def test_verify_command_collection_not_found() -> None:
    """Test verify command fails when collection is missing."""
    runner = CliRunner()

    with (
        patch("image_search_service.scripts.bootstrap_qdrant.get_qdrant_client") as mock_get_client,
        patch("image_search_service.scripts.bootstrap_qdrant.get_settings") as mock_get_settings,
    ):
        mock_client = MagicMock()
        # No collections exist
        mock_client.get_collections.return_value = CollectionsResponse(collections=[])
        mock_get_client.return_value = mock_client

        mock_settings = Mock()
        mock_settings.qdrant_collection = "image_assets"
        mock_settings.embedding_dim = 768  # Image search uses 768-dim
        mock_settings.qdrant_face_collection = "test_faces"
        mock_settings.qdrant_centroid_collection = "test_person_centroids"
        mock_settings.use_siglip = False
        mock_settings.siglip_rollout_percentage = 0
        mock_get_settings.return_value = mock_settings

        result = runner.invoke(app, ["verify"])

        assert result.exit_code == 1
        assert "not found" in result.stdout
