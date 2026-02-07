"""Tests for validate_qdrant_collections() function."""

from unittest.mock import Mock, patch

from image_search_service.vector.qdrant import validate_qdrant_collections


class TestValidateQdrantCollections:
    """Tests for validate_qdrant_collections()."""

    @patch("image_search_service.vector.qdrant.get_qdrant_client")
    @patch("image_search_service.vector.qdrant.get_settings")
    def test_all_collections_exist(self, mock_get_settings: Mock, mock_get_client: Mock) -> None:
        """Should return empty list when all required collections exist."""
        # Setup: all collections exist
        mock_settings = Mock()
        mock_settings.qdrant_collection = "image_assets"
        mock_settings.qdrant_face_collection = "faces"
        mock_settings.qdrant_centroid_collection = "person_centroids"
        mock_settings.use_siglip = False
        mock_get_settings.return_value = mock_settings

        mock_client = Mock()
        # Create mock collections with .name attribute
        mock_col1 = Mock()
        mock_col1.name = "image_assets"
        mock_col2 = Mock()
        mock_col2.name = "faces"
        mock_col3 = Mock()
        mock_col3.name = "person_centroids"
        mock_client.get_collections.return_value.collections = [mock_col1, mock_col2, mock_col3]
        mock_get_client.return_value = mock_client

        # Execute
        missing = validate_qdrant_collections()

        # Verify
        assert missing == []

    @patch("image_search_service.vector.qdrant.get_qdrant_client")
    @patch("image_search_service.vector.qdrant.get_settings")
    def test_missing_required_collection(
        self, mock_get_settings: Mock, mock_get_client: Mock
    ) -> None:
        """Should return list of missing collections."""
        # Setup: only image_assets exists
        mock_settings = Mock()
        mock_settings.qdrant_collection = "image_assets"
        mock_settings.qdrant_face_collection = "faces"
        mock_settings.qdrant_centroid_collection = "person_centroids"
        mock_settings.use_siglip = False
        mock_get_settings.return_value = mock_settings

        mock_client = Mock()
        # Create mock collection with .name attribute
        mock_col = Mock()
        mock_col.name = "image_assets"
        mock_client.get_collections.return_value.collections = [mock_col]
        mock_get_client.return_value = mock_client

        # Execute
        missing = validate_qdrant_collections()

        # Verify
        assert "faces" in missing
        assert "person_centroids" in missing
        assert len(missing) == 2

    @patch("image_search_service.vector.qdrant.get_qdrant_client")
    @patch("image_search_service.vector.qdrant.get_settings")
    def test_siglip_required_when_enabled(
        self, mock_get_settings: Mock, mock_get_client: Mock
    ) -> None:
        """Should require SigLIP collection when feature enabled."""
        # Setup: SigLIP enabled but collection missing
        mock_settings = Mock()
        mock_settings.qdrant_collection = "image_assets"
        mock_settings.qdrant_face_collection = "faces"
        mock_settings.qdrant_centroid_collection = "person_centroids"
        mock_settings.siglip_collection = "image_assets_siglip"
        mock_settings.use_siglip = True
        mock_get_settings.return_value = mock_settings

        mock_client = Mock()
        # Create mock collections with .name attribute
        mock_col1 = Mock()
        mock_col1.name = "image_assets"
        mock_col2 = Mock()
        mock_col2.name = "faces"
        mock_col3 = Mock()
        mock_col3.name = "person_centroids"
        mock_client.get_collections.return_value.collections = [mock_col1, mock_col2, mock_col3]
        mock_get_client.return_value = mock_client

        # Execute
        missing = validate_qdrant_collections()

        # Verify
        assert "image_assets_siglip" in missing

    @patch("image_search_service.vector.qdrant.get_qdrant_client")
    @patch("image_search_service.vector.qdrant.get_settings")
    def test_siglip_not_required_when_disabled(
        self, mock_get_settings: Mock, mock_get_client: Mock
    ) -> None:
        """Should NOT require SigLIP collection when feature disabled."""
        # Setup: SigLIP disabled
        mock_settings = Mock()
        mock_settings.qdrant_collection = "image_assets"
        mock_settings.qdrant_face_collection = "faces"
        mock_settings.qdrant_centroid_collection = "person_centroids"
        mock_settings.use_siglip = False
        mock_get_settings.return_value = mock_settings

        mock_client = Mock()
        # Create mock collections with .name attribute
        mock_col1 = Mock()
        mock_col1.name = "image_assets"
        mock_col2 = Mock()
        mock_col2.name = "faces"
        mock_col3 = Mock()
        mock_col3.name = "person_centroids"
        mock_client.get_collections.return_value.collections = [mock_col1, mock_col2, mock_col3]
        mock_get_client.return_value = mock_client

        # Execute
        missing = validate_qdrant_collections()

        # Verify: should be empty (SigLIP not required)
        assert missing == []

    @patch("image_search_service.vector.qdrant.get_qdrant_client")
    @patch("image_search_service.vector.qdrant.get_settings")
    def test_qdrant_connection_failure(
        self, mock_get_settings: Mock, mock_get_client: Mock
    ) -> None:
        """Should return all collections as missing on connection failure."""
        # Setup: connection fails
        mock_settings = Mock()
        mock_settings.qdrant_collection = "image_assets"
        mock_settings.qdrant_face_collection = "faces"
        mock_settings.qdrant_centroid_collection = "person_centroids"
        mock_settings.use_siglip = False
        mock_get_settings.return_value = mock_settings

        mock_client = Mock()
        mock_client.get_collections.side_effect = Exception("Connection refused")
        mock_get_client.return_value = mock_client

        # Execute
        missing = validate_qdrant_collections()

        # Verify: treats all as missing
        assert len(missing) >= 3  # At least the 3 required collections
        assert "image_assets" in missing
        assert "faces" in missing
        assert "person_centroids" in missing
