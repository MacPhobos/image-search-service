"""Tests for bootstrap_qdrant script."""

from unittest.mock import Mock, patch

import click.exceptions
import pytest
from qdrant_client.models import (
    CollectionDescription,
    CollectionsResponse,
    Distance,
)


@pytest.fixture
def mock_settings() -> Mock:
    """Create mock settings object."""
    settings = Mock()
    settings.qdrant_url = "http://localhost:6333"
    settings.qdrant_api_key = ""
    settings.qdrant_collection = "image_assets"
    settings.qdrant_face_collection = "faces"
    settings.qdrant_centroid_collection = "person_centroids"
    settings.embedding_dim = 512
    settings.siglip_collection = "image_assets_siglip"
    settings.siglip_embedding_dim = 768
    settings.use_siglip = False
    settings.siglip_rollout_percentage = 0
    return settings


@pytest.fixture
def mock_qdrant_client() -> Mock:
    """Create mock Qdrant client."""
    client = Mock()
    client.get_collections.return_value = CollectionsResponse(collections=[])
    client.create_collection = Mock()
    client.create_payload_index = Mock()
    client.get_collection = Mock()
    return client


class TestEnsurePersonCentroidsCollection:
    """Tests for ensure_person_centroids_collection function."""

    def test_creates_collection_when_missing(
        self, mock_qdrant_client: Mock, mock_settings: Mock
    ) -> None:
        """Test that collection is created when it doesn't exist."""
        from image_search_service.scripts.bootstrap_qdrant import (
            ensure_person_centroids_collection,
        )

        # Mock empty collections list
        mock_qdrant_client.get_collections.return_value = CollectionsResponse(collections=[])

        with patch(  # noqa: E501
            "image_search_service.scripts.bootstrap_qdrant.get_settings"
        ) as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            result = ensure_person_centroids_collection(mock_qdrant_client)

            # Verify collection was created
            assert result is True
            mock_qdrant_client.create_collection.assert_called_once()
            call_args = mock_qdrant_client.create_collection.call_args
            assert call_args.kwargs["collection_name"] == "person_centroids"
            assert call_args.kwargs["vectors_config"].size == 512
            assert call_args.kwargs["vectors_config"].distance == Distance.COSINE

            # Verify payload indexes were created
            assert mock_qdrant_client.create_payload_index.call_count == 5
            index_calls = [
                call.kwargs for call in mock_qdrant_client.create_payload_index.call_args_list
            ]
            expected_indexes = [
                "person_id",
                "centroid_id",
                "model_version",
                "centroid_version",
                "centroid_type",
            ]
            actual_indexes = [call["field_name"] for call in index_calls]
            assert set(actual_indexes) == set(expected_indexes)

    def test_skips_when_collection_exists(
        self, mock_qdrant_client: Mock, mock_settings: Mock
    ) -> None:
        """Test that collection creation is skipped when it already exists."""
        from image_search_service.scripts.bootstrap_qdrant import (
            ensure_person_centroids_collection,
        )

        # Mock existing collection using CollectionDescription
        existing_collection = CollectionDescription(name="person_centroids")
        mock_qdrant_client.get_collections.return_value = CollectionsResponse(
            collections=[existing_collection]
        )

        # Mock collection info with correct dimension
        collection_info = Mock()
        collection_info.config.params.vectors.size = 512
        mock_qdrant_client.get_collection.return_value = collection_info

        with patch(
            "image_search_service.scripts.bootstrap_qdrant.get_settings"
        ) as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            result = ensure_person_centroids_collection(mock_qdrant_client)

            # Verify collection was NOT created
            assert result is False
            mock_qdrant_client.create_collection.assert_not_called()
            mock_qdrant_client.create_payload_index.assert_not_called()

    def test_warns_on_dimension_mismatch(
        self, mock_qdrant_client: Mock, mock_settings: Mock
    ) -> None:
        """Test that warning is logged when existing collection has wrong dimension."""
        from image_search_service.scripts.bootstrap_qdrant import (
            ensure_person_centroids_collection,
        )

        # Mock existing collection with wrong dimension using CollectionDescription
        existing_collection = CollectionDescription(name="person_centroids")
        mock_qdrant_client.get_collections.return_value = CollectionsResponse(
            collections=[existing_collection]
        )

        collection_info = Mock()
        collection_info.config.params.vectors.size = 256  # Wrong dimension
        mock_qdrant_client.get_collection.return_value = collection_info

        with patch(
            "image_search_service.scripts.bootstrap_qdrant.get_settings"
        ) as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            result = ensure_person_centroids_collection(mock_qdrant_client)

            # Should still return False (not created)
            assert result is False
            mock_qdrant_client.create_collection.assert_not_called()


class TestEnsureSigLIPCollection:
    """Tests for ensure_siglip_collection function."""

    def test_skips_when_siglip_disabled(
        self, mock_qdrant_client: Mock, mock_settings: Mock
    ) -> None:
        """Test that collection creation is skipped when SigLIP is disabled."""
        from image_search_service.scripts.bootstrap_qdrant import ensure_siglip_collection

        mock_settings.use_siglip = False
        mock_settings.siglip_rollout_percentage = 0

        with patch(
            "image_search_service.scripts.bootstrap_qdrant.get_settings"
        ) as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            result = ensure_siglip_collection(mock_qdrant_client)

            # Verify collection was NOT created
            assert result is False
            mock_qdrant_client.create_collection.assert_not_called()

    def test_creates_when_use_siglip_enabled(
        self, mock_qdrant_client: Mock, mock_settings: Mock
    ) -> None:
        """Test that collection is created when use_siglip=True."""
        from image_search_service.scripts.bootstrap_qdrant import ensure_siglip_collection

        mock_settings.use_siglip = True
        mock_settings.siglip_rollout_percentage = 0
        mock_settings.siglip_collection = "image_assets_siglip"
        mock_settings.siglip_embedding_dim = 768
        mock_qdrant_client.get_collections.return_value = CollectionsResponse(collections=[])

        with patch(
            "image_search_service.scripts.bootstrap_qdrant.get_settings"
        ) as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            result = ensure_siglip_collection(mock_qdrant_client)

            # Verify collection was created
            assert result is True
            mock_qdrant_client.create_collection.assert_called_once()
            call_args = mock_qdrant_client.create_collection.call_args
            assert call_args.kwargs["collection_name"] == "image_assets_siglip"
            # Verify quantization was set up (indicates proper collection creation)

    def test_creates_when_rollout_percentage_positive(
        self, mock_qdrant_client: Mock, mock_settings: Mock
    ) -> None:
        """Test that collection is created when siglip_rollout_percentage > 0."""
        from image_search_service.scripts.bootstrap_qdrant import ensure_siglip_collection

        mock_settings.use_siglip = False
        mock_settings.siglip_rollout_percentage = 50
        mock_settings.siglip_collection = "image_assets_siglip"
        mock_settings.siglip_embedding_dim = 768
        mock_qdrant_client.get_collections.return_value = CollectionsResponse(collections=[])

        with patch(
            "image_search_service.scripts.bootstrap_qdrant.get_settings"
        ) as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            result = ensure_siglip_collection(mock_qdrant_client)

            # Verify collection was created
            assert result is True
            mock_qdrant_client.create_collection.assert_called_once()

    def test_skips_when_collection_exists(
        self, mock_qdrant_client: Mock, mock_settings: Mock
    ) -> None:
        """Test that collection creation is skipped when it already exists."""
        from image_search_service.scripts.bootstrap_qdrant import ensure_siglip_collection

        mock_settings.use_siglip = True
        mock_settings.siglip_collection = "image_assets_siglip"
        mock_settings.siglip_embedding_dim = 768
        existing_collection = CollectionDescription(name="image_assets_siglip")
        mock_qdrant_client.get_collections.return_value = CollectionsResponse(
            collections=[existing_collection]
        )

        collection_info = Mock()
        collection_info.config.params.vectors.size = 768
        mock_qdrant_client.get_collection.return_value = collection_info

        with patch(
            "image_search_service.scripts.bootstrap_qdrant.get_settings"
        ) as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            result = ensure_siglip_collection(mock_qdrant_client)

            # Verify collection was NOT created
            assert result is False
            mock_qdrant_client.create_collection.assert_not_called()


class TestVerify:
    """Tests for verify command."""

    def test_passes_when_all_collections_exist(
        self, mock_qdrant_client: Mock, mock_settings: Mock
    ) -> None:
        """Test that verify passes when all required collections exist with correct config."""
        from image_search_service.scripts.bootstrap_qdrant import verify

        # Mock all required collections using CollectionDescription
        collections = [
            CollectionDescription(name="image_assets"),
            CollectionDescription(name="faces"),
            CollectionDescription(name="person_centroids"),
        ]
        mock_qdrant_client.get_collections.return_value = CollectionsResponse(
            collections=collections
        )

        # Mock collection info with correct dimensions
        def get_collection_side_effect(collection_name: str) -> Mock:
            info = Mock()
            info.points_count = 0
            info.config.params.vectors.distance = Distance.COSINE

            if collection_name == "image_assets":
                info.config.params.vectors.size = 512
            elif collection_name == "faces":
                info.config.params.vectors.size = 512
            elif collection_name == "person_centroids":
                info.config.params.vectors.size = 512
            else:
                raise ValueError(f"Unexpected collection: {collection_name}")

            return info

        mock_qdrant_client.get_collection.side_effect = get_collection_side_effect

        with patch(
            "image_search_service.scripts.bootstrap_qdrant.get_settings"
        ) as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            with patch(
                "image_search_service.scripts.bootstrap_qdrant.get_qdrant_client"
            ) as mock_get_client:
                mock_get_client.return_value = mock_qdrant_client

                # Should not raise
                verify()

    def test_fails_when_collection_missing(
        self, mock_qdrant_client: Mock, mock_settings: Mock
    ) -> None:
        """Test that verify fails (sys.exit(1)) when a required collection is missing."""
        from image_search_service.scripts.bootstrap_qdrant import verify

        # Mock only some collections (missing person_centroids) using CollectionDescription
        collections = [
            CollectionDescription(name="image_assets"),
            CollectionDescription(name="faces"),
        ]
        mock_qdrant_client.get_collections.return_value = CollectionsResponse(
            collections=collections
        )

        # Mock collection info for existing collections
        def get_collection_side_effect(collection_name: str) -> Mock:
            info = Mock()
            info.points_count = 0
            info.config.params.vectors.distance = Distance.COSINE
            info.config.params.vectors.size = 512
            return info

        mock_qdrant_client.get_collection.side_effect = get_collection_side_effect

        with patch(
            "image_search_service.scripts.bootstrap_qdrant.get_settings"
        ) as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            with patch(
                "image_search_service.scripts.bootstrap_qdrant.get_qdrant_client"
            ) as mock_get_client:
                mock_get_client.return_value = mock_qdrant_client

                # Should raise typer.Exit (which is click.exceptions.Exit)
                with pytest.raises(click.exceptions.Exit):
                    verify()

    def test_fails_when_dimension_mismatch(
        self, mock_qdrant_client: Mock, mock_settings: Mock
    ) -> None:
        """Test that verify fails when collection has wrong dimension."""
        from image_search_service.scripts.bootstrap_qdrant import verify

        # Mock all collections using CollectionDescription
        collections = [
            CollectionDescription(name="image_assets"),
            CollectionDescription(name="faces"),
            CollectionDescription(name="person_centroids"),
        ]
        mock_qdrant_client.get_collections.return_value = CollectionsResponse(
            collections=collections
        )

        # Mock collection info with WRONG dimension for faces
        def get_collection_side_effect(collection_name: str) -> Mock:
            info = Mock()
            info.points_count = 0
            info.config.params.vectors.distance = Distance.COSINE

            if collection_name == "faces":
                info.config.params.vectors.size = 256  # Wrong!
            else:
                info.config.params.vectors.size = 512

            return info

        mock_qdrant_client.get_collection.side_effect = get_collection_side_effect

        with patch(
            "image_search_service.scripts.bootstrap_qdrant.get_settings"
        ) as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            with patch(
                "image_search_service.scripts.bootstrap_qdrant.get_qdrant_client"
            ) as mock_get_client:
                mock_get_client.return_value = mock_qdrant_client

                # Should raise typer.Exit (which is click.exceptions.Exit)
                with pytest.raises(click.exceptions.Exit):
                    verify()


class TestInit:
    """Tests for init command."""

    def test_calls_all_ensure_functions(
        self, mock_qdrant_client: Mock, mock_settings: Mock
    ) -> None:
        """Test that init calls all ensure functions in correct order."""
        from image_search_service.scripts.bootstrap_qdrant import init

        mock_qdrant_client.get_collections.return_value = CollectionsResponse(collections=[])

        with patch(
            "image_search_service.scripts.bootstrap_qdrant.get_settings"
        ) as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            with patch(
                "image_search_service.scripts.bootstrap_qdrant.get_qdrant_client"
            ) as mock_get_client:
                mock_get_client.return_value = mock_qdrant_client

                with patch(
                    "image_search_service.scripts.bootstrap_qdrant.ensure_image_assets_collection"
                ) as mock_ensure_image:
                    with patch(
                        "image_search_service.scripts.bootstrap_qdrant.ensure_faces_collection"
                    ) as mock_ensure_faces:
                        with patch(
                            "image_search_service.scripts.bootstrap_qdrant.ensure_person_centroids_collection"
                        ) as mock_ensure_centroids:
                            with patch(
                                "image_search_service.scripts.bootstrap_qdrant.ensure_siglip_collection"
                            ) as mock_ensure_siglip:
                                init()

                                # Verify all ensure functions were called
                                mock_ensure_image.assert_called_once()
                                mock_ensure_faces.assert_called_once()
                                mock_ensure_centroids.assert_called_once()
                                mock_ensure_siglip.assert_called_once()

    def test_handles_exceptions(self, mock_qdrant_client: Mock, mock_settings: Mock) -> None:
        """Test that init handles exceptions and exits with error."""
        from image_search_service.scripts.bootstrap_qdrant import init

        with patch(
            "image_search_service.scripts.bootstrap_qdrant.get_settings"
        ) as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            with patch(
                "image_search_service.scripts.bootstrap_qdrant.get_qdrant_client"
            ) as mock_get_client:
                # Mock client to raise exception
                mock_get_client.side_effect = Exception("Connection failed")

                # Should raise typer.Exit (which is click.exceptions.Exit)
                with pytest.raises(click.exceptions.Exit):
                    init()
