"""Tests for validate_qdrant_collections() function."""

from unittest.mock import Mock

import pytest

from image_search_service.vector.qdrant import validate_qdrant_collections


def _make_mock_collection(name: str) -> Mock:
    """Create a mock collection object with a .name attribute."""
    col = Mock()
    col.name = name
    return col


@pytest.fixture()
def mock_get_settings(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Monkeypatch get_settings in the qdrant module and return the mock."""
    mock = Mock()
    monkeypatch.setattr("image_search_service.vector.qdrant.get_settings", mock)
    return mock


@pytest.fixture()
def mock_get_client(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Monkeypatch get_qdrant_client in the qdrant module and return the mock."""
    mock = Mock()
    monkeypatch.setattr("image_search_service.vector.qdrant.get_qdrant_client", mock)
    return mock


def _setup_mocks(
    mock_get_settings: Mock,
    mock_get_client: Mock,
    *,
    existing_names: list[str],
    use_siglip: bool = False,
    siglip_collection: str = "image_assets_siglip",
    connection_error: bool = False,
) -> None:
    """Configure mock settings and client for validate_qdrant_collections tests."""
    mock_settings = Mock()
    mock_settings.qdrant_collection = "image_assets"
    mock_settings.qdrant_face_collection = "faces"
    mock_settings.qdrant_centroid_collection = "person_centroids"
    mock_settings.use_siglip = use_siglip
    if use_siglip:
        mock_settings.siglip_collection = siglip_collection
    mock_get_settings.return_value = mock_settings

    mock_client = Mock()
    if connection_error:
        mock_client.get_collections.side_effect = Exception("Connection refused")
    else:
        mock_client.get_collections.return_value.collections = [
            _make_mock_collection(name) for name in existing_names
        ]
    mock_get_client.return_value = mock_client


class TestValidateQdrantCollections:
    """Tests for validate_qdrant_collections()."""

    @pytest.mark.parametrize(
        "existing_names,expected_missing",
        [
            pytest.param(
                ["image_assets", "faces", "person_centroids"],
                [],
                id="all-exist",
            ),
            pytest.param(
                ["image_assets"],
                ["faces", "person_centroids"],
                id="missing-faces-and-centroids",
            ),
            pytest.param(
                [],
                ["image_assets", "faces", "person_centroids"],
                id="none-exist",
            ),
        ],
    )
    def test_validate_collections(
        self,
        mock_get_settings: Mock,
        mock_get_client: Mock,
        existing_names: list[str],
        expected_missing: list[str],
    ) -> None:
        """Should return correct missing collections based on what exists."""
        _setup_mocks(
            mock_get_settings,
            mock_get_client,
            existing_names=existing_names,
        )

        missing = validate_qdrant_collections()

        assert sorted(missing) == sorted(expected_missing)

    def test_siglip_required_when_enabled(
        self, mock_get_settings: Mock, mock_get_client: Mock
    ) -> None:
        """Should require SigLIP collection when feature enabled."""
        _setup_mocks(
            mock_get_settings,
            mock_get_client,
            existing_names=["image_assets", "faces", "person_centroids"],
            use_siglip=True,
        )

        missing = validate_qdrant_collections()

        assert "image_assets_siglip" in missing

    def test_siglip_not_required_when_disabled(
        self, mock_get_settings: Mock, mock_get_client: Mock
    ) -> None:
        """Should NOT require SigLIP collection when feature disabled."""
        _setup_mocks(
            mock_get_settings,
            mock_get_client,
            existing_names=["image_assets", "faces", "person_centroids"],
            use_siglip=False,
        )

        missing = validate_qdrant_collections()

        assert missing == []

    def test_qdrant_connection_failure(
        self, mock_get_settings: Mock, mock_get_client: Mock
    ) -> None:
        """Should return all collections as missing on connection failure."""
        _setup_mocks(
            mock_get_settings,
            mock_get_client,
            existing_names=[],
            connection_error=True,
        )

        missing = validate_qdrant_collections()

        assert len(missing) >= 3  # At least the 3 required collections
        assert "image_assets" in missing
        assert "faces" in missing
        assert "person_centroids" in missing
