"""Unit tests for API schemas."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from image_search_service.api.schemas import Asset, CameraMetadata, LocationMetadata


class MockImageAsset:
    """Mock ImageAsset model for testing."""

    def __init__(
        self,
        id: int = 1,
        path: str = "/test/image.jpg",
        created_at: datetime | None = None,
        indexed_at: datetime | None = None,
        taken_at: datetime | None = None,
        camera_make: str | None = None,
        camera_model: str | None = None,
        gps_latitude: float | None = None,
        gps_longitude: float | None = None,
    ):
        self.id = id
        self.path = path
        self.created_at = created_at or datetime.now(timezone.utc)
        self.indexed_at = indexed_at
        self.taken_at = taken_at
        self.camera_make = camera_make
        self.camera_model = camera_model
        self.gps_latitude = gps_latitude
        self.gps_longitude = gps_longitude


class TestLocationMetadata:
    """Tests for LocationMetadata schema."""

    def test_location_with_aliases(self):
        """Should use lat/lng aliases for JSON serialization."""
        location = LocationMetadata(latitude=37.7749, longitude=-122.4194)

        data = location.model_dump(by_alias=True)

        assert "lat" in data
        assert "lng" in data
        assert data["lat"] == 37.7749
        assert data["lng"] == -122.4194
        # snake_case should not exist
        assert "latitude" not in data
        assert "longitude" not in data

    def test_location_required_fields(self):
        """Should require both latitude and longitude."""
        from pydantic import ValidationError

        # Missing latitude
        with pytest.raises(ValidationError):
            LocationMetadata(longitude=-122.4194)

        # Missing longitude
        with pytest.raises(ValidationError):
            LocationMetadata(latitude=37.7749)


class TestCameraMetadata:
    """Tests for CameraMetadata schema."""

    def test_camera_with_both_fields(self):
        """Should serialize camera with both make and model."""
        camera = CameraMetadata(make="Canon", model="EOS 5D Mark IV")

        data = camera.model_dump(by_alias=True)

        assert data["make"] == "Canon"
        assert data["model"] == "EOS 5D Mark IV"

    def test_camera_with_make_only(self):
        """Should allow only make to be set."""
        camera = CameraMetadata(make="Canon", model=None)

        data = camera.model_dump(by_alias=True)

        assert data["make"] == "Canon"
        assert data["model"] is None

    def test_camera_with_model_only(self):
        """Should allow only model to be set."""
        camera = CameraMetadata(make=None, model="iPhone 14 Pro")

        data = camera.model_dump(by_alias=True)

        assert data["make"] is None
        assert data["model"] == "iPhone 14 Pro"

    def test_camera_empty(self):
        """Should allow both fields to be None."""
        camera = CameraMetadata(make=None, model=None)

        data = camera.model_dump(by_alias=True)

        assert data["make"] is None
        assert data["model"] is None


class TestAsset:
    """Tests for Asset schema."""

    def test_asset_basic_fields(self):
        """Should serialize basic asset fields correctly."""
        mock_asset = MockImageAsset(
            id=123,
            path="/photos/vacation/IMG_001.jpg",
            created_at=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
            indexed_at=datetime(2024, 1, 15, 12, 5, 0, tzinfo=timezone.utc),
        )

        asset = Asset.model_validate(mock_asset)

        assert asset.id == 123
        assert asset.path == "/photos/vacation/IMG_001.jpg"
        assert asset.created_at == datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        assert asset.indexed_at == datetime(2024, 1, 15, 12, 5, 0, tzinfo=timezone.utc)

    def test_asset_computed_fields(self):
        """Should generate computed fields correctly."""
        mock_asset = MockImageAsset(id=456, path="/test/photo.jpg")

        asset = Asset.model_validate(mock_asset)

        assert asset.url == "/api/v1/images/456/full"
        assert asset.thumbnail_url == "/api/v1/images/456/thumbnail"
        assert asset.filename == "photo.jpg"

    def test_asset_filename_extraction(self):
        """Should extract filename from various path formats."""
        test_cases = [
            ("/photos/vacation/IMG_001.jpg", "IMG_001.jpg"),
            ("/photos/IMG_002.JPG", "IMG_002.JPG"),
            ("/path/to/deeply/nested/photo.png", "photo.png"),
            ("simple.jpg", "simple.jpg"),
        ]

        for path, expected_filename in test_cases:
            mock_asset = MockImageAsset(path=path)
            asset = Asset.model_validate(mock_asset)
            assert asset.filename == expected_filename

    def test_asset_with_exif_taken_at(self):
        """Should include taken_at from EXIF metadata."""
        taken_at = datetime(2023, 6, 15, 14, 30, 0, tzinfo=timezone.utc)
        mock_asset = MockImageAsset(taken_at=taken_at)

        asset = Asset.model_validate(mock_asset)

        assert asset.taken_at == taken_at

    def test_asset_with_camera_metadata(self):
        """Should construct camera metadata from flat DB fields."""
        mock_asset = MockImageAsset(camera_make="Canon", camera_model="EOS 5D Mark IV")

        asset = Asset.model_validate(mock_asset)

        assert asset.camera is not None
        assert asset.camera.make == "Canon"
        assert asset.camera.model == "EOS 5D Mark IV"

    def test_asset_with_partial_camera_metadata(self):
        """Should construct camera metadata with only make."""
        mock_asset = MockImageAsset(camera_make="Apple", camera_model=None)

        asset = Asset.model_validate(mock_asset)

        assert asset.camera is not None
        assert asset.camera.make == "Apple"
        assert asset.camera.model is None

    def test_asset_without_camera_metadata(self):
        """Should have None camera when no camera fields present."""
        mock_asset = MockImageAsset(camera_make=None, camera_model=None)

        asset = Asset.model_validate(mock_asset)

        assert asset.camera is None

    def test_asset_with_location_metadata(self):
        """Should construct location metadata from GPS fields."""
        mock_asset = MockImageAsset(gps_latitude=37.7749, gps_longitude=-122.4194)

        asset = Asset.model_validate(mock_asset)

        assert asset.location is not None
        assert asset.location.latitude == 37.7749
        assert asset.location.longitude == -122.4194

    def test_asset_without_location_metadata(self):
        """Should have None location when GPS fields are None."""
        mock_asset = MockImageAsset(gps_latitude=None, gps_longitude=None)

        asset = Asset.model_validate(mock_asset)

        assert asset.location is None

    def test_asset_with_partial_location_metadata(self):
        """Should have None location when only one GPS field present."""
        # Only latitude
        mock_asset_lat = MockImageAsset(gps_latitude=37.7749, gps_longitude=None)
        asset_lat = Asset.model_validate(mock_asset_lat)
        assert asset_lat.location is None

        # Only longitude
        mock_asset_lng = MockImageAsset(gps_latitude=None, gps_longitude=-122.4194)
        asset_lng = Asset.model_validate(mock_asset_lng)
        assert asset_lng.location is None

    def test_asset_with_full_exif_metadata(self):
        """Should include all EXIF metadata fields."""
        mock_asset = MockImageAsset(
            id=789,
            path="/photos/IMG_003.jpg",
            taken_at=datetime(2023, 7, 20, 10, 15, 0, tzinfo=timezone.utc),
            camera_make="Sony",
            camera_model="Alpha 7 IV",
            gps_latitude=40.7128,
            gps_longitude=-74.0060,
        )

        asset = Asset.model_validate(mock_asset)

        assert asset.id == 789
        assert asset.taken_at == datetime(2023, 7, 20, 10, 15, 0, tzinfo=timezone.utc)
        assert asset.camera is not None
        assert asset.camera.make == "Sony"
        assert asset.camera.model == "Alpha 7 IV"
        assert asset.location is not None
        assert asset.location.latitude == 40.7128
        assert asset.location.longitude == -74.0060

    def test_asset_json_serialization_with_exif(self):
        """Should serialize EXIF metadata to JSON with camelCase."""
        mock_asset = MockImageAsset(
            id=999,
            path="/test.jpg",
            created_at=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            taken_at=datetime(2023, 12, 25, 10, 0, 0, tzinfo=timezone.utc),
            camera_make="Nikon",
            camera_model="D850",
            gps_latitude=51.5074,
            gps_longitude=-0.1278,
        )

        asset = Asset.model_validate(mock_asset)
        data = asset.model_dump(by_alias=True)

        # Check basic fields use camelCase
        assert "createdAt" in data
        assert "indexedAt" in data
        assert "takenAt" in data
        assert data["takenAt"] == datetime(2023, 12, 25, 10, 0, 0, tzinfo=timezone.utc)

        # Check camera nested object
        assert "camera" in data
        assert data["camera"]["make"] == "Nikon"
        assert data["camera"]["model"] == "D850"

        # Check location nested object uses lat/lng aliases
        assert "location" in data
        assert data["location"]["lat"] == 51.5074
        assert data["location"]["lng"] == -0.1278
        # Verify nested object uses aliases, not latitude/longitude
        assert "latitude" not in data["location"]
        assert "longitude" not in data["location"]

    def test_asset_from_dict(self):
        """Should construct Asset from dict (e.g., from JSON)."""
        data = {
            "id": 1,
            "path": "/test.jpg",
            "created_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "indexed_at": None,
            "taken_at": datetime(2023, 12, 25, tzinfo=timezone.utc),
            "camera": {"make": "Canon", "model": "EOS R5"},
            "location": {"lat": 37.7749, "lng": -122.4194},
        }

        asset = Asset.model_validate(data)

        assert asset.id == 1
        assert asset.taken_at == datetime(2023, 12, 25, tzinfo=timezone.utc)
        assert asset.camera is not None
        assert asset.camera.make == "Canon"
        assert asset.location is not None
        assert asset.location.latitude == 37.7749
