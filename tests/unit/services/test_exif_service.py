"""Test EXIF metadata extraction service."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from PIL import Image
from PIL.TiffImagePlugin import IFDRational

from image_search_service.services.exif_service import (
    EXIF_TAG_DATETIME_DIGITIZED,
    EXIF_TAG_DATETIME_ORIGINAL,
    EXIF_TAG_MAKE,
    EXIF_TAG_MODEL,
    ExifService,
    get_exif_service,
)


@pytest.fixture
def exif_service() -> ExifService:
    """Create ExifService instance for testing."""
    return ExifService()


def create_image_with_exif(
    path: Path,
    datetime_original: str | None = None,
    datetime_digitized: str | None = None,
    camera_make: str | None = None,
    camera_model: str | None = None,
    gps_latitude: tuple | None = None,
    gps_longitude: tuple | None = None,
    gps_latitude_ref: str | None = None,
    gps_longitude_ref: str | None = None,
) -> Path:
    """Create test image with EXIF data.

    Args:
        path: Path to save image
        datetime_original: EXIF DateTimeOriginal value
        datetime_digitized: EXIF DateTimeDigitized value
        camera_make: Camera manufacturer
        camera_model: Camera model
        gps_latitude: GPS latitude tuple ((degrees, 1), (minutes, 1), (seconds, 100))
        gps_longitude: GPS longitude tuple
        gps_latitude_ref: "N" or "S"
        gps_longitude_ref: "E" or "W"

    Returns:
        Path to created image
    """
    # Create simple RGB image
    img = Image.new("RGB", (100, 100), color=(128, 128, 128))

    # Build EXIF data
    exif = Image.Exif()

    if datetime_original:
        exif[EXIF_TAG_DATETIME_ORIGINAL] = datetime_original
    if datetime_digitized:
        exif[EXIF_TAG_DATETIME_DIGITIZED] = datetime_digitized
    if camera_make:
        exif[EXIF_TAG_MAKE] = camera_make
    if camera_model:
        exif[EXIF_TAG_MODEL] = camera_model

    # Add GPS data if provided
    if all([gps_latitude, gps_longitude, gps_latitude_ref, gps_longitude_ref]):
        # GPS IFD (Image File Directory) structure
        # Convert tuples to IFDRational for proper EXIF encoding
        gps_lat_rationals = tuple(IFDRational(num, den) for num, den in gps_latitude)
        gps_lon_rationals = tuple(IFDRational(num, den) for num, den in gps_longitude)

        gps_ifd = {
            1: gps_latitude_ref,  # GPSLatitudeRef
            2: gps_lat_rationals,  # GPSLatitude
            3: gps_longitude_ref,  # GPSLongitudeRef
            4: gps_lon_rationals,  # GPSLongitude
        }
        exif[34853] = gps_ifd  # GPSInfo tag

    # Save image with EXIF
    img.save(path, exif=exif)

    return path


def test_extract_exif_with_datetime_original(
    exif_service: ExifService, tmp_path: Path
) -> None:
    """Test extraction of DateTimeOriginal from EXIF."""
    image_path = create_image_with_exif(
        tmp_path / "test.jpg",
        datetime_original="2023:07:15 14:30:00"
    )

    result = exif_service.extract_exif(str(image_path))

    assert result["taken_at"] is not None
    assert result["taken_at"] == datetime(2023, 7, 15, 14, 30, 0, tzinfo=UTC)


def test_extract_exif_with_datetime_digitized(
    exif_service: ExifService, tmp_path: Path
) -> None:
    """Test fallback to DateTimeDigitized when DateTimeOriginal not present."""
    image_path = create_image_with_exif(
        tmp_path / "test.jpg",
        datetime_digitized="2023:08:20 09:15:30"
    )

    result = exif_service.extract_exif(str(image_path))

    assert result["taken_at"] is not None
    assert result["taken_at"] == datetime(2023, 8, 20, 9, 15, 30, tzinfo=UTC)


def test_extract_exif_datetime_original_takes_precedence(
    exif_service: ExifService, tmp_path: Path
) -> None:
    """Test that DateTimeOriginal takes precedence over DateTimeDigitized."""
    image_path = create_image_with_exif(
        tmp_path / "test.jpg",
        datetime_original="2023:07:15 14:30:00",
        datetime_digitized="2023:08:20 09:15:30"
    )

    result = exif_service.extract_exif(str(image_path))

    # Should use DateTimeOriginal, not DateTimeDigitized
    assert result["taken_at"] == datetime(2023, 7, 15, 14, 30, 0, tzinfo=UTC)


def test_extract_exif_no_datetime_returns_none(
    exif_service: ExifService, tmp_path: Path
) -> None:
    """Test that missing datetime returns None, not inferred date."""
    image_path = create_image_with_exif(tmp_path / "test.jpg")

    result = exif_service.extract_exif(str(image_path))

    assert result["taken_at"] is None


def test_extract_exif_camera_make_and_model(
    exif_service: ExifService, tmp_path: Path
) -> None:
    """Test extraction of camera make and model."""
    image_path = create_image_with_exif(
        tmp_path / "test.jpg",
        camera_make="Apple",
        camera_model="iPhone 14 Pro"
    )

    result = exif_service.extract_exif(str(image_path))

    assert result["camera_make"] == "Apple"
    assert result["camera_model"] == "iPhone 14 Pro"


def test_extract_exif_camera_make_trimmed_to_100_chars(
    exif_service: ExifService, tmp_path: Path
) -> None:
    """Test that camera make is limited to 100 characters."""
    long_make = "A" * 150  # 150 characters
    image_path = create_image_with_exif(
        tmp_path / "test.jpg",
        camera_make=long_make
    )

    result = exif_service.extract_exif(str(image_path))

    assert result["camera_make"] == "A" * 100
    assert len(result["camera_make"]) == 100


def test_extract_exif_gps_coordinates(
    exif_service: ExifService, tmp_path: Path
) -> None:
    """Test extraction of GPS coordinates."""
    # NYC coordinates: 40.7128° N, 74.0060° W
    # GPS format: ((degrees, 1), (minutes, 1), (seconds, 100))
    lat_dms = ((40, 1), (42, 1), (4608, 100))  # 40°42'46.08"
    lon_dms = ((74, 1), (0, 1), (2160, 100))   # 74°0'21.6"

    image_path = create_image_with_exif(
        tmp_path / "test.jpg",
        gps_latitude=lat_dms,
        gps_longitude=lon_dms,
        gps_latitude_ref="N",
        gps_longitude_ref="W"
    )

    result = exif_service.extract_exif(str(image_path))

    assert result["gps_latitude"] is not None
    assert result["gps_longitude"] is not None

    # Check approximate values (GPS conversion may have small precision differences)
    assert abs(result["gps_latitude"] - 40.7128) < 0.01
    assert abs(result["gps_longitude"] - (-74.0060)) < 0.01


def test_extract_exif_gps_southern_western_hemisphere(
    exif_service: ExifService, tmp_path: Path
) -> None:
    """Test GPS coordinates in southern and western hemispheres are negative."""
    # Sydney coordinates: 33.8688° S, 151.2093° E
    lat_dms = ((33, 1), (52, 1), (760, 100))  # 33°52'7.6"
    lon_dms = ((151, 1), (12, 1), (3348, 100))  # 151°12'33.48"

    image_path = create_image_with_exif(
        tmp_path / "test.jpg",
        gps_latitude=lat_dms,
        gps_longitude=lon_dms,
        gps_latitude_ref="S",  # Southern hemisphere = negative
        gps_longitude_ref="E"  # Eastern hemisphere = positive
    )

    result = exif_service.extract_exif(str(image_path))

    assert result["gps_latitude"] is not None
    assert result["gps_longitude"] is not None

    # South = negative latitude, East = positive longitude
    assert result["gps_latitude"] < 0
    assert result["gps_longitude"] > 0
    assert abs(result["gps_latitude"] - (-33.8688)) < 0.01
    assert abs(result["gps_longitude"] - 151.2093) < 0.01


def test_extract_exif_no_gps_returns_none(
    exif_service: ExifService, tmp_path: Path
) -> None:
    """Test that missing GPS data returns None."""
    image_path = create_image_with_exif(tmp_path / "test.jpg")

    result = exif_service.extract_exif(str(image_path))

    assert result["gps_latitude"] is None
    assert result["gps_longitude"] is None


def test_extract_exif_stores_all_tags_in_metadata(
    exif_service: ExifService, tmp_path: Path
) -> None:
    """Test that all EXIF tags are stored in exif_metadata dict."""
    image_path = create_image_with_exif(
        tmp_path / "test.jpg",
        datetime_original="2023:07:15 14:30:00",
        camera_make="Apple",
        camera_model="iPhone 14 Pro"
    )

    result = exif_service.extract_exif(str(image_path))

    # Should have exif_metadata dict with readable tag names
    assert "exif_metadata" in result
    assert isinstance(result["exif_metadata"], dict)

    # Check some expected tags (may vary by PIL version)
    # At minimum, should have Make and Model
    metadata = result["exif_metadata"]
    assert "Make" in metadata or "Model" in metadata


def test_extract_exif_no_exif_data_returns_empty_dict(
    exif_service: ExifService, tmp_path: Path
) -> None:
    """Test that image without EXIF returns empty metadata dict."""
    # Create image without EXIF
    img = Image.new("RGB", (100, 100))
    image_path = tmp_path / "no_exif.jpg"
    img.save(image_path)

    result = exif_service.extract_exif(str(image_path))

    assert result["taken_at"] is None
    assert result["camera_make"] is None
    assert result["camera_model"] is None
    assert result["gps_latitude"] is None
    assert result["gps_longitude"] is None
    assert result["exif_metadata"] == {}


def test_extract_exif_nonexistent_file_returns_none_values(
    exif_service: ExifService
) -> None:
    """Test that nonexistent file returns None values without crashing."""
    result = exif_service.extract_exif("/nonexistent/path/to/image.jpg")

    assert result["taken_at"] is None
    assert result["camera_make"] is None
    assert result["camera_model"] is None
    assert result["gps_latitude"] is None
    assert result["gps_longitude"] is None
    assert result["exif_metadata"] == {}


def test_extract_exif_corrupt_image_returns_partial_data(
    exif_service: ExifService, tmp_path: Path
) -> None:
    """Test that corrupt image returns None values without crashing."""
    # Create corrupt image file
    corrupt_path = tmp_path / "corrupt.jpg"
    corrupt_path.write_bytes(b"NOT A VALID IMAGE FILE")

    result = exif_service.extract_exif(str(corrupt_path))

    # Should return None values, not raise exception
    assert result["taken_at"] is None
    assert result["camera_make"] is None
    assert result["camera_model"] is None


def test_extract_exif_invalid_datetime_format_returns_none(
    exif_service: ExifService
) -> None:
    """Test that invalid datetime format returns None."""
    # Test _parse_exif_datetime directly with invalid formats
    assert exif_service._parse_exif_datetime("invalid") is None
    assert exif_service._parse_exif_datetime("2023-07-15 14:30:00") is None  # Wrong separator
    assert exif_service._parse_exif_datetime("") is None
    assert exif_service._parse_exif_datetime(None) is None


def test_parse_exif_datetime_valid_format(exif_service: ExifService) -> None:
    """Test parsing of valid EXIF datetime string."""
    result = exif_service._parse_exif_datetime("2023:07:15 14:30:00")

    assert result == datetime(2023, 7, 15, 14, 30, 0, tzinfo=UTC)


def test_parse_exif_datetime_handles_bytes(exif_service: ExifService) -> None:
    """Test parsing of EXIF datetime as bytes."""
    result = exif_service._parse_exif_datetime(b"2023:07:15 14:30:00")

    assert result == datetime(2023, 7, 15, 14, 30, 0, tzinfo=UTC)


def test_convert_to_degrees_valid_gps_tuple(exif_service: ExifService) -> None:
    """Test conversion of GPS DMS tuple to decimal degrees."""
    # 40°42'46.08" = 40.7128°
    gps_tuple = ((40, 1), (42, 1), (4608, 100))

    result = exif_service._convert_to_degrees(gps_tuple)

    assert abs(result - 40.7128) < 0.01


def test_parse_gps_valid_coordinates(exif_service: ExifService) -> None:
    """Test parsing of valid GPS info dict."""
    gps_info = {
        1: "N",  # GPSLatitudeRef
        2: ((40, 1), (42, 1), (4608, 100)),  # GPSLatitude
        3: "W",  # GPSLongitudeRef
        4: ((74, 1), (0, 1), (2160, 100)),  # GPSLongitude
    }

    lat, lon = exif_service._parse_gps(gps_info)

    assert lat is not None
    assert lon is not None
    assert abs(lat - 40.7128) < 0.01
    assert abs(lon - (-74.0060)) < 0.01


def test_parse_gps_missing_data_returns_none(exif_service: ExifService) -> None:
    """Test that incomplete GPS data returns None."""
    # Missing longitude
    gps_info = {
        1: "N",
        2: ((40, 1), (42, 1), (4608, 100)),
    }

    lat, lon = exif_service._parse_gps(gps_info)

    assert lat is None
    assert lon is None


def test_parse_gps_invalid_coordinates_returns_none(exif_service: ExifService) -> None:
    """Test that invalid coordinates (outside valid range) return None."""
    # Invalid latitude (>90°)
    gps_info = {
        1: "N",
        2: ((95, 1), (0, 1), (0, 1)),  # 95° is invalid
        3: "W",
        4: ((74, 1), (0, 1), (0, 1)),
    }

    lat, lon = exif_service._parse_gps(gps_info)

    assert lat is None
    assert lon is None


def test_get_exif_service_returns_singleton() -> None:
    """Test that get_exif_service returns same instance."""
    service1 = get_exif_service()
    service2 = get_exif_service()

    assert service1 is service2


def test_extract_exif_thread_safe(
    exif_service: ExifService, tmp_path: Path
) -> None:
    """Test that EXIF extraction is thread-safe (uses lock internally)."""
    import threading

    image_path = create_image_with_exif(
        tmp_path / "test.jpg",
        datetime_original="2023:07:15 14:30:00",
        camera_make="Apple"
    )

    results = []
    errors = []

    def extract_exif_thread() -> None:
        try:
            result = exif_service.extract_exif(str(image_path))
            results.append(result)
        except Exception as e:
            errors.append(e)

    # Run 10 threads simultaneously
    threads = [threading.Thread(target=extract_exif_thread) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All threads should succeed
    assert len(errors) == 0
    assert len(results) == 10

    # All results should be identical
    for result in results:
        assert result["taken_at"] == datetime(2023, 7, 15, 14, 30, 0, tzinfo=UTC)
        assert result["camera_make"] == "Apple"


def test_extract_exif_handles_whitespace_in_camera_fields(
    exif_service: ExifService, tmp_path: Path
) -> None:
    """Test that whitespace is trimmed from camera make/model."""
    image_path = create_image_with_exif(
        tmp_path / "test.jpg",
        camera_make="  Apple  ",
        camera_model="  iPhone 14 Pro  "
    )

    result = exif_service.extract_exif(str(image_path))

    assert result["camera_make"] == "Apple"
    assert result["camera_model"] == "iPhone 14 Pro"


def test_extract_exif_complete_example(
    exif_service: ExifService, tmp_path: Path
) -> None:
    """Test extraction of complete EXIF data with all fields."""
    # NYC coordinates
    lat_dms = ((40, 1), (42, 1), (4608, 100))
    lon_dms = ((74, 1), (0, 1), (2160, 100))

    image_path = create_image_with_exif(
        tmp_path / "complete.jpg",
        datetime_original="2023:07:15 14:30:00",
        camera_make="Apple",
        camera_model="iPhone 14 Pro",
        gps_latitude=lat_dms,
        gps_longitude=lon_dms,
        gps_latitude_ref="N",
        gps_longitude_ref="W"
    )

    result = exif_service.extract_exif(str(image_path))

    # All fields should be populated
    assert result["taken_at"] == datetime(2023, 7, 15, 14, 30, 0, tzinfo=UTC)
    assert result["camera_make"] == "Apple"
    assert result["camera_model"] == "iPhone 14 Pro"
    assert abs(result["gps_latitude"] - 40.7128) < 0.01
    assert abs(result["gps_longitude"] - (-74.0060)) < 0.01
    assert len(result["exif_metadata"]) > 0
