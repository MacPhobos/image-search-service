"""EXIF metadata extraction service for image assets.

Extracts structured metadata from image EXIF data, with robust error handling
for corrupt or malformed EXIF data.

CRITICAL DATE EXTRACTION RULES:
- ONLY use EXIF tags: DateTimeOriginal (36867) or DateTimeDigitized (36868)
- NEVER use DateTime (306) - this is file modification date
- NEVER infer date from filename, directory, or filesystem
- If no valid EXIF date → return None (no assumptions)
"""

import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image
from PIL.ExifTags import IFD, TAGS

from image_search_service.core.logging import get_logger

logger = get_logger(__name__)

# Lock for thread-safe EXIF parsing
# PIL's EXIF parser has known thread-safety issues with malformed data
_exif_parse_lock = threading.Lock()

# EXIF tag IDs for date extraction
EXIF_TAG_DATETIME_ORIGINAL = 36867  # DateTimeOriginal - when photo was taken
EXIF_TAG_DATETIME_DIGITIZED = 36868  # DateTimeDigitized - when photo was digitized
EXIF_TAG_DATETIME = 306  # DateTime - file modification time (DO NOT USE)

# EXIF tag IDs for camera info
EXIF_TAG_MAKE = 271  # Camera manufacturer
EXIF_TAG_MODEL = 272  # Camera model

# GPS tag ID
EXIF_TAG_GPS_INFO = 34853  # GPS metadata

# EXIF sub-IFD tag ID (where DateTimeOriginal/DateTimeDigitized are stored)
EXIF_IFD_TAG = 34665  # EXIF sub-IFD pointer


class ExifService:
    """Service for extracting EXIF metadata from images.

    Handles corrupt/malformed EXIF data gracefully with thread-safe parsing.
    """

    def extract_exif(self, image_path: str) -> dict[str, Any]:
        """Extract EXIF metadata from image file.

        Returns structured metadata with proper type conversions for database storage.
        Handles corrupt EXIF data gracefully - never raises exceptions, logs warnings instead.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with keys:
                - taken_at: datetime | None (from EXIF DateTimeOriginal/DateTimeDigitized only)
                - camera_make: str | None
                - camera_model: str | None
                - gps_latitude: float | None
                - gps_longitude: float | None
                - exif_metadata: dict[str, Any] (all readable EXIF tags)

        Example:
            {
                "taken_at": datetime(2023, 7, 15, 14, 30, 0, tzinfo=timezone.utc),
                "camera_make": "Apple",
                "camera_model": "iPhone 14 Pro",
                "gps_latitude": 40.7128,
                "gps_longitude": -74.0060,
                "exif_metadata": {
                    "FocalLength": "6.86mm",
                    "FNumber": "f/1.78",
                    "ISOSpeedRatings": 64,
                    ...
                }
            }
        """
        # Initialize result with None values
        result: dict[str, Any] = {
            "taken_at": None,
            "camera_make": None,
            "camera_model": None,
            "gps_latitude": None,
            "gps_longitude": None,
            "exif_metadata": {},
        }

        # Validate file exists
        path = Path(image_path)
        if not path.exists():
            logger.warning(f"Image file not found: {image_path}")
            return result

        # Thread-safe EXIF extraction
        with _exif_parse_lock:
            try:
                with Image.open(path) as img:
                    # Get raw EXIF data
                    exif_data = img.getexif()
                    if not exif_data:
                        logger.debug(f"No EXIF data found in image: {image_path}")
                        return result

                    # Extract all readable EXIF tags for storage
                    exif_dict: dict[str, Any] = {}
                    for tag_id, value in exif_data.items():
                        tag_name = TAGS.get(tag_id, str(tag_id))

                        # Convert value to JSON-serializable type
                        try:
                            # Handle bytes (common in EXIF)
                            if isinstance(value, bytes):
                                try:
                                    exif_dict[tag_name] = value.decode("utf-8", errors="ignore")
                                except Exception:
                                    exif_dict[tag_name] = str(value)
                            # Handle tuples (GPS coordinates, etc.)
                            elif isinstance(value, tuple):
                                exif_dict[tag_name] = list(value)
                            # Handle basic types
                            elif isinstance(value, (str, int, float, bool)):
                                exif_dict[tag_name] = value
                            else:
                                exif_dict[tag_name] = str(value)
                        except Exception as e:
                            logger.debug(
                                f"Could not serialize EXIF tag {tag_name}: {e}"
                            )
                            continue

                    # Sanitize EXIF metadata for JSON/JSONB storage
                    # (removes null bytes that PostgreSQL JSONB cannot store)
                    result["exif_metadata"] = _sanitize_for_json(exif_dict)

                    # Extract taken_at (CRITICAL: only from DateTimeOriginal/DateTimeDigitized)
                    # These tags are in the EXIF sub-IFD, NOT the main IFD
                    taken_at_str = None
                    try:
                        # Access EXIF sub-IFD where DateTimeOriginal/DateTimeDigitized are stored
                        exif_ifd = exif_data.get_ifd(IFD.Exif)
                        taken_at_str = exif_ifd.get(EXIF_TAG_DATETIME_ORIGINAL)
                        if not taken_at_str:
                            taken_at_str = exif_ifd.get(EXIF_TAG_DATETIME_DIGITIZED)
                    except KeyError:
                        # No EXIF sub-IFD present
                        logger.debug(f"No EXIF sub-IFD found in image: {image_path}")
                    except Exception as e:
                        logger.debug(f"Could not access EXIF sub-IFD: {e}")

                    if taken_at_str:
                        result["taken_at"] = self._parse_exif_datetime(taken_at_str)

                    # Extract camera info (these ARE in the main IFD)
                    camera_make = exif_data.get(EXIF_TAG_MAKE)
                    if camera_make:
                        result["camera_make"] = str(camera_make).strip()[:100]  # Limit to 100 chars

                    camera_model = exif_data.get(EXIF_TAG_MODEL)
                    if camera_model:
                        result["camera_model"] = str(camera_model).strip()[:100]

                    # Extract GPS coordinates
                    # GPS data is stored in a separate IFD (Image File Directory)
                    try:
                        gps_ifd = exif_data.get_ifd(EXIF_TAG_GPS_INFO)
                        if gps_ifd:
                            lat, lon = self._parse_gps(gps_ifd)
                            result["gps_latitude"] = lat
                            result["gps_longitude"] = lon
                    except KeyError:
                        # No GPS data in EXIF
                        pass
                    except Exception as e:
                        logger.debug(f"Could not extract GPS data: {e}")

            except Exception as e:
                # Log warning but don't crash - return partial data
                logger.warning(
                    f"Failed to extract EXIF from {image_path}: {e}",
                    exc_info=True
                )

        return result

    def _parse_exif_datetime(self, value: Any) -> datetime | None:
        """Parse EXIF datetime string to datetime object.

        EXIF datetime format: "YYYY:MM:DD HH:MM:SS"
        Assumes UTC timezone if not specified in EXIF.

        Args:
            value: EXIF datetime value (string or bytes)

        Returns:
            datetime with timezone or None if parsing fails
        """
        try:
            # Convert bytes to string if needed
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="ignore")

            # Ensure it's a string
            datetime_str = str(value).strip()

            # Parse EXIF datetime format: "YYYY:MM:DD HH:MM:SS"
            dt = datetime.strptime(datetime_str, "%Y:%m:%d %H:%M:%S")

            # Assume UTC if timezone not specified
            # (EXIF rarely includes timezone info)
            return dt.replace(tzinfo=UTC)

        except Exception as e:
            logger.debug(f"Could not parse EXIF datetime '{value}': {e}")
            return None

    def _parse_gps(self, gps_info: dict[int, Any]) -> tuple[float | None, float | None]:
        """Convert GPS EXIF data to decimal degree coordinates.

        GPS data is stored as:
            GPSLatitude: ((degrees, 1), (minutes, 1), (seconds, 100))
            GPSLatitudeRef: "N" or "S"
            GPSLongitude: ((degrees, 1), (minutes, 1), (seconds, 100))
            GPSLongitudeRef: "E" or "W"

        Args:
            gps_info: GPS EXIF dictionary

        Returns:
            Tuple of (latitude, longitude) as floats or (None, None) if invalid
        """
        try:
            # Extract GPS data with proper tag names
            gps_latitude = gps_info.get(2)  # GPSLatitude
            gps_latitude_ref = gps_info.get(1)  # GPSLatitudeRef (N/S)
            gps_longitude = gps_info.get(4)  # GPSLongitude
            gps_longitude_ref = gps_info.get(3)  # GPSLongitudeRef (E/W)

            if not all([gps_latitude, gps_latitude_ref, gps_longitude, gps_longitude_ref]):
                return (None, None)

            # Ensure we have tuples before converting
            if not isinstance(gps_latitude, tuple) or not isinstance(gps_longitude, tuple):
                return (None, None)

            # Convert to decimal degrees
            lat = self._convert_to_degrees(gps_latitude)
            if gps_latitude_ref == "S":
                lat = -lat

            lon = self._convert_to_degrees(gps_longitude)
            if gps_longitude_ref == "W":
                lon = -lon

            # Validate coordinates
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                logger.warning(f"Invalid GPS coordinates: lat={lat}, lon={lon}")
                return (None, None)

            return (lat, lon)

        except Exception as e:
            logger.debug(f"Could not parse GPS data: {e}")
            return (None, None)

    def _convert_to_degrees(self, value: tuple[Any, ...]) -> float:
        """Convert GPS coordinate from DMS (degrees, minutes, seconds) to decimal degrees.

        Handles multiple formats:
        1. Tuple of numeric values (IFDRational, int, float): (degrees, minutes, seconds)
        2. Tuple of tuples: ((degrees, 1), (minutes, 1), (seconds, 100))

        Args:
            value: DMS coordinate tuple

        Returns:
            Decimal degree value

        Raises:
            ValueError: If value format is invalid
        """
        # Check format and extract degrees, minutes, seconds
        if len(value) != 3:
            raise ValueError(f"Invalid GPS coordinate tuple length: {len(value)}")

        # Check if first element is a tuple (legacy format)
        if isinstance(value[0], tuple) and len(value[0]) == 2:
            # Legacy format: ((degrees, divisor), (minutes, divisor), (seconds, divisor))
            d = float(value[0][0]) / float(value[0][1])
            m = float(value[1][0]) / float(value[1][1])
            s = float(value[2][0]) / float(value[2][1])
        else:
            # Modern PIL format: (degrees, minutes, seconds) as numeric values
            # Works with IFDRational, int, float - anything that can convert to float
            try:
                d = float(value[0])
                m = float(value[1])
                s = float(value[2])
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid GPS coordinate format: {value}") from e

        # Convert to decimal degrees
        return d + (m / 60.0) + (s / 3600.0)


def _sanitize_for_json(value: Any) -> Any:
    """Remove null bytes and other non-JSON-safe characters from values.

    PostgreSQL JSONB columns cannot store null bytes (\x00 or \u0000).
    Some EXIF data (especially MakerNote fields) contains these characters.

    Args:
        value: EXIF value to sanitize

    Returns:
        Sanitized value safe for JSON/JSONB storage
    """
    if isinstance(value, str):
        # Remove null bytes (both representations)
        return value.replace('\x00', '').replace('\u0000', '')
    elif isinstance(value, bytes):
        # Decode bytes, removing null bytes
        try:
            decoded = value.decode('utf-8', errors='replace')
            return decoded.replace('\x00', '').replace('\u0000', '')
        except Exception:
            # If decoding fails completely, return None
            return None
    elif isinstance(value, dict):
        # Recursively sanitize dictionary values
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        # Recursively sanitize list/tuple items
        sanitized = [_sanitize_for_json(v) for v in value]
        # Preserve tuple type
        return tuple(sanitized) if isinstance(value, tuple) else sanitized
    else:
        # Return other types as-is (int, float, bool, None)
        return value


# Global service instance (lazy initialization)
_exif_service: ExifService | None = None


def get_exif_service() -> ExifService:
    """Get global ExifService instance (lazy initialization).

    Returns:
        Singleton ExifService instance
    """
    global _exif_service
    if _exif_service is None:
        _exif_service = ExifService()
    return _exif_service
