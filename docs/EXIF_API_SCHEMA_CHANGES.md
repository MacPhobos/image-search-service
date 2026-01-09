# EXIF Metadata in API Schemas

## Overview

Updated the API schemas to expose EXIF metadata fields in API responses. The Asset schema now includes camera information and GPS location data extracted from image files.

## Changes

### New Schemas

#### LocationMetadata
```python
class LocationMetadata(BaseModel):
    """GPS location from EXIF."""
    latitude: float = Field(alias="lat")
    longitude: float = Field(alias="lng")
```

**JSON Output:**
```json
{
  "lat": 37.7749,
  "lng": -122.4194
}
```

#### CameraMetadata
```python
class CameraMetadata(BaseModel):
    """Camera info from EXIF."""
    make: str | None = None
    model: str | None = None
```

**JSON Output:**
```json
{
  "make": "Canon",
  "model": "EOS 5D Mark IV"
}
```

### Updated Asset Schema

The `Asset` schema now includes three new fields:

1. **takenAt** (datetime | None): Photo capture timestamp from EXIF DateTimeOriginal
2. **camera** (CameraMetadata | None): Camera make and model
3. **location** (LocationMetadata | None): GPS coordinates (only present when both lat and lng are available)

**Example Response:**
```json
{
  "id": 123,
  "path": "/photos/vacation/IMG_4567.jpg",
  "createdAt": "2024-01-15T12:00:00Z",
  "indexedAt": "2024-01-15T12:05:00Z",
  "takenAt": "2023-08-20T14:30:00Z",
  "camera": {
    "make": "Canon",
    "model": "EOS 5D Mark IV"
  },
  "location": {
    "lat": 37.7749,
    "lng": -122.4194
  },
  "url": "/api/v1/images/123/full",
  "thumbnailUrl": "/api/v1/images/123/thumbnail",
  "filename": "IMG_4567.jpg"
}
```

## Technical Implementation

### Database to API Transformation

The database stores EXIF data in flat fields:
- `taken_at`
- `camera_make`
- `camera_model`
- `gps_latitude`
- `gps_longitude`

The API schema transforms these into nested objects using a Pydantic `@model_validator`:

```python
@model_validator(mode="before")
@classmethod
def build_nested_metadata(cls, data: Any) -> dict[str, Any]:
    """Transform flat DB fields into nested EXIF metadata objects."""
    # ... transforms ImageAsset model into nested structure
```

### Rules

1. **camera** is only present when at least one of `camera_make` or `camera_model` is non-null
2. **location** is only present when BOTH `gps_latitude` AND `gps_longitude` are non-null
3. **takenAt** is only populated from EXIF DateTimeOriginal/DateTimeDigitized (never inferred)

## Backward Compatibility

This is a **backward-compatible change**:
- All new fields are optional (can be null)
- Existing clients that don't expect these fields will ignore them
- Existing API endpoints return the same structure with additional fields

## Testing

Comprehensive tests added in `tests/unit/api/test_schemas.py`:

- Location metadata serialization with lat/lng aliases
- Camera metadata with partial/full data
- Asset transformation from database models
- Nested object construction rules
- JSON serialization with camelCase

All 19 new tests pass ✅

## Next Steps

1. **Frontend Integration**: Update TypeScript types by regenerating from OpenAPI spec
   ```bash
   cd image-search-ui
   npm run gen:api
   ```

2. **API Contract**: Update `docs/api-contract.md` to document the new fields

3. **EXIF Extraction**: Ensure the EXIF extraction job populates these fields when ingesting images

## Related Files

- `src/image_search_service/api/schemas.py` - Updated schemas
- `src/image_search_service/db/models.py` - Database model (already has EXIF fields)
- `tests/unit/api/test_schemas.py` - Comprehensive test suite
- `docs/api-examples/asset-with-exif.json` - Example JSON response
