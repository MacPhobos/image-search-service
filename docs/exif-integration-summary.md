# EXIF Extraction Integration Summary

## Overview
Integrated EXIF metadata extraction into the background thumbnail generation jobs. EXIF data (date taken, camera info, GPS coordinates) is now automatically extracted when thumbnails are generated.

## Integration Points

### 1. Thumbnail Jobs (`src/image_search_service/queue/thumbnail_jobs.py`)

Two background jobs now extract EXIF data:

#### `generate_single_thumbnail(asset_id)`
- **When**: Individual asset processing (API-triggered)
- **Behavior**: 
  - Generates thumbnail
  - Extracts EXIF data (taken_at, camera, GPS)
  - Updates ImageAsset record with all metadata
  - Logs INFO level for successful extraction

#### `generate_thumbnails_batch(session_id)`
- **When**: Batch processing for training sessions
- **Behavior**:
  - Generates thumbnails for all assets in session
  - Extracts EXIF data for each asset
  - Updates ImageAsset records in single commit per asset
  - Logs DEBUG level to avoid log spam in batch mode

## Error Handling Strategy

EXIF extraction uses **graceful degradation**:

1. **EXIF extraction failure does NOT fail the job**
   - Thumbnail generation is the primary goal
   - EXIF extraction is a bonus feature
   - Job continues even if EXIF parsing fails

2. **Error logging levels**:
   - Single thumbnail job: `logger.warning()` with exception traceback
   - Batch job: `logger.debug()` to avoid log spam

3. **Database updates**:
   - Only update fields that have non-None values
   - Use `if exif_data.get("field")` checks before assignment
   - Handle GPS coordinates with `is not None` check (allows 0.0 values)

## Data Flow

```
Image uploaded
    ↓
Thumbnail job enqueued
    ↓
├─ Generate thumbnail ✓ (primary task)
│  └─ Update: thumbnail_path, width, height
│
└─ Extract EXIF ✓ (optional enhancement)
   └─ Update: taken_at, camera_make, camera_model, 
      gps_latitude, gps_longitude, exif_metadata
    ↓
Single database commit (atomic)
```

## Implementation Details

### Key Design Decisions

1. **Integrated with thumbnail generation** - Reuses existing job infrastructure
2. **Synchronous database session** - Uses `get_sync_session()` (RQ workers are sync)
3. **Lazy service initialization** - `get_exif_service()` follows project pattern
4. **Thread-safe parsing** - ExifService uses lock for PIL's EXIF parser
5. **Atomic commits** - Single commit updates both thumbnail and EXIF data

### Code Pattern

```python
# Extract EXIF metadata (graceful degradation)
try:
    exif_service = get_exif_service()
    exif_data = exif_service.extract_exif(asset.path)

    if exif_data:
        # Update asset with EXIF data (only if values are present)
        if exif_data.get("taken_at"):
            asset.taken_at = exif_data["taken_at"]
        if exif_data.get("camera_make"):
            asset.camera_make = exif_data["camera_make"]
        # ... (other fields)
        
        logger.info(f"Extracted EXIF data for asset {asset_id}")
except Exception as e:
    # Log warning but don't fail the job - EXIF extraction is optional
    logger.warning(f"Failed to extract EXIF data: {e}", exc_info=True)

# Commit all changes (thumbnail + EXIF)
db_session.commit()
```

## Testing

### Existing Test Coverage

- **ExifService tests**: `tests/unit/services/test_exif_service.py`
  - Tests EXIF extraction logic
  - Tests date parsing (DateTimeOriginal, DateTimeDigitized)
  - Tests GPS coordinate conversion
  - Tests error handling for corrupt EXIF

### Integration Testing

To test the integration:

1. **Start services**:
   ```bash
   make db-up
   make dev     # Terminal 1: API
   make worker  # Terminal 2: Background worker
   ```

2. **Upload an image with EXIF data**:
   ```bash
   curl -X POST http://localhost:8000/api/v1/assets/upload \
     -F "file=@photo_with_exif.jpg"
   ```

3. **Verify EXIF extraction**:
   - Check logs for "Extracted EXIF data for asset X"
   - Query database to verify fields populated
   - Use API to retrieve asset and check metadata

## Database Schema

EXIF fields in `ImageAsset` model:

```python
# EXIF metadata (extracted from image files)
taken_at: Mapped[datetime | None]           # When photo was taken (EXIF only)
camera_make: Mapped[str | None]             # Camera manufacturer (e.g., "Apple")
camera_model: Mapped[str | None]            # Camera model (e.g., "iPhone 14 Pro")
gps_latitude: Mapped[float | None]          # Decimal degrees latitude
gps_longitude: Mapped[float | None]         # Decimal degrees longitude
exif_metadata: Mapped[dict | None]          # Full EXIF blob (JSONB)
```

## Backfill Script

For existing assets that were uploaded before EXIF integration, use the backfill script:

```bash
# Backfill all assets without EXIF data (taken_at IS NULL)
make exif-backfill

# Process limited number with custom batch size
make exif-backfill LIMIT=1000 BATCH_SIZE=50

# Dry run to preview what would be done
make exif-backfill DRY_RUN=1 LIMIT=100

# Direct script invocation
python scripts/backfill_exif.py --limit 1000 --batch-size 100 --dry-run
```

**Script behavior**:
- Queries assets where `taken_at IS NULL`
- Extracts EXIF metadata from image files
- Updates database in batches (default: 100 assets per commit)
- Reports statistics: processed, updated, skipped (no EXIF), failed (errors)
- Gracefully handles missing files and corrupt EXIF data

**Output example**:
```
Processing batch 1/50... (87 assets updated, 13 skipped)
Processing batch 2/50... (92 assets updated, 8 skipped)
...
Done! Processed: 5000, Updated: 4350, Skipped: 620, Failed: 30
```

## Future Enhancements

1. ✅ **Backfill existing assets** - Implemented via `scripts/backfill_exif.py`
2. **API response** - Return EXIF metadata in asset GET endpoints
3. **Search filters** - Enable filtering by date taken, camera, location
4. **Map visualization** - Display photos on map using GPS coordinates

## Quality Checks

✅ **Linting**: `ruff check src/image_search_service/queue/thumbnail_jobs.py`
✅ **Type checking**: `mypy --strict src/image_search_service/queue/thumbnail_jobs.py`
✅ **No breaking changes**: Existing thumbnail job behavior preserved
✅ **Graceful degradation**: EXIF failure doesn't break thumbnail generation
✅ **Follows patterns**: Uses existing service pattern, logging, error handling

## Related Files

- **Integration**: `src/image_search_service/queue/thumbnail_jobs.py`
- **Service**: `src/image_search_service/services/exif_service.py`
- **Model**: `src/image_search_service/db/models.py` (ImageAsset)
- **Backfill Script**: `scripts/backfill_exif.py`
- **Tests**: `tests/unit/services/test_exif_service.py`

---

**Date**: 2026-01-09
**Status**: ✅ Integrated and tested
