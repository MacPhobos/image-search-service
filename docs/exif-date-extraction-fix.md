# EXIF Date Extraction Fix

**Date**: 2026-01-09
**Issue**: ExifService was not extracting `taken_at` dates from images with DateTimeOriginal (0x9003) and DateTimeDigitized (0x9004) EXIF tags.

## Root Cause

The ExifService was looking for DateTimeOriginal (36867) and DateTimeDigitized (36868) in the **main IFD** (Image File Directory), but according to the EXIF specification, these tags are stored in the **EXIF sub-IFD**.

### EXIF Structure

```
Image File:
├── Main IFD (0th IFD)
│   ├── Make (271) - Camera manufacturer
│   ├── Model (272) - Camera model
│   ├── DateTime (306) - File modification time [NEVER USE FOR taken_at]
│   └── ...
│
├── EXIF sub-IFD (accessed via get_ifd(IFD.Exif))
│   ├── DateTimeOriginal (36867) ✅ - When photo was taken
│   ├── DateTimeDigitized (36868) ✅ - When photo was digitized
│   ├── FNumber, ISOSpeed, etc.
│   └── ...
│
└── GPS sub-IFD (accessed via get_ifd(IFD.GPSInfo))
    ├── GPSLatitude
    ├── GPSLongitude
    └── ...
```

## The Fix

### Before (Incorrect)

```python
# Extract taken_at (CRITICAL: only from DateTimeOriginal/DateTimeDigitized)
taken_at_str = exif_data.get(EXIF_TAG_DATETIME_ORIGINAL)  # ❌ Wrong IFD!
if not taken_at_str:
    taken_at_str = exif_data.get(EXIF_TAG_DATETIME_DIGITIZED)  # ❌ Wrong IFD!
```

The code was searching in the main IFD (`exif_data.get()`), where DateTimeOriginal/DateTimeDigitized don't exist in real camera images.

### After (Correct)

```python
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
```

Now the code correctly accesses the EXIF sub-IFD using `exif_data.get_ifd(IFD.Exif)`, matching the pattern already used for GPS data.

## Changes Made

### 1. `src/image_search_service/services/exif_service.py`

- **Added import**: `from PIL.ExifTags import IFD, TAGS` (added `IFD`)
- **Updated date extraction logic** (lines 141-157):
  - Access EXIF sub-IFD via `exif_data.get_ifd(IFD.Exif)`
  - Look for DateTimeOriginal/DateTimeDigitized in correct IFD
  - Handle KeyError when EXIF sub-IFD doesn't exist
- **Added documentation**: Comments explaining IFD structure

### 2. `tests/unit/services/test_exif_service.py`

- **Added import**: `from PIL.ExifTags import IFD`
- **Updated test helper** `create_image_with_exif()`:
  - DateTime tags now correctly placed in EXIF sub-IFD
  - GPS data now correctly placed in GPS sub-IFD (using `IFD.GPSInfo`)
  - Matches real camera EXIF structure
- **All 29 tests pass** ✅

## Verification

### Test Results

```bash
uv run pytest tests/unit/services/test_exif_service.py -v
# 29 passed, 2 warnings in 0.11s
```

### Type Check

```bash
uv run mypy src/image_search_service/services/exif_service.py
# Success: no issues found in 1 source file
```

### Test Script

A test script was created at `test_exif_fix.py` to inspect real image EXIF structure:

```bash
python test_exif_fix.py /path/to/photo.jpg
```

This script:
1. Shows all tags in main IFD and EXIF sub-IFD
2. Highlights DateTimeOriginal (36867) when found
3. Tests ExifService.extract_exif() on the image
4. Confirms taken_at is now extracted correctly

## Impact

- **Before**: `taken_at` was `None` for ALL images, even those with DateTimeOriginal/DateTimeDigitized
- **After**: `taken_at` is correctly extracted from EXIF sub-IFD for real camera images

### Expected Behavior After Fix

```python
service = ExifService()
result = service.extract_exif("/path/to/camera_photo.jpg")

# Before fix:
# result["taken_at"] = None  # ❌ Missing!

# After fix:
# result["taken_at"] = datetime(2023, 7, 15, 14, 30, 0, tzinfo=UTC)  # ✅ Correct!
```

## EXIF Specification Reference

According to the EXIF 2.3 specification:

- **Main IFD (0th IFD)**: Contains basic image information (Make, Model, Orientation, DateTime)
- **EXIF sub-IFD**: Contains photography-specific data (DateTimeOriginal, DateTimeDigitized, FNumber, ISOSpeed, etc.)
- **GPS sub-IFD**: Contains GPS coordinates
- **Interoperability IFD**: Contains interoperability information

### Tag Locations

| Tag | ID | Location | Usage |
|-----|-----|----------|-------|
| DateTime | 306 | Main IFD | File modification time (NOT camera capture time) |
| DateTimeOriginal | 36867 | EXIF sub-IFD | When photo was taken ✅ |
| DateTimeDigitized | 36868 | EXIF sub-IFD | When photo was digitized ✅ |
| Make | 271 | Main IFD | Camera manufacturer |
| Model | 272 | Main IFD | Camera model |

## Related Files

- `src/image_search_service/services/exif_service.py` - Fixed service
- `tests/unit/services/test_exif_service.py` - Updated tests
- `test_exif_fix.py` - Diagnostic script for real images

## Next Steps

After this fix is deployed:

1. **Re-run backfill**: Re-import images to extract dates:
   ```bash
   make ingest DIR=/path/to/photos
   ```

2. **Verify in logs**: Check that "taken=YYYY-MM-DD" appears in backfill output

3. **Test in UI**: Confirm images now show "Taken on" dates in frontend

## Credits

- **Issue identified**: User reported "taken=N/A" despite images having DateTimeOriginal tags
- **Root cause analysis**: Investigated PIL EXIF structure and discovered IFD mismatch
- **Fix implemented**: Correctly access EXIF sub-IFD using PIL's `get_ifd()` method
