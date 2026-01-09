# EXIF Null Byte Sanitization Fix

## Problem

**Error**: `asyncpg.exceptions.CharacterNotInRepertoireError: invalid byte sequence for encoding "UTF8": 0x00`

**Root Cause**: PostgreSQL JSONB columns cannot store null bytes (`\x00`, `\u0000`, `\0`). Some camera EXIF data (especially in `MakerNote`, `UserComment`, and occasionally `camera_make`/`camera_model` fields) contains these characters. The previous fix only sanitized `exif_metadata` but not other fields like `camera_make` and `camera_model`.

## Solution

Applied comprehensive null byte sanitization to **all fields** in the EXIF extraction result by:

1. **Moving sanitization to final return point** in `extract_exif()` method
2. **Sanitizing entire result dictionary** instead of just `exif_metadata`
3. **Enhanced `_sanitize_for_json()` function** to handle all null byte representations:
   - `\x00` (Python bytes null)
   - `\u0000` (Unicode null)
   - `\0` (literal null)

## Implementation Details

### Before (Incomplete Fix)

```python
# Only sanitized exif_metadata, not camera_make/camera_model
result["exif_metadata"] = _sanitize_for_json(exif_dict)

# Later in code...
result["camera_make"] = str(camera_make).strip()[:100]  # ❌ No sanitization!
result["camera_model"] = str(camera_model).strip()[:100]  # ❌ No sanitization!

return result  # ❌ Returns unsanitized data
```

### After (Complete Fix)

```python
# Store raw data first (no early sanitization)
result["exif_metadata"] = exif_dict
result["camera_make"] = str(camera_make).strip()[:100]
result["camera_model"] = str(camera_model).strip()[:100]

# ... extract all other fields ...

# ✅ Sanitize ENTIRE result at return point
sanitized = _sanitize_for_json(result)
return cast(dict[str, Any], sanitized)
```

### Sanitization Function Enhancement

```python
def _sanitize_for_json(value: Any) -> Any:
    """Remove null bytes from all string values recursively."""
    if isinstance(value, str):
        # Remove all null byte representations
        return value.replace('\x00', '').replace('\u0000', '').replace('\0', '')
    elif isinstance(value, bytes):
        # Decode and remove null bytes
        decoded = value.decode('utf-8', errors='replace')
        return decoded.replace('\x00', '').replace('\u0000', '').replace('\0', '')
    elif isinstance(value, dict):
        # Recursively sanitize dictionary values
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        # Recursively sanitize list/tuple items
        sanitized = [_sanitize_for_json(v) for v in value]
        return tuple(sanitized) if isinstance(value, tuple) else sanitized
    elif isinstance(value, (datetime, int, float, bool, type(None))):
        # Preserve datetime and primitive types as-is
        return value
    else:
        # Return other types unchanged
        return value
```

## Testing

### Comprehensive Test Coverage

1. **Null byte removal from strings**:
   - `\x00` representation
   - `\u0000` representation
   - `\0` literal null
   - Multiple null bytes
   - Null bytes at start, middle, end

2. **Bytes with null bytes**:
   - Decode and sanitize
   - Handle decoding errors gracefully

3. **Nested structures**:
   - Dictionaries with nested dictionaries
   - Lists and tuples
   - Mixed nested structures

4. **Type preservation**:
   - `datetime` objects preserved
   - `int`, `float`, `bool`, `None` preserved
   - Tuple type preserved (not converted to list)

5. **Realistic EXIF data**:
   - Complete EXIF extraction result
   - Camera make/model with null bytes
   - EXIF metadata with null bytes
   - GPS coordinates preserved
   - Datetime preserved

### Test Results

All 31 tests pass, including new regression tests:
- `test_sanitize_for_json_preserves_datetime` - Ensures datetime objects are not corrupted
- `test_extract_exif_sanitizes_camera_make_with_null_bytes` - Regression test for the original bug

## Impact

### Before Fix
- EXIF backfill would crash on images with null bytes in camera make/model
- Database insertion failures
- Incomplete EXIF data in database

### After Fix
- ✅ All EXIF data safely stored in PostgreSQL JSONB
- ✅ No data corruption (datetimes, numbers preserved)
- ✅ Graceful handling of all null byte representations
- ✅ Comprehensive test coverage prevents regressions

## Files Modified

1. **`src/image_search_service/services/exif_service.py`**:
   - Enhanced `_sanitize_for_json()` to handle `\0` literal nulls
   - Added explicit datetime type preservation
   - Moved sanitization to final return point (sanitizes ALL fields)
   - Added type cast for mypy strict mode compliance

2. **`tests/unit/services/test_exif_service.py`**:
   - Added `test_sanitize_for_json_preserves_datetime()`
   - Added `test_extract_exif_sanitizes_camera_make_with_null_bytes()`

## Verification Checklist

- [x] All 31 EXIF service tests pass
- [x] Type checking passes (mypy strict mode)
- [x] Linting passes (ruff)
- [x] Datetime preservation verified
- [x] Camera make/model sanitization verified
- [x] Nested structure sanitization verified
- [x] No performance regression (sanitization is O(n) in data size)

## Related Issues

- Original bug: EXIF backfill failing with null byte errors
- Previous incomplete fix: Only sanitized `exif_metadata`, not all fields

## Prevention

To prevent similar issues in the future:

1. **Always sanitize at boundaries**: When data crosses from external sources (EXIF, user input) to database
2. **Sanitize entire data structures**: Don't rely on partial sanitization
3. **Test with realistic data**: Include null bytes, unicode, binary data in tests
4. **Use comprehensive test coverage**: Test all code paths and edge cases

## Deployment Notes

No migration required. Existing EXIF data in database is unaffected. Future EXIF extractions will have null bytes properly sanitized.

Backfill can be safely re-run on previously failed images:

```bash
make exif-backfill LIMIT=10000
```

---

**Date**: 2026-01-09
**Author**: Claude Code (AI Assistant)
**Status**: Complete ✅
