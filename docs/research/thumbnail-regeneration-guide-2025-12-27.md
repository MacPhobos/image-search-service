# Thumbnail Regeneration Guide

**Research Date**: 2025-12-27
**Context**: Fixed EXIF orientation in thumbnail_service.py, need to regenerate cached thumbnails
**Status**: Complete

## Summary

After fixing EXIF orientation handling in `thumbnail_service.py`, existing cached thumbnails still have incorrect orientation. This guide documents multiple methods to regenerate thumbnails.

## Thumbnail Storage Architecture

### Storage Location
- **Directory**: `/tmp/thumbnails` (configurable via `THUMBNAIL_DIR` environment variable)
- **Structure**: Hash-sharded subdirectories (2-char MD5 prefix)
  ```
  /tmp/thumbnails/
    ├── 00/
    ├── 01/
    ├── 02/
    ...
    └── ff/
  ```
- **Filename**: `{asset_id}.jpg` (JPEG format with 85% quality)

### Configuration
From `/export/workspace/image-search/image-search-service/src/image_search_service/core/config.py`:
```python
thumbnail_dir: str = Field(default="/tmp/thumbnails", alias="THUMBNAIL_DIR")
thumbnail_size: int = Field(default=256, alias="THUMBNAIL_SIZE")
```

### Caching Behavior
- Thumbnails are cached on-disk and regenerated only when:
  1. File doesn't exist at expected path
  2. Forced regeneration via API endpoint
- Database stores `thumbnail_path` for each asset
- Service checks file existence before generating new thumbnail

## Methods to Regenerate Thumbnails

### Method 1: Delete Cache Directory (Simplest)

**Best for**: Regenerating ALL thumbnails

```bash
# Stop the service first (optional but recommended)
# Delete entire cache
rm -rf /tmp/thumbnails

# Restart service - thumbnails will regenerate on-demand when accessed
```

**Pros**:
- Simplest approach
- All thumbnails regenerated with correct orientation
- No risk of missing any thumbnails

**Cons**:
- Thumbnails regenerate lazily (only when requested)
- Temporary performance impact on first access

---

### Method 2: Delete Selective Thumbnails

**Best for**: Regenerating specific subsets

```bash
# Delete specific hash shard
rm -rf /tmp/thumbnails/0a

# Delete specific thumbnail
rm /tmp/thumbnails/0a/12345.jpg

# Delete all thumbnails older than X days
find /tmp/thumbnails -name "*.jpg" -mtime +7 -delete
```

---

### Method 3: API Endpoint for Single Asset (Existing)

**Best for**: Testing or regenerating individual images

**Endpoint**: `POST /api/v1/images/{asset_id}/thumbnail/generate`

**Implementation**: `/export/workspace/image-search/image-search-service/src/image_search_service/api/routes/images.py:205-266`

**Example**:
```bash
# Regenerate thumbnail for asset ID 12345
curl -X POST http://localhost:8000/api/v1/images/12345/thumbnail/generate

# Response:
# {
#   "assetId": 12345,
#   "thumbnailPath": "/tmp/thumbnails/0a/12345.jpg",
#   "width": 256,
#   "height": 192,
#   "status": "generated"
# }
```

**Behavior**:
- Forces regeneration even if thumbnail exists
- Updates database with new dimensions
- Returns thumbnail metadata
- Handles errors gracefully (404 if image missing, 500 on failure)

---

### Method 4: API Endpoint for Batch Regeneration

**Best for**: Regenerating all thumbnails via API

**Current Status**: No batch regeneration endpoint exists

**Workaround Script**:
```bash
#!/bin/bash
# regenerate_all_thumbnails.sh

# Get all asset IDs from database
ASSET_IDS=$(psql -d image_search -t -c "SELECT id FROM image_assets ORDER BY id;" | xargs)

# Regenerate each thumbnail
for asset_id in $ASSET_IDS; do
  echo "Regenerating thumbnail for asset $asset_id..."
  curl -X POST http://localhost:8000/api/v1/images/$asset_id/thumbnail/generate
  sleep 0.1  # Rate limiting
done

echo "Done!"
```

**Alternative with Python**:
```python
# regenerate_thumbnails.py
import asyncio
import httpx
from sqlalchemy import select
from image_search_service.db.models import ImageAsset
from image_search_service.db.session import get_async_session

async def regenerate_all():
    async with get_async_session() as db:
        result = await db.execute(select(ImageAsset.id))
        asset_ids = [row[0] for row in result.all()]

    async with httpx.AsyncClient() as client:
        for asset_id in asset_ids:
            print(f"Regenerating asset {asset_id}...")
            response = await client.post(
                f"http://localhost:8000/api/v1/images/{asset_id}/thumbnail/generate"
            )
            print(f"  Status: {response.status_code}")

if __name__ == "__main__":
    asyncio.run(regenerate_all())
```

---

### Method 5: Background Job for Session Thumbnails (Existing)

**Best for**: Regenerating thumbnails for a specific training session

**Endpoint**: `POST /api/v1/training/sessions/{session_id}/thumbnails`

**Implementation**: `/export/workspace/image-search/image-search-service/src/image_search_service/api/routes/training.py:623-674`

**Background Job**: `generate_thumbnails_batch()` in `/export/workspace/image-search/image-search-service/src/image_search_service/queue/thumbnail_jobs.py`

**Behavior**:
- Enqueues background job via RQ
- Processes all assets in a training session
- Skips thumbnails that already exist (not suitable for regeneration)

**Limitation**: This endpoint SKIPS existing thumbnails, so it won't regenerate them. Not suitable for fixing orientation issues.

---

## Recommended Approach

### For Development/Testing

**Option A: Quick Full Regeneration**
```bash
# 1. Delete cache
rm -rf /tmp/thumbnails

# 2. Access any image in UI
# Thumbnails regenerate on-demand with correct orientation
```

**Option B: Verify Single Image First**
```bash
# 1. Test with one asset
curl -X POST http://localhost:8000/api/v1/images/12345/thumbnail/generate

# 2. If successful, proceed with full regeneration
rm -rf /tmp/thumbnails
```

### For Production

**Staged Approach**:
```bash
# 1. Backup existing thumbnails (optional)
cp -r /tmp/thumbnails /tmp/thumbnails.backup

# 2. Delete cache during low-traffic period
rm -rf /tmp/thumbnails

# 3. Pre-warm cache with batch script (optional)
# Run regenerate_all_thumbnails.sh in background

# 4. Monitor logs for regeneration activity
tail -f /path/to/app.log | grep "thumbnail"
```

---

## Implementation Details

### Thumbnail Generation Flow

From `/export/workspace/image-search/image-search-service/src/image_search_service/services/thumbnail_service.py:94-146`:

```python
def generate_thumbnail(self, original_path: str, asset_id: int) -> tuple[str, int, int]:
    """Generate thumbnail for image with EXIF orientation handling."""

    # 1. Open original image
    with Image.open(original) as img:
        # 2. Apply EXIF orientation transformation (FIX APPLIED HERE)
        img = ImageOps.exif_transpose(img) or img

        # 3. Convert RGBA to RGB for JPEG
        if img.mode in ("RGBA", "LA", "P"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])
            img = background

        # 4. Resize maintaining aspect ratio
        img.thumbnail((self.thumbnail_size, self.thumbnail_size), Image.Resampling.LANCZOS)

        # 5. Save as JPEG
        img.save(thumb_path, "JPEG", quality=85, optimize=True)
```

**Key Fix**: `ImageOps.exif_transpose(img)` now correctly handles smartphone photo rotation.

### On-Demand Regeneration

From `/export/workspace/image-search/image-search-service/src/image_search_service/api/routes/images.py:64-132`:

```python
@router.get("/{asset_id}/thumbnail")
async def get_thumbnail(asset_id: int, db: AsyncSession = Depends(get_db)):
    """Serve thumbnail, generating on-the-fly if missing."""

    # Check if cached thumbnail exists
    if asset.thumbnail_path and Path(asset.thumbnail_path).exists():
        thumb_path = Path(asset.thumbnail_path)
    else:
        # Generate on-demand
        thumb_path_str, width, height = thumb_service.generate_thumbnail(...)
        asset.thumbnail_path = thumb_path_str
        await db.commit()
```

**Behavior**: Thumbnails regenerate automatically if file is missing (e.g., after cache deletion).

---

## Testing Verification

### Verify Orientation Fix

```bash
# 1. Find an asset with known orientation issue
ASSET_ID=12345

# 2. Regenerate thumbnail
curl -X POST http://localhost:8000/api/v1/images/$ASSET_ID/thumbnail/generate

# 3. Fetch thumbnail and verify orientation
curl http://localhost:8000/api/v1/images/$ASSET_ID/thumbnail -o test_thumb.jpg

# 4. Open test_thumb.jpg and verify it's correctly oriented
```

### Monitor Regeneration Progress

```bash
# Watch thumbnail directory size
watch -n 1 'du -sh /tmp/thumbnails'

# Count thumbnails
watch -n 1 'find /tmp/thumbnails -name "*.jpg" | wc -l'

# Monitor API logs
tail -f /path/to/app.log | grep "thumbnail"
```

---

## Future Improvements

### Recommended: Add Bulk Regeneration Endpoint

**Proposed API**:
```
POST /api/v1/admin/thumbnails/regenerate
{
  "force": true,
  "assetIds": [1, 2, 3],  // Optional: specific assets
  "fromDate": "2024-01-01",  // Optional: filter by date
  "toDate": "2024-12-31"
}
```

**Implementation Location**: `/export/workspace/image-search/image-search-service/src/image_search_service/api/routes/admin.py`

**Background Job**: Create `regenerate_thumbnails_batch()` in `queue/thumbnail_jobs.py`

### Recommended: Add CLI Command

**Proposed Makefile target**:
```makefile
thumbnails-regenerate: ## Regenerate all thumbnails
	uv run python -m image_search_service.scripts.cli thumbnails regenerate --force
```

**Implementation**: Add `thumbnails` command group to `scripts/cli.py`

---

## Related Files

- **Service**: `src/image_search_service/services/thumbnail_service.py`
- **API Routes**: `src/image_search_service/api/routes/images.py`
- **Background Jobs**: `src/image_search_service/queue/thumbnail_jobs.py`
- **Config**: `src/image_search_service/core/config.py`
- **Database Model**: `src/image_search_service/db/models.py` (ImageAsset.thumbnail_path)

---

## Appendix: Cache Statistics

Current cache status (as of research):
```bash
$ ls -la /tmp/thumbnails | head -20
total 1160
drwxrwxr-x 256 mac  mac    4096 Dec 27 20:36 .
drwxrwxrwt  57 root root 139264 Dec 27 20:39 ..
drwxrwxr-x   2 mac  mac    4096 Dec 27 14:11 00
drwxrwxr-x   2 mac  mac    4096 Dec 27 17:05 01
drwxrwxr-x   2 mac  mac    4096 Dec 27 19:08 02
...
```

- 256 shard directories (00-ff)
- Thumbnails distributed across shards
- Ready for cache deletion and regeneration
