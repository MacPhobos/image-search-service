# File Watcher Quick Start Guide

## Overview

The file watcher automatically detects new images added to configured directories and creates database records for them. Optionally, it can also enqueue detected images for training.

## Quick Start

### 1. Enable File Watching

Add to your `.env` file:

```bash
WATCH_ENABLED=true
IMAGE_ROOT_DIR=/path/to/your/images
WATCH_AUTO_TRAIN=false  # Set to true to auto-train
```

### 2. Start the Service

```bash
# Terminal 1: Start backend services
make db-up
make migrate
make dev

# Terminal 2: Start background worker
make worker
```

### 3. Verify Watcher is Running

```bash
curl http://localhost:8000/api/v1/system/watcher/status
```

Expected response:
```json
{
  "enabled": true,
  "running": true,
  "watch_paths": ["/path/to/your/images"],
  "file_watcher_active": true,
  "auto_train": false,
  "debounce_seconds": 1.0
}
```

### 4. Add a Test Image

```bash
cp test-image.jpg /path/to/your/images/
```

The watcher will:
1. Detect the new file within 1 second
2. Enqueue a background job
3. Create an ImageAsset record
4. (Optional) Enqueue for training if `WATCH_AUTO_TRAIN=true`

### 5. Verify Image was Detected

Check the database:
```bash
psql -d image_search -c "SELECT id, path, training_status FROM image_assets ORDER BY id DESC LIMIT 1;"
```

Or use the API:
```bash
curl http://localhost:8000/api/v1/assets?page=1&pageSize=10
```

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WATCH_ENABLED` | `false` | Enable/disable file watching |
| `IMAGE_ROOT_DIR` | `""` | Directory to watch for new images |
| `WATCH_DEBOUNCE_SECONDS` | `1.0` | Delay before processing (prevents duplicates) |
| `WATCH_AUTO_TRAIN` | `false` | Auto-enqueue detected images for training |

### Debounce Explanation

The debounce delay prevents duplicate processing when files are being copied or modified. For example:
- File copy starts → Event triggered
- File copy in progress → Multiple events triggered
- File copy completes → Final event triggered
- **After 1 second of no events** → Image is processed once

Increase `WATCH_DEBOUNCE_SECONDS` if you're copying large files over slow network.

## Manual Scanning

### Incremental Scan Endpoint

Use this to scan a directory without enabling continuous watching:

```bash
curl -X POST "http://localhost:8000/api/v1/training/scan/incremental?directory=/path/to/images&auto_train=false"
```

Response:
```json
{
  "directory": "/path/to/images",
  "discovered": 150,
  "created": 42,
  "skipped": 108
}
```

- **discovered**: Total image files found
- **created**: New ImageAsset records created
- **skipped**: Files already in database

### Batch Import Workflow

For large batch imports:

1. **Disable auto-train** to avoid overwhelming the queue:
   ```bash
   export WATCH_AUTO_TRAIN=false
   ```

2. **Copy all files**:
   ```bash
   rsync -av /source/images/ /path/to/your/images/
   ```

3. **Wait for processing** (check worker logs)

4. **Verify all images detected**:
   ```bash
   curl "http://localhost:8000/api/v1/training/scan/incremental?directory=/path/to/your/images"
   ```

5. **Create training session** to process all images:
   ```bash
   curl -X POST http://localhost:8000/api/v1/training/sessions \
     -H "Content-Type: application/json" \
     -d '{"name": "Batch Import", "root_path": "/path/to/your/images"}'
   ```

## API Endpoints

### Watcher Control

#### Get Status
```bash
GET /api/v1/system/watcher/status
```

#### Start Watcher
```bash
POST /api/v1/system/watcher/start
```

#### Stop Watcher
```bash
POST /api/v1/system/watcher/stop
```

### Scanning

#### Incremental Scan
```bash
POST /api/v1/training/scan/incremental?directory=/path&auto_train=false
```

## Troubleshooting

### Watcher Not Starting

**Problem**: Status shows `running: false`

**Solutions**:
1. Check `WATCH_ENABLED=true` in environment
2. Verify `IMAGE_ROOT_DIR` is set and directory exists
3. Check application logs for errors
4. Ensure directory has read permissions

### Images Not Detected

**Problem**: New images not appearing in database

**Solutions**:
1. Check file extension is supported (.jpg, .jpeg, .png, .gif, .webp, .bmp)
2. Verify RQ worker is running (`make worker`)
3. Check worker logs for errors
4. Increase debounce time if files are large
5. Try manual scan to verify path is correct

### Duplicate Processing

**Problem**: Same image processed multiple times

**Solutions**:
1. Increase `WATCH_DEBOUNCE_SECONDS` (try 2.0 or 5.0)
2. Check for file system events being triggered multiple times
3. Database constraint prevents duplicate paths (safe)

### Performance Issues

**Problem**: High CPU/memory usage

**Solutions**:
1. Reduce watch directory scope (don't watch entire filesystem)
2. Use manual incremental scans instead of continuous watching
3. Disable recursive watching if not needed
4. Set `WATCH_AUTO_TRAIN=false` for large imports

## Best Practices

### Development

- Use `WATCH_AUTO_TRAIN=false` during development
- Set `WATCH_DEBOUNCE_SECONDS=0.5` for faster feedback
- Watch a small test directory

### Production

- Enable watching only if needed (network shares may not work well)
- Use incremental scans for scheduled imports
- Set `WATCH_DEBOUNCE_SECONDS=2.0` for reliability
- Monitor worker queue length
- Consider using manual scans for large batch imports

### Large Datasets

For datasets > 10,000 images:

1. **Disable continuous watching**:
   ```bash
   WATCH_ENABLED=false
   ```

2. **Use scheduled incremental scans** (via cron or similar):
   ```bash
   0 * * * * curl -X POST "http://localhost:8000/api/v1/training/scan/incremental?directory=/path"
   ```

3. **Process in batches** using training sessions

## Examples

### Example 1: Development Setup

```bash
# .env
WATCH_ENABLED=true
IMAGE_ROOT_DIR=/Users/dev/test-images
WATCH_DEBOUNCE_SECONDS=0.5
WATCH_AUTO_TRAIN=false
```

Add images:
```bash
cp ~/Downloads/*.jpg /Users/dev/test-images/
```

Check results:
```bash
curl http://localhost:8000/api/v1/assets?page=1
```

### Example 2: Production Setup

```bash
# .env
WATCH_ENABLED=false  # Use manual scans
IMAGE_ROOT_DIR=/mnt/production-images
```

Scheduled scan (crontab):
```bash
# Scan every hour
0 * * * * curl -X POST "http://localhost:8000/api/v1/training/scan/incremental?directory=/mnt/production-images&auto_train=true"
```

### Example 3: Dropbox Integration

```bash
# .env
WATCH_ENABLED=true
IMAGE_ROOT_DIR=/Users/user/Dropbox/Images
WATCH_DEBOUNCE_SECONDS=5.0  # Dropbox sync can be slow
WATCH_AUTO_TRAIN=true
```

Drop images into Dropbox folder → Automatically detected and trained

## Monitoring

### Check Watcher Health

```bash
# Status check
curl http://localhost:8000/api/v1/system/watcher/status | jq .

# Recent assets
curl http://localhost:8000/api/v1/assets?page=1&pageSize=10 | jq .

# Worker status (if using RQ dashboard)
open http://localhost:9181
```

### Logs

Watch application logs for watcher events:
```bash
# API server logs
tail -f logs/api.log | grep -i watcher

# Worker logs
tail -f logs/worker.log | grep -i "process_new_image"
```

## Security Notes

- **Path validation**: Only configured directories are watched
- **Extension filtering**: Only image files are processed
- **Directory traversal protection**: Paths are validated before scanning
- **No user uploads**: File watching is for server-side directories only

## Support

For issues or questions:
1. Check application logs
2. Verify configuration with `/system/watcher/status`
3. Test with manual incremental scan
4. Review worker logs for background job errors
