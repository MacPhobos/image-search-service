# Training System Usage Guide

Quick reference for using the background training system.

## Architecture Overview

```
API Request → TrainingService.enqueue_training()
              ↓
              1. Discover assets in selected subdirectories
              2. Create TrainingJob records
              3. Update session status to RUNNING
              4. Enqueue train_session() to RQ (high priority)
              ↓
RQ Worker → train_session(session_id)
            ↓
            Process jobs in batches
            ├── train_batch(session_id, asset_ids, batch_num)
            │   ├── train_single_asset(job_id, asset_id, session_id)
            │   │   ├── Load asset from DB
            │   │   ├── Generate embedding with OpenCLIP
            │   │   ├── Store in Qdrant
            │   │   ├── Create TrainingEvidence record
            │   │   └── Update TrainingJob status
            │   ├── Check for cancellation
            │   └── Update progress
            └── Return summary stats
```

## Basic Workflow

### 1. Create Training Session

```python
from image_search_service.services.training_service import TrainingService
from image_search_service.api.training_schemas import TrainingSessionCreate

service = TrainingService()

# Create session with subdirectories
session_data = TrainingSessionCreate(
    name="My Training Session",
    rootPath="/path/to/images",
    subdirectories=[
        "/path/to/images/vacation",
        "/path/to/images/family"
    ],
    config={
        "recursive": True,
        "extensions": ["jpg", "jpeg", "png"],
        "batch_size": 32
    }
)

session = await service.create_session(db, session_data)
```

### 2. Start Training

```python
# Enqueue training job
rq_job_id = await service.enqueue_training(db, session.id)

# Session is now in RUNNING state
# RQ worker will process in background
```

### 3. Monitor Progress

```python
# Get current progress
progress = await service.get_session_progress(db, session.id)

print(f"Status: {progress.status}")
print(f"Progress: {progress.progress.percentage}%")
print(f"Processed: {progress.progress.current}/{progress.progress.total}")
print(f"ETA: {progress.progress.etaSeconds} seconds")
print(f"Rate: {progress.progress.imagesPerMinute} images/min")

# Job status breakdown
print(f"Pending: {progress.jobsSummary.pending}")
print(f"Running: {progress.jobsSummary.running}")
print(f"Completed: {progress.jobsSummary.completed}")
print(f"Failed: {progress.jobsSummary.failed}")
```

### 4. Cancel or Pause

```python
from image_search_service.db.models import SessionStatus

# Cancel training
session.status = SessionStatus.CANCELLED.value
await db.commit()
# Worker will stop after current batch completes

# Or pause training
session.status = SessionStatus.PAUSED.value
await db.commit()
# Can resume later
```

### 5. Resume Paused Session

```python
# Change status back to RUNNING
session.status = SessionStatus.RUNNING.value
await db.commit()

# Re-enqueue pending jobs
rq_job_id = await service.enqueue_training(db, session.id)
```

## Direct Job Enqueueing (Advanced)

```python
from image_search_service.queue.worker import get_queue, QUEUE_HIGH
from image_search_service.queue.training_jobs import train_session

# Get high-priority queue
queue = get_queue(QUEUE_HIGH)

# Enqueue training session
job = queue.enqueue(
    train_session,
    session_id=123,
    job_timeout="1h",  # 1 hour timeout
    result_ttl=3600     # Keep result for 1 hour
)

print(f"Job ID: {job.id}")
print(f"Job status: {job.get_status()}")
```

## Queue Priorities

```python
from image_search_service.queue.worker import (
    QUEUE_HIGH,     # "training-high" - User-initiated training
    QUEUE_NORMAL,   # "training-normal" - Scheduled tasks
    QUEUE_LOW,      # "training-low" - Thumbnails, cleanup
    get_queue
)

# Enqueue to specific priority
high_queue = get_queue(QUEUE_HIGH)
normal_queue = get_queue(QUEUE_NORMAL)
low_queue = get_queue(QUEUE_LOW)
```

## Error Handling

### Check for Failed Jobs

```python
from image_search_service.db.models import JobStatus

# Get failed jobs
query = (
    select(TrainingJob)
    .where(TrainingJob.session_id == session_id)
    .where(TrainingJob.status == JobStatus.FAILED.value)
)
failed_jobs = await db.execute(query)

for job in failed_jobs.scalars():
    print(f"Asset {job.asset_id}: {job.error_message}")
```

### Retry Failed Jobs

```python
# Get pending assets (includes failed)
pending_asset_ids = await service.get_pending_assets(db, session_id)

# Create new jobs for pending assets
job_count = await service.create_training_jobs(db, session_id, pending_asset_ids)

# Re-enqueue
rq_job_id = await service.enqueue_training(db, session_id)
```

## Asset Discovery

```python
from image_search_service.services.asset_discovery import AssetDiscoveryService

discovery = AssetDiscoveryService(extensions=["jpg", "png"])

# Discover all assets for a session
assets = await discovery.discover_assets(db, session_id)
print(f"Found {len(assets)} images")

# Count images without creating records
count = await discovery.count_images_in_directory("/path/to/images", recursive=True)
print(f"Directory contains {count} images")
```

## Progress Tracking (Internal)

```python
from image_search_service.queue.progress import ProgressTracker

tracker = ProgressTracker(session_id)

# In RQ worker (sync context)
db_session = get_sync_session()

# Update progress
tracker.update_progress(db_session, processed=100, failed=5)

# Check for cancellation
if tracker.should_stop(db_session):
    print("Session was cancelled or paused")
    return

# Get current stats
progress = tracker.get_current_progress(db_session)
print(f"Progress: {progress['percentage']}%")

# Calculate ETA
eta = tracker.calculate_eta(start_time, processed=100, total=500)
print(f"ETA: {eta}")

# Calculate rate
rate = tracker.calculate_rate(start_time, processed=100)
print(f"Rate: {rate} images/min")
```

## Configuration

### Environment Variables

```bash
# Batch size for training
TRAINING_BATCH_SIZE=32

# RQ job timeout (default: 1 hour)
# Set in code: queue.enqueue(..., job_timeout="2h")
```

### Session Config

```python
config = {
    "recursive": True,              # Scan subdirectories recursively
    "extensions": ["jpg", "png"],   # File extensions to process
    "batch_size": 32                # Number of images per batch
}
```

## Monitoring

### Check Worker Status

```bash
# In terminal
make worker

# Should show:
# Processing queues in order: ['training-high', 'training-normal', 'training-low', 'default']
```

### Check Queue Status (Redis)

```python
from image_search_service.queue.worker import get_queue, QUEUE_HIGH

queue = get_queue(QUEUE_HIGH)

print(f"Queue: {queue.name}")
print(f"Pending jobs: {len(queue)}")
print(f"Failed jobs: {queue.failed_job_registry.count}")
print(f"Started jobs: {queue.started_job_registry.count}")
```

### Check Evidence Records

```python
from image_search_service.db.models import TrainingEvidence

# Get evidence for a session
query = (
    select(TrainingEvidence)
    .where(TrainingEvidence.session_id == session_id)
    .order_by(TrainingEvidence.created_at.desc())
)
evidence = await db.execute(query)

for record in evidence.scalars():
    print(f"Asset {record.asset_id}:")
    print(f"  Model: {record.model_name} {record.model_version}")
    print(f"  Device: {record.device}")
    print(f"  Time: {record.processing_time_ms}ms")
    print(f"  Checksum: {record.embedding_checksum}")
    if record.error_message:
        print(f"  Error: {record.error_message}")
```

## Common Patterns

### Batch Training with Progress Updates

```python
async def train_with_monitoring(session_id: int):
    """Train session with real-time progress monitoring."""
    service = TrainingService()

    # Start training
    rq_job_id = await service.enqueue_training(db, session_id)

    # Monitor progress
    while True:
        progress = await service.get_session_progress(db, session_id)

        if progress.status in ["completed", "failed", "cancelled"]:
            break

        print(f"{progress.progress.percentage}% - "
              f"{progress.progress.current}/{progress.progress.total} - "
              f"ETA: {progress.progress.etaSeconds}s")

        await asyncio.sleep(5)  # Check every 5 seconds

    return progress
```

### Selective Retraining

```python
async def retrain_failed_only(session_id: int):
    """Retry only failed jobs."""
    service = TrainingService()

    # Get failed asset IDs
    query = (
        select(TrainingJob.asset_id)
        .where(TrainingJob.session_id == session_id)
        .where(TrainingJob.status == JobStatus.FAILED.value)
    )
    result = await db.execute(query)
    failed_asset_ids = list(result.scalars())

    if not failed_asset_ids:
        print("No failed jobs to retry")
        return

    # Create new jobs for failed assets
    job_count = await service.create_training_jobs(db, session_id, failed_asset_ids)

    # Enqueue
    rq_job_id = await service.enqueue_training(db, session_id)

    print(f"Retrying {job_count} failed jobs")
```

## Performance Tips

1. **Batch Size**: Larger batches = fewer DB updates, but less granular progress
   - Default: 32 images
   - Adjust based on image size and available memory

2. **Queue Priority**: Use `QUEUE_HIGH` for user-initiated training
   - Background tasks use `QUEUE_NORMAL` or `QUEUE_LOW`

3. **Worker Count**: Run multiple workers for parallel processing
   ```bash
   # Terminal 1
   make worker

   # Terminal 2
   make worker

   # Terminal 3
   make worker
   ```

4. **Database Connections**: Workers use sync connections
   - Pool size configured in `get_sync_engine()`

5. **Progress Update Frequency**: Updated after each batch
   - More batches = more frequent updates
   - Balance between responsiveness and DB load

## Troubleshooting

### Jobs Not Processing

1. Check worker is running: `make worker`
2. Check Redis connection: `redis-cli ping`
3. Check queue has jobs:
   ```python
   queue = get_queue(QUEUE_HIGH)
   print(len(queue))  # Should be > 0
   ```

### High Memory Usage

1. Reduce batch size: `TRAINING_BATCH_SIZE=16`
2. Process fewer queues simultaneously
3. Restart worker periodically

### Slow Processing

1. Check device: CPU vs GPU (logged in evidence)
2. Check image sizes (resize large images)
3. Increase worker count for parallelization

### Failed Jobs

1. Check error messages in `TrainingJob.error_message`
2. Check evidence records for details
3. Common issues:
   - File not found (moved/deleted)
   - Corrupted image (cannot open)
   - Out of memory (reduce batch size)

## API Integration Example

```python
# In API route
@router.post("/training/sessions/{session_id}/start")
async def start_training(
    session_id: int,
    db: AsyncSession = Depends(get_db)
):
    service = TrainingService()

    # Enqueue training
    rq_job_id = await service.enqueue_training(db, session_id)

    return {
        "sessionId": session_id,
        "rqJobId": rq_job_id,
        "status": "running",
        "message": "Training started successfully"
    }
```
