# Test RQ Worker - Reproduction Script

This directory contains a standalone test script to reproduce and debug the RQ worker crash issue on macOS.

## Problem

On macOS, the RQ worker crashes with:
```
rq.job - DEBUG - Job <id>: handling failure: Work-horse terminated unexpectedly;
```

This happens because:
1. RQ uses `spawn()` on macOS (not `fork()`), creating fresh Python interpreters
2. The subprocess Metal compiler service fails to initialize properly
3. The environment variable `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` is a workaround but masks the deeper issue

## Solution: Isolated Test Reproduction

The test script allows you to:
- **Quickly reproduce** the crash in isolation without affecting production
- **Gather DEBUG information** about GPU initialization and memory
- **Iterate on fixes** by restarting the worker and re-running the test
- **Compare x86 vs macOS** behavior side-by-side

## Files

- **`test_rq_worker.py`** - Main test script (Python)
- **`run_test_worker.sh`** - Helper script to start worker with proper environment
- **`README_TEST_RQ_WORKER.md`** - This file

## Quick Start

### Prerequisites

```bash
# Make sure dependencies are installed
uv sync --dev

# Start Redis and Qdrant locally
make db-up

# Migrations (if needed)
make migrate
```

### Run the Test

**Step 1: Terminal 1 - Start the test worker**

```bash
# x86 (Linux/standard)
make worker

# OR macOS (with fork safety disabled - THIS IS THE CASE WE'RE DEBUGGING)
OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES make worker

# OR use the helper script (automatically detects macOS)
./scripts/run_test_worker.sh
```

You should see output like:
```
2026-01-05 21:15:23 - image_search_service.queue.worker - INFO - Starting RQ worker with priority queues
2026-01-05 21:15:23 - image_search_service.queue.worker - INFO - Redis URL: redis://localhost:6379/0
2026-01-05 21:15:23 - image_search_service.queue.worker - INFO - Processing queues in order: ['training-high', ...]
```

**Step 2: Terminal 2 - Run the test client**

```bash
# Enqueue a job and monitor execution
uv run python scripts/test_rq_worker.py --mode client

# OR with debug logging
uv run python scripts/test_rq_worker.py --mode client --debug
```

You should see output like:
```
════════════════════════════════════════════════════════════════════════
Environment Information
════════════════════════════════════════════════════════════════════════
Platform: Darwin 25.1.0
Python: 3.12.0
PyTorch version: 2.x.x
CUDA available: False
MPS available: True

════════════════════════════════════════════════════════════════════════
Test RQ Worker - Initialization
════════════════════════════════════════════════════════════════════════
✓ Connected to Redis: redis://localhost:6379/1
✓ Created test database: sqlite:///:memory:
✓ Using Qdrant collection: test_rq_worker_assets
✓ Created training session: ID=1
✓ Created image asset: ID=1, path=Charlize Theron_30.jpg
✓ Created training job: ID=1
✓ Enqueued job: abc123def456...
  Status: queued
  Queue: training-high

Waiting for worker to process job...
Make sure the worker is running in another terminal:
  make worker  (or with OBJC_DISABLE_INITIALIZE_FORK_SAFETY on macOS)

════════════════════════════════════════════════════════════════════════
Monitoring Job Execution
════════════════════════════════════════════════════════════════════════
GPU Memory before job: 0 MB
Timeout: 120 seconds
Polling job status (every 2 seconds)...
[002s] Job status: started
[015s] Job status: finished
════════════════════════════════════════════════════════════════════════
✓ JOB SUCCEEDED
════════════════════════════════════════════════════════════════════════
Elapsed time: 17.42s
Result: {'status': 'completed', 'asset_id': 1, ...}

════════════════════════════════════════════════════════════════════════
Test Results
════════════════════════════════════════════════════════════════════════
Job ID: abc123def456...
Result: ✓ SUCCESS
════════════════════════════════════════════════════════════════════════
```

## What Gets Tested

### Isolated Environment
- **Redis**: Uses test database (`redis://localhost:6379/1` by default)
- **Database**: SQLite in-memory (no external dependencies)
- **Qdrant**: Uses `test_rq_worker_assets` collection (separate from production)
- **Sample Image**: `tests/sample-images/Charlize Theron_30.jpg`

### Job Pipeline
1. Creates a test training session in database
2. Creates an image asset record pointing to the sample image
3. Creates a training job record
4. Enqueues `train_single_asset()` job via RQ to `training-high` queue
5. Monitors job execution with:
   - Status changes (queued → started → finished)
   - Elapsed time and GPU memory
   - Full error messages if job fails
   - Worker process info if available

### Debug Information Captured
- Platform info (macOS/Linux, kernel version)
- Python version and PyTorch version
- GPU/MPS availability and capabilities
- CUDA/MPS device name and memory
- Environment variables (OBJC_DISABLE_INITIALIZE_FORK_SAFETY, WORKER_DEBUG)
- RQ queue configuration
- Job execution timeline
- GPU memory before/after
- Error stack traces on failure

## Interpreting Results

### Success ✓
```
✓ JOB SUCCEEDED
Elapsed time: 17.42s
Result: {'status': 'completed', ...}
```

The job completed successfully. The embedding was generated and stored in Qdrant.

### Failure ✗
```
✗ JOB FAILED
Elapsed time: 2.15s
Error message: Work-horse terminated unexpectedly; ...
```

The worker crashed. Check the worker terminal (Terminal 1) for the full stack trace. This is the case we're debugging on macOS.

### Timeout
```
Job did not complete within 120 seconds
```

The job didn't finish in time. Check if the worker is running and healthy.

## Common Issues

### "Sample image not found"
```
FileNotFoundError: Sample image not found: tests/sample-images/Charlize Theron_30.jpg
```

Make sure you're running the script from the project root:
```bash
cd /path/to/image-search-service
uv run python scripts/test_rq_worker.py --mode client
```

### "Failed to connect to Redis"
```
✗ Failed to connect to Redis: Connection refused
```

Make sure Redis is running:
```bash
make db-up  # Start Redis and Postgres
```

### "Worker not found / Job timeout"
```
[120s] Job did not complete within 120 seconds
```

Make sure the worker is running in another terminal:
```bash
# Terminal 1
make worker
```

## Advanced Usage

### Use Different Redis Database
```bash
# Use Redis DB 2 instead of default (1)
REDIS_TEST_DB=2 uv run python scripts/test_rq_worker.py --mode client
```

### Enable Full Debug Logging
```bash
# Captures DEBUG-level logs from all modules
uv run python scripts/test_rq_worker.py --mode client --debug
```

### Use Custom Qdrant Collection
```bash
# Use a specific Qdrant collection name
QDRANT_TEST_COLLECTION=my_test_collection uv run python scripts/test_rq_worker.py --mode client
```

### Run Worker with Helper Script
```bash
# Automatically handles OBJC_DISABLE_INITIALIZE_FORK_SAFETY on macOS
./scripts/run_test_worker.sh
```

## Workflow for Debugging macOS Crashes

1. **Setup** (one-time):
   ```bash
   make db-up
   make migrate
   ```

2. **Start worker** (Terminal 1):
   ```bash
   ./scripts/run_test_worker.sh  # or "make worker" on x86
   ```

3. **Run test** (Terminal 2):
   ```bash
   uv run python scripts/test_rq_worker.py --mode client --debug
   ```

4. **Iterate**:
   - If job crashes: Check worker output (Terminal 1), identify the issue
   - Apply fix to `src/image_search_service/queue/worker.py` or `services/embedding.py`
   - Stop and restart worker (Ctrl+C, then re-run step 2)
   - Re-run test (step 3)
   - Repeat until job succeeds

5. **Verify**:
   - Once test passes consistently, run production tests:
     ```bash
     make test
     ```

## No Production Code Changes

This test script:
- ✅ Uses isolated test environment (separate Redis DB, SQLite)
- ✅ Uses sample image from test fixtures
- ✅ Creates test-only database records
- ✅ Uses test-only Qdrant collection
- ✅ Does NOT modify any production code
- ✅ Does NOT affect running API or workers
- ✅ Can be deleted or kept for future regression testing

## Architecture

The test script works by:

1. **Client Process** (Python script):
   - Sets up isolated test environment
   - Creates test data in SQLite DB
   - Enqueues job via RQ to test Redis DB
   - Monitors job status with polling

2. **Worker Process** (RQ worker):
   - Connects to test Redis DB
   - Fetches job from queue
   - Loads real OpenCLIP model in work-horse subprocess
   - Processes sample image with actual GPU inference
   - Updates test SQLite database with results
   - Stores vectors in test Qdrant collection

3. **Communication**:
   - Client and worker communicate via Redis
   - Results stored in SQLite (for test DB records)
   - Vectors stored in Qdrant (for search functionality)
   - No external API calls needed

## See Also

- `CLAUDE.md` - Project setup and architecture
- `src/image_search_service/queue/worker.py` - RQ worker implementation
- `src/image_search_service/queue/training_jobs.py` - Training job definitions
- `src/image_search_service/services/embedding.py` - OpenCLIP embedding service
