# Test RQ Worker - Quick Reference Card

## 🚀 Start Testing in 2 Minutes

### Terminal 1: Start Worker
```bash
# macOS (the case we're debugging)
./scripts/run_test_worker.sh

# OR x86/Linux
make worker
```

### Terminal 2: Run Test
```bash
# Basic test
uv run python scripts/test_rq_worker.py --mode client

# With debug logging
uv run python scripts/test_rq_worker.py --mode client --debug
```

## 📊 Interpreting Results

### ✓ Success
```
✓ JOB SUCCEEDED
Elapsed time: 17.42s
GPU Memory delta: +512 MB
```
→ Everything works! Job completed successfully.

### ✗ Failure (macOS Crash)
```
✗ JOB FAILED
Error message: Work-horse terminated unexpectedly;
```
→ This is the bug we're debugging. Check worker output (Terminal 1) for crash details.

### ⏱️ Timeout
```
Job did not complete within 120 seconds
```
→ Make sure worker is running in Terminal 1.

## 🛠️ Debugging Workflow

### 1. Setup (one-time)
```bash
make db-up     # Start Redis and Postgres
make migrate   # Run migrations
```

### 2. Reproduce Crash
```bash
# Terminal 1
./scripts/run_test_worker.sh

# Terminal 2
uv run python scripts/test_rq_worker.py --mode client --debug
```

### 3. Apply Fix
Edit: `src/image_search_service/queue/worker.py` or `src/image_search_service/services/embedding.py`

### 4. Test Again
```bash
# Terminal 1: Stop (Ctrl+C) and restart worker
./scripts/run_test_worker.sh

# Terminal 2: Re-run test
uv run python scripts/test_rq_worker.py --mode client --debug
```

### 5. Repeat until ✓ Success

## 📁 Key Files

```
scripts/test_rq_worker.py              # Main test script
scripts/run_test_worker.sh             # Worker startup helper
scripts/README_TEST_RQ_WORKER.md       # Full documentation
tests/sample-images/Charlize Theron_30.jpg  # Test image
```

## 🔍 Debug Information Captured

- Platform (macOS/Linux)
- Python & PyTorch versions
- GPU availability (CUDA/MPS)
- GPU memory before/after
- Job execution timeline
- Full error stack traces
- Environment variables
- RQ configuration

## ⚙️ Advanced Options

```bash
# Use different Redis database
REDIS_TEST_DB=2 uv run python scripts/test_rq_worker.py --mode client

# Use different Qdrant collection
QDRANT_TEST_COLLECTION=my_test uv run python scripts/test_rq_worker.py --mode client

# Full debug output
LOG_LEVEL=DEBUG uv run python scripts/test_rq_worker.py --mode client --debug
```

## ✅ What Gets Tested

- ✓ Real OpenCLIP GPU inference (not mocked)
- ✓ RQ worker subprocess execution
- ✓ Model loading in work-horse process
- ✓ GPU memory management
- ✓ Database operations (create/update)
- ✓ Qdrant vector storage

## ❌ What's NOT Tested

- ❌ Production code (isolated test environment)
- ❌ API endpoints
- ❌ Main app configuration
- ❌ Other background jobs

## 📖 Full Documentation

For detailed documentation, see:
- `scripts/README_TEST_RQ_WORKER.md` - Comprehensive guide
- `TEST_RQ_WORKER_IMPLEMENTATION_SUMMARY.md` - Complete implementation details

## 🎯 Typical Debug Session (20 minutes)

```
[05 min] make db-up                    # Start services
[10 min] ./scripts/run_test_worker.sh  # Worker running
[02 min] uv run python scripts/test_rq_worker.py --mode client --debug  # Test 1
[03 min] Apply fix to worker code      # Based on crash info
[00 min] Ctrl+C and restart worker     # Pick up changes
[02 min] Re-run test                   # Verify fix
[02 min] make test                     # Full test suite
```

## 🚦 Common Commands

```bash
# Setup
make db-up

# Start worker
make worker
./scripts/run_test_worker.sh

# Run test
uv run python scripts/test_rq_worker.py --mode client
uv run python scripts/test_rq_worker.py --mode client --debug

# Stop worker
Ctrl+C

# View logs
tail -f /var/log/image-search-service.log

# Run all tests
make test

# Type check
make typecheck

# Format
make format
```

## 💡 Pro Tips

1. **Keep logs visible**: Arrange terminals side-by-side so you can see both worker output and test output simultaneously

2. **Check GPU memory**: If test passes locally but fails with larger batches, GPU memory might be the issue

3. **Test both platforms**: Run the test on both macOS (with OBJC_DISABLE_INITIALIZE_FORK_SAFETY) and x86 to compare behavior

4. **Compare timings**: Note how long the job takes on each platform—large differences might indicate GPU initialization overhead

5. **Use debug mode**: Always use `--debug` flag when investigating failures

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Sample image not found" | Run from project root: `cd image-search-service` |
| "Failed to connect to Redis" | Run: `make db-up` |
| "Job did not complete within 120 seconds" | Make sure worker is running in Terminal 1 |
| Import errors | Run: `uv sync --dev` |
| GPU memory not tracked | Normal on MPS (not exposed like CUDA) |

## 📞 Questions?

See `scripts/README_TEST_RQ_WORKER.md` for detailed documentation.

---

**Remember**: This is a **standalone test environment**. It won't affect your production code or data. Feel free to run it as many times as you need!
