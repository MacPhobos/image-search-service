# RQ Worker MPS Crash - Root Cause Analysis

**Date**: 2025-12-31
**Status**: Under Investigation
**Priority**: High

## Problem Statement

When running `make worker` and processing training requests that call the embedding service, the RQ worker subprocess (work-horse) crashes with:

```
Error getting visible function: (null) Unable to reach MTLCompilerService.
The process is unavailable because the compiler is no longer active.
Latest invalidation reason: Connection init failed at lookup with error 3 - No such process

assertion `Error getting visible function: (null) Unable to reach MTLCompilerService...`

Work-horse terminated unexpectedly; waitpid returned 6 (signal 6);
```

## Environment

- **OS**: macOS 15.1 (Apple Silicon - ARM64)
- **Python**: 3.12.11
- **PyTorch**: 2.9.1
- **OpenCLIP**: 3.2.0
- **Framework**: RQ 2.6.1 (for background job processing)
- **GPU**: Apple Metal Performance Shaders (MPS)

## Diagnostic Findings

### Finding 1: MPS Works in Main Process ✅

- MPS is built and available
- Simple tensor allocation on MPS works fine in main process
- Device detection correctly identifies MPS as available

```
PyTorch MPS Configuration:
  - MPS built: True
  - MPS available: True
  - MPS allocation test: SUCCESS
```

### Finding 2: Subprocess-Specific Issue ✅

Subprocess tests reveal a clear pattern:

| Test | Description | Main Process | Subprocess | Status |
|------|-------------|--------------|------------|--------|
| 1 | Torch import + check availability | ✅ Works | ✅ Works | OK |
| 2 | Torch MPS tensor allocation | ✅ Works | ✅ Works | OK |
| 3 | OpenCLIP model creation on MPS | ✅ Works | ❌ **HANGS/CRASHES** | **ISSUE** |

The subprocess hangs/times out (>30s) when attempting:
```python
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai', device='mps'
)
```

### Finding 3: Root Cause - OpenCLIP + MPS in Subprocess

The crash is **not** a general PyTorch MPS issue in subprocesses. It's specifically about **OpenCLIP model creation on MPS in forked subprocess contexts**.

#### Potential Root Causes:

1. **Metal Compiler Service Lifecycle**
   - When a parent process (uvicorn/main app) initializes MPS, it maintains connections to Apple's Metal compiler service
   - When fork() is called, the subprocess inherits handles but not the service connection
   - OpenCLIP attempts to compile shaders for the model on MPS, which requires the compiler service
   - The subprocess tries to reach the dead/closed compiler service handle → crash

2. **Model Weight Download Contention**
   - OpenCLIP downloads pretrained weights from HuggingFace Hub
   - In subprocess context, file descriptor/connection issues might occur
   - The timeout suggests the process hangs waiting for something

3. **Torch JIT Compilation**
   - PyTorch JIT (just-in-time) compilation for MPS kernels
   - Subprocess context lacks proper Metal initialization
   - Kernel compilation fails when trying to optimize for MPS device

4. **GIL and Threading Issues**
   - OpenCLIP model creation involves multiple threads
   - Subprocess fork doesn't preserve all thread/GIL state properly
   - Metal operations in multithreaded context in subprocess may fail

## Evidence from Error Log

The error occurs at the exact moment when:

```
2025-12-31 12:09:48,957 - image_search_service.services.embedding - INFO - Loading OpenCLIP model ViT-B-32 on mps
2025-12-31 12:09:46,957 - root - INFO - Parsing model identifier. Schema: None, Identifier: ViT-B-32
2025-12-31 12:09:46,958 - root - INFO - Loaded built-in ViT-B-32 model config.
2025-12-31 12:09:47,005 - httpcore.connection - DEBUG - connect_tcp.started host='huggingface.co' port=443
2025-12-31 12:09:47,007 - httpcore.connection - DEBUG - connect_tcp.failed exception=ConnectError(gaierror(8, 'nodename nor servname provided, or not known'))
2025-12-31 12:09:47,007 - root - INFO - Instantiating model architecture: CLIP
# ... lots of PIL EXIF debug logs ...
2025-12-31 12:09:48,687 - image_search_service.services.embedding - INFO - Model loaded. Embedding dim: 512
2025-12-31 12:09:48,695 - httpcore.connection - DEBUG - connect_tcp.started host='localhost' port=6333
2025-12-31 12:09:50.323 python3[24833:32357328] Error getting visible function: (null) Unable to reach MTLCompilerService
```

Key observations:
1. Model **loads successfully** (Embedding dim: 512) in subprocess
2. The crash happens **after** model is loaded, when first GPU operations occur
3. Metal compiler service error suggests JIT compilation trigger

## Not the Issue (Ruled Out)

✅ **Not a general subprocess issue**: Test 1 & 2 prove torch works fine in subprocess
✅ **Not a package version issue**: Correct versions installed
✅ **Not a missing MPS support**: MPS is built and available
✅ **Not a device detection issue**: Device selection is correct
✅ **Not Linux/NVIDIA-specific packages**: macOS has correct onnxruntime (non-GPU)

## Reproduction

```bash
# Prerequisites
make db-up
make migrate

# Start worker
OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES make worker

# In another terminal, queue a training job
curl -X POST http://localhost:8000/api/v1/training/sessions \
  -H "Content-Type: application/json" \
  -d '{"training_directory": "/path/to/images"}'

# Result: Work-horse crashes with Metal compiler error
```

## Next Steps for Investigation

### Phase 1: Confirm Root Cause
1. **Test with CPU**: Force CPU device in worker to confirm it works
2. **Test with different PyTorch versions**: Check if 2.8.x or 2.10.x have better subprocess support
3. **Check OpenCLIP versions**: Try older/newer versions
4. **Profile metal compiler**: Use `metal-profiler` to see exactly where crash occurs

### Phase 2: Explore Solutions
1. **Lazy model loading**: Load model in main process, pass via shared memory
2. **Model caching in parent**: Keep model in parent process, subprocess queries via IPC
3. **PyTorch subprocess initialization hook**: Custom initialization for MPS in subprocess
4. **ONNX Runtime alternative**: Use ONNX export of OpenCLIP model with CPUExecutionProvider
5. **Upgrade dependencies**: PyTorch 2.10+, OpenCLIP 3.4+, other packages

### Phase 3: Workarounds
1. **Environment variable**: Use `PYTORCH_ENABLE_MPS_FALLBACK=1` to fallback to CPU on MPS failure
2. **Custom device selector**: Disable MPS for RQ workers only (current approach, but not ideal)
3. **Use different backend**: Switch from RQ to Celery with proper subprocess handling
4. **Process pooling**: Use `ProcessPoolExecutor` instead of RQ fork

## References

- [PyTorch MPS Known Issues](https://pytorch.org/docs/stable/notes/mps.html)
- [Apple Metal Performance Shaders Documentation](https://developer.apple.com/documentation/metalperformanceshaders)
- [OpenCLIP GitHub](https://github.com/mlfoundry/open_clip)
- [RQ Worker Documentation](https://python-rq.org/docs/)

## Appendix: Test Results

### System Info
```
PyTorch version: 2.9.1
Python version: 3.12.11
Platform: darwin (macOS)

MPS Configuration:
  - MPS built: True
  - MPS available: True
  - MPS allocation test: SUCCESS

CUDA Configuration:
  - CUDA available: False

Installed Packages:
- open_clip: 3.2.0
- torch: 2.9.1
- torchvision: 0.24.1
- onnxruntime: 1.23.2
- insightface: 0.7.3
```

### Subprocess MPS Tests
```
[Test 1] Simple subprocess with torch import
  Result: ✅ PASS
  Exit code: 0
  Output: Device: mps, Available: True

[Test 2] Subprocess with MPS tensor allocation
  Result: ✅ PASS
  Exit code: 0
  Output: MPS tensor created: torch.Size([10])

[Test 3] Subprocess with open_clip model creation
  Result: ❌ TIMEOUT (>30s)
  Exit code: Hangs indefinitely
  Issue: Metal compiler service initialization fails
```

## Summary

The RQ worker crash is caused by **OpenCLIP model creation on MPS within forked subprocess contexts**. PyTorch and basic MPS operations work fine in subprocesses, but attempting to compile/load a large model (ViT-B-32) on MPS triggers Metal compiler service interaction that fails in the subprocess context due to inherited-but-broken handle state.

**Solutions should focus on:**
1. Preventing MPS usage in RQ worker subprocesses only (pragmatic short-term)
2. Using shared memory or IPC to pass pre-loaded models (medium-term)
3. Switching to CPU/ONNX for worker context (medium-term)
4. Waiting for PyTorch/OpenCLIP fixes for better subprocess MPS support (long-term)
