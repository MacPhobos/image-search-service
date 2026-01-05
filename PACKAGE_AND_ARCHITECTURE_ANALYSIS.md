# Package Analysis & Architectural Root Cause

**Date**: 2026-01-05  
**Status**: ⚠️ CRITICAL ARCHITECTURAL LIMITATION IDENTIFIED  
**Severity**: High

---

## Executive Summary

The crash is **NOT primarily a package issue**. It's a fundamental architectural limitation:

**The preload approach CANNOT work because:**
1. RQ creates work-horse subprocesses using Python's **spawn** method (default on macOS)
2. spawn() creates a completely fresh Python interpreter with no inherited memory
3. The preload in the worker main process doesn't help the work-horse
4. Work-horse still tries to load the model fresh → MPS crash

---

## Package Analysis

### pyproject.toml Configuration
```
open-clip-torch>=2.24.0         ✅ Works fine on macOS
torch>=2.0.0                    ✅ Has MPS support
onnxruntime-gpu (Linux only)    ✅ Correctly excluded on macOS
onnxruntime (macOS CPU)         ✅ Correct for macOS
insightface>=0.7.3              ✅ Works with CoreML on macOS
```

### Key Finding: Packages are NOT the Issue

All packages have correct configurations:
- **PyTorch**: MPS is built and available ✓
- **OpenCLIP**: Works on MPS (though defaults to CPU without explicit device arg) ✓
- **ONNX Runtime**: CoreML provider available for macOS ✓
- **InsightFace**: Initializes with CoreML correctly ✓

**Verdict**: Packages are NOT tailored incorrectly for macOS. They work fine.

---

## Architectural Root Cause

### The Real Issue: RQ's Work-Horse Spawning

**Process Tree When Training Job is Queued:**

```
Terminal 1: make dev
  └─ FastAPI main process (PID 1000)
     └─ preload_embedding_model() runs
     └─ Model loaded in memory ✓

Terminal 2: make worker  
  └─ RQ Worker main process (PID 2000)
     └─ preload_embedding_model() called at import
     └─ Model loaded in memory ✓
     └─ Worker waits for jobs...

Terminal 3: POST /training/sessions
  └─ Job queued to Redis
  └─ RQ Worker picks up job
  └─ **Work-horse subprocess spawned (PID 3000)**
     ├─ NEW Python interpreter (fresh process)
     ├─ Module globals = None
     ├─ preload() NOT re-run (not at module import)
     ├─ Job calls embed_images_batch()
     ├─ _load_model() called
     ├─ _model is None (fresh globals)
     ├─ OpenCLIP initialization on MPS
     └─ ❌ CRASH
```

### Why The Preload Doesn't Help Work-Horse

On **macOS**, Python's default multiprocessing method is **SPAWN**:

```python
import multiprocessing
multiprocessing.get_start_method()  # Returns: 'spawn'
```

**What spawn() does:**
1. Creates completely new Python process
2. Executes `python -m module_name` (or equivalent)
3. Fresh interpreter = no inherited memory
4. All module globals initialized to defaults

**What fork() would do (Linux):**
1. Duplicates parent process memory
2. Child inherits all parent state (including loaded model)
3. Module globals have parent's values
4. preload's effect IS inherited

### The Two-Level Process Separation

```
Layer 1: RQ Worker Main Process
  ├─ Imports worker.py → preload_embedding_model() runs
  ├─ Model loaded: _model = <CLIP model>
  └─ Waits for jobs

Layer 2: Work-Horse Subprocess (spawned by RQ for each job)
  ├─ Fresh Python interpreter
  ├─ All imports happen fresh
  ├─ Module globals: _model = None
  ├─ preload_embedding_model() NOT called
  ├─ First job call → _load_model()
  ├─ _model is None → loads fresh
  └─ OpenCLIP on MPS → CRASH
```

---

## Why This is Different from What We Tested

### What Our Tests Showed (Wrong Model)

We tested with `subprocess.run()`:
```python
result = subprocess.run([sys.executable, '-c', code])
```

This DOES import worker.py in the subprocess, so preload was called. ✓

### What RQ Actually Does (The Real Model)

RQ creates work-horse differently:
```python
# Simplified - RQ actually does something similar
worker.prepare_job_and_execute()  # Runs in work-horse process
```

Work-horse is created ONCE per job and runs with a fresh interpreter that:
1. Does NOT necessarily import the full worker module
2. Only imports what's needed for the job
3. Module globals are fresh
4. Preload from worker main doesn't carry over

---

## Why The Package Question Was Valid

Your suspicion that packages might be an issue was reasonable because:

1. **OpenCLIP Behavior**: 
   - Without explicit device arg, defaults to CPU
   - Our code does `create_model_and_transforms()` then `.to(device)`
   - But if the model object gets recreated somewhere...

2. **ONNX Runtime on macOS**:
   - Only has CoreML provider (no GPU provider like CUDA)
   - Any GPU-expecting code will fall back to CPU

3. **Packages Could Be Misconfigured**:
   - If we installed GPU versions of packages on macOS = wrong
   - If PyTorch doesn't have MPS = problem
   - If ONNX Runtime can't use CoreML = issue

But our analysis confirms:
- **Packages are configured correctly** ✓
- **The issue is architectural** (process spawning) ❌

---

## Summary of Findings

### ✅ Package Level
- PyTorch: MPS available and built ✓
- OpenCLIP: Works on MPS ✓
- ONNX Runtime: CoreML available ✓
- All dependency constraints correct for macOS ✓

### ❌ Architecture Level
- RQ uses spawn() for work-horse on macOS
- spawn() creates fresh Python interpreter
- Preload in worker main ≠ preload in work-horse
- Work-horse can't access preloaded model
- Work-horse tries fresh load → MPS crash ❌

### 🔴 The Real Root Cause
**Not packages. Not device selection. It's the process spawning model.**

RQ's fork/spawn architecture on macOS doesn't support the "preload in parent" pattern for GPU models that need Metal compiler service initialization.

---

## What This Means For The Current Fix

The preload code we added to `queue/worker.py`:
- ✅ Runs in worker main process ✓
- ❌ Does NOT help work-horse subprocess ✗
- ❌ Work-horse still crashes ✗

The fix is **incomplete** because it addresses the wrong process layer.

---

## Alternative Approaches (Not Implemented)

### Option 1: Force CPU for Work-Horse (Pragmatic But Slow)
```python
# In work-horse context
os.environ['DEVICE'] = 'cpu'
```
- Pros: Simple, reliable, no crashes
- Cons: Slow (no GPU), defeats the purpose

### Option 2: Use Model Server Pattern
```
Main API → Model Server (separate GPU process)
Worker → Model Server (HTTP/IPC)
```
- Pros: Works with any backend, GPU always available
- Cons: Complex, more infrastructure

### Option 3: ONNX Export + CoreML
```
Export OpenCLIP to ONNX
Use onnxruntime with CoreML provider
```
- Pros: Native macOS optimization
- Cons: Requires model export, different inference path

### Option 4: Custom RQ Work-Horse Hook
```python
def work_horse_init():
    preload_embedding_model()

Worker(work_horse_init=work_horse_init)
```
- Pros: Runs in work-horse process
- Cons: RQ might not support this pattern

---

## Conclusion

The issue is **not a package problem** - it's an architectural limitation of how RQ spawns work-horse subprocesses on macOS.

The current fix (preload in worker main) doesn't solve the problem because:
1. Work-horse is spawned as a fresh Python process
2. Fresh interpreter = no inherited model
3. Work-horse still needs to load the model itself
4. That load happens in subprocess context → MPS crash

**The fix we implemented was based on incomplete understanding of RQ's process model.**

