# Multi-Platform ML Device Support Implementation Plan

**Created**: 2025-12-26
**Status**: Planning Complete - Awaiting Approval
**Scope**: Add Apple Silicon (MPS) support alongside existing NVIDIA CUDA

---

## Executive Summary

This plan adds support for Apple Silicon (M1/M2/M3/M4) GPU acceleration using PyTorch's MPS (Metal Performance Shaders) backend, while maintaining full compatibility with existing NVIDIA CUDA infrastructure.

### Goals
1. **Device Abstraction**: Centralized device management replacing hardcoded logic
2. **MPS Support**: Enable GPU acceleration on Apple Silicon Macs
3. **Auto-Detection**: Automatic selection of best available accelerator
4. **Configuration**: Environment variable overrides for device selection
5. **Backward Compatibility**: Zero changes to existing CUDA behavior

### Scope
- **In Scope**: PyTorch device selection, ONNX Runtime providers, dependency management
- **Out of Scope**: Multi-GPU support, TensorRT integration, model quantization

### Effort Estimate
- **Total**: 1.5-2 weeks
- **Phase 1**: 2-3 days (device abstraction)
- **Phase 2**: 1-2 days (embedding service)
- **Phase 3**: 1-2 days (face detection)
- **Phase 4**: 1 day (metadata)
- **Phase 5**: 2-3 days (testing & docs)

---

## Current State Analysis

### Overview

The service uses **two distinct ML frameworks**:
1. **PyTorch + OpenCLIP** - CLIP embeddings for image/text similarity
2. **ONNX Runtime + InsightFace** - Face detection and recognition

### Current Device Support Matrix

| Platform | Hardware | CLIP Embeddings | Face Detection | Status |
|----------|----------|-----------------|----------------|--------|
| Linux x86_64 | NVIDIA GPU | CUDA | CUDA | Supported |
| Linux x86_64 | CPU | CPU | CPU | Supported |
| macOS ARM64 | Apple M-series | CPU (no MPS) | CPU | Not Optimized |
| macOS x86_64 | CPU | CPU | CPU | Legacy |

### Identified Issues

#### 1. Hardcoded Device Logic (`services/embedding.py`)

**Lines 35 and 68** contain identical binary device selection:
```python
device = "cuda" if torch.cuda.is_available() else "cpu"  # NO MPS SUPPORT
```

**Problems**:
- Binary choice ignores MPS backend entirely
- Device selection logic duplicated in two places
- No configuration override capability
- No device validation or error handling

#### 2. Missing MPS Provider (`faces/detector.py`)

**Line 33** specifies ONNX Runtime providers:
```python
providers=["CUDAExecutionProvider", "CPUExecutionProvider"]  # Missing CoreMLExecutionProvider
```

**Status**:
- CUDA detection works correctly (`_has_cuda()` helper)
- CPU fallback works
- No CoreML provider for Apple Neural Engine
- macOS gets CPU-only InsightFace

#### 3. Incomplete Dependencies (`pyproject.toml`)

**Current ML dependencies**:
```toml
"torch>=2.0.0",                                           # Not version-pinned
"open-clip-torch>=2.24.0",                               # Not version-pinned
"onnxruntime-gpu>=1.16.0; sys_platform != 'darwin'",    # Platform-specific (good)
"onnxruntime>=1.16.0; sys_platform == 'darwin'",        # CPU-only on macOS (suboptimal)
```

**Issues**:
- PyTorch version not pinned (stability risk)
- No platform-specific PyTorch installation guidance
- ONNX Runtime correctly uses platform markers, but no GPU support on macOS

#### 4. Incomplete Metadata Capture (`queue/training_jobs.py`)

**Lines 475-482**:
```python
environment_meta["cuda_available"] = torch.cuda.is_available()
if torch.cuda.is_available():
    environment_meta["gpu_name"] = torch.cuda.get_device_name(0)  # CUDA-specific
```

**Issue**: Assumes CUDA for all GPU detection, no MPS tracking

### Files Requiring Changes

| File | Priority | Change Type |
|------|----------|-------------|
| `core/device.py` (NEW) | High | Create centralized device manager |
| `services/embedding.py` | High | Replace hardcoded device logic |
| `faces/detector.py` | Medium | Add CoreML provider |
| `core/config.py` | Medium | Add device configuration |
| `queue/training_jobs.py` | Low | Update metadata capture |
| `pyproject.toml` | Medium | Document platform dependencies |

---

## Solution Architecture

### Design Principles

1. **Single Source of Truth**: All device logic in `core/device.py`
2. **Explicit Over Implicit**: Clear device selection with logging
3. **Fail-Safe Defaults**: Always fall back to CPU if GPU unavailable
4. **Configurable**: Environment variables override auto-detection
5. **Testable**: Device manager mockable for unit tests

### New Module: `core/device.py`

```python
"""Centralized device management for ML inference.

Device Selection Priority:
1. DEVICE environment variable (explicit override)
2. FORCE_CPU=true (force CPU mode)
3. Auto-detect: CUDA > MPS > CPU

Usage:
    from image_search_service.core.device import get_device, get_device_info

    device = get_device()  # Returns "cuda", "mps", or "cpu"
    info = get_device_info()  # Returns comprehensive device info dict
"""

import os
import platform
from functools import lru_cache
from typing import Any

import torch


@lru_cache(maxsize=1)
def get_device() -> str:
    """Get the best available PyTorch device.

    Priority:
    1. DEVICE env var (explicit override)
    2. FORCE_CPU env var (forces CPU)
    3. CUDA (if available)
    4. MPS (if available on Apple Silicon)
    5. CPU (fallback)

    Returns:
        Device string: "cuda", "cuda:0", "mps", or "cpu"
    """
    # Priority 1: Explicit device override
    if device := os.getenv("DEVICE"):
        _validate_device(device)
        return device

    # Priority 2: Force CPU mode
    if os.getenv("FORCE_CPU", "").lower() in ("true", "1", "yes"):
        return "cpu"

    # Priority 3: Auto-detect best available
    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def _validate_device(device: str) -> None:
    """Validate device string and raise if invalid."""
    valid_prefixes = ("cuda", "mps", "cpu")
    if not any(device.startswith(prefix) for prefix in valid_prefixes):
        raise ValueError(
            f"Invalid DEVICE '{device}'. Must be 'cuda', 'cuda:N', 'mps', or 'cpu'"
        )

    # Validate CUDA device ID if specified
    if device.startswith("cuda:"):
        try:
            device_id = int(device.split(":")[1])
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                if device_id >= device_count:
                    raise ValueError(
                        f"CUDA device {device_id} not available. "
                        f"Only {device_count} device(s) found."
                    )
        except ValueError:
            raise ValueError(f"Invalid CUDA device ID in '{device}'")


@lru_cache(maxsize=1)
def get_device_info() -> dict[str, Any]:
    """Get comprehensive device and platform information.

    Returns:
        Dict with platform, PyTorch, CUDA, and MPS information.
    """
    info: dict[str, Any] = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "selected_device": get_device(),
    }

    # CUDA information
    info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)

    # MPS information (Apple Silicon)
    if hasattr(torch.backends, "mps"):
        info["mps_built"] = torch.backends.mps.is_built()
        info["mps_available"] = torch.backends.mps.is_available()
    else:
        info["mps_built"] = False
        info["mps_available"] = False

    return info


def get_onnx_providers() -> list[str]:
    """Get ONNX Runtime execution providers in priority order.

    Returns:
        List of available providers, prioritized: CUDA > CoreML > CPU
    """
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
    except ImportError:
        return ["CPUExecutionProvider"]

    # Priority order
    priority = [
        "CUDAExecutionProvider",
        "CoreMLExecutionProvider",  # Apple Neural Engine
        "CPUExecutionProvider",
    ]

    return [p for p in priority if p in available]


def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon (M1/M2/M3/M4)."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"
```

### Configuration Changes (`core/config.py`)

Add new settings:
```python
# Device settings
device: str = Field(default="auto", alias="DEVICE")
force_cpu: bool = Field(default=False, alias="FORCE_CPU")
```

Note: The `DEVICE` environment variable values:
- `"auto"` - Auto-detect (default)
- `"cuda"` or `"cuda:0"` - Force CUDA
- `"mps"` - Force MPS (Apple Silicon)
- `"cpu"` - Force CPU

---

## Implementation Phases

### Phase 1: Device Abstraction Layer (2-3 days)

#### Task 1.1: Create Device Manager Module

**File**: `src/image_search_service/core/device.py` (NEW)

**Implementation**:
1. Create module with `get_device()`, `get_device_info()`, `get_onnx_providers()`
2. Add device validation and error handling
3. Implement `lru_cache` for performance
4. Add comprehensive logging

**Acceptance Criteria**:
- [ ] `get_device()` returns correct device on all platforms
- [ ] Environment variable override works (`DEVICE=cpu`)
- [ ] Invalid device values raise clear errors
- [ ] Device info includes all relevant metadata

#### Task 1.2: Add Configuration Settings

**File**: `src/image_search_service/core/config.py`

**Changes**:
```python
# Add to Settings class (after line 74)
device: str = Field(default="auto", alias="DEVICE")
force_cpu: bool = Field(default=False, alias="FORCE_CPU")
onnx_providers: str = Field(default="", alias="ONNX_PROVIDERS")
```

**Acceptance Criteria**:
- [ ] Settings load from environment variables
- [ ] Default values are sensible (auto-detect)
- [ ] Settings accessible via `get_settings()`

#### Task 1.3: Add Unit Tests

**File**: `tests/core/test_device.py` (NEW)

**Test Cases**:
```python
@patch("torch.cuda.is_available", return_value=True)
def test_get_device_cuda_available():
    assert get_device() == "cuda"

@patch("torch.cuda.is_available", return_value=False)
@patch("torch.backends.mps.is_available", return_value=True)
def test_get_device_mps_fallback():
    assert get_device() == "mps"

@patch.dict(os.environ, {"DEVICE": "cpu"})
def test_get_device_env_override():
    assert get_device() == "cpu"

def test_get_device_invalid_raises():
    with patch.dict(os.environ, {"DEVICE": "tpu"}):
        with pytest.raises(ValueError):
            get_device()
```

---

### Phase 2: Embedding Service Update (1-2 days)

#### Task 2.1: Replace Hardcoded Device Logic

**File**: `src/image_search_service/services/embedding.py`

**Before** (line 35):
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

**After**:
```python
from image_search_service.core.device import get_device

device = get_device()
```

**Also update** (line 68):
```python
# BEFORE
self._device = "cuda" if torch.cuda.is_available() else "cpu"

# AFTER
from image_search_service.core.device import get_device
self._device = get_device()
```

#### Task 2.2: Add Startup Logging

**File**: `src/image_search_service/services/embedding.py`

**Add to `_load_model()`**:
```python
from image_search_service.core.device import get_device, get_device_info

device = get_device()
device_info = get_device_info()
logger.info(
    f"Loading OpenCLIP model {settings.clip_model_name} on {device}",
    extra={"device_info": device_info}
)
```

#### Task 2.3: Update Tests

**File**: `tests/services/test_embedding.py`

**Update fixtures** to mock `get_device()` instead of `torch.cuda.is_available()`:
```python
@patch("image_search_service.services.embedding.get_device", return_value="cpu")
def test_embed_text_cpu(mock_device):
    service = EmbeddingService()
    result = service.embed_text("test")
    assert len(result) == 512
```

---

### Phase 3: Face Detection Update (1-2 days)

#### Task 3.1: Update ONNX Runtime Provider Selection

**File**: `src/image_search_service/faces/detector.py`

**Before** (line 33):
```python
_face_analysis = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
```

**After**:
```python
from image_search_service.core.device import get_onnx_providers

_face_analysis = FaceAnalysis(
    name="buffalo_l",
    providers=get_onnx_providers(),
)
```

#### Task 3.2: Update CUDA Detection Helper

**File**: `src/image_search_service/faces/detector.py`

**Replace** `_has_cuda()` function:
```python
# BEFORE
def _has_cuda() -> bool:
    try:
        import onnxruntime as ort
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except ImportError:
        return False

# AFTER
from image_search_service.core.device import get_onnx_providers

def _has_gpu_provider() -> bool:
    """Check if GPU-accelerated ONNX provider is available."""
    providers = get_onnx_providers()
    gpu_providers = {"CUDAExecutionProvider", "CoreMLExecutionProvider"}
    return any(p in gpu_providers for p in providers)
```

#### Task 3.3: Update Context ID Logic

**File**: `src/image_search_service/faces/detector.py`

**Update** model preparation:
```python
# BEFORE
ctx_id = 0 if _has_cuda() else -1

# AFTER
ctx_id = 0 if _has_gpu_provider() else -1
```

---

### Phase 4: Metadata & Monitoring (1 day)

#### Task 4.1: Update Training Job Metadata

**File**: `src/image_search_service/queue/training_jobs.py`

**Replace** (lines 475-482):
```python
# BEFORE
try:
    import torch
    environment_meta["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        environment_meta["gpu_name"] = torch.cuda.get_device_name(0)
except ImportError:
    environment_meta["cuda_available"] = False

# AFTER
from image_search_service.core.device import get_device_info

try:
    device_info = get_device_info()
    environment_meta.update({
        "device": device_info["selected_device"],
        "cuda_available": device_info["cuda_available"],
        "mps_available": device_info.get("mps_available", False),
        "platform": device_info["platform"],
        "machine": device_info["machine"],
    })
    if device_info["cuda_available"]:
        environment_meta["gpu_name"] = device_info.get("cuda_device_name")
except Exception as e:
    logger.warning(f"Failed to capture device info: {e}")
    environment_meta["device"] = "unknown"
```

#### Task 4.2: Add Device Info to Health Endpoint

**File**: `src/image_search_service/api/routes/health.py`

**Add optional device info** (if health route doesn't exist, skip):
```python
from image_search_service.core.device import get_device_info

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": get_device_info(),
    }
```

---

### Phase 5: Documentation & Testing (2-3 days)

#### Task 5.1: Update pyproject.toml

**Add installation notes** (not code changes, documentation):
```toml
# NOTE: PyTorch installation varies by platform
#
# NVIDIA CUDA (Linux/Windows):
#   pip install torch --index-url https://download.pytorch.org/whl/cu118
#
# Apple Silicon (macOS ARM64):
#   pip install torch  # MPS support included in standard wheel
#
# CPU-only:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
```

#### Task 5.2: Update Environment Documentation

**Update** `CLAUDE.md` Environment Variables section:
```markdown
## Device Configuration (NEW)

```bash
# Auto-detect best device (default)
DEVICE=auto

# Force specific device
DEVICE=cuda       # NVIDIA GPU
DEVICE=mps        # Apple Silicon
DEVICE=cpu        # CPU only

# Force CPU even if GPU available (debugging)
FORCE_CPU=true
```
```

#### Task 5.3: Create Platform Setup Guide

**File**: `docs/platform-setup.md` (NEW)

**Contents**:
1. Linux + NVIDIA GPU setup
2. macOS + Apple Silicon setup
3. CPU-only setup
4. Troubleshooting device detection
5. Performance expectations per platform

#### Task 5.4: Integration Testing

**Test Matrix**:

| Platform | Test Case | Expected Result |
|----------|-----------|-----------------|
| Linux CUDA | Default startup | Device = "cuda" |
| Linux CUDA | DEVICE=cpu | Device = "cpu" |
| macOS ARM64 | Default startup | Device = "mps" |
| macOS ARM64 | FORCE_CPU=true | Device = "cpu" |
| Any | DEVICE=invalid | Raises ValueError |

**Manual Tests**:
1. Embed an image on each platform
2. Verify embedding generation succeeds
3. Check logs for correct device selection
4. Run face detection on each platform
5. Verify model loads with correct provider

---

## Dependency Changes

### pyproject.toml Recommendations

**Current**:
```toml
"torch>=2.0.0",
"onnxruntime-gpu>=1.16.0; sys_platform != 'darwin'",
"onnxruntime>=1.16.0; sys_platform == 'darwin'",
```

**Recommended** (documentation only, no changes required):

The current setup works because:
1. PyTorch standard wheel includes MPS support on macOS ARM64
2. ONNX Runtime CPU-only on macOS is acceptable (CoreML optional)
3. No code changes needed in pyproject.toml

**Optional Enhancement** (future consideration):
```toml
# Consider for future if CoreML support needed:
# "onnxruntime-silicon>=1.16.0; sys_platform == 'darwin' and platform_machine == 'arm64'",
```

---

## Risk Assessment

### High Risks

#### MPS Backend Stability
**Risk**: MPS is newer than CUDA (PyTorch 1.12+), may have edge cases
**Impact**: Incorrect embeddings or crashes on Apple Silicon
**Mitigation**:
- Extensive testing on M1/M2/M3/M4 hardware
- Keep CPU fallback always available
- Log MPS-specific errors distinctly
- Add `FORCE_CPU` escape hatch

#### ONNX Runtime CoreML Support
**Risk**: CoreML provider may not support all InsightFace operations
**Impact**: Face detection may fall back to CPU on macOS
**Mitigation**:
- Document CPU fallback behavior
- Test InsightFace on Apple Silicon
- Accept CPU fallback if CoreML unsupported

### Medium Risks

#### Performance Regressions
**Risk**: Device abstraction adds overhead
**Impact**: Slightly slower startup
**Mitigation**:
- Use `lru_cache` for device detection
- Profile before/after changes
- Minimal overhead (< 10ms)

#### Testing Coverage
**Risk**: Limited hardware for cross-platform testing
**Impact**: Bugs discovered in production
**Mitigation**:
- CI with mocked devices
- Developer testing on real hardware
- Staged rollout

### Low Risks

#### Configuration Confusion
**Risk**: Users set invalid device values
**Impact**: Clear error at startup
**Mitigation**:
- Validate device values immediately
- Clear error messages
- Documentation with examples

---

## Success Criteria

### Functional Requirements

- [ ] Embedding generation works on CUDA, MPS, and CPU
- [ ] Face detection works on CUDA and CPU (MPS optional via CoreML)
- [ ] Device selection configurable via `DEVICE` environment variable
- [ ] `FORCE_CPU=true` works on all platforms
- [ ] No hardcoded device assumptions in codebase
- [ ] Device info included in training job metadata

### Non-Functional Requirements

- [ ] Device detection completes in < 100ms
- [ ] No performance regression for existing CUDA path
- [ ] All existing tests pass
- [ ] New unit tests for device module
- [ ] Documentation updated

### Verification Checklist

- [ ] Run `make test` - all tests pass
- [ ] Run `make typecheck` - no type errors
- [ ] Test on Linux with NVIDIA GPU
- [ ] Test on macOS with Apple Silicon
- [ ] Test with `DEVICE=cpu` override
- [ ] Verify startup logs show correct device
- [ ] Embed an image and verify embedding quality

---

## Implementation Order

### Recommended Sequence

1. **Phase 1.1**: Create `core/device.py` - foundation for all other changes
2. **Phase 1.2**: Add configuration settings
3. **Phase 1.3**: Add unit tests for device module
4. **Phase 2.1-2.3**: Update embedding service (highest user impact)
5. **Phase 3.1-3.3**: Update face detection
6. **Phase 4.1-4.2**: Update metadata (can be done in parallel)
7. **Phase 5.1-5.4**: Documentation and integration testing

### Parallel Work Opportunities

- **Phase 1.3** (tests) can parallel with **Phase 2** (embedding)
- **Phase 3** (face detection) can parallel with **Phase 4** (metadata)
- **Phase 5** should happen after all code changes

---

## Rollback Plan

If issues discovered after deployment:

1. **Immediate**: Set `FORCE_CPU=true` in environment
2. **Quick Fix**: Revert device module, restore hardcoded logic
3. **Full Rollback**: Revert all commits from this implementation

**No database changes** means clean rollback is always possible.

---

## Appendix A: Environment Variable Reference

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `DEVICE` | `auto`, `cuda`, `cuda:N`, `mps`, `cpu` | `auto` | PyTorch device selection |
| `FORCE_CPU` | `true`, `false` | `false` | Force CPU mode |
| `ONNX_PROVIDERS` | Comma-separated list | Auto | ONNX Runtime providers |

**Examples**:
```bash
# Production (Linux + NVIDIA GPU)
DEVICE=cuda

# Development (macOS + Apple Silicon)
DEVICE=mps

# Testing/CI (CPU-only)
FORCE_CPU=true
DEVICE=cpu

# Multi-GPU selection
DEVICE=cuda:1
```

---

## Appendix B: Device Detection Algorithm

```
START
  │
  ├─ Check DEVICE env var
  │    ├─ Set? → Validate → Use it
  │    └─ Not set? → Continue
  │
  ├─ Check FORCE_CPU env var
  │    ├─ true? → Return "cpu"
  │    └─ false? → Continue
  │
  ├─ Check torch.cuda.is_available()
  │    ├─ true? → Return "cuda"
  │    └─ false? → Continue
  │
  ├─ Check torch.backends.mps.is_available()
  │    ├─ true? → Return "mps"
  │    └─ false? → Continue
  │
  └─ Return "cpu" (fallback)
END
```

---

## Appendix C: File Change Summary

| File | Action | Lines Changed (Est.) |
|------|--------|---------------------|
| `core/device.py` | CREATE | ~80 lines |
| `core/config.py` | MODIFY | +3 lines |
| `services/embedding.py` | MODIFY | ~10 lines |
| `faces/detector.py` | MODIFY | ~15 lines |
| `queue/training_jobs.py` | MODIFY | ~10 lines |
| `tests/core/test_device.py` | CREATE | ~60 lines |
| `tests/services/test_embedding.py` | MODIFY | ~10 lines |
| `docs/platform-setup.md` | CREATE | ~100 lines |
| `CLAUDE.md` | MODIFY | +10 lines |

**Total**: ~300 lines of code/tests, ~100 lines of documentation

---

**Status**: Planning Complete - Awaiting Approval

**Next Steps**:
1. Review and approve this plan
2. Begin Phase 1 implementation
3. Test on target hardware platforms
