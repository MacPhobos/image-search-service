# ML/GPU/CUDA Implementation Analysis

**Date**: 2025-12-26
**Author**: Research Agent
**Purpose**: Analyze current ML/GPU/CUDA usage to inform multi-platform device support (NVIDIA CUDA + Apple Silicon MPS)

---

## Executive Summary

The image-search-service uses two distinct ML frameworks:
1. **PyTorch + OpenCLIP** for image/text embeddings (CLIP model)
2. **ONNX Runtime + InsightFace** for face detection and recognition

**Current Device Support**:
- ✅ CUDA detection and fallback to CPU exists for both frameworks
- ❌ NO Apple Silicon (MPS) support currently
- ⚠️ Platform-specific dependencies exist but incomplete (onnxruntime vs onnxruntime-gpu)
- ⚠️ Hardcoded device assumptions in embedding service

**Key Findings**:
- Device selection happens at model load time (lazy initialization)
- No centralized device management abstraction
- Device information captured in metadata but not configurable
- Face detection already has platform-aware ONNX Runtime selection
- CLIP embeddings hardcode "cuda" vs "cpu" binary choice

---

## 1. Dependencies Analysis

### 1.1 PyProject.toml Dependencies

```toml
# ML/GPU Dependencies (lines 18-28)
"open-clip-torch>=2.24.0",
"torch>=2.0.0",
"pillow>=10.0.0",
"insightface>=0.7.3",
"onnxruntime-gpu>=1.16.0; sys_platform != 'darwin'",   # ← Platform-specific
"onnxruntime>=1.16.0; sys_platform == 'darwin'",       # ← Platform-specific
"opencv-python>=4.8.0",
"hdbscan>=0.8.33",
"tqdm>=4.67.1",
```

**Key Observations**:
- **PyTorch version**: `>=2.0.0` (NOT pinned - could cause issues)
- **ONNX Runtime**: Already has platform-specific handling (Darwin vs non-Darwin)
- **OpenCLIP version**: `>=2.24.0` (NOT pinned)
- **No MPS-specific dependencies** or Metal Performance Shaders integration

**Implications**:
1. ONNX Runtime CPU-only on macOS (no GPU acceleration for InsightFace)
2. PyTorch installation defaults may vary by platform
3. No version pinning = potential compatibility issues

---

## 2. Configuration Analysis

### 2.1 Core Config (`core/config.py`)

**CLIP Model Settings** (lines 46-49):
```python
clip_model_name: str = Field(default="ViT-B-32", alias="CLIP_MODEL_NAME")
clip_pretrained: str = Field(default="laion2b_s34b_b79k", alias="CLIP_PRETRAINED")
embedding_dim: int = Field(default=512, alias="EMBEDDING_DIM")
```

**Face Recognition Settings** (lines 65-74):
```python
face_model_name: str = Field(default="buffalo_l", alias="FACE_MODEL_NAME")
face_model_checkpoint: str = Field(default="", alias="FACE_MODEL_CHECKPOINT")
face_training_enabled: bool = Field(default=False, alias="FACE_TRAINING_ENABLED")

# Training hyperparameters
face_triplet_margin: float = Field(default=0.2, alias="FACE_TRIPLET_MARGIN")
face_training_epochs: int = Field(default=20, alias="FACE_TRAINING_EPOCHS")
face_batch_size: int = Field(default=32, alias="FACE_BATCH_SIZE")
face_learning_rate: float = Field(default=0.0001, alias="FACE_LEARNING_RATE")
```

**⚠️ Missing**:
- NO device selection configuration
- NO `DEVICE` or `PYTORCH_DEVICE` environment variable
- NO `ONNX_EXECUTION_PROVIDER` override
- NO `FORCE_CPU` or `DISABLE_GPU` options

---

## 3. ML Code Analysis

### 3.1 CLIP Embeddings (`services/embedding.py`)

#### Device Selection Logic (lines 34-42):

```python
def _load_model() -> tuple[Any, Any, Any]:
    import open_clip
    import torch

    settings = get_settings()
    device = "cuda" if torch.cuda.is_available() else "cpu"  # ← HARDCODED

    logger.info(f"Loading OpenCLIP model {settings.clip_model_name} on {device}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        settings.clip_model_name, pretrained=settings.clip_pretrained
    )
    model = model.to(device)  # ← HARDCODED device placement
    model.eval()
```

**Issues**:
1. **Binary choice**: Only "cuda" or "cpu" (no MPS support)
2. **Hardcoded logic**: No abstraction for device selection
3. **No override**: Cannot force CPU even if CUDA is available
4. **Global state**: `_model`, `_preprocess`, `_tokenizer` module-level globals

#### Device Property (lines 62-69):

```python
@property
def device(self) -> str:
    """Get device for model inference."""
    if self._device is None:
        import torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
    return self._device
```

**Redundant device detection** - duplicated logic from `_load_model()`

#### Inference Code (lines 77-119):

```python
def embed_text(self, text: str) -> list[float]:
    model, _, tokenizer = _load_model()
    with torch.no_grad():
        tokens = tokenizer([text]).to(self.device)  # ← Uses device property
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    result: list[float] = text_features[0].cpu().numpy().tolist()
    return result

def embed_image(self, image_path: str | Path) -> list[float]:
    model, preprocess, _ = _load_model()
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(self.device)  # ← Uses device property
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    result: list[float] = image_features[0].cpu().numpy().tolist()
    return result
```

**Observations**:
- Tensors explicitly moved to device via `.to(self.device)`
- Results always moved back to CPU (`.cpu().numpy().tolist()`)
- No mixed precision (fp16/bf16) for GPU optimization

---

### 3.2 Face Detection (`faces/detector.py`)

#### CUDA Detection (lines 14-22):

```python
def _has_cuda() -> bool:
    """Check if CUDA is available."""
    try:
        import onnxruntime as ort
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except ImportError:
        return False
```

**Good**: Uses ONNX Runtime's provider detection (not PyTorch)

#### Model Loading (lines 24-44):

```python
def _ensure_model_loaded() -> Any:
    global _face_analysis
    if _face_analysis is None:
        try:
            from insightface.app import FaceAnalysis

            _face_analysis = FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],  # ← Provider list
            )
            ctx_id = 0 if _has_cuda() else -1  # ← 0=GPU, -1=CPU
            _face_analysis.prepare(ctx_id=ctx_id, det_size=(640, 640))
            device = "CUDA" if _has_cuda() else "CPU"
            logger.info(f"Loaded InsightFace model (buffalo_l) on {device}")
        except ImportError as e:
            logger.error("InsightFace not installed. Run: pip install insightface onnxruntime-gpu")
```

**Observations**:
1. **Provider fallback**: Lists CUDA first, then CPU (ONNX Runtime auto-selects)
2. **ctx_id semantics**: 0=GPU, -1=CPU (InsightFace convention)
3. **No MPS provider**: Would need `CoreMLExecutionProvider` or custom handling

---

### 3.3 Face Processing Service (`faces/service.py`)

**GPU-aware design** (lines 152-174):
```python
def process_assets_batch(
    self,
    asset_ids: list[int],
    prefetch_batch_size: int = 8,
    io_workers: int = 4,
    qdrant_batch_size: int = 100,
    progress_callback: Any = None,
) -> dict[str, Any]:
    """Process multiple assets with parallel image loading and cross-asset Qdrant batching.

    Uses ThreadPoolExecutor to prefetch images while GPU processes previous batch.
    This overlaps disk I/O with GPU inference for better throughput.

    Returns:
        Summary dict with counts, throughput, and timing metrics:
        - processed: Number of assets processed
        - total_faces: Total faces detected
        - errors: Number of errors
        - throughput: Assets per second
        - elapsed_time: Total wall clock time
        - io_time: Total I/O time (loading images from disk)
        - gpu_time: Total GPU time (face detection)  # ← GPU time tracked
    """
```

**Good practices**:
- I/O parallelization with ThreadPoolExecutor
- Separate timing for I/O vs GPU operations
- Batching optimized for GPU throughput

---

### 3.4 Training Jobs (`queue/training_jobs.py`)

**Environment Metadata Capture** (lines 475-482):

```python
try:
    import torch
    environment_meta["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        environment_meta["gpu_name"] = torch.cuda.get_device_name(0)  # ← CUDA-specific
except ImportError:
    environment_meta["cuda_available"] = False
```

**Issue**: Assumes CUDA for GPU detection (no MPS handling)

**Device Recording** (lines 340, 396):

```python
"device": embedding_service.device,  # ← Captured in evidence metadata
```

**Good**: Device info tracked in training evidence for reproducibility

---

## 4. Hardcoded CUDA Assumptions

### 4.1 Critical Hardcoded Patterns

| File | Line | Pattern | Issue |
|------|------|---------|-------|
| `services/embedding.py` | 35 | `device = "cuda" if torch.cuda.is_available() else "cpu"` | Binary choice, no MPS |
| `services/embedding.py` | 68 | `self._device = "cuda" if torch.cuda.is_available() else "cpu"` | Duplicate logic |
| `queue/training_jobs.py` | 478-480 | `torch.cuda.is_available()` + `torch.cuda.get_device_name(0)` | CUDA-only metadata |
| `faces/detector.py` | 33 | `providers=["CUDAExecutionProvider", "CPUExecutionProvider"]` | No MPS provider |

### 4.2 Device String Usage

**Device strings found**:
- `"cuda"` - NVIDIA GPU
- `"cpu"` - CPU fallback
- `"CUDA"` - String representation in logs

**Missing**:
- `"mps"` - Apple Metal Performance Shaders
- `"cuda:0"`, `"cuda:1"` - Multi-GPU support
- Device ID specification

---

## 5. Existing Abstractions

### 5.1 What Exists

✅ **ONNX Runtime Platform Detection**:
- `pyproject.toml` uses `sys_platform` markers for Darwin vs others
- `onnxruntime` (CPU-only) on macOS
- `onnxruntime-gpu` on Linux/Windows

✅ **Lazy Model Loading**:
- Models loaded on first use (not at import time)
- Global module state for singleton models

✅ **Device Property Pattern**:
- `EmbeddingService.device` property for device access
- Cached device detection

### 5.2 What's Missing

❌ **Centralized Device Manager**:
- No single source of truth for device selection
- Device logic duplicated across files

❌ **Configuration Override**:
- Cannot force specific device via environment variable
- Cannot disable GPU acceleration

❌ **Device Abstraction Layer**:
- No `DeviceConfig` or `DeviceStrategy` class
- No device capabilities detection

❌ **Platform Detection**:
- No unified platform detection (macOS/Linux/Windows)
- No Apple Silicon (M1/M2/M3/M4) specific handling

---

## 6. Platform-Specific Dependencies

### 6.1 Current Platform Handling

**ONNX Runtime** (pyproject.toml lines 24-25):
```toml
"onnxruntime-gpu>=1.16.0; sys_platform != 'darwin'",
"onnxruntime>=1.16.0; sys_platform == 'darwin'",
```

**Analysis**:
- ✅ Prevents `onnxruntime-gpu` installation on macOS (correct)
- ❌ macOS gets CPU-only ONNX Runtime (no MPS/CoreML)
- ⚠️ No explicit M1/M2/M3/M4 optimization

### 6.2 PyTorch Platform Handling

**Current** (pyproject.toml line 19):
```toml
"torch>=2.0.0",
```

**Issues**:
1. Generic `torch` package (defaults to CPU on pip install)
2. No CUDA version specification (torch+cu118, torch+cu121, etc.)
3. No explicit platform-specific wheels
4. Apple Silicon gets CPU-only PyTorch by default

**Recommended patterns**:
```toml
# Option 1: Platform markers (like ONNX Runtime)
"torch>=2.0.0; sys_platform == 'darwin'",              # macOS (includes MPS)
"torch>=2.0.0+cu118; sys_platform != 'darwin'",         # Linux/Windows with CUDA 11.8

# Option 2: Separate requirement files
# requirements-cuda.txt, requirements-cpu.txt, requirements-mps.txt

# Option 3: Optional dependency groups
[project.optional-dependencies]
cuda = ["torch>=2.0.0+cu118"]
mps = ["torch>=2.0.0"]  # MPS included in standard macOS wheel
```

---

## 7. Device Detection Gaps

### 7.1 PyTorch Device Detection

**What exists**:
```python
torch.cuda.is_available()  # CUDA detection only
```

**What's missing**:
```python
torch.backends.mps.is_available()        # Apple Metal Performance Shaders
torch.backends.mps.is_built()            # MPS support compiled in
torch.cuda.device_count()                # Multi-GPU support
torch.version.cuda                       # CUDA version
```

### 7.2 ONNX Runtime Provider Detection

**What exists**:
```python
"CUDAExecutionProvider" in ort.get_available_providers()
```

**What's missing**:
```python
ort.get_available_providers()
# Possible values:
# - 'CUDAExecutionProvider'    (NVIDIA GPU)
# - 'CPUExecutionProvider'      (CPU fallback)
# - 'CoreMLExecutionProvider'   (Apple Neural Engine)
# - 'TensorrtExecutionProvider' (NVIDIA TensorRT)
# - 'DmlExecutionProvider'      (DirectML - Windows)
```

---

## 8. Recommended Device Support Matrix

### 8.1 Target Platforms

| Platform | Hardware | PyTorch Device | ONNX Runtime Provider | Priority |
|----------|----------|----------------|----------------------|----------|
| **Linux x86_64** | NVIDIA GPU | `cuda` | `CUDAExecutionProvider` | ✅ High (current prod) |
| **Linux x86_64** | CPU | `cpu` | `CPUExecutionProvider` | ✅ High (fallback) |
| **macOS ARM64** | Apple M1/M2/M3/M4 | `mps` | `CoreMLExecutionProvider` | 🔶 Medium (dev/testing) |
| **macOS ARM64** | CPU | `cpu` | `CPUExecutionProvider` | 🔶 Medium (fallback) |
| **macOS x86_64** | CPU | `cpu` | `CPUExecutionProvider` | 🔵 Low (legacy) |
| **Windows x86_64** | NVIDIA GPU | `cuda` | `CUDAExecutionProvider` | 🔵 Low (not tested) |

### 8.2 Device Selection Priority

**Recommended device selection order** (configurable via env var):

```python
# Priority 1: Explicit environment variable override
DEVICE = os.getenv("PYTORCH_DEVICE")  # e.g., "cuda", "cpu", "mps", "cuda:1"

# Priority 2: Auto-detect best available device
if DEVICE is None:
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"

# Priority 3: Validate device availability
device = torch.device(DEVICE)
```

---

## 9. Code Changes Required

### 9.1 High-Priority Changes

1. **Centralized Device Management** (`core/device.py`):
   - Detect platform (macOS ARM64, Linux x86_64, etc.)
   - Detect available accelerators (CUDA, MPS, CPU)
   - Provide device selection with env var override
   - Singleton pattern for device config

2. **Update Embedding Service** (`services/embedding.py`):
   - Replace hardcoded `"cuda"` with device manager
   - Support MPS device string
   - Add device validation

3. **Update Face Detector** (`faces/detector.py`):
   - Add CoreML provider for macOS
   - Update provider list based on platform
   - Add MPS support for future PyTorch-based models

4. **Update Training Jobs** (`queue/training_jobs.py`):
   - Capture MPS availability in metadata
   - Record actual device used (not just CUDA)
   - Add platform info to environment metadata

### 9.2 Medium-Priority Changes

5. **Configuration** (`core/config.py`):
   - Add `device: str` setting with env var `DEVICE`
   - Add `force_cpu: bool` setting for debugging
   - Add `onnx_providers: list[str]` for ONNX Runtime customization

6. **Dependency Pinning** (`pyproject.toml`):
   - Pin PyTorch version (e.g., `torch==2.0.1`)
   - Add CUDA version suffix for Linux (e.g., `torch==2.0.1+cu118`)
   - Document installation for different platforms

### 9.3 Low-Priority Changes

7. **Multi-GPU Support**:
   - Support device IDs (`cuda:0`, `cuda:1`, etc.)
   - Add GPU selection configuration
   - Implement device affinity for batch jobs

8. **Performance Optimization**:
   - Mixed precision (fp16/bf16) for faster inference
   - TensorRT support via ONNX Runtime
   - Model quantization for CPU deployment

---

## 10. Implementation Strategy

### Phase 1: Device Abstraction (Week 1)
- [ ] Create `core/device.py` device manager
- [ ] Add platform detection utilities
- [ ] Implement device selection with fallback chain
- [ ] Add configuration overrides

### Phase 2: Embedding Service Update (Week 1)
- [ ] Replace hardcoded device logic in `services/embedding.py`
- [ ] Add MPS device support
- [ ] Test on both CUDA and MPS platforms
- [ ] Update tests with device mocking

### Phase 3: Face Detection Update (Week 2)
- [ ] Add CoreML provider to `faces/detector.py`
- [ ] Update provider selection logic
- [ ] Test InsightFace on macOS (CPU-only initially)
- [ ] Document performance differences

### Phase 4: Metadata & Monitoring (Week 2)
- [ ] Update training job metadata capture
- [ ] Add device info to health check endpoint
- [ ] Log device selection on startup
- [ ] Add device utilization metrics

### Phase 5: Documentation & Testing (Week 3)
- [ ] Update CLAUDE.md with device setup instructions
- [ ] Document platform-specific installation
- [ ] Add device-specific CI/CD tests
- [ ] Create performance benchmarks

---

## 11. Risk Assessment

### 11.1 High Risks

⚠️ **PyTorch MPS Stability**:
- MPS backend is newer than CUDA (introduced PyTorch 1.12)
- Potential edge cases or bugs on Apple Silicon
- Not all operations have MPS kernels (fallback to CPU)

**Mitigation**: Extensive testing on M1/M2/M3/M4 hardware, fallback to CPU

⚠️ **ONNX Runtime CoreML Support**:
- CoreML provider may not support all InsightFace operations
- Performance may be worse than CUDA on equivalent hardware
- Limited to macOS (no cross-platform benefit)

**Mitigation**: Keep CPU fallback, consider PyTorch-based face detection alternative

⚠️ **Dependency Version Conflicts**:
- PyTorch version pinning may conflict with other dependencies
- Platform-specific wheels increase complexity
- Installation documentation becomes platform-specific

**Mitigation**: Document all platforms, use virtual environments, CI testing

### 11.2 Medium Risks

⚠️ **Performance Regressions**:
- Device abstraction layer may add minimal overhead
- Device detection adds startup time
- Fallback chains increase code complexity

**Mitigation**: Benchmark before/after, profile device detection, cache results

⚠️ **Configuration Complexity**:
- More environment variables to document
- Device selection priority may be confusing
- Override behavior needs clear documentation

**Mitigation**: Sensible defaults, clear logging, comprehensive docs

### 11.3 Low Risks

⚠️ **Testing Coverage**:
- Need hardware access for each platform
- CI/CD may not have GPU runners
- Manual testing required for MPS/CUDA

**Mitigation**: Developer testing, cloud GPU instances, staged rollout

---

## 12. Testing Strategy

### 12.1 Unit Tests (Mock-based)

```python
# Test device detection with mocked hardware
@patch('torch.cuda.is_available', return_value=True)
def test_device_detection_cuda(mock_cuda):
    device_manager = DeviceManager()
    assert device_manager.device == "cuda"

@patch('torch.cuda.is_available', return_value=False)
@patch('torch.backends.mps.is_available', return_value=True)
def test_device_detection_mps(mock_cuda, mock_mps):
    device_manager = DeviceManager()
    assert device_manager.device == "mps"
```

### 12.2 Integration Tests (Hardware-based)

**CUDA Platform** (Linux x86_64 + NVIDIA GPU):
- [ ] Embedding generation uses CUDA
- [ ] Face detection uses CUDAExecutionProvider
- [ ] Metadata captures GPU name
- [ ] Training jobs use CUDA device

**MPS Platform** (macOS ARM64 + M1/M2/M3/M4):
- [ ] Embedding generation uses MPS
- [ ] Face detection falls back to CPU (or uses CoreML)
- [ ] Metadata captures MPS availability
- [ ] No CUDA references in logs

**CPU Fallback** (Any platform):
- [ ] Works when `DEVICE=cpu` env var set
- [ ] Works when no accelerators available
- [ ] Performance degradation acceptable
- [ ] All features functional

### 12.3 Performance Benchmarks

**Metrics to compare** (CUDA vs MPS vs CPU):
- Embedding generation throughput (images/sec)
- Face detection throughput (images/sec)
- Model load time (seconds)
- Memory usage (GB)
- Accuracy (should be identical)

---

## 13. Open Questions

1. **ONNX Runtime MPS Support**:
   - Does InsightFace work with CoreMLExecutionProvider?
   - Should we switch to PyTorch-based face detection for MPS?
   - What's the performance comparison?

2. **PyTorch Wheel Distribution**:
   - Should we pre-install platform-specific PyTorch wheels?
   - How to handle CUDA version selection (11.8 vs 12.1)?
   - Should we use `torch.hub` for model downloads?

3. **Production Deployment**:
   - Will production always have CUDA available?
   - Is MPS support needed for production or just development?
   - Should we support multi-GPU training in future?

4. **Backward Compatibility**:
   - Can existing embeddings be used with new device code?
   - Do we need to re-embed existing assets?
   - What about existing Qdrant vectors?

---

## 14. Conclusion

### Current State
- ✅ CUDA detection and CPU fallback work correctly
- ✅ ONNX Runtime has platform-specific dependency handling
- ❌ No Apple Silicon (MPS) support
- ❌ Device selection hardcoded in multiple places
- ❌ No centralized device management

### Recommended Approach
1. **Create device abstraction layer** (`core/device.py`)
2. **Update embedding service** to use abstraction
3. **Test on both platforms** (CUDA + MPS)
4. **Document platform-specific setup**
5. **Add device monitoring** to health checks

### Effort Estimate
- **Device abstraction**: 2-3 days
- **Embedding service update**: 1-2 days
- **Face detection update**: 1-2 days (if CoreML supported)
- **Testing & documentation**: 2-3 days
- **Total**: ~1.5-2 weeks for full MPS support

### Success Criteria
- ✅ Embedding generation works on CUDA, MPS, and CPU
- ✅ Face detection works on CUDA and CPU (MPS optional)
- ✅ Device selection configurable via environment variable
- ✅ No hardcoded device assumptions remain
- ✅ All tests pass on both platforms
- ✅ Performance acceptable on both platforms

---

## Appendix A: File Inventory

### Files with ML/GPU Code

1. **`src/image_search_service/services/embedding.py`** (126 lines)
   - CLIP model loading and inference
   - Device detection and tensor placement
   - **Changes required**: High priority

2. **`src/image_search_service/faces/detector.py`** (155 lines)
   - InsightFace model loading
   - ONNX Runtime provider selection
   - **Changes required**: Medium priority

3. **`src/image_search_service/queue/training_jobs.py`** (~500 lines)
   - Training job orchestration
   - Device metadata capture
   - **Changes required**: Low priority (metadata only)

4. **`src/image_search_service/faces/service.py`** (~400 lines)
   - Batch processing with I/O parallelization
   - GPU timing metrics
   - **Changes required**: None (device-agnostic)

5. **`src/image_search_service/core/config.py`** (109 lines)
   - Application configuration
   - **Changes required**: Medium priority (add device settings)

### Files to Create

1. **`src/image_search_service/core/device.py`** (new)
   - Centralized device management
   - Platform detection
   - Device selection with fallback

---

## Appendix B: Device Detection Code Examples

### PyTorch Device Detection

```python
import torch
import platform

def detect_best_device() -> str:
    """Detect best available PyTorch device."""
    # Check for CUDA
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"

    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"

    # Fallback to CPU
    return "cpu"

def get_device_info() -> dict:
    """Get comprehensive device information."""
    info = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
    }

    # CUDA info
    if torch.cuda.is_available():
        info["cuda_available"] = True
        info["cuda_version"] = torch.version.cuda
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
    else:
        info["cuda_available"] = False

    # MPS info
    if hasattr(torch.backends, 'mps'):
        info["mps_built"] = torch.backends.mps.is_built()
        info["mps_available"] = torch.backends.mps.is_available()
    else:
        info["mps_built"] = False
        info["mps_available"] = False

    return info
```

### ONNX Runtime Provider Detection

```python
import onnxruntime as ort

def get_onnx_providers() -> list[str]:
    """Get available ONNX Runtime execution providers in priority order."""
    available = ort.get_available_providers()

    # Priority order: CUDA > CoreML > CPU
    priority = [
        "CUDAExecutionProvider",
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]

    return [p for p in priority if p in available]

def create_onnx_session(model_path: str) -> ort.InferenceSession:
    """Create ONNX Runtime session with best available provider."""
    providers = get_onnx_providers()
    return ort.InferenceSession(model_path, providers=providers)
```

---

## Appendix C: Relevant Environment Variables

### Current (Implicit)
- `DATABASE_URL` - PostgreSQL connection
- `REDIS_URL` - Redis connection
- `QDRANT_URL` - Qdrant vector database
- `CLIP_MODEL_NAME` - CLIP model variant
- `FACE_MODEL_NAME` - InsightFace model variant

### Proposed (Device Management)
- `DEVICE` - PyTorch device (`cuda`, `mps`, `cpu`, `cuda:0`, etc.)
- `FORCE_CPU` - Force CPU mode (boolean, overrides auto-detection)
- `ONNX_PROVIDERS` - Comma-separated ONNX Runtime providers
- `CUDA_VISIBLE_DEVICES` - Standard CUDA device masking (already supported by PyTorch)

### Example Configuration

```bash
# Production (Linux + NVIDIA GPU)
DEVICE=cuda
ONNX_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider

# Development (macOS + Apple Silicon)
DEVICE=mps
ONNX_PROVIDERS=CoreMLExecutionProvider,CPUExecutionProvider

# Testing/CI (CPU-only)
FORCE_CPU=true
DEVICE=cpu
ONNX_PROVIDERS=CPUExecutionProvider
```

---

**End of Analysis**
