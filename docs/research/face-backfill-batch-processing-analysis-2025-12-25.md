# Face Detection Backfill: Batch Processing Analysis

**Date**: 2025-12-25
**Objective**: Investigate current face detection backfill implementation to identify opportunities for GPU batch processing optimization

## Executive Summary

The current face detection backfill processes images **sequentially** (one at a time), which is inefficient for GPU processing. InsightFace's underlying detection model processes one image per `get()` call, but opportunities exist for:

1. **Image pre-loading and batching**: Load N images concurrently while GPU processes previous batch
2. **Multi-threaded image I/O**: Decouple disk I/O from GPU inference using ThreadPoolExecutor
3. **Qdrant batch optimization**: Already batching vectors, but can increase batch size

**Key Constraint**: InsightFace `FaceAnalysis.get()` processes **one image at a time** - no native batch inference API exists.

---

## Current Implementation Flow

### CLI Entry Point
**File**: `/export/workspace/image-search/image-search-service/src/image_search_service/scripts/faces.py`

```python
@faces_app.command("backfill")
def backfill_faces(
    limit: int = 1000,
    offset: int = 0,
    min_confidence: float = 0.5,
    queue: bool = False,
) -> None:
```

**Flow**:
1. Queries database for assets without faces (using subquery exclusion)
2. Calls `service.process_assets_batch()` with list of asset IDs
3. Returns statistics: `processed`, `total_faces`, `errors`

**Key Parameters**:
- `limit`: Number of assets to process (default: 1000)
- `offset`: Pagination offset (default: 0)
- `min_confidence`: Detection threshold (default: 0.5)
- `queue`: Run as background RQ job vs. direct execution

---

### Face Processing Service
**File**: `/export/workspace/image-search/image-search-service/src/image_search_service/faces/service.py`

#### Current `process_assets_batch()` Implementation

```python
def process_assets_batch(
    self,
    asset_ids: list[int],
    min_confidence: float = 0.5,
    min_face_size: int = 20,
) -> dict[str, int]:
    """Process multiple assets in batch."""
    total_faces = 0
    processed = 0
    errors = 0

    for asset_id in asset_ids:  # ⚠️ SEQUENTIAL LOOP
        asset = self.db.get(ImageAsset, asset_id)
        if not asset:
            errors += 1
            continue

        try:
            faces = self.process_asset(asset, min_confidence, min_face_size)  # ⚠️ ONE AT A TIME
            total_faces += len(faces)
            processed += 1
        except Exception as e:
            logger.error(f"Error processing asset {asset_id}: {e}")
            errors += 1

    return {"processed": processed, "total_faces": total_faces, "errors": errors}
```

**Problems**:
1. **Sequential processing**: Processes one asset at a time in a for-loop
2. **GPU underutilization**: GPU sits idle between image loads
3. **No parallelism**: Database fetch → disk I/O → GPU inference all happen serially

#### Current `process_asset()` Implementation

```python
def process_asset(
    self,
    asset: ImageAsset,
    min_confidence: float = 0.5,
    min_face_size: int = 20,
) -> list[FaceInstance]:
    """Process a single asset: detect faces, store in DB and Qdrant."""

    # 1. Resolve image path
    image_path = self._resolve_asset_path(asset)

    # 2. Detect faces (calls InsightFace)
    detected = detect_faces_from_path(
        str(image_path),
        min_confidence=min_confidence,
        min_face_size=min_face_size,
    )

    # 3. Create FaceInstance records
    face_instances = []
    qdrant_points = []
    for face in detected:
        # Check for duplicates (idempotency)
        existing = self._find_existing_face(asset.id, face.bbox)
        if existing:
            continue

        # Create DB record
        face_instance = FaceInstance(...)
        self.db.add(face_instance)
        face_instances.append(face_instance)

        # Prepare Qdrant point
        qdrant_points.append({
            "point_id": uuid.uuid4(),
            "embedding": face.embedding.tolist(),
            "asset_id": asset.id,
            # ... metadata ...
        })

    # 4. Commit to database
    self.db.commit()

    # 5. Batch upsert to Qdrant
    if qdrant_points:
        self.qdrant.upsert_faces_batch(qdrant_points)  # ✅ Already batched!

    return face_instances
```

**Good**: Qdrant operations are already batched
**Problem**: Each asset is still processed sequentially

---

### Face Detection Layer
**File**: `/export/workspace/image-search/image-search-service/src/image_search_service/faces/detector.py`

#### InsightFace Model Loading

```python
_face_analysis: Any | None = None  # Global singleton

def _ensure_model_loaded() -> Any:
    """Lazy load InsightFace model."""
    global _face_analysis
    if _face_analysis is None:
        from insightface.app import FaceAnalysis

        _face_analysis = FaceAnalysis(
            name="buffalo_l",  # Good balance of speed/accuracy
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        ctx_id = 0 if _has_cuda() else -1
        _face_analysis.prepare(ctx_id=ctx_id, det_size=(640, 640))

    return _face_analysis
```

**Key Observations**:
- Model is loaded once (singleton pattern)
- Supports CUDA via ONNX Runtime
- Detection size: 640x640 (fixed)
- Buffalo_l model: Good speed/accuracy balance

#### Detection Function

```python
def detect_faces_from_path(
    image_path: str,
    min_confidence: float = 0.5,
    min_face_size: int = 20,
) -> list[DetectedFace]:
    """Detect faces from an image file path."""
    import cv2

    image = cv2.imread(image_path)  # ⚠️ DISK I/O BLOCKS
    if image is None:
        return []

    return detect_faces(image, min_confidence, min_face_size)

def detect_faces(
    image: np.ndarray,
    min_confidence: float = 0.5,
    min_face_size: int = 20,
) -> list[DetectedFace]:
    """Detect faces in an image."""
    app = _ensure_model_loaded()

    # InsightFace expects BGR format (OpenCV default)
    faces = app.get(image)  # ⚠️ SINGLE IMAGE INFERENCE

    results = []
    for face in faces:
        # Filter by confidence and size
        if face.det_score < min_confidence:
            continue
        if (face.bbox[2] - face.bbox[0]) < min_face_size:
            continue

        results.append(DetectedFace(
            bbox=(...),
            confidence=face.det_score,
            landmarks=face.kps,
            embedding=face.embedding,  # 512-d normalized
        ))

    return results
```

**Critical Findings**:
1. `cv2.imread()` is synchronous/blocking
2. `app.get(image)` processes **one image at a time**
3. No batch inference API in InsightFace

---

## InsightFace Batch Processing Capabilities

### Investigation of `FaceAnalysis.get()` Method

**Source**: `/export/workspace/image-search/image-search-service/.venv/lib/python3.12/site-packages/insightface/app/face_analysis.py`

```python
def get(self, img, max_num=0):
    # 1. Detection model inference (single image)
    bboxes, kpss = self.det_model.detect(img, max_num=max_num, metric='default')

    if bboxes.shape[0] == 0:
        return []

    # 2. For each detected face, run recognition models
    ret = []
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i, 0:4]
        det_score = bboxes[i, 4]
        kps = kpss[i] if kpss is not None else None

        face = Face(bbox=bbox, kps=kps, det_score=det_score)

        # Run all models (embedding, age, gender, etc.)
        for taskname, model in self.models.items():
            if taskname == 'detection':
                continue
            model.get(img, face)  # Extracts face region and runs model

        ret.append(face)

    return ret
```

**Conclusions**:
- ❌ **No batch inference API**: `get()` only accepts a single image
- ✅ **Batch processing per face**: After detection, processes all faces in the image
- 💡 **Optimization opportunity**: The underlying ONNX models *could* support batching, but InsightFace doesn't expose this

---

## Optimization Opportunities

### 1. **Parallel Image Loading with Threading**

**Current**: Load image → Detect faces → Load next image
**Proposed**: Load N images in parallel while GPU processes current batch

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_assets_batch_parallel(
    self,
    asset_ids: list[int],
    batch_size: int = 8,  # GPU can process while loading next batch
    max_workers: int = 4,  # I/O threads
):
    """Process assets with parallel image loading."""

    # Pre-fetch assets from database
    assets = [self.db.get(ImageAsset, aid) for aid in asset_ids]

    # Process in batches
    for i in range(0, len(assets), batch_size):
        batch = assets[i:i+batch_size]

        # Load images in parallel (I/O bound)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_load_image, asset): asset
                for asset in batch
            }

            # Process as images become available
            for future in as_completed(futures):
                asset = futures[future]
                image = future.result()

                if image is not None:
                    # GPU inference (still sequential per image)
                    detected = detect_faces(image, ...)
                    self._store_faces(asset, detected)
```

**Benefits**:
- Overlaps disk I/O with GPU inference
- GPU stays busy while next batch loads
- Simple to implement (no changes to InsightFace)

**Limitations**:
- Still processes images sequentially through GPU
- GIL limits Python threading (but I/O releases GIL)

---

### 2. **Increase Qdrant Batch Size**

**Current**: Batches all faces from one asset
**Proposed**: Accumulate faces from multiple assets before upserting

```python
def process_assets_batch_optimized(
    self,
    asset_ids: list[int],
    qdrant_batch_size: int = 100,  # Upsert every 100 faces
):
    qdrant_buffer = []

    for asset_id in asset_ids:
        asset = self.db.get(ImageAsset, asset_id)
        detected = detect_faces_from_path(...)

        for face in detected:
            # Create DB record
            face_instance = FaceInstance(...)
            self.db.add(face_instance)

            # Add to buffer
            qdrant_buffer.append({
                "point_id": face_instance.qdrant_point_id,
                "embedding": face.embedding.tolist(),
                # ...
            })

            # Flush buffer when full
            if len(qdrant_buffer) >= qdrant_batch_size:
                self.qdrant.upsert_faces_batch(qdrant_buffer)
                qdrant_buffer.clear()

    # Flush remaining
    if qdrant_buffer:
        self.qdrant.upsert_faces_batch(qdrant_buffer)
```

**Benefits**:
- Reduces network round-trips to Qdrant
- Better amortization of batch overhead

---

### 3. **Progress Reporting for Long-Running Backfills**

**Current**: No progress feedback during execution
**Proposed**: Use `tqdm` or logging every N assets

```python
from tqdm import tqdm

def process_assets_batch(self, asset_ids: list[int], ...):
    total_faces = 0
    processed = 0
    errors = 0

    with tqdm(total=len(asset_ids), desc="Processing assets") as pbar:
        for asset_id in asset_ids:
            try:
                faces = self.process_asset(...)
                total_faces += len(faces)
                processed += 1
            except Exception as e:
                errors += 1
            finally:
                pbar.update(1)
                pbar.set_postfix(faces=total_faces, errors=errors)
```

---

### 4. **Async I/O with asyncio (Advanced)**

**Not recommended for this codebase** because:
- InsightFace is synchronous (blocking GPU inference)
- SQLAlchemy sync session (would need async rewrite)
- cv2.imread is synchronous

**Would require**:
- Rewrite service layer with async/await
- Use aiofiles for image loading
- Run GPU inference in executor pool

**Complexity**: High
**Benefit**: Moderate (I/O parallelism already achievable with threads)

---

## Recommended Optimizations (Priority Order)

### Priority 1: Parallel Image Loading (Quick Win)
- **Implementation**: ThreadPoolExecutor for `cv2.imread()`
- **Impact**: Overlaps I/O with GPU inference
- **Complexity**: Low (2-3 hour implementation)
- **Risk**: Low (backward compatible)

### Priority 2: Increase Qdrant Batch Size
- **Implementation**: Buffer faces across multiple assets
- **Impact**: Reduces network overhead
- **Complexity**: Low (1 hour implementation)
- **Risk**: Low (Qdrant already supports batching)

### Priority 3: Progress Reporting
- **Implementation**: Add tqdm progress bar
- **Impact**: Better UX for long-running jobs
- **Complexity**: Very low (30 minutes)
- **Risk**: None

### Priority 4: Investigate ONNX Batch Inference (Future)
- **Implementation**: Bypass InsightFace, use ONNX Runtime directly
- **Impact**: True GPU batch processing (potentially 2-5x faster)
- **Complexity**: High (requires ONNX model expertise)
- **Risk**: Medium (model compatibility, testing required)

---

## Key Files Reference

| File | Purpose | Lines of Interest |
|------|---------|-------------------|
| `src/image_search_service/scripts/faces.py` | CLI command entry point | Lines 13-74 (`backfill_faces`) |
| `src/image_search_service/faces/service.py` | High-level face processing | Lines 114-147 (`process_assets_batch`), Lines 25-112 (`process_asset`) |
| `src/image_search_service/faces/detector.py` | InsightFace integration | Lines 85-154 (`detect_faces`, `detect_faces_from_path`) |
| `src/image_search_service/queue/face_jobs.py` | RQ background job wrapper | Lines 188-240 (`backfill_faces_job`) |

---

## Existing Patterns in Codebase

### Async Code Usage
- **API routes**: Use FastAPI's async/await (e.g., `api/routes/faces.py`)
- **Database**: SQLAlchemy async session in API layer
- **CLI/Jobs**: Use **sync** code (sync session, blocking I/O)

### Batch Processing Patterns
- **Qdrant operations**: Already batched (see `face_qdrant.py`)
- **Database queries**: Batch fetch assets with single query
- **No GPU batch processing**: All inference is single-image

---

## Testing Considerations

### Before Optimization
1. Measure current throughput (assets/second, faces/second)
2. Profile GPU utilization (should be <100% due to I/O waits)
3. Measure Qdrant network latency

### After Optimization
1. Verify throughput improvement (target: 1.5-2x for image loading parallelism)
2. Confirm GPU utilization increases
3. Test with various batch sizes (4, 8, 16, 32)
4. Ensure idempotency still works (no duplicate faces)
5. Test error handling (corrupted images, missing files)

### Test Dataset
- Small: 100 assets (quick validation)
- Medium: 1,000 assets (performance comparison)
- Large: 10,000 assets (production-like)

---

## Conclusion

The current face detection backfill is **GPU-underutilized** due to sequential processing. While InsightFace doesn't provide a batch inference API, we can achieve **significant speedups** (estimated 1.5-2x) through:

1. **Parallel image loading** (ThreadPoolExecutor)
2. **Larger Qdrant batches** (reduce network overhead)
3. **Progress reporting** (better UX)

For even greater speedups (2-5x), we would need to bypass InsightFace and implement custom ONNX batch inference, but this is a complex undertaking requiring model expertise.

**Recommended next step**: Implement Priority 1 (parallel image loading) as a proof-of-concept to validate performance gains before committing to more complex optimizations.
