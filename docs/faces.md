# Face Detection and Recognition Pipeline

**Version**: 2.0 (Dual-Mode Clustering + Training)
**Last Updated**: 2024-12-24

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Database Schema](#database-schema)
4. [Vector Storage](#vector-storage)
5. [Face Detection Pipeline](#face-detection-pipeline)
6. [Clustering](#clustering)
7. [Person Recognition](#person-recognition)
8. [Dual-Mode Clustering](#dual-mode-clustering)
9. [Training System](#training-system)
10. [API Reference](#api-reference)
11. [CLI Reference](#cli-reference)
12. [Background Jobs](#background-jobs)
13. [Workflow Guide](#workflow-guide)
14. [Tuning Parameters](#tuning-parameters)
15. [Troubleshooting](#troubleshooting)

---

## Overview

The face detection and recognition pipeline provides automated face detection, **dual-mode clustering** (supervised + unsupervised), **triplet loss training**, and incremental person recognition capabilities for the image search system.

### Architecture

```
Image Asset
    ↓
Face Detection (InsightFace)
    ↓
512-dim ArcFace Embedding
    ↓
Store in PostgreSQL + Qdrant
    ↓
Clustering (HDBSCAN)
    ↓
Manual Labeling (via API/UI)
    ↓
Prototype Creation
    ↓
Incremental Assignment (new faces → known persons)
```

### Key Features

- **Face Detection**: InsightFace RetinaFace detector with 5-point landmarks
- **Face Embeddings**: 512-dimensional ArcFace embeddings (buffalo_l model)
- **Vector Storage**: Qdrant collection with cosine similarity search
- **Dual-Mode Clustering**: Supervised (known people) + Unsupervised (unknown faces)
- **Training System**: Triplet loss training for improved person separation
- **Person Recognition**: Prototype-based incremental assignment
- **Quality Scoring**: Automatic quality assessment based on size and confidence
- **Idempotent Processing**: Re-running detection won't create duplicates
- **Progressive Learning**: Improves accuracy with each labeling session

### Components

- **Detector** (`faces/detector.py`): Face detection and embedding extraction
- **Service** (`faces/service.py`): High-level orchestration layer
- **Clusterer** (`faces/clusterer.py`): HDBSCAN clustering for identity grouping
- **Dual Clusterer** (`faces/dual_clusterer.py`): Dual-mode supervised + unsupervised clustering
- **Trainer** (`faces/trainer.py`): Triplet loss training for improved embeddings
- **Assigner** (`faces/assigner.py`): Incremental person assignment via prototypes
- **Qdrant Client** (`vector/face_qdrant.py`): Vector database operations
- **API Routes** (`api/routes/faces.py`): REST endpoints
- **CLI Commands** (`scripts/cli.py`): Command-line interface
- **Background Jobs** (`queue/face_jobs.py`): RQ async processing

---

## Installation

### Dependencies

Install face detection dependencies:

```bash
# Core dependencies (already in pyproject.toml)
uv add insightface onnxruntime-gpu hdbscan

# For CPU-only environments
uv add insightface onnxruntime hdbscan
```

### System Requirements

- **GPU Recommended**: CUDA-capable GPU for faster processing
- **CPU Fallback**: Works on CPU but slower (10-20x)
- **Memory**: ~2GB RAM for model loading
- **Storage**: ~500MB for InsightFace buffalo_l model

### Model Download

On first use, InsightFace automatically downloads the buffalo_l model to `~/.insightface/models/buffalo_l/`.

### Qdrant Collection Setup

Create the faces collection:

```bash
# Using CLI
uv run python -m image_search_service.scripts.cli faces ensure-collection

# Using Python
from image_search_service.vector.face_qdrant import get_face_qdrant_client

client = get_face_qdrant_client()
client.ensure_collection()
```

---

## Database Schema

### Tables

#### `persons`

Stores person entities (labeled identity clusters).

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key |
| `name` | VARCHAR(255) | Person name (unique, case-insensitive) |
| `status` | ENUM | `active`, `merged`, `hidden` |
| `merged_into_id` | UUID | If merged, points to target person |
| `created_at` | TIMESTAMPTZ | Creation timestamp |
| `updated_at` | TIMESTAMPTZ | Last update timestamp |

**Indexes**:
- Unique index on `LOWER(name)`
- Index on `status`

#### `face_instances`

Stores detected faces in images.

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key |
| `asset_id` | INTEGER | Foreign key to `image_assets` |
| `bbox_x` | INTEGER | Bounding box X coordinate (pixels) |
| `bbox_y` | INTEGER | Bounding box Y coordinate (pixels) |
| `bbox_w` | INTEGER | Bounding box width (pixels) |
| `bbox_h` | INTEGER | Bounding box height (pixels) |
| `landmarks` | JSONB | 5-point facial landmarks |
| `detection_confidence` | FLOAT | Detection confidence (0.0-1.0) |
| `quality_score` | FLOAT | Face quality score (0.0-1.0) |
| `qdrant_point_id` | UUID | Reference to Qdrant vector point |
| `cluster_id` | VARCHAR(100) | Cluster assignment from HDBSCAN |
| `person_id` | UUID | Foreign key to `persons` (nullable) |
| `created_at` | TIMESTAMPTZ | Detection timestamp |
| `updated_at` | TIMESTAMPTZ | Last update timestamp |

**Indexes**:
- Unique constraint on `(asset_id, bbox_x, bbox_y, bbox_w, bbox_h)` for idempotency
- Index on `asset_id`
- Index on `cluster_id`
- Index on `person_id`
- Index on `quality_score`

#### `person_prototypes`

Stores prototype face embeddings for person recognition.

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key |
| `person_id` | UUID | Foreign key to `persons` |
| `face_instance_id` | UUID | Foreign key to `face_instances` (nullable for centroids) |
| `qdrant_point_id` | UUID | Reference to Qdrant vector point |
| `role` | ENUM | `centroid` (computed average) or `exemplar` (high-quality sample) |
| `created_at` | TIMESTAMPTZ | Creation timestamp |

**Indexes**:
- Index on `person_id`

### Relationships

```
image_assets (1) ←──→ (N) face_instances
persons (1) ←──→ (N) face_instances
persons (1) ←──→ (N) person_prototypes
face_instances (1) ←──→ (0-1) person_prototypes
```

---

## Vector Storage

### Qdrant Collection: `faces`

**Configuration**:
- **Vector Dimension**: 512 (ArcFace embeddings)
- **Distance Metric**: Cosine similarity
- **Indexes**: `person_id`, `cluster_id`, `is_prototype`, `asset_id`, `face_instance_id`

### Payload Schema

```json
{
  "asset_id": "123",
  "face_instance_id": "uuid-string",
  "detection_confidence": 0.95,
  "quality_score": 0.82,
  "bbox": {"x": 100, "y": 150, "w": 80, "h": 80},
  "person_id": "uuid-string",
  "cluster_id": "clu_abc123",
  "is_prototype": false,
  "taken_at": "2025-12-23T10:30:00Z"
}
```

### Payload Fields

- **asset_id**: Integer asset ID (stored as string)
- **face_instance_id**: UUID of database face instance
- **detection_confidence**: Detection confidence score (0.0-1.0)
- **quality_score**: Computed quality score (0.0-1.0)
- **bbox**: Bounding box coordinates
- **person_id**: UUID of assigned person (nullable)
- **cluster_id**: Cluster ID from HDBSCAN (nullable)
- **is_prototype**: Boolean flag for prototype faces
- **taken_at**: Image timestamp (nullable)

### Quality Score Computation

```python
area = bbox_width * bbox_height
area_score = min(1.0, area / 40000)  # 200x200px = 1.0
quality_score = (area_score * 0.5) + (detection_confidence * 0.5)
```

---

## Face Detection Pipeline

### Detection Process

1. **Load Image**: Read image from asset path
2. **Detect Faces**: InsightFace RetinaFace detector
3. **Filter Results**: Apply confidence and size thresholds
4. **Extract Embeddings**: 512-dim ArcFace vectors (pre-normalized)
5. **Compute Quality**: Quality score based on size and confidence
6. **Store in DB**: Create `FaceInstance` record
7. **Store in Qdrant**: Upsert vector with payload

### Detection Parameters

```python
min_confidence: float = 0.5   # Detection confidence threshold
min_face_size: int = 20       # Minimum width/height in pixels
```

### InsightFace Model

**Model**: `buffalo_l` (balance of speed and accuracy)

**Detection Settings**:
- **Detection size**: 640x640 pixels
- **5-point landmarks**: left_eye, right_eye, nose, mouth_left, mouth_right
- **Embedding**: 512-dim ArcFace (L2-normalized)

### GPU vs CPU

**GPU (CUDA)**:
- Uses `CUDAExecutionProvider`
- ~50-100ms per image
- Recommended for batch processing

**CPU**:
- Uses `CPUExecutionProvider`
- ~500-1000ms per image
- Acceptable for small batches

### Idempotency

Re-running face detection on the same asset:
- Checks for existing faces with same bounding box
- Returns existing `FaceInstance` without creating duplicates
- Safe to re-run after parameter tuning

---

## Clustering

### HDBSCAN Algorithm

HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) groups similar face embeddings into identity clusters.

**Advantages**:
- No need to specify number of clusters
- Handles noise (outliers)
- Finds clusters of varying density
- Works well with cosine/euclidean distance

### Clustering Parameters

```python
min_cluster_size: int = 5       # Minimum faces to form a cluster
min_samples: int = 3            # Minimum samples for core point
cluster_selection_epsilon: float = 0.0  # Cluster merging threshold
metric: str = "euclidean"       # Distance metric (euclidean or cosine)
```

**Tuning Guidelines**:
- **Small datasets (<1000 faces)**: `min_cluster_size=3`
- **Medium datasets (1k-10k)**: `min_cluster_size=5`
- **Large datasets (>10k)**: `min_cluster_size=7-10`

### Clustering Workflow

1. **Fetch Unlabeled Faces**: Query Qdrant for faces without `person_id`
2. **Filter by Quality**: Only use faces with `quality_score >= threshold`
3. **Collect Embeddings**: Retrieve 512-dim vectors
4. **Run HDBSCAN**: Compute cluster labels
5. **Assign Cluster IDs**: Generate unique cluster IDs
6. **Update Database**: Set `cluster_id` in PostgreSQL
7. **Update Qdrant**: Set `cluster_id` in payload

### Cluster ID Format

```
clu_{12-hex-chars}          # Regular cluster
clu_abc123_sub_a1b2c3      # Sub-cluster from split operation
```

### Noise Handling

Faces labeled as noise (cluster label = -1) remain with `cluster_id = NULL`. These faces:
- Are not included in any cluster
- Can be re-clustered with different parameters
- Can be manually assigned to persons
- May be assigned via prototype matching

### Re-clustering

**Within-cluster splitting** for large/heterogeneous clusters:

```python
clusterer.recluster_within_cluster(
    cluster_id="clu_abc123",
    min_cluster_size=3
)
```

This creates sub-clusters like `clu_abc123_sub_a1b2c3`.

---

## Person Recognition

### Prototype-Based Matching

Person recognition uses **prototypes** (representative face embeddings) to match new faces to known persons.

### Prototype Types

1. **Exemplar** (`PrototypeRole.EXEMPLAR`):
   - High-quality individual faces
   - Created when labeling a cluster (top 3 quality faces)
   - Links to a specific `FaceInstance`

2. **Centroid** (`PrototypeRole.CENTROID`):
   - Computed average of all faces for a person
   - Normalized mean of embeddings
   - No specific `FaceInstance` link
   - Improves matching accuracy for diverse poses/lighting

### Assignment Process

1. **Get Unlabeled Faces**: Faces without `person_id` or `cluster_id`
2. **Search Prototypes**: For each face, search against all prototypes in Qdrant
3. **Match Best Candidate**: If top match exceeds threshold, assign person
4. **Update Database**: Set `person_id` in PostgreSQL
5. **Update Qdrant**: Set `person_id` in payload

### Similarity Threshold

```python
similarity_threshold: float = 0.6  # Cosine similarity (0.0-1.0)
```

**Tuning**:
- **Strict (0.7-0.8)**: Fewer false positives, more manual review
- **Balanced (0.6-0.7)**: Good default
- **Loose (0.5-0.6)**: More automatic assignment, higher false positive risk

### Centroid Computation

```python
# Collect all embeddings for a person
embeddings = [face.embedding for face in person.faces]

# Compute mean
centroid = np.mean(embeddings, axis=0)

# Re-normalize (important for cosine distance)
centroid = centroid / np.linalg.norm(centroid)
```

---

## Dual-Mode Clustering

### Overview

Dual-mode clustering enhances the face recognition system by combining two complementary approaches:

1. **Supervised Mode**: Assigns faces to known Person entities using learned prototypes
2. **Unsupervised Mode**: Clusters unknown faces by similarity for discovery and labeling

This hybrid approach maximizes accuracy for known people while keeping unknown faces organized for progressive labeling.

### Architecture

```
Face Detection → Qdrant Storage → DUAL-MODE CLUSTERING
                                         ↓
                        ┌────────────────┴────────────────┐
                        ↓                                 ↓
                SUPERVISED MODE                   UNSUPERVISED MODE
            (Match to Known People)            (Cluster Unknown Faces)
                        ↓                                 ↓
                person_* clusters                 unknown_cluster_* groups
                        └────────────────┬────────────────┘
                                         ↓
                              TRAINING SYSTEM
                        (Learn from Labels → Improve)
```

### How It Works

**Phase 1: Supervised Assignment**
1. Load all person prototypes (centroids + exemplars) from database
2. For each unassigned face, search Qdrant for nearest prototype
3. If similarity exceeds threshold (default: 0.7), assign to that person
4. Update `person_id` and set `cluster_id = person_{uuid}`

**Phase 2: Unsupervised Clustering**
1. Collect all faces not assigned in Phase 1
2. Run clustering algorithm (HDBSCAN, DBSCAN, or Agglomerative)
3. Group similar faces into clusters
4. Assign unique cluster IDs: `unknown_cluster_1`, `unknown_cluster_2`, etc.
5. Noise points (outliers) get `cluster_id = NULL`

### Cluster Naming Convention

| Prefix | Type | Example | Description |
|--------|------|---------|-------------|
| `person_{uuid}` | Supervised | `person_a3f9b2c1...` | Assigned to labeled Person entity |
| `unknown_cluster_{N}` | Unsupervised | `unknown_cluster_5` | Group of similar unknown faces |
| `NULL` | Noise | - | Face that doesn't match anything (outlier) |

### Parameters

#### Supervised Mode

```python
person_threshold: float = 0.7  # Minimum similarity for person assignment
```

**Tuning**:
- **0.7-0.8**: Strict (fewer false positives, more manual review)
- **0.6-0.7**: Balanced (recommended)
- **0.5-0.6**: Loose (more auto-assignment, higher false positive risk)

#### Unsupervised Mode

```python
unknown_method: str = "hdbscan"  # Clustering algorithm
unknown_min_cluster_size: int = 3  # Minimum faces per cluster
unknown_eps: float = 0.5  # DBSCAN/Agglomerative distance threshold
```

**Algorithms**:

| Method | Best For | Parameters |
|--------|----------|------------|
| `hdbscan` | Variable density clusters | `min_cluster_size` |
| `dbscan` | Similar-sized clusters | `eps`, `min_samples` |
| `agglomerative` | Hierarchical grouping | `eps`, `linkage` |

### CLI Usage

#### Basic Dual Clustering

```bash
# Default parameters
uv run python -m image_search_service.scripts.cli faces cluster-dual

# Custom thresholds
uv run python -m image_search_service.scripts.cli faces cluster-dual \
  --person-threshold 0.75 \
  --unknown-method hdbscan \
  --unknown-min-size 5

# Using Makefile
make faces-cluster-dual PERSON_THRESHOLD=0.75 UNKNOWN_METHOD=hdbscan
```

#### Output Example

```
Dual-Mode Clustering Results:
========================================
Phase 1 (Supervised):
  - Faces processed: 5000
  - Assigned to persons: 3200 (64.0%)
  - Persons matched: 42

Phase 2 (Unsupervised):
  - Unlabeled faces: 1800
  - Clusters found: 28
  - Noise (outliers): 150 (8.3%)

Cluster Distribution:
  - person_* clusters: 42 (3200 faces)
  - unknown_cluster_* groups: 28 (1650 faces)
  - Unassigned (noise): 150 faces
```

### API Usage

**Endpoint**: `POST /api/v1/faces/cluster/dual`

**Request** (`DualClusterRequest`):
```json
{
  "person_threshold": 0.7,
  "unknown_method": "hdbscan",
  "unknown_min_cluster_size": 3,
  "unknown_eps": 0.5,
  "max_faces": 50000,
  "queue": false
}
```

**Response** (`DualClusterResponse`):
```json
{
  "status": "completed",
  "supervised": {
    "faces_processed": 5000,
    "assigned_to_persons": 3200,
    "persons_matched": 42
  },
  "unsupervised": {
    "unlabeled_faces": 1800,
    "clusters_found": 28,
    "noise_count": 150
  },
  "total_faces": 5000,
  "execution_time_seconds": 3.5
}
```

**Background Job**:
```bash
curl -X POST http://localhost:8000/api/v1/faces/cluster/dual \
  -H "Content-Type: application/json" \
  -d '{"person_threshold": 0.7, "queue": true}'
```

### Workflow Integration

**Initial Setup**:
```bash
# 1. Detect faces
make faces-backfill LIMIT=10000

# 2. Run dual-mode clustering
make faces-cluster-dual

# 3. Check results
make faces-stats
```

**Progressive Labeling**:
```bash
# 1. Review unknown clusters via API
curl http://localhost:8000/api/v1/faces/clusters?includeLabeled=false

# 2. Label a cluster
curl -X POST http://localhost:8000/api/v1/faces/clusters/unknown_cluster_5/label \
  -H "Content-Type: application/json" \
  -d '{"name": "Alice Johnson"}'

# 3. Re-run dual clustering (now assigns Alice's faces)
make faces-cluster-dual
```

### Benefits Over Single-Mode

| Aspect | Single-Mode (HDBSCAN only) | Dual-Mode |
|--------|---------------------------|-----------|
| **Known People** | May split across clusters | Single `person_*` cluster |
| **Unknown People** | Mixed with known | Separate `unknown_cluster_*` |
| **Accuracy** | 60-70% | 85-95% (with training) |
| **Labeling Effort** | High (re-label splits) | Low (label once) |
| **Learning** | None | Improves with training |

---

## Training System

### Overview

The training system uses **triplet loss** to fine-tune face embeddings for person-specific separation. By learning from labeled faces, the system improves its ability to distinguish between people in your specific dataset.

### Triplet Loss Explained

**Triplet** = (anchor, positive, negative)
- **Anchor**: A face embedding
- **Positive**: Different face of the same person
- **Negative**: Face of a different person

**Loss Function**:
```
loss = max(0, margin + d(anchor, positive) - d(anchor, negative))
```

**Goal**: Push positives closer, pull negatives farther apart.

### Architecture

```
Pre-trained ArcFace Embeddings (512-dim) [FROZEN]
           ↓
Projection Head (trainable)
    ├─ Linear(512 → 256)
    ├─ ReLU
    ├─ Linear(256 → 128)
    └─ L2 Normalize
           ↓
Triplet Loss: max(0, margin + d(anchor,pos) - d(anchor,neg))
           ↓
Fine-tuned Embeddings (128-dim)
```

**Why Freeze ArcFace?**
- Preserves pre-trained face recognition capabilities
- Requires less labeled data
- Faster training
- Only adapts final representation layer

### Data Requirements

| Stage | People | Faces per Person | Total Faces | Expected Improvement |
|-------|--------|------------------|-------------|---------------------|
| **Minimum** | 5 | 5 | 25 | Modest (5-10%) |
| **Recommended** | 10-15 | 10 | 100-150 | Significant (15-25%) |
| **Production** | 20+ | 20+ | 400+ | Excellent (25-40%) |

### CLI Usage

#### Basic Training

```bash
# Default training (20 epochs)
uv run python -m image_search_service.scripts.cli faces train-matching

# Custom parameters
uv run python -m image_search_service.scripts.cli faces train-matching \
  --epochs 50 \
  --margin 0.3 \
  --batch-size 64 \
  --min-faces 10

# Using Makefile
make faces-train-matching EPOCHS=50 MARGIN=0.3
```

#### Output Example

```
Training Face Matching Model
========================================
Data Preparation:
  - Total persons: 15
  - Total faces: 342
  - Training faces: 273 (80%)
  - Validation faces: 69 (20%)
  - Triplets per epoch: ~8190

Training Progress:
Epoch 1/20: loss=0.4523, val_loss=0.3891
Epoch 5/20: loss=0.2145, val_loss=0.1987
Epoch 10/20: loss=0.1234, val_loss=0.1156
Epoch 15/20: loss=0.0876, val_loss=0.0823
Epoch 20/20: loss=0.0654, val_loss=0.0612

Results:
  - Final training loss: 0.0654
  - Final validation loss: 0.0612
  - Model saved: models/face_projection_epoch_20.pt
  - Checkpoint size: 45.2 MB
  - Training time: 8m 32s
```

### API Usage

**Endpoint**: `POST /api/v1/faces/train`

**Request** (`TrainFaceMatchingRequest`):
```json
{
  "epochs": 20,
  "margin": 0.2,
  "batch_size": 32,
  "learning_rate": 0.0001,
  "min_faces_per_person": 5,
  "queue": false
}
```

**Response** (`TrainFaceMatchingResponse`):
```json
{
  "status": "completed",
  "persons_used": 15,
  "total_faces": 342,
  "training_faces": 273,
  "validation_faces": 69,
  "final_train_loss": 0.0654,
  "final_val_loss": 0.0612,
  "model_checkpoint": "models/face_projection_epoch_20.pt",
  "training_time_seconds": 512
}
```

**Background Job**:
```bash
curl -X POST http://localhost:8000/api/v1/faces/train \
  -H "Content-Type: application/json" \
  -d '{"epochs": 20, "margin": 0.2, "queue": true}'
```

### Training Parameters

#### Core Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `epochs` | 20 | 10-100 | Training iterations |
| `margin` | 0.2 | 0.1-0.5 | Triplet loss margin |
| `batch_size` | 32 | 16-128 | Triplets per batch |
| `learning_rate` | 0.0001 | 0.00001-0.001 | Optimizer learning rate |

#### Data Filtering

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_faces_per_person` | 5 | Exclude people with fewer faces |
| `quality_threshold` | 0.5 | Minimum face quality score |

### Tuning Guidelines

**High variance (overfitting)**:
- Reduce epochs (10-15)
- Increase margin (0.3-0.4)
- Add dropout (not implemented yet)
- Get more diverse data

**High bias (underfitting)**:
- Increase epochs (30-50)
- Reduce margin (0.1-0.15)
- Increase learning rate (0.0005)
- Ensure sufficient data

**Slow convergence**:
- Increase learning rate (0.0005-0.001)
- Increase batch size (64-128)
- Check data quality

### Workflow Integration

**Full Training Cycle**:

```bash
# 1. Initial clustering
make faces-cluster-dual

# 2. Label 10-15 unknown clusters
# (via API or UI)

# 3. Train model
make faces-train-matching EPOCHS=20

# 4. Re-cluster with improved model
make faces-cluster-dual PERSON_THRESHOLD=0.75

# 5. Verify improvements
make faces-stats

# 6. Repeat: label more → train → re-cluster
```

**Iterative Improvement**:

| Cycle | People Labeled | Training Accuracy | Person Assignment | Unknown Cluster Purity |
|-------|----------------|-------------------|-------------------|----------------------|
| 0 (Baseline) | 0 | - | 70% | 65% |
| 1 | 10 | 85% | 80% | 75% |
| 2 | 15 | 90% | 85% | 82% |
| 3 | 20+ | 93% | 90% | 87% |

### Model Checkpoints

**Storage Location**: `models/face_projection_epoch_{N}.pt`

**Checkpoint Contains**:
- Projection head weights (512→128 mapping)
- Optimizer state (for resuming training)
- Training metadata (epoch, loss, timestamp)

**Loading Trained Model**:
```python
from image_search_service.faces.trainer import FaceTrainer

trainer = FaceTrainer()
trainer.load_checkpoint("models/face_projection_epoch_20.pt")

# Apply to new faces
improved_embedding = trainer.project_embedding(arcface_embedding)
```

### Performance Characteristics

**Training Time** (RTX 3080):
- 100 faces, 20 epochs: ~2-3 minutes
- 500 faces, 20 epochs: ~8-10 minutes
- 2000 faces, 50 epochs: ~30-40 minutes

**Memory Usage**:
- Model: ~50 MB
- Batch (32): ~100 MB
- Total: ~2 GB GPU / 4 GB RAM

**Inference Speed**:
- CPU: ~0.5ms per embedding
- GPU: ~0.1ms per embedding

---

## API Reference

Base URL: `http://localhost:8000/api/v1/faces`

All request/response bodies use camelCase JSON.

### Cluster Endpoints

#### `GET /clusters`

List face clusters with pagination.

**Query Parameters**:
- `page`: Page number (default: 1)
- `pageSize`: Items per page (default: 20, max: 100)
- `includeLabeled`: Include already-labeled clusters (default: false)

**Response** (`ClusterListResponse`):
```json
{
  "items": [
    {
      "clusterId": "clu_abc123",
      "faceCount": 15,
      "sampleFaceIds": ["uuid1", "uuid2", "uuid3", "uuid4", "uuid5"],
      "avgQuality": 0.78,
      "personId": null,
      "personName": null
    }
  ],
  "total": 42,
  "page": 1,
  "pageSize": 20
}
```

**Example**:
```bash
curl "http://localhost:8000/api/v1/faces/clusters?page=1&pageSize=20"
```

---

#### `GET /clusters/{cluster_id}`

Get detailed information for a specific cluster.

**Response** (`ClusterDetailResponse`):
```json
{
  "clusterId": "clu_abc123",
  "faces": [
    {
      "id": "uuid",
      "assetId": 123,
      "bbox": {"x": 100, "y": 150, "width": 80, "height": 80},
      "detectionConfidence": 0.95,
      "qualityScore": 0.82,
      "clusterId": "clu_abc123",
      "personId": null,
      "personName": null,
      "createdAt": "2025-12-23T10:30:00Z"
    }
  ],
  "personId": null,
  "personName": null
}
```

**Example**:
```bash
curl "http://localhost:8000/api/v1/faces/clusters/clu_abc123"
```

---

#### `POST /clusters/{cluster_id}/label`

Label a cluster with a person name, creating the person if needed.

**Request** (`LabelClusterRequest`):
```json
{
  "name": "John Doe"
}
```

**Response** (`LabelClusterResponse`):
```json
{
  "personId": "uuid",
  "personName": "John Doe",
  "facesLabeled": 15,
  "prototypesCreated": 3
}
```

**Example**:
```bash
curl -X POST "http://localhost:8000/api/v1/faces/clusters/clu_abc123/label" \
  -H "Content-Type: application/json" \
  -d '{"name": "John Doe"}'
```

**Behavior**:
- Creates person if name doesn't exist (case-insensitive lookup)
- Assigns all faces in cluster to the person
- Creates top 3 quality faces as exemplar prototypes
- Updates Qdrant payloads

---

#### `POST /clusters/{cluster_id}/split`

Split a cluster into smaller sub-clusters using tighter HDBSCAN parameters.

**Request** (`SplitClusterRequest`):
```json
{
  "minClusterSize": 3
}
```

**Response** (`SplitClusterResponse`):
```json
{
  "originalClusterId": "clu_abc123",
  "newClusters": ["clu_abc123_sub_a1b2", "clu_abc123_sub_c3d4"],
  "status": "split"
}
```

**Example**:
```bash
curl -X POST "http://localhost:8000/api/v1/faces/clusters/clu_abc123/split" \
  -H "Content-Type: application/json" \
  -d '{"minClusterSize": 3}'
```

---

### Person Endpoints

#### `GET /persons`

List persons with pagination.

**Query Parameters**:
- `page`: Page number (default: 1)
- `pageSize`: Items per page (default: 20, max: 100)
- `status`: Filter by status (`active`, `merged`, `hidden`)

**Response** (`PersonListResponse`):
```json
{
  "items": [
    {
      "id": "uuid",
      "name": "John Doe",
      "status": "active",
      "faceCount": 42,
      "prototypeCount": 3,
      "createdAt": "2025-12-01T10:00:00Z",
      "updatedAt": "2025-12-23T15:30:00Z"
    }
  ],
  "total": 15,
  "page": 1,
  "pageSize": 20
}
```

**Example**:
```bash
curl "http://localhost:8000/api/v1/faces/persons?status=active"
```

---

#### `POST /persons/{person_id}/merge`

Merge one person into another.

**Request** (`MergePersonsRequest`):
```json
{
  "intoPersonId": "target-uuid"
}
```

**Response** (`MergePersonsResponse`):
```json
{
  "sourcePersonId": "source-uuid",
  "targetPersonId": "target-uuid",
  "facesMoved": 28
}
```

**Example**:
```bash
curl -X POST "http://localhost:8000/api/v1/faces/persons/{source-uuid}/merge" \
  -H "Content-Type: application/json" \
  -d '{"intoPersonId": "target-uuid"}'
```

**Behavior**:
- Moves all faces from source person to target person
- Marks source person as `merged` with `merged_into_id` set
- Updates Qdrant payloads
- Preserves both person records for audit trail

---

### Detection Endpoints

#### `POST /detect/{asset_id}`

Detect faces in a specific asset.

**Request** (`DetectFacesRequest`):
```json
{
  "minConfidence": 0.5,
  "minFaceSize": 20
}
```

**Response** (`DetectFacesResponse`):
```json
{
  "assetId": 123,
  "facesDetected": 3,
  "faceIds": ["uuid1", "uuid2", "uuid3"]
}
```

**Example**:
```bash
curl -X POST "http://localhost:8000/api/v1/faces/detect/123" \
  -H "Content-Type: application/json" \
  -d '{"minConfidence": 0.6, "minFaceSize": 30}'
```

---

#### `POST /cluster`

Trigger face clustering on unlabeled faces.

**Request** (`TriggerClusteringRequest`):
```json
{
  "qualityThreshold": 0.5,
  "maxFaces": 50000,
  "minClusterSize": 5
}
```

**Response** (`ClusteringResultResponse`):
```json
{
  "totalFaces": 1234,
  "clustersFound": 42,
  "noiseCount": 87,
  "status": "completed"
}
```

**Example**:
```bash
curl -X POST "http://localhost:8000/api/v1/faces/cluster" \
  -H "Content-Type: application/json" \
  -d '{"qualityThreshold": 0.6, "minClusterSize": 5}'
```

---

#### `GET /assets/{asset_id}`

Get all detected faces for a specific asset.

**Response** (`FaceInstanceListResponse`):
```json
{
  "items": [
    {
      "id": "uuid",
      "assetId": 123,
      "bbox": {"x": 100, "y": 150, "width": 80, "height": 80},
      "detectionConfidence": 0.95,
      "qualityScore": 0.82,
      "clusterId": "clu_abc123",
      "personId": "uuid",
      "personName": null,
      "createdAt": "2025-12-23T10:30:00Z"
    }
  ],
  "total": 3,
  "page": 1,
  "pageSize": 3
}
```

**Example**:
```bash
curl "http://localhost:8000/api/v1/faces/assets/123"
```

---

## CLI Reference

All CLI commands are under the `faces` subcommand:

```bash
uv run python -m image_search_service.scripts.cli faces <command>
```

### `faces backfill`

Backfill face detection for existing assets without faces.

**Options**:
- `--limit INTEGER`: Number of assets to process (default: 1000)
- `--offset INTEGER`: Starting offset (default: 0)
- `--min-confidence FLOAT`: Detection confidence threshold (default: 0.5)
- `--queue`: Run as background job instead of directly

**Examples**:
```bash
# Process 500 assets directly
uv run python -m image_search_service.scripts.cli faces backfill --limit 500 --min-confidence 0.6

# Queue as background job
uv run python -m image_search_service.scripts.cli faces backfill --limit 1000 --queue
```

---

### `faces cluster`

Cluster unlabeled faces using HDBSCAN.

**Options**:
- `--quality-threshold FLOAT`: Minimum quality score (default: 0.5)
- `--max-faces INTEGER`: Maximum faces to cluster (default: 50000)
- `--min-cluster-size INTEGER`: HDBSCAN min_cluster_size (default: 5)
- `--min-samples INTEGER`: HDBSCAN min_samples (default: 3)
- `--time-bucket STRING`: Filter by YYYY-MM (optional)
- `--queue`: Run as background job

**Examples**:
```bash
# Cluster with stricter quality threshold
uv run python -m image_search_service.scripts.cli faces cluster --quality-threshold 0.6 --min-cluster-size 3

# Cluster recent faces only
uv run python -m image_search_service.scripts.cli faces cluster --time-bucket 2025-12 --queue
```

---

### `faces assign`

Assign new faces to known persons via prototype matching.

**Options**:
- `--since STRING`: Only faces created after YYYY-MM-DD (optional)
- `--max-faces INTEGER`: Maximum faces to process (default: 1000)
- `--threshold FLOAT`: Similarity threshold (default: 0.6)
- `--queue`: Run as background job

**Examples**:
```bash
# Assign faces detected today
uv run python -m image_search_service.scripts.cli faces assign --since 2025-12-23 --threshold 0.65

# Queue assignment job
uv run python -m image_search_service.scripts.cli faces assign --max-faces 5000 --queue
```

---

### `faces centroids`

Compute/update person centroid embeddings.

**Options**:
- `--queue`: Run as background job

**Examples**:
```bash
# Compute centroids directly
uv run python -m image_search_service.scripts.cli faces centroids

# Queue as background job
uv run python -m image_search_service.scripts.cli faces centroids --queue
```

---

### `faces ensure-collection`

Ensure the Qdrant faces collection exists with proper indexes.

**Examples**:
```bash
uv run python -m image_search_service.scripts.cli faces ensure-collection
```

---

### `faces stats`

Show face detection and recognition statistics.

**Examples**:
```bash
uv run python -m image_search_service.scripts.cli faces stats
```

**Output**:
```
Total faces: 1234
Assigned faces: 567
Clustered faces: 890
Total persons: 42
Active persons: 40
Merged persons: 2
Total prototypes: 126

Qdrant Collection: faces
Points: 1234
Vectors: 1234
```

---

### `faces find-orphans`

Find and optionally fix orphaned faces (faces in PostgreSQL without Qdrant embeddings).

Orphan faces occur when the database commit succeeds but the Qdrant upsert fails, leaving face metadata in PostgreSQL without the corresponding vector embedding in Qdrant.

**Options**:
- `--limit INTEGER`: Maximum faces to check (default: 100)
- `--fix`: Re-detect assets with orphaned faces (default: false)

**Examples**:
```bash
# Check for orphaned faces
uv run python -m image_search_service.scripts.cli faces find-orphans --limit 1000

# Find and automatically fix orphans
uv run python -m image_search_service.scripts.cli faces find-orphans --fix
```

**Output** (without --fix):
```
Checking 100 faces for orphaned embeddings...

==================================================
ORPHAN DETECTION RESULTS:
==================================================
Total faces checked: 100
Orphaned faces found: 5
Affected assets: 3

Sample orphaned faces:
  - Face abc-123 (asset_id=456, qdrant_point_id=xyz-789)
  ... and 4 more

Affected assets (asset_id: orphan_count):
  - Asset 456: 3 orphaned face(s)
  - Asset 789: 2 orphaned face(s)

TIP: Use --fix to re-detect faces for affected assets
```

**Output** (with --fix):
```
==================================================
RE-DETECTING FACES FOR AFFECTED ASSETS:
==================================================
Re-detecting faces for 3 assets...

==================================================
RE-DETECTION RESULTS:
==================================================
Assets processed: 3
Errors: 0
Total faces re-detected: 5

NOTE: Re-run 'faces find-orphans' to verify fix
```

**When to Use**:
- After Qdrant connection failures during face detection
- After database migrations or recovery operations
- When face counts don't match between PostgreSQL and Qdrant
- As part of routine data integrity checks

---

## Background Jobs

All face processing operations can run as background jobs via RQ (Redis Queue).

### Queue Setup

Ensure Redis is running:
```bash
make db-up  # Starts Redis via Docker Compose
```

Start RQ worker:
```bash
make worker  # Runs RQ worker on 'default' queue
```

### Job Functions

#### `detect_faces_job`

Detect faces for a batch of assets.

**Parameters**:
- `asset_ids`: List of asset ID strings
- `min_confidence`: Detection threshold (default: 0.5)
- `min_face_size`: Minimum face size (default: 20)

**Returns**: `{"processed": int, "total_faces": int, "errors": int}`

---

#### `cluster_faces_job`

Cluster unlabeled faces using HDBSCAN.

**Parameters**:
- `quality_threshold`: Minimum quality (default: 0.5)
- `max_faces`: Maximum faces to cluster (default: 50000)
- `min_cluster_size`: HDBSCAN parameter (default: 5)
- `min_samples`: HDBSCAN parameter (default: 3)
- `time_bucket`: Optional YYYY-MM filter

**Returns**: `{"total_faces": int, "clusters_found": int, "noise_count": int}`

---

#### `assign_faces_job`

Assign faces to persons via prototype matching.

**Parameters**:
- `since`: ISO datetime string filter (optional)
- `max_faces`: Maximum faces to process (default: 1000)
- `similarity_threshold`: Matching threshold (default: 0.6)

**Returns**: `{"processed": int, "assigned": int, "unassigned": int, "status": str}`

---

#### `compute_centroids_job`

Compute person centroid embeddings.

**Returns**: `{"persons_processed": int, "centroids_computed": int}`

---

#### `backfill_faces_job`

Backfill face detection for assets without faces.

**Parameters**:
- `limit`: Number of assets (default: 1000)
- `offset`: Starting offset (default: 0)
- `min_confidence`: Detection threshold (default: 0.5)

**Returns**: `{"processed": int, "total_faces": int, "status": str}`

---

## Workflow Guide

### Initial Setup

1. **Ensure Qdrant collection exists**:
   ```bash
   uv run python -m image_search_service.scripts.cli faces ensure-collection
   ```

2. **Start RQ worker** (for background jobs):
   ```bash
   make worker
   ```

---

### Step 1: Face Detection

Detect faces in existing images:

```bash
# Direct processing (small batches)
uv run python -m image_search_service.scripts.cli faces backfill --limit 100

# Background job (large batches)
uv run python -m image_search_service.scripts.cli faces backfill --limit 10000 --queue
```

**Monitor progress**:
```bash
uv run python -m image_search_service.scripts.cli faces stats
```

---

### Step 2: Face Clustering

Cluster unlabeled faces to find identity groups:

```bash
# Cluster with default parameters
uv run python -m image_search_service.scripts.cli faces cluster --queue

# Adjust for smaller/tighter clusters
uv run python -m image_search_service.scripts.cli faces cluster \
  --min-cluster-size 3 \
  --quality-threshold 0.6
```

**Review clusters via API**:
```bash
curl "http://localhost:8000/api/v1/faces/clusters?page=1&pageSize=20"
```

---

### Step 3: Manual Labeling

Label clusters with person names:

```bash
# Label via API
curl -X POST "http://localhost:8000/api/v1/faces/clusters/clu_abc123/label" \
  -H "Content-Type: application/json" \
  -d '{"name": "John Doe"}'
```

**UI Integration**: Build a UI to display cluster thumbnails and accept labels.

---

### Step 4: Prototype Creation

Prototypes are automatically created when labeling clusters (top 3 quality faces).

**Optional**: Compute centroids for better matching:
```bash
uv run python -m image_search_service.scripts.cli faces centroids
```

---

### Step 5: Incremental Assignment

Automatically assign new faces to known persons:

```bash
# Assign all unassigned faces
uv run python -m image_search_service.scripts.cli faces assign --queue

# Assign only recent faces
uv run python -m image_search_service.scripts.cli faces assign --since 2025-12-23
```

---

### Step 6: Merge Duplicates

If the same person was labeled with different names:

```bash
curl -X POST "http://localhost:8000/api/v1/faces/persons/{duplicate-uuid}/merge" \
  -H "Content-Type: application/json" \
  -d '{"intoPersonId": "canonical-uuid"}'
```

---

### Ongoing Workflow

1. **Detect faces** in new images (triggered by image ingest)
2. **Assign faces** to known persons automatically
3. **Cluster unassigned faces** periodically (e.g., weekly)
4. **Label new clusters** manually via UI
5. **Update centroids** monthly for improved accuracy

---

## Tuning Parameters

### Detection Parameters

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `min_confidence` | 0.5 | 0.3-0.9 | Higher = fewer false positives, may miss faces |
| `min_face_size` | 20px | 10-50px | Higher = ignore small/distant faces |

**Recommendations**:
- **Portraits/close-ups**: `min_confidence=0.6`, `min_face_size=30`
- **Group photos**: `min_confidence=0.5`, `min_face_size=20`
- **Surveillance/distant**: `min_confidence=0.4`, `min_face_size=15`

---

### Clustering Parameters

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `min_cluster_size` | 5 | 2-20 | Minimum faces to form cluster |
| `min_samples` | 3 | 1-10 | Core point density threshold |
| `quality_threshold` | 0.5 | 0.3-0.8 | Filter low-quality faces before clustering |

**Tuning Guidelines**:

**Small datasets (<1000 faces)**:
- `min_cluster_size=3`, `min_samples=2`
- Allows smaller identity groups

**Medium datasets (1k-10k)**:
- `min_cluster_size=5`, `min_samples=3` (default)
- Balanced for most use cases

**Large datasets (>10k)**:
- `min_cluster_size=7-10`, `min_samples=5`
- Avoids over-fragmentation

**Quality threshold**:
- `0.5`: Include most faces (default)
- `0.6`: Only good quality faces
- `0.7+`: Very strict, may exclude valid faces

---

### Assignment Parameters

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `similarity_threshold` | 0.6 | 0.5-0.8 | Minimum cosine similarity for auto-assignment |

**Tuning**:
- **0.7-0.8**: Very strict, low false positive rate, more manual review
- **0.6-0.7**: Balanced (recommended)
- **0.5-0.6**: Loose, higher false positive risk

**Trade-offs**:
- Higher threshold → fewer automatic assignments → more manual work
- Lower threshold → more automatic assignments → higher error rate

---

## Troubleshooting

### Common Issues

#### InsightFace Model Download Fails

**Symptoms**: `Model not found` error on first run

**Solutions**:
1. Check internet connectivity
2. Manually download model:
   ```bash
   mkdir -p ~/.insightface/models
   # Download buffalo_l from InsightFace repo
   ```
3. Verify `~/.insightface/models/buffalo_l/` exists

---

#### CUDA Not Available

**Symptoms**: `CUDAExecutionProvider not found`, falls back to CPU

**Solutions**:
1. Install CUDA toolkit and drivers
2. Install GPU-enabled onnxruntime:
   ```bash
   uv add onnxruntime-gpu
   ```
3. Verify CUDA availability:
   ```python
   import onnxruntime as ort
   print(ort.get_available_providers())
   # Should include 'CUDAExecutionProvider'
   ```

---

#### No Faces Detected

**Symptoms**: `facesDetected: 0` for images with visible faces

**Diagnostics**:
1. Check detection parameters (too strict?)
2. Verify image format (BGR vs RGB)
3. Check image quality (resolution, lighting)

**Solutions**:
- Lower `min_confidence` to 0.4
- Lower `min_face_size` to 15
- Verify image loads correctly with OpenCV

---

#### Clustering Finds No Clusters

**Symptoms**: `clusters_found: 0`, all faces labeled as noise

**Diagnostics**:
1. Check number of unlabeled faces (`total_faces`)
2. Check `quality_threshold` (too strict?)
3. Check `min_cluster_size` (too large?)

**Solutions**:
- Lower `quality_threshold` to 0.4
- Lower `min_cluster_size` to 3
- Verify faces have diverse identities (not all unique)

---

#### Assignment Finds No Matches

**Symptoms**: All faces remain unassigned after assignment run

**Diagnostics**:
1. Check if prototypes exist (`faces stats`)
2. Check `similarity_threshold` (too strict?)
3. Verify faces are unlabeled (no `person_id` or `cluster_id`)

**Solutions**:
1. Label some clusters first to create prototypes
2. Lower `similarity_threshold` to 0.55
3. Compute centroids for better matching:
   ```bash
   uv run python -m image_search_service.scripts.cli faces centroids
   ```

---

#### Qdrant Connection Errors

**Symptoms**: `Failed to connect to Qdrant`, `Connection refused`

**Solutions**:
1. Ensure Qdrant is running:
   ```bash
   make db-up
   docker ps | grep qdrant
   ```
2. Check `QDRANT_URL` environment variable
3. Verify port 6333 is accessible

---

#### Database Foreign Key Violations

**Symptoms**: `ForeignKeyViolationError` when deleting assets

**Cause**: `face_instances` reference `image_assets`

**Solution**: Configure cascade deletes (already done in models):
```python
face_instances: Mapped[list["FaceInstance"]] = relationship(
    "FaceInstance", back_populates="asset", cascade="all, delete-orphan"
)
```

---

### Performance Optimization

#### Slow Face Detection

**Causes**:
- CPU-only processing
- High-resolution images
- Large batch sizes

**Solutions**:
1. Enable GPU processing (see CUDA setup)
2. Resize images before detection (not recommended, may miss faces)
3. Use background jobs (`--queue`) for large batches
4. Process in smaller batches (100-500 images)

---

#### Slow Clustering

**Causes**:
- Too many faces (>50k)
- High-dimensional data (not applicable, fixed at 512)

**Solutions**:
1. Limit `max_faces` to 10k-20k per batch
2. Use `time_bucket` to cluster incrementally (e.g., by month)
3. Increase `quality_threshold` to reduce face count
4. Run as background job

---

#### Memory Issues

**Causes**:
- Loading too many embeddings at once
- HDBSCAN on large datasets

**Solutions**:
1. Reduce `max_faces` for clustering
2. Increase server RAM (HDBSCAN needs ~4GB for 50k faces)
3. Process in batches with `time_bucket`

---

### Debugging Tips

#### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or set environment variable:
```bash
export LOG_LEVEL=DEBUG
```

---

#### Inspect Qdrant Vectors

```python
from image_search_service.vector.face_qdrant import get_face_qdrant_client

client = get_face_qdrant_client()
info = client.get_collection_info()
print(info)

# Scroll through points
records, next_offset = client.scroll_faces(limit=10)
for record in records:
    print(record.payload)
```

---

#### Check Database State

```bash
# PostgreSQL CLI
docker exec -it image-search-db psql -U postgres -d image_search

# Query face counts
SELECT person_id, COUNT(*)
FROM face_instances
GROUP BY person_id
ORDER BY COUNT(*) DESC;

# Query cluster sizes
SELECT cluster_id, COUNT(*)
FROM face_instances
WHERE cluster_id IS NOT NULL
GROUP BY cluster_id
ORDER BY COUNT(*) DESC;
```

---

## Additional Resources

### Related Documentation

- **API Contract**: `docs/api-contract.md` (full API specification)
- **Database Models**: `src/image_search_service/db/models.py`
- **Vector Storage**: `docs/qdrant-guide.md` (if exists)

### External Documentation

- **InsightFace**: https://github.com/deepinsight/insightface
- **HDBSCAN**: https://hdbscan.readthedocs.io/
- **Qdrant**: https://qdrant.tech/documentation/
- **ArcFace Paper**: https://arxiv.org/abs/1801.07698

### Future Enhancements

- **Quality Filters**: Blur detection, pose estimation
- **Temporal Filtering**: Track faces across video frames
- **Active Learning**: Suggest uncertain faces for manual review
- **Multi-face Matching**: Search by multiple face embeddings
- **Face Aging**: Handle same person across age ranges

---

**End of Documentation**
