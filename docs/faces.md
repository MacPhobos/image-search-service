# Face Detection and Recognition Pipeline

**Version**: 1.0
**Last Updated**: 2025-12-23

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Database Schema](#database-schema)
4. [Vector Storage](#vector-storage)
5. [Face Detection Pipeline](#face-detection-pipeline)
6. [Clustering](#clustering)
7. [Person Recognition](#person-recognition)
8. [API Reference](#api-reference)
9. [CLI Reference](#cli-reference)
10. [Background Jobs](#background-jobs)
11. [Workflow Guide](#workflow-guide)
12. [Tuning Parameters](#tuning-parameters)
13. [Troubleshooting](#troubleshooting)

---

## Overview

The face detection and recognition pipeline provides automated face detection, unsupervised clustering, and incremental person recognition capabilities for the image search system.

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
- **Clustering**: HDBSCAN unsupervised clustering for identity grouping
- **Person Recognition**: Prototype-based incremental assignment
- **Quality Scoring**: Automatic quality assessment based on size and confidence
- **Idempotent Processing**: Re-running detection won't create duplicates

### Components

- **Detector** (`faces/detector.py`): Face detection and embedding extraction
- **Service** (`faces/service.py`): High-level orchestration layer
- **Clusterer** (`faces/clusterer.py`): HDBSCAN clustering for identity grouping
- **Assigner** (`faces/assigner.py`): Incremental person assignment via prototypes
- **Qdrant Client** (`vector/face_qdrant.py`): Vector database operations
- **API Routes** (`api/routes/faces.py`): REST endpoints
- **CLI Commands** (`scripts/faces.py`): Command-line interface
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
uv run python -m image_search_service.cli faces ensure-collection

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
uv run python -m image_search_service.cli faces <command>
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
uv run python -m image_search_service.cli faces backfill --limit 500 --min-confidence 0.6

# Queue as background job
uv run python -m image_search_service.cli faces backfill --limit 1000 --queue
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
uv run python -m image_search_service.cli faces cluster --quality-threshold 0.6 --min-cluster-size 3

# Cluster recent faces only
uv run python -m image_search_service.cli faces cluster --time-bucket 2025-12 --queue
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
uv run python -m image_search_service.cli faces assign --since 2025-12-23 --threshold 0.65

# Queue assignment job
uv run python -m image_search_service.cli faces assign --max-faces 5000 --queue
```

---

### `faces centroids`

Compute/update person centroid embeddings.

**Options**:
- `--queue`: Run as background job

**Examples**:
```bash
# Compute centroids directly
uv run python -m image_search_service.cli faces centroids

# Queue as background job
uv run python -m image_search_service.cli faces centroids --queue
```

---

### `faces ensure-collection`

Ensure the Qdrant faces collection exists with proper indexes.

**Examples**:
```bash
uv run python -m image_search_service.cli faces ensure-collection
```

---

### `faces stats`

Show face detection and recognition statistics.

**Examples**:
```bash
uv run python -m image_search_service.cli faces stats
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
   uv run python -m image_search_service.cli faces ensure-collection
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
uv run python -m image_search_service.cli faces backfill --limit 100

# Background job (large batches)
uv run python -m image_search_service.cli faces backfill --limit 10000 --queue
```

**Monitor progress**:
```bash
uv run python -m image_search_service.cli faces stats
```

---

### Step 2: Face Clustering

Cluster unlabeled faces to find identity groups:

```bash
# Cluster with default parameters
uv run python -m image_search_service.cli faces cluster --queue

# Adjust for smaller/tighter clusters
uv run python -m image_search_service.cli faces cluster \
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
uv run python -m image_search_service.cli faces centroids
```

---

### Step 5: Incremental Assignment

Automatically assign new faces to known persons:

```bash
# Assign all unassigned faces
uv run python -m image_search_service.cli faces assign --queue

# Assign only recent faces
uv run python -m image_search_service.cli faces assign --since 2025-12-23
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
   uv run python -m image_search_service.cli faces centroids
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
