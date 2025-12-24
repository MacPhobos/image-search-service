# Dual-Mode Face Clustering Implementation Plan

**Version**: 1.0
**Created**: 2024-12-24
**Target**: Image Search Service - Face Recognition Enhancement

---

## Executive Summary

This document provides a comprehensive implementation plan for enhancing the face recognition system with **dual-mode clustering**:

1. **Supervised Mode**: Assigns faces to known Person entities using similarity matching with trained embeddings
2. **Unsupervised Mode**: Clusters unknown faces by similarity for progressive labeling and discovery

The goal is to maximize clustering accuracy for known people while maintaining organization of unknown faces for easy labeling and discovery.

---

## Current System Analysis

### Architecture Overview

The current system implements a **cluster-first, label-later** workflow:

```
Image Asset → Face Detection → Embedding → Qdrant Storage → HDBSCAN Clustering → Manual Labeling → Prototype Assignment
```

### Existing Components

#### Database Models (PostgreSQL)

- **`persons`**: Person entities with name, status (active/merged/hidden)
- **`face_instances`**: Detected faces with bbox, confidence, quality_score, cluster_id, person_id
- **`person_prototypes`**: Prototype embeddings (centroid or exemplar) for person recognition
- **`face_assignment_events`**: Audit log for face assignment operations

#### Vector Storage (Qdrant)

- **Collection**: `faces` (512-dim ArcFace embeddings, cosine similarity)
- **Payload**: asset_id, face_instance_id, person_id, cluster_id, quality_score, detection_confidence

#### Processing Pipeline

1. **Face Detection** (`faces/detector.py`): InsightFace buffalo_l model
   - RetinaFace detector with 5-point landmarks
   - 512-dim ArcFace embeddings (normalized)
   - Quality scoring based on size + confidence

2. **Face Service** (`faces/service.py`): High-level orchestration
   - Batch processing of assets
   - Idempotent face detection (won't create duplicates)
   - Database + Qdrant synchronization

3. **Clustering** (`faces/clusterer.py`): HDBSCAN unsupervised clustering
   - Groups unlabeled faces by similarity
   - Configurable min_cluster_size (default: 5), min_samples (default: 3)
   - Uses euclidean distance on normalized vectors
   - Handles noise/outliers (label: -1)

4. **Assignment** (`faces/assigner.py`): Prototype-based matching
   - Matches new faces to known person prototypes
   - Uses cosine similarity threshold (default: 0.6)
   - Computes person centroids from labeled faces

### Current Workflow

```bash
# 1. Detect faces in images
make faces-backfill LIMIT=1000

# 2. Cluster unlabeled faces
make faces-cluster MAX_FACES=50000

# 3. Label clusters via API
curl -X POST /api/v1/faces/clusters/{cluster_id}/label -d '{"name": "Alice"}'

# 4. Assign new faces to known people
make faces-assign MAX_FACES=1000

# 5. Compute centroids for better matching
make faces-centroids
```

### Limitations

1. **No Identity Awareness in Clustering**: HDBSCAN groups by similarity without knowing person identity
   - Results in split clusters (same person in multiple clusters)
   - Mixed clusters (multiple people in one cluster)
   - Poor precision/recall for person identification

2. **No Model Training**: Pre-trained embeddings don't learn from user labels
   - Labeling clusters doesn't improve future clustering
   - No fine-tuning on dataset-specific facial characteristics
   - No adaptation to user's specific people

3. **Binary Workflow**: Faces are either "clustered" or "assigned to person"
   - No simultaneous handling of known vs unknown faces
   - Reclustering affects all faces equally

---

## Solution Architecture

### Dual-Mode Clustering Design

```
┌─────────────────────────────────────────────────────────────┐
│                     Face Detection                          │
│              (InsightFace buffalo_l)                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Store in PostgreSQL + Qdrant                   │
│         (512-dim ArcFace embeddings)                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌───────────────────┐   ┌────────────────────┐
│  SUPERVISED MODE  │   │ UNSUPERVISED MODE  │
│                   │   │                    │
│ Match to Known    │   │ Cluster Unknown    │
│ Person Entities   │   │ Faces by Similarity│
│                   │   │                    │
│ • Prototype Match │   │ • HDBSCAN         │
│ • Threshold: 0.7  │   │ • min_size: 3     │
│ • Cosine Sim      │   │ • Noise handling  │
└────────┬──────────┘   └──────────┬─────────┘
         │                         │
         └────────────┬────────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │   Store Assignments    │
         │                        │
         │ • person_* clusters    │
         │ • unknown_* clusters   │
         │ • unknown_noise_* solo │
         └────────────────────────┘
```

### Key Improvements

1. **Identity-Aware Clustering**: Known people get precise assignments
2. **Organized Unknown Faces**: Unmapped faces stay grouped by similarity
3. **Progressive Learning**: Train model on labeled data → better boundaries
4. **No Data Loss**: Unknown faces aren't forced into wrong person clusters
5. **Incremental Workflow**: Label → Train → Recluster → Repeat

---

## Implementation Tasks

### Phase 1: Configuration Updates

**File**: `src/image_search_service/core/config.py`

Add new configuration fields for dual-mode clustering and training:

```python
# Face recognition model settings
face_model_name: str = Field(default="buffalo_l", alias="FACE_MODEL_NAME")
face_model_checkpoint: str = Field(default="", alias="FACE_MODEL_CHECKPOINT")
face_training_enabled: bool = Field(default=False, alias="FACE_TRAINING_ENABLED")

# Training hyperparameters (Triplet Loss)
face_triplet_margin: float = Field(default=0.2, alias="FACE_TRIPLET_MARGIN")
face_training_epochs: int = Field(default=20, alias="FACE_TRAINING_EPOCHS")
face_batch_size: int = Field(default=32, alias="FACE_BATCH_SIZE")
face_learning_rate: float = Field(default=0.0001, alias="FACE_LEARNING_RATE")

# Supervised clustering (known people)
face_person_match_threshold: float = Field(default=0.7, alias="FACE_PERSON_MATCH_THRESHOLD")

# Unsupervised clustering (unknown faces)
face_unknown_clustering_method: str = Field(default="hdbscan", alias="FACE_UNKNOWN_CLUSTERING_METHOD")
face_unknown_min_cluster_size: int = Field(default=3, alias="FACE_UNKNOWN_MIN_CLUSTER_SIZE")
face_unknown_eps: float = Field(default=0.5, alias="FACE_UNKNOWN_EPS")
```

**Dependencies**: Add to `pyproject.toml`:
```toml
dependencies = [
    # ...existing dependencies...
    "torch>=2.0.0",           # Already present
    "scikit-learn>=1.3.0",    # For AgglomerativeClustering
]
```

### Phase 2: Dual-Mode Clustering Service

**New File**: `src/image_search_service/faces/dual_clusterer.py`

Implement the core dual-mode clustering logic:

#### Class: `DualModeClusterer`

```python
class DualModeClusterer:
    """Dual-mode face clustering: supervised + unsupervised."""

    def __init__(
        self,
        db_session: SyncSession,
        person_match_threshold: float = 0.7,
        unknown_min_cluster_size: int = 3,
        unknown_method: str = "hdbscan",
    ):
        """Initialize dual-mode clusterer."""
        pass

    def cluster_all_faces(self) -> dict:
        """Run dual-mode clustering on all faces.

        Returns:
            {
                'assigned_to_people': int,
                'unknown_clusters': int,
                'total_processed': int,
                'noise_faces': int,
            }
        """
        pass
```

#### Key Methods

1. **`dual_mode_face_clustering()`**: Main orchestrator
   - Query all faces (labeled and unlabeled)
   - Separate by person_id (labeled vs unlabeled)
   - Call supervised assignment
   - Call unsupervised clustering on remaining
   - Save results to database

2. **`assign_to_known_people()`**: Supervised assignment
   - Build person centroids from labeled faces
   - For each unlabeled face:
     - Calculate cosine similarity to all person centroids
     - If best_similarity >= threshold: assign to person
     - Else: add to unknown pool
   - Return (assigned_faces, still_unknown)

3. **`cluster_unknown_faces()`**: Unsupervised clustering
   - Support multiple algorithms: HDBSCAN, DBSCAN, AgglomerativeClustering
   - Use cosine distance metric
   - Label clusters: `unknown_cluster_N`
   - Handle noise: `unknown_noise_{face_id}`
   - Return {face_id: cluster_label}

4. **`save_dual_mode_results()`**: Persist to database
   - For assigned faces:
     - Find/create cluster for person
     - Update face_instances.person_id
     - Update face_instances.cluster_id
     - Update Qdrant payload
   - For unknown clusters:
     - Create cluster entries with cluster_label
     - Update face_instances.cluster_id
     - Update Qdrant payload

#### Database Changes

**Cluster Naming Convention**:
- `person_{uuid}`: Cluster assigned to a Person entity
- `unknown_cluster_{N}`: Unsupervised cluster of unknown faces
- `unknown_noise_{face_id}`: Singleton face (doesn't match anything)

**Note**: The current schema already supports this with nullable `person_id` in `face_instances`.

### Phase 3: Training System

**New File**: `src/image_search_service/faces/trainer.py`

Implement triplet loss training for person-aware embeddings:

#### Class: `FaceTrainer`

```python
class FaceTrainer:
    """Train face recognition model using triplet loss."""

    def __init__(
        self,
        db_session: SyncSession,
        settings: Settings,
    ):
        """Initialize face trainer."""
        pass

    def train_for_person_clustering(self) -> dict:
        """Train model to maximize person identity separation.

        Returns:
            {
                'epochs': int,
                'final_loss': float,
                'people_count': int,
                'faces_used': int,
                'checkpoint_path': str,
            }
        """
        pass
```

#### Key Components

1. **`TripletFaceDataset`**: PyTorch Dataset
   - Input: `{person_id: [(face_id, embedding), ...]}`
   - Generate triplets: (anchor, positive, negative)
     - Anchor: random face from person A
     - Positive: different face from same person A
     - Negative: face from different person B
   - Filter people with < 2 faces
   - Oversample to create enough training examples

2. **`get_labeled_faces_by_person()`**: Query builder
   ```sql
   SELECT p.id as person_id, f.id as face_id, f.qdrant_point_id
   FROM face_instances f
   JOIN persons p ON f.person_id = p.id
   WHERE p.status = 'active'
   ```
   - Group results by person_id
   - Return dict for TripletFaceDataset

3. **`fine_tune_for_person_clustering()`**: Training loop
   - Load labeled faces grouped by person
   - Validate: need 2+ people with 2+ faces each
   - Create TripletFaceDataset + DataLoader
   - Define model architecture:
     ```
     Pre-trained ArcFace (frozen)
       ↓
     Projection Head (trainable)
       Linear(512 → 256)
       ReLU
       Linear(256 → 128)
       L2 Normalize
     ```
   - Train with TripletMarginLoss:
     ```python
     loss = max(0, margin + dist(anchor, pos) - dist(anchor, neg))
     ```
   - Save checkpoint: `models/face_projection_{model_name}.pth`

4. **`regenerate_embeddings()`**: Re-embed all faces
   - Load trained projection head
   - For each face_instance:
     - Get original 512-dim embedding from Qdrant
     - Apply projection → 128-dim embedding
     - Update Qdrant with new embedding
   - **Note**: This requires Qdrant collection reconfiguration (512 → 128 dims)

#### Training Requirements

- **Minimum Data**: 10+ people with 5+ faces each (50+ total faces)
- **Recommended**: 20+ people with 10+ faces each
- **GPU**: Recommended but not required (CPU is slower)
- **Training Time**: ~5-10 minutes per 20 epochs on GPU

### Phase 4: CLI Commands

**File**: `src/image_search_service/scripts/faces.py`

Add new commands for dual-mode clustering and training:

#### Command: `faces cluster-dual`

```python
@faces_app.command("cluster-dual")
def cluster_faces_dual(
    person_threshold: float = typer.Option(0.7, help="Person match threshold"),
    unknown_method: str = typer.Option("hdbscan", help="Unknown clustering method"),
    unknown_min_size: int = typer.Option(3, help="Min cluster size for unknown"),
    queue: bool = typer.Option(False, help="Run as background job"),
) -> None:
    """Run dual-mode clustering (supervised + unsupervised).

    Example:
        faces cluster-dual --person-threshold 0.75 --unknown-method hdbscan
    """
```

#### Command: `faces train-matching`

```python
@faces_app.command("train-matching")
def train_person_matching(
    epochs: int = typer.Option(20, help="Training epochs"),
    margin: float = typer.Option(0.2, help="Triplet loss margin"),
    batch_size: int = typer.Option(32, help="Batch size"),
    learning_rate: float = typer.Option(0.0001, help="Learning rate"),
    queue: bool = typer.Option(False, help="Run as background job"),
) -> None:
    """Train model to improve person identity separation.

    Example:
        faces train-matching --epochs 30 --margin 0.25
    """
```

#### Command: `faces reassign`

```python
@faces_app.command("reassign")
def reassign_faces(
    person_threshold: float = typer.Option(0.75, help="Person match threshold"),
    use_trained_model: bool = typer.Option(False, help="Use trained projection head"),
    queue: bool = typer.Option(False, help="Run as background job"),
) -> None:
    """Reassign all faces using updated model/thresholds.

    Example:
        faces reassign --person-threshold 0.8 --use-trained-model
    """
```

### Phase 5: Makefile Targets

**File**: `Makefile`

Add convenient targets for new workflow:

```makefile
# Dual-mode clustering
faces-cluster-dual: ## Run dual-mode clustering (supervised + unsupervised)
	@echo "Running dual-mode clustering..."
	uv run python -m image_search_service.scripts.cli faces cluster-dual \
		--person-threshold $(or $(PERSON_THRESHOLD),0.7) \
		--unknown-method $(or $(UNKNOWN_METHOD),hdbscan)

# Train person matching model
faces-train-person-matching: ## Train model to improve person identity separation
	@echo "Training person matching model..."
	uv run python -m image_search_service.scripts.cli faces train-matching \
		--epochs $(or $(EPOCHS),20) \
		--margin $(or $(MARGIN),0.2)

# Reassign faces after training
faces-reassign-smart: ## Reassign all faces with trained model
	@echo "Reassigning faces with trained model..."
	uv run python -m image_search_service.scripts.cli faces reassign \
		--person-threshold $(or $(PERSON_THRESHOLD),0.75) \
		--use-trained-model

# Full iterative pipeline
faces-pipeline-iterative: faces-cluster-dual faces-train-person-matching faces-reassign-smart faces-stats ## Run iterative training pipeline
	@echo "Iterative face pipeline complete!"
```

### Phase 6: Background Jobs

**File**: `src/image_search_service/queue/face_jobs.py`

Add RQ job functions for async processing:

```python
def cluster_dual_job(
    person_threshold: float = 0.7,
    unknown_method: str = "hdbscan",
    unknown_min_size: int = 3,
) -> dict:
    """Background job for dual-mode clustering."""
    pass

def train_person_matching_job(
    epochs: int = 20,
    margin: float = 0.2,
    batch_size: int = 32,
    learning_rate: float = 0.0001,
) -> dict:
    """Background job for training person matching model."""
    pass

def reassign_faces_job(
    person_threshold: float = 0.75,
    use_trained_model: bool = False,
) -> dict:
    """Background job for reassigning faces."""
    pass
```

### Phase 7: API Enhancements (Optional)

**File**: `src/image_search_service/api/routes/faces.py`

Add endpoints for triggering dual-mode operations via API:

#### Endpoint: `POST /faces/cluster/dual`

```python
@router.post("/cluster/dual", response_model=ClusteringResultResponse)
async def trigger_dual_clustering(
    request: DualClusteringRequest,
    db: AsyncSession = Depends(get_db),
) -> ClusteringResultResponse:
    """Trigger dual-mode clustering operation."""
```

#### Endpoint: `POST /faces/train`

```python
@router.post("/train", response_model=TrainingResultResponse)
async def trigger_training(
    request: TrainingRequest,
    db: AsyncSession = Depends(get_db),
) -> TrainingResultResponse:
    """Trigger person matching training."""
```

---

## Workflow Guide

### Initial Setup (First Time)

```bash
# 1. Detect faces in all images
make faces-backfill LIMIT=10000

# 2. Run initial dual-mode clustering
make faces-cluster-dual PERSON_THRESHOLD=0.7

# 3. Check statistics
make faces-stats
```

### Label Unknown Clusters

```bash
# List unknown clusters via API
curl http://localhost:8000/api/v1/faces/clusters?include_labeled=false

# Label a cluster as a person
curl -X POST http://localhost:8000/api/v1/faces/clusters/unknown_cluster_5/label \
  -H "Content-Type: application/json" \
  -d '{"name": "Alice Johnson"}'

# Repeat for 10-20 clusters to get good training data
```

### Iterative Training Loop

```bash
# 1. Train model on labeled data (requires 10+ people with 5+ faces each)
make faces-train-person-matching EPOCHS=20

# 2. Re-cluster with improved model
make faces-cluster-dual PERSON_THRESHOLD=0.75

# 3. Check results
make faces-stats

# 4. Label more unknown clusters
# ... via API ...

# 5. Train again (repeat 2-3 times for best results)
make faces-train-person-matching EPOCHS=30
make faces-cluster-dual PERSON_THRESHOLD=0.8
```

### Progressive Improvement

After each iteration:
- **More faces** assigned to known people (higher recall)
- **Fewer false positives** in person clusters (higher precision)
- **Tighter unknown clusters** (easier to label)
- **Better person boundaries** (fewer split clusters)

### Production Workflow

```bash
# Daily/weekly: process new images
make faces-backfill LIMIT=1000
make faces-cluster-dual PERSON_THRESHOLD=0.75

# Monthly: retrain with new labels
make faces-train-person-matching EPOCHS=20
make faces-reassign-smart PERSON_THRESHOLD=0.75
```

---

## Expected Results

### Performance Metrics

#### Initial State (No Training)
- **Person Assignment Accuracy**: 70-80% (using pre-trained embeddings)
- **Unknown Cluster Purity**: 60-70% (similar-looking people grouped)
- **Recall**: 50-60% (many faces of known people in unknown clusters)

#### After 1st Training Iteration (10+ labeled people)
- **Person Assignment Accuracy**: 80-85%
- **Unknown Cluster Purity**: 75-80%
- **Recall**: 65-75%

#### After 3rd Training Iteration (20+ labeled people)
- **Person Assignment Accuracy**: 85-90%
- **Unknown Cluster Purity**: 80-85%
- **Recall**: 75-85%

#### Production Quality (50+ labeled people)
- **Person Assignment Accuracy**: 90-95%
- **Unknown Cluster Purity**: 85-90%
- **Recall**: 85-90%

### Qualitative Improvements

1. **Known People**: High precision clusters with minimal false positives
2. **Unknown People**: Organized by similarity, easy to review and label
3. **Edge Cases**: Better handling of profile views, occlusions, varying ages
4. **Dataset Adaptation**: Model learns specific characteristics of your dataset

---

## Technical Considerations

### Computational Requirements

#### Dual-Mode Clustering
- **CPU**: ~1-2 seconds per 1000 faces
- **Memory**: ~500MB peak (for 50K faces)
- **Disk I/O**: Minimal (Qdrant is in-memory)

#### Training
- **GPU**: 5-10 minutes per 20 epochs (RTX 3080)
- **CPU**: 30-60 minutes per 20 epochs (16-core)
- **Memory**: ~2GB for model + training data
- **Checkpoint Size**: ~50MB per saved model

#### Re-embedding (if using projection head)
- **GPU**: ~10 seconds per 1000 faces
- **CPU**: ~1 minute per 1000 faces
- **Note**: Only needed if you want to use trained embeddings for all operations

### Scalability

- **10K faces**: All operations in seconds/minutes
- **100K faces**: Clustering in minutes, training in hours
- **1M+ faces**: Consider batching, distributed processing

### Data Requirements

#### Minimum (Proof of Concept)
- **5 people** with **5 faces each** = 25 labeled faces
- Training possible but results modest

#### Recommended (Good Results)
- **10-15 people** with **10+ faces each** = 100-150 labeled faces
- Significant improvement over baseline

#### Production (Excellent Results)
- **20+ people** with **20+ faces each** = 400+ labeled faces
- Near-human-level accuracy for known people

### Limitations

1. **Cold Start**: Need labeled data before training improves results
2. **Class Imbalance**: People with many faces dominate training
3. **Model Drift**: Need periodic retraining as dataset grows
4. **Computational Cost**: Training adds overhead vs. simple clustering

---

## Testing Strategy

### Unit Tests

Create test files in `tests/faces/`:

#### `test_dual_clusterer.py`
- Test supervised assignment logic
- Test unsupervised clustering
- Test cluster naming conventions
- Test database updates

#### `test_trainer.py`
- Test triplet dataset generation
- Test training loop (mocked)
- Test checkpoint saving/loading
- Test embedding regeneration

#### `test_integration_dual_mode.py`
- End-to-end test: detection → dual clustering → training → reassignment
- Verify cluster assignments
- Verify person accuracy improvements

### Integration Tests

```python
def test_dual_mode_workflow():
    # 1. Create test assets with faces
    # 2. Run face detection
    # 3. Label some clusters
    # 4. Run dual-mode clustering
    # 5. Verify supervised assignments
    # 6. Verify unknown clusters
    # 7. Train model
    # 8. Verify improved accuracy
```

### Manual Testing Checklist

- [ ] Initial clustering creates unknown_* clusters
- [ ] Labeling cluster creates Person entity
- [ ] Dual clustering assigns faces to labeled person
- [ ] Unknown faces stay in unknown_* clusters
- [ ] Training requires sufficient labeled data
- [ ] Training improves person boundary accuracy
- [ ] Reassignment uses trained model correctly
- [ ] Statistics reflect dual-mode state
- [ ] API endpoints return correct cluster types

---

## Rollout Plan

### Phase 1: Core Implementation (Week 1)
- [ ] Add configuration fields
- [ ] Implement `DualModeClusterer` class
- [ ] Add CLI command: `faces cluster-dual`
- [ ] Add Makefile target: `faces-cluster-dual`
- [ ] Write unit tests
- [ ] Manual testing with sample dataset

### Phase 2: Training System (Week 2)
- [ ] Implement `FaceTrainer` class
- [ ] Add CLI command: `faces train-matching`
- [ ] Add Makefile target: `faces-train-person-matching`
- [ ] Write training tests
- [ ] Validate on sample dataset with 10+ people

### Phase 3: Integration (Week 3)
- [ ] Add background job support
- [ ] Implement API endpoints (optional)
- [ ] Add CLI command: `faces reassign`
- [ ] Create integration tests
- [ ] Performance testing with large datasets

### Phase 4: Documentation & Refinement (Week 4)
- [ ] Update `docs/faces.md` with dual-mode guide
- [ ] Create tutorial notebook/guide
- [ ] Performance optimization
- [ ] User feedback incorporation

---

## Migration Strategy

### Backward Compatibility

The dual-mode system is **fully backward compatible** with existing workflows:

1. **Existing clusters remain valid**: No migration needed
2. **Old commands still work**: `faces-cluster`, `faces-assign` unchanged
3. **New commands are additive**: `faces-cluster-dual` is a new option
4. **Database schema unchanged**: No migrations required

### Migration Path

Users can transition gradually:

```bash
# Old workflow (still works)
make faces-cluster
make faces-assign

# New workflow (coexists)
make faces-cluster-dual
make faces-train-person-matching
```

### Data Migration

If you want to convert existing clusters to dual-mode format:

```bash
# Re-run dual-mode clustering (preserves labeled assignments)
make faces-cluster-dual PERSON_THRESHOLD=0.7

# This will:
# - Keep all person_id assignments
# - Recluster unlabeled faces as unknown_*
# - No data loss
```

---

## Monitoring & Observability

### Key Metrics

Track these metrics to measure system effectiveness:

1. **Assignment Accuracy**: % of correctly assigned faces to people
2. **Unknown Cluster Purity**: % of faces in unknown clusters that belong together
3. **Person Recall**: % of person's faces correctly assigned
4. **Training Loss**: Triplet loss convergence over epochs
5. **Cluster Count**: Number of person vs unknown clusters
6. **Noise Rate**: % of faces in singleton unknown_noise_* clusters

### Logging

Add structured logging at key points:

```python
logger.info("Dual-mode clustering started", extra={
    'total_faces': len(all_faces),
    'labeled_people': len(known_people),
})

logger.info("Supervised assignment complete", extra={
    'assigned': assigned_count,
    'unassigned': unassigned_count,
    'threshold': person_threshold,
})

logger.info("Training complete", extra={
    'epochs': epochs,
    'final_loss': final_loss,
    'people_count': people_count,
})
```

### Dashboard (Future Enhancement)

Consider adding admin dashboard showing:
- Clustering quality metrics
- Training progress/history
- Person-by-person accuracy
- Cluster browser (person vs unknown)

---

## Security Considerations

### Data Privacy

- **Face embeddings**: Store securely (already in Qdrant + PostgreSQL)
- **Person names**: PII - handle according to privacy policy
- **Training data**: Labeled faces used for training contain PII

### Access Control

- **Labeling operations**: Require authentication
- **Training triggers**: Admin-only
- **Cluster viewing**: Consider privacy settings per person

### Audit Trail

- **`face_assignment_events`**: Already tracks assignment changes
- **Training logs**: Add logging for training operations
- **Cluster labeling**: Audit who labeled which clusters

---

## Future Enhancements

### Short Term

1. **Active Learning**: Suggest uncertain clusters for labeling
2. **Confidence Scores**: Return confidence with each assignment
3. **Multi-face Scenes**: Better handling of group photos
4. **Quality Filtering**: Automatically exclude low-quality faces

### Medium Term

1. **Online Learning**: Incremental model updates without full retraining
2. **Transfer Learning**: Fine-tune on domain-specific datasets
3. **Ensemble Methods**: Combine multiple clustering approaches
4. **Auto-labeling**: Suggest person names based on metadata

### Long Term

1. **Demographic Attributes**: Age, gender estimation (optional)
2. **Facial Attributes**: Glasses, beard, emotion detection
3. **Cross-Dataset Learning**: Train on public datasets
4. **Real-time Processing**: Live face recognition in video streams

---

## Appendix A: Algorithm Details

### Triplet Loss

Triplet loss optimizes embeddings to satisfy:

```
||anchor - positive||² + margin < ||anchor - negative||²
```

Where:
- **anchor**: Face of person A
- **positive**: Different face of same person A
- **negative**: Face of different person B
- **margin**: Minimum separation between positive and negative pairs

The loss function:

```python
loss = max(0, ||anchor - positive||² - ||anchor - negative||² + margin)
```

### HDBSCAN vs DBSCAN vs Agglomerative

| Algorithm | Pros | Cons | Best For |
|-----------|------|------|----------|
| **HDBSCAN** | Variable density, hierarchical, robust | Slower, memory-intensive | Default choice |
| **DBSCAN** | Fast, simple, well-understood | Fixed density, parameter-sensitive | Large datasets |
| **Agglomerative** | Deterministic, interpretable | Memory O(n²), slow | Small datasets |

### Cosine vs Euclidean Distance

For **normalized vectors** (as used in face embeddings):

```
cosine_distance(a, b) = 1 - (a · b) / (||a|| × ||b||)
euclidean_distance(a, b) = ||a - b||

For unit vectors: euclidean² = 2 × (1 - cosine)
```

Both metrics are equivalent for normalized vectors, but:
- **Cosine**: More intuitive (0 = identical, 1 = orthogonal)
- **Euclidean**: Faster computation for some algorithms

---

## Appendix B: Configuration Examples

### Development (Fast, Less Accurate)

```bash
# .env
FACE_PERSON_MATCH_THRESHOLD=0.65
FACE_UNKNOWN_MIN_CLUSTER_SIZE=2
FACE_UNKNOWN_CLUSTERING_METHOD=dbscan
FACE_TRAINING_EPOCHS=10
FACE_BATCH_SIZE=16
```

### Production (Slower, More Accurate)

```bash
# .env
FACE_PERSON_MATCH_THRESHOLD=0.75
FACE_UNKNOWN_MIN_CLUSTER_SIZE=5
FACE_UNKNOWN_CLUSTERING_METHOD=hdbscan
FACE_TRAINING_EPOCHS=30
FACE_BATCH_SIZE=32
```

### Conservative (High Precision)

```bash
# .env
FACE_PERSON_MATCH_THRESHOLD=0.85  # Fewer false positives
FACE_UNKNOWN_MIN_CLUSTER_SIZE=8   # Tighter clusters
FACE_UNKNOWN_CLUSTERING_METHOD=hdbscan
FACE_TRAINING_EPOCHS=40
```

### Aggressive (High Recall)

```bash
# .env
FACE_PERSON_MATCH_THRESHOLD=0.60  # More assignments
FACE_UNKNOWN_MIN_CLUSTER_SIZE=2   # Accept smaller clusters
FACE_UNKNOWN_CLUSTERING_METHOD=agglomerative
FACE_TRAINING_EPOCHS=20
```

---

## Appendix C: Troubleshooting

### Issue: "Not enough labeled data for training"

**Cause**: Fewer than 2 people with 2+ faces each

**Solution**:
```bash
# Check current labels
make faces-stats

# Label more clusters
curl -X POST .../clusters/unknown_cluster_N/label -d '{"name": "Person Name"}'

# Need: 10+ people with 5+ faces each for good results
```

### Issue: "Training loss not decreasing"

**Cause**: Learning rate too high/low, insufficient data, or bad initialization

**Solution**:
```bash
# Try different learning rates
make faces-train-person-matching EPOCHS=20 LEARNING_RATE=0.0001  # Lower
make faces-train-person-matching EPOCHS=20 LEARNING_RATE=0.001   # Higher

# Increase training epochs
make faces-train-person-matching EPOCHS=50

# Verify labeled data quality (faces actually belong to labeled person)
```

### Issue: "Dual clustering assigns too few faces to people"

**Cause**: Threshold too high

**Solution**:
```bash
# Lower the threshold
make faces-cluster-dual PERSON_THRESHOLD=0.65

# Check threshold vs accuracy tradeoff
# Lower threshold = more assignments (higher recall, lower precision)
# Higher threshold = fewer assignments (lower recall, higher precision)
```

### Issue: "Unknown clusters are too large/mixed"

**Cause**: Unknown clustering parameters too loose

**Solution**:
```bash
# Increase min cluster size
make faces-cluster-dual UNKNOWN_MIN_SIZE=5

# Try different clustering method
make faces-cluster-dual UNKNOWN_METHOD=agglomerative

# Use tighter epsilon for DBSCAN
# Edit .env: FACE_UNKNOWN_EPS=0.3
```

### Issue: "Out of memory during training"

**Cause**: Batch size too large, too many triplets

**Solution**:
```bash
# Reduce batch size
make faces-train-person-matching BATCH_SIZE=16

# Train on subset of people (modify code to sample)

# Use CPU instead of GPU (slower but more memory)
```

---

## Appendix D: References

### Research Papers

1. **FaceNet** (Schroff et al., 2015): Triplet loss for face recognition
2. **ArcFace** (Deng et al., 2019): Additive angular margin loss
3. **HDBSCAN** (McInnes et al., 2017): Density-based hierarchical clustering
4. **InsightFace** (Guo et al., 2018): Open-source face recognition toolkit

### Libraries

- **InsightFace**: https://github.com/deepinsight/insightface
- **PyTorch**: https://pytorch.org/
- **HDBSCAN**: https://hdbscan.readthedocs.io/
- **scikit-learn**: https://scikit-learn.org/
- **Qdrant**: https://qdrant.tech/

### Internal Documentation

- `docs/faces.md`: Complete face pipeline documentation
- `docs/api-contract.md`: API reference
- `PHASE*_IMPLEMENTATION.md`: Project history and architecture decisions

---

## Conclusion

This implementation plan provides a comprehensive roadmap for adding dual-mode face clustering with supervised learning to the image search service. The approach balances:

- **Accuracy**: High precision for known people, organized unknown faces
- **Usability**: Incremental workflow, progressive improvement
- **Compatibility**: No breaking changes, gradual migration path
- **Performance**: Efficient processing, scalable to large datasets

The system enables a powerful **label → train → improve** feedback loop that continuously enhances face recognition accuracy as users label more data.

**Next Steps**: Review this plan, prioritize phases, and begin Phase 1 implementation.

