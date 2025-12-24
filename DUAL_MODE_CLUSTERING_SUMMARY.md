# Dual-Mode Face Clustering - Implementation Summary

**Status**: Planning Phase
**Date**: 2024-12-24
**Full Plan**: [docs/implementation/dual-mode-clustering-plan.md](docs/implementation/dual-mode-clustering-plan.md)

---

## Quick Overview

This enhancement adds **dual-mode clustering** to the face recognition system:

1. **Supervised Mode**: Assigns faces to known Person entities (high precision)
2. **Unsupervised Mode**: Clusters unknown faces by similarity (for discovery & labeling)

**Goal**: Maximize accuracy for known people while keeping unknown faces organized for progressive labeling.

---

## Current vs Enhanced System

### Current System

```
Face Detection → Qdrant Storage → HDBSCAN Clustering → Manual Labeling → Prototype Assignment
```

**Limitations**:
- ❌ No identity awareness in clustering
- ❌ No learning from user labels
- ❌ Split clusters (same person in multiple groups)
- ❌ Mixed clusters (multiple people in one group)

### Enhanced System

```
Face Detection → Qdrant Storage → DUAL-MODE CLUSTERING
                                         ↓
                        ┌────────────────┴────────────────┐
                        ↓                                 ↓
                SUPERVISED MODE                   UNSUPERVISED MODE
            (Match to Known People)            (Cluster Unknown Faces)
                        ↓                                 ↓
                person_* clusters                 unknown_* clusters
                        └────────────────┬────────────────┘
                                         ↓
                              TRAINING SYSTEM
                        (Learn from Labels → Improve)
```

**Benefits**:
- ✅ Identity-aware assignments
- ✅ Progressive learning from labels
- ✅ High precision for known people
- ✅ Organized unknown faces
- ✅ No data loss

---

## Implementation Phases

### Phase 1: Core Dual-Mode Clustering (Week 1)

**Files to Create/Modify**:
- `src/image_search_service/core/config.py` - Add configuration fields
- `src/image_search_service/faces/dual_clusterer.py` - New dual-mode clustering service
- `src/image_search_service/scripts/faces.py` - Add `cluster-dual` command
- `Makefile` - Add `faces-cluster-dual` target

**Key Components**:
- `DualModeClusterer` class with:
  - `assign_to_known_people()` - Supervised matching
  - `cluster_unknown_faces()` - Unsupervised grouping
  - `save_dual_mode_results()` - Database persistence

### Phase 2: Training System (Week 2)

**Files to Create/Modify**:
- `src/image_search_service/faces/trainer.py` - New training service
- `src/image_search_service/scripts/faces.py` - Add `train-matching` command
- `Makefile` - Add `faces-train-person-matching` target

**Key Components**:
- `FaceTrainer` class with:
  - `TripletFaceDataset` - PyTorch dataset for triplet loss
  - `fine_tune_for_person_clustering()` - Training loop
  - `regenerate_embeddings()` - Apply trained model

### Phase 3: Integration (Week 3)

**Files to Create/Modify**:
- `src/image_search_service/queue/face_jobs.py` - Background jobs
- `src/image_search_service/api/routes/faces.py` - API endpoints (optional)
- `src/image_search_service/scripts/faces.py` - Add `reassign` command
- `tests/faces/` - Integration tests

### Phase 4: Documentation (Week 4)

**Files to Update**:
- `docs/faces.md` - Add dual-mode guide
- `README.md` - Update with new workflow
- Create tutorial/examples

---

## New CLI Commands

### 1. Dual-Mode Clustering

```bash
# Run dual-mode clustering
uv run python -m image_search_service.cli faces cluster-dual \
  --person-threshold 0.7 \
  --unknown-method hdbscan

# Or via Makefile
make faces-cluster-dual PERSON_THRESHOLD=0.7
```

### 2. Train Person Matching

```bash
# Train model on labeled data
uv run python -m image_search_service.cli faces train-matching \
  --epochs 20 \
  --margin 0.2

# Or via Makefile
make faces-train-person-matching EPOCHS=20
```

### 3. Reassign with Trained Model

```bash
# Reassign all faces
uv run python -m image_search_service.cli faces reassign \
  --person-threshold 0.75 \
  --use-trained-model

# Or via Makefile
make faces-reassign-smart PERSON_THRESHOLD=0.75
```

---

## New Configuration Fields

Add to `.env`:

```bash
# Face recognition model
FACE_MODEL_NAME=buffalo_l
FACE_MODEL_CHECKPOINT=
FACE_TRAINING_ENABLED=false

# Training hyperparameters
FACE_TRIPLET_MARGIN=0.2
FACE_TRAINING_EPOCHS=20
FACE_BATCH_SIZE=32
FACE_LEARNING_RATE=0.0001

# Supervised clustering (known people)
FACE_PERSON_MATCH_THRESHOLD=0.7

# Unsupervised clustering (unknown faces)
FACE_UNKNOWN_CLUSTERING_METHOD=hdbscan
FACE_UNKNOWN_MIN_CLUSTER_SIZE=3
FACE_UNKNOWN_EPS=0.5
```

---

## Enhanced Workflow

### Initial Setup

```bash
# 1. Detect faces
make faces-backfill LIMIT=10000

# 2. Run dual-mode clustering
make faces-cluster-dual PERSON_THRESHOLD=0.7

# 3. Check statistics
make faces-stats
```

### Iterative Improvement

```bash
# 1. Label 10-20 unknown clusters via API
curl -X POST http://localhost:8000/api/v1/faces/clusters/unknown_cluster_5/label \
  -H "Content-Type: application/json" \
  -d '{"name": "Alice Johnson"}'

# 2. Train model on labeled data
make faces-train-person-matching EPOCHS=20

# 3. Re-cluster with improved model
make faces-cluster-dual PERSON_THRESHOLD=0.75

# 4. Check improvements
make faces-stats

# 5. Repeat: label more → train → recluster
```

### Production Workflow

```bash
# Daily: process new images
make faces-backfill LIMIT=1000
make faces-cluster-dual PERSON_THRESHOLD=0.75

# Monthly: retrain with new labels
make faces-train-person-matching EPOCHS=20
make faces-reassign-smart PERSON_THRESHOLD=0.75
```

---

## Expected Results

### Accuracy Progression

| Stage | Person Assignment | Unknown Cluster Purity | Recall |
|-------|-------------------|------------------------|--------|
| **Baseline** (No training) | 70-80% | 60-70% | 50-60% |
| **After 1st Training** (10+ people) | 80-85% | 75-80% | 65-75% |
| **After 3rd Training** (20+ people) | 85-90% | 80-85% | 75-85% |
| **Production** (50+ people) | 90-95% | 85-90% | 85-90% |

### Qualitative Improvements

- ✅ Known people: High precision clusters
- ✅ Unknown people: Organized by similarity
- ✅ Better handling of edge cases (profiles, occlusions)
- ✅ Dataset-specific adaptation

---

## Data Requirements

### Minimum (Proof of Concept)
- **5 people** × **5 faces each** = 25 labeled faces
- Training possible but modest results

### Recommended (Good Results)
- **10-15 people** × **10 faces each** = 100-150 labeled faces
- Significant improvement

### Production (Excellent Results)
- **20+ people** × **20+ faces each** = 400+ labeled faces
- Near-human-level accuracy

---

## Cluster Naming Convention

The system uses clear naming to distinguish cluster types:

| Prefix | Type | Example | Description |
|--------|------|---------|-------------|
| `person_{uuid}` | Supervised | `person_a3f9b2c1...` | Assigned to labeled Person entity |
| `unknown_cluster_{N}` | Unsupervised | `unknown_cluster_5` | Group of similar unknown faces |
| `unknown_noise_{face_id}` | Singleton | `unknown_noise_abc123...` | Face that doesn't match anything |

---

## Technical Architecture

### Key Classes

#### `DualModeClusterer` (New)
```python
class DualModeClusterer:
    def cluster_all_faces() -> dict:
        """Main orchestrator for dual-mode clustering."""

    def assign_to_known_people() -> tuple:
        """Supervised: match faces to person centroids."""

    def cluster_unknown_faces() -> dict:
        """Unsupervised: cluster remaining faces."""

    def save_dual_mode_results() -> None:
        """Persist assignments to database."""
```

#### `FaceTrainer` (New)
```python
class FaceTrainer:
    def train_for_person_clustering() -> dict:
        """Train with triplet loss to improve boundaries."""

    class TripletFaceDataset(Dataset):
        """Generate (anchor, positive, negative) triplets."""

    def regenerate_embeddings() -> dict:
        """Apply trained projection to all faces."""
```

### Training Architecture

```
Pre-trained ArcFace (512-dim) [FROZEN]
           ↓
Projection Head (trainable)
    Linear(512 → 256)
    ReLU
    Linear(256 → 128)
    L2 Normalize
           ↓
Triplet Loss: max(0, margin + d(anchor,pos) - d(anchor,neg))
```

---

## Database Schema

**No schema changes required!** Uses existing tables:

### Existing Schema (No Changes)

- `persons` - Person entities (name, status)
- `face_instances` - Detected faces (bbox, person_id, cluster_id)
- `person_prototypes` - Prototype embeddings (centroid/exemplar)
- `face_assignment_events` - Audit log

### New Usage Patterns

- `face_instances.cluster_id` now stores: `person_*`, `unknown_*`, or `unknown_noise_*`
- `face_instances.person_id` assigned by supervised mode
- No migrations needed - fully backward compatible

---

## Performance Characteristics

### Dual-Mode Clustering
- **Speed**: ~1-2 seconds per 1000 faces
- **Memory**: ~500MB peak (for 50K faces)
- **Scalability**: Linear with face count

### Training
- **GPU**: 5-10 minutes per 20 epochs (RTX 3080)
- **CPU**: 30-60 minutes per 20 epochs (16-core)
- **Memory**: ~2GB for model + data
- **Checkpoint**: ~50MB per saved model

---

## Backward Compatibility

✅ **Fully backward compatible** - no breaking changes:

- Existing clusters remain valid
- Old commands (`faces-cluster`, `faces-assign`) still work
- New commands are additive
- No database migrations required
- Users can transition gradually

---

## Migration Path

### Option 1: Keep Using Old System
```bash
# Nothing changes
make faces-cluster
make faces-assign
```

### Option 2: Transition to Dual-Mode
```bash
# Start using new commands
make faces-cluster-dual
make faces-train-person-matching
```

### Option 3: Hybrid Approach
```bash
# Use dual-mode for new processing
make faces-backfill LIMIT=1000
make faces-cluster-dual

# Keep old assignments intact
# They automatically become supervised assignments
```

---

## Testing Strategy

### Unit Tests
- `test_dual_clusterer.py` - Supervised/unsupervised logic
- `test_trainer.py` - Training loop and triplet generation
- `test_integration_dual_mode.py` - End-to-end workflow

### Manual Testing Checklist
- [ ] Initial clustering creates unknown_* clusters
- [ ] Labeling creates Person entity
- [ ] Dual clustering assigns to labeled person
- [ ] Unknown faces stay organized
- [ ] Training improves accuracy
- [ ] Statistics reflect dual-mode state

---

## Monitoring Metrics

Track these to measure effectiveness:

1. **Assignment Accuracy**: % correctly assigned to people
2. **Unknown Cluster Purity**: % of faces that belong together
3. **Person Recall**: % of person's faces correctly assigned
4. **Training Loss**: Convergence over epochs
5. **Cluster Distribution**: person vs unknown vs noise counts

---

## Troubleshooting Quick Reference

### "Not enough labeled data"
- **Need**: 10+ people with 5+ faces each
- **Check**: `make faces-stats`
- **Fix**: Label more unknown clusters via API

### "Training loss not decreasing"
- **Try**: Different learning rates (0.0001 vs 0.001)
- **Try**: More epochs (30-50)
- **Check**: Data quality (faces match labels)

### "Too few faces assigned to people"
- **Cause**: Threshold too high
- **Fix**: Lower threshold (0.65 instead of 0.7)
- **Trade-off**: Lower precision, higher recall

### "Unknown clusters too large/mixed"
- **Cause**: Clustering parameters too loose
- **Fix**: Increase min_cluster_size (5 instead of 3)
- **Try**: Different method (agglomerative vs hdbscan)

---

## Next Steps

### For Implementation
1. Review [full plan](docs/implementation/dual-mode-clustering-plan.md)
2. Start with Phase 1: Core dual-mode clustering
3. Test with sample dataset (10+ people, 100+ faces)
4. Iterate based on results

### For Users (After Implementation)
1. Run initial dual-mode clustering
2. Label 10-20 unknown clusters
3. Train model (first iteration)
4. Observe improvements
5. Repeat labeling and training

---

## References

- **Full Implementation Plan**: [docs/implementation/dual-mode-clustering-plan.md](docs/implementation/dual-mode-clustering-plan.md)
- **Face Pipeline Docs**: [docs/faces.md](docs/faces.md)
- **API Reference**: [docs/api-contract.md](docs/api-contract.md)

---

## Key Advantages

1. **Progressive Improvement**: Each labeling session improves the system
2. **High Precision**: Known people get accurate clusters
3. **Organization**: Unknown faces stay grouped for easy review
4. **No Data Loss**: Nothing gets misclassified or lost
5. **Backward Compatible**: Works alongside existing system
6. **Production Ready**: Scales to large datasets with monitoring

---

**For detailed implementation instructions, see the [full plan document](docs/implementation/dual-mode-clustering-plan.md).**

