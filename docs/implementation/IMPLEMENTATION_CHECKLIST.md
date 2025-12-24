# Dual-Mode Face Clustering - Implementation Checklist

**For Claude Code Agent**

This checklist provides step-by-step instructions for implementing the dual-mode face clustering system.

---

## Prerequisites

Before starting, ensure:
- [ ] Project builds and tests pass: `make test`
- [ ] Face detection working: `make faces-backfill LIMIT=10`
- [ ] Database has face data: Check `make faces-stats`
- [ ] Review full plan: [dual-mode-clustering-plan.md](dual-mode-clustering-plan.md)
- [ ] Review summary: [DUAL_MODE_CLUSTERING_SUMMARY.md](../../DUAL_MODE_CLUSTERING_SUMMARY.md)

---

## Phase 1: Configuration & Core Clustering (Week 1)

### Task 1.1: Add Configuration Fields

**File**: `src/image_search_service/core/config.py`

**Action**: Add new configuration fields after the existing face-related settings:

```python
# Face recognition model settings
face_model_name: str = Field(default="buffalo_l", alias="FACE_MODEL_NAME")
face_model_checkpoint: str = Field(default="", alias="FACE_MODEL_CHECKPOINT")
face_training_enabled: bool = Field(default=False, alias="FACE_TRAINING_ENABLED")

# Training hyperparameters
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

**Verification**:
```bash
uv run python -c "from image_search_service.core.config import get_settings; s = get_settings(); print(s.face_person_match_threshold)"
# Should output: 0.7
```

**Checklist**:
- [ ] Fields added to Settings class
- [ ] Default values set appropriately
- [ ] Field aliases use correct environment variable names
- [ ] Type hints correct
- [ ] Verification command passes

---

### Task 1.2: Create Dual-Mode Clusterer

**File**: `src/image_search_service/faces/dual_clusterer.py` (NEW)

**Action**: Create new file with the following structure:

```python
"""Dual-mode face clustering: supervised + unsupervised."""

import logging
import uuid
from typing import Optional

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session as SyncSession

logger = logging.getLogger(__name__)


class DualModeClusterer:
    """Clusters faces in dual mode: supervised (known people) + unsupervised (unknown faces)."""

    def __init__(
        self,
        db_session: SyncSession,
        person_match_threshold: float = 0.7,
        unknown_min_cluster_size: int = 3,
        unknown_method: str = "hdbscan",
        unknown_eps: float = 0.5,
    ):
        """Initialize dual-mode clusterer.

        Args:
            db_session: Synchronous SQLAlchemy session
            person_match_threshold: Minimum similarity for assignment to person (0-1)
            unknown_min_cluster_size: Minimum cluster size for unknown faces
            unknown_method: Clustering method (hdbscan, dbscan, agglomerative)
            unknown_eps: Distance threshold for DBSCAN/Agglomerative
        """
        self.db = db_session
        self.person_match_threshold = person_match_threshold
        self.unknown_min_cluster_size = unknown_min_cluster_size
        self.unknown_method = unknown_method
        self.unknown_eps = unknown_eps

    def cluster_all_faces(self, max_faces: Optional[int] = None) -> dict:
        """Run dual-mode clustering on all faces.

        Process:
        1. Get all faces (labeled and unlabeled)
        2. Separate by person_id
        3. Assign unlabeled faces to known people (supervised)
        4. Cluster remaining unknown faces (unsupervised)
        5. Save results

        Args:
            max_faces: Optional limit on number of faces to process

        Returns:
            Summary dict with counts
        """
        # TODO: Implement
        pass

    def assign_to_known_people(
        self,
        unlabeled_faces: list,
        labeled_faces: list,
    ) -> tuple[list, list]:
        """Assign unlabeled faces to known Person entities (supervised).

        Args:
            unlabeled_faces: Faces without person_id
            labeled_faces: Faces with person_id

        Returns:
            (assigned_faces, still_unknown)
        """
        # TODO: Implement
        pass

    def cluster_unknown_faces(self, unknown_faces: list) -> dict:
        """Cluster unknown faces using unsupervised learning.

        Args:
            unknown_faces: Faces still unassigned after supervised phase

        Returns:
            {face_id: cluster_label}
        """
        # TODO: Implement
        pass

    def save_dual_mode_results(
        self,
        assigned_faces: list,
        unknown_clusters: dict,
    ) -> None:
        """Save clustering results to database and Qdrant.

        Args:
            assigned_faces: List of dicts with face_id, person_id, similarity
            unknown_clusters: Dict mapping face_id to cluster_label
        """
        # TODO: Implement
        pass


def get_dual_mode_clusterer(
    db_session: SyncSession,
    person_match_threshold: float = 0.7,
    unknown_min_cluster_size: int = 3,
    unknown_method: str = "hdbscan",
) -> DualModeClusterer:
    """Factory function for DualModeClusterer."""
    return DualModeClusterer(
        db_session=db_session,
        person_match_threshold=person_match_threshold,
        unknown_min_cluster_size=unknown_min_cluster_size,
        unknown_method=unknown_method,
    )
```

**Implementation Details**:

See [dual-mode-clustering-plan.md](dual-mode-clustering-plan.md) Phase 2 for detailed algorithm implementations.

**Checklist**:
- [ ] File created with proper imports
- [ ] `DualModeClusterer` class defined
- [ ] `cluster_all_faces()` method stub
- [ ] `assign_to_known_people()` method stub
- [ ] `cluster_unknown_faces()` method stub
- [ ] `save_dual_mode_results()` method stub
- [ ] Factory function `get_dual_mode_clusterer()` defined
- [ ] Type hints correct
- [ ] Docstrings complete

---

### Task 1.3: Implement `assign_to_known_people()`

**File**: `src/image_search_service/faces/dual_clusterer.py`

**Action**: Implement supervised assignment logic:

```python
def assign_to_known_people(
    self,
    unlabeled_faces: list,
    labeled_faces: list,
) -> tuple[list, list]:
    """Assign unlabeled faces to known Person entities (supervised)."""

    if not labeled_faces:
        return [], unlabeled_faces

    from collections import defaultdict

    # Build person centroids from labeled faces
    person_embeddings = defaultdict(list)

    for face in labeled_faces:
        embedding = self._get_face_embedding(face['qdrant_point_id'])
        if embedding:
            person_embeddings[face['person_id']].append(embedding)

    # Calculate centroid for each person
    person_centroids = {}
    for person_id, embeddings in person_embeddings.items():
        centroid = np.mean(embeddings, axis=0)
        # Re-normalize
        person_centroids[person_id] = centroid / np.linalg.norm(centroid)

    assigned = []
    still_unknown = []

    # Match each unlabeled face to nearest person
    for face in unlabeled_faces:
        embedding = self._get_face_embedding(face['qdrant_point_id'])
        if not embedding:
            still_unknown.append(face)
            continue

        best_person = None
        best_similarity = -1

        for person_id, centroid in person_centroids.items():
            # Cosine similarity
            similarity = np.dot(embedding, centroid) / (
                np.linalg.norm(embedding) * np.linalg.norm(centroid)
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_person = person_id

        # Assign if above threshold
        if best_similarity >= self.person_match_threshold:
            assigned.append({
                'face_id': face['id'],
                'person_id': best_person,
                'similarity': best_similarity,
                'qdrant_point_id': face['qdrant_point_id'],
            })
        else:
            still_unknown.append(face)

    logger.info(
        f"Supervised assignment: {len(assigned)} assigned, "
        f"{len(still_unknown)} still unknown"
    )

    return assigned, still_unknown

def _get_face_embedding(self, qdrant_point_id: uuid.UUID) -> Optional[np.ndarray]:
    """Get face embedding from Qdrant by point ID."""
    from image_search_service.vector.face_qdrant import get_face_qdrant_client

    qdrant = get_face_qdrant_client()

    try:
        points = qdrant.client.retrieve(
            collection_name="faces",
            ids=[str(qdrant_point_id)],
            with_vectors=True,
        )

        if points and points[0].vector:
            vector = points[0].vector
            # Handle both dict and list formats
            if isinstance(vector, dict):
                return np.array(list(vector.values())[0])
            return np.array(vector)
        return None
    except Exception as e:
        logger.error(f"Error retrieving embedding: {e}")
        return None
```

**Checklist**:
- [ ] Method implemented with proper logic
- [ ] Centroid computation correct
- [ ] Cosine similarity calculation correct
- [ ] Threshold comparison correct
- [ ] Helper method `_get_face_embedding()` implemented
- [ ] Logging added
- [ ] Error handling in place

---

### Task 1.4: Implement `cluster_unknown_faces()`

**File**: `src/image_search_service/faces/dual_clusterer.py`

**Action**: Implement unsupervised clustering:

```python
def cluster_unknown_faces(self, unknown_faces: list) -> dict:
    """Cluster unknown faces using unsupervised learning."""

    if not unknown_faces:
        return {}

    # Get embeddings
    embeddings = []
    face_ids = []

    for face in unknown_faces:
        embedding = self._get_face_embedding(face['qdrant_point_id'])
        if embedding is not None:
            embeddings.append(embedding)
            face_ids.append(face['id'])

    if len(embeddings) < self.unknown_min_cluster_size:
        # Not enough for clustering
        return {fid: f"unknown_noise_{fid}" for fid in face_ids}

    embeddings_array = np.array(embeddings)

    # Choose clustering method
    if self.unknown_method == "hdbscan":
        cluster_labels = self._cluster_hdbscan(embeddings_array)
    elif self.unknown_method == "dbscan":
        cluster_labels = self._cluster_dbscan(embeddings_array)
    elif self.unknown_method == "agglomerative":
        cluster_labels = self._cluster_agglomerative(embeddings_array)
    else:
        raise ValueError(f"Unknown clustering method: {self.unknown_method}")

    # Map face_id to cluster_label
    clusters = {}
    for face_id, label in zip(face_ids, cluster_labels):
        if label == -1:  # Noise
            clusters[face_id] = f"unknown_noise_{face_id}"
        else:
            clusters[face_id] = f"unknown_cluster_{label}"

    logger.info(
        f"Unsupervised clustering: {len(set(clusters.values()))} clusters, "
        f"{sum(1 for v in clusters.values() if 'noise' in v)} noise"
    )

    return clusters

def _cluster_hdbscan(self, X: np.ndarray) -> np.ndarray:
    """Cluster using HDBSCAN."""
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=self.unknown_min_cluster_size,
        metric='cosine',
        cluster_selection_method='eom',
    )
    return clusterer.fit_predict(X)

def _cluster_dbscan(self, X: np.ndarray) -> np.ndarray:
    """Cluster using DBSCAN."""
    from sklearn.cluster import DBSCAN

    clusterer = DBSCAN(
        eps=self.unknown_eps,
        min_samples=self.unknown_min_cluster_size,
        metric='cosine',
    )
    return clusterer.fit_predict(X)

def _cluster_agglomerative(self, X: np.ndarray) -> np.ndarray:
    """Cluster using Agglomerative Clustering."""
    from sklearn.cluster import AgglomerativeClustering

    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=self.unknown_eps,
        linkage='average',
        metric='cosine',
    )
    return clusterer.fit_predict(X)
```

**Checklist**:
- [ ] Method implemented with all clustering algorithms
- [ ] HDBSCAN clustering implemented
- [ ] DBSCAN clustering implemented
- [ ] Agglomerative clustering implemented
- [ ] Cluster labeling correct (unknown_cluster_N, unknown_noise_*)
- [ ] Logging added
- [ ] Edge cases handled (not enough faces)

---

### Task 1.5: Implement `save_dual_mode_results()` and `cluster_all_faces()`

**File**: `src/image_search_service/faces/dual_clusterer.py`

**Action**: Implement database persistence and main orchestrator.

See detailed implementation in [dual-mode-clustering-plan.md](dual-mode-clustering-plan.md) Phase 2.

**Key Points**:
- Update `face_instances.person_id` for assigned faces
- Update `face_instances.cluster_id` for all faces
- Update Qdrant payloads for person_id and cluster_id
- Create cluster entries if needed
- Commit transaction

**Checklist**:
- [ ] `save_dual_mode_results()` implemented
- [ ] `cluster_all_faces()` orchestrator implemented
- [ ] Database updates correct
- [ ] Qdrant updates correct
- [ ] Transaction handling proper
- [ ] Error handling in place

---

### Task 1.6: Add CLI Command

**File**: `src/image_search_service/scripts/faces.py`

**Action**: Add new command after existing face commands:

```python
@faces_app.command("cluster-dual")
def cluster_faces_dual(
    person_threshold: float = typer.Option(0.7, help="Person match threshold (0-1)"),
    unknown_method: str = typer.Option("hdbscan", help="Unknown clustering method"),
    unknown_min_size: int = typer.Option(3, help="Min cluster size for unknown"),
    max_faces: Optional[int] = typer.Option(None, help="Max faces to process"),
    queue: bool = typer.Option(False, help="Run as background job"),
) -> None:
    """Run dual-mode clustering (supervised + unsupervised).

    Example:
        faces cluster-dual --person-threshold 0.75 --unknown-method hdbscan
    """
    if queue:
        from redis import Redis
        from rq import Queue
        from image_search_service.core.config import get_settings
        from image_search_service.queue.face_jobs import cluster_dual_job

        settings = get_settings()
        redis = Redis.from_url(settings.redis_url)
        q = Queue("default", connection=redis)

        job = q.enqueue(
            cluster_dual_job,
            person_threshold=person_threshold,
            unknown_method=unknown_method,
            unknown_min_size=unknown_min_size,
            max_faces=max_faces,
        )
        typer.echo(f"Queued dual clustering job: {job.id}")
    else:
        from image_search_service.db.sync_operations import get_sync_session
        from image_search_service.faces.dual_clusterer import get_dual_mode_clusterer

        typer.echo(f"Running dual-mode clustering...")

        db_session = get_sync_session()
        try:
            clusterer = get_dual_mode_clusterer(
                db_session=db_session,
                person_match_threshold=person_threshold,
                unknown_min_cluster_size=unknown_min_size,
                unknown_method=unknown_method,
            )

            result = clusterer.cluster_all_faces(max_faces=max_faces)

            typer.echo(f"Assigned to people: {result['assigned_to_people']}")
            typer.echo(f"Unknown clusters: {result['unknown_clusters']}")
            typer.echo(f"Total processed: {result['total_processed']}")
        finally:
            db_session.close()
```

**Checklist**:
- [ ] Command added with proper decorator
- [ ] Options defined correctly
- [ ] Queue support added
- [ ] Direct execution implemented
- [ ] Error handling in place
- [ ] Output messages clear

---

### Task 1.7: Add Makefile Target

**File**: `Makefile`

**Action**: Add new target after existing face targets:

```makefile
faces-cluster-dual: ## Run dual-mode clustering (supervised + unsupervised)
	@echo "Running dual-mode clustering..."
	uv run python -m image_search_service.scripts.cli faces cluster-dual \
		--person-threshold $(or $(PERSON_THRESHOLD),0.7) \
		--unknown-method $(or $(UNKNOWN_METHOD),hdbscan) \
		--unknown-min-size $(or $(UNKNOWN_MIN_SIZE),3)
```

**Checklist**:
- [ ] Target added to Makefile
- [ ] Help comment included (##)
- [ ] Parameters configurable via environment
- [ ] Default values match config

---

### Task 1.8: Testing Phase 1

**Action**: Create test file and verify implementation.

**File**: `tests/faces/test_dual_clusterer.py` (NEW)

```python
"""Tests for dual-mode face clustering."""

import pytest
from unittest.mock import Mock, patch
import numpy as np
import uuid

# TODO: Implement tests for:
# - assign_to_known_people()
# - cluster_unknown_faces()
# - save_dual_mode_results()
# - cluster_all_faces()
```

**Manual Testing**:

```bash
# 1. Verify configuration
uv run python -c "from image_search_service.core.config import get_settings; print(get_settings().face_person_match_threshold)"

# 2. Run dual clustering on test data
make faces-cluster-dual PERSON_THRESHOLD=0.7

# 3. Check results
make faces-stats

# 4. Verify cluster labels in database
# Should see unknown_cluster_* for unlabeled faces
```

**Checklist**:
- [ ] Unit tests created
- [ ] Manual test passes
- [ ] Cluster labels correct (person_*, unknown_*)
- [ ] Database updates correct
- [ ] Qdrant updates correct
- [ ] No errors in logs

---

## Phase 2: Training System (Week 2)

### Task 2.1: Create Training Module

**File**: `src/image_search_service/faces/trainer.py` (NEW)

See detailed implementation in [dual-mode-clustering-plan.md](dual-mode-clustering-plan.md) Phase 3.

**Checklist**:
- [ ] File created with `FaceTrainer` class
- [ ] `TripletFaceDataset` implemented
- [ ] `get_labeled_faces_by_person()` query implemented
- [ ] `fine_tune_for_person_clustering()` training loop implemented
- [ ] Checkpoint saving implemented
- [ ] Factory function added

---

### Task 2.2: Add Training CLI Command

**File**: `src/image_search_service/scripts/faces.py`

**Checklist**:
- [ ] `train-matching` command added
- [ ] Options for epochs, margin, batch_size, learning_rate
- [ ] Queue support added
- [ ] Direct execution implemented

---

### Task 2.3: Add Training Makefile Target

**File**: `Makefile`

**Checklist**:
- [ ] `faces-train-person-matching` target added
- [ ] Parameters configurable
- [ ] Default values set

---

## Phase 3: Integration (Week 3)

### Task 3.1: Add Background Jobs

**File**: `src/image_search_service/queue/face_jobs.py`

**Checklist**:
- [ ] `cluster_dual_job()` function added
- [ ] `train_person_matching_job()` function added
- [ ] `reassign_faces_job()` function added

---

### Task 3.2: Add API Endpoints (Optional)

**File**: `src/image_search_service/api/routes/faces.py`

**Checklist**:
- [ ] POST `/faces/cluster/dual` endpoint
- [ ] POST `/faces/train` endpoint
- [ ] Request/response schemas defined

---

### Task 3.3: Integration Testing

**File**: `tests/faces/test_integration_dual_mode.py` (NEW)

**Checklist**:
- [ ] End-to-end test created
- [ ] Tests full workflow: detect → cluster → label → train → recluster
- [ ] Verifies improvements in accuracy

---

## Phase 4: Documentation (Week 4)

### Task 4.1: Update Documentation

**Files to Update**:
- [ ] `docs/faces.md` - Add dual-mode section
- [ ] `README.md` - Add new commands to reference
- [ ] `DUAL_MODE_CLUSTERING_SUMMARY.md` - Keep updated

**Checklist**:
- [ ] Dual-mode workflow documented
- [ ] Training workflow documented
- [ ] Configuration options documented
- [ ] Examples provided

---

## Final Verification

### Functional Tests

- [ ] Basic clustering works: `make faces-cluster-dual`
- [ ] Training works: `make faces-train-person-matching EPOCHS=10`
- [ ] Reassignment works: `make faces-reassign-smart`
- [ ] Statistics show dual-mode clusters: `make faces-stats`

### Integration Tests

- [ ] Full pipeline: backfill → cluster → label → train → recluster
- [ ] Accuracy improves after training
- [ ] Unknown clusters stay organized
- [ ] No data loss

### Performance Tests

- [ ] 1K faces: clustering < 5 seconds
- [ ] 10K faces: clustering < 30 seconds
- [ ] Training 10 people: < 10 minutes on GPU

### Backward Compatibility

- [ ] Old commands still work: `make faces-cluster`, `make faces-assign`
- [ ] Existing clusters unaffected
- [ ] No database migration needed

---

## Success Criteria

Implementation is complete when:

1. ✅ **Dual-mode clustering works**: Assigns to people + clusters unknown
2. ✅ **Training improves accuracy**: Measurable improvement after 2-3 iterations
3. ✅ **CLI commands functional**: All new commands work as expected
4. ✅ **Tests pass**: Unit and integration tests pass
5. ✅ **Documentation complete**: Users can follow workflow guide
6. ✅ **Backward compatible**: No breaking changes to existing system
7. ✅ **Performance acceptable**: Meets performance targets

---

## Support Resources

- **Full Implementation Plan**: [dual-mode-clustering-plan.md](dual-mode-clustering-plan.md)
- **Quick Summary**: [DUAL_MODE_CLUSTERING_SUMMARY.md](../../DUAL_MODE_CLUSTERING_SUMMARY.md)
- **Face Pipeline Docs**: [../faces.md](../faces.md)
- **Current Code**: See `src/image_search_service/faces/` for existing implementations

---

**Good luck with implementation! Follow this checklist step by step for a successful deployment.**

