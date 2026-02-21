# Phase 1.5: PCA in FaceClusterer + Suggestion Job Batching

**Priority**: HIGH -- Addresses two bottlenecks ignored by Phases 1-4 (covers 90% of training pipeline time)
**Estimated Effort**: 4-6 hours (two work streams)
**Expected Speedup**: ~80% total pipeline reduction (111.6s -> 15-25s)
**Risk Level**: Low-Medium (PCA validated by Phase 1; batching introduces error isolation tradeoff)

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [Scope](#2-scope)
3. [Work Stream A: PCA in FaceClusterer](#3-work-stream-a-pca-in-faceclusterer-p0--highest-impact)
4. [Work Stream B: Suggestion Job Batching](#4-work-stream-b-suggestion-job-batching-p1)
5. [Work Stream C: Supporting Fixes](#5-work-stream-c-supporting-fixes-p2-p3)
6. [Files Changed](#6-files-changed)
7. [Testing Plan](#7-testing-plan)
8. [Estimated Impact](#8-estimated-impact)
9. [Risk Assessment](#9-risk-assessment)
10. [Dependencies](#10-dependencies)
11. [Rollout Plan](#11-rollout-plan)

---

## 1. Motivation

Phase 1 implemented PCA preprocessing in `discover_unknown_persons_job` (`face_jobs.py` lines 2218-2275), but the **primary training workflow** calls `FaceClusterer._run_hdbscan()` in `clusterer.py` (lines 188-218) -- a completely independent code path with **zero dimensionality reduction**. Worker logs from training 6 images show HDBSCAN consuming 72.5 seconds (65% of the 111.6s pipeline) operating at full 512 dimensions. Meanwhile, 90 post-training suggestion jobs consume 28.1 seconds (25%) due to serial RQ execution overhead.

Combined, these two issues account for **90% of pipeline time** and neither is addressed by Phase 2+ plans.

### Current Training Pipeline (measured, 6-image training session)

| Component | Time | % of Total |
|-----------|------|-----------|
| HDBSCAN clustering (512d, Prim's, single-threaded) | 72.5s | 65% |
| Post-training suggestion jobs (90 individual RQ jobs) | 28.1s | 25% |
| Everything else (detect, embed, save, assign) | 11.0s | 10% |
| **Total** | **111.6s** | **100%** |

The training clustering path (`detect_faces_for_session_job` at line 685-720) calls `get_face_clusterer()` which invokes `FaceClusterer.cluster_unlabeled_faces()` -> `_run_hdbscan()`. This path has never received the Phase 1 PCA treatment.

### Root Cause of 90-Job Overhead

Each individual suggestion job consumes ~245ms before doing any useful work:
- RQ dequeue latency: ~80ms
- DB session initialization: ~60ms
- Qdrant client setup (×2 clients): ~90ms
- Per-job Python import overhead: ~15ms

92% of individual jobs produce zero suggestions (persons with no matching unassigned faces). Eliminating the overhead via batching saves ~25 seconds with no change to business logic.

---

## 2. Scope

Two work streams, ordered by impact:

| Work Stream | Problem | Solution | Est. Saving |
|------------|---------|----------|------------|
| **A (P0)** | FaceClusterer runs HDBSCAN at 512d (Prim's, O(N²×D)) | Port PCA from discover path into FaceClusterer via shared utility | 67-71s |
| **B (P1)** | 90 individual suggestion jobs, each with ~245ms RQ overhead | Replace per-person enqueue with batch variants | 24-26s |
| **C (P2-P3)** | Memory formula overstates requirements; prototype pre-check missing | Corrected formula, pre-filter no-ops | 1-3s |

---

## 3. Work Stream A: PCA in FaceClusterer (P0 -- Highest Impact)

### Problem

`FaceClusterer._run_hdbscan()` (clusterer.py lines 188-218) operates on raw 512d embeddings:

```python
def _run_hdbscan(self, embeddings_array: np.ndarray) -> np.ndarray:
    # ...
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=self.min_cluster_size,
        min_samples=self.min_samples,
        cluster_selection_epsilon=self.cluster_selection_epsilon,
        metric=self.metric,
        core_dist_n_jobs=-1,  # Use all CPU cores
    )

    cluster_labels = clusterer.fit_predict(embeddings_array)
    return cluster_labels
```

At 512d, HDBSCAN selects Prim's MST algorithm (`X.shape[1] > 60` threshold in hdbscan 0.8.x), which:
- Is single-threaded (ignores `core_dist_n_jobs=-1` -- the parameter is silently discarded)
- Has O(N²×D) complexity
- For 10K faces: 72.5s. Projected for 50K: ~30 minutes.

There are **6 callers** of `get_face_clusterer()` across the codebase:
1. `detect_faces_for_session_job()` -- training path (line 689, this log)
2. `cluster_faces_job()` -- standalone RQ job
3. `scripts/faces.py` CLI -- cluster command
4. `face_clustering_restart_service.py` -- restart service
5. `api/routes/faces.py` -- API endpoint (line 407)
6. `api/routes/faces.py` -- API endpoint (line 1534)

All 6 callers will benefit from PCA since they all perform the same task (cluster unlabeled faces).

### A1. Create Shared PCA Utility Module

**File**: `src/image_search_service/faces/embedding_preprocessing.py` (NEW)

This extracts the PCA logic currently inline in `discover_unknown_persons_job` (lines 2218-2303) into a reusable module, avoiding code duplication across two HDBSCAN call sites.

```python
"""Shared embedding preprocessing for face clustering pipelines."""
from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def reduce_dimensions_pca(
    embeddings: NDArray[np.float32],
    target_dim: int = 50,
    random_state: int = 42,
    variance_warning_threshold: float = 0.90,
    job_id: str = "no-job",
) -> tuple[NDArray[np.float32], dict[str, float]]:
    """Reduce embedding dimensionality via PCA for faster HDBSCAN clustering.

    At 512 dimensions, HDBSCAN selects Prim's MST algorithm (dimensions > 60
    threshold in hdbscan 0.8.x), which is single-threaded O(N^2 * D).
    PCA to <=50 dimensions triggers Boruvka algorithm selection, which IS
    parallel and uses KD-tree pruning: O(N * log^2(N)).

    Args:
        embeddings: Array of shape (n_samples, n_dims)
        target_dim: Target number of dimensions (0 to disable PCA)
        random_state: Random seed for deterministic PCA
        variance_warning_threshold: Log warning if retained variance falls below this
        job_id: Job identifier for structured log context

    Returns:
        Tuple of (reduced_embeddings, stats_dict) where stats_dict contains:
        - original_dims: int
        - reduced_dims: int
        - explained_variance: float
        - applied: bool
    """
    from sklearn.decomposition import PCA  # lazy import -- sklearn is a transitive dep

    n_samples, n_dims = embeddings.shape
    stats: dict[str, float] = {
        "original_dims": float(n_dims),
        "reduced_dims": float(n_dims),
        "explained_variance": 1.0,
        "applied": 0.0,
    }

    if target_dim <= 0 or n_dims <= target_dim or n_samples <= target_dim:
        logger.info(
            f"[{job_id}] PCA skipped: {n_samples} samples × {n_dims}d "
            f"(target={target_dim}d)"
        )
        return embeddings, stats

    n_components = min(target_dim, n_samples - 1)
    pca = PCA(n_components=n_components, random_state=random_state)
    reduced = pca.fit_transform(embeddings)

    explained_variance = float(pca.explained_variance_ratio_.sum())
    stats.update(
        {
            "reduced_dims": float(n_components),
            "explained_variance": explained_variance,
            "applied": 1.0,
        }
    )

    logger.info(
        f"[{job_id}] PCA: {n_dims}d -> {n_components}d "
        f"({explained_variance:.1%} variance retained)"
    )

    if explained_variance < variance_warning_threshold:
        logger.warning(
            f"[{job_id}] PCA retained only {explained_variance:.1%} variance "
            f"(threshold: {variance_warning_threshold:.0%}). "
            "Clustering quality may be degraded."
        )

    return reduced.astype(np.float32), stats


def select_hdbscan_params(
    n_dims: int,
    pca_applied: bool,
) -> dict[str, object]:
    """Select optimal HDBSCAN algorithm and parallelism based on dimensionality.

    At low dimensions (<=50, i.e., below the hdbscan 0.8.x threshold of >60),
    Boruvka KD-tree gives O(N log²N) complexity with `core_dist_n_jobs` parallelism.
    At high dimensions (>60), HDBSCAN auto-selects Prim's MST: O(N²D),
    single-threaded (core_dist_n_jobs is silently ignored).

    Note: when algorithm="boruvka_kdtree" is explicitly set, hdbscan validates
    metric against BALLTREE_VALID_METRICS (a library quirk), but since
    BALLTREE_VALID_METRICS is a superset of KDTREE_VALID_METRICS, "euclidean"
    works correctly with the explicit override.

    Args:
        n_dims: Number of dimensions after any preprocessing
        pca_applied: Whether PCA was applied to reduce dimensions

    Returns:
        Dict of keyword arguments to pass to hdbscan.HDBSCAN constructor
    """
    if pca_applied and n_dims <= 60:
        return {
            "algorithm": "boruvka_kdtree",
            "core_dist_n_jobs": -1,  # Boruvka uses this; Prim's ignores it
        }
    else:
        return {
            "algorithm": "best",
            "core_dist_n_jobs": 1,  # Prim's ignores this anyway; set to 1 explicitly
        }
```

### A2. Integrate PCA into FaceClusterer._run_hdbscan()

**File**: `src/image_search_service/faces/clusterer.py`

**Current code** (lines 188-218):

```python
def _run_hdbscan(self, embeddings_array: np.ndarray) -> np.ndarray:
    """Run HDBSCAN clustering on embedding matrix."""
    try:
        import hdbscan
    except ImportError:
        logger.error("hdbscan not installed. Run: pip install hdbscan")
        raise

    # For normalized face embeddings, euclidean distance works well
    # (it's proportional to cosine distance for unit vectors)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=self.min_cluster_size,
        min_samples=self.min_samples,
        cluster_selection_epsilon=self.cluster_selection_epsilon,
        metric=self.metric,
        core_dist_n_jobs=-1,  # Use all CPU cores
    )

    cluster_labels = clusterer.fit_predict(embeddings_array)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    logger.debug(f"HDBSCAN found {n_clusters} clusters")

    return cluster_labels
```

**Modified approach**:

```python
def _run_hdbscan(
    self,
    embeddings_array: np.ndarray,
    pca_target_dim: int = 50,
) -> np.ndarray:
    """Run HDBSCAN clustering on embedding matrix.

    Args:
        embeddings_array: Numpy array of shape (n_samples, embedding_dim)
        pca_target_dim: Target dimensions for PCA preprocessing (0 to disable).
            Defaults to 50, which is below hdbscan's 60-dimension threshold for
            Boruvka algorithm selection.

    Returns:
        Cluster labels array of shape (n_samples,)
    """
    try:
        import hdbscan
    except ImportError:
        logger.error("hdbscan not installed. Run: pip install hdbscan")
        raise

    from image_search_service.faces.embedding_preprocessing import (
        reduce_dimensions_pca,
        select_hdbscan_params,
    )

    # Phase 1.5: Dimensionality reduction for performance.
    # FaceClusterer previously ran HDBSCAN at 512d (Prim's, single-threaded, O(N²D)).
    # PCA to <=50d triggers Boruvka (parallel, O(N log²N)).
    reduced_embeddings, pca_stats = reduce_dimensions_pca(
        embeddings_array,
        target_dim=pca_target_dim,
    )

    hdbscan_params = select_hdbscan_params(
        n_dims=int(pca_stats["reduced_dims"]),
        pca_applied=bool(pca_stats["applied"]),
    )

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=self.min_cluster_size,
        min_samples=self.min_samples,
        cluster_selection_epsilon=self.cluster_selection_epsilon,
        metric=self.metric,
        cluster_selection_method="eom",  # Excess of mass (more stable)
        **hdbscan_params,
    )

    cluster_labels = clusterer.fit_predict(reduced_embeddings)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    logger.debug(
        f"HDBSCAN found {n_clusters} clusters "
        f"(dims={int(pca_stats['reduced_dims'])}, pca={bool(pca_stats['applied'])})"
    )

    return cluster_labels
```

### A3. Add pca_target_dim to FaceClusterer Constructor

Add the parameter alongside existing constructor params to allow callers to override the default:

```python
def __init__(
    self,
    db_session: SyncSession,
    qdrant_client,
    min_cluster_size: int = 5,
    min_samples: int = 3,
    cluster_selection_epsilon: float = 0.0,
    metric: str = "euclidean",
    pca_target_dim: int = 50,  # NEW: 0 to disable PCA
):
    # ...
    self.pca_target_dim = pca_target_dim
```

Then pass `self.pca_target_dim` to `_run_hdbscan()` in `cluster_unlabeled_faces()`:

```python
cluster_labels = self._run_hdbscan(embeddings_array, pca_target_dim=self.pca_target_dim)
```

### A4. Refactor discover_unknown_persons_job to Use Shared Utility

**File**: `src/image_search_service/queue/face_jobs.py`

Replace the inline PCA code (lines 2218-2275) with a call to `reduce_dimensions_pca()`, and replace the inline algorithm-selection logic (lines 2280-2303) with a call to `select_hdbscan_params()`.

This eliminates code duplication between the discover path and the clusterer path. Both paths will use identical preprocessing logic maintained in one place.

**Before** (inline at lines 2218-2303, ~85 lines):
```python
        # PHASE 3.5: Dimensionality reduction for efficient clustering
        import time as _time
        n_samples, n_dims = embeddings.shape
        if pca_target_dim > 0 and n_dims > pca_target_dim and n_samples > pca_target_dim:
            # ... inline PCA code ...
        else:
            embeddings_for_clustering = embeddings
            # ... inline skip logging ...

        # Select HDBSCAN algorithm based on whether PCA was applied
        pca_was_applied = embeddings_for_clustering.shape[1] < embeddings.shape[1]
        hdbscan_algorithm = "boruvka_kdtree" if pca_was_applied else "best"
        hdbscan_n_jobs = -1 if pca_was_applied else 1
        # ... inline logging ...
```

**After** (using shared utility, ~15 lines):
```python
        # PHASE 3.5: Dimensionality reduction for efficient clustering
        from image_search_service.faces.embedding_preprocessing import (
            reduce_dimensions_pca,
            select_hdbscan_params,
        )

        embeddings_for_clustering, pca_stats = reduce_dimensions_pca(
            embeddings,
            target_dim=pca_target_dim,
            job_id=job_id,
        )

        hdbscan_params = select_hdbscan_params(
            n_dims=int(pca_stats["reduced_dims"]),
            pca_applied=bool(pca_stats["applied"]),
        )
```

### A5. Add Memory Ceiling Check to FaceClusterer

The `FaceClusterer` training path has **no memory check** (unlike the discover path at lines 2184-2199). Without PCA, the existing discover formula `(N² × 8) / (1024³)` incorrectly estimates a full N×N distance matrix (Boruvka never builds one). After PCA, memory is dominated by the reduced embedding matrix, not a distance matrix.

Add a pre-clustering check in `cluster_unlabeled_faces()` after the embeddings array is built:

```python
        # Memory ceiling check (after embeddings built, before HDBSCAN)
        # After PCA to 50d, Boruvka uses KD-tree -- no full distance matrix.
        # 5x multiplier covers: embedding matrix + KD-tree + working memory.
        max_clustering_memory_gb = 4
        n_faces = len(embeddings)
        embedding_dim = embeddings_array.shape[1]
        estimated_memory_gb = (n_faces * embedding_dim * 4 * 5) / (1024**3)
        if estimated_memory_gb > max_clustering_memory_gb:
            logger.warning(
                f"Memory ceiling check: {estimated_memory_gb:.2f} GB estimated "
                f"for {n_faces} faces × {embedding_dim}d "
                f"(max: {max_clustering_memory_gb} GB). Proceeding anyway -- "
                "PCA will reduce actual usage significantly."
            )
```

Note: In the clusterer path this is a warning (not an error-abort), because PCA will be applied immediately after and the true memory usage will be far lower than the formula estimates. The discover path's hard abort at the pre-PCA stage is more conservative; the clusterer can be more permissive since it always applies PCA when n_dims > pca_target_dim.

---

## 4. Work Stream B: Suggestion Job Batching (P1)

### Problem

After clustering, the training job enqueues one RQ job per person (lines 786-829). With 90 persons:

```
~90 jobs × 245ms overhead = ~22s of pure queue/setup overhead
```

Each individual job creates its own DB session, Qdrant client (face collection), and Qdrant client (centroid collection). 92% of jobs produce zero new suggestions (the person already has max pending suggestions or no matching unassigned faces).

### B1. Extract Per-Person Logic into Helper Functions

Before creating batch jobs, extract the core per-person logic from the existing standalone jobs. This is a prerequisite for B2/B3 and also improves the standalone jobs' testability.

**For `find_more_centroid_suggestions_job`** (line 1693): Extract lines 1761-1900 into:
```python
def _find_centroid_suggestions_for_person(
    db_session: SyncSession,
    face_qdrant_client: Any,
    centroid_qdrant_client: Any,
    person_id: str,
    min_similarity: float = 0.70,
    max_results: int = 50,
    unassigned_only: bool = True,
    job_id: str = "no-job",
) -> dict[str, Any]:
    """Core logic for centroid-based suggestion finding.

    Accepts externally-managed sessions/clients for batch reuse.
    """
    # ... extracted logic ...
```

The existing standalone job becomes a thin wrapper:
```python
def find_more_centroid_suggestions_job(...) -> dict[str, Any]:
    db_session = get_sync_session()
    face_qdrant = get_face_qdrant_client()
    centroid_qdrant = get_centroid_qdrant_client()
    try:
        return _find_centroid_suggestions_for_person(
            db_session, face_qdrant, centroid_qdrant, ...
        )
    finally:
        db_session.close()
```

**For `propagate_person_label_multiproto_job`** (line 1433): Same pattern -- extract core logic into `_propagate_labels_for_person()` helper.

### B2. Create find_centroid_suggestions_batch_job

```python
def find_centroid_suggestions_batch_job(
    person_ids: list[str],
    min_similarity: float = 0.70,
    max_results_per_person: int = 50,
    unassigned_only: bool = True,
) -> dict[str, Any]:
    """Batch centroid suggestion job.

    Uses a single DB session and Qdrant client pair for all persons,
    eliminating the per-job connection setup overhead (~245ms per person).
    Each person is processed independently with skip-and-continue error handling.

    Args:
        person_ids: List of UUID strings for persons to process
        min_similarity: Minimum cosine similarity threshold
        max_results_per_person: Maximum suggestions to create per person
        unassigned_only: If True, only suggest unassigned faces

    Returns:
        dict with: persons_processed, persons_skipped, total_suggestions,
        failed_count, failed_person_ids
    """
    job = get_current_job()
    job_id = job.id if job else "no-job"

    db_session = get_sync_session()
    face_qdrant = get_face_qdrant_client()
    centroid_qdrant = get_centroid_qdrant_client()

    total_suggestions = 0
    persons_processed = 0
    persons_skipped = 0
    failed_person_ids: list[str] = []

    logger.info(
        f"[{job_id}] Starting centroid suggestion batch for "
        f"{len(person_ids)} persons"
    )

    try:
        for person_id in person_ids:
            try:
                result = _find_centroid_suggestions_for_person(
                    db_session=db_session,
                    face_qdrant_client=face_qdrant,
                    centroid_qdrant_client=centroid_qdrant,
                    person_id=person_id,
                    min_similarity=min_similarity,
                    max_results=max_results_per_person,
                    unassigned_only=unassigned_only,
                    job_id=job_id,
                )
                # Commit per-person to prevent one failure rolling back all work
                db_session.commit()

                created = result.get("suggestions_created", 0)
                if created > 0:
                    persons_processed += 1
                    total_suggestions += created
                else:
                    persons_skipped += 1

            except Exception as e:
                logger.warning(
                    f"[{job_id}] Failed to process person {person_id}: {e}"
                )
                db_session.rollback()  # Roll back this person's changes only
                failed_person_ids.append(person_id)
                continue

    finally:
        db_session.close()

    logger.info(
        f"[{job_id}] Centroid batch complete: "
        f"{persons_processed} processed, {persons_skipped} skipped, "
        f"{len(failed_person_ids)} failed, {total_suggestions} total suggestions"
    )

    return {
        "persons_processed": persons_processed,
        "persons_skipped": persons_skipped,
        "total_suggestions": total_suggestions,
        "failed_count": len(failed_person_ids),
        "failed_person_ids": failed_person_ids,
    }
```

### B3. Create propagate_labels_batch_job

Same pattern as B2 but for the prototype-based path:

```python
def propagate_labels_batch_job(
    person_ids: list[str],
    max_suggestions: int = 50,
    min_confidence: float = 0.7,
    preserve_existing: bool = True,
) -> dict[str, Any]:
    """Batch multi-prototype suggestion job.

    Processes multiple persons in a single job with shared DB session
    and Qdrant client, with per-person commit and skip-and-continue error handling.
    """
    # ... same structure as B2 but calling _propagate_labels_for_person() ...
```

### B4. Update Enqueue Site

**File**: `src/image_search_service/queue/face_jobs.py` lines 786-829

**Before** (per-person loop, 90 individual jobs):
```python
                for person in persons:
                    if use_centroids and person.face_count >= min_faces_for_centroid:
                        job = queue.enqueue(
                            "...find_more_centroid_suggestions_job",
                            person_id=str(person.id),
                            min_similarity=0.70,
                            max_results=50,
                            ...
                            job_timeout="10m",
                        )
                        centroid_jobs_queued += 1
                    else:
                        job = queue.enqueue(
                            "...propagate_person_label_multiproto_job",
                            person_id=str(person.id),
                            max_suggestions=50,
                            min_confidence=0.7,
                            ...
                            job_timeout="10m",
                        )
                        prototype_jobs_queued += 1
                    suggestions_jobs_queued += 1
```

**After** (two batch jobs maximum):
```python
                # Partition persons by job type
                centroid_person_ids = [
                    str(p.id) for p in persons
                    if use_centroids and p.face_count >= min_faces_for_centroid
                ]
                prototype_person_ids = [
                    str(p.id) for p in persons
                    if not (use_centroids and p.face_count >= min_faces_for_centroid)
                ]

                if centroid_person_ids:
                    queue.enqueue(
                        "image_search_service.queue.face_jobs"
                        ".find_centroid_suggestions_batch_job",
                        person_ids=centroid_person_ids,
                        min_similarity=0.70,
                        max_results_per_person=50,
                        unassigned_only=True,
                        job_timeout="30m",
                    )
                    centroid_jobs_queued = 1
                    suggestions_jobs_queued += 1

                if prototype_person_ids:
                    queue.enqueue(
                        "image_search_service.queue.face_jobs"
                        ".propagate_labels_batch_job",
                        person_ids=prototype_person_ids,
                        max_suggestions=50,
                        min_confidence=0.7,
                        preserve_existing=True,
                        job_timeout="30m",
                    )
                    prototype_jobs_queued = 1
                    suggestions_jobs_queued += 1
```

### B5. Pre-filter No-op Persons

Before building the person_ids lists, add a lightweight pre-check to exclude persons unlikely to produce suggestions:

```python
                # Pre-filter: exclude persons with max pending suggestions already
                # (single batch query instead of per-person check in each job)
                from image_search_service.db.models import FaceSuggestion, SuggestionStatus

                persons_with_pending = set(
                    row[0] for row in db_session.execute(
                        select(FaceSuggestion.person_id)
                        .where(FaceSuggestion.status == SuggestionStatus.PENDING)
                        .group_by(FaceSuggestion.person_id)
                        .having(func.count(FaceSuggestion.id) >= 50)
                    ).all()
                )

                eligible_persons = [
                    p for p in persons
                    if p.id not in persons_with_pending
                ]
```

---

## 5. Work Stream C: Supporting Fixes (P2-P3)

### C1. Fix Memory Ceiling Formula in discover_unknown_persons_job

**File**: `src/image_search_service/queue/face_jobs.py` lines 2184-2199

**Current formula** (line 2189):
```python
estimated_memory_gb = (total_faces**2 * 8) / (1024**3)
```

This formula assumes a full N×N distance matrix. At 23K faces this estimates 4GB, which triggers the abort. Boruvka never builds a full distance matrix.

**Corrected formula** that accounts for PCA:
```python
        # After PCA to 50d, Boruvka uses KD-tree -- no full distance matrix.
        # Memory breakdown for N faces at D dimensions:
        #   - Embedding matrix (float32): N × D × 4 bytes
        #   - KD-tree internal nodes: ~3× embedding matrix
        #   - Core distances array: N × 8 bytes
        #   - MST edges (sparse): ~N × 24 bytes
        # Use 5× multiplier over embedding matrix as conservative estimate.
        pca_dims = min(pca_target_dim, total_faces - 1) if pca_target_dim > 0 else 512
        embedding_bytes = total_faces * pca_dims * 4  # float32
        estimated_memory_gb = (embedding_bytes * 5) / (1024**3)
```

This allows clustering up to ~500K faces within the 4GB limit (vs. the current ~23K ceiling).

### C2. Pre-check Prototype Existence Before Enqueuing Propagation Jobs

**File**: `src/image_search_service/queue/face_jobs.py` lines 808-816

Currently, `propagate_person_label_multiproto_job` is enqueued for persons with insufficient face counts for centroids. Many of these persons have 0 prototypes, causing the job to return immediately with `"No prototypes"` after spending ~245ms on setup.

Add a pre-check query before building `prototype_person_ids`:

```python
                # Pre-check: only enqueue propagation for persons with prototypes
                from image_search_service.db.models import PersonPrototype

                persons_with_prototypes = set(
                    row[0] for row in db_session.execute(
                        select(PersonPrototype.person_id)
                        .where(PersonPrototype.person_id.in_(
                            [p.id for p in persons]
                        ))
                        .group_by(PersonPrototype.person_id)
                    ).all()
                )

                prototype_person_ids = [
                    str(p.id) for p in persons
                    if p.id in persons_with_prototypes
                    and not (use_centroids and p.face_count >= min_faces_for_centroid)
                ]
```

### C3. Fix [no-job] Log Context

Both `find_more_centroid_suggestions_job` (line 1726) and `propagate_person_label_multiproto_job` (line 1467) use `job_id = job.id if job else "no-job"`. When invoked from within a batch job, `get_current_job()` returns the batch job's ID, not "no-job". This is correct behavior -- the `[no-job]` log prefix only appears when the helper is called outside an RQ context (e.g., in tests or CLI). No fix needed for the batch path.

However, if the helper functions (`_find_centroid_suggestions_for_person` etc.) are extracted as in B1, they should accept `job_id` as an explicit parameter rather than calling `get_current_job()` internally.

---

## 6. Files Changed

| File | Changes | Work Stream | LOC Delta (est.) |
|------|---------|-------------|-----------------|
| `src/image_search_service/faces/embedding_preprocessing.py` | **NEW** -- shared PCA utility | A1 | +95 |
| `src/image_search_service/faces/clusterer.py` | Add PCA to `_run_hdbscan()`, `pca_target_dim` to constructor, memory check | A2, A3, A5 | +25 |
| `src/image_search_service/queue/face_jobs.py` | Refactor discover PCA to use shared util, extract per-person helpers, add batch jobs, fix memory formula, update enqueue site, prototype pre-check | A4, B1-B5, C1-C3 | +160 / -90 |
| `tests/unit/test_embedding_preprocessing.py` | **NEW** -- unit tests for shared utility | A1 | +80 |
| `tests/unit/test_clusterer.py` | Add PCA integration tests | A2, A3 | +40 |
| `tests/unit/test_face_jobs_batch.py` | **NEW** -- batch job tests | B1-B4 | +100 |

**Net LOC Delta**: approximately +410 added, -90 removed = **+320 net**

(Most additions are tests and the new shared utility. The face_jobs.py changes are net-negative.)

---

## 7. Testing Plan

### Unit Tests

**`test_embedding_preprocessing.py`**:

1. `test_reduce_dimensions_pca_applies_when_above_target()` -- verify 512d → 50d
2. `test_reduce_dimensions_pca_skips_when_below_target()` -- 30d input unchanged
3. `test_reduce_dimensions_pca_handles_fewer_samples_than_target()` -- 20 samples → 19 components
4. `test_reduce_dimensions_pca_disabled_when_target_zero()` -- pca_target_dim=0 skips PCA
5. `test_reduce_dimensions_pca_logs_warning_below_variance_threshold()` -- warning at <90%
6. `test_select_hdbscan_params_boruvka_when_pca_applied()` -- returns boruvka_kdtree at 50d
7. `test_select_hdbscan_params_best_when_no_pca()` -- returns best at 512d

**`test_clusterer.py`** additions:

8. `test_clusterer_run_hdbscan_applies_pca_by_default()` -- verify PCA log entries
9. `test_clusterer_pca_target_dim_zero_disables_pca()` -- pca_target_dim=0 passthrough
10. `test_clusterer_pca_target_dim_configurable_via_constructor()` -- custom dim respected

**`test_face_jobs_batch.py`**:

11. `test_find_centroid_suggestions_batch_processes_all_persons()` -- all persons attempted
12. `test_find_centroid_suggestions_batch_commits_per_person()` -- one commit per person
13. `test_find_centroid_suggestions_batch_continues_after_person_failure()` -- skip-and-continue
14. `test_find_centroid_suggestions_batch_reports_failed_person_ids()` -- failed list populated
15. `test_propagate_labels_batch_processes_all_persons()` -- same structure

### Integration Tests

16. Run `detect_faces_for_session_job` with a 6-image dataset -- verify PCA log entries appear from the `clusterer.py` path (not just the discover path)
17. Run batch suggestion job with 10 persons (mock Qdrant) -- verify total time is significantly less than 10 × 245ms = 2.45s
18. Verify clustering results equivalence: same synthetic data should produce equivalent cluster groupings with PCA vs. without PCA (ARI >= 0.85)

### Performance Verification

1. **Baseline**: Run training session, record HDBSCAN time from worker logs
2. **After Work Stream A**: Re-run, confirm "PCA: 512d -> 50d" log from clusterer.py path
3. **After Work Stream B**: Re-run, confirm 2 batch jobs instead of 90 individual jobs in RQ dashboard
4. **Before/after comparison**: Total pipeline time should drop from ~111.6s to 15-25s

---

## 8. Estimated Impact

| Component | Before | After | Reduction |
|-----------|--------|-------|-----------|
| HDBSCAN clustering (training path) | 72.5s | 1-5s | **93-99%** |
| Post-training suggestions (90 jobs → 2 batch jobs) | 28.1s | 2-4s | **86-93%** |
| Everything else | 11.0s | 10-15s | neutral (slight overhead from PCA) |
| **Total pipeline** | **111.6s** | **~15-25s** | **~80%** |

### Projected Scaling

| Dataset Size | Before (Prim's, 512d) | After (Boruvka, 50d) |
|-------------|----------------------|----------------------|
| 5K faces | ~20s | <1s |
| 10K faces | ~72s | 1-5s |
| 50K faces | ~30 min | 15-30s |
| 100K faces | ~2+ hours | 45-90s |

---

## 9. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| PCA degrades clustering quality | Low | Medium | Phase 1 already validated this approach in the discover path; 90% variance warning threshold; `pca_target_dim=0` opt-out; ARI comparison test |
| Batch job timeout on large datasets | Low | Low | 30m timeout is generous; individual persons are fast (~0.25s each even without batching) |
| Shared utility refactor breaks discover path | Medium | Medium | Implement in two sub-phases: A (new code) then B (refactor existing); integration tests verify equivalence |
| Per-person commit pattern causes performance regression | Low | Low | 90 commits vs. 1 commit adds ~15ms total; acceptable for correctness benefit |
| Memory check formula too permissive after fix | Low | High | 5× multiplier is conservative; log actual memory usage during development for tuning |
| Prototype pre-check query misses persons | Very Low | Low | Query is additive -- persons without prototypes are simply excluded from the batch, not erroneously included |

---

## 10. Dependencies

- **Phase 1 complete**: ✅ PCA code exists in discover path as reference implementation (lines 2218-2303)
- **sklearn already a dependency**: ✅ Used by discover path; confirmed in uv.lock (scikit-learn 1.8.0)
- **hdbscan already a dependency**: ✅ Confirmed in pyproject.toml
- **No new external dependencies required**
- **No API contract changes** -- purely backend/worker optimization
- **No database schema changes**

---

## 11. Rollout Plan

### Phase 1.5a: Add PCA to clusterer.py (NEW code only, minimal risk)

1. Create `embedding_preprocessing.py` with shared utility
2. Modify `clusterer.py` `_run_hdbscan()` to call the utility
3. Run tests: `make lint && make typecheck && make test`
4. Deploy and verify worker logs show "PCA: 512d -> 50d" from the training path

### Phase 1.5b: Refactor discover path to use shared utility (modify existing code)

5. Replace inline PCA in `discover_unknown_persons_job` with calls to shared utility
6. Run tests and verify discover path still produces equivalent results
7. Deploy and verify discover path log format is unchanged (just sourced from utility)

### Phase 1.5c: Batch suggestion jobs

8. Extract per-person helper functions (B1)
9. Create batch job functions (B2, B3)
10. Update enqueue site (B4) and add pre-filter (B5)
11. Run tests: verify batch job structure, error handling, and skip-and-continue
12. Deploy and verify RQ dashboard shows 2 jobs instead of 90+

### Phase 1.5d: Supporting fixes

13. Fix memory formula (C1) -- low risk, straightforward
14. Add prototype pre-check (C2) -- eliminates "No prototypes" log noise
15. Deploy and run full pipeline test with before/after timing comparison

---

## References

- Phase 1 implementation: `src/image_search_service/queue/face_jobs.py` lines 2218-2303
- Training path calling clusterer: `src/image_search_service/queue/face_jobs.py` lines 685-720
- FaceClusterer source: `src/image_search_service/faces/clusterer.py` lines 188-218
- Memory ceiling code: `src/image_search_service/queue/face_jobs.py` lines 2184-2199
- find_more_centroid_suggestions_job: `src/image_search_service/queue/face_jobs.py` line 1693
- propagate_person_label_multiproto_job: `src/image_search_service/queue/face_jobs.py` line 1433
- Suggestion enqueue site: `src/image_search_service/queue/face_jobs.py` lines 786-829
- Research: `docs/research/clusterer-fit-predict-perf/findings.md`
- Research: `docs/research/clusterer-fit-predict-perf/verification.md`
- Prior devil's advocate: `docs/plans/clusterer-fit-predict-perf/devils-advocate-review.md`
