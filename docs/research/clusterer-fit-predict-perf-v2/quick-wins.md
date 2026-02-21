# Quick Wins: Performance Improvements Beyond Existing Plans

**Date**: 2026-02-21
**Context**: Improvements NOT already outlined in phases 2-5 of `docs/plans/clusterer-fit-predict-perf/`. Each is validated against actual worker log evidence from the training-6-images pipeline run.

---

## QW-1: Add PCA to `clusterer.py._run_hdbscan()` [CRITICAL]

**Impact**: 72.5s to ~1-5s (50-70x speedup)
**Effort**: Low (~20 lines of code)
**Confidence**: High -- same pattern already proven in `discover_unknown_persons_job`

### Problem

Phase 1 PCA was implemented only in `discover_unknown_persons_job` (`face_jobs.py` lines 2218-2275), but the training workflow calls `FaceClusterer._run_hdbscan()` in `clusterer.py` (lines 188-218), which operates on raw 512-dimensional embeddings.

At 512 dimensions, HDBSCAN internally selects the `prims_balltree` algorithm for minimum spanning tree construction. This algorithm:
- Ignores the `core_dist_n_jobs=-1` parallelism parameter (forced single-threaded)
- Scales approximately O(N^2) with face count
- Results in 72.5 seconds for 10K faces

### Solution

Add the same PCA preprocessing to `clusterer.py._run_hdbscan()`:

1. Import sklearn PCA
2. Before HDBSCAN `fit_predict`, reduce embeddings from 512d to 50d
3. At 50d, HDBSCAN switches from Prim's MST to the Boruvka algorithm (which supports parallelism via `core_dist_n_jobs`)
4. Set `algorithm='boruvka_kdtree'` explicitly for reliability

### Files

- `src/image_search_service/faces/clusterer.py` lines 188-218

### Evidence

Log shows 72.5 seconds of wall time between "Clustering 10000 unlabeled faces" (06:59:05.586) and the first cluster_id update (07:00:18.132). No PCA log entries are present anywhere in the 1489-line log.

---

## QW-2: Batch Post-Training Suggestion Jobs [HIGH IMPACT]

**Impact**: 28.1s to ~2-4s
**Effort**: Medium (new batch job function)
**Confidence**: High -- overhead clearly measured in logs

### Problem

After clustering completes, 75+ individual RQ jobs are enqueued serially (one per person) for centroid suggestions, plus ~15 propagation jobs. Each job carries approximately 245ms of dequeue overhead plus its own Qdrant collection existence check. Total: 28.1 seconds for ~90 jobs.

The overhead breakdown per job:
- ~100ms RQ dequeue and deserialization
- ~50ms Qdrant collection existence check (redundant, same collection every time)
- ~50ms Qdrant client initialization
- ~45ms job setup/teardown

### Solution

Replace individual per-person jobs with batch variants:

1. `find_more_centroid_suggestions_batch_job(person_ids=[...])` -- iterates internally over all person IDs in a single job
2. `propagate_labels_batch_job(person_ids=[...])` -- iterates internally over all person IDs in a single job
3. Single Qdrant client initialization and single collection check per batch
4. Eliminates ~90 job dequeue/setup/teardown cycles

### Files

- `src/image_search_service/queue/face_jobs.py` lines 786-827 (enqueue site)

### Evidence

Log timestamps show a consistent ~245ms inter-job gap across all 90 jobs. First centroid job starts at 07:00:20.996, last propagation job ends at 07:00:49.363. Total elapsed: 28.1 seconds.

---

## QW-3: Fix Memory Ceiling Formula [LOW EFFORT, HIGH VALUE]

**Impact**: Unblocks clustering above ~23K faces
**Effort**: Very low (change one formula)
**Confidence**: High -- documented in `devils-advocate-review.md` Priority 7 but not implemented

### Problem

The memory check formula in `face_jobs.py` lines 2187-2199:

```python
estimated_memory_gb = (total_faces**2 * 8) / (1024**3)
```

This assumes a full N x N distance matrix. HDBSCAN with euclidean metric does NOT build a full distance matrix -- it uses sparse distance computation via the Boruvka or Prim's algorithm.

With PCA to 50 dimensions, actual memory usage for 23K faces is approximately 250MB (input array + working copy + sparse graph). But the formula estimates 3.95GB, hitting the 4GB cap and blocking clustering entirely.

### Solution

Update `face_jobs.py` lines 2187-2199 with a realistic memory estimate:

**Option A** -- formula based on actual HDBSCAN memory model:
```python
estimated_memory_gb = (total_faces * reduced_dims * 8 * 3) / (1024**3)
```
This accounts for: input array + working copy + sparse graph overhead.

**Option B** -- raise the cap after PCA is confirmed working, since PCA-reduced clustering uses far less memory than the quadratic estimate implies.

### Files

- `src/image_search_service/queue/face_jobs.py` lines 2187-2199

---

## QW-4: Skip No-Op Suggestion Jobs [QUICK WIN]

**Impact**: Eliminates ~68 of 75 wasted jobs, saves ~16s
**Effort**: Low (pre-filter query)
**Confidence**: High -- 92% waste rate visible in logs

### Problem

Most centroid suggestion jobs produce zero actionable suggestions. Out of ~75 jobs:
- ~25 found 0 candidates (no faces in nearby clusters)
- ~40 found only duplicates (all candidates already suggested previously)
- Only ~7 produced actual new suggestions

Each wasted job still pays the full ~245ms overhead cost.

### Solution

Before enqueuing suggestion jobs at `face_jobs.py` line 786:

1. Query for persons whose cluster assignments changed in the current clustering run (these are the only ones that could have new suggestions)
2. Skip persons whose clusters were unchanged -- they cannot have new candidates
3. Alternative approach: batch query to check which persons have faces in recently-updated clusters, then enqueue only for those persons

### Evidence

Log shows repeated patterns of:
- "No candidates found for person X" (zero results from similarity search)
- "All N candidates already suggested for person Y" (duplicates filtered out, nothing new)

These patterns account for approximately 68 of 75 centroid suggestion jobs.

---

## QW-5: Fix `[no-job]` Log Context [DEBUGGING AID]

**Impact**: Improved debuggability (no performance impact)
**Effort**: Low
**Confidence**: High

### Problem

Every log entry across all 1489 lines shows `[no-job]` for the `job_id` field. The `get_current_job()` RQ function returns `None` because the `ListenerWorker` does not properly set the RQ execution context before calling job functions.

This makes it impossible to correlate log entries with specific RQ jobs when debugging production issues or analyzing performance.

### Solution

Either:
- Pass `job_id` explicitly through the function call chain (from the worker entry point down to each job function)
- Or configure the `ListenerWorker` to set the RQ execution context (via `rq.worker.set_current_job()` or equivalent) before invoking job functions

### Evidence

All 1489 log lines show `[no-job]` context. No job ID is ever populated.

---

## QW-6: Pre-Check Prototype Existence Before Propagation [QUICK FIX]

**Impact**: Eliminates warning noise, saves ~0.25s per wasted job
**Effort**: Very low (add one filter)
**Confidence**: High

### Problem

Propagation jobs are enqueued for persons without any prototype faces. These jobs immediately fail with a warning ("No prototypes found for person ...") and exit without doing useful work. One concrete instance found: person `0e3b6fb9-f053-49fe-8119-14c0e073d205`.

### Solution

At `face_jobs.py` lines 808-816, filter the `person_ids` list to include only those persons with at least one prototype face before enqueuing propagation jobs. This can be done with a single DB query joining persons to their prototype faces.

---

## Summary Priority Matrix

| ID | Quick Win | Impact | Effort | Priority |
|----|-----------|--------|--------|----------|
| QW-1 | PCA in `clusterer.py` | 72.5s to ~1-5s | Low | **P0 -- Do First** |
| QW-2 | Batch suggestion jobs | 28.1s to ~2-4s | Medium | **P1** |
| QW-3 | Fix memory ceiling | Unblocks 23K+ faces | Very Low | **P2** |
| QW-4 | Skip no-op suggestions | ~16s saved | Low | **P3** |
| QW-5 | Fix log context | Debugging aid | Low | **P4** |
| QW-6 | Pre-check prototypes | Warning fix | Very Low | **P5** |

### Combined Impact

**QW-1 + QW-2**: Pipeline time drops from **111.6s to ~15-25s** (approximately 80% reduction).

**QW-1 + QW-2 + QW-4**: Pipeline time drops to **~10-15s** (approximately 87% reduction).

**All quick wins**: Pipeline time of ~10-15s, clustering unblocked above 23K faces, clean logs with proper job context, and no wasted propagation jobs.
