# Worker Log Analysis: Training 6 Images Pipeline (2026-02-21)

**Date**: 2026-02-21
**Source**: `testrun.txt` worker log (1489 lines), cross-referenced against plans in `docs/plans/clusterer-fit-predict-perf/`
**Pipeline**: `detect_faces_for_session_job` triggered by training 6 images

---

## 1. Pipeline Timing Breakdown

The end-to-end pipeline for training 6 images took **111.6 seconds** total.

| Stage | Duration | % of Total |
|-------|----------|------------|
| Training session (6 images) | 3.2s | 3% |
| Face detection (21 faces) | 1.7s | 2% |
| Auto-assignment (21 faces vs 106 persons) | 0.2s | <1% |
| Qdrant face scroll (10 pages, 10K faces) | 2.5s | 2% |
| **HDBSCAN clustering (10K faces at 512d)** | **72.5s** | **65%** |
| Cluster ID updates to Qdrant (~243 clusters) | 2.4s | 2% |
| DB/metadata finalization | 0.2s | <1% |
| Post-training centroid jobs (~75 jobs) | 23.8s | 21% |
| Post-training multi-proto jobs (~15 jobs) | 4.3s | 4% |

**Key takeaway**: HDBSCAN is 65% of the total pipeline. Post-training suggestion jobs account for an additional 25%. Together these two stages consume 90% of wall time.

---

## 2. Critical Gap -- PCA Not Applied in clusterer.py

**This is the most important finding.**

Phase 1 of the existing performance plan implemented PCA preprocessing in `discover_unknown_persons_job` (`face_jobs.py` lines 2218-2275), but the log shows clustering runs through `FaceClusterer._run_hdbscan()` in `clusterer.py` (lines 188-218), which was **NOT** updated with PCA.

### Evidence

- **Log line 84**: Logger name is `image_search_service.faces.clusterer` -- this is the `FaceClusterer` code path, not the `discover_unknown_persons_job` path.
- **No PCA-related log entries**: Across all 1489 lines, there are zero mentions of PCA, dimensionality reduction, or `n_components`. The Phase 1 implementation logs PCA parameters when it runs; their absence proves it was not invoked.
- **HDBSCAN at 512d selects Prim's MST algorithm**: At 512 dimensions, HDBSCAN internally selects the `prims_balltree` algorithm for minimum spanning tree construction. This algorithm ignores the `core_dist_n_jobs=-1` parallelism parameter, forcing single-threaded execution.
- **Result**: 72.5 seconds for 10K faces with the Phase 1 fix completely bypassed.

### Why This Happened

The `detect_faces_for_session_job` workflow (triggered by training images) calls `FaceClusterer.cluster()` in `clusterer.py`. This is a **separate code path** from `discover_unknown_persons_job` in `face_jobs.py`, where Phase 1 PCA was implemented. The training workflow -- which is the most common user-facing use case -- does not benefit from Phase 1 at all.

---

## 3. Plan Coverage Assessment

Cross-referencing against `docs/plans/clusterer-fit-predict-perf/overview.md` and phase documents:

### What the plans DO address (and status)

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | HDBSCAN dimension reduction via PCA | Implemented, but only in `discover_unknown_persons_job` code path |
| Phase 2 | Batch DB updates | Implemented in discover path; already existed in `clusterer.py` |
| Phase 3 | Progress feedback | Not yet implemented |
| Phase 4 | Consolidation of duplicate clustering code paths | Not yet implemented |
| Phase 5 | Future GPU acceleration | Not yet implemented |

Phase 4 (consolidation) would have caught this gap -- the existence of two separate clustering code paths is exactly the problem Phase 4 aims to solve. However, Phase 4 was deferred in favor of the earlier phases.

### What the plans DON'T address (gaps)

1. **Post-training suggestion jobs serial execution** (28.1s, 25% of pipeline): After clustering completes, 75+ individual centroid suggestion jobs and ~15 propagation jobs are enqueued serially. Each job carries ~245ms of dequeue/setup overhead. The plans do not address this at all.

2. **92% waste rate in suggestion jobs**: Out of ~75 centroid suggestion jobs, only ~7 produce actual suggestions. Approximately 25 find zero candidates and ~40 find only duplicates already suggested. The plans do not discuss pre-filtering to avoid wasted work.

3. **Redundant Qdrant collection checks** (75+ per run): Each individual suggestion job performs its own Qdrant collection existence check. With 75+ jobs per pipeline run, this produces 75+ redundant checks against the same collection.

4. **Memory ceiling formula is overly conservative** (blocks at ~23K faces): The formula `estimated_memory_gb = (total_faces**2 * 8) / (1024**3)` assumes a full N x N distance matrix. HDBSCAN with euclidean metric and Boruvka algorithm does NOT build a full distance matrix. With PCA to 50 dimensions, actual memory for 23K faces is approximately 250MB, but the formula estimates 3.95GB and hits the 4GB cap. This is mentioned in `devils-advocate-review.md` Priority 7 but is not part of any implementation phase.

5. **Auto-assignment serial HTTP pattern** (O(N) Qdrant calls per face): During auto-assignment, each detected face triggers an individual Qdrant similarity search. For 21 faces this is negligible (0.2s), but at scale (e.g., batch import of 500 images with 2000 faces) this becomes a bottleneck.

6. **`[no-job]` log context issue**: Every log entry across all 1489 lines shows `[no-job]` for the RQ job ID. `get_current_job()` returns None, likely because the ListenerWorker does not properly set RQ execution context before invoking job functions. This is a debugging/observability gap, not a performance issue.

---

## 4. Warning and Error Catalog

### 4.1 FutureWarning: scikit-image deprecation

**Source**: `insightface/utils/face_align.py:23`
**Message**: `estimate` method deprecated since scikit-image 0.26, will be removed in version 2.2.
**Action**: Pin scikit-image below 2.2, or report upstream to InsightFace project for a fix. This warning is emitted once per face detection call.

### 4.2 No prototypes warning

**Affected person**: `0e3b6fb9-f053-49fe-8119-14c0e073d205`
**Message**: Person had no prototypes but was still queued for label propagation.
**Action**: Add a pre-check filter before enqueuing propagation jobs. Persons without at least one prototype face cannot meaningfully propagate labels.

### 4.3 Missing centroids

**Frequency**: 5 persons (~7% of active persons)
**Message**: "No active centroid found, computing..."
**Action**: These persons triggered on-demand centroid computation during suggestion jobs, adding latency. Indicates incomplete centroid pre-computation after clustering. A post-clustering centroid refresh step would eliminate this.

### 4.4 `[no-job]` log context

**Frequency**: All 1489 log lines
**Message**: Every entry shows `[no-job]` for `job_id`.
**Cause**: `get_current_job()` returns `None`. The `ListenerWorker` likely does not set the RQ execution context before calling job functions, so the structured logging middleware cannot retrieve the job ID.
**Action**: Either pass job_id explicitly through the function call chain, or configure the ListenerWorker to set RQ execution context.

---

## 5. Scaling Projections

| Component | 10K faces | 50K faces | 100K faces |
|-----------|-----------|-----------|------------|
| HDBSCAN (512d, Prim's MST) | 72.5s | **~30 min** | **~2 hours** |
| HDBSCAN (50d PCA, Boruvka) | ~1-5s | ~15-30s | ~30-60s |
| Qdrant scroll | 2.5s | ~12s | ~24s |
| Post-training (per 100 persons) | 28s | 28s* | 28s* |
| Post-training (per 500 persons) | -- | ~135s | ~135s |
| Post-training (per 2000 persons) | -- | -- | ~540s |

*Person count determines post-training time, not face count.

### Important caveats

- The `max_faces=10000` cap in `face_jobs.py` limits clustering to a 10K face sample. At 100K total faces, 90% are excluded from clustering entirely. This sampling strategy may need revisiting as the dataset grows.
- HDBSCAN at 512d scales approximately as O(N^2) due to Prim's MST. With PCA to 50d and Boruvka, scaling improves to approximately O(N log N).
- Post-training job time scales linearly with person count, not face count. Each person triggers one centroid suggestion job and potentially one propagation job.
- Qdrant scroll time scales linearly with total face count (fixed page size of 1000, so pages = total_faces / 1000).
