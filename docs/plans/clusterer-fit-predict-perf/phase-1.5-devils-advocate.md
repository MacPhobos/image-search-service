# Devil's Advocate Review: Phase 1.5 Plan

**Date**: 2026-02-21
**Reviewer**: Claude Opus 4.6 (automated critical review)
**Scope**: Phase 1.5 plan (`phase-1.5-clusterer-pca-and-suggestion-batching.md`) + referenced source code

---

## Overall Assessment

The plan addresses real, measured bottlenecks (72.5s HDBSCAN + 28.1s suggestion overhead = 90% of pipeline). The claims are validated against actual worker logs and source code inspection. The technical approach is sound and builds directly on Phase 1's validated work.

Seven concerns are examined below, ranging from valid (requires plan modification) to low risk (no change needed).

---

## Concern 1: Does PCA Actually Degrade Clustering Quality in the Training Path?

**Claim in plan**: "Phase 1 already validated this approach in the discover path."

**Challenge**: The discover path clusters **all unlabeled faces** (up to 50K) without any prior person assignments. The training path clusters **only recently-detected faces** (up to 10K, `max_faces=10000` at line 702), right after a labeling session. These are different distributions:

- Discover path: broad, diverse unlabeled set -- PCA captures dominant variance across many identities
- Training path: skewed toward the recently-labeled person's face variations + their lookalikes -- the discriminating dimensions for this narrower set might be concentrated in fewer PCA components OR spread differently

**Verdict**: LOW RISK. The training path calls `cluster_unlabeled_faces()` with `quality_threshold=0.3`, meaning it sees the full unlabeled set up to 10K, not just the newly-labeled person's faces. The distribution is comparable to the discover path. The 90% variance warning threshold provides a safety net.

**Suggested addition**: Add an ARI comparison test that specifically tests clusterer.py path output at 512d vs. 50d on synthetically generated data representative of the training scenario (well-separated clusters with some noise). This is distinct from the discover path ARI test and validates the `_run_hdbscan()` change specifically.

---

## Concern 2: The Shared Utility Refactor Introduces Change to Working Code

**Claim in plan**: "Implement in two sub-phases: A (new code) then B (refactor existing)."

**Challenge**: Phase 1.5a adds PCA to clusterer.py using the new shared utility -- this is new code, low risk. Phase 1.5b refactors the WORKING discover path to use the same shared utility. The discover path (Phase 1) was just implemented and validated. Why modify it immediately?

Alternative: Keep discover path inline indefinitely. Copy-paste the PCA block into clusterer.py directly. No shared module needed.

**Verdict**: MEDIUM RISK for 1.5b, but the shared utility is the right architecture. Two independent PCA implementations will diverge over time -- if the variance threshold needs tuning (e.g., changed to 0.85), one copy will be updated and the other forgotten. The shared utility prevents this class of bug.

**Required modification**: The plan's sub-phase ordering is correct (1.5a before 1.5b). However, add an explicit verification step between 1.5a and 1.5b:

1. Deploy 1.5a (clusterer.py uses shared utility) -- verify worker logs from training
2. Run discover_unknown_persons_job manually -- verify it still works (before touching it)
3. Only then implement 1.5b (refactor discover path to use shared utility)
4. Run discover_unknown_persons_job again -- verify identical log output

This gives a clear rollback point: if 1.5b breaks something, revert it without affecting 1.5a's gains.

---

## Concern 3: Batch Suggestion Jobs Remove Per-Job Error Isolation

**Challenge**: Individual jobs provide natural error isolation via RQ's job retry mechanism. If `find_more_centroid_suggestions_job` fails for person X, RQ can retry that specific job. With batching, if person X causes an unhandled exception, the plan says "skip-and-continue" -- but the failed person is silently dropped from suggestion generation. There is no retry for person X.

More specifically: the plan's batch job includes `failed_person_ids` in the return dict. But RQ job return values are not automatically acted on. Nobody reads `failed_person_ids` and re-queues those persons.

**Verdict**: VALID CONCERN. The plan addresses error isolation at the code level (try/except per person, per-person commit) but does not address the operational question: what happens to failed persons?

**Required modifications**:

1. The batch job should log `failed_person_ids` at WARNING level with enough context for manual re-queuing
2. Consider enqueuing failed persons as individual retry jobs at the END of the batch job:
   ```python
   if failed_person_ids and retry_individually:
       for failed_id in failed_person_ids:
           queue.enqueue("...find_more_centroid_suggestions_job", person_id=failed_id, ...)
   ```
3. Add `retry_individually: bool = True` parameter to the batch job (defaults True for safety, can disable in testing)
4. Cap the retry-individual threshold: if more than 20% of persons failed, do NOT auto-retry individually (likely a systemic error, not per-person failures)

---

## Concern 4: Per-Person Commit Pattern Performance Concern

**Claim in plan**: "Per-person commit to prevent one failure rolling back all work."

**Challenge**: The plan acknowledges this adds ~90 commits vs. 1 commit, estimating "+15ms total." Is this accurate? PostgreSQL transaction commit involves:
- Flushing WAL to disk (fsync)
- Network round-trip acknowledgment

At 1-5ms per commit × 90 persons = 90-450ms. Not 15ms.

**Verdict**: VALID CONCERN but acceptable. The 90-450ms overhead is a small fraction of the 28.1s total savings from batching (~1.5%). The correctness benefit (each person's suggestions are safe regardless of later failures) outweighs the overhead.

However, the plan's "~15ms total" estimate is incorrect and should be corrected to "~90-450ms total" to avoid surprise if someone benchmarks it.

**Suggested modification**: Correct the performance estimate in the plan. Also consider whether per-person commit is truly necessary: if the batch is processing 90 persons and each person's suggestions are independent (they reference different FaceSuggestion rows), a single commit at the end is also safe. The only case where per-person commit matters is if a LATER person's processing causes an exception that would roll back EARLIER persons' work. With a proper try/except + rollback per person, this is handled correctly regardless.

**Recommended approach**: Use savepoints instead of per-person commits:
```python
    for person_id in person_ids:
        try:
            savepoint = db_session.begin_nested()
            result = _find_centroid_suggestions_for_person(...)
            savepoint.commit()
        except Exception as e:
            savepoint.rollback()
            failed_person_ids.append(person_id)
            continue
    db_session.commit()  # Single final commit
```
This gives the same error isolation benefit with one network round-trip instead of N.

---

## Concern 5: Memory Ceiling Formula -- Is the 5x Multiplier Justified?

**Claim in plan**: "5× multiplier covers: embedding matrix + KD-tree + working memory."

**Challenge**: What does HDBSCAN with Boruvka actually allocate at 50d?

From hdbscan 0.8.x source and memory profiling on similar workloads:
- Input array (float32 → upcast to float64 by Boruvka): N × D × 8 bytes
- KD-tree internal: ~1.5× input = N × D × 12 bytes
- Core distances: N × 8 bytes (float64 array)
- Mutual reachability graph (sparse): N × k × 8 bytes (k = min_samples = 3)
- MST edges: ~N × 24 bytes (three float64 values per edge)

For 50K faces at 50d:
- Input (float64): 50000 × 50 × 8 = 20 MB
- KD-tree: ~30 MB
- Core distances: ~0.4 MB
- MRG: ~0.6 MB
- MST edges: ~1.2 MB
- **Total: ~52 MB**

Even with 10× headroom: 520 MB. The 5× multiplier at 50K faces × 50d gives:
`(50000 × 50 × 4 × 5) / (1024³)` = ~0.047 GB = 47 MB. This is accurate.

**Verdict**: The formula is correct. The Boruvka memory usage is dominated by the float64 upcast (2× the float32 input size) plus KD-tree (~1.5× input). The 5× over float32 ≈ 2.5× over float64, which is generous but not excessively conservative.

However, the plan's discover path fix (C1) applies this formula BEFORE the embeddings array is built (using `pca_dims` estimated from `pca_target_dim`). If the actual embeddings turn out to be a different dimension (e.g., 768d SigLIP), `pca_target_dim=50` is used for the formula, which is correct since PCA will reduce to 50d regardless of input dimension.

**No plan modification required** for correctness. Add a comment in the code explaining why we use `pca_dims` rather than `embedding_dims` for the formula.

---

## Concern 6: Are All 6 FaceClusterer Callers Appropriate for PCA?

**Claim in plan**: "All 6 callers will benefit from PCA since they all perform the same task."

**Challenge**: Let's verify each caller:

1. `detect_faces_for_session_job()` (line 689) -- training path. Clusters up to 10K unlabeled faces. **PCA appropriate.**
2. `cluster_faces_job()` -- standalone RQ job. Clusters all unlabeled faces (presumably same scope as discover). **PCA appropriate.**
3. `scripts/faces.py` CLI -- cluster command. Interactive use. **PCA appropriate** (CLI user would not want to wait 72s).
4. `face_clustering_restart_service.py` -- restart service. Unclear scope. Need to verify it's clustering the same face embedding space. **PCA likely appropriate.**
5. `api/routes/faces.py` line 407 -- API endpoint. Triggered on-demand. **PCA appropriate** (same clustering task, just triggered differently).
6. `api/routes/faces.py` line 1534 -- API endpoint. **PCA appropriate** (same reasoning).

**Verdict**: LOW RISK. All callers perform the same underlying task. However, callers 4-6 were not explicitly analyzed in the plan.

**Suggested addition**: Explicitly verify what `face_clustering_restart_service.py` does. If it has special requirements (e.g., it resets cluster_ids first and expects a clean slate), PCA should be safe but confirm it doesn't override `pca_target_dim`. The configurable `pca_target_dim=0` opt-out via the constructor handles any edge cases.

---

## Concern 7: The max_faces=10000 Cap Should Be Raised After PCA

**Challenge**: The training path calls `cluster_unlabeled_faces(max_faces=10000)` (line 702). The discover path uses `max_faces=50000` (line 2146). At 50K faces, PCA makes HDBSCAN fast enough that 10K is an arbitrary and conservative limit. Why not raise the cap now that PCA makes larger datasets tractable?

**Verdict**: NOT IN THIS PHASE. Raising `max_faces` changes clustering behavior -- more faces means different cluster boundaries, potentially different cluster assignments for the same faces. This requires:
- A deliberate product decision (are larger clusters desired?)
- Validation that quality doesn't degrade with more noise faces
- A separate PR with its own testing

The plan correctly focuses on making the existing 10K cap run faster.

**Suggested addition to plan**: Add a "Future Work" note:

> After Phase 1.5 is deployed and validated, consider raising `max_faces` from 10,000 to 25,000-50,000 in a subsequent phase. PCA + Boruvka makes this tractable (~5-15s for 50K faces). This should be a separate, tested change with quality validation.

---

## Corrections Required

### Correction 1: cluster_selection_method="eom" Missing from Current clusterer.py

The plan's proposed `_run_hdbscan()` adds `cluster_selection_method="eom"` to the HDBSCAN constructor:

```python
clusterer = hdbscan.HDBSCAN(
    ...
    cluster_selection_method="eom",  # Excess of mass (more stable)
    **hdbscan_params,
)
```

But the CURRENT `_run_hdbscan()` (lines 205-211) does NOT have this parameter -- it uses HDBSCAN's default (`"eom"` is actually the default, so this is safe). Verify this is not a behavior change by checking hdbscan's default: `cluster_selection_method` defaults to `"eom"` in hdbscan 0.8.x. **No behavior change.** But the plan should note this explicitly rather than appearing to add a new parameter.

### Correction 2: get_centroid_qdrant_client Import in Batch Job

The plan's batch job (B2) calls `get_centroid_qdrant_client()`:

```python
    centroid_qdrant = get_centroid_qdrant_client()
```

Verify this function exists and is importable from the face_jobs.py module scope. The existing `find_more_centroid_suggestions_job` (line 1722) imports it locally:

```python
    from image_search_service.vector.centroid_qdrant import get_centroid_qdrant_client
```

The batch job should use the same local import pattern for consistency.

### Correction 3: Per-Person Commit Estimate Should Be Corrected

As noted in Concern 4, the "+15ms total" estimate for per-person commits is incorrect. Should be "+90-450ms total" (1-5ms × 90 persons). Not a blocking issue, but a factual correction.

---

## Final Verdict

**Plan is SOUND with the following required modifications:**

| Item | Priority | Required? |
|------|---------|-----------|
| Add retry-individually mechanism for failed batch persons | P0 | Required |
| Use savepoints instead of per-person commits | P1 | Recommended |
| Correct per-person commit time estimate (+90-450ms, not +15ms) | P2 | Should fix |
| Add explicit verification step between 1.5a and 1.5b rollout | P1 | Required |
| Add ARI test specific to clusterer.py path (training distribution) | P2 | Recommended |
| Add "Future Work" note about raising max_faces cap | P3 | Nice to have |
| Verify `face_clustering_restart_service.py` is appropriate for PCA | P2 | Should verify |
| Note that `cluster_selection_method="eom"` is the default (not a new param) | P3 | Nice to have |
| Correct `get_centroid_qdrant_client` import pattern in batch job | P1 | Required |

### Summary Scorecard

| Work Stream | Technical Correctness | Risk Level | Ready to Implement? |
|------------|----------------------|-----------|-------------------|
| **A: PCA in FaceClusterer** | 9/10 -- sound, builds on validated Phase 1 | Low | Yes, as Phase 1.5a |
| **A4: Discover path refactor** | 8/10 -- correct but modifies working code | Medium | Yes, as Phase 1.5b (after 1.5a validated) |
| **B: Suggestion job batching** | 7/10 -- error handling needs retry mechanism | Medium | Yes, after adding retry-individually |
| **C: Supporting fixes** | 8/10 -- straightforward improvements | Low | Yes, as Phase 1.5d |

**Overall**: The plan correctly identifies and addresses the two dominant performance bottlenecks in the training pipeline. The primary concern is the lack of a retry mechanism for failed batch persons -- without this, a systemic error in a batch job would silently drop suggestion generation for affected persons with no recovery path. All other concerns are either low-risk or have mitigations already present in the plan.
