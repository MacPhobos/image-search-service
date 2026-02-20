# Training Session Progress Reporting Delay: Root Cause Analysis and Solution

**Date**: 2026-02-19
**Status**: Investigation Complete — Solution Proposed
**Severity**: High (blocks UI, degrades service for all users during scan)
**Files Affected**: See Section 8

---

## 1. Problem Statement

When a user clicks "Start Training" in the `TrainingControlPanel`, the UI enters a dead spinner state — no progress visible, no status feedback — for the entire duration of asset discovery. On a large photo library (10,000+ images), this blocking period can last minutes.

The symptom looks like the session failed to start, but the real cause is that the HTTP response is not returned until all synchronous work completes.

**What the user experiences:**
1. Clicks "Start"
2. UI disables the button and shows a spinner
3. Nothing happens for 10–1000+ seconds
4. Eventually the spinner stops and training progress appears at whatever percentage the worker has reached

**What the logs show during the delay:**

```
INFO  training_service: Discovering assets for session 42
INFO  asset_discovery: Scanning directory: /photos/family/2023
INFO  asset_discovery: Scanning directory: /photos/family/2024
INFO  asset_discovery: Scanning directory: /photos/travel
INFO  asset_discovery: Total assets discovered for session 42: 8741
INFO  training_service: Created 8741 training jobs for session 42
INFO  training_service: Enqueued training session 42 as RQ job abc-123 with 8741 assets
# ^^^ Only at this point does the HTTP 200 return to the frontend
```

The frontend `onStatusChange()` handler fires only after the HTTP response arrives, and polling only begins when the status transitions to `running`. By that point, the RQ worker may already be processing images with no visible progress for the user during the entire discovery phase.

---

## 2. Root Cause Analysis

### 2.1 The Execution Path

The blocking chain begins in the FastAPI route handler and does not release the HTTP connection until all discovery and job creation work is done:

```
User clicks Start
  → POST /api/v1/training/sessions/{id}/start
    → routes/training.py: start_training()
      → service.start_training(db, session_id)          [TrainingService]
        → self.enqueue_training(db, session_id)
          ① → discovery_service.discover_assets(db, session_id)
                → for each subdir: _scan_directory()
                    → path.glob("**/*")                 [BLOCKS event loop]
                    → for each file: ensure_asset_exists()
                        → SELECT from image_assets       [1 round-trip/file]
                        → db.flush() on new assets
          ② → self.create_training_jobs(db, session_id, asset_ids)
                → for each asset missing a hash:
                    → compute_perceptual_hash(asset.path) [CPU: PIL resize+dHash]
                → db.commit()
                → for each asset: db.refresh(asset)
                → bulk INSERT TrainingJob rows
          ③ → session.status = RUNNING
          ④ → db.commit()
          ⑤ → queue.enqueue(train_session, session_id)  [Redis LPUSH]
        ← returns rq_job_id
      ← returns updated session (status: "running")
  ← HTTP 200 {"status": "running"}
Frontend: onStatusChange() fires → polling starts
```

Steps ① and ② are the problem. They are entirely synchronous from the HTTP response's perspective — the client waits for all of it.

### 2.2 The Synchronous Glob Blocks the Asyncio Event Loop

`AssetDiscoveryService._scan_directory()` at `src/image_search_service/services/asset_discovery.py` line 103 uses:

```python
for file_path in path.glob(pattern):
```

`Path.glob("**/*")` is a synchronous blocking generator. It calls OS `readdir()` syscalls on the main asyncio event loop thread. While this generator is running:

- FastAPI **cannot serve any other requests** — every other endpoint in the service is frozen
- Health checks return timeouts
- Concurrent users get connection refused or request timeouts

This is not an academic concern. On a network-mounted filesystem (NAS, NFS, SMB), a recursive glob over 10,000 files can block for 30+ seconds.

### 2.3 The DB Round-Trip Amplification

`ensure_asset_exists()` at `asset_discovery.py` line 117 executes one `SELECT` per file path:

```python
query = select(ImageAsset).where(ImageAsset.path == path)
result = await db.execute(query)
existing_asset = result.scalar_one_or_none()
```

For N files, this is N sequential round-trips to PostgreSQL. At ~1ms each on a local connection, N=10,000 files takes ~10 seconds in DB queries alone.

### 2.4 The CPU-Bound Hash Computation Holds the Event Loop

`create_training_jobs()` at `training_service.py` lines 416–425 runs perceptual hash computation inline on the event loop:

```python
for asset in assets:
    if asset.perceptual_hash is None:
        asset.perceptual_hash = compute_perceptual_hash(asset.path)
```

`compute_perceptual_hash()` in `services/perceptual_hash.py` opens the image with PIL, converts to grayscale, and resizes to 9×8. This is CPU-bound and runs on the asyncio event loop thread. Each call takes 10–100ms depending on image size and disk speed. For 10,000 images with no existing hashes, this is 100–1,000 seconds of CPU work blocking the event loop.

### 2.5 The Status Transition Is the Last Thing That Happens

Because `session.status = SessionStatus.RUNNING.value` is set at line 353 of `training_service.py`, immediately before `db.commit()` and the RQ enqueue call, the session status stays `pending` for the entire duration of steps ① and ②. The frontend polls `GET /api/v1/training/sessions/{id}` and receives `{"status": "pending"}` for the whole duration — indistinguishable from "hasn't started yet."

---

## 3. Scale of Impact

The following estimates assume a 10,000-image library on a local SSD. Network-attached storage multiplies each figure by 2–10x.

| Operation | Per-image cost | For 10,000 images | Notes |
|---|---|---|---|
| `path.glob("**/*")` | — | 1–30s | Blocks event loop; NAS can be 10x worse |
| `SELECT` in `ensure_asset_exists()` | ~1ms | ~10s | Sequential, not batched |
| `compute_perceptual_hash()` (PIL dHash) | 10–100ms | 100–1,000s | CPU-bound, on event loop |
| `TrainingJob` INSERT (bulk) | negligible | <1s | Already somewhat batched |
| Redis LPUSH (enqueue) | <1ms | <1ms | Instantaneous |
| HTTP response to frontend | — | Only after ALL above | |

**Worst-case end-to-end delay before frontend polling begins**: ~17 minutes for 10,000 images on a NAS with no existing hashes.

**Service degradation scope**: While `path.glob()` and `compute_perceptual_hash()` run, all other FastAPI endpoints are unresponsive. This affects every user and every feature of the application — search, thumbnail loading, face recognition — not just the user who clicked Start Training.

---

## 4. The Existing Correct Pattern (Face Detection Sessions)

The face detection feature in `src/image_search_service/api/routes/face_sessions.py` (lines 92–129) demonstrates the correct architectural pattern:

```python
@router.post("", response_model=FaceDetectionSessionResponse, status_code=201)
async def create_session(
    request: CreateFaceDetectionSessionRequest,
    db: AsyncSession = Depends(get_db),
) -> FaceDetectionSessionResponse:
    """Create and start a new face detection session."""
    # Step 1: Create minimal DB record (milliseconds)
    session = FaceDetectionSession(
        training_session_id=request.training_session_id,
        status=DBFaceDetectionSessionStatus.PENDING.value,
        ...
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)

    # Step 2: Enqueue RQ background job (Redis LPUSH — milliseconds)
    queue = get_queue(QUEUE_HIGH)
    job = queue.enqueue(
        detect_faces_for_session_job,
        session_id=str(session.id),
        job_timeout="24h",
    )
    session.job_id = job.id
    await db.commit()

    # Step 3: Return immediately with PENDING status
    return _session_to_response(session)
```

Total HTTP handler time: <50ms regardless of how many assets exist.
All discovery, processing, and heavy work: happens in the RQ worker subprocess.

The training session start handler must be refactored to match this pattern.

---

## 5. Devil's Advocate Analysis

The following challenges each aspect of the proposed solution. Each must be addressed before implementation.

### Challenge 1: "Just move discovery to the worker — what could go wrong with progress display?"

**The concern**: If `enqueue_training()` returns immediately before discovery, the session's `total_images` will be 0. The frontend progress bar will show 0/0, which renders as 0% or NaN%. Any component doing `processed / total * 100` will divide by zero or show a misleading "100% complete" for 0 images.

The progress polling endpoint (`GET /api/v1/training/sessions/{id}/progress`) likely returns a percentage based on `processed_images / total_images`. With `total_images = 0`, this is undefined.

**Analysis of the concern**: This is a real problem. The UI already assumes `total_images` is populated when status is `running`. A naive migration that simply moves discovery to the worker and jumps straight from `pending` to `running` would produce broken progress displays during the discovery phase.

**Resolution**: Introduce a `DISCOVERING` intermediate state between `PENDING` and `RUNNING`. This state communicates a specific, meaningful phase:

- `pending` — session created, not started
- `discovering` — worker is scanning directories, total_images not yet known
- `running` — discovery complete, total_images populated, training in progress
- `paused` / `completed` / `failed` — as before

The frontend progress endpoint must return a special response for `discovering` status — an indeterminate progress indicator rather than a percentage. The UI `SessionDetailView.svelte` must handle this state with "Discovering assets..." and an indeterminate progress bar (not a percentage bar).

This is a complete resolution. The intermediate state makes the UI correct and provides better UX than the current design.

---

### Challenge 2: "What if the RQ worker crashes during discovery?"

**The concern**: If the worker process crashes mid-scan — OOM kill, unhandled exception, machine restart — the session is left in `discovering` state with partially-created `ImageAsset` and `TrainingJob` records. A subsequent restart attempt has no clear recovery path: does it re-run discovery from scratch (risk of duplicates) or try to resume from where it left off (complex checkpoint logic)?

**Analysis of the concern**: This concern is valid for any stateful worker. However, the current design has the same problem: if the worker crashes during hash computation or batch processing, the session is in `running` state with partial jobs.

The discovery phase crash scenario is actually easier to recover from than a mid-training crash because:
1. `ensure_asset_exists()` is already effectively idempotent — it does a `SELECT` before `INSERT`. Running it again for the same file path returns the existing record.
2. `create_training_jobs()` at line 396–408 already checks for existing jobs (`existing_asset_ids`) and skips assets that already have jobs. The recovery path is re-run discovery from scratch — any assets already processed are no-ops.

**Resolution**: The worker's `train_session()` function should use `INSERT ... ON CONFLICT DO NOTHING` semantics (already partially implemented via the existing job existence check). On restart from `discovering` state, the worker re-runs discovery. Files already inserted return quickly (SELECT finds existing record). Jobs already created are skipped. This is idempotent by construction.

For belt-and-suspenders protection: the RQ job `timeout="1h"` already handles hung workers. A watchdog can transition timed-out `discovering` sessions to `failed` for user-initiated retry.

---

### Challenge 3: "Why not use FastAPI BackgroundTasks instead of moving to RQ?"

**The concern**: FastAPI has a built-in `BackgroundTasks` mechanism. Could we use `background_tasks.add_task(enqueue_training, db, session_id)` to defer the work without adding RQ complexity?

**Analysis**: FastAPI `BackgroundTasks` runs in the same uvicorn process, on the same asyncio event loop, after the HTTP response is sent. This addresses the HTTP response delay problem but does not solve the event loop blocking problem.

The specific issues that remain:
1. `path.glob("**/*")` is a synchronous blocking call. When `BackgroundTasks` runs it, it blocks the event loop exactly as before — just after the HTTP response instead of before. All other requests are still frozen.
2. `compute_perceptual_hash()` is CPU-bound. Python's GIL means CPU-bound work in an asyncio background task still blocks other coroutines from running.
3. `BackgroundTasks` run in the same process — if uvicorn is restarted or the worker process is killed, background tasks are lost with no recovery mechanism.
4. There is no job status tracking, retry logic, or priority queuing available for `BackgroundTasks`.

**Resolution**: RQ workers run in separate subprocesses with their own Python interpreter and their own GIL. CPU-bound work in an RQ worker has zero impact on the FastAPI event loop. This is the architecturally correct choice for mixed CPU/IO-bound workloads. The existing `training_jobs.py` module already runs in this context (`train_session()` is already an RQ job) — the fix is simply moving discovery into that existing job.

---

### Challenge 4: "What about the frontend optimistic update approach?"

**The concern**: Could the frontend skip waiting for the HTTP response and immediately show "running" status before the API responds? The `handleStart()` function in `TrainingControlPanel.svelte` currently awaits the response before calling `onStatusChange()`. If it fires `onStatusChange()` optimistically (before await completes), polling starts immediately and the spinner goes away sooner.

**Pros of this approach**:
- Zero backend changes required
- Ships in hours, not days
- User sees immediate visual feedback

**Cons of this approach**:
- Masks the architectural problem entirely. The API process is still blocked, still unable to serve other requests.
- The `path.glob()` call still freezes the event loop, blocking search requests, thumbnail loads, face recognition, and health checks for all users simultaneously.
- If the start request fails (e.g., no subdirectories selected, invalid session state), the frontend shows "running" optimistically but the API returns an error — the UI is now inconsistent with the DB state.
- Polling would begin while the session is still `pending` in the database, producing confusing results.

**Verdict**: This is a valid short-term UI improvement that can ship immediately to reduce user confusion, but it does not solve the backend architectural problem. Other users' requests are still blocked during the discovery phase. The frontend fix should be applied as a first patch, and the backend fix must follow. Do not treat the frontend fix as sufficient.

---

### Challenge 5: "Should we batch the DB operations instead of refactoring?"

**The concern**: The N round-trips in `ensure_asset_exists()` are a significant contributor to the delay. Could we fix just that — replacing the per-file `SELECT`/`INSERT` loop with a bulk `SELECT ... WHERE path IN (...)` followed by a bulk `INSERT ... ON CONFLICT DO NOTHING`? This would reduce the 10,000 round-trips to ~10 batched queries and bring the DB time from ~10s to ~0.1s.

**Analysis**: This is a genuine optimization worth doing, but it addresses only one of three contributing problems:

| Problem | Addressed by batch DB? | Still blocks event loop? |
|---|---|---|
| `path.glob()` filesystem scan | No | Yes |
| `SELECT` per file (10,000 round-trips) | **Yes** | No (still async) |
| `compute_perceptual_hash()` CPU work | No | Yes |

Even after batching, a 10,000-image scan over a NAS with 30s of `glob()` time plus 1,000s of hash computation is still fundamentally broken. Batching the DB operations alone does not make the HTTP handler return in <100ms.

**Verdict**: DB batching is a high-value follow-up optimization (Phase 2) and should be implemented after the primary architectural fix. It is not a substitute for moving discovery to the worker.

---

### Challenge 6: "What about asyncio event loop blocking from glob() in the API process?"

**The concern**: Even after moving discovery to the RQ worker, the `_scan_directory()` method and `count_images_in_directory()` in `asset_discovery.py` still use synchronous `path.glob()`. If any code path in the API process calls these methods (e.g., a directory preview endpoint, or if `AssetDiscoveryService` is reused elsewhere), the event loop blocking problem would reappear.

The concern is also forward-looking: a future developer might call `_scan_directory()` from a new async API endpoint without realizing it blocks the event loop.

**Analysis**: In the proposed solution, `_scan_directory()` would only be called from the RQ worker subprocess (synchronous context), where event loop blocking is irrelevant — workers use synchronous DB sessions (`get_sync_session()`) and have no asyncio event loop.

However, the concern about future misuse is valid. `directory_service.py` at line 77 already has a similar synchronous glob call:

```python
count += len(list(dir_path.glob(f"*.{ext.lower()}")))
```

**Resolution**: Wrap the `path.glob()` calls in `asyncio.to_thread()` for any method that remains callable from async context. This offloads the blocking syscall to a thread pool without blocking the event loop:

```python
import asyncio

async def count_images_in_directory(self, directory: str, recursive: bool = True) -> int:
    def _sync_count() -> int:
        count = 0
        for file_path in path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.extensions_set:
                count += 1
        return count

    return await asyncio.to_thread(_sync_count)
```

This is a defensive fix that protects against future misuse. It should be applied in Phase 2 to `count_images_in_directory()` and `_scan_directory()` when called from async contexts.

---

## 6. Recommended Solution

### Phase 1: Move Discovery to the RQ Worker (Primary Fix)

This eliminates the HTTP blocking entirely. The HTTP handler returns in <100ms regardless of library size.

#### 6.1 Add `DISCOVERING` State to `SessionStatus`

**File**: `src/image_search_service/db/models.py`

Add the new state between `PENDING` and `RUNNING`:

```python
class SessionStatus(str, Enum):
    """Status enum for training sessions."""

    PENDING = "pending"
    DISCOVERING = "discovering"   # NEW: worker is scanning directories
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
```

A database migration is required to widen the `status` column's CHECK constraint (or enum type, depending on the column definition).

#### 6.2 Rewrite `enqueue_training()` to Return Immediately

**File**: `src/image_search_service/services/training_service.py`

Replace the current blocking implementation (lines 309–368) with:

```python
async def enqueue_training(self, db: AsyncSession, session_id: int) -> str:
    """Enqueue training session for background processing.

    This method returns immediately. Asset discovery, hash computation,
    and job creation all happen inside the RQ worker subprocess.

    State transition: pending → discovering (set here)
    Worker transitions: discovering → running → completed/failed

    Args:
        db: Database session
        session_id: Training session ID

    Returns:
        RQ job ID

    Raises:
        ValueError: If session not found or has no selected subdirectories
    """
    session = await self.get_session(db, session_id)
    if not session:
        raise ValueError(f"Training session {session_id} not found")

    # Validate that subdirectories are selected before enqueuing
    # (cheap check — avoids enqueueing a job that will immediately fail)
    subdirs_query = (
        select(TrainingSubdirectory)
        .where(TrainingSubdirectory.session_id == session_id)
        .where(TrainingSubdirectory.selected == True)  # noqa: E712
    )
    result = await db.execute(subdirs_query)
    selected_subdirs = list(result.scalars().all())

    if not selected_subdirs:
        raise ValueError(f"No subdirectories selected for session {session_id}")

    # Transition immediately to DISCOVERING — cheap, returns fast
    session.status = SessionStatus.DISCOVERING.value
    await db.commit()

    # Enqueue RQ job — Redis LPUSH, <1ms
    from image_search_service.queue.training_jobs import train_session

    queue = get_queue(QUEUE_HIGH)
    rq_job = queue.enqueue(train_session, session_id, job_timeout="1h")

    logger.info(
        f"Enqueued training session {session_id} as RQ job {rq_job.id} "
        f"(discovery + training will happen in worker)"
    )

    return str(rq_job.id)
```

**HTTP handler time after this change**: <100ms for any library size.

#### 6.3 Move Discovery into the RQ Worker

**File**: `src/image_search_service/queue/training_jobs.py`

The existing `train_session()` function (line 129) queries pre-populated `TrainingJob` rows and processes them. It must be extended to run discovery first when called from the `DISCOVERING` state.

The worker already uses synchronous DB operations (`get_sync_session()`), so `path.glob()` blocking is a non-issue in the subprocess context.

```python
def train_session(session_id: int) -> dict[str, object]:
    """Main training job: discovery + batch processing.

    State transitions managed by this function:
      discovering → running (after asset discovery + job creation)
      running → completed (after all jobs processed)
      any → failed (on unhandled exception)

    This function is idempotent: safe to retry on crash.
    """
    logger.info(f"Starting training session {session_id}")
    start_time = datetime.now(UTC)

    db_session = get_sync_session()
    tracker = ProgressTracker(session_id)

    try:
        session = get_session_by_id_sync(db_session, session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            return {"status": "error", "message": "Session not found"}

        # Phase 1: Asset discovery (only needed from DISCOVERING state)
        if session.status == SessionStatus.DISCOVERING.value:
            logger.info(f"Session {session_id}: starting asset discovery phase")

            # Run sync discovery (safe in worker subprocess — no event loop)
            _discover_and_create_jobs_sync(db_session, session_id)

            # Transition to RUNNING with total_images populated
            session = get_session_by_id_sync(db_session, session_id)
            if not session:
                return {"status": "error", "message": "Session not found after discovery"}

            session.status = SessionStatus.RUNNING.value
            session.started_at = datetime.now(UTC)
            db_session.commit()

            logger.info(
                f"Session {session_id}: discovery complete, "
                f"{session.total_images} assets, transitioning to RUNNING"
            )

        # Phase 2: Process pending TrainingJob rows (existing logic)
        query = (
            select(TrainingJob)
            .where(TrainingJob.session_id == session_id)
            .where(TrainingJob.status == JobStatus.PENDING.value)
        )
        result = db_session.execute(query)
        pending_jobs = list(result.scalars().all())

        if not pending_jobs:
            logger.warning(f"No pending jobs found for session {session_id}")
            return {"status": "completed", "session_id": session_id, "processed": 0, "failed": 0}

        # ... (existing batch processing loop unchanged) ...
```

The new helper `_discover_and_create_jobs_sync()` wraps the existing `AssetDiscoveryService` and `create_training_jobs` logic in synchronous form, using `get_sync_session()` for DB operations. This keeps the discovery logic in one place and avoids duplication.

#### 6.4 Update the `start_training()` State Machine

**File**: `src/image_search_service/services/training_service.py`

The `start_training()` method's `valid_states` list (line 591) must be updated to reflect that `DISCOVERING` is now a valid state to detect (worker already running):

```python
valid_states = [
    SessionStatus.PENDING.value,
    SessionStatus.PAUSED.value,
    SessionStatus.FAILED.value,
]

# Also handle: if already DISCOVERING or RUNNING, return current state
already_active = [
    SessionStatus.DISCOVERING.value,
    SessionStatus.RUNNING.value,
]
if session.status in already_active:
    logger.warning(f"Session {session_id} already active (status: {session.status})")
    return session
```

#### 6.5 Update Pydantic Schemas

**File**: `src/image_search_service/api/schemas.py`

Add `discovering` to any schema `SessionStatus` literal or enum that mirrors the DB model. The progress endpoint response must handle the `discovering` state:

```python
class SessionProgressResponse(BaseModel):
    session_id: int
    status: str  # "pending" | "discovering" | "running" | ...
    total_images: int
    processed_images: int
    failed_images: int
    skipped_images: int
    percentage: float | None  # None when status is "discovering"
    is_indeterminate: bool    # True when status is "discovering"
    message: str | None
```

The `percentage` field must be `None` (not 0.0) during `discovering` to prevent division-by-zero and incorrect "0% complete" display.

#### 6.6 Database Migration

A migration is required to allow the new `discovering` value in the `status` column:

```bash
make makemigrations
# Enter message: "add discovering status to training sessions"
```

Review the generated migration to ensure:
- The column constraint or enum type is widened to include `"discovering"`
- The migration is idempotent (uses `if_exists=True` patterns where applicable)
- Existing `pending`/`running` sessions are unaffected

#### 6.7 Frontend Changes

**File**: `image-search-ui/src/routes/training/SessionDetailView.svelte`

Handle the new `discovering` status:

```typescript
// In the status display logic:
if (session.status === 'discovering') {
    // Show indeterminate progress bar
    // Show "Discovering assets..." message
    // Do NOT show a percentage
    // Continue polling (keep polling interval active)
}
```

**File**: `image-search-ui/src/routes/training/TrainingControlPanel.svelte`

Start polling immediately after the API responds (before status is `running`):

```typescript
async function handleStart() {
    const response = await api.post(`/training/sessions/${sessionId}/start`);
    // Fire onStatusChange immediately — status is now "discovering"
    // This starts polling even during the discovery phase
    onStatusChange(response.status);
}
```

The frontend type regeneration step (`npm run gen:api`) must be run after the backend schema is updated.

---

### Phase 2: Optimizations (Follow-up Work)

These are valuable improvements but should not block the Phase 1 fix.

#### 6.8 Batch DB Operations in `ensure_asset_exists()`

Replace the per-file `SELECT` + `flush()` pattern with:

1. Collect all file paths from `path.glob()` first
2. `SELECT * FROM image_assets WHERE path = ANY(:paths)` — one round-trip for N files
3. `INSERT INTO image_assets ... ON CONFLICT (path) DO NOTHING` — one round-trip for new assets

This reduces N DB round-trips to 2, cutting DB time from ~10s to ~20ms for 10,000 files.

#### 6.9 Fix Asyncio Event Loop Blocking for Remaining Callers

Wrap `path.glob()` calls in `asyncio.to_thread()` in `count_images_in_directory()` and any other async callers in `asset_discovery.py` and `directory_service.py`:

```python
async def count_images_in_directory(self, directory: str, recursive: bool = True) -> int:
    def _sync_count() -> int:
        count = 0
        for file_path in Path(directory).glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.extensions_set:
                count += 1
        return count

    return await asyncio.to_thread(_sync_count)
```

#### 6.10 Add Discovery Progress Reporting

During the discovery phase, the worker can periodically update a counter in the DB or Redis:

```python
# In _discover_and_create_jobs_sync(), every 500 files:
if discovered_count % 500 == 0:
    # UPDATE training_sessions SET discovered_images = :count WHERE id = :session_id
    update_discovered_count_sync(db_session, session_id, discovered_count)
```

The frontend can display "Discovering assets... 2,341 found so far" during the `discovering` phase.

---

## 7. Files Affected

### Backend — Required Changes (Phase 1)

| File | Change |
|---|---|
| `src/image_search_service/db/models.py` | Add `DISCOVERING = "discovering"` to `SessionStatus` enum |
| `src/image_search_service/services/training_service.py` | Rewrite `enqueue_training()` to return immediately; update `start_training()` valid states |
| `src/image_search_service/queue/training_jobs.py` | Add discovery phase to `train_session()`; add `_discover_and_create_jobs_sync()` helper |
| `src/image_search_service/api/schemas.py` | Add `discovering` to status literals; add `is_indeterminate` field to progress response |
| `db/migrations/versions/*.py` | New migration: add `discovering` to session status column |
| `tests/unit/test_training_service.py` | Update tests for new `enqueue_training()` behavior |
| `tests/api/test_training.py` | Add tests for `discovering` status transitions |

### Frontend — Required Changes (Phase 1)

| File | Change |
|---|---|
| `src/lib/api/generated.ts` | Regenerated automatically via `npm run gen:api` |
| `src/routes/training/SessionDetailView.svelte` | Handle `discovering` status with indeterminate progress |
| `src/routes/training/TrainingControlPanel.svelte` | Fire `onStatusChange()` immediately after API responds |

### Backend — Optional Improvements (Phase 2)

| File | Change |
|---|---|
| `src/image_search_service/services/asset_discovery.py` | Batch `ensure_asset_exists()`; wrap `glob()` in `asyncio.to_thread()` |
| `src/image_search_service/services/directory_service.py` | Wrap `glob()` calls in `asyncio.to_thread()` |
| `src/image_search_service/queue/training_jobs.py` | Add periodic discovered-count updates during discovery |
| `src/image_search_service/api/schemas.py` | Add `discovered_images` field to progress response |

---

## 8. Risk Assessment

### Risk 1: Worker restarts during discovery leave orphaned records

**Probability**: Low (worker crashes are rare; RQ has job timeout handling)
**Impact**: Medium (session stuck in `discovering` state)
**Mitigation**: The existing `_check_existing_jobs` logic in `create_training_jobs()` is already idempotent. Add a DB cleanup job that transitions sessions stuck in `discovering` for >2x the expected discovery time to `failed`, allowing user retry.

### Risk 2: `total_images = 0` during `discovering` causes division-by-zero in progress queries

**Probability**: High (any call to the progress endpoint before discovery completes)
**Impact**: Medium (NaN/Infinity in frontend, possibly 500 error if not handled)
**Mitigation**: The schema change (Phase 1, section 6.5) makes `percentage` a nullable `float | None` and adds `is_indeterminate: bool`. The frontend checks `is_indeterminate` before rendering a percentage bar. The backend progress computation must guard: `percentage = (processed / total * 100) if total > 0 else None`.

### Risk 3: Frontend type mismatch after backend schema change

**Probability**: Medium (developer forgets to run `npm run gen:api`)
**Impact**: Low (TypeScript compile error catches it before runtime)
**Mitigation**: The frontend enforces types at compile time. Running `npm run gen:api` after the backend change produces updated `generated.ts`. TypeScript will surface any component that doesn't handle `discovering` or the new `is_indeterminate` field.

### Risk 4: Paused/failed session restart skips discovery phase

**Probability**: Low (the `PAUSED`/`FAILED` → `RUNNING` path in `start_training()` already skips `enqueue_training()` and re-enqueues `train_session()` directly)
**Impact**: None (the existing jobs are already created; the worker only re-runs discovery when status is `DISCOVERING`)
**Mitigation**: The `train_session()` function checks `session.status` at entry. Only `DISCOVERING` triggers the discovery phase. A resume from `PAUSED` or `FAILED` sets status directly to `RUNNING` (existing behavior), and the worker finds existing `PENDING` jobs and processes them.

### Risk 5: Migration adds `discovering` but existing sessions use `pending`

**Probability**: None (additive change only)
**Impact**: None
**Mitigation**: The `DISCOVERING` value is only written by the new `enqueue_training()` code path. Existing sessions with `pending`, `running`, etc. are unaffected. The migration widens the allowed value set without changing existing data.

---

## Summary

The training session progress reporting delay is caused by a single architectural mistake: running multi-minute synchronous work (filesystem scanning, per-file DB queries, CPU-bound hash computation) inside the HTTP request handler, blocking both the HTTP response and the asyncio event loop.

The fix follows the pattern already established by the face detection session feature: return immediately with a new `discovering` state, do all heavy work in the RQ worker subprocess. The worker transitions through `discovering` → `running` → `completed`, giving the frontend meaningful state information at each phase.

Phase 1 is a focused, low-risk change to `enqueue_training()`, `train_session()`, and the status enum. It eliminates the HTTP blocking entirely and makes the service responsive to all users during discovery. Phase 2 optimizations (DB batching, asyncio-safe glob) provide defense-in-depth and improve throughput.
