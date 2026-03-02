# Coverage Gaps

This document identifies source modules and behaviors that lack test coverage, ordered by engineering risk.

## Critical Gaps (HIGH risk)

### 1. `src/image_search_service/queue/jobs.py` -- No Direct Execution Tests

The core background job module containing `index_asset()` and `update_asset_person_ids_job()` has no dedicated test file that exercises job execution logic. Indirect references exist: `tests/api/test_ingest.py` references `index_asset` for enqueue verification, and `tests/unit/queue/test_enqueue_person_ids_update.py` references `update_asset_person_ids_job` for enqueue. However, these only verify that jobs are enqueued -- not that they execute correctly.

**What `index_asset()` does**: Loads an asset from PostgreSQL, calls `get_embedding_service().embed_image()`, upserts the vector to Qdrant, and updates `indexed_at` timestamp. This is the primary indexing pipeline.

**What is untested**:
- Asset not found in database (line 43-45)
- Embedding failure handling (line 50-52)
- Qdrant upsert with correct payload construction (lines 56-64)
- `indexed_at` timestamp update and commit (lines 67-68)
- The `int(asset_id)` conversion on line 40 (what happens with non-numeric IDs?)

**What `update_asset_person_ids_job()` does**: Queries FaceInstance records for an asset, collects distinct person_ids, and updates the Qdrant payload.

**What is untested**:
- The guard against deleted assets (lines 92-104) -- added in commit `92f45d8` but never tested
- Person ID collection query (lines 108-114)
- Qdrant payload update success and failure paths

**Risk**: These are the core data pipeline jobs. A regression here means images stop being searchable or person associations silently break.

### 2. `src/image_search_service/services/config_service.py` -- No Behavioral Tests

The database-backed configuration service provides runtime-configurable settings. It supports both async and sync interfaces (`ConfigService` and `SyncConfigService`). Some structural tests exist: `tests/unit/test_config_keys_sync.py` tests `ConfigService.DEFAULTS` dict structure, and `tests/unit/test_config_unknown_person.py` tests unknown person settings. However, the core behavioral surface is untested.

**What is untested**:
- `get()` reading values with defaults
- `set()` writing and persisting values
- Type coercion (`get_int`, `get_float`, `get_bool`)
- Cache behavior (if any)
- Error handling for invalid keys or database failures
- `SyncConfigService` wrapper used by background jobs

**Risk**: Configuration drives thresholds for face clustering, suggestion expiry, and other critical behaviors. Bugs here silently change system behavior.

### 3. `src/image_search_service/services/file_watcher.py` -- Zero Tests

The `ImageFileHandler` processes filesystem events (file created, modified, deleted) with debouncing and extension filtering.

**What is untested**:
- File creation event triggers indexing
- File modification event triggers re-indexing
- File deletion event triggers cleanup
- Debouncing (same file modified rapidly only triggers one job)
- Extension filtering (only `.jpg`, `.png`, etc. are processed)
- Error handling for inaccessible files

**Risk**: The file watcher is the primary ingest mechanism for automatic photo discovery. Bugs cause missed photos or duplicate processing.

### 4. `src/image_search_service/services/periodic_scanner.py` -- Zero Tests

The `PeriodicScannerService` runs async scan loops at configurable intervals.

**What is untested**:
- Scan loop lifecycle (start, stop, interval timing)
- Handling of scan errors (does the loop continue or crash?)
- Concurrent scan prevention
- Integration with asset discovery

**Risk**: If the scanner crashes silently, new photos stop being discovered until a manual restart.

## Significant Gaps (MEDIUM risk)

### 5. Real Embedding Service Contract Tests

As detailed in `weaknesses.md`, both `test_embedding_service.py` and `test_siglip_embedding_service.py` test mock objects. The real `EmbeddingService` class at `src/image_search_service/services/embedding.py` lacks tests for:

- Model lazy loading (verified for SigLIP but not CLIP)
- `embed_text()` output normalization
- `embed_image()` input validation (corrupt images, missing files)
- `embed_images_batch()` batching logic
- Singleton behavior of `get_embedding_service()`
- Device selection (CPU vs GPU vs MPS)

### 6. `src/image_search_service/services/training_service.py` -- Partial Coverage

The `TrainingService` has indirect coverage through route tests and race condition tests, but lacks dedicated unit tests for:

- Session creation with validation
- Subdirectory scanning and validation
- Resume logic (which assets to skip on resume)
- Hash deduplication during training
- The full start -> pause -> resume -> complete lifecycle with real DB state

### 7. `src/image_search_service/services/upload_service.py` -- Unknown Coverage

Storage upload orchestration. While `tests/unit/storage/test_upload_service.py` exists, the coverage depth is unknown relative to the service complexity.

### 8. `src/image_search_service/queue/training_jobs.py` -- train_batch Untested

While `test_training_jobs_coverage.py` tests `train_session` and `train_single_asset`, the `train_batch` function is always mocked:

```python
monkeypatch.setattr(
    "image_search_service.queue.training_jobs.train_batch",
    mock_train_batch,
)
```

`train_batch` handles GPU batch processing, memory cleanup, and error aggregation. It is never tested directly.

### 9. `src/image_search_service/services/evidence_service.py` -- Unknown Coverage

Evidence tracking for training audit trails. No dedicated test file found.

### 10. `src/image_search_service/services/directory_service.py` -- Unknown Coverage

Directory management for organizing photo libraries. No dedicated test file found.

## Lower Priority Gaps (LOW risk)

### 11. `src/image_search_service/queue/auto_detection_jobs.py`

Automatic face detection triggering after training completion. The module exists but has no dedicated tests.

### 12. `src/image_search_service/queue/hash_backfill_jobs.py`

Perceptual hash backfill for existing assets. While the perceptual hash algorithm is well-tested, the backfill job orchestration is not.

### 13. `src/image_search_service/queue/thumbnail_jobs.py`

Thumbnail generation jobs. No dedicated tests found.

### 14. `src/image_search_service/queue/storage_jobs.py`

Storage sync/upload background jobs. Covered indirectly through storage route tests but lacks unit-level coverage.

### 15. `src/image_search_service/services/watcher_manager.py`

Coordinates multiple file watchers. The `WatcherManager` is always mocked in `test_main.py`.

### 16. `src/image_search_service/services/asset_discovery.py` -- No Tests Found

Asset discovery logic for scanning directories and identifying new images. No dedicated test file exists, and no indirect coverage was identified.

### 17. `src/image_search_service/services/thumbnail_service.py` -- No Tests Found

Thumbnail generation service for creating preview images. No dedicated test file exists.

### 18. `src/image_search_service/queue/progress.py` -- No Tests Found

Progress tracking for background jobs (used by `ProgressTracker` references throughout queue tests). The tracker itself is always mocked.

### 19. `src/image_search_service/queue/worker.py` -- No Tests Found

Custom worker configuration and lifecycle management. Worker behavior is tested indirectly through `test_listener_worker.py`, but the worker module itself has no dedicated tests.

## Missing Test Categories

### End-to-End Feature Workflow Tests

Integration tests exist in `tests/integration/` with 6 files covering database constraints (`test_postgres_constraints.py`, `test_postgres_jsonb.py`), migrations (`test_postgres_migrations.py`, `test_temporal_migration.py`), worker lifecycle (`test_listener_worker.py`), and restart workflows (`test_restart_workflows.py`). These provide solid infrastructure-level integration coverage.

What is missing are end-to-end *feature workflow* tests that exercise cross-service pipelines:

- **Ingest -> Index -> Search**: No test verifies that ingesting an image makes it searchable
- **Detect Faces -> Cluster -> Assign**: Individual steps are tested, but the pipeline is not
- **Training -> Centroid Computation -> Suggestion Generation**: Each step is isolated

### Error Recovery Tests

The suite lacks systematic tests for service recovery after failures:

- What happens when Qdrant is down during indexing? Does the job retry?
- What happens when a training session's worker crashes mid-batch?
- What happens when the database connection drops during a long scan?

The race condition documentation tests identify these gaps but do not test recovery behavior.

### Configuration-Driven Behavior Tests

No tests verify that changing configuration values actually changes system behavior:

- Does changing `face_suggestion_expiry_days` affect which suggestions are expired?
- Does changing `centroid_min_faces` affect which persons get centroids?
- Does changing `siglip_rollout_percentage` correctly route requests?

The embedding router tests cover the rollout percentage, but this pattern is not applied elsewhere.
