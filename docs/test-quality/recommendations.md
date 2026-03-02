# Recommendations

Prioritized action items ordered by engineering risk reduction per effort invested.

## Priority 1: Close Critical Coverage Gaps

### R1. Write tests for `queue/jobs.py`

**Effort**: 2-3 hours
**Risk reduction**: HIGH

`index_asset()` and `update_asset_person_ids_job()` are the core data pipeline. Use the `sync_db_session` fixture pattern from `test_face_jobs_coverage.py`:

1. Create `tests/unit/queue/test_jobs.py`
2. Create ImageAsset records in the test database
3. Mock only `get_embedding_service()` (return a fixed vector) and `upsert_vector()` / `update_vector_payload()` (capture arguments)
4. Verify:
   - Asset not found returns `{"status": "error"}`
   - Successful indexing calls `upsert_vector` with correct `asset_id`, `vector`, and `payload`
   - `indexed_at` is set on the asset after commit
   - Embedding failure is caught and logged
   - `update_asset_person_ids_job` correctly queries FaceInstance records and builds `person_ids` list
   - The deleted-asset guard (commit `92f45d8`) returns `{"status": "skipped"}`

### R2. Write tests for `services/config_service.py`

**Effort**: 2-3 hours
**Risk reduction**: HIGH

Use the `db_session` (async) fixture:

1. Create `tests/unit/services/test_config_service.py`
2. Test `ConfigService.get()` with and without defaults
3. Test `ConfigService.set()` persists values
4. Test type coercion: `get_int`, `get_float`, `get_bool`
5. Test `SyncConfigService` wraps async correctly (using `sync_db_session`)
6. Test missing key returns default, not exception

### R3. Write tests for `services/file_watcher.py`

**Effort**: 3-4 hours
**Risk reduction**: HIGH

Use `tmp_path` and the `monkeypatch` fixture:

1. Create `tests/unit/services/test_file_watcher.py`
2. Test `ImageFileHandler.on_created()` with supported extensions (`.jpg`, `.png`, `.heic`)
3. Test unsupported extensions are ignored
4. Test debouncing: rapid file modifications produce only one job
5. Mock the queue enqueue function and verify it is called with correct arguments
6. Test error handling for inaccessible files

## Priority 2: Fix Structural Test Problems

### R4. Replace embedding mock tests with real service tests

**Effort**: 2-3 hours
**Risk reduction**: MEDIUM

Rewrite `tests/unit/test_embedding_service.py`:

1. Import the real `EmbeddingService` class
2. Mock only the underlying OpenCLIP model call (`open_clip.create_model_and_transforms`)
3. Verify that `embed_text()`:
   - Calls the tokenizer with the input text
   - Passes tokens through the model
   - Normalizes the output vector
   - Returns a list of floats with the correct dimension
4. Verify that `embed_image()`:
   - Opens the image file
   - Applies preprocessing transforms
   - Passes the tensor through the model
   - Normalizes and returns
5. Keep the existing mock tests but rename the file to `test_mock_embedding.py` and mark it as test infrastructure validation, not production code testing.

### R5. Refactor `test_main.py` to reduce patch stacking

**Effort**: 1-2 hours
**Risk reduction**: MEDIUM (reduces maintenance cost)

Option A (quick fix): Convert all `@patch` decorators to `monkeypatch.setattr()` within the test body. This eliminates the reversed-parameter-order risk.

Option B (better fix): Create a fixture that sets up the lifespan dependencies:

```python
@pytest.fixture
def lifespan_mocks(monkeypatch):
    mocks = {}
    mocks["validate"] = MagicMock(return_value=[])
    mocks["settings"] = MagicMock()
    mocks["settings"].qdrant_strict_startup = True
    monkeypatch.setattr("image_search_service.vector.qdrant.validate_qdrant_collections", mocks["validate"])
    monkeypatch.setattr("image_search_service.core.config.get_settings", lambda: mocks["settings"])
    # ... etc
    return mocks
```

### R6. Replace PersonService internal method mocks with real DB tests

**Effort**: 3-4 hours
**Risk reduction**: MEDIUM

1. Use `db_session` to create Person records with different types (identified, unidentified clusters, noise)
2. Create FaceInstance records associated with persons
3. Call `get_all_people()` directly
4. Verify sorting, filtering, and count accuracy against real data
5. This catches SQL query bugs that the current mock-based approach cannot detect

### R7. Extend `@pytest.mark.parametrize` to repetitive tests

**Effort**: 2-3 hours
**Risk reduction**: LOW (improves maintainability)

The pattern is already used effectively in `tests/unit/test_temporal_service.py` (5 uses, 19+ cases in `TestClassifyAgeEra`) and `tests/core/test_device.py` (5 uses). Use these as reference examples and apply the same approach to:

```python
# test_validate_qdrant_collections.py
@pytest.mark.parametrize("existing_collections,expected_missing", [
    (["image_assets", "faces", "person_centroids"], []),
    (["image_assets"], ["faces", "person_centroids"]),
    ([], ["image_assets", "faces", "person_centroids"]),
])
def test_validate_collections(existing_collections, expected_missing, ...):
    ...

# test_embedding_router.py
@pytest.mark.parametrize("use_siglip,rollout_pct,user_id,expected_service", [
    (False, 0, None, "clip"),
    (True, 0, None, "siglip"),
    (False, 50, 23, "siglip"),
    (False, 50, 76, "clip"),
    (False, 100, None, "siglip"),
])
def test_router_selection(use_siglip, rollout_pct, user_id, expected_service, ...):
    ...
```

## Priority 3: Improve Test Quality

### R8. Fix `os.environ` mutation in safety guard tests

**Effort**: 30 minutes
**Risk reduction**: LOW (prevents potential test pollution)

In `tests/unit/test_qdrant_safety.py`, replace lines 63 and 87:

```python
# Before (fragile):
os.environ["QDRANT_FACE_COLLECTION"] = "faces"
# ... test ...
os.environ["QDRANT_FACE_COLLECTION"] = "test_faces"  # Manual cleanup

# After (safe):
monkeypatch.setenv("QDRANT_FACE_COLLECTION", "faces")
# ... test ...
# Cleanup is automatic
```

### R9. Replace `np.random.seed()` with local RNG

**Effort**: 1-2 hours (19 instances across 4 files, plus determinism verification)
**Risk reduction**: LOW-MEDIUM (prevents flaky tests with random ordering; 19 instances makes this a systemic issue)

Affected files:
- `tests/faces/test_trainer.py`: 11 instances
- `tests/faces/test_clusterer.py`: 4 instances
- `tests/unit/services/test_face_clustering_service.py`: 1 instance (line 106)
- `tests/unit/test_cluster_confidence.py`: 3 instances

```python
# Before:
np.random.seed(42)
base_vector = np.random.rand(512)

# After:
rng = np.random.default_rng(42)
base_vector = rng.random(512)
```

After conversion, verify determinism by running each affected file in isolation and in reversed order (`pytest --reversed`).

### R10. Mark race condition tests with `xfail`

**Effort**: 15 minutes
**Risk reduction**: LOW (prevents confusion when bugs are fixed)

```python
@pytest.mark.xfail(
    reason="Known race condition: double-deprecation without row locking. "
           "Remove xfail when SELECT FOR UPDATE is implemented.",
    strict=True,  # Fail if the test unexpectedly passes (bug was fixed)
)
async def test_compute_centroids_when_concurrent_same_person_then_double_deprecation():
    ...
```

### R11. Add named constants for magic numbers

**Effort**: 1 hour
**Risk reduction**: LOW (improves readability)

Create `tests/constants.py`:

```python
CLIP_EMBEDDING_DIM = 768
SIGLIP_EMBEDDING_DIM = 768
FACE_EMBEDDING_DIM = 512
SHA256_HEX_LENGTH = 64
PERCEPTUAL_HASH_LENGTH = 16
```

Import these in test files instead of using bare integers.

## Priority 4: Long-Term Improvements

### R12. Extend existing integration tests with end-to-end workflow tests

**Effort**: 4-6 hours
**Risk reduction**: MEDIUM (catches cross-service bugs)

Integration test infrastructure already exists in `tests/integration/` with 6 files covering database constraints (`test_postgres_constraints.py`, `test_postgres_jsonb.py`), migrations (`test_postgres_migrations.py`, `test_temporal_migration.py`), worker lifecycle (`test_listener_worker.py`), and restart workflows (`test_restart_workflows.py`). The `tests/conftest_postgres.py` testcontainers setup provides a solid foundation.

Build on this infrastructure by adding `tests/integration/test_workflows.py`:

1. **Ingest-to-search**: Create an image, run `index_asset()`, verify the vector exists in Qdrant (in-memory), then run a text search query and verify the image appears in results.
2. **Face pipeline**: Create assets with face instances, run `cluster_dual_job()`, verify cluster assignments in the database.
3. **Training lifecycle**: Create a session, run `train_session()`, verify all jobs complete and evidence is created.

These tests use real in-memory databases and Qdrant but mock only the ML models.

### R13. Standardize mocking approach

**Effort**: 4-6 hours (can be done incrementally)
**Risk reduction**: LOW (maintenance improvement)

Adopt `monkeypatch.setattr()` as the standard approach. Convert `@patch` and `with patch()` usage in:

1. `tests/test_main.py` (4 tests, highest priority due to fragility)
2. `tests/unit/test_validate_qdrant_collections.py` (5 tests)
3. `tests/unit/test_embedding_router.py` (7 tests)

Do not convert tests that work well with their current approach (e.g., `test_face_jobs_coverage.py` uses `with patch()` consistently and readably).

### R14. Split large test files

**Effort**: 2-3 hours per file
**Risk reduction**: LOW (organizational improvement)

- `tests/api/test_search.py` (~50KB): Split by search type (text search, image search, combined search, filter tests)
- `tests/api/test_storage_routes.py` (~58KB): Split by feature (upload, download, status, cancel, folders)
- `tests/unit/queue/test_face_jobs_coverage.py` (1054 lines): Already well-organized by class; split each class into its own file

## Summary Table

| ID | Action | Effort | Risk Reduction | Priority |
|----|--------|--------|---------------|----------|
| R1 | Test queue/jobs.py | 2-3h | HIGH | P1 |
| R2 | Test config_service.py | 2-3h | HIGH | P1 |
| R3 | Test file_watcher.py | 3-4h | HIGH | P1 |
| R4 | Real embedding service tests | 2-3h | MEDIUM | P2 |
| R5 | Refactor test_main.py patching | 1-2h | MEDIUM | P2 |
| R6 | Real DB tests for PersonService | 3-4h | MEDIUM | P2 |
| R7 | Parametrize repetitive tests | 2-3h | LOW | P2 |
| R8 | Fix os.environ mutation | 30min | LOW | P3 |
| R9 | Replace np.random.seed() (19 instances) | 1-2h | LOW-MEDIUM | P3 |
| R10 | Mark race condition tests xfail | 15min | LOW | P3 |
| R11 | Named constants for magic numbers | 1h | LOW | P3 |
| R12 | Extend integration tests with workflow tests | 4-6h | MEDIUM | P4 |
| R13 | Standardize mocking approach | 4-6h | LOW | P4 |
| R14 | Split large test files | 6-9h | LOW | P4 |
