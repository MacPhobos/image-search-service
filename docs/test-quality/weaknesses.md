# Test Suite Weaknesses

## 1. Embedding Tests Only Validate Mock Implementations

**Files**: `tests/unit/test_embedding_service.py`, `tests/unit/test_siglip_embedding_service.py`

Both of these test files test mock objects rather than the actual embedding services. `test_embedding_service.py` imports `MockEmbeddingService` from `tests/conftest.py` and asserts that it returns 768-dimensional vectors. `test_siglip_embedding_service.py` defines its own `MockSigLIPEmbeddingService` inline and runs similar assertions.

**Why this matters**: These tests provide zero assurance that the real `EmbeddingService` or `SigLIPEmbeddingService` classes work correctly. They validate that a hash-based mock produces vectors of the right dimension -- a property of the test infrastructure, not the production code. The last two tests in `test_siglip_embedding_service.py` (`test_siglip_service_lazy_loading` and `test_siglip_service_uses_correct_config`) do test the real class, but only verify that `_load_model` is called, not that embeddings are correct.

**What should exist instead**: Tests that verify the real `EmbeddingService.__init__`, `embed_text()`, and `embed_image()` interface contracts using lightweight mocks of only the underlying OpenCLIP/SigLIP model, not the entire service. At minimum, verify that the service correctly calls the model, preprocesses inputs, and normalizes outputs.

## 2. Patch-Stacking in test_main.py

**File**: `tests/test_main.py`

Every test in this file uses 4-6 stacked `@patch` decorators:

```python
@patch("image_search_service.vector.qdrant.validate_qdrant_collections")
@patch("image_search_service.core.config.get_settings")
@patch("image_search_service.services.embedding.preload_embedding_model")
@patch("image_search_service.services.watcher_manager.WatcherManager")
@patch("image_search_service.main.close_db", new_callable=AsyncMock)
@patch("image_search_service.main.close_qdrant")
async def test_startup_success_all_collections_exist(self, mock1, mock2, ...):
```

**Why this matters**: The reversed parameter order of `@patch` (outermost decorator maps to last parameter) is a known source of bugs. More importantly, these tests are extremely fragile -- any change to the lifespan function's import structure breaks every test. The tests verify that certain functions are called, but cannot verify the actual startup/shutdown sequence because every dependency is replaced.

**Root cause**: The `lifespan()` function has too many direct dependencies. The tests compensate by mocking everything, which makes them test wiring rather than behavior.

## 3. PersonService Tests Mock Internal Methods

**File**: `tests/unit/services/test_person_service.py`

The async tests replace private methods on the service instance:

```python
service._get_identified_people = AsyncMock(return_value=mock_identified)
service._get_unidentified_clusters = AsyncMock(return_value=[])
service._get_noise_faces = AsyncMock(return_value=None)
```

**Nuance**: Not all tests in this file are hollow. Approximately 25% test real logic: `_generate_display_name` has 6 solid test cases covering defaults, named persons, cluster labeling, unnamed clusters, noise faces, and `None` handling. The mocked async tests DO validate real sorting, filtering, and count logic -- the data fetching is mocked but the data processing is real.

**Why the approach still matters**: These tests cannot catch bugs in the actual SQL queries that fetch people from the database. If `_get_identified_people` is refactored (renamed, merged, split), every mocked test breaks despite no behavioral change. The tests couple to internal structure rather than observable behavior.

**What should exist instead**: Tests that use the real `db_session` fixture (like `test_unknown_person_service.py` does) to create actual Person/FaceInstance records and call `get_all_people()` against the real database. The existing `_generate_display_name` tests and data-processing assertions can remain as-is.

## 4. Queue Tests Primarily Verify Mock Behavior

**File**: `tests/api/test_queues.py`

This file defines elaborate mock classes (`MockQueueService`, `MockQueueServiceRedisDown`) that replicate the queue service's interface, then runs HTTP requests against the API that uses these mocks. The tests verify that the API correctly serializes mock data into JSON responses.

**Nuance**: The tests are not useless. They also verify HTTP status codes, 404 handling for missing queues, pagination logic, Redis-down error scenarios, and worker detail endpoints. They validate the API layer's routing, error handling, and serialization contract -- which has real value.

**The real gap**: The mock queue service contains hardcoded return values (`total_jobs=42`, `workers_count=2`). While the API layer behavior is validated, there is no assurance that the real `QueueService` correctly queries Redis, counts jobs, or handles connection failures. The `QueueService` interaction with Redis is the untested layer, not the API routing itself.

## 5. Qdrant Collection Validation Tests Are Heavily Mocked

**File**: `tests/unit/test_validate_qdrant_collections.py`

Every test creates a `Mock()` for each collection object, manually setting `.name` attributes:

```python
mock_col1 = Mock()
mock_col1.name = "image_assets"
mock_col2 = Mock()
mock_col2.name = "faces"
# ... repeated for every test
```

There is extensive code duplication: the same settings mock and collection mock setup is repeated across all 5 tests with only minor variations. The tests also validate a narrow slice of behavior (collection name checking) while missing error recovery and retry logic.

## 6. Face Job Tests Primarily Verify Mock Orchestration

**File**: `tests/unit/queue/test_face_jobs_coverage.py`

While comprehensive in scope (1054 lines), these tests follow a consistent pattern: mock all dependencies with `patch()`, call the job function, and assert the mock return value appears in the result. For example:

```python
mock_clusterer.cluster_all_faces.return_value = {
    "total_processed": 5,
    "assigned_to_people": 2,
    ...
}
result = cluster_dual_job()
assert result["total_processed"] == 5  # Just checking the mock's return value
```

The `expire_old_suggestions_job` tests are an exception -- they create real database records and verify that the correct records are updated. This pattern should be extended to other job tests.

## 7. Parametrize Underutilized

`@pytest.mark.parametrize` is used effectively in two files -- `tests/unit/test_temporal_service.py` (5 uses with 19+ parametrized cases across `TestClassifyAgeEra`) and `tests/core/test_device.py` (5 uses) -- but these are the exception. Many other test files contain repeated patterns that would benefit from the same approach:

- `test_validate_qdrant_collections.py`: Same setup with different collection configurations
- `test_embedding_router.py`: Seven nearly identical test functions with different settings combinations
- `test_perceptual_hash.py`: `TestHammingDistance` could parametrize hash pairs and expected distances
- `test_fusion.py`: Weighted RRF tests with different weight configurations

The existing usage in `test_temporal_service.py` and `test_device.py` demonstrates the team knows the pattern well. Extending it to the files listed above would reduce code duplication, make edge cases easier to add, and improve test readability.

## 8. Inconsistent Use of monkeypatch vs @patch

The test suite inconsistently uses `monkeypatch.setattr()` and `@patch()`/`with patch()`:

- `test_training_jobs_coverage.py` uses `monkeypatch` exclusively (cleaner, auto-cleanup)
- `test_face_jobs_coverage.py` uses `with patch()` context managers
- `test_embedding_router.py` uses nested `with patch()` blocks (up to 3 levels deep)
- `test_main.py` uses stacked `@patch` decorators

This inconsistency makes the test suite harder to maintain. `monkeypatch` is generally preferable because it is scoped to the test function automatically and does not require understanding `@patch` parameter ordering.

## 9. Weak Assertion Patterns

Several test files use assertions that pass too easily:

- `test_face_clustering_service.py` line 157: `assert 0.0 <= confidence <= 1.0` -- this accepts any valid float, providing no meaningful constraint on the algorithm's output for random vectors.
- `test_face_jobs_coverage.py` line 775: `assert result["total_images"] > 0` -- only checks the value is positive, not that it matches the number of created assets.
- `test_validate_qdrant_collections.py` line 154: `assert len(missing) >= 3` -- uses `>=` when the exact count is known (should be `== 3`).

## 10. Test File Size and Organization

Several test files are excessively large:

- `tests/api/test_search.py`: ~50KB (over 1000 lines)
- `tests/api/test_storage_routes.py`: ~58KB
- `tests/unit/queue/test_face_jobs_coverage.py`: 1054 lines

These files would benefit from being split by feature or behavior. Large test files make it harder to find relevant tests, increase merge conflict risk, and discourage developers from adding new test cases.
