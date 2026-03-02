# Test Smells

Specific anti-patterns found in the test code, with file locations and explanations of why they reduce test value.

## 1. Testing the Mock Instead of the Production Code

**Files**:
- `tests/unit/test_embedding_service.py` (entire file)
- `tests/unit/test_siglip_embedding_service.py` (lines 1-79, `MockSigLIPEmbeddingService` class)

**Pattern**: The test imports a mock class and asserts properties of the mock.

```python
# tests/unit/test_embedding_service.py
from tests.conftest import MockEmbeddingService

def test_mock_embedding_service_embed_text_returns_vector():
    service = MockEmbeddingService()
    vector = service.embed_text("a beautiful sunset")
    assert len(vector) == 768
```

**Why it is a problem**: This test will pass forever regardless of whether the real `EmbeddingService` is broken. It validates test infrastructure, not production code. A developer reading the test name might assume the embedding service is tested when it is not.

**Fix**: Replace with tests that instantiate the real `EmbeddingService` with a mocked underlying model (mock OpenCLIP/SigLIP, not the entire service).

## 2. Excessive Patch Stacking

**File**: `tests/test_main.py`

**Pattern**: Multiple `@patch` decorators on a single test, requiring careful parameter ordering.

```python
@patch("image_search_service.vector.qdrant.validate_qdrant_collections")
@patch("image_search_service.core.config.get_settings")
@patch("image_search_service.services.embedding.preload_embedding_model")
@patch("image_search_service.services.watcher_manager.WatcherManager")
@patch("image_search_service.main.close_db", new_callable=AsyncMock)
@patch("image_search_service.main.close_qdrant")
async def test_startup_success(self, mock_close_qdrant, mock_close_db,
    mock_watcher_manager, mock_preload, mock_get_settings, mock_validate):
```

**Why it is a problem**:
1. The reversed parameter mapping of `@patch` is a known source of bugs -- adding/removing a decorator silently shifts all mock bindings.
2. The test is extremely brittle: adding any new import to `lifespan()` breaks all four tests.
3. With everything mocked, the test only verifies that functions are called in some order, not that the startup sequence actually works.

**Fix**: Refactor `lifespan()` to accept a dependency container or use FastAPI's dependency injection, then test with lightweight fakes rather than patching every import.

## 3. Mocking Private Methods

**File**: `tests/unit/services/test_person_service.py`

**Pattern**: Replacing private methods (`_get_identified_people`, `_get_unidentified_clusters`, `_get_noise_faces`) on the service instance.

```python
service._get_identified_people = AsyncMock(return_value=mock_identified)
service._get_unidentified_clusters = AsyncMock(return_value=[])
service._get_noise_faces = AsyncMock(return_value=None)
```

**Why it is a problem**: Private methods are implementation details. The test breaks if the method is renamed, merged with another method, or refactored internally -- even though the public behavior (`get_all_people()`) might be identical. This couples tests to internal structure rather than observable behavior.

**Fix**: Create real Person, FaceInstance, and cluster records in the test database and call `get_all_people()` against the real implementation.

## 4. Asserting Mock Return Values

**Files**:
- `tests/unit/queue/test_face_jobs_coverage.py` (TestClusterDualJob, TestComputeCentroidsJob, TestBackfillFacesJob)
- `tests/api/test_queues.py` (all tests)

**Pattern**: Setting a mock's return value, calling the function, then asserting the mock's return value appears in the result.

```python
mock_clusterer.cluster_all_faces.return_value = {
    "total_processed": 5,
    "assigned_to_people": 2,
}
result = cluster_dual_job()
assert result["total_processed"] == 5  # This is the mock's return value
```

**Why it is a problem**: This test passes regardless of what `cluster_dual_job()` actually does with the clusterer's return value. It only verifies that the return value is forwarded, not that the job correctly sets up inputs, handles errors, or updates state.

**Fix**: For job tests, use the real database with the `sync_db_session` fixture, mock only the heavy ML computation (InsightFace, OpenCLIP), and verify database state changes after the job runs. The `expire_old_suggestions_job` tests in the same file demonstrate this better pattern.

## 5. Duplicated Mock Setup

**File**: `tests/unit/test_validate_qdrant_collections.py`

**Pattern**: Every test repeats the same 15-line mock setup with minor variations.

```python
# Repeated in EVERY test:
mock_settings = Mock()
mock_settings.qdrant_collection = "image_assets"
mock_settings.qdrant_face_collection = "faces"
mock_settings.qdrant_centroid_collection = "person_centroids"
mock_settings.use_siglip = False
mock_get_settings.return_value = mock_settings

mock_client = Mock()
mock_col1 = Mock()
mock_col1.name = "image_assets"
mock_col2 = Mock()
mock_col2.name = "faces"
...
```

**Why it is a problem**: Duplication makes tests harder to update (change to settings structure requires editing every test), harder to read (signal buried in noise), and discourages adding new test cases.

**Fix**: Extract a fixture or helper function that creates the settings mock and collection mocks with configurable parameters. Use `@pytest.mark.parametrize` for the different collection configurations.

## 6. Nested Context Managers for Patching

**File**: `tests/unit/test_embedding_router.py`

**Pattern**: Three levels of `with patch(...)` nesting.

```python
with patch("...get_settings") as mock_settings:
    settings_obj = MagicMock()
    ...
    with patch("...get_embedding_service") as mock_clip, \
         patch("...get_siglip_service") as mock_siglip:
        mock_clip_service = MagicMock()
        ...
        service, collection = get_search_embedding_service()
```

**Why it is a problem**: Deep nesting obscures the test's intent. The reader must trace through multiple indentation levels to find the actual assertion. This pattern is repeated identically across 7 test functions.

**Fix**: Use `monkeypatch.setattr()` in a fixture, or use `@pytest.fixture` with `mock.patch.dict` to set up the routing configuration once. Then each test becomes 3-5 lines of setup and assertion.

## 7. Overly Broad Exception Assertions

**File**: `tests/unit/queue/test_face_jobs_coverage.py`, line 724

```python
result = detect_faces_for_session_job(invalid_session_id)
assert result["status"] == "failed"
assert "not found" in result["error"].lower()
```

**Why it is a problem**: The `.lower()` check on the error message is fragile and could pass with unrelated error messages containing "not found". It does not verify the specific error type or the session ID in the error context.

**Better alternative**: Assert the exact error structure or use a more specific substring.

## 8. `os.environ` Mutation Without Cleanup

**File**: `tests/unit/test_qdrant_safety.py`, lines 63 and 87

```python
os.environ["QDRANT_FACE_COLLECTION"] = "faces"
# ... test runs ...
os.environ["QDRANT_FACE_COLLECTION"] = "test_faces"  # Manual cleanup
```

**Why it is a problem**: If the test fails before reaching the cleanup line, the environment is left polluted for subsequent tests. The `monkeypatch.setenv()` approach used elsewhere in the same file (lines 20, 43) handles cleanup automatically.

**Fix**: Replace `os.environ[...] = ...` with `monkeypatch.setenv()` throughout.

## 9. Magic Numbers in Assertions

**Files**: Multiple

```python
# tests/unit/test_embedding_service.py
assert len(vector) == 768  # Why 768? Where does this come from?

# tests/unit/test_unknown_person_service.py
assert len(hash1) == 64  # SHA-256 hex digest, but not documented

# tests/api/test_queues.py
assert data["total_jobs"] == 42  # Hardcoded in mock, matches mock
```

**Why it is a problem**: Magic numbers make tests harder to understand and maintain. When the embedding dimension changes from 768 to 512, which assertions need updating?

**Fix**: Use named constants (`CLIP_EMBEDDING_DIM = 768`, `SHA256_HEX_LENGTH = 64`) imported from production code or test constants.

## 10. `np.random.seed()` in Tests -- Pervasive Global State Mutation

**Files** (19 instances across 4 files):
- `tests/faces/test_trainer.py`: **11 occurrences**
- `tests/faces/test_clusterer.py`: **4 occurrences**
- `tests/unit/services/test_face_clustering_service.py`: **1 occurrence** (line 106)
- `tests/unit/test_cluster_confidence.py`: **3 occurrences**

```python
np.random.seed(42)
base_vector = np.random.rand(512)
```

**Why this is a problem**: With 19 instances, this is not an isolated oversight but a systemic pattern. Global random seed affects all subsequent numpy operations in the test process. If test ordering changes (e.g., via pytest-randomly), seeds set in one test may not be active when another test expects them, causing flaky failures. The heaviest concentration is in `test_trainer.py` (11 instances), where multiple tests each set the global seed -- meaning earlier tests' seed state leaks into later tests' setup phases.

**Severity**: MEDIUM -- the number of instances increases the likelihood of order-dependent test failures and makes the test suite incompatible with randomized test execution.

**Fix**: Replace all 19 instances with `np.random.default_rng(42)` for a local random generator that does not affect global state. Verify determinism is preserved by running each affected test file in isolation and in reversed order.

## 11. Tests That Document Bugs but Cannot Detect Fixes

**Files**: `tests/unit/services/test_centroid_race_conditions.py`, `tests/unit/services/test_training_race_conditions.py`

These tests are valuable as documentation (see `strengths.md`), but they have a subtle problem: they assert that the bug EXISTS. If someone fixes the race condition (e.g., adding `SELECT ... FOR UPDATE`), the test will FAIL because the fix prevents double-deprecation.

```python
# This asserts the bug exists:
assert deprecation_count == 2, "Expected 2 deprecation calls (race condition)"
```

**Fix**: Add a comment or marker (`@pytest.mark.xfail(reason="Known race condition")`) so that when the fix is applied, the test failure is expected and points to updating the test rather than reverting the fix.

## 12. Inconsistent Test Class Usage

Some test files use classes (`class TestClusterDualJob`, `class TestPerceptualHashComputation`), while others use standalone functions. The class-based tests sometimes use `self` only to reference fixtures, gaining no organizational benefit.

More problematic: `tests/unit/services/test_person_service.py` uses a class with `@pytest.fixture` methods that are only used within that class, but the `service` fixture depends on `mock_db` which is also a fixture -- a fixture chain that is hard to follow without reading the entire class.

**Fix**: Adopt a consistent convention. For this codebase, standalone functions with descriptive names (like the ExifService tests or fusion tests) tend to be more readable than class-based organization.
