# Test Suite Strengths

## 1. ExifService Tests Are Exemplary

`tests/unit/services/test_exif_service.py` is the gold standard in this test suite. It demonstrates how unit tests should be written:

- **Real test images**: Uses `create_image_with_exif()` to build actual JPEG files with PIL, embedding specific EXIF tags. This validates the actual parsing path rather than mocking PIL internals.
- **Comprehensive edge cases**: Tests GPS coordinates in all four hemispheres (N/S/E/W), null byte sanitization in string fields (`_sanitize_for_json`), IFDRational type handling, truncation of long strings, whitespace trimming.
- **Thread safety**: Includes a concurrency test spinning up 10 threads to verify the service handles parallel access without corruption.
- **Error handling**: Tests for corrupt files, missing EXIF data, and graceful fallbacks.
- **Clean assertion style**: Each test validates one specific behavior with descriptive names like `test_extract_metadata_gps_southern_hemisphere`.

This file should serve as the reference template when writing new test modules.

## 2. Race Condition Documentation Tests

`tests/unit/services/test_centroid_race_conditions.py` and `tests/unit/services/test_training_race_conditions.py` are unusual and valuable. They use `asyncio.gather()` to simulate concurrent operations on the same resource, explicitly documenting known bugs:

- Double-deprecation during concurrent centroid computation
- Inconsistent state when training session start and cancel run simultaneously
- Transactional gap where Qdrant failure leaves a person with no active centroid

Each test includes detailed docstrings explaining the expected (buggy) behavior and the correct behavior that a fix should implement. This approach turns tests into living documentation of known concurrency issues, making it clear to any developer what needs fixing and why.

## 3. Qdrant Safety Guard Tests

`tests/unit/test_qdrant_safety.py` contains critical regression tests that prevent production data destruction:

- Verifies collection names come from environment variables, not hardcoded strings
- Confirms that calling `reset_collection()` on production collection names raises `RuntimeError` when `PYTEST_CURRENT_TEST` is set
- Validates that test fixtures properly override collection names to `test_*` prefixes
- Checks that settings cache clearing prevents cross-test contamination

These tests protect against a catastrophic failure mode (wiping production Qdrant data during test runs) and are exactly the kind of safety net every system with destructive operations needs.

## 4. Real Database Fixtures with In-Memory SQLite

The root `tests/conftest.py` provides a well-designed `db_session` fixture that creates a real async SQLite database with all SQLAlchemy models:

```python
@pytest.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite://", ...)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    ...
```

This lets tests like `test_unknown_person_service.py` exercise actual SQL queries (INSERT, SELECT with WHERE, DISTINCT) against a real schema without requiring PostgreSQL. The `sync_db_session` companion fixture (used by queue job tests) provides the same pattern for synchronous SQLAlchemy sessions.

The factory fixtures (`create_person`, `create_face_instance`, `create_image_asset`, `create_suggestion`, `create_detection_session`) in `tests/unit/queue/test_face_jobs_coverage.py` are well-structured and handle FK relationships correctly.

## 5. Comprehensive Fusion Algorithm Tests

`tests/unit/test_fusion.py` is a strong example of algorithm testing:

- Tests reciprocal rank fusion with empty lists, single lists, two lists with same/reversed order, non-overlapping results, and varying k parameters.
- Tests weighted RRF with equal weights (verifying equivalence to standard RRF), text-heavy, and image-heavy weightings.
- Validates mathematical properties: RRF scores are monotonically decreasing, combined ranks are sequential, score ratios change correctly with k parameter.
- Uses a clean `MockAsset(BaseModel)` pattern rather than over-complicated mocks.

## 6. Perceptual Hash Tests with Real Image Operations

`tests/unit/test_perceptual_hash.py` tests the hashing algorithm using real images:

- Generates gradient images (horizontal vs vertical) to verify distinct hashes
- Tests JPEG compression at different quality levels to verify hash stability
- Tests color vs grayscale versions of the same image
- Validates Hamming distance computation with exact bit-level expectations
- Tests boundary conditions on similarity thresholds

## 7. Face Clustering Service Tests

`tests/unit/services/test_face_clustering_service.py` demonstrates good mathematical testing:

- Tests cosine similarity with identical, orthogonal, opposite, and zero-norm vectors
- Tests cluster confidence calculation with known embedding patterns (identical, orthogonal, high-similarity, random)
- Validates sampling behavior for large clusters (50 faces sampled to 20)
- Tests representative face selection logic (quality score, bbox size tiebreaking)

## 8. Listener Worker Tests

`tests/unit/queue/test_listener_worker.py` covers the custom RQ worker lifecycle thoroughly:

- Worker registration/deregistration with Redis
- State transitions (idle to busy to idle) tracked via spy functions
- Job registry management (FinishedJobRegistry, FailedJobRegistry)
- Graceful failure handling when Redis is unavailable
- RQ 2.x compatibility (StartedJobRegistry removed)

## 9. Training Job Coverage

`tests/unit/queue/test_training_jobs_coverage.py` and `tests/unit/queue/test_face_jobs_coverage.py` cover complex multi-step job orchestration:

- Session lifecycle: DISCOVERING -> RUNNING -> COMPLETED
- Cancellation and pause handling with ProgressTracker mocks
- Partial failure scenarios (1 of 3 jobs fails)
- Evidence metadata generation with proper structure validation
- Uses `monkeypatch` instead of `@patch` for cleaner dependency injection

## 10. Embedding Router Tests

`tests/unit/test_embedding_router.py` covers the A/B testing routing logic well:

- Tests the three routing paths: `use_siglip=True`, `use_siglip=False`, rollout percentage
- Boundary conditions: 0% rollout (always CLIP), 100% rollout (always SigLIP)
- Deterministic bucketing with explicit user_id values
- Override precedence: `use_siglip=True` overrides rollout percentage
- Random bucketing when no user_id provided (mocks `random.randint`)

## 11. Unknown Person Service with Real DB

`tests/unit/test_unknown_person_service.py` tests hash computation and dismissal logic against the real SQLite-backed session:

- Order-independent hash computation verified with multiple orderings
- Persistence validation: creates dismissal records and queries them back
- Set operations: `get_dismissed_hashes` returns proper `set` for filtering
- Edge cases: empty lists, single IDs, subsets producing different hashes

## 12. PostgreSQL Testcontainers Integration

`tests/conftest_postgres.py` (173 lines) provides real PostgreSQL 16 via testcontainers with per-test rollback, fresh DB creation for migration tests, and safety assertions preventing production DB connections. This infrastructure is used by 5 integration test files under `tests/integration/`. Running tests against real PostgreSQL catches dialect-specific SQL issues that in-memory SQLite cannot reproduce (e.g., JSONB operators, constraint behaviors, migration idempotency).

## 13. Concurrency and Failure Injection Helpers

Two dedicated helper modules provide mature fault injection patterns:

- `tests/helpers/concurrency.py` (163 lines): `race_requests()` for simulating concurrent API calls, `OperationLogger` for tracing call sequences, `DelayedMockSession` for introducing artificial latency in DB operations.
- `tests/helpers/failure_injection.py` (117 lines): `FailAfterNCallsQdrant` for simulating Qdrant failures after N successful operations, `SelectiveFailureQdrant` for failing only on specific collection/operation combinations.

These demonstrate a testing maturity beyond basic mocking, enabling systematic exploration of failure modes.

## 14. SemanticMockEmbeddingService

`tests/conftest.py` (lines 222-397) contains a sophisticated mock embedding service that produces semantically meaningful embeddings using concept clusters (nature, animal, food, urban, people). Unlike a trivial mock returning random or zero vectors, this service generates deterministic vectors where semantically related terms produce closer embeddings. This allows search relevance tests to verify ranking behavior rather than just asserting that results are returned.

## 15. Autouse Safety Fixtures

Three `autouse=True` fixtures in `tests/conftest.py` provide global test isolation without requiring per-test opt-in:

- `test_settings`: Clears settings cache, sets safe Qdrant collection names (`test_*` prefixes), neutralizes Google Drive environment variables to prevent accidental cloud operations during tests.
- `clear_embedding_cache`: Patches `EmbeddingService` globally with `SemanticMockEmbeddingService`, preventing tests from loading multi-gigabyte ML models.
- `validate_embedding_dimensions`: Session-scoped guard that verifies embedding dimension consistency across all test runs.

This autouse pattern ensures that even newly added test files inherit safety protections without the author needing to remember to apply them.

## 16. Google Drive Storage Test Suite

9 test files under `tests/unit/storage/` provide thorough coverage of the Google Drive integration layer:

- OAuth flow handling and credential management
- Async wrappers for Drive API operations
- Path resolution and folder hierarchy navigation
- Upload service with retry and error handling
- Storage status and quota tracking

This suite demonstrates good practice for testing external service integrations by mocking at the HTTP boundary while validating the full application-layer logic.
