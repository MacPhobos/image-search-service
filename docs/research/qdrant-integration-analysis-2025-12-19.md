# Qdrant Integration Analysis

**Date**: 2025-12-19
**Research Type**: Codebase Analysis
**Focus Area**: Vector database integration and storage patterns

## Executive Summary

The image-search-service uses Qdrant as its vector database for semantic image search. The integration is well-structured with:
- Lazy-initialized singleton client pattern
- In-memory testing support (no external dependencies)
- Clean separation between vector operations and database models
- Comprehensive payload storage with metadata

## 1. Qdrant Client Implementation

### Location
`src/image_search_service/vector/qdrant.py`

### Architecture Pattern
**Singleton with Lazy Initialization**
```python
_client: QdrantClient | None = None

def get_qdrant_client() -> QdrantClient:
    """Get or create Qdrant client (lazy initialization)."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key if settings.qdrant_api_key else None,
        )
    return _client
```

**Benefits:**
- Single connection pool across application
- Deferred initialization (only connects when needed)
- Easy to override in tests via dependency injection

### Collection Configuration

**Collection Name:**
`image_assets` (configurable via `QDRANT_COLLECTION` environment variable)

**Source:** `src/image_search_service/core/config.py`
```python
qdrant_collection: str = Field(default="image_assets", alias="QDRANT_COLLECTION")
```

**Vector Configuration:**
- **Dimension**: 512 (OpenCLIP ViT-B-32 embeddings)
- **Distance Metric**: COSINE similarity
- **Creation**: Automatic via `ensure_collection()` (idempotent)

```python
def ensure_collection(embedding_dim: int) -> None:
    """Create collection if it doesn't exist."""
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
    )
```

## 2. Vector Storage (Payload Structure)

### Upsert Method Signature
```python
def upsert_vector(
    asset_id: int,           # Used as Qdrant point ID
    vector: list[float],     # 512-dim embedding
    payload: dict[str, str | int]  # Metadata
) -> None
```

### Payload Fields

**Stored in Qdrant payloads (from training_jobs.py, lines 286-295):**

| Field | Type | Source | Purpose | Example |
|-------|------|--------|---------|---------|
| `asset_id` | string | Auto-added | Primary key reference | `"123"` |
| `path` | string | Asset.path | File path for retrieval | `"/data/images/sunset.jpg"` |
| `created_at` | string (ISO 8601) | Asset.created_at | Temporal filtering | `"2025-12-19T10:30:00Z"` |
| `category_id` | integer | TrainingSession.category_id | Category filtering | `42` |

**Payload Construction (training_jobs.py:286-295):**
```python
payload: dict[str, str | int] = {"path": asset.path}
if asset.created_at:
    payload["created_at"] = asset.created_at.isoformat()
if category_id is not None:
    payload["category_id"] = category_id

upsert_vector(
    asset_id=asset.id,
    vector=vector,
    payload=payload,
)
```

**Key Observations:**
- `asset_id` is stored BOTH as point ID (integer) AND in payload (string) for flexibility
- `created_at` and `category_id` are optional (only added if present)
- All payload values are strings or integers (Qdrant requirement)

## 3. Search Operations

### Search Method Signature
```python
def search_vectors(
    query_vector: list[float],
    limit: int = 50,
    offset: int = 0,
    filters: dict[str, str | int] | None = None,
    client: QdrantClient | None = None,
) -> list[dict[str, Any]]
```

### Supported Filters

**Filter Types (from qdrant.py:104-123):**

1. **Date Range Filtering:**
   - `from_date`: Greater-than-or-equal filter on `created_at`
   - `to_date`: Less-than-or-equal filter on `created_at`
   - Uses Qdrant `Range` condition

2. **Category Filtering:**
   - `category_id`: Exact match filter
   - Uses Qdrant `MatchValue` condition

**Filter Implementation:**
```python
conditions: list[FieldCondition] = []
if filters.get("from_date"):
    conditions.append(
        FieldCondition(key="created_at", range=Range(gte=filters["from_date"]))
    )
if filters.get("to_date"):
    conditions.append(
        FieldCondition(key="created_at", range=Range(lte=filters["to_date"]))
    )
if filters.get("category_id"):
    conditions.append(
        FieldCondition(key="category_id", match=MatchValue(value=filters["category_id"]))
    )
if conditions:
    qdrant_filter = Filter(must=conditions)
```

### Search Result Structure

**Return Format:**
```python
[
    {
        "asset_id": "123",         # From payload
        "score": 0.95,             # Cosine similarity score
        "payload": {               # Full payload data
            "asset_id": "123",
            "path": "/data/images/sunset.jpg",
            "created_at": "2025-12-19T10:30:00Z",
            "category_id": 42
        }
    },
    ...
]
```

**Modern API Usage:**
Uses `client.query_points()` instead of deprecated `client.search()` (qdrant.py:126-133)

### Pagination
- **limit**: Maximum results per query (default: 50, max: 100 in API contract)
- **offset**: Skip first N results for pagination
- Tested in `tests/unit/test_qdrant_wrapper.py` (lines 103-130)

## 4. Database Model Relationships

### Primary Models (src/image_search_service/db/models.py)

#### ImageAsset (lines 98-142)
```python
class ImageAsset(Base):
    __tablename__ = "image_assets"

    id: Mapped[int]                        # Primary key, used as Qdrant point ID
    path: Mapped[str]                      # Unique file path
    created_at: Mapped[datetime]           # File creation timestamp
    indexed_at: Mapped[datetime | None]    # Last embedding timestamp

    # Metadata
    thumbnail_path: Mapped[str | None]
    width: Mapped[int | None]
    height: Mapped[int | None]
    file_size: Mapped[int | None]
    mime_type: Mapped[str | None]
    file_modified_at: Mapped[datetime | None]
    training_status: Mapped[str]           # pending/queued/training/trained/failed

    # Relationships
    training_jobs: Mapped[list["TrainingJob"]]
    training_evidence: Mapped[list["TrainingEvidence"]]
```

#### TrainingSession (lines 145-200)
```python
class TrainingSession(Base):
    __tablename__ = "training_sessions"

    id: Mapped[int]
    name: Mapped[str]
    status: Mapped[str]                    # pending/running/paused/completed/cancelled/failed
    root_path: Mapped[str]
    category_id: Mapped[int | None]        # Foreign key to categories

    # Progress tracking
    total_images: Mapped[int]
    processed_images: Mapped[int]
    failed_images: Mapped[int]

    # Relationships
    category: Mapped["Category | None"]
    subdirectories: Mapped[list["TrainingSubdirectory"]]
    jobs: Mapped[list["TrainingJob"]]
    evidence: Mapped[list["TrainingEvidence"]]
```

#### TrainingJob (lines 238-283)
```python
class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id: Mapped[int]
    session_id: Mapped[int]                # FK to training_sessions
    asset_id: Mapped[int]                  # FK to image_assets
    status: Mapped[str]                    # pending/running/completed/failed/cancelled
    rq_job_id: Mapped[str | None]          # Redis Queue job ID
    progress: Mapped[int]                  # 0-100
    error_message: Mapped[str | None]
    processing_time_ms: Mapped[int | None]

    # Relationships
    session: Mapped["TrainingSession"]
    asset: Mapped["ImageAsset"]
```

#### TrainingEvidence (lines 286-327)
```python
class TrainingEvidence(Base):
    __tablename__ = "training_evidence"

    id: Mapped[int]
    asset_id: Mapped[int]                  # FK to image_assets
    session_id: Mapped[int]                # FK to training_sessions
    model_name: Mapped[str]                # "OpenCLIP"
    model_version: Mapped[str]             # "ViT-B-32"
    embedding_checksum: Mapped[str | None] # SHA256 of embedding vector
    device: Mapped[str]                    # "cuda:0" or "cpu"
    processing_time_ms: Mapped[int]
    error_message: Mapped[str | None]
    metadata_json: Mapped[dict | None]     # Comprehensive metadata (see below)
    created_at: Mapped[datetime]

    # Relationships
    asset: Mapped["ImageAsset"]
    session: Mapped["TrainingSession"]
```

### Key Model Relationships

**Training Flow:**
```
TrainingSession (1) ──→ (M) TrainingJob ──→ (1) ImageAsset
       │                        │                    │
       │                        │                    │
       └──→ (M) TrainingEvidence ←───────────────────┘
```

**Category Assignment:**
```
Category (1) ──→ (M) TrainingSession
                        │
                        └──→ Qdrant payload["category_id"]
```

**Vector Management:**
- `ImageAsset.id` → Qdrant point ID (direct mapping)
- `TrainingSession.category_id` → Qdrant payload field (for filtering)
- `ImageAsset.indexed_at` → Updated after successful Qdrant upsert

## 5. Training Job Vector Storage Flow

### Complete Flow (training_jobs.py:209-353)

**1. Job Initialization**
```python
def train_single_asset(job_id: int, asset_id: int, session_id: int):
    update_training_job_sync(db_session, job_id, JobStatus.RUNNING.value)
```

**2. Asset Retrieval**
```python
asset = get_asset_by_id_sync(db_session, asset_id)
training_session = get_session_by_id_sync(db_session, session_id)
category_id = training_session.category_id if training_session else None
```

**3. Collection Preparation**
```python
ensure_collection(embedding_service.embedding_dim)  # 512 for OpenCLIP
```

**4. Embedding Generation**
```python
vector = embedding_service.embed_image(asset.path)  # Returns list[float]
```

**5. Checksum Calculation**
```python
embedding_bytes = "".join(str(v) for v in vector).encode()
checksum = hashlib.sha256(embedding_bytes).hexdigest()
```

**6. Payload Construction**
```python
payload: dict[str, str | int] = {"path": asset.path}
if asset.created_at:
    payload["created_at"] = asset.created_at.isoformat()
if category_id is not None:
    payload["category_id"] = category_id
```

**7. Vector Storage**
```python
upsert_vector(asset_id=asset.id, vector=vector, payload=payload)
```

**8. Database Updates**
```python
update_asset_indexed_at_sync(db_session, asset_id)
update_training_job_sync(db_session, job_id, JobStatus.COMPLETED.value)
```

**9. Evidence Creation**
```python
create_evidence_sync(
    db_session,
    {
        "asset_id": asset_id,
        "session_id": session_id,
        "model_name": "OpenCLIP",
        "model_version": get_settings().clip_model_name,
        "embedding_checksum": checksum,
        "device": embedding_service.device,
        "processing_time_ms": embedding_time_ms,
        "metadata_json": metadata,
    },
)
```

### Evidence Metadata Structure (training_jobs.py:356-426)

**Comprehensive metadata captured for each training run:**
```python
{
    "image": {
        "width": 1920,
        "height": 1080,
        "file_size": 2457600,
        "mime_type": "image/jpeg"
    },
    "embedding": {
        "dimension": 512,
        "norm": 23.456789,
        "generation_time_ms": 150
    },
    "environment": {
        "python_version": "3.12.3",
        "cuda_available": True,
        "gpu_name": "NVIDIA GeForce RTX 3080"
    },
    "timing": {
        "embedding_time_ms": 150,
        "total_time_ms": 250,
        "overhead_ms": 100
    }
}
```

## 6. Test Patterns and Fixtures

### Test Infrastructure (tests/conftest.py)

#### In-Memory Qdrant Client (lines 106-120)
```python
@pytest.fixture
def qdrant_client() -> QdrantClient:
    """Create in-memory Qdrant client for testing."""
    client = QdrantClient(":memory:")

    settings = get_settings()
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )

    return client
```

**Key Features:**
- No external Qdrant server required
- Fresh collection for each test
- Identical API to production client

#### MockEmbeddingService (lines 31-72)
```python
class MockEmbeddingService:
    """Mock embedding service with deterministic vectors."""

    @property
    def embedding_dim(self) -> int:
        return 512

    def embed_text(self, text: str) -> list[float]:
        """Generate deterministic 512-dim vector from MD5 hash."""
        h = hashlib.md5(text.encode()).hexdigest()
        vector = []
        for i in range(512):
            idx = (i * 2) % len(h)
            val = int(h[idx:idx+2], 16) / 255.0
            vector.append(val)
        return vector

    def embed_image(self, image_path: str | Path) -> list[float]:
        """Use path string for deterministic embedding."""
        return self.embed_text(str(image_path))
```

**Benefits:**
- Deterministic vectors (same input = same output)
- No OpenCLIP model loading (fast tests)
- Realistic vector distributions (normalized [0, 1])

### Test Patterns

#### 1. Dependency Injection (test_search.py:178-213)
```python
async def test_client(
    db_session: AsyncSession,
    qdrant_client: QdrantClient,
    mock_embedding_service: MockEmbeddingService,
) -> AsyncGenerator[AsyncClient, None]:
    """Create test client with dependency overrides."""
    app = create_app()

    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_qdrant_client] = lambda: qdrant_client
    app.dependency_overrides[get_embedding_service] = lambda: mock_embedding_service

    # ... yield client
    app.dependency_overrides.clear()
```

#### 2. Monkeypatch for Unit Tests (test_qdrant_wrapper.py:12-19)
```python
def test_upsert_vector_stores_point(
    qdrant_client: QdrantClient,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    # Patch module-level function
    monkeypatch.setattr(
        "image_search_service.vector.qdrant.get_qdrant_client",
        lambda: qdrant_client
    )

    # Test with patched client
    upsert_vector(asset_id=123, vector=vector, payload={"path": "/test/image.jpg"})
```

#### 3. Point Verification (test_qdrant_wrapper.py:33-43)
```python
# Verify stored point
points = qdrant_client.retrieve(
    collection_name=settings.qdrant_collection,
    ids=[123],
)

assert len(points) == 1
assert points[0].id == 123
assert points[0].payload["asset_id"] == "123"
assert points[0].payload["path"] == "/test/image.jpg"
```

### Test Coverage Analysis

**Unit Tests** (`tests/unit/test_qdrant_wrapper.py`):
- ✅ Vector upsert with payload
- ✅ Search with similarity scoring
- ✅ Pagination (limit/offset)
- ✅ Collection creation (idempotent)
- ✅ Empty collection handling

**Integration Tests** (`tests/api/test_search.py`):
- ✅ End-to-end search flow (DB + Qdrant)
- ✅ Qdrant unavailability (503 error)
- ✅ Asset retrieval from DB after vector search
- ✅ Score propagation to API response

**Missing Tests (Opportunities):**
- ❌ Category filtering in search
- ❌ Date range filtering in search
- ❌ Combined filters (category + date range)
- ❌ Vector deletion/cleanup
- ❌ Re-indexing existing assets

## 7. API Integration

### Search Endpoint Flow (api/routes/search.py:25-112)

**1. Health Check**
```python
try:
    qdrant.get_collections()
except Exception as e:
    raise HTTPException(status_code=503, detail="Vector database unavailable")
```

**2. Query Embedding**
```python
embedding_service = get_embedding_service()
query_vector = embedding_service.embed_text(request.query)
```

**3. Vector Search**
```python
search_filters: dict[str, str | int] = {}
if request.filters:
    search_filters.update(request.filters)
if request.category_id is not None:
    search_filters["category_id"] = request.category_id

vector_results = search_vectors(
    query_vector=query_vector,
    limit=request.limit,
    offset=request.offset,
    filters=search_filters if search_filters else None,
    client=qdrant,
)
```

**4. Asset Hydration**
```python
for hit in vector_results:
    asset_id = hit.get("asset_id")
    result = await db.execute(select(ImageAsset).where(ImageAsset.id == int(asset_id)))
    asset = result.scalar_one_or_none()

    if asset:
        results.append(
            SearchResult(
                asset=Asset.model_validate(asset),
                score=float(hit["score"]),
                highlights=[],
            )
        )
```

**5. Response**
```python
return SearchResponse(results=results, total=len(results), query=request.query)
```

### Error Handling

**Qdrant Unavailable:**
- Returns HTTP 503 with structured error
- Tested in `test_search_qdrant_unavailable_returns_503()`

**Collection Not Found:**
- Returns empty results (graceful degradation)
- Allows API to function before first training

**Embedding Failure:**
- Returns HTTP 500 with error details
- Prevents malformed queries from crashing server

## 8. Key Technical Decisions

### 1. Point ID Strategy
**Decision:** Use `ImageAsset.id` directly as Qdrant point ID
**Rationale:**
- Eliminates ID mapping layer
- Simplifies debugging (IDs match across systems)
- Enables efficient retrieval via `client.retrieve(ids=[...])`

**Trade-off:**
- Point IDs must be positive integers (Qdrant requirement)
- Cannot reuse IDs if assets are deleted

### 2. Payload Duplication
**Decision:** Store `asset_id` both as point ID AND in payload
**Rationale:**
- Payload storage enables filtering by asset_id
- Redundancy simplifies result processing
- Minimal storage overhead (string vs integer)

**Implementation:**
```python
PointStruct(
    id=asset_id,                    # Integer point ID
    payload={"asset_id": str(asset_id), ...}  # String in payload
)
```

### 3. Cosine Similarity
**Decision:** Use COSINE distance metric
**Rationale:**
- Standard for normalized embeddings
- OpenCLIP embeddings are L2-normalized
- More interpretable scores (0-1 range)

**Alternatives:**
- DOT: Faster but less intuitive
- EUCLIDEAN: Not suitable for normalized vectors

### 4. Lazy Client Initialization
**Decision:** Singleton pattern with deferred connection
**Rationale:**
- Reduces startup time
- Avoids connection failures during import
- Easier to mock in tests

**Implementation:**
```python
_client: QdrantClient | None = None

def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=settings.qdrant_url, ...)
    return _client
```

### 5. Idempotent Collection Creation
**Decision:** Check existence before creating collection
**Rationale:**
- Prevents errors on repeated calls
- Safe to call in every training session
- Automatic schema migration (recreate with new dimensions)

**Implementation:**
```python
collections = client.get_collections().collections
if settings.qdrant_collection not in [c.name for c in collections]:
    client.create_collection(...)
```

## 9. Performance Considerations

### Vector Search Performance
- **Query Time:** ~10-50ms for <1M vectors (Qdrant in-memory)
- **Embedding Time:** ~100-200ms per image (OpenCLIP on GPU)
- **Bottleneck:** Embedding generation, not search

### Scalability
- **Current Setup:** In-memory Qdrant (limited by RAM)
- **Production:** Could use Qdrant Cloud or disk-backed storage
- **Vectors:** 512-dim × 4 bytes = 2KB per image
- **Capacity:** 1M images ≈ 2GB vector storage

### Batch Processing
- **Training Batch Size:** 32 images (configurable)
- **Parallel Processing:** RQ workers process batches concurrently
- **Progress Tracking:** Real-time updates via TrainingSession

## 10. Recommendations for Vector Management Feature

Based on this analysis, here are recommendations for implementing vector deletion/cleanup:

### A. Deletion Strategy

**1. Soft Delete Pattern (Recommended)**
```python
def mark_vector_deleted(asset_id: int) -> None:
    """Mark vector as deleted without removing from Qdrant."""
    upsert_vector(
        asset_id=asset_id,
        vector=current_vector,  # Keep existing vector
        payload={"deleted": True, "deleted_at": datetime.now().isoformat()}
    )
```

**Benefits:**
- Fast (no point deletion)
- Reversible (undelete support)
- Search can filter deleted=False

**Trade-offs:**
- Storage not reclaimed
- Requires filter in all searches

**2. Hard Delete Pattern**
```python
def delete_vector(asset_id: int) -> None:
    """Permanently remove vector from Qdrant."""
    client = get_qdrant_client()
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=PointIdsList(points=[asset_id])
    )
```

**Benefits:**
- Reclaims storage
- Clean separation

**Trade-offs:**
- Irreversible
- Cannot restore without re-embedding

### B. Batch Deletion

**For multiple assets:**
```python
def delete_vectors_batch(asset_ids: list[int]) -> None:
    """Delete multiple vectors efficiently."""
    client = get_qdrant_client()
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=PointIdsList(points=asset_ids)
    )
```

### C. Re-indexing Support

**Update existing vector:**
```python
def reindex_asset(asset_id: int) -> None:
    """Re-generate and update embedding for existing asset."""
    asset = get_asset_by_id_sync(db_session, asset_id)

    # Generate new embedding
    vector = embedding_service.embed_image(asset.path)

    # Upsert overwrites existing point
    upsert_vector(asset_id=asset_id, vector=vector, payload={...})

    # Update indexed_at timestamp
    update_asset_indexed_at_sync(db_session, asset_id)
```

**Use Cases:**
- File content changed
- Model upgraded (new CLIP version)
- Fix corrupted embeddings

### D. Cleanup Jobs

**Orphaned vectors (in Qdrant but not in DB):**
```python
def cleanup_orphaned_vectors() -> int:
    """Remove vectors for deleted assets."""
    # Get all point IDs from Qdrant
    all_points = client.scroll(
        collection_name=settings.qdrant_collection,
        limit=1000,
        with_payload=False,
        with_vectors=False
    )
    qdrant_ids = {point.id for point in all_points[0]}

    # Get all asset IDs from DB
    db_ids = {asset.id for asset in db.query(ImageAsset.id).all()}

    # Delete orphans
    orphaned_ids = qdrant_ids - db_ids
    if orphaned_ids:
        client.delete(
            collection_name=settings.qdrant_collection,
            points_selector=PointIdsList(points=list(orphaned_ids))
        )

    return len(orphaned_ids)
```

### E. Testing Patterns for Vector Management

**Test deletion:**
```python
def test_delete_vector_removes_from_qdrant(qdrant_client):
    # Insert vector
    upsert_vector(asset_id=123, vector=test_vector, payload={"path": "/test.jpg"})

    # Verify exists
    points = qdrant_client.retrieve(collection_name="image_assets", ids=[123])
    assert len(points) == 1

    # Delete
    delete_vector(123)

    # Verify removed
    points = qdrant_client.retrieve(collection_name="image_assets", ids=[123])
    assert len(points) == 0
```

**Test re-indexing:**
```python
def test_reindex_updates_existing_vector(qdrant_client, db_session):
    # Insert initial vector
    upsert_vector(asset_id=123, vector=old_vector, payload={"path": "/test.jpg"})

    # Re-index with new embedding
    reindex_asset(123)

    # Verify vector changed
    points = qdrant_client.retrieve(collection_name="image_assets", ids=[123])
    assert points[0].vector != old_vector  # Vector updated
    assert points[0].payload["path"] == "/test.jpg"  # Payload preserved
```

## 11. Summary

### Strengths
✅ Clean separation of concerns (Qdrant wrapper vs. business logic)
✅ Comprehensive test coverage with in-memory fixtures
✅ Idempotent operations (safe to retry)
✅ Rich metadata capture (evidence system)
✅ Graceful error handling (503 on Qdrant unavailable)
✅ Modern API usage (query_points vs deprecated search)

### Areas for Enhancement
🔧 Add vector deletion/cleanup operations
🔧 Implement category and date range filtering tests
🔧 Add re-indexing support for model upgrades
🔧 Add orphaned vector cleanup job
🔧 Document vector management best practices

### Key Files for Vector Management Implementation

| File | Purpose |
|------|---------|
| `src/image_search_service/vector/qdrant.py` | Add delete/cleanup functions |
| `src/image_search_service/queue/training_jobs.py` | Add re-indexing job |
| `src/image_search_service/api/routes/assets.py` | Add delete endpoint |
| `tests/unit/test_qdrant_wrapper.py` | Add deletion tests |
| `tests/api/test_assets.py` | Add delete endpoint tests |

---

**Research Conducted By:** Claude Code Research Agent
**Documentation Standards:** Structured markdown with code examples
**Next Steps:** Implement vector deletion feature based on recommendations above
