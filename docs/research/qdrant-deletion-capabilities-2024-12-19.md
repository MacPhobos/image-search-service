# Qdrant Integration Research: Vector Deletion Capabilities

**Research Date**: 2024-12-19
**Project**: image-search-service
**Researcher**: Claude Code Research Agent

## Executive Summary

This research analyzes the current Qdrant vector database integration to determine what deletion capabilities would be needed for managing trained image embeddings. The system currently has **no vector deletion functionality** - it only supports upsertion and search operations.

**Key Findings**:
- Point IDs use `asset_id` (integer) directly as the Qdrant point ID
- Payload includes `asset_id` (string), `path`, `created_at`, and `category_id`
- Database has cascade deletes configured for related records
- Qdrant client supports deletion via `PointIdsList` (by IDs) and `FilterSelector` (by payload filters)
- **Critical gap**: No Qdrant deletion occurs when database records are deleted

---

## 1. Qdrant Collection Structure

### Collection Configuration

**Collection Name**: `image_assets` (configurable via `QDRANT_COLLECTION` env var)

**Vector Configuration**:
- **Dimension**: 512 (default, configurable via `EMBEDDING_DIM`)
- **Distance Metric**: COSINE
- **Model**: OpenCLIP ViT-B-32 (configurable)

**Collection Initialization** (`src/image_search_service/vector/qdrant.py:39-59`):
```python
def ensure_collection(embedding_dim: int) -> None:
    """Create collection if it doesn't exist."""
    client = get_qdrant_client()
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    if settings.qdrant_collection not in collection_names:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
        )
```

### Point ID Format

**Point ID**: Uses `asset_id` (integer) directly

**Code Reference** (`src/image_search_service/vector/qdrant.py:61-77`):
```python
def upsert_vector(asset_id: int, vector: list[float], payload: dict[str, str | int]) -> None:
    """Upsert a vector point into Qdrant."""
    client.upsert(
        collection_name=settings.qdrant_collection,
        points=[
            PointStruct(id=asset_id, vector=vector, payload={**payload, "asset_id": str(asset_id)})
        ],
    )
```

**Important**: The point ID is the **integer asset_id**, while payload contains a **string version** of asset_id for filtering.

### Payload Structure

Vectors are stored with the following payload metadata:

| Field | Type | Source | Required | Notes |
|-------|------|--------|----------|-------|
| `asset_id` | string | Asset ID (converted to string) | Yes | Always included, used for filtering |
| `path` | string | ImageAsset.path | Yes | File system path to image |
| `created_at` | string | ImageAsset.created_at (ISO 8601) | Optional | Image creation timestamp |
| `category_id` | integer | TrainingSession.category_id | Optional | Only if session has category |

**Code Reference** (`src/image_search_service/queue/training_jobs.py:286-296`):
```python
# Build payload for Qdrant vector
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

---

## 2. Vector Storage Workflow

### Training Job Flow

**Entry Point**: `POST /api/v1/training/sessions/{session_id}/start`

**Background Job Chain**:
1. **`train_session(session_id)`** - Main orchestration job
   - Discovers assets via `AssetDiscoveryService`
   - Creates `TrainingJob` records for each asset
   - Processes jobs in batches (default: 32 images per batch)

2. **`train_batch(session_id, asset_ids, batch_num)`** - Batch processing
   - Processes multiple assets in parallel
   - Calls `train_single_asset()` for each

3. **`train_single_asset(job_id, asset_id, session_id)`** - Individual asset training
   - Generates embedding using OpenCLIP
   - **Upserts vector to Qdrant** with payload metadata
   - Updates `ImageAsset.indexed_at` timestamp
   - Creates `TrainingEvidence` record with checksum and metadata

**Code Reference** (`src/image_search_service/queue/training_jobs.py:241-296`):
```python
def train_single_asset(job_id: int, asset_id: int, session_id: int) -> dict[str, object]:
    # Get training session to access category_id
    training_session = get_session_by_id_sync(db_session, session_id)
    category_id = training_session.category_id if training_session else None

    # Ensure Qdrant collection exists
    ensure_collection(embedding_service.embedding_dim)

    # Generate embedding
    vector = embedding_service.embed_image(asset.path)

    # Calculate embedding checksum
    embedding_bytes = "".join(str(v) for v in vector).encode()
    checksum = hashlib.sha256(embedding_bytes).hexdigest()

    # Upsert to Qdrant
    payload: dict[str, str | int] = {"path": asset.path}
    if asset.created_at:
        payload["created_at"] = asset.created_at.isoformat()
    if category_id is not None:
        payload["category_id"] = category_id

    upsert_vector(asset_id=asset.id, vector=vector, payload=payload)

    # Update asset indexed_at timestamp
    update_asset_indexed_at_sync(db_session, asset_id)
```

### Metadata Stored in PostgreSQL

**TrainingEvidence Table** (`src/image_search_service/db/models.py:286-327`):

Records comprehensive metadata about each training operation:

| Column | Type | Purpose |
|--------|------|---------|
| `id` | Integer (PK) | Evidence record ID |
| `asset_id` | Integer (FK) | References ImageAsset |
| `session_id` | Integer (FK) | References TrainingSession |
| `model_name` | String | "OpenCLIP" |
| `model_version` | String | e.g., "ViT-B-32" |
| `embedding_checksum` | String(64) | SHA256 hash of embedding vector |
| `device` | String | "cuda" or "cpu" |
| `processing_time_ms` | Integer | Time to generate embedding |
| `error_message` | Text | Error if training failed |
| `metadata_json` | JSON | Additional metadata (image dims, L2 norm, etc.) |
| `created_at` | DateTime | Training timestamp |

**Checksum Verification**: The `embedding_checksum` allows verification that Qdrant vectors match expected values.

---

## 3. Existing Deletion Capabilities

### Database Cascade Deletes

**ImageAsset Cascades** (`src/image_search_service/db/models.py:129-134`):
```python
class ImageAsset(Base):
    # Relationships with cascade delete
    training_jobs: Mapped[list["TrainingJob"]] = relationship(
        "TrainingJob", back_populates="asset", cascade="all, delete-orphan"
    )
    training_evidence: Mapped[list["TrainingEvidence"]] = relationship(
        "TrainingEvidence", back_populates="asset", cascade="all, delete-orphan"
    )
```

**TrainingSession Cascades** (`src/image_search_service/db/models.py:184-192`):
```python
class TrainingSession(Base):
    subdirectories: Mapped[list["TrainingSubdirectory"]] = relationship(
        "TrainingSubdirectory", back_populates="session", cascade="all, delete-orphan"
    )
    jobs: Mapped[list["TrainingJob"]] = relationship(
        "TrainingJob", back_populates="session", cascade="all, delete-orphan"
    )
    evidence: Mapped[list["TrainingEvidence"]] = relationship(
        "TrainingEvidence", back_populates="session", cascade="all, delete-orphan"
    )
```

**Foreign Key Cascades**:
- `TrainingSubdirectory.session_id` → `ondelete="CASCADE"`
- `TrainingJob.session_id` → `ondelete="CASCADE"`
- `TrainingJob.asset_id` → `ondelete="CASCADE"`
- `TrainingEvidence.asset_id` → `ondelete="CASCADE"`
- `TrainingEvidence.session_id` → `ondelete="CASCADE"`
- `TrainingSession.category_id` → `ondelete="SET NULL"` (preserves session if category deleted)

### API Delete Endpoints

**Current DELETE endpoint** (`src/image_search_service/api/routes/training.py:159-179`):
```python
@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: int, db: AsyncSession = Depends(get_db)) -> None:
    """Delete a training session."""
    service = TrainingService()
    deleted = await service.delete_session(db, session_id)

    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
```

**Service Implementation** (`src/image_search_service/services/training_service.py:185-203`):
```python
async def delete_session(self, db: AsyncSession, session_id: int) -> bool:
    """Delete a training session."""
    session = await self.get_session(db, session_id)
    if not session:
        return False

    await db.delete(session)  # Triggers cascade deletes for jobs, evidence, subdirectories
    await db.commit()

    logger.info(f"Deleted training session {session_id}")
    return True
```

**Critical Gap**: This deletes database records but **does NOT delete Qdrant vectors**.

### No Qdrant Deletion Functions

**Current Qdrant wrapper functions** (`src/image_search_service/vector/qdrant.py`):
- `get_qdrant_client()` - Get/create client
- `ensure_collection(embedding_dim)` - Create collection if missing
- `upsert_vector(asset_id, vector, payload)` - Insert/update vector
- `search_vectors(query_vector, limit, offset, filters)` - Search similar vectors
- `ping()` - Health check
- `close_qdrant()` - Close client connection

**Missing**: No `delete_vector()`, `delete_vectors()`, or similar deletion functions.

---

## 4. Qdrant Client Delete Capabilities

### Qdrant Python SDK Delete API

The `qdrant_client` library provides deletion via the `delete()` method with two selector types:

**1. PointIdsList** - Delete by specific point IDs:
```python
from qdrant_client.models import PointIdsList

client.delete(
    collection_name="image_assets",
    points_selector=PointIdsList(
        points=[1, 2, 3, 42, 100]  # List of asset IDs
    )
)
```

**2. FilterSelector** - Delete by payload filter:
```python
from qdrant_client.models import FilterSelector, Filter, FieldCondition, MatchValue

client.delete(
    collection_name="image_assets",
    points_selector=FilterSelector(
        filter=Filter(
            must=[
                FieldCondition(
                    key="category_id",
                    match=MatchValue(value=5)
                )
            ]
        )
    )
)
```

**Code Evidence** (from `.venv/lib/python3.12/site-packages/qdrant_client/http/models/models.py`):
```python
class PointIdsList(BaseModel, extra="forbid"):
    points: List[ExtendedPointId] = Field(..., description="")

class FilterSelector(BaseModel, extra="forbid"):
    filter: "Filter" = Field(..., description="")

PointsSelector = Union[
    PointIdsList,
    FilterSelector,
]
```

### Delete Method Signature

```python
client.delete(
    collection_name: str,
    points_selector: PointsSelector,  # PointIdsList or FilterSelector
    wait: bool = True,  # Wait for operation to complete
    ordering: Optional[WriteOrdering] = None,
    shard_key_selector: Optional[ShardKeySelector] = None,
)
```

---

## 5. Deletion Scenarios and Requirements

### Scenario 1: Delete Single Asset's Vector

**Use Case**: User deletes an image file or removes asset from system

**Current Behavior**:
1. ImageAsset record deleted from PostgreSQL
2. CASCADE deletes TrainingJob and TrainingEvidence records
3. **Vector remains in Qdrant** (orphaned)

**Required Implementation**:
```python
def delete_vector(asset_id: int) -> None:
    """Delete a single vector from Qdrant by asset ID."""
    client = get_qdrant_client()
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=PointIdsList(points=[asset_id])
    )
```

**Integration Point**: Add to `ImageAsset` deletion flow or create separate cleanup endpoint.

---

### Scenario 2: Delete All Vectors for a Training Session

**Use Case**: User deletes an entire training session

**Current Behavior**:
1. TrainingSession deleted (CASCADE deletes jobs, evidence, subdirectories)
2. ImageAsset records **remain** (not cascade deleted)
3. **All vectors remain in Qdrant**

**Challenge**: TrainingSession is not directly linked to vectors in Qdrant payload.

**Option 2A - Delete by Asset IDs**:
```python
async def delete_session_vectors(db: AsyncSession, session_id: int) -> int:
    """Delete all vectors associated with a training session."""
    # Get all asset IDs from training jobs
    query = select(TrainingJob.asset_id).where(TrainingJob.session_id == session_id)
    result = await db.execute(query)
    asset_ids = list(result.scalars().all())

    if not asset_ids:
        return 0

    # Delete vectors from Qdrant
    client = get_qdrant_client()
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=PointIdsList(points=asset_ids)
    )

    return len(asset_ids)
```

**Option 2B - Track session_id in payload** (requires migration):
- Add `session_id` to vector payload during training
- Delete by filter: `session_id == X`
- **Limitation**: Asset can belong to multiple sessions (re-training)

**Recommendation**: Use Option 2A (delete by asset IDs collected from jobs).

---

### Scenario 3: Delete All Vectors for a Category

**Use Case**: User deletes a category or wants to retrain all category images

**Current Behavior**:
1. Category deleted (TrainingSession.category_id set to NULL via `ondelete="SET NULL"`)
2. **Vectors with `category_id` remain in Qdrant**

**Required Implementation**:
```python
def delete_category_vectors(category_id: int) -> int:
    """Delete all vectors with a specific category_id."""
    client = get_qdrant_client()

    # Use FilterSelector to delete by category_id
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=FilterSelector(
            filter=Filter(
                must=[
                    FieldCondition(
                        key="category_id",
                        match=MatchValue(value=category_id)
                    )
                ]
            )
        )
    )

    # Return count of deleted vectors (requires separate count query)
    # Qdrant delete() doesn't return count
```

**Limitation**: Qdrant `delete()` doesn't return count of deleted points. Would need to:
1. Count points with filter before deletion
2. Or accept that count is unknown

---

### Scenario 4: Delete All Vectors (Full Reset)

**Use Case**: Complete system reset, test cleanup, or collection rebuild

**Option 4A - Delete Collection**:
```python
def delete_collection() -> None:
    """Delete entire Qdrant collection."""
    client = get_qdrant_client()
    client.delete_collection(collection_name=settings.qdrant_collection)
```

**Option 4B - Delete All Points**:
```python
def delete_all_vectors() -> None:
    """Delete all points from collection (keeps collection structure)."""
    client = get_qdrant_client()

    # Scroll through all points to get IDs
    offset = None
    all_ids = []

    while True:
        result = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=1000,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )

        points, next_offset = result
        all_ids.extend([p.id for p in points])

        if next_offset is None:
            break
        offset = next_offset

    # Delete all points
    if all_ids:
        client.delete(
            collection_name=settings.qdrant_collection,
            points_selector=PointIdsList(points=all_ids)
        )
```

**Recommendation**: Use Option 4A (delete collection) for full reset - simpler and faster.

---

### Scenario 5: Orphaned Vector Cleanup

**Use Case**: Find and delete vectors that don't have corresponding ImageAsset records

**Implementation**:
```python
async def cleanup_orphaned_vectors(db: AsyncSession) -> int:
    """Delete vectors from Qdrant that don't have corresponding ImageAsset records."""
    # Get all asset IDs from database
    query = select(ImageAsset.id)
    result = await db.execute(query)
    valid_asset_ids = set(result.scalars().all())

    # Scroll through all Qdrant vectors
    client = get_qdrant_client()
    offset = None
    orphaned_ids = []

    while True:
        result = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=1000,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )

        points, next_offset = result

        # Find points with IDs not in database
        for point in points:
            if point.id not in valid_asset_ids:
                orphaned_ids.append(point.id)

        if next_offset is None:
            break
        offset = next_offset

    # Delete orphaned vectors
    if orphaned_ids:
        client.delete(
            collection_name=settings.qdrant_collection,
            points_selector=PointIdsList(points=orphaned_ids)
        )

    return len(orphaned_ids)
```

**Use Case**: Run as periodic maintenance job or manual cleanup operation.

---

## 6. Related Data Relationships

### Data Model Diagram

```
Category (1) ──┐
               │ SET NULL on delete
               ▼
TrainingSession (1) ──┬── CASCADE ──> TrainingSubdirectory (N)
                      │
                      ├── CASCADE ──> TrainingJob (N) ──┐
                      │                                 │ CASCADE
                      └── CASCADE ──> TrainingEvidence (N)
                                                        │
                                                        │
ImageAsset (1) ────────────────────────────────────────┴─ CASCADE

Qdrant Vector (asset_id as point ID) ──[NO LINK]──> ImageAsset
```

### Deletion Impact Matrix

| Action | ImageAsset | TrainingJob | TrainingEvidence | Qdrant Vector | Impact |
|--------|------------|-------------|------------------|---------------|--------|
| **Delete ImageAsset** | ✓ Deleted | ✓ CASCADE | ✓ CASCADE | ✗ **Orphaned** | Vector remains with invalid asset_id |
| **Delete TrainingSession** | ✗ Remains | ✓ CASCADE | ✓ CASCADE | ✗ **Remains** | Vectors still searchable |
| **Delete Category** | ✗ Remains | ✗ Remains | ✗ Remains | ✗ **Remains** | category_id set to NULL in session |
| **Delete TrainingJob** | ✗ Remains | ✓ Deleted | ✗ Remains | ✗ **Remains** | Evidence still exists |

**Key Insight**: Qdrant vectors are **never automatically deleted** by current database operations.

---

## 7. Proposed Deletion Functions

### Core Deletion Functions (Add to `src/image_search_service/vector/qdrant.py`)

```python
def delete_vector(asset_id: int) -> None:
    """Delete a single vector from Qdrant by asset ID.

    Args:
        asset_id: Asset ID (used as point ID)
    """
    client = get_qdrant_client()
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=PointIdsList(points=[asset_id])
    )
    logger.info(f"Deleted vector for asset {asset_id} from Qdrant")


def delete_vectors(asset_ids: list[int]) -> None:
    """Delete multiple vectors from Qdrant by asset IDs.

    Args:
        asset_ids: List of asset IDs to delete
    """
    if not asset_ids:
        return

    client = get_qdrant_client()
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=PointIdsList(points=asset_ids)
    )
    logger.info(f"Deleted {len(asset_ids)} vectors from Qdrant")


def delete_vectors_by_category(category_id: int) -> None:
    """Delete all vectors with a specific category_id.

    Args:
        category_id: Category ID to filter by
    """
    client = get_qdrant_client()
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=FilterSelector(
            filter=Filter(
                must=[
                    FieldCondition(
                        key="category_id",
                        match=MatchValue(value=category_id)
                    )
                ]
            )
        )
    )
    logger.info(f"Deleted all vectors for category {category_id} from Qdrant")


def delete_collection() -> None:
    """Delete the entire Qdrant collection.

    Warning: This is destructive and removes all vectors.
    """
    client = get_qdrant_client()
    client.delete_collection(collection_name=settings.qdrant_collection)
    logger.warning(f"Deleted Qdrant collection '{settings.qdrant_collection}'")


async def cleanup_orphaned_vectors(db: AsyncSession) -> int:
    """Delete vectors from Qdrant that don't have corresponding ImageAsset records.

    Args:
        db: Database session

    Returns:
        Number of orphaned vectors deleted
    """
    from sqlalchemy import select
    from image_search_service.db.models import ImageAsset

    # Get all valid asset IDs from database
    query = select(ImageAsset.id)
    result = await db.execute(query)
    valid_asset_ids = set(result.scalars().all())

    # Scroll through Qdrant to find orphaned vectors
    client = get_qdrant_client()
    offset = None
    orphaned_ids = []

    while True:
        result = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=1000,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )

        points, next_offset = result

        for point in points:
            if point.id not in valid_asset_ids:
                orphaned_ids.append(point.id)

        if next_offset is None:
            break
        offset = next_offset

    # Delete orphaned vectors
    if orphaned_ids:
        delete_vectors(orphaned_ids)
        logger.info(f"Cleaned up {len(orphaned_ids)} orphaned vectors from Qdrant")

    return len(orphaned_ids)
```

### Service Layer Integration

**Add to `src/image_search_service/services/training_service.py`**:

```python
async def delete_session(self, db: AsyncSession, session_id: int, delete_vectors: bool = True) -> bool:
    """Delete a training session and optionally its vectors.

    Args:
        db: Database session
        session_id: Session ID
        delete_vectors: If True, also delete vectors from Qdrant

    Returns:
        True if deleted, False if not found
    """
    session = await self.get_session(db, session_id)
    if not session:
        return False

    # Optionally delete vectors from Qdrant before deleting session
    if delete_vectors:
        from image_search_service.vector.qdrant import delete_vectors
        from sqlalchemy import select
        from image_search_service.db.models import TrainingJob

        # Get all asset IDs from training jobs
        query = select(TrainingJob.asset_id).where(TrainingJob.session_id == session_id)
        result = await db.execute(query)
        asset_ids = list(result.scalars().all())

        if asset_ids:
            delete_vectors(asset_ids)
            logger.info(f"Deleted {len(asset_ids)} vectors for session {session_id}")

    # Delete session (cascades to jobs, evidence, subdirectories)
    await db.delete(session)
    await db.commit()

    logger.info(f"Deleted training session {session_id}")
    return True
```

---

## 8. API Endpoint Recommendations

### New Deletion Endpoints

**1. Delete Vectors for Training Session**:
```python
@router.delete("/sessions/{session_id}/vectors", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session_vectors(
    session_id: int,
    db: AsyncSession = Depends(get_db)
) -> None:
    """Delete all Qdrant vectors associated with a training session.

    Does not delete database records - only removes vectors from Qdrant.
    """
```

**2. Cleanup Orphaned Vectors**:
```python
@router.post("/vectors/cleanup", response_model=dict)
async def cleanup_orphaned_vectors(
    db: AsyncSession = Depends(get_db)
) -> dict[str, int]:
    """Find and delete orphaned vectors from Qdrant.

    Returns:
        Dictionary with count of deleted vectors
    """
```

**3. Delete Collection (Admin)**:
```python
@router.delete("/vectors/reset", status_code=status.HTTP_204_NO_CONTENT)
async def reset_qdrant_collection(
    confirm: bool = Query(..., description="Must be true to confirm deletion")
) -> None:
    """Delete entire Qdrant collection (destructive operation).

    Requires explicit confirmation parameter.
    """
```

**4. Update Session Delete Endpoint**:
```python
@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: int,
    delete_vectors: bool = Query(True, description="Also delete vectors from Qdrant"),
    db: AsyncSession = Depends(get_db)
) -> None:
    """Delete a training session and optionally its vectors."""
```

---

## 9. Testing Requirements

### Unit Tests (Add to `tests/unit/test_qdrant_wrapper.py`)

```python
def test_delete_vector_removes_point(qdrant_client: QdrantClient):
    """Test that delete_vector removes a single point."""
    # Insert test vector
    upsert_vector(asset_id=123, vector=[...], payload={"path": "/test.jpg"})

    # Delete vector
    delete_vector(123)

    # Verify deletion
    points = qdrant_client.retrieve(collection_name=settings.qdrant_collection, ids=[123])
    assert len(points) == 0


def test_delete_vectors_removes_multiple_points(qdrant_client: QdrantClient):
    """Test that delete_vectors removes multiple points."""
    # Insert test vectors
    for i in [1, 2, 3]:
        upsert_vector(asset_id=i, vector=[...], payload={"path": f"/test{i}.jpg"})

    # Delete vectors
    delete_vectors([1, 2, 3])

    # Verify deletion
    points = qdrant_client.retrieve(collection_name=settings.qdrant_collection, ids=[1, 2, 3])
    assert len(points) == 0


def test_delete_vectors_by_category_filters_correctly(qdrant_client: QdrantClient):
    """Test that delete_vectors_by_category only deletes matching vectors."""
    # Insert vectors with different categories
    upsert_vector(asset_id=1, vector=[...], payload={"path": "/test1.jpg", "category_id": 5})
    upsert_vector(asset_id=2, vector=[...], payload={"path": "/test2.jpg", "category_id": 10})

    # Delete only category 5
    delete_vectors_by_category(5)

    # Verify category 5 deleted, category 10 remains
    points1 = qdrant_client.retrieve(collection_name=settings.qdrant_collection, ids=[1])
    points2 = qdrant_client.retrieve(collection_name=settings.qdrant_collection, ids=[2])

    assert len(points1) == 0
    assert len(points2) == 1
```

### Integration Tests

**Test Session Deletion with Vectors**:
```python
async def test_delete_session_with_vectors(db: AsyncSession, qdrant_client: QdrantClient):
    """Test that deleting session optionally deletes vectors."""
    # Create session and train assets
    service = TrainingService()
    session = await service.create_session(db, ...)
    # ... train assets, creating vectors ...

    # Delete session with vectors
    deleted = await service.delete_session(db, session.id, delete_vectors=True)

    # Verify database records deleted
    assert deleted is True

    # Verify Qdrant vectors deleted
    points = qdrant_client.retrieve(collection_name=settings.qdrant_collection, ids=[...])
    assert len(points) == 0
```

---

## 10. Migration and Rollout Strategy

### Phase 1: Add Core Functions (Non-Breaking)
1. Add deletion functions to `qdrant.py`
2. Add unit tests for deletion functions
3. Add orphaned vector cleanup endpoint
4. Deploy and monitor

### Phase 2: Integrate with Existing Flows (Breaking Change)
1. Update `delete_session()` to include `delete_vectors` parameter (default: True)
2. Update API endpoint to accept `delete_vectors` query param
3. Update API contract documentation
4. Communicate breaking change to UI team
5. Deploy backend
6. Update UI to handle new parameter

### Phase 3: Maintenance and Cleanup
1. Run orphaned vector cleanup on production
2. Add periodic cleanup job (daily or weekly)
3. Monitor vector count vs. database asset count
4. Document vector deletion procedures in ops runbook

---

## 11. Risks and Mitigations

### Risk 1: Accidental Vector Deletion

**Risk**: Deleting session or assets accidentally removes trained vectors that took hours to generate.

**Mitigation**:
- Add confirmation parameter to destructive endpoints (`confirm=true`)
- Implement soft-delete for ImageAsset (add `deleted_at` column)
- Provide vector backup/export before deletion
- Log all deletions with asset IDs and timestamps

### Risk 2: Orphaned Vectors Accumulation

**Risk**: Database records deleted without cleaning Qdrant, leading to stale vectors.

**Mitigation**:
- Make `delete_vectors=True` the default behavior
- Periodic orphaned vector cleanup job
- Monitoring/alerting for vector count vs. asset count divergence
- Dashboard showing orphaned vector count

### Risk 3: Performance Impact of Large Deletions

**Risk**: Deleting thousands of vectors in one operation could timeout or impact search performance.

**Mitigation**:
- Batch deletions in chunks (e.g., 1000 vectors per call)
- Use background jobs for large deletions (>1000 vectors)
- Monitor Qdrant performance during deletions
- Rate limit deletion operations

### Risk 4: Category Filter Deletion Accuracy

**Risk**: `delete_vectors_by_category()` might delete wrong vectors if payload isn't properly set.

**Mitigation**:
- Add count preview before deletion (show how many vectors match filter)
- Require explicit confirmation with count acknowledgment
- Log all deleted point IDs for audit trail
- Provide rollback via backup/restore if needed

---

## 12. Recommendations

### Immediate Actions (High Priority)

1. **Add Core Deletion Functions**
   - Implement `delete_vector()`, `delete_vectors()`, and `delete_collection()` in `qdrant.py`
   - Add unit tests for all deletion functions
   - Deploy to staging for testing

2. **Integrate with Session Deletion**
   - Update `TrainingService.delete_session()` to optionally delete vectors
   - Make `delete_vectors=True` the default behavior
   - Update API endpoint with `delete_vectors` query parameter

3. **Add Orphaned Vector Cleanup**
   - Implement `cleanup_orphaned_vectors()` async function
   - Add admin endpoint: `POST /api/v1/training/vectors/cleanup`
   - Run initial cleanup on production to establish baseline

### Medium Priority

4. **Add Vector Deletion Endpoints**
   - `DELETE /api/v1/training/sessions/{session_id}/vectors` - Delete session vectors only
   - `DELETE /api/v1/training/vectors/reset` - Full collection reset (with confirmation)

5. **Monitoring and Metrics**
   - Add Prometheus metrics: `qdrant_vector_count`, `database_asset_count`, `orphaned_vector_count`
   - Alert if orphaned vector count exceeds threshold (e.g., >100)
   - Dashboard showing vector count trends

### Low Priority

6. **Category-Based Deletion**
   - Add `delete_vectors_by_category(category_id)` function
   - Integrate with category deletion flow
   - Add confirmation UI with count preview

7. **Backup and Restore**
   - Implement vector export functionality
   - Add snapshot creation before destructive operations
   - Document vector restore procedures

---

## Appendix A: Code Locations

| Component | File Path | Lines |
|-----------|-----------|-------|
| Qdrant client wrapper | `src/image_search_service/vector/qdrant.py` | 1-169 |
| Vector upsert | `src/image_search_service/vector/qdrant.py` | 61-77 |
| Training job (vector creation) | `src/image_search_service/queue/training_jobs.py` | 209-354 |
| Training service (session management) | `src/image_search_service/services/training_service.py` | 1-806 |
| Database models | `src/image_search_service/db/models.py` | 1-328 |
| Training API endpoints | `src/image_search_service/api/routes/training.py` | 1-739 |
| Qdrant configuration | `src/image_search_service/core/config.py` | 25-44 |
| Qdrant tests | `tests/unit/test_qdrant_wrapper.py` | 1-208 |

---

## Appendix B: External References

- **Qdrant Python Client Documentation**: https://qdrant.tech/documentation/frameworks/python/
- **Qdrant Delete API**: https://qdrant.tech/documentation/concepts/points/#delete-points
- **OpenCLIP Model**: https://github.com/mlfoundations/open_clip
- **SQLAlchemy Cascade Deletes**: https://docs.sqlalchemy.org/en/20/orm/cascades.html

---

**End of Research Report**
