# Face Assignment 404 Bug Analysis

**Date**: 2025-12-25
**Investigator**: Claude (Research Agent)
**Status**: Root Cause Identified

## Bug Summary

When assigning a face to an existing person via `POST /api/v1/faces/faces/{face_id}/assign`, the API returns a 404 error from Qdrant:

```json
{
    "detail": "Failed to assign face: Unexpected Response: 404 (Not Found)\nRaw response content:\nb'{\"status\":{\"error\":\"Not found: No point with id eb39509d-3d1b-4267-bdac-0c86bbfd3c2a found\"},\"time\":0.002607214}'"
}
```

## Root Cause

**Critical Bug**: Database-Qdrant synchronization failure due to improper transaction ordering in face detection pipeline.

### Code Flow Analysis

#### 1. Face Detection Flow (`src/image_search_service/faces/service.py`)

```python
# Lines 106-112
# Commit DB changes
self.db.commit()  # ← DB COMMITTED FIRST

# Batch upsert to Qdrant
if qdrant_points:
    self.qdrant.upsert_faces_batch(qdrant_points)  # ← QDRANT SECOND
    logger.info(f"Stored {len(qdrant_points)} new faces for asset {asset.id}")
```

**Problem**:
- Database transaction commits BEFORE Qdrant upsert
- No error handling or rollback mechanism
- If Qdrant upsert fails (network error, service down, etc.), database has record but Qdrant doesn't have the point

#### 2. Face Assignment Flow (`src/image_search_service/api/routes/faces.py`)

```python
# Lines 845-905
@router.post("/faces/{face_id}/assign", response_model=AssignFaceResponse)
async def assign_face_to_person(
    face_id: UUID,
    request: AssignFaceRequest,
    db: AsyncSession = Depends(get_db),
) -> AssignFaceResponse:
    # Get face instance from DB
    face = await db.get(FaceInstance, face_id)  # ← FACE EXISTS IN DB

    # ...

    # Update Qdrant payload with new person_id
    qdrant = get_face_qdrant_client()
    qdrant.update_person_ids([face.qdrant_point_id], person.id)  # ← 404 ERROR HERE
```

**Problem**:
- Code assumes if face exists in DB, it also exists in Qdrant
- No validation that Qdrant point actually exists before attempting update
- `update_person_ids` calls Qdrant's `set_payload` which fails with 404 if point doesn't exist

#### 3. Qdrant Update Flow (`src/image_search_service/vector/face_qdrant.py`)

```python
# Lines 344-377
def update_person_ids(
    self,
    point_ids: list[uuid.UUID],
    person_id: uuid.UUID | None,
) -> None:
    # ...
    self.client.set_payload(
        collection_name=FACE_COLLECTION_NAME,
        payload={"person_id": str(person_id)},
        points=[str(point_id) for point_id in point_ids],  # ← FAILS IF POINT MISSING
    )
```

**Problem**:
- No check if point exists before attempting update
- Qdrant returns 404 for missing points
- Error propagates up to API endpoint

## Failure Scenarios

### Scenario 1: Qdrant Service Interruption
1. Face detection runs, DB commits successfully
2. Qdrant service is down/unreachable
3. `upsert_faces_batch()` fails (exception or silent failure)
4. FaceInstance exists in PostgreSQL with `qdrant_point_id = UUID`
5. Qdrant has no point with that UUID
6. Later assignment attempt → 404 error

### Scenario 2: Qdrant Collection Reset
1. Faces detected and stored in both DB and Qdrant
2. Admin runs `qdrant.reset_collection()` (line 714-746 in face_qdrant.py)
3. Qdrant collection is deleted and recreated empty
4. Database still has all FaceInstance records with qdrant_point_id values
5. Assignment attempt → 404 error

### Scenario 3: Partial Batch Failure
1. Batch of 100 faces being processed
2. First 50 upsert to Qdrant successfully
3. Network error or rate limit hit
4. Remaining 50 fail to upsert to Qdrant
5. All 100 exist in database
6. Last 50 faces → 404 on assignment

## Data Inconsistency Evidence

The error message shows a specific UUID: `eb39509d-3d1b-4267-bdac-0c86bbfd3c2a`

This UUID is stored in `FaceInstance.qdrant_point_id` in PostgreSQL, but Qdrant has no corresponding point.

## Impact Assessment

**Severity**: High
**Affected Operations**:
- Face assignment to persons (POST /faces/{face_id}/assign)
- Cluster labeling (may have similar issues)
- Person merging (may have similar issues)
- Any operation that updates Qdrant payloads for existing faces

**Data Integrity**:
- Database and Qdrant are out of sync
- Unknown number of orphaned FaceInstance records
- Cannot assign affected faces to persons without manual intervention

## Recommended Fixes

### Fix 1: Immediate - Add Defensive Check in Assignment (Quick Fix)

**File**: `src/image_search_service/api/routes/faces.py`
**Location**: Lines 845-905 in `assign_face_to_person`

```python
@router.post("/faces/{face_id}/assign", response_model=AssignFaceResponse)
async def assign_face_to_person(
    face_id: UUID,
    request: AssignFaceRequest,
    db: AsyncSession = Depends(get_db),
) -> AssignFaceResponse:
    from image_search_service.vector.face_qdrant import get_face_qdrant_client

    # Get face instance from DB
    face = await db.get(FaceInstance, face_id)
    if not face:
        raise HTTPException(status_code=404, detail=f"Face {face_id} not found")

    # Get target person from DB
    person = await db.get(Person, request.person_id)
    if not person:
        raise HTTPException(
            status_code=404, detail=f"Person {request.person_id} not found"
        )

    # NEW: Verify Qdrant point exists before attempting update
    qdrant = get_face_qdrant_client()
    try:
        # Retrieve the point to verify it exists
        qdrant.client.retrieve(
            collection_name=FACE_COLLECTION_NAME,
            ids=[str(face.qdrant_point_id)],
        )
    except Exception as e:
        logger.error(f"Face {face_id} missing from Qdrant: {e}")
        raise HTTPException(
            status_code=409,
            detail=f"Face data is incomplete. Point {face.qdrant_point_id} not found in Qdrant. "
                   f"Re-run face detection for asset {face.asset_id} to fix.",
        )

    # Track previous assignment for audit log
    previous_person_id = face.person_id

    # Update face.person_id to new person
    face.person_id = person.id

    try:
        # Update Qdrant payload with new person_id
        qdrant.update_person_ids([face.qdrant_point_id], person.id)

        # ... rest of existing code
```

**Benefits**:
- Prevents 500 error with better error message
- Informs user of data inconsistency
- Suggests remediation (re-run face detection)
- No changes to core detection flow

**Drawbacks**:
- Extra Qdrant API call (retrieve before update)
- Doesn't prevent the root cause
- Reactive rather than proactive

### Fix 2: Proper Transaction Ordering (Correct Fix)

**File**: `src/image_search_service/faces/service.py`
**Location**: Lines 106-114 in `process_asset`

```python
# BEFORE (BUGGY):
# Commit DB changes
self.db.commit()

# Batch upsert to Qdrant
if qdrant_points:
    self.qdrant.upsert_faces_batch(qdrant_points)
    logger.info(f"Stored {len(qdrant_points)} new faces for asset {asset.id}")

return face_instances


# AFTER (FIXED):
# Batch upsert to Qdrant FIRST (before commit)
if qdrant_points:
    try:
        self.qdrant.upsert_faces_batch(qdrant_points)
        logger.info(f"Stored {len(qdrant_points)} new faces for asset {asset.id}")
    except Exception as e:
        # Rollback DB if Qdrant fails
        self.db.rollback()
        logger.error(f"Failed to upsert faces to Qdrant for asset {asset.id}: {e}")
        raise RuntimeError(f"Face detection failed: Qdrant upsert error: {e}") from e

# Commit DB changes ONLY if Qdrant succeeded
self.db.commit()

return face_instances
```

**Benefits**:
- Prevents database-Qdrant desync at the source
- Proper error handling with rollback
- Maintains atomicity: both succeed or both fail
- No orphaned database records

**Drawbacks**:
- Requires testing to ensure no side effects
- Doesn't fix existing broken data (see Fix 3)

### Fix 3: Data Repair Utility (Cleanup)

Create admin endpoint or CLI command to detect and repair orphaned faces:

```python
async def repair_orphaned_faces(db: AsyncSession) -> dict[str, int]:
    """Find and fix FaceInstance records missing from Qdrant."""
    from image_search_service.vector.face_qdrant import get_face_qdrant_client

    qdrant = get_face_qdrant_client()
    orphaned_count = 0
    repaired_count = 0
    failed_count = 0

    # Get all face instances
    query = select(FaceInstance)
    result = await db.execute(query)
    faces = result.scalars().all()

    for face in faces:
        # Check if point exists in Qdrant
        try:
            qdrant.client.retrieve(
                collection_name=FACE_COLLECTION_NAME,
                ids=[str(face.qdrant_point_id)],
            )
        except Exception:
            # Point missing - this is an orphan
            orphaned_count += 1
            logger.warning(f"Orphaned face: {face.id} (point {face.qdrant_point_id} missing)")

            # Try to recreate the point if we can get the embedding
            # (This requires re-running face detection or having embeddings cached)
            # For now, just log the issue

    return {
        "total_faces": len(faces),
        "orphaned": orphaned_count,
        "repaired": repaired_count,
        "failed": failed_count,
    }
```

**Benefits**:
- Identifies extent of data corruption
- Can be used to repair existing broken data
- Useful for production diagnostics

**Drawbacks**:
- Requires re-running face detection to regenerate embeddings
- Slow for large databases (requires checking each face)

## Testing Recommendations

### Test Case 1: Qdrant Failure During Detection
```python
def test_face_detection_qdrant_failure():
    """Ensure DB rollback when Qdrant fails."""
    # Mock Qdrant to raise exception
    # Run face detection
    # Assert: No FaceInstance records created
    # Assert: Proper error message returned
```

### Test Case 2: Assignment of Orphaned Face
```python
def test_assign_orphaned_face():
    """Ensure proper error when face missing from Qdrant."""
    # Create FaceInstance in DB without Qdrant point
    # Attempt assignment
    # Assert: 409 Conflict with helpful error message
    # Assert: No partial updates
```

### Test Case 3: Recovery After Qdrant Reset
```python
def test_recovery_after_qdrant_reset():
    """Test repair utility after collection reset."""
    # Detect faces (DB + Qdrant)
    # Reset Qdrant collection
    # Run repair utility
    # Assert: All faces repaired or marked for re-detection
```

## Related Code Locations

**Primary Bug Location**:
- `src/image_search_service/faces/service.py:106-114` - Transaction ordering issue

**Affected Endpoints**:
- `POST /api/v1/faces/faces/{face_id}/assign` (Line 845-905 in routes/faces.py)
- `POST /api/v1/faces/clusters/{cluster_id}/label` (Line 157-224 in routes/faces.py)
- `POST /api/v1/faces/persons/{person_id}/merge` (Line 465-517 in routes/faces.py)
- `POST /api/v1/faces/persons/{person_id}/photos/bulk-move` (Line 611-761 in routes/faces.py)
- `POST /api/v1/faces/persons/{person_id}/photos/bulk-remove` (Line 519-609 in routes/faces.py)

**Qdrant Update Methods**:
- `src/image_search_service/vector/face_qdrant.py:285-316` - update_payload
- `src/image_search_service/vector/face_qdrant.py:344-377` - update_person_ids
- `src/image_search_service/vector/face_qdrant.py:318-342` - update_cluster_ids

## Migration Path

1. **Immediate** (Today):
   - Deploy Fix 1 (defensive check in assignment endpoint)
   - Document known issue in API docs

2. **Short-term** (This week):
   - Implement Fix 2 (proper transaction ordering)
   - Add comprehensive tests
   - Deploy to production with monitoring

3. **Medium-term** (This month):
   - Implement Fix 3 (data repair utility)
   - Run repair on production database
   - Add monitoring for DB-Qdrant sync health

4. **Long-term** (Ongoing):
   - Add background job to detect orphaned records
   - Implement Qdrant health checks
   - Add retry logic with exponential backoff
   - Consider distributed transaction patterns (saga, 2PC)

## Conclusion

The 404 error occurs because face detection commits database records before confirming Qdrant storage succeeds. Any failure in the Qdrant upsert step leaves orphaned FaceInstance records that cannot be assigned to persons.

**Recommended Action**: Implement Fix 2 (transaction ordering) immediately, followed by Fix 3 (data repair) to clean up existing broken records. Fix 1 can serve as a temporary band-aid but doesn't address the root cause.
