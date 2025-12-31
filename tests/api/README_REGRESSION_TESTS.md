# Regression Tests for personName Bug

## Bug Summary

**Fixed Bug**: API endpoints were returning `personName: null` even when `personId` was set.

**Root Cause**: Queries were not JOINing with the `Person` table to fetch person names.

**Affected Endpoints**:
- `GET /api/v1/faces/clusters/{cluster_id}`
- `GET /api/v1/faces/assets/{asset_id}`

**Fix**: Both endpoints now use `.outerjoin(Person, FaceInstance.person_id == Person.id)` to fetch person names alongside face data.

## Test Coverage

### File: `test_person_name_regression.py`

Comprehensive regression tests to prevent this bug from reoccurring.

#### Test Class: `TestGetClusterPersonName`

Tests for `GET /api/v1/faces/clusters/{cluster_id}`:

1. **`test_get_cluster_returns_person_name_when_assigned`**
   - Creates a face assigned to a person
   - Verifies both cluster-level and face-level `personName` are populated
   - **Critical assertion**: `personName == "Test Person"` (not null)

2. **`test_get_cluster_returns_null_person_when_unassigned`**
   - Creates a face with no person assignment
   - Verifies `personId` and `personName` are both null

3. **`test_get_cluster_with_multiple_faces_same_person`**
   - Creates 3 faces all assigned to the same person
   - Verifies all faces have the same `personName`

#### Test Class: `TestGetFacesForAssetPersonName`

Tests for `GET /api/v1/faces/assets/{asset_id}`:

1. **`test_get_faces_for_asset_returns_person_name_when_assigned`**
   - Creates a face assigned to a person
   - Verifies `personName` is populated
   - **Critical assertion**: `personName == "Test Person"` (not null)

2. **`test_get_faces_for_asset_returns_null_person_when_unassigned`**
   - Creates a face with no person assignment
   - Verifies `personId` and `personName` are both null

3. **`test_get_faces_for_asset_with_multiple_faces_different_persons`**
   - Creates 3 faces: one for Alice, one for Bob, one unassigned
   - Verifies each face has the correct `personName`
   - Tests that multiple persons in one asset work correctly

### File: `test_get_faces_for_asset.py`

Pre-existing tests (also cover the bug):

1. **`test_get_faces_for_asset_with_person_name`**
   - Similar to regression test, verifies personName is "Chantal"

2. **`test_get_faces_for_asset_without_person`**
   - Verifies personName is null when no person assigned

## Running the Tests

```bash
# Run all regression tests
uv run pytest tests/api/test_person_name_regression.py -v

# Run all personName related tests
uv run pytest tests/api/test_person_name_regression.py tests/api/test_get_faces_for_asset.py -v

# Run full test suite
make test
```

## What the Tests Verify

### The Critical Regression (MUST NOT FAIL)

```python
# This was the bug:
assert face_response["personId"] == "123e4567-e89b-12d3-a456-426614174000"
assert face_response["personName"] == None  # ❌ BUG: Should be "Test Person"

# Fixed behavior:
assert face_response["personId"] == "123e4567-e89b-12d3-a456-426614174000"
assert face_response["personName"] == "Test Person"  # ✅ CORRECT
```

### Edge Cases Covered

- ✅ Face assigned to person → personName populated
- ✅ Face not assigned → personName is null
- ✅ Multiple faces in same cluster/asset → each has correct personName
- ✅ Multiple different persons in same asset → each face has correct personName
- ✅ Mixed assigned/unassigned faces → correct personName per face

## Implementation Pattern

Both endpoints follow this pattern:

```python
query = (
    select(FaceInstance, Person.name)
    .outerjoin(Person, FaceInstance.person_id == Person.id)
    .where(...)
)
result = await db.execute(query)
faces_data = result.all()

# Convert to response using helper
return [_face_to_response(face, person_name) for face, person_name in faces_data]
```

**Key points**:
- Use `.outerjoin()` (not `.join()`) to include faces without persons
- Select `Person.name` alongside `FaceInstance`
- Pass `person_name` to `_face_to_response()` helper

## Maintenance

When modifying face-related endpoints:
1. Always JOIN with Person table if returning personName
2. Run regression tests before committing
3. Add new test cases for new edge cases
4. Never delete these regression tests without replacement coverage
