# People Filter Implementation Plan

**Date:** 2025-12-29
**Status:** Draft - Pending Review
**Detailed Analysis:** [people-filter-analysis-2025-12-29.md](./people-filter-analysis-2025-12-29.md)

---

## Executive Summary

The People Filter UI exists in the frontend but **does not work**. The backend completely ignores the `personId` filter parameter. This plan outlines the implementation required to make it functional.

### Expected Behavior

| Scenario | Expected Result |
|----------|-----------------|
| People Filter specified | Search results only include images with faces labeled to the specified person |
| No People Filter | Full image search (current behavior) |

### Current State

| Component | Status | Issue |
|-----------|--------|-------|
| Frontend UI | Implemented | Multi-select UI but only sends first person ID |
| API Contract | Documented | `personId` parameter defined |
| Backend Route | Passes filter | Filter forwarded to Qdrant search |
| Qdrant Search | **NOT IMPLEMENTED** | `personId` filter completely ignored |
| Qdrant Payload | **MISSING DATA** | No `person_ids` field stored |

---

## Implementation Approach

### Approach Decision: Payload-Based vs Post-Filter

| Approach | Pros | Cons | Recommendation |
|----------|------|------|----------------|
| **Payload-Based** | Fast queries, scales well | Requires payload updates on face changes | **Recommended** |
| Post-Filter | Simpler to implement | Slower queries, pagination issues | Not recommended |

**Decision:** Use Payload-Based approach for performance at scale.

---

## Phase 1: Backend Core Implementation

**Effort:** 4-6 hours
**Priority:** Critical

### Task 1.1: Update Qdrant Vector Payloads

**File:** `image-search-service/src/image_search_service/vector/qdrant.py`

Add `person_ids` to payload structure during vector upsert:

```python
# Current payload:
{
    "asset_id": "123",
    "path": "/photos/beach.jpg",
    "created_at": "2024-12-19T10:00:00Z",
    "category_id": 5
}

# Required payload:
{
    "asset_id": "123",
    "path": "/photos/beach.jpg",
    "created_at": "2024-12-19T10:00:00Z",
    "category_id": 5,
    "person_ids": ["person-uuid-1", "person-uuid-2"]  # NEW
}
```

**Acceptance Criteria:**
- [ ] `upsert_vector()` accepts optional `person_ids` parameter
- [ ] Payload includes `person_ids` array (empty array if no faces)
- [ ] Existing vectors without `person_ids` handled gracefully

### Task 1.2: Implement Filter Logic in search_vectors()

**File:** `image-search-service/src/image_search_service/vector/qdrant.py`
**Function:** `search_vectors()` (line ~160)

Add filter handling for `personId`:

```python
# Add after line 173 (category_id handling):
if filters.get("personId") or filters.get("person_id"):
    person_id = filters.get("personId") or filters.get("person_id")
    conditions.append(
        FieldCondition(
            key="person_ids",
            match=MatchAny(any=[person_id])  # Matches if person_id in array
        )
    )
```

**Acceptance Criteria:**
- [ ] `personId` filter creates Qdrant FieldCondition
- [ ] Filter uses `MatchAny` for array membership check
- [ ] Works in combination with existing date/category filters
- [ ] Returns empty results if person has no photos (not error)

### Task 1.3: Update Indexing Pipeline

**File:** `image-search-service/src/image_search_service/services/embedding.py` (or equivalent)

Modify embedding/indexing to include person_ids:

```python
async def index_asset(asset_id: int, db: AsyncSession):
    # Get faces for this asset
    faces = await get_faces_for_asset(asset_id, db)
    person_ids = [f.person_id for f in faces if f.person_id]

    # Include in payload
    payload = {
        "asset_id": str(asset_id),
        "person_ids": person_ids,  # NEW
        # ... other fields
    }

    upsert_vector(vector=embedding, payload=payload)
```

**Acceptance Criteria:**
- [ ] New assets indexed with current `person_ids`
- [ ] Assets with no faces have empty `person_ids` array
- [ ] Re-index job available for existing assets

---

## Phase 2: Payload Sync on Face Changes

**Effort:** 3-4 hours
**Priority:** High

### Task 2.1: Update Payload on Face Assignment

**File:** `image-search-service/src/image_search_service/api/routes/faces.py`
**Endpoint:** `POST /api/v1/faces/faces/{faceId}/assign`

After assigning face to person, update Qdrant payload:

```python
@router.post("/faces/{face_id}/assign")
async def assign_face_to_person(...):
    # ... existing assignment logic ...

    # NEW: Update Qdrant payload
    person_ids = await get_person_ids_for_asset(face.asset_id, db)
    await update_vector_payload(
        asset_id=face.asset_id,
        payload_updates={"person_ids": person_ids}
    )
```

**Acceptance Criteria:**
- [ ] Face assignment triggers Qdrant payload update
- [ ] Payload reflects all persons in the image
- [ ] Async update doesn't block API response

### Task 2.2: Update Payload on Face Unassignment

**File:** `image-search-service/src/image_search_service/api/routes/faces.py`
**Endpoint:** `DELETE /api/v1/faces/faces/{faceId}/person`

After removing face assignment, update Qdrant payload:

```python
@router.delete("/faces/{face_id}/person")
async def unassign_face(...):
    # ... existing unassignment logic ...

    # NEW: Update Qdrant payload
    person_ids = await get_person_ids_for_asset(face.asset_id, db)
    await update_vector_payload(
        asset_id=face.asset_id,
        payload_updates={"person_ids": person_ids}
    )
```

**Acceptance Criteria:**
- [ ] Face unassignment triggers Qdrant payload update
- [ ] Empty array if no more faces assigned
- [ ] Handles edge case: face without asset gracefully

### Task 2.3: Update Payload on Person Merge

**File:** `image-search-service/src/image_search_service/api/routes/people.py`
**Endpoint:** `POST /api/v1/people/merge`

After merging persons, update all affected asset payloads:

```python
@router.post("/merge")
async def merge_persons(...):
    # ... existing merge logic ...

    # NEW: Update all affected Qdrant payloads
    affected_asset_ids = await get_assets_for_persons([source_id, target_id], db)
    for asset_id in affected_asset_ids:
        person_ids = await get_person_ids_for_asset(asset_id, db)
        await update_vector_payload(
            asset_id=asset_id,
            payload_updates={"person_ids": person_ids}
        )
```

**Acceptance Criteria:**
- [ ] Person merge updates all affected Qdrant payloads
- [ ] Merged person ID replaced with target person ID
- [ ] Batch update for efficiency (not one-by-one)

---

## Phase 3: Frontend Alignment

**Effort:** 1-2 hours
**Priority:** Medium

### Task 3.1: Fix Single vs Multi-Select Mismatch

**File:** `image-search-ui/src/lib/components/FiltersPanel.svelte`

**Option A: Convert to Single-Select (Recommended for MVP)**

```typescript
// Change from array to single value:
let selectedPersonId = $state<string | null>(null);

// Update filter construction:
let filters = $derived<SearchFilters>({
    ...(dateFrom && { dateFrom }),
    ...(dateTo && { dateTo }),
    ...(categoryId && { categoryId }),
    ...(selectedPersonId && { personId: selectedPersonId })  // Single value
});
```

**Option B: Keep Multi-Select (Requires API Contract Change)**

- Update API contract to accept `personIds: string[]`
- Implement OR logic in backend
- Higher effort, defer to Phase 4

**Acceptance Criteria (Option A):**
- [ ] UI shows single-select dropdown (not multi-select)
- [ ] Selected person displayed as single value/chip
- [ ] Clear button removes selection
- [ ] Filter correctly populated with single personId

### Task 3.2: Update Type Comment

**File:** `image-search-ui/src/lib/types.ts`

```typescript
export interface SearchFilters {
    dateFrom?: string;
    dateTo?: string;
    personId?: string;    // Filter by person ID (implemented) ← Update comment
    categoryId?: number;
}
```

---

## Phase 4: Testing

**Effort:** 2-3 hours
**Priority:** High

### Backend Tests

**File:** `image-search-service/tests/api/test_search.py`

```python
async def test_search_with_person_filter():
    """Search with personId filter returns only matching assets."""
    # Setup: Create person and assign faces
    person = await create_test_person(name="Alice")
    asset_with_alice = await create_test_asset()
    asset_without_alice = await create_test_asset()
    await assign_face(asset_with_alice.id, person.id)

    # Index both assets
    await index_assets([asset_with_alice, asset_without_alice])

    # Search with person filter
    response = await client.post("/api/v1/search", json={
        "query": "test",
        "filters": {"personId": str(person.id)}
    })

    # Verify: Only asset_with_alice returned
    assert response.status_code == 200
    result_ids = [r["asset"]["id"] for r in response.json()["results"]]
    assert asset_with_alice.id in result_ids
    assert asset_without_alice.id not in result_ids


async def test_search_with_person_filter_no_matches():
    """Search with personId filter returns empty when no matches."""
    person = await create_test_person(name="Unknown")
    # Don't assign any faces

    response = await client.post("/api/v1/search", json={
        "query": "test",
        "filters": {"personId": str(person.id)}
    })

    assert response.status_code == 200
    assert response.json()["results"] == []


async def test_search_combined_filters():
    """Person filter works with other filters."""
    # Test person + category + date filters together
    ...
```

**File:** `image-search-service/tests/unit/test_qdrant.py`

```python
def test_search_vectors_with_person_filter():
    """search_vectors applies personId filter correctly."""
    mock_client = MagicMock()

    search_vectors(
        query_vector=[0.1] * 512,
        filters={"personId": "person-uuid-123"}
    )

    # Verify FieldCondition created for person_ids
    call_args = mock_client.search.call_args
    filter_arg = call_args.kwargs.get("query_filter")
    assert any(
        cond.key == "person_ids"
        for cond in filter_arg.must
    )
```

### Frontend Tests

**File:** `image-search-ui/src/tests/components/FiltersPanel.test.ts`

```typescript
it('sends personId filter when person selected', async () => {
    const onFilterChange = vi.fn();
    render(FiltersPanel, { props: { onFilterChange } });

    // Select a person
    await userEvent.click(screen.getByLabelText('People Filter'));
    await userEvent.click(screen.getByText('Alice'));

    // Verify filter includes personId
    expect(onFilterChange).toHaveBeenCalledWith(
        expect.objectContaining({ personId: 'alice-uuid' })
    );
});

it('clears personId filter when selection removed', async () => {
    // ... test clearing filter
});
```

---

## Phase 5: Migration & Data Backfill (Optional)

**Effort:** 1-2 hours
**Priority:** Low (for existing data)

### Task 5.1: Backfill Existing Vectors

Create script to update existing Qdrant vectors with person_ids:

```python
# scripts/backfill_person_ids.py
async def backfill_person_ids():
    """Update all existing Qdrant vectors with person_ids."""
    async with get_db() as db:
        # Get all assets
        assets = await db.execute(select(ImageAsset))

        for asset in assets.scalars():
            person_ids = await get_person_ids_for_asset(asset.id, db)
            await update_vector_payload(
                asset_id=asset.id,
                payload_updates={"person_ids": person_ids}
            )

        logger.info(f"Backfilled {len(assets)} vectors")
```

**Run:** `python -m scripts.backfill_person_ids`

---

## Implementation Summary

### Minimum Viable (Phases 1-3)

| Phase | Tasks | Effort | Outcome |
|-------|-------|--------|---------|
| 1 | Backend filter implementation | 4-6h | People filter works |
| 2 | Payload sync on face changes | 3-4h | Filter stays current |
| 3 | Frontend alignment | 1-2h | UI matches backend |
| **Total MVP** | | **8-12h** | **Functional people filter** |

### Production Ready (+ Phase 4-5)

| Phase | Tasks | Effort | Outcome |
|-------|-------|--------|---------|
| 4 | Testing | 2-3h | Confidence in implementation |
| 5 | Migration | 1-2h | Existing data works |
| **Total Prod** | | **11-17h** | **Robust, tested feature** |

---

## Dependencies

- [ ] Face detection system operational (appears to exist)
- [ ] Person management API operational (appears to exist)
- [ ] Qdrant collection accessible for payload updates

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Qdrant payload update fails | Low | High | Queue updates, retry logic |
| Performance with large payloads | Medium | Medium | Benchmark with 10K+ vectors |
| Migration corrupts existing data | Low | High | Test on copy first, rollback plan |
| Multi-select expected by users | Medium | Low | Document single-select for MVP |

---

## Decision Points for Review

1. **Single-select vs Multi-select for MVP?**
   - Recommendation: Single-select (simpler, matches current API contract)
   - Multi-select can be Phase 2 enhancement

2. **Sync vs Async payload updates?**
   - Recommendation: Async (queue) to avoid blocking API responses
   - Immediate consistency vs eventual consistency trade-off

3. **Backfill existing data?**
   - Recommendation: Yes, but as separate script (not migration)
   - Can be run during low-traffic period

---

## Approval

- [ ] Plan reviewed
- [ ] Approach approved
- [ ] Ready to implement

---

*For detailed code analysis, see [people-filter-analysis-2025-12-29.md](./people-filter-analysis-2025-12-29.md)*
