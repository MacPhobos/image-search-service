# People Filter Implementation Analysis

**Date:** 2025-12-29
**Researcher:** Research Agent
**System:** Image Search (image-search-ui + image-search-service)

---

## Executive Summary

The People Filter functionality in the image search system is **partially implemented**. The UI has a complete multi-select people filter with search capabilities, but the backend **does not currently support filtering by person ID**. The filter parameter is sent to the API but is ignored during vector search execution.

### Key Findings

1. **Frontend:** Fully implemented with multi-select UI, person search, and chip display
2. **API Contract:** Specifies `personId` query parameter for search endpoint (line 443)
3. **Backend Implementation:** `personId` filter is **not applied** to Qdrant search queries
4. **Data Flow Gap:** Filter reaches backend but is discarded before vector search

---

## API Contract Specification

**Source:** `docs/api-contract.md` (identical in both repos)

### Search Endpoint: `GET /api/v1/search`

```typescript
// Line 431-447
| Parameter    | Type    | Default    | Description                        |
|--------------|---------|------------|------------------------------------|
| `q`          | string  | _required_ | Search query text                  |
| `page`       | integer | 1          | Page number                        |
| `pageSize`   | integer | 20         | Results per page (max: 100)        |
| `minScore`   | number  | 0.0        | Minimum similarity score (0.0-1.0) |
| `personId`   | string  | -          | Filter by person ID                |
| `categoryId` | integer | -          | Filter by category ID              |
| `dateFrom`   | string  | -          | Filter: date taken >= (ISO 8601)   |
| `dateTo`     | string  | -          | Filter: date taken <= (ISO 8601)   |
```

**Contract Status:** `personId` is documented as a supported filter parameter.

---

## Frontend Implementation (image-search-ui)

### 1. Search Page (`src/routes/+page.svelte`)

**Location:** Lines 1-119

**Key Components:**
- Imports `FiltersPanel` component
- Maintains `filters` state of type `SearchFilters`
- Passes filters to `searchImages()` API call
- Re-executes search when filters change

```svelte
// Lines 49-55
function handleFilterChange(newFilters: SearchFilters) {
    filters = newFilters;
    // Re-run search if we already have a query
    if (query) {
        handleSearch(query);
    }
}
```

**Data Flow:**
```
FiltersPanel (user input)
  → handleFilterChange(newFilters)
  → filters state update
  → handleSearch(query)
  → searchImages({ query, filters })
```

---

### 2. Filters Panel (`src/lib/components/FiltersPanel.svelte`)

**Location:** Lines 1-580

**Implementation Details:**

#### Person Filter UI (Lines 209-295)

**Features Implemented:**
1. **Multi-select dropdown** with search functionality
2. **Selected persons displayed as chips** with remove buttons
3. **Person search** with real-time filtering
4. **Loading states** during person data fetch
5. **Empty state** with link to face labeling when no people exist

#### State Management (Lines 30-36)

```typescript
// Person filter (multi-select)
let selectedPersonIds = $state<string[]>([]);
let persons = $state<Person[]>([]);
let personsLoading = $state(true);
let personSearchQuery = $state('');
let showPersonDropdown = $state(false);
```

#### Filter Construction (Lines 79-85)

```typescript
let filters = $derived<SearchFilters>({
    ...(dateFrom && { dateFrom }),
    ...(dateTo && { dateTo }),
    ...(categoryId && { categoryId }),
    ...(selectedPersonIds.length > 0 && { personId: selectedPersonIds[0] })
    //                                              ^^^^^^^^^^^^^^^^^^^
    //                                              ONLY FIRST PERSON!
});
```

**Critical Finding:** Only the **first selected person ID** is passed to the API, even though the UI supports multi-select.

#### Person Selection Logic (Lines 117-129)

```typescript
function handlePersonSelect(person: Person) {
    if (selectedPersonIds.includes(person.id)) {
        // Remove if already selected
        selectedPersonIds = selectedPersonIds.filter((id) => id !== person.id);
    } else {
        // Add to selection
        selectedPersonIds = [...selectedPersonIds, person.id];
    }
}
```

**UI Behavior:**
- Users can select multiple people via dropdown
- Selected people appear as removable chips
- Chips display person name with X button for removal

---

### 3. API Client (`src/lib/api/client.ts`)

**Location:** Lines 57-90

**Search Request Construction:**

```typescript
// Lines 57-90
export async function searchImages(params: SearchParams): Promise<SearchResponse> {
    const requestBody: SearchRequest = {
        query: params.query,
        limit: params.limit ?? 50,
        offset: params.offset ?? 0,
        filters: undefined
    };

    // Convert date filters to API format if provided
    if (params.filters) {
        const apiFilters: Record<string, string> = {};
        if (params.filters.dateFrom) {
            apiFilters['dateFrom'] = params.filters.dateFrom;
        }
        if (params.filters.dateTo) {
            apiFilters['dateTo'] = params.filters.dateTo;
        }
        if (params.filters.personId) {
            apiFilters['personId'] = params.filters.personId;  // ← Sent to API
        }
        if (params.filters.categoryId) {
            apiFilters['categoryId'] = params.filters.categoryId.toString();
        }
        if (Object.keys(apiFilters).length > 0) {
            requestBody.filters = apiFilters;
        }
    }

    return apiRequest<SearchResponse>('/api/v1/search', {
        method: 'POST',
        body: JSON.stringify(requestBody)
    });
}
```

**Behavior:**
- `personId` filter is included in request body if provided
- Filter sent as string value in `filters` object
- Request uses POST method to `/api/v1/search`

---

### 4. Type Definitions (`src/lib/types.ts`)

**Location:** Lines 34-40

```typescript
// Search filters for UI (date range + future face filter)
export interface SearchFilters {
    dateFrom?: string;    // ISO 8601 date
    dateTo?: string;      // ISO 8601 date
    personId?: string;    // Face filter (future) ← Comment says "future"
    categoryId?: number;  // Category filter
}
```

**Note:** Comment incorrectly marks `personId` as "future" despite implementation existing.

---

## Backend Implementation (image-search-service)

### 1. Search Route (`src/image_search_service/api/routes/search.py`)

**Location:** Lines 25-112

**Request Handling:**

```python
# Lines 25-30
@router.post("", response_model=SearchResponse, responses={503: {"model": ErrorResponse}})
async def search_assets(
    request: SearchRequest,
    db: AsyncSession = Depends(get_db),
    qdrant: QdrantClient = Depends(get_qdrant_client),
) -> SearchResponse:
```

**Filter Processing (Lines 70-83):**

```python
# Build filters dict from request
search_filters: dict[str, str | int] = {}
if request.filters:
    search_filters.update(request.filters)  # ← personId included here
if request.category_id is not None:
    search_filters["category_id"] = request.category_id

vector_results = search_vectors(
    query_vector=query_vector,
    limit=request.limit,
    offset=request.offset,
    filters=search_filters if search_filters else None,  # ← Passed to Qdrant
    client=qdrant,
)
```

**Behavior:**
- `personId` from request.filters is added to `search_filters` dict
- Filter dict is passed to `search_vectors()` function
- No validation or transformation of `personId` value

---

### 2. Search Request Schema (`src/image_search_service/api/schemas.py`)

**Location:** Lines 56-66

```python
class SearchRequest(BaseModel):
    """Request to search for assets."""

    model_config = ConfigDict(populate_by_name=True)

    query: str
    limit: int = 50
    offset: int = 0
    filters: dict[str, str] | None = None  # from_date, to_date
    category_id: int | None = Field(None, alias="categoryId", description="Filter by category ID")
```

**Schema Details:**
- `filters` accepts any `dict[str, str]` (no schema validation)
- Comment only mentions `from_date, to_date` - **`personId` undocumented**
- `category_id` handled separately (has dedicated field)

---

### 3. Qdrant Vector Search (`src/image_search_service/vector/qdrant.py`)

**Location:** Lines 133-195

**Filter Application (Lines 156-176):**

```python
def search_vectors(
    query_vector: list[float],
    limit: int = 50,
    offset: int = 0,
    filters: dict[str, str | int] | None = None,
    client: QdrantClient | None = None,
) -> list[dict[str, Any]]:
    """Search for similar vectors.

    Args:
        filters: Optional filters (from_date, to_date, category_id)
                                   ^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^
                                   personId NOT mentioned
    """
    # Build filter if date range or category_id provided
    qdrant_filter = None
    if filters:
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
        # ← NO HANDLING FOR personId OR person_id!

        if conditions:
            qdrant_filter = Filter(must=conditions)
```

**Critical Gap:**
- Only `from_date`, `to_date`, and `category_id` are processed
- **`personId` filter is completely ignored**
- No error raised if unknown filter keys are present
- Filters are silently dropped

---

## Data Flow Analysis

### Current Implementation Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│ FRONTEND (image-search-ui)                                         │
├─────────────────────────────────────────────────────────────────────┤
│ 1. User selects person(s) in FiltersPanel                          │
│    selectedPersonIds = ['person-uuid-1', 'person-uuid-2']          │
│                                                                     │
│ 2. Filter object created (ONLY FIRST PERSON!)                      │
│    filters = { personId: 'person-uuid-1' }                         │
│                                                                     │
│ 3. API client sends POST /api/v1/search                            │
│    body = {                                                         │
│      query: "beach photos",                                         │
│      filters: { personId: "person-uuid-1" }                         │
│    }                                                                │
└─────────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ BACKEND (image-search-service)                                     │
├─────────────────────────────────────────────────────────────────────┤
│ 4. FastAPI receives SearchRequest                                  │
│    request.filters = { "personId": "person-uuid-1" }               │
│                                                                     │
│ 5. Search route builds search_filters dict                         │
│    search_filters = { "personId": "person-uuid-1" }                │
│                                                                     │
│ 6. Calls search_vectors(filters=search_filters)                    │
└─────────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ QDRANT SEARCH (vector/qdrant.py)                                   │
├─────────────────────────────────────────────────────────────────────┤
│ 7. Processes filters:                                               │
│    ✓ from_date    → FieldCondition(key="created_at", range=...)   │
│    ✓ to_date      → FieldCondition(key="created_at", range=...)   │
│    ✓ category_id  → FieldCondition(key="category_id", match=...)  │
│    ✗ personId     → IGNORED (no handler)                           │
│                                                                     │
│ 8. Executes Qdrant query WITHOUT person filter                     │
│    Results: All assets matching query (ignoring person filter)     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Gap Analysis

### 1. Multi-Select Support

**Current State:**
- UI allows selecting multiple people
- State management supports arrays: `selectedPersonIds: string[]`
- **Only first person ID sent to API**

**API Contract:**
- Specifies single `personId` parameter (not array)
- No support for filtering by multiple people simultaneously

**Gap:** UI promises multi-select but backend only supports single person filter.

---

### 2. Backend Filter Processing

**Current State:**
- `personId` filter accepted in request schema
- Filter passed through to `search_vectors()`
- **Completely ignored in Qdrant filter construction**

**Expected Behavior:**
```python
# Missing implementation:
if filters.get("personId") or filters.get("person_id"):
    conditions.append(
        FieldCondition(key="person_id", match=MatchValue(value=filters["personId"]))
    )
```

**Gap:** Zero implementation of person-based filtering in vector search.

---

### 3. Data Model Relationship

**Required for Person Filtering:**

```
ImageAsset (DB)
    ↓ 1:N
FaceInstance (DB)
    ↓ N:1
Person (DB)
    ↓ metadata
person_id stored in Qdrant payload
```

**Current Qdrant Payload:**
- `asset_id`: string
- `created_at`: timestamp
- `category_id`: integer (optional)
- `path`: string
- **Missing:** `person_id` or `person_ids` field

**Gap:** Qdrant vectors don't contain person metadata required for filtering.

---

### 4. Face-to-Asset Relationship

**Database Schema (Expected):**

```sql
-- ImageAsset table
CREATE TABLE image_assets (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL,
    created_at TIMESTAMP,
    category_id INTEGER REFERENCES categories(id)
);

-- FaceInstance table (links assets to people)
CREATE TABLE face_instances (
    id TEXT PRIMARY KEY,  -- UUID
    asset_id INTEGER REFERENCES image_assets(id),
    person_id TEXT REFERENCES persons(id),
    bounding_box JSONB,
    confidence FLOAT,
    created_at TIMESTAMP
);

-- Person table
CREATE TABLE persons (
    id TEXT PRIMARY KEY,  -- UUID
    name TEXT UNIQUE NOT NULL,
    status TEXT,
    face_count INTEGER,
    created_at TIMESTAMP
);
```

**Query Logic Required:**

To filter images by person, the backend needs to:

1. **Option A: Payload-based (Faster)**
   - Store `person_ids: ["uuid1", "uuid2"]` in Qdrant payload during indexing
   - Filter vectors using `FieldCondition(key="person_ids", match=...)`
   - Update payload when faces are assigned/unassigned

2. **Option B: Post-filter (Simpler)**
   - Execute normal vector search (no Qdrant filter)
   - Query `face_instances` table for matching `asset_ids`:
     ```sql
     SELECT DISTINCT asset_id
     FROM face_instances
     WHERE person_id = 'person-uuid'
     ```
   - Filter search results to only include matching asset IDs

**Gap:** Neither approach is currently implemented.

---

## Behavioral Analysis

### User Experience (Current)

```
┌──────────────────────────────────────────────────────────────┐
│ User Action: Select "Alice" in People Filter                │
│ Expected Result: Show only photos containing Alice          │
│ Actual Result: Shows all photos matching search query       │
│                (person filter silently ignored)              │
└──────────────────────────────────────────────────────────────┘
```

**Symptoms:**
1. **Silent Failure:** No error message shown to user
2. **False Positive UX:** UI suggests filter is active (chip displayed)
3. **Confusing Results:** User sees photos without selected person
4. **Multi-select Illusion:** Can select multiple people, but only first matters (and even that is ignored)

---

### Edge Cases

#### Case 1: Multiple People Selected
```typescript
// Frontend state:
selectedPersonIds = ["person-A", "person-B", "person-C"]

// Sent to API:
filters = { personId: "person-A" }  // Only first

// Backend processes:
search_filters = { "personId": "person-A" }

// Qdrant filters:
// (ignored - no filter applied)
```

**Result:** Same as selecting zero people.

---

#### Case 2: Person Filter + Category Filter
```typescript
// Frontend:
filters = {
    personId: "person-A",
    categoryId: 5
}

// Backend:
search_filters = { "personId": "person-A", "category_id": 5 }

// Qdrant:
conditions = [
    FieldCondition(key="category_id", match=5)  // ✓ Applied
    // personId ignored
]
```

**Result:** Category filter works, person filter silently dropped.

---

#### Case 3: Clear Filters
```typescript
// Frontend:
selectedPersonIds = []  // Clear all people

// Sent to API:
filters = {}  // No personId field

// Backend:
// No filters applied (correct behavior)
```

**Result:** Works correctly (no spurious filtering).

---

## Missing Implementation Components

### 1. Database Schema Changes

**Required Tables:**
- ✅ `persons` table (exists per API contract)
- ✅ `face_instances` table (exists per API contract)
- ❓ `image_assets.person_ids` JSONB column (if using payload approach)

**Required Indexes:**
```sql
-- For efficient person-to-asset lookup
CREATE INDEX idx_face_instances_person_id
ON face_instances(person_id);

CREATE INDEX idx_face_instances_asset_id
ON face_instances(asset_id);
```

---

### 2. Qdrant Payload Updates

**Current Payload Structure:**
```python
{
    "asset_id": "123",
    "path": "/photos/beach.jpg",
    "created_at": "2024-12-19T10:00:00Z",
    "category_id": 5
}
```

**Required Additions:**
```python
{
    "asset_id": "123",
    "path": "/photos/beach.jpg",
    "created_at": "2024-12-19T10:00:00Z",
    "category_id": 5,
    "person_ids": ["person-uuid-1", "person-uuid-2"]  # ← NEW
}
```

**Update Triggers:**
- When face is assigned to person (`POST /api/v1/faces/faces/{faceId}/assign`)
- When face is unassigned (`DELETE /api/v1/faces/faces/{faceId}/person`)
- When person is merged (`POST /api/v1/people/merge`)
- During initial indexing/embedding generation

---

### 3. Backend Filter Logic

**File:** `src/image_search_service/vector/qdrant.py`
**Function:** `search_vectors()` (line 133)

**Required Implementation:**
```python
# Add to filter construction (after line 173):
if filters.get("personId") or filters.get("person_id"):
    person_id = filters.get("personId") or filters.get("person_id")
    conditions.append(
        FieldCondition(
            key="person_ids",  # Payload field
            match=MatchValue(value=person_id)
        )
    )
```

**Alternative (Post-Filter Approach):**
```python
# In search.py, after vector search:
if request.filters and request.filters.get("personId"):
    person_id = request.filters["personId"]

    # Query face_instances for assets with this person
    result = await db.execute(
        select(FaceInstance.asset_id)
        .where(FaceInstance.person_id == person_id)
        .distinct()
    )
    allowed_asset_ids = {row[0] for row in result.fetchall()}

    # Filter vector results
    results = [
        r for r in results
        if r.asset.id in allowed_asset_ids
    ]
```

---

### 4. Frontend Multi-Select Support

**Current Limitation:**
```typescript
// Only sends first person:
...(selectedPersonIds.length > 0 && { personId: selectedPersonIds[0] })
```

**Options for Fix:**

**Option A: Remove Multi-Select (Simplest)**
```typescript
// Change to single-select:
let selectedPersonId = $state<string | null>(null);

// Update filter:
...(selectedPersonId && { personId: selectedPersonId })
```

**Option B: Support Multiple People (API Contract Change)**
```typescript
// Send array (requires API contract v2.0):
...(selectedPersonIds.length > 0 && { personIds: selectedPersonIds })
```

**API Contract Update:**
```typescript
interface SearchFilters {
    personIds?: string[];  // Filter by multiple people (OR logic)
}
```

**Backend Logic:**
```python
if filters.get("personIds"):
    person_ids = filters["personIds"]
    # Qdrant "should" condition (OR logic):
    should_conditions = [
        FieldCondition(key="person_ids", match=MatchValue(value=pid))
        for pid in person_ids
    ]
    conditions.append(Filter(should=should_conditions))
```

---

## Recommendations

### Immediate Actions (Short-term)

#### 1. Fix Frontend Multi-Select Mismatch
**Priority:** HIGH
**Effort:** LOW

**Action:** Convert to single-select to match API contract.

**Implementation:**
```typescript
// FiltersPanel.svelte - Change from array to single value:
let selectedPersonId = $state<string | null>(null);

// Update filter:
...(selectedPersonId && { personId: selectedPersonId })

// Simplify UI: Replace chips with single dropdown selection
```

**Rationale:** Eliminates false expectations until multi-select is supported.

---

#### 2. Document Person Filter as Non-Functional
**Priority:** HIGH
**Effort:** LOW

**Action:** Update UI to indicate person filter is not yet implemented.

**Implementation:**
```svelte
<!-- FiltersPanel.svelte -->
<label for="personFilter">
    People Filter
    <span class="badge-beta">Coming Soon</span>
</label>
```

**Rationale:** Sets correct user expectations until backend is implemented.

---

### Core Implementation (Medium-term)

#### 3. Implement Backend Person Filtering
**Priority:** HIGH
**Effort:** MEDIUM

**Subtasks:**
1. **Add person_ids to Qdrant payload**
   - Update `upsert_vector()` calls during indexing
   - Populate from `face_instances` table
   - Handle face assignment/unassignment events

2. **Implement filter logic in `search_vectors()`**
   - Add `personId` handling in Qdrant filter construction
   - Test with various person IDs
   - Validate filter behavior with no matches

3. **Update API documentation**
   - Document `personId` filter in schema docstring
   - Add example request with person filter
   - Update `api-contract.md` if behavior differs

**Acceptance Criteria:**
- Search returns only images containing specified person
- Filter works in combination with date/category filters
- Empty results when person has no photos
- Performance acceptable with 10K+ vectors

---

### Advanced Features (Long-term)

#### 4. Support Multi-Person Filtering
**Priority:** MEDIUM
**Effort:** HIGH

**Requirements:**
1. **API Contract v2.0:**
   - Add `personIds: string[]` parameter
   - Define OR vs AND semantics (OR recommended)
   - Deprecate single `personId` parameter

2. **Backend Implementation:**
   - Update Qdrant filter with "should" conditions (OR logic)
   - Consider AND logic option: `personIds` AND `personIdsAll`

3. **Frontend Enhancement:**
   - Re-enable multi-select chips UI
   - Send full `selectedPersonIds` array
   - Update filter display to show all selected people

**Use Cases:**
- "Show photos with Alice OR Bob" (OR logic)
- "Show photos with both Alice AND Bob" (AND logic - future)

---

#### 5. Add Face Count to Search Results
**Priority:** LOW
**Effort:** MEDIUM

**Feature:** Display number of faces detected in each search result.

**Implementation:**
```typescript
interface SearchResult {
    asset: Asset;
    score: float;
    highlights: string[];
    faceCount?: number;      // NEW
    personNames?: string[];  // NEW - names of people in photo
}
```

**UI Enhancement:**
```svelte
<div class="result-card">
    <img src={result.asset.thumbnailUrl} />
    <div class="result-metadata">
        <span class="score">Score: {result.score.toFixed(2)}</span>
        {#if result.faceCount}
            <span class="face-count">{result.faceCount} faces</span>
        {/if}
        {#if result.personNames}
            <div class="person-tags">
                {#each result.personNames as name}
                    <span class="person-tag">{name}</span>
                {/each}
            </div>
        {/if}
    </div>
</div>
```

---

## Testing Requirements

### Frontend Tests

**File:** `src/tests/components/FiltersPanel.test.ts`

**Test Cases:**
1. ✅ Loads people from API on mount
2. ✅ Displays loading state while fetching people
3. ✅ Shows empty state when no people exist
4. ✅ Renders person dropdown with search
5. ⚠️ **Missing:** Verifies only single person sent to API (not full array)
6. ⚠️ **Missing:** Validates filter callback includes personId
7. ⚠️ **Missing:** Tests person search filtering
8. ⚠️ **Missing:** Tests chip removal updates filter

---

### Backend Tests

**File:** `tests/api/test_search.py`

**Test Cases:**
1. ⚠️ **Missing:** Search with `personId` filter returns only matching assets
2. ⚠️ **Missing:** Search with invalid `personId` returns empty results
3. ⚠️ **Missing:** Person filter combined with category filter
4. ⚠️ **Missing:** Person filter combined with date range filter
5. ⚠️ **Missing:** Multiple people in same photo (future multi-select)

**File:** `tests/unit/test_qdrant.py`

**Test Cases:**
1. ⚠️ **Missing:** `search_vectors()` applies personId filter
2. ⚠️ **Missing:** Qdrant filter construction includes person_ids condition
3. ⚠️ **Missing:** Filter handles missing personId gracefully

---

### Integration Tests

**Test Scenario:**
```python
async def test_search_by_person_integration():
    # Setup: Create person and assign faces
    person = await create_person(name="Alice")
    asset1 = await create_asset(path="/photo1.jpg")
    asset2 = await create_asset(path="/photo2.jpg")
    asset3 = await create_asset(path="/photo3.jpg")

    # Assign faces to Alice
    await assign_face(asset_id=asset1.id, person_id=person.id)
    await assign_face(asset_id=asset2.id, person_id=person.id)
    # asset3 has no faces assigned

    # Index all assets
    await index_assets([asset1, asset2, asset3])

    # Search with person filter
    response = await client.post("/api/v1/search", json={
        "query": "photo",
        "filters": {"personId": person.id}
    })

    # Verify: Only asset1 and asset2 returned
    assert len(response.results) == 2
    assert {r.asset.id for r in response.results} == {asset1.id, asset2.id}
```

---

## Code Locations Reference

### Frontend (image-search-ui)

| Component | File Path | Lines | Description |
|-----------|-----------|-------|-------------|
| Search Page | `src/routes/+page.svelte` | 1-119 | Main dashboard with search and filters |
| Filters Panel | `src/lib/components/FiltersPanel.svelte` | 1-580 | Person filter UI with multi-select |
| API Client | `src/lib/api/client.ts` | 57-90 | `searchImages()` function |
| Type Definitions | `src/lib/types.ts` | 34-40 | `SearchFilters` interface |
| Persons API | `src/lib/api/faces.ts` | N/A | `listPersons()` function |

### Backend (image-search-service)

| Component | File Path | Lines | Description |
|-----------|-----------|-------|-------------|
| Search Route | `src/.../api/routes/search.py` | 25-112 | POST /api/v1/search endpoint |
| Search Schemas | `src/.../api/schemas.py` | 56-82 | `SearchRequest`, `SearchResponse` |
| Qdrant Client | `src/.../vector/qdrant.py` | 133-195 | `search_vectors()` function |
| DB Models | `src/.../db/models.py` | N/A | `Person`, `FaceInstance` models |

### Documentation

| Document | Path | Relevant Sections |
|----------|------|-------------------|
| API Contract | `docs/api-contract.md` | Lines 431-447 (Search endpoint) |
| API Contract | `docs/api-contract.md` | Lines 498-737 (People/Faces) |
| Frontend Guide | `image-search-ui/CLAUDE.md` | N/A |
| Backend Guide | `image-search-service/CLAUDE.md` | N/A |

---

## Appendix: Related Endpoints

The People Filter implementation depends on these related endpoints:

### 1. List People (`GET /api/v1/people`)
**Status:** ✅ Implemented
**Usage:** FiltersPanel loads people for dropdown
**Contract:** Lines 570-583

### 2. Assign Face (`POST /api/v1/faces/faces/{faceId}/assign`)
**Status:** ✅ Implemented
**Impact:** Should trigger Qdrant payload update (not implemented)
**Contract:** Lines 643-695

### 3. Unassign Face (`DELETE /api/v1/faces/faces/{faceId}/person`)
**Status:** ✅ Implemented
**Impact:** Should trigger Qdrant payload update (not implemented)
**Contract:** Lines 697-737

### 4. Merge Persons (`POST /api/v1/people/merge`)
**Status:** ✅ Implemented
**Impact:** Should update Qdrant payloads for all affected assets
**Contract:** Lines 605-629

---

## Summary

**Current State:**
- ✅ Frontend UI: Fully implemented (with multi-select)
- ✅ API Contract: Documented
- ⚠️ Frontend Logic: Sends only first person ID (bug)
- ❌ Backend Processing: Filter completely ignored
- ❌ Qdrant Payloads: Missing person_ids field
- ❌ Tests: No coverage for person filtering

**To Make People Filter Work:**

**Minimum Viable:**
1. Add `person_ids` to Qdrant payloads during indexing
2. Implement filter logic in `search_vectors()` function
3. Update payloads when faces are assigned/unassigned
4. Fix frontend to send single personId (or update for multi-select)

**Production Ready:**
5. Add comprehensive tests (frontend + backend + integration)
6. Update API documentation with examples
7. Add performance benchmarks for filtered queries
8. Implement multi-person filtering with OR/AND logic
9. Add person metadata to search results (face count, names)

---

**Research Classification:** Actionable
**Estimated Implementation Effort:** 2-3 days (single person), 4-5 days (multi-person)
**Blocking Dependencies:** Face detection and person labeling systems (already exist)
