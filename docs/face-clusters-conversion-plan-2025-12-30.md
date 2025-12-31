# Face Clusters View Conversion Plan

**Date**: 2025-12-30
**Type**: Feature Conversion/Simplification
**Scope**: Frontend + Backend
**Impact**: Breaking change to Face Clusters UI

---

## Executive Summary

Convert the existing "Face Clusters" view from a dual-tab interface (showing both labeled and unlabeled clusters) into a simplified, single-view interface that ONLY displays high-confidence groups of **unknown/unlabeled faces**. This transformation eliminates UI complexity while creating a focused workflow for discovering and labeling new people in the photo collection.

**Key Changes**:
- Remove tab navigation ("All Clusters" / "Unlabeled")
- Filter to show ONLY unlabeled clusters with high intra-cluster similarity
- Add configurable confidence threshold (via Admin/Settings)
- Add configurable minimum cluster size (via Admin/Settings)
- Maintain existing labeling workflow (click cluster → label to person)

---

## Current State Analysis

### Existing Implementation

**Frontend Components**:
- **Route**: `/export/workspace/image-search/image-search-ui/src/routes/faces/clusters/+page.svelte`
- **Card Component**: `/export/workspace/image-search/image-search-ui/src/lib/components/faces/ClusterCard.svelte`
- **Detail Route**: `/export/workspace/image-search/image-search-ui/src/routes/faces/clusters/[clusterId]/+page.svelte`

**Backend Endpoints**:
- **List Clusters**: `GET /api/v1/faces/clusters?include_labeled={bool}`
- **Get Cluster Details**: `GET /api/v1/faces/clusters/{cluster_id}`
- **Label Cluster**: `POST /api/v1/faces/clusters/{cluster_id}/label`

**Database Schema** (`FaceInstance` model):
```python
cluster_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
person_id: Mapped[uuid.UUID | None] = mapped_column(
    UUID(as_uuid=True),
    ForeignKey("persons.id"),
    nullable=True
)
```

**Current UI Features**:
1. **Tabs**: "Unlabeled" (default) and "All Clusters"
2. **Sorting**: By face count or average quality
3. **Pagination**: Load more button (100 items per page)
4. **Cluster Cards**: Show face count, sample faces (6 thumbnails), person badge
5. **Click Navigation**: Cluster card → detail view → label workflow

**Current Backend Logic** (`/api/v1/faces/clusters`):
- Aggregates `FaceInstance` by `cluster_id`
- Filters by `include_labeled` parameter
- Returns: `cluster_id`, `face_count`, `sample_face_ids`, `avg_quality`, `person_id`, `person_name`
- Uses PostgreSQL `array_agg()` for face IDs (SQLite: `group_concat()`)
- Uses PostgreSQL `bool_or()` to detect labeled clusters

---

## Desired State (After Conversion)

### Target User Experience

**Simplified View**:
```
┌─────────────────────────────────────────────────────────┐
│  Unknown Faces                                          │
│  Review and label groups of similar unidentified faces  │
└─────────────────────────────────────────────────────────┘

[NO TABS - Single clean grid]

Sort by: [Face Count ▼]     Showing 45 groups

┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ 23 faces    │  │ 18 faces    │  │ 12 faces    │
│ 👤👤👤👤👤👤  │  │ 👤👤👤👤👤👤  │  │ 👤👤👤👤👤👤  │
│ Quality: 82%│  │ Quality: 91%│  │ Quality: 74%│
│ Unlabeled   │  │ Unlabeled   │  │ Unlabeled   │
└─────────────┘  └─────────────┘  └─────────────┘
     ↓ Click to label
```

**Configuration in Admin/Settings**:
```
┌─────────────────────────────────────────────┐
│  Unknown Face Clustering Settings          │
├─────────────────────────────────────────────┤
│  Confidence Threshold: [========|] 85%      │
│  Only show groups with 85%+ similarity      │
│                                             │
│  Minimum Cluster Size: [5]                  │
│  Ignore groups with fewer than 5 faces      │
├─────────────────────────────────────────────┤
│  [Save Settings]  [Reset]                   │
└─────────────────────────────────────────────┘
```

### Key Behavioral Changes

**What Gets Removed**:
1. Tab navigation component (lines 136-157 in `+page.svelte`)
2. `activeTab` state variable and tab change logic
3. Display of labeled clusters (clusters with `person_id != NULL`)
4. "All Clusters" view entirely

**What Gets Simplified**:
1. Single grid view (no tab switching logic)
2. Always filters to `include_labeled=false`
3. Clearer page title: "Unknown Faces" instead of "Face Clusters"
4. Updated subtitle explaining purpose

**What Gets Added**:
1. **Backend**: Cluster confidence/similarity score calculation
2. **Backend**: Filter by confidence threshold
3. **Backend**: Filter by minimum cluster size
4. **Backend**: Settings storage for thresholds
5. **Frontend**: Admin settings UI for threshold configuration
6. **Frontend**: Representative face thumbnail extraction

---

## Architecture: Before vs After

### Before (Current Architecture)

```
┌──────────────────────────────────────────┐
│  Face Clusters Page                     │
├──────────────────────────────────────────┤
│  [Unlabeled] [All Clusters] ← Tabs      │
│                                          │
│  ┌────────────────────────────────┐     │
│  │  API: GET /faces/clusters      │     │
│  │  Params: include_labeled=true  │     │
│  └────────────────────────────────┘     │
│                                          │
│  Shows: ALL clusters (labeled +         │
│         unlabeled) when "All" tab       │
│         active                           │
└──────────────────────────────────────────┘
                  │
                  ▼
         Database Query
    ┌──────────────────────────┐
    │ SELECT cluster_id,       │
    │        person_id,        │
    │        COUNT(*) AS cnt   │
    │ FROM face_instances      │
    │ GROUP BY cluster_id      │
    │ -- No confidence filter  │
    └──────────────────────────┘
```

### After (Simplified Architecture)

```
┌──────────────────────────────────────────┐
│  Unknown Faces Page                     │
├──────────────────────────────────────────┤
│  NO TABS - Single Grid View             │
│                                          │
│  ┌────────────────────────────────┐     │
│  │  API: GET /faces/unknown       │     │
│  │  Params: min_confidence=0.85   │     │
│  │          min_cluster_size=5    │     │
│  └────────────────────────────────┘     │
│                                          │
│  Shows: ONLY unlabeled clusters with    │
│         confidence ≥ threshold           │
└──────────────────────────────────────────┘
                  │
                  ▼
       Enhanced Database Query
    ┌──────────────────────────────┐
    │ WITH cluster_stats AS (      │
    │   SELECT cluster_id,         │
    │          COUNT(*) AS cnt,    │
    │          AVG(confidence) AS  │
    │            cluster_conf      │
    │   FROM face_instances        │
    │   WHERE person_id IS NULL    │
    │   GROUP BY cluster_id        │
    │   HAVING COUNT(*) >= 5       │
    │      AND AVG(conf) >= 0.85   │
    │ )                            │
    └──────────────────────────────┘
                  │
                  ▼
         Admin Settings Store
    ┌──────────────────────────────┐
    │ Config Table/Endpoint        │
    │ - unknown_face_threshold     │
    │ - unknown_face_min_size      │
    └──────────────────────────────┘
```

---

## Detailed Implementation Plan

### Phase 1: Backend Changes

#### 1.1 Configuration Storage

**New Config Endpoint** (extend existing `/api/v1/config/face-matching`):

**File**: `/image-search-service/src/image_search_service/api/routes/config.py`

**Add to schema** (`api/schemas.py` or `api/config_schemas.py`):
```python
class UnknownFaceClusteringConfig(CamelCaseModel):
    """Configuration for unknown face clustering display."""
    min_confidence: float = Field(
        0.85,
        ge=0.0,
        le=1.0,
        description="Minimum intra-cluster confidence threshold"
    )
    min_cluster_size: int = Field(
        5,
        ge=1,
        le=100,
        description="Minimum number of faces required per cluster"
    )
```

**Extend existing config endpoint**:
```python
@router.get("/face-clustering-unknown", response_model=UnknownFaceClusteringConfig)
async def get_unknown_clustering_config():
    """Get configuration for unknown face clustering display."""
    return UnknownFaceClusteringConfig(
        min_confidence=settings.unknown_face_cluster_confidence,
        min_cluster_size=settings.unknown_face_cluster_min_size,
    )

@router.put("/face-clustering-unknown", response_model=UnknownFaceClusteringConfig)
async def update_unknown_clustering_config(
    config: UnknownFaceClusteringConfig
):
    """Update configuration for unknown face clustering."""
    # Store in database config table or settings file
    # For now, can use in-memory or extend Config model
    settings.unknown_face_cluster_confidence = config.min_confidence
    settings.unknown_face_cluster_min_size = config.min_cluster_size
    return config
```

**Add to settings** (`core/config.py`):
```python
class Settings(BaseSettings):
    # ... existing settings ...

    # Unknown face clustering display
    unknown_face_cluster_confidence: float = 0.85
    unknown_face_cluster_min_size: int = 5
```

#### 1.2 Cluster Confidence Calculation

**Problem**: Current clustering doesn't store intra-cluster confidence scores.

**Solution Options**:

**Option A: Calculate on-the-fly** (Recommended for MVP)
- During cluster listing, query Qdrant for all face embeddings in cluster
- Calculate average pairwise cosine similarity
- Return as `cluster_confidence` field
- **Pros**: No schema migration, accurate real-time data
- **Cons**: Slightly slower query (mitigated by caching)

**Option B: Pre-compute and store**
- Add `cluster_confidence` column to `FaceInstance` or new `Cluster` table
- Compute during clustering job
- **Pros**: Faster queries
- **Cons**: Requires migration, can drift from reality

**Recommendation**: Start with **Option A** for faster iteration.

**Implementation** (Option A):

**File**: `/image-search-service/src/image_search_service/api/routes/faces.py`

**Add helper function**:
```python
def calculate_cluster_confidence(
    qdrant: FaceQdrantClient,
    face_qdrant_point_ids: list[int],
) -> float:
    """Calculate average pairwise similarity within a cluster."""
    if len(face_qdrant_point_ids) < 2:
        return 1.0  # Single face = perfect confidence

    embeddings = [
        qdrant.get_embedding_by_point_id(pid)
        for pid in face_qdrant_point_ids
    ]

    # Calculate pairwise cosine similarities
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            similarities.append(sim)

    return sum(similarities) / len(similarities)
```

**Update endpoint**:
```python
@router.get("/clusters", response_model=ClusterListResponse)
async def list_clusters(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    include_labeled: bool = Query(False),
    min_confidence: float | None = Query(None, ge=0.0, le=1.0),
    min_cluster_size: int | None = Query(None, ge=1),
    db: AsyncSession = Depends(get_db),
) -> ClusterListResponse:
    """List face clusters with optional confidence filtering."""

    # ... existing aggregation query ...

    # Add HAVING clause for min_cluster_size
    if min_cluster_size:
        cluster_query = cluster_query.having(
            func.count(FaceInstance.id) >= min_cluster_size
        )

    # ... execute query ...

    # Post-process: calculate confidence and filter
    filtered_items = []
    for row in rows:
        # Get embeddings for confidence calculation
        if min_confidence:
            qdrant_ids = [...]  # Extract from row
            confidence = calculate_cluster_confidence(qdrant, qdrant_ids)
            if confidence < min_confidence:
                continue
        else:
            confidence = None

        filtered_items.append(
            ClusterSummary(
                cluster_id=row.cluster_id,
                face_count=row.face_count,
                sample_face_ids=face_ids_list[:5],
                avg_quality=row.avg_quality,
                cluster_confidence=confidence,  # NEW FIELD
                person_id=person_id,
                person_name=person_name,
            )
        )

    return ClusterListResponse(items=filtered_items, total=len(filtered_items), ...)
```

**Update schema** (`api/face_schemas.py`):
```python
class ClusterSummary(CamelCaseModel):
    """Summary of a face cluster."""
    cluster_id: str
    face_count: int
    sample_face_ids: list[UUID]
    avg_quality: float | None = None
    cluster_confidence: float | None = None  # NEW: Intra-cluster similarity
    person_id: UUID | None = None
    person_name: str | None = None
```

#### 1.3 Representative Face Thumbnail Selection

**Current State**: `sample_face_ids` returns up to 5 random face IDs

**Desired State**: Select the **highest quality** face as the primary representative

**Change**:
```python
# Current (in list_clusters endpoint):
sample_face_ids=face_ids_list[:5]

# After:
# Sort by quality_score descending before slicing
sorted_faces = sorted(
    zip(face_ids_list, quality_scores),
    key=lambda x: x[1] or 0,
    reverse=True
)
sample_face_ids = [fid for fid, _ in sorted_faces[:5]]
representative_face_id = sample_face_ids[0] if sample_face_ids else None
```

**Add to response schema**:
```python
class ClusterSummary(CamelCaseModel):
    # ... existing fields ...
    representative_face_id: UUID | None = None  # Highest quality face
```

**Frontend will use**: `/api/v1/images/{asset_id}/face-thumbnail?face_id={representative_face_id}`

---

### Phase 2: Frontend Changes

#### 2.1 Remove Tab Navigation

**File**: `/image-search-ui/src/routes/faces/clusters/+page.svelte`

**Delete**:
- Lines 18-19: `let activeTab = $state<'unlabeled' | 'all'>('unlabeled');`
- Lines 44-63: Tab change effect and tracking
- Lines 110-113: `handleTabChange()` function
- Lines 136-157: Tab navigation HTML

**Simplify**:
```svelte
<script lang="ts">
  // Remove activeTab state entirely
  // Remove previousTab tracking

  // Hardcode behavior to unlabeled only
  async function loadClusters(reset: boolean = false) {
    // ...
    const includeLabeled = false; // Always false
    const response = await listClusters(currentPage, PAGE_SIZE, includeLabeled);
    // ...
  }
</script>

<main class="clusters-page">
  <header class="page-header">
    <h1>Unknown Faces</h1>
    <p class="subtitle">
      Review and label groups of similar unidentified faces.
      These are face clusters that haven't been assigned to any person yet.
    </p>
  </header>

  <!-- NO TABS - directly show content -->
  <section class="content" aria-live="polite">
    <!-- ... existing loading/error/empty/grid states ... -->
  </section>
</main>
```

**Update empty state message**:
```svelte
<div class="empty-state">
  <h2>No unknown faces found</h2>
  <p>
    All face clusters have been labeled! Great work identifying everyone in your photos.
  </p>
</div>
```

#### 2.2 Update Cluster Card Component

**File**: `/image-search-ui/src/lib/components/faces/ClusterCard.svelte`

**Changes**:
1. Remove labeled/unlabeled badge logic (always unlabeled)
2. Add confidence display (if available)
3. Use representative face thumbnail

**Updated card header**:
```svelte
<div class="card-header">
  <div class="cluster-info">
    <span class="face-count">{cluster.faceCount} faces</span>
    {#if cluster.clusterConfidence}
      <span class="confidence-badge">
        {(cluster.clusterConfidence * 100).toFixed(0)}% similar
      </span>
    {/if}
  </div>
  <span class="cluster-id" title={cluster.clusterId}>
    {shortenClusterId(cluster.clusterId)}
  </span>
</div>
```

**Remove person badge logic**:
```svelte
<!-- DELETE THIS BLOCK: -->
{#if cluster.personName}
  <span class="person-badge">{cluster.personName}</span>
{:else}
  <span class="unlabeled-badge">Unlabeled</span>
{/if}
```

#### 2.3 Admin Settings Integration

**File**: `/image-search-ui/src/lib/components/admin/FaceMatchingSettings.svelte`

**Add new section** (after Prototype Settings):
```svelte
<div class="other-settings">
  <h3>Unknown Face Clustering</h3>
  <p class="section-description">
    Control which face clusters appear in the Unknown Faces view.
    Higher thresholds show only very similar faces, reducing noise.
  </p>

  <div class="form-grid">
    <div class="form-field">
      <label for="unknownConfidenceThreshold">
        Minimum Cluster Confidence
        <span class="field-hint">
          Only show clusters with this average similarity (0-100%)
        </span>
      </label>
      <div class="slider-container">
        <input
          id="unknownConfidenceThreshold"
          type="range"
          min="0.5"
          max="1.0"
          step="0.05"
          bind:value={unknownClusterConfig.minConfidence}
        />
        <output class="slider-value">
          {(unknownClusterConfig.minConfidence * 100).toFixed(0)}%
        </output>
      </div>
    </div>

    <div class="form-field">
      <label for="unknownMinClusterSize">
        Minimum Faces Per Cluster
        <span class="field-hint">
          Only show clusters with at least this many faces
        </span>
      </label>
      <input
        id="unknownMinClusterSize"
        type="number"
        min="1"
        max="50"
        bind:value={unknownClusterConfig.minClusterSize}
      />
    </div>
  </div>
</div>
```

**Add state**:
```svelte
<script lang="ts">
  let unknownClusterConfig = $state({
    minConfidence: 0.85,
    minClusterSize: 5
  });

  async function loadConfig() {
    // ... existing code ...

    // Load unknown clustering config
    const unknownResponse = await fetch(
      `${API_BASE_URL}/api/v1/config/face-clustering-unknown`
    );
    if (unknownResponse.ok) {
      const unknownData = await unknownResponse.json();
      unknownClusterConfig = {
        minConfidence: unknownData.minConfidence,
        minClusterSize: unknownData.minClusterSize
      };
    }
  }

  async function saveConfig() {
    // ... existing code ...

    // Save unknown clustering config
    await fetch(`${API_BASE_URL}/api/v1/config/face-clustering-unknown`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        min_confidence: unknownClusterConfig.minConfidence,
        min_cluster_size: unknownClusterConfig.minClusterSize
      })
    });
  }
</script>
```

#### 2.4 Update API Client

**File**: `/image-search-ui/src/lib/api/faces.ts`

**Update `listClusters` function**:
```typescript
export async function listClusters(
  page: number = 1,
  pageSize: number = 20,
  includeLabeled: boolean = false,
  minConfidence?: number,
  minClusterSize?: number
): Promise<ClusterListResponse> {
  const params = new URLSearchParams({
    page: page.toString(),
    page_size: pageSize.toString(),
    include_labeled: includeLabeled.toString()
  });

  if (minConfidence !== undefined) {
    params.append('min_confidence', minConfidence.toString());
  }
  if (minClusterSize !== undefined) {
    params.append('min_cluster_size', minClusterSize.toString());
  }

  return apiRequest<ClusterListResponse>(
    `/api/v1/faces/clusters?${params.toString()}`
  );
}
```

**Update type definition** (`src/lib/types.ts`):
```typescript
export interface ClusterSummary {
  clusterId: string;
  faceCount: number;
  sampleFaceIds: string[];
  avgQuality: number | null;
  clusterConfidence?: number; // NEW
  representativeFaceId?: string; // NEW
  personId: string | null;
  personName: string | null;
}
```

#### 2.5 Load Settings and Apply Filters

**File**: `/image-search-ui/src/routes/faces/clusters/+page.svelte`

**Add settings state**:
```svelte
<script lang="ts">
  let clusterSettings = $state({
    minConfidence: 0.85,
    minClusterSize: 5
  });

  onMount(async () => {
    await loadSettings();
    loadClusters(true);
  });

  async function loadSettings() {
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/v1/config/face-clustering-unknown`
      );
      if (response.ok) {
        const data = await response.json();
        clusterSettings = {
          minConfidence: data.minConfidence,
          minClusterSize: data.minClusterSize
        };
      }
    } catch (err) {
      console.warn('Failed to load cluster settings, using defaults');
    }
  }

  async function loadClusters(reset: boolean = false) {
    // ... existing code ...

    const response = await listClusters(
      currentPage,
      PAGE_SIZE,
      false, // Never include labeled
      clusterSettings.minConfidence,
      clusterSettings.minClusterSize
    );

    // ... rest of existing code ...
  }
</script>
```

---

### Phase 3: Migration & Rollout

#### 3.1 Database Migration

**No migration required** for Option A (on-the-fly confidence calculation).

If implementing Option B (stored confidence):
```sql
-- migration: add_cluster_confidence_20250130.sql
ALTER TABLE face_instances
  ADD COLUMN cluster_confidence REAL;

CREATE INDEX idx_face_instances_cluster_confidence
  ON face_instances(cluster_confidence)
  WHERE cluster_id IS NOT NULL;
```

#### 3.2 Backward Compatibility

**API Contract Changes**:
- **Non-breaking**: New optional query params (`min_confidence`, `min_cluster_size`)
- **Non-breaking**: New optional response fields (`clusterConfidence`, `representativeFaceId`)
- **Breaking**: Frontend no longer uses `include_labeled=true` (but API still supports it)

**Deprecation Plan**:
1. Keep existing endpoint behavior (default `include_labeled=false`)
2. Frontend stops using `include_labeled=true` entirely
3. Future: Consider deprecating `include_labeled` param or creating separate `/faces/unknown` endpoint

#### 3.3 Rollout Checklist

**Backend**:
- [ ] Add settings to `core/config.py`
- [ ] Create config endpoint (`/api/v1/config/face-clustering-unknown`)
- [ ] Add confidence calculation helper
- [ ] Update `/faces/clusters` endpoint with filtering
- [ ] Update `ClusterSummary` schema
- [ ] Add tests for confidence calculation
- [ ] Add tests for filtering logic
- [ ] Update API contract documentation

**Frontend**:
- [ ] Remove tab navigation from `+page.svelte`
- [ ] Update page title and subtitle
- [ ] Simplify cluster loading logic
- [ ] Update `ClusterCard.svelte` (remove person badges)
- [ ] Add settings UI to `FaceMatchingSettings.svelte`
- [ ] Update `listClusters()` API client
- [ ] Update TypeScript types
- [ ] Run `npm run gen:api` after backend changes
- [ ] Test settings persistence
- [ ] Test cluster filtering behavior

**Testing**:
- [ ] Unit tests: Confidence calculation accuracy
- [ ] Unit tests: Filtering by min_confidence and min_cluster_size
- [ ] Integration tests: Settings CRUD operations
- [ ] E2E tests: Unknown faces workflow
- [ ] Manual QA: Verify no labeled clusters appear
- [ ] Manual QA: Verify threshold changes take effect

**Documentation**:
- [ ] Update `docs/api-contract.md` with new fields/params
- [ ] Update user guide (if exists) with new workflow
- [ ] Add migration notes for existing users

---

## Configuration Management Details

### Default Values

**Backend Defaults** (`core/config.py`):
```python
UNKNOWN_FACE_CLUSTER_CONFIDENCE: float = 0.85  # 85% similarity
UNKNOWN_FACE_CLUSTER_MIN_SIZE: int = 5         # At least 5 faces
```

**Rationale**:
- **85% confidence**: Balances precision (avoid false positives) with recall (don't miss valid clusters)
- **5 faces minimum**: Avoids noise from small random groupings, focuses on meaningful patterns

### Storage Strategy

**Option A: Environment Variables** (Simpler, recommended for MVP)
- Store in `core/config.py` as `Settings` fields
- Load from environment variables
- Persist via config file or `.env`

**Option B: Database Configuration Table** (More flexible)
- Create `app_config` table with key-value pairs
- Allows runtime updates without restart
- Requires migration

**Recommended**: Start with **Option A**, migrate to **Option B** if multi-tenancy needed.

---

## Representative Thumbnail Selection

### Current Behavior
- `sample_face_ids` returns first 5 face IDs from cluster
- Frontend displays all 6 sample faces in grid
- No prioritization by quality

### New Behavior
- Backend sorts faces by `quality_score` descending
- Returns `representative_face_id` as highest quality face
- Frontend displays representative face prominently

### Implementation

**Backend** (in `list_clusters` endpoint):
```python
# Before returning ClusterSummary:
faces_with_quality = [
    (fid, quality)
    for fid, quality in zip(face_ids_list, quality_scores)
]
faces_sorted = sorted(
    faces_with_quality,
    key=lambda x: x[1] or 0,
    reverse=True
)

representative_face_id = faces_sorted[0][0] if faces_sorted else None
sample_face_ids = [fid for fid, _ in faces_sorted[:5]]
```

**Frontend** (`ClusterCard.svelte`):
```svelte
<div class="representative-thumbnail">
  {#if cluster.representativeFaceId}
    <FaceThumbnail
      faceId={cluster.representativeFaceId}
      size={96}
      alt="Representative face"
    />
  {/if}
</div>

<div class="sample-thumbnails">
  {#each cluster.sampleFaceIds.slice(1, 5) as faceId}
    <FaceThumbnail {faceId} size={48} />
  {/each}
</div>
```

---

## Risk Analysis & Mitigation

### Risk 1: Confidence Calculation Performance

**Risk**: Computing pairwise similarities on-the-fly may be slow for large clusters.

**Impact**: Page load times increase, poor UX.

**Mitigation**:
1. **Limit cluster size**: Only calculate for clusters with ≤ 50 faces
2. **Caching**: Cache confidence scores in Redis (TTL: 1 hour)
3. **Async computation**: Pre-compute in background job, store in DB
4. **Sampling**: Calculate confidence on 10 random face pairs (not all pairs)

**Monitoring**: Add performance logging to track calculation times.

### Risk 2: Users Miss Labeled Clusters

**Risk**: Removing "All Clusters" tab means users can't review all clusters.

**Impact**: Loss of visibility into labeled data.

**Mitigation**:
1. **Document change**: Communicate that labeled clusters now appear in "People" view
2. **Add link**: Include navigation hint: "To review labeled faces, visit the People page"
3. **Preserve detail view**: Clicking a cluster from old bookmarks still works

### Risk 3: Threshold Too Restrictive

**Risk**: Default 85% confidence filters out valid clusters.

**Impact**: Users don't see faces they should label.

**Mitigation**:
1. **Configurable defaults**: Allow admins to adjust threshold
2. **Show counts**: Display "X clusters hidden by confidence threshold"
3. **Quick toggle**: Add UI toggle to temporarily lower threshold
4. **Analytics**: Log how many clusters are filtered to tune defaults

### Risk 4: Breaking Existing Workflows

**Risk**: Users accustomed to tab navigation get confused.

**Impact**: Support burden, user frustration.

**Mitigation**:
1. **Release notes**: Clear communication about UI simplification
2. **In-app message**: Show one-time banner explaining changes
3. **Preserve URLs**: Old URLs redirect to new single view
4. **Phased rollout**: Deploy to staging first, gather feedback

---

## Testing Strategy

### Unit Tests (Backend)

**File**: `tests/api/test_faces_clusters.py`

```python
def test_cluster_confidence_calculation():
    """Test pairwise similarity calculation for cluster confidence."""
    # Given: 3 face embeddings with known similarities
    embeddings = [np.array([1, 0, 0]), np.array([0.9, 0.1, 0]), ...]

    # When: Calculate cluster confidence
    confidence = calculate_cluster_confidence(mock_qdrant, point_ids)

    # Then: Confidence matches expected average pairwise similarity
    assert 0.85 <= confidence <= 0.95

def test_list_clusters_min_confidence_filter():
    """Test that clusters below confidence threshold are filtered."""
    # Given: Database with 2 clusters (confidence 0.90 and 0.70)
    # When: Request clusters with min_confidence=0.85
    response = await list_clusters(min_confidence=0.85)

    # Then: Only high-confidence cluster returned
    assert len(response.items) == 1
    assert response.items[0].cluster_confidence >= 0.85

def test_list_clusters_min_size_filter():
    """Test that small clusters are filtered."""
    # Given: Clusters with 3, 8, 15 faces
    # When: Request with min_cluster_size=5
    response = await list_clusters(min_cluster_size=5)

    # Then: Only clusters with ≥5 faces returned
    assert all(c.face_count >= 5 for c in response.items)
```

### Integration Tests

```python
async def test_unknown_clustering_config_crud():
    """Test configuration CRUD operations."""
    # Create/Update
    response = await client.put("/api/v1/config/face-clustering-unknown", json={
        "min_confidence": 0.90,
        "min_cluster_size": 10
    })
    assert response.status_code == 200

    # Read
    response = await client.get("/api/v1/config/face-clustering-unknown")
    config = response.json()
    assert config["minConfidence"] == 0.90
    assert config["minClusterSize"] == 10
```

### Frontend Tests

**File**: `src/tests/routes/face-clusters.test.ts`

```typescript
test('renders unknown faces view without tabs', async () => {
  mockResponse('/api/v1/faces/clusters', { items: [], total: 0 });

  render(FaceClustersPage);

  // Should NOT have tab navigation
  expect(screen.queryByRole('tab', { name: /all clusters/i })).not.toBeInTheDocument();
  expect(screen.queryByRole('tab', { name: /unlabeled/i })).not.toBeInTheDocument();
});

test('applies confidence and size filters from settings', async () => {
  mockResponse('/api/v1/config/face-clustering-unknown', {
    minConfidence: 0.90,
    minClusterSize: 8
  });
  mockResponse('/api/v1/faces/clusters?min_confidence=0.90&min_cluster_size=8', {
    items: mockClusters,
    total: 2
  });

  render(FaceClustersPage);
  await waitFor(() => {
    expect(screen.getByText(/2 groups/i)).toBeInTheDocument();
  });

  // Verify correct API call
  assertCalled('/api/v1/faces/clusters', {
    query: expect.objectContaining({
      min_confidence: '0.90',
      min_cluster_size: '8'
    })
  });
});
```

---

## Rollback Plan

### Trigger Conditions
- Confidence calculation causes >2s page load times
- >20% of clusters disappear due to overly restrictive threshold
- Critical bug in filtering logic

### Rollback Steps
1. **Backend**: Revert endpoint changes, remove filtering params
2. **Frontend**: Revert to tab navigation UI
3. **Config**: Remove unknown clustering settings
4. **Database**: No rollback needed (no schema changes)

### Rollback Testing
- Verify tabs reappear correctly
- Verify all clusters visible again
- Verify labeling workflow still functions

---

## Future Enhancements (Out of Scope)

### Phase 2 Improvements
1. **Cluster merging**: Allow users to merge similar clusters
2. **Cluster splitting**: Auto-split low-confidence clusters
3. **Confidence heatmap**: Visual indicator of similarity distribution
4. **Face similarity search**: "Find more like this face"
5. **Batch labeling**: Select multiple clusters, label to same person

### Advanced Features
1. **Temporal filtering**: Show clusters from specific date ranges
2. **Quality filtering**: Additional filter by face quality score
3. **Smart suggestions**: "These 3 clusters might be the same person"
4. **Export clusters**: Download cluster data for external analysis

---

## Success Metrics

### Performance Metrics
- Page load time: < 2 seconds (p95)
- Confidence calculation: < 500ms per cluster (p95)
- Settings update latency: < 200ms

### Usage Metrics
- % of clusters labeled within 30 days of detection
- Average time to label a cluster (click → save)
- Number of clusters filtered by confidence threshold

### Quality Metrics
- False positive rate: Clusters with mixed people < 5%
- User satisfaction: Survey rating ≥ 4/5 for new UI
- Support tickets: Decrease by 30% (due to simpler UI)

---

## Timeline Estimate

**Total: 2-3 weeks** (1 backend engineer + 1 frontend engineer)

| Phase | Tasks | Time |
|-------|-------|------|
| **Backend** | Config endpoint, confidence calculation, filtering | 3-4 days |
| **Frontend** | Remove tabs, update UI, settings integration | 3-4 days |
| **Testing** | Unit tests, integration tests, QA | 2-3 days |
| **Documentation** | API contract, user guide, migration notes | 1 day |
| **Review & Deploy** | Code review, staging deployment, monitoring | 1-2 days |

---

## Approval & Sign-off

- [ ] Product Owner: Approve simplified UI approach
- [ ] Engineering Lead: Approve technical architecture
- [ ] QA Lead: Approve test coverage plan
- [ ] Design Review: Approve updated UI mockups (if needed)

---

## Appendix

### A. API Contract Diff

**Before**:
```yaml
/api/v1/faces/clusters:
  get:
    parameters:
      - name: include_labeled
        type: boolean
        default: false
    responses:
      200:
        schema: ClusterListResponse
          items:
            - clusterId: string
              faceCount: int
              avgQuality: float
              personId: uuid | null
```

**After**:
```yaml
/api/v1/faces/clusters:
  get:
    parameters:
      - name: include_labeled
        type: boolean
        default: false
      - name: min_confidence         # NEW
        type: float
        required: false
      - name: min_cluster_size       # NEW
        type: int
        required: false
    responses:
      200:
        schema: ClusterListResponse
          items:
            - clusterId: string
              faceCount: int
              avgQuality: float
              clusterConfidence: float | null    # NEW
              representativeFaceId: uuid | null  # NEW
              personId: uuid | null
```

### B. UI Component Tree Changes

**Before**:
```
FaceClustersPage
├── Tabs [REMOVED]
│   ├── Unlabeled Tab
│   └── All Clusters Tab
├── Results Header
├── Clusters Grid
│   └── ClusterCard (x N)
│       ├── Face Count
│       ├── Person Badge OR Unlabeled Badge
│       └── Sample Faces (6)
└── Load More Button
```

**After**:
```
UnknownFacesPage
├── Results Header
├── Clusters Grid
│   └── ClusterCard (x N)
│       ├── Face Count
│       ├── Confidence Badge [NEW]
│       ├── Representative Face [NEW]
│       └── Sample Faces (4)
└── Load More Button

AdminSettings
└── FaceMatchingSettings
    └── Unknown Clustering Section [NEW]
        ├── Confidence Threshold Slider
        └── Min Cluster Size Input
```

### C. Database Query Comparison

**Before** (simple aggregation):
```sql
SELECT
  cluster_id,
  COUNT(*) AS face_count,
  AVG(quality_score) AS avg_quality,
  ARRAY_AGG(id) AS face_ids,
  BOOL_OR(person_id IS NOT NULL) AS has_person
FROM face_instances
WHERE cluster_id IS NOT NULL
GROUP BY cluster_id
HAVING BOOL_OR(person_id IS NOT NULL) IS NOT TRUE
LIMIT 100;
```

**After** (with filtering):
```sql
WITH cluster_agg AS (
  SELECT
    cluster_id,
    COUNT(*) AS face_count,
    AVG(quality_score) AS avg_quality,
    ARRAY_AGG(id ORDER BY quality_score DESC) AS face_ids,
    ARRAY_AGG(qdrant_point_id) AS qdrant_ids
  FROM face_instances
  WHERE cluster_id IS NOT NULL
    AND person_id IS NULL
  GROUP BY cluster_id
  HAVING COUNT(*) >= 5  -- min_cluster_size filter
)
SELECT * FROM cluster_agg
LIMIT 100;

-- Then in Python: calculate confidence, filter by min_confidence
```

---

**End of Conversion Plan**

This plan provides a comprehensive roadmap for converting the Face Clusters view into a simplified, unknown-faces-only interface. The conversion maintains existing labeling workflows while reducing UI complexity and adding intelligent filtering based on cluster confidence.
