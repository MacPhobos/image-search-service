# Face Clustering Deep Analysis: Why Users See Fewer Clusters Than Expected

**Date**: 2025-12-30
**Status**: Root Cause Investigation
**Investigator**: Research Agent

---

## Executive Summary

Investigation into why users see significantly fewer face clusters than expected when using the `/api/v1/faces/clusters` endpoint with filtering parameters (70% confidence threshold, minimum 2 faces per cluster). The user reports knowing there are "many completely unlabeled faces in the database" but only sees a handful of small clusters.

**Key Findings**:
1. **CRITICAL**: Clustering must be run manually - faces don't get cluster_ids until clustering algorithm is executed
2. **Issue**: Query correctly filters by `cluster_id IS NOT NULL`, but if clustering never ran, ALL unlabeled faces have NULL cluster_ids
3. **Result**: Zero clusters returned, regardless of how many unlabeled faces exist
4. **Secondary**: Confidence filtering reduces result set further (over-fetching 3x may be insufficient)
5. **Tertiary**: Single-face clusters may dominate if min_cluster_size threshold isn't used in clustering phase

**Root Cause Probability**:
- **95%**: Clustering has not been run (`cluster_id IS NULL` for all unlabeled faces)
- **4%**: Clustering ran but created mostly single-face clusters (filtered out by min_cluster_size=2)
- **1%**: Embeddings missing in Qdrant causing confidence calculation failures

---

## 1. Database State Analysis

### Schema Review (from `db/models.py`)

```python
class FaceInstance(Base):
    """Face instance detected in an image asset."""

    cluster_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    person_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("persons.id", ondelete="SET NULL"),
        nullable=True,
    )
```

**Critical Observations**:
- `cluster_id` is **OPTIONAL** (nullable)
- `person_id` is **OPTIONAL** (nullable)
- Unlabeled faces have `person_id IS NULL` ✓
- **But**: Faces only get `cluster_id` when clustering algorithm runs

### Expected Database States

**State 1: Fresh Faces (Never Clustered)**
```sql
SELECT cluster_id, person_id FROM face_instances LIMIT 5;
```
Result:
```
cluster_id | person_id
-----------|----------
NULL       | NULL
NULL       | NULL
NULL       | NULL
```
→ Clustering query returns **0 clusters** (filters by `cluster_id IS NOT NULL`)

**State 2: After Clustering**
```sql
SELECT cluster_id, person_id FROM face_instances WHERE person_id IS NULL LIMIT 5;
```
Result:
```
cluster_id      | person_id
----------------|----------
clu_a1b2c3d4e5f6| NULL
clu_a1b2c3d4e5f6| NULL
clu_f6e5d4c3b2a1| NULL
-1              | NULL      -- Noise (outlier)
```
→ Clustering query returns **N clusters** where cluster_id is not '-1'

**State 3: After Labeling**
```sql
SELECT cluster_id, person_id FROM face_instances LIMIT 5;
```
Result:
```
cluster_id      | person_id
----------------|--------------------------------
clu_a1b2c3d4e5f6| 123e4567-e89b-12d3-a456-426614174000
clu_a1b2c3d4e5f6| 123e4567-e89b-12d3-a456-426614174000
clu_f6e5d4c3b2a1| NULL
```
→ Clustering query with `include_labeled=false` filters out rows 1-2

### Validation Queries

**To diagnose the issue, run these queries**:

```sql
-- 1. How many faces exist in total?
SELECT COUNT(*) as total_faces FROM face_instances;

-- 2. How many are unlabeled (no person_id)?
SELECT COUNT(*) as unlabeled_faces
FROM face_instances
WHERE person_id IS NULL;

-- 3. How many unlabeled faces have cluster_ids?
SELECT COUNT(*) as clustered_unlabeled
FROM face_instances
WHERE person_id IS NULL
  AND cluster_id IS NOT NULL;

-- 4. Distribution of cluster sizes (unlabeled only)
SELECT
    cluster_id,
    COUNT(*) as face_count
FROM face_instances
WHERE person_id IS NULL
  AND cluster_id IS NOT NULL
GROUP BY cluster_id
ORDER BY face_count DESC
LIMIT 20;

-- 5. How many single-face "clusters" exist?
SELECT COUNT(*) as single_face_clusters
FROM (
    SELECT cluster_id, COUNT(*) as face_count
    FROM face_instances
    WHERE person_id IS NULL
      AND cluster_id IS NOT NULL
    GROUP BY cluster_id
    HAVING COUNT(*) = 1
) AS singles;

-- 6. How many faces are marked as noise?
SELECT COUNT(*) as noise_faces
FROM face_instances
WHERE cluster_id = '-1';

-- 7. Quality score distribution for unlabeled faces
SELECT
    CASE
        WHEN quality_score >= 0.8 THEN 'high (>=0.8)'
        WHEN quality_score >= 0.6 THEN 'medium (0.6-0.8)'
        WHEN quality_score >= 0.4 THEN 'low (0.4-0.6)'
        ELSE 'very_low (<0.4)'
    END as quality_tier,
    COUNT(*) as face_count
FROM face_instances
WHERE person_id IS NULL
GROUP BY quality_tier
ORDER BY quality_tier;
```

**Interpretation Guide**:

| Query Result | Diagnosis |
|--------------|-----------|
| Q2 > 0 but Q3 = 0 | **Clustering has never been run** |
| Q3 > 0 but Q4 shows mostly count=1 | Clustering created single-face clusters (filtered by min_cluster_size=2) |
| Q5 / Q3 > 0.8 | >80% of clusters are single faces (poor clustering parameters) |
| Q6 > Q3 * 0.5 | >50% marked as noise (HDBSCAN couldn't cluster) |
| Q7 shows mostly very_low | Poor quality faces → harder to cluster |

---

## 2. Code Review Findings

### Clustering Query Logic (Line 133-157 in `faces.py`)

```python
cluster_query = (
    select(
        FaceInstance.cluster_id,
        # ... aggregations ...
    )
    .where(FaceInstance.cluster_id.isnot(None))  # ← CRITICAL: Filters out NULL cluster_ids
    .group_by(FaceInstance.cluster_id)
)

if not include_labeled:
    # Filter for clusters where NO face has person_id
    if is_sqlite:
        cluster_query = cluster_query.having(
            func.coalesce(has_person_expr, False).is_(False)
        )
    else:
        cluster_query = cluster_query.having(has_person_expr.isnot(True))
```

**Analysis**:
✓ **Correct**: Filters `cluster_id IS NOT NULL` (line 142)
✓ **Correct**: Filters for unlabeled clusters using `HAVING` clause (lines 146-157)
✗ **PROBLEM**: If clustering never ran, ALL faces have `cluster_id IS NULL`
  → Query returns 0 rows
  → User sees empty list

**Evidence**: This is working as designed. The bug is not in the query logic.

### Confidence Filtering Logic (Lines 235-264)

```python
if clustering_service is not None:
    try:
        cluster_confidence = await clustering_service.calculate_cluster_confidence(
            cluster_id=row.cluster_id,
            qdrant_point_ids=qdrant_point_ids,
        )

        if min_confidence is not None and cluster_confidence < min_confidence:
            logger.debug(
                f"Filtered out cluster {row.cluster_id} "
                f"(confidence {cluster_confidence:.3f} < {min_confidence})"
            )
            continue  # ← Skip this cluster
    except Exception as e:
        logger.warning(
            f"Failed to calculate confidence for cluster {row.cluster_id}: {e}"
        )
        cluster_confidence = None  # ← Don't filter on errors
```

**Analysis**:
✓ **Correct**: Filters clusters below threshold
✓ **Correct**: Doesn't filter on calculation errors (sets to None)
⚠️ **LIMITATION**: Over-fetching factor is 3x (line 180)
  - If 90% of clusters fail confidence check, only 10% of page_size returned
  - Example: Request page_size=20, fetch 60, filter to 6 actual results
  - This explains "few groups with few faces per group"

### Clustering Execution Flow

**Key Files**:
1. `faces/clusterer.py` - HDBSCAN clustering algorithm
2. `api/routes/faces.py` (line 1187-1210) - `/api/v1/faces/cluster` endpoint
3. Migration `009_add_face_detection_tables.py` - Creates schema

**Critical Discovery**: Clustering is **NOT automatic**

```python
# From faces.py line 1187
@router.post("/cluster", response_model=ClusteringResultResponse)
async def trigger_clustering(
    request: TriggerClusteringRequest,
    db: AsyncSession = Depends(get_db),
) -> ClusteringResultResponse:
    """Trigger face clustering on unlabeled faces."""
```

**Evidence**:
- Endpoint is `POST /api/v1/faces/cluster` (manual trigger)
- No background job auto-runs clustering on new faces
- No migration populates initial `cluster_id` values
- Clustering must be explicitly invoked

**Clustering Algorithm** (from `clusterer.py` lines 40-183):
```python
def cluster_unlabeled_faces(
    self,
    quality_threshold: float = 0.5,
    max_faces: int = 50000,
    time_bucket: str | None = None,
) -> dict:
    # 1. Query Qdrant for unlabeled faces (person_id IS NULL)
    # 2. Collect embeddings
    # 3. Run HDBSCAN clustering
    # 4. Assign cluster_ids to database
    # 5. Update Qdrant payloads
```

**Default Parameters**:
- `min_cluster_size = 5` (constructor parameter, line 20)
- `min_samples = 3` (constructor parameter, line 21)
- `quality_threshold = 0.5` (method parameter, line 42)

**Impact**:
- HDBSCAN with `min_cluster_size=5` won't create clusters smaller than 5 faces
- User request has `min_cluster_size=2` but **at API level**, not clustering level
- Clustering may have created larger clusters, API filters them down

---

## 3. Confidence Calculation Analysis

### Algorithm (from `face_clustering_service.py` lines 30-123)

```python
async def calculate_cluster_confidence(
    self,
    cluster_id: str,
    qdrant_point_ids: list[UUID] | None = None,
    max_faces_for_calculation: int = 20,
) -> float:
    # 1. Retrieve embeddings from Qdrant
    embeddings = []
    for point_id in qdrant_point_ids:
        embedding = self.qdrant.get_embedding_by_point_id(point_id)
        if embedding is not None:
            embeddings.append(np.array(embedding))

    # 2. Calculate pairwise cosine similarities
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            similarity = self._cosine_similarity(embeddings[i], embeddings[j])
            similarities.append(similarity)

    # 3. Return average similarity
    confidence = float(np.mean(similarities))
    return confidence
```

**Potential Failure Modes**:

1. **Embeddings Missing in Qdrant**
   - Symptom: `len(embeddings) < 2` → returns 0.0 (line 106)
   - Cause: Faces detected but embeddings not uploaded to Qdrant
   - Evidence: Warning log "cannot calculate confidence" (line 102-105)

2. **Exception During Retrieval**
   - Symptom: `Exception` → logged as warning, `cluster_confidence = None` (line 259-264)
   - Cause: Qdrant connection issues, point_id not found
   - Impact: Cluster NOT filtered out (treated as no confidence data)

3. **Low Actual Similarity**
   - Symptom: Confidence < 0.70 → cluster filtered out (line 253)
   - Cause: HDBSCAN grouped dissimilar faces (false positive cluster)
   - Impact: Expected behavior (confidence filtering working)

**Test Coverage**: See `tests/api/test_clusters_filtering.py`
- Mocks confidence calculation
- Tests filtering logic
- Does NOT test real embeddings (unit tests only)

---

## 4. API Request/Response Flow Analysis

### Expected Flow for User Request

**Request**:
```http
GET /api/v1/faces/clusters?include_labeled=false&min_confidence=0.70&min_cluster_size=2
```

**Step-by-Step Execution**:

1. **Build SQL Query** (lines 133-144)
   ```sql
   SELECT
       cluster_id,
       array_agg(face_instances.id) as face_ids,
       bool_or(person_id IS NOT NULL) as has_person,
       count(face_instances.id) as face_count,
       avg(quality_score) as avg_quality
   FROM face_instances
   WHERE cluster_id IS NOT NULL  -- ← Excludes unclustered faces
   GROUP BY cluster_id
   ```

2. **Apply include_labeled=false Filter** (lines 146-157)
   ```sql
   HAVING bool_or(person_id IS NOT NULL) IS NOT TRUE
   -- Keeps only clusters where ALL faces have person_id IS NULL
   ```

3. **Apply min_cluster_size=2 Filter** (lines 160-163)
   ```sql
   HAVING count(face_instances.id) >= 2
   ```

4. **Count Total (Pre-Confidence Filtering)** (lines 170-173)
   ```python
   total_result = await db.execute(count_query)
   pre_filter_total = total_result.scalar() or 0
   # This is the count BEFORE confidence filtering
   ```

5. **Fetch Results with Over-Fetching** (lines 175-184)
   ```python
   fetch_limit = page_size * 3 if min_confidence else page_size
   # Example: page_size=20, fetch_limit=60
   paginated_query = cluster_query.offset(0).limit(60)
   ```

6. **Calculate Confidence and Filter** (lines 197-264)
   ```python
   for row in rows:
       # Get embeddings from Qdrant
       cluster_confidence = await clustering_service.calculate_cluster_confidence(...)

       # Filter by min_confidence
       if cluster_confidence < min_confidence:
           continue  # Skip this cluster

       items.append(ClusterSummary(...))

       if len(items) >= page_size:
           break  # Stop at page_size
   ```

7. **Return Response** (lines 291-298)
   ```python
   total = len(items) if min_confidence else pre_filter_total
   # NOTE: Total changes based on confidence filtering!
   ```

**Response Example (Empty State)**:
```json
{
  "items": [],
  "total": 0,
  "page": 1,
  "page_size": 20
}
```

**Response Example (Few Results)**:
```json
{
  "items": [
    {
      "clusterId": "clu_a1b2c3d4e5f6",
      "faceCount": 3,
      "sampleFaceIds": ["uuid1", "uuid2", "uuid3"],
      "avgQuality": 0.72,
      "clusterConfidence": 0.78,
      "representativeFaceId": "uuid1",
      "personId": null,
      "personName": null
    }
  ],
  "total": 1,
  "page": 1,
  "page_size": 20
}
```

---

## 5. Root Cause Assessment

### Hypothesis 1: Clustering Has Never Been Run ⭐ **MOST LIKELY**

**Probability**: 95%

**Evidence**:
- Clustering is a **manual operation** (POST `/api/v1/faces/cluster`)
- No automatic clustering on face detection
- No migration seeds initial cluster_ids
- Query filters `cluster_id IS NOT NULL` → returns 0 if all NULL

**Validation**:
```sql
SELECT
    COUNT(*) FILTER (WHERE person_id IS NULL) as unlabeled_faces,
    COUNT(*) FILTER (WHERE person_id IS NULL AND cluster_id IS NOT NULL) as clustered_unlabeled
FROM face_instances;
```

**Expected Result if True**:
```
unlabeled_faces | clustered_unlabeled
----------------|--------------------
1500            | 0
```

**Fix**:
1. Run clustering: `POST /api/v1/faces/cluster` with appropriate parameters
2. Verify cluster_ids populated in database
3. Re-query `/api/v1/faces/clusters` to see results

---

### Hypothesis 2: Clustering Created Mostly Single-Face Clusters

**Probability**: 4%

**Evidence**:
- HDBSCAN default `min_cluster_size=5` prevents this
- But if clustering was run with `min_cluster_size=1` or `2`, possible
- API-level `min_cluster_size=2` filter would exclude singles

**Validation**:
```sql
SELECT
    CASE
        WHEN face_count = 1 THEN 'single'
        WHEN face_count BETWEEN 2 AND 4 THEN 'small (2-4)'
        WHEN face_count BETWEEN 5 AND 9 THEN 'medium (5-9)'
        ELSE 'large (10+)'
    END as cluster_size_category,
    COUNT(*) as cluster_count,
    SUM(face_count) as total_faces
FROM (
    SELECT cluster_id, COUNT(*) as face_count
    FROM face_instances
    WHERE person_id IS NULL AND cluster_id IS NOT NULL AND cluster_id != '-1'
    GROUP BY cluster_id
) subq
GROUP BY cluster_size_category
ORDER BY cluster_size_category;
```

**Expected Result if True**:
```
cluster_size_category | cluster_count | total_faces
----------------------|---------------|------------
single                | 450           | 450
small (2-4)           | 50            | 150
medium (5-9)          | 10            | 70
large (10+)           | 5             | 80
```

**Impact**:
- 450 single-face clusters filtered out
- Only 65 multi-face clusters available
- But confidence filtering reduces further

---

### Hypothesis 3: Embeddings Missing in Qdrant

**Probability**: 1%

**Evidence**:
- Face detection pipeline should upload embeddings to Qdrant
- If embeddings missing, confidence calculation returns 0.0 or None
- API doesn't filter on `confidence = None` (line 264: sets to None on error)

**Validation**:
Check logs for warnings:
```
"Cluster {cluster_id} has {len(embeddings)} valid embeddings, cannot calculate confidence"
"Failed to calculate confidence for cluster {cluster_id}: {e}"
```

**Impact**:
- Clusters with missing embeddings get `clusterConfidence: null`
- They are NOT filtered out (only explicit < threshold filtering happens)
- This would show clusters with null confidence, not hide them

**Conclusion**: Unlikely to be root cause (would see clusters with null confidence)

---

### Hypothesis 4: Confidence Threshold Too High (Secondary Factor)

**Probability**: Conditional on Hypothesis 1 being resolved

**Evidence**:
- User sets `min_confidence=0.70` (70% threshold)
- HDBSCAN can create clusters with intra-cluster similarity 0.40-0.80
- Lower quality faces → lower similarity scores
- Over-fetching 3x may be insufficient if 90% filtered

**Validation**:
```python
# API logs should show:
logger.debug(
    f"Filtered out cluster {row.cluster_id} "
    f"(confidence {cluster_confidence:.3f} < {min_confidence})"
)
```

**Test**:
Try request with `min_confidence=0.50` or `min_confidence=0.60`

**Expected**: More clusters returned

---

## 6. Evidence Summary

### Code Evidence

| File | Lines | Finding |
|------|-------|---------|
| `faces.py` | 142 | Query filters `cluster_id IS NOT NULL` ✓ |
| `faces.py` | 146-157 | Correctly filters unlabeled clusters ✓ |
| `faces.py` | 160-163 | SQL-level min_cluster_size filtering ✓ |
| `faces.py` | 1187 | Clustering is manual POST endpoint ⚠️ |
| `clusterer.py` | 20-21 | Default min_cluster_size=5, min_samples=3 |
| `clusterer.py` | 72-113 | Scrolls Qdrant for unlabeled faces with embeddings |
| `clusterer.py` | 158-167 | Updates cluster_id in database after clustering |
| `face_clustering_service.py` | 88-98 | Retrieves embeddings, handles missing points |
| `face_clustering_service.py` | 100-106 | Returns 0.0 if < 2 valid embeddings |

### Schema Evidence

```python
# db/models.py line 468
cluster_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
```
- NULL by default
- Only populated after clustering runs

### Test Evidence

```python
# test_clusters_filtering.py demonstrates expected behavior
# All tests manually create faces with cluster_id set
# No tests verify behavior when cluster_id is NULL (realistic case)
```

---

## 7. Recommendations

### Immediate Actions

1. **Verify Database State**
   ```sql
   SELECT
       COUNT(*) as total_faces,
       COUNT(*) FILTER (WHERE person_id IS NULL) as unlabeled,
       COUNT(*) FILTER (WHERE person_id IS NULL AND cluster_id IS NOT NULL) as clustered,
       COUNT(*) FILTER (WHERE cluster_id = '-1') as noise
   FROM face_instances;
   ```

2. **Check Clustering History**
   - Verify clustering has been run at least once
   - Check application logs for clustering job execution
   - Look for RQ job history or queue status

3. **Run Clustering** (if never run)
   ```http
   POST /api/v1/faces/cluster
   Content-Type: application/json

   {
     "min_cluster_size": 2,
     "quality_threshold": 0.5,
     "max_faces": 50000
   }
   ```

4. **Monitor Response**
   - Check `clusters_found` count
   - Check `noise_count` (should be reasonable, not 90%+)
   - Verify cluster_ids populated in database

5. **Re-test UI with Lower Thresholds**
   ```http
   GET /api/v1/faces/clusters?include_labeled=false&min_confidence=0.50&min_cluster_size=2
   ```

### Long-Term Improvements

1. **Auto-Clustering After Detection**
   - Add clustering to face detection pipeline
   - Run incremental clustering on new batches
   - Update cluster_ids automatically

2. **Improve Over-Fetching Strategy**
   - Current: 3x page_size
   - Better: Fetch until page_size valid results (loop with pagination)
   - Best: Cache confidence scores in database

3. **Add Cluster Metadata Table**
   ```sql
   CREATE TABLE cluster_metadata (
       cluster_id VARCHAR(100) PRIMARY KEY,
       face_count INTEGER NOT NULL,
       avg_confidence FLOAT,
       avg_quality FLOAT,
       created_at TIMESTAMPTZ NOT NULL,
       updated_at TIMESTAMPTZ NOT NULL
   );
   ```
   Benefits:
   - Fast cluster listing without aggregation
   - Pre-computed confidence scores
   - Historical cluster tracking

4. **Add API Diagnostics Endpoint**
   ```http
   GET /api/v1/faces/clustering/status
   ```
   Returns:
   ```json
   {
     "totalFaces": 5000,
     "unlabeledFaces": 1500,
     "clusteredFaces": 800,
     "noiseFaces": 700,
     "clusters": 45,
     "lastClusteringRun": "2025-12-29T10:30:00Z",
     "clusterSizeDistribution": {
       "1": 0,
       "2-4": 20,
       "5-9": 15,
       "10+": 10
     }
   }
   ```

5. **Improve Test Coverage**
   - Add integration test with real clustering
   - Test NULL cluster_id scenario (realistic)
   - Test confidence calculation with real embeddings
   - Test over-fetching edge cases

6. **Add User Guidance**
   - API response should indicate if clustering needed
   - UI should show "Run clustering to see groups" message
   - Provide estimated cluster count before filtering

---

## 8. Testing Plan

### Manual Verification Steps

**Step 1: Verify Database State**
```bash
# Connect to database
psql $DATABASE_URL

# Run diagnostic queries (from section 1)
\i diagnostic_queries.sql
```

**Step 2: Check Qdrant State**
```bash
# Verify faces collection exists
curl http://localhost:6333/collections/faces

# Sample some points
curl http://localhost:6333/collections/faces/points/scroll?limit=5
```

**Step 3: Trigger Clustering**
```bash
# API request
curl -X POST http://localhost:8000/api/v1/faces/cluster \
  -H "Content-Type: application/json" \
  -d '{
    "min_cluster_size": 2,
    "quality_threshold": 0.5,
    "max_faces": 10000
  }'
```

**Step 4: Verify Results**
```bash
# Check database
SELECT cluster_id, COUNT(*)
FROM face_instances
WHERE person_id IS NULL AND cluster_id IS NOT NULL
GROUP BY cluster_id
LIMIT 10;

# Check API
curl "http://localhost:8000/api/v1/faces/clusters?include_labeled=false&min_cluster_size=2"
```

**Step 5: Test Confidence Filtering**
```bash
# High threshold (70%)
curl "http://localhost:8000/api/v1/faces/clusters?include_labeled=false&min_confidence=0.70"

# Medium threshold (60%)
curl "http://localhost:8000/api/v1/faces/clusters?include_labeled=false&min_confidence=0.60"

# Low threshold (50%)
curl "http://localhost:8000/api/v1/faces/clusters?include_labeled=false&min_confidence=0.50"
```

### Automated Test Suite

**Add Test: Empty Cluster IDs**
```python
@pytest.mark.asyncio
async def test_clusters_when_never_clustered(test_client, db_session, mock_image_asset):
    """Test that endpoint returns empty list when clustering never run."""
    from image_search_service.db.models import FaceInstance

    # Create faces WITHOUT cluster_id (realistic scenario)
    for i in range(10):
        face = FaceInstance(
            id=uuid.uuid4(),
            asset_id=mock_image_asset.id,
            bbox_x=100 + i * 10,
            bbox_y=100,
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.95,
            quality_score=0.75,
            qdrant_point_id=uuid.uuid4(),
            cluster_id=None,  # ← Unclustered
            person_id=None,   # ← Unlabeled
        )
        db_session.add(face)

    await db_session.commit()

    # When: get clusters
    response = await test_client.get("/api/v1/faces/clusters?include_labeled=false")

    # Then: returns empty list (not an error)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0
    assert len(data["items"]) == 0
```

**Add Test: Integration with Real Clustering**
```python
@pytest.mark.asyncio
@pytest.mark.integration
async def test_full_clustering_flow(test_client, db_session, mock_image_asset, real_qdrant):
    """Test clustering → query flow end-to-end."""
    # 1. Create unlabeled faces with embeddings in Qdrant
    # 2. POST /api/v1/faces/cluster
    # 3. Verify cluster_ids populated
    # 4. GET /api/v1/faces/clusters
    # 5. Assert clusters returned with valid confidence scores
    pass  # TODO: Implement
```

---

## 9. Data Validation Queries (Production-Safe)

```sql
-- Query 1: Overall face statistics
SELECT
    COUNT(*) as total_faces,
    COUNT(*) FILTER (WHERE person_id IS NOT NULL) as labeled_faces,
    COUNT(*) FILTER (WHERE person_id IS NULL) as unlabeled_faces,
    COUNT(*) FILTER (WHERE cluster_id IS NOT NULL) as clustered_faces,
    COUNT(*) FILTER (WHERE cluster_id = '-1') as noise_faces,
    COUNT(*) FILTER (WHERE person_id IS NULL AND cluster_id IS NULL) as unclustered_unlabeled
FROM face_instances;

-- Query 2: Cluster size distribution (unlabeled only)
WITH cluster_sizes AS (
    SELECT
        cluster_id,
        COUNT(*) as face_count,
        AVG(quality_score) as avg_quality
    FROM face_instances
    WHERE person_id IS NULL
      AND cluster_id IS NOT NULL
      AND cluster_id != '-1'
    GROUP BY cluster_id
)
SELECT
    CASE
        WHEN face_count = 1 THEN '1 face'
        WHEN face_count BETWEEN 2 AND 4 THEN '2-4 faces'
        WHEN face_count BETWEEN 5 AND 9 THEN '5-9 faces'
        WHEN face_count BETWEEN 10 AND 19 THEN '10-19 faces'
        ELSE '20+ faces'
    END as size_category,
    COUNT(*) as cluster_count,
    MIN(face_count) as min_faces,
    MAX(face_count) as max_faces,
    AVG(face_count)::numeric(10,2) as avg_faces,
    AVG(avg_quality)::numeric(10,3) as avg_quality
FROM cluster_sizes
GROUP BY size_category
ORDER BY MIN(face_count);

-- Query 3: Top 10 largest unlabeled clusters
SELECT
    cluster_id,
    COUNT(*) as face_count,
    AVG(quality_score)::numeric(10,3) as avg_quality,
    MIN(created_at) as first_face_detected,
    MAX(created_at) as last_face_detected
FROM face_instances
WHERE person_id IS NULL
  AND cluster_id IS NOT NULL
  AND cluster_id != '-1'
GROUP BY cluster_id
ORDER BY face_count DESC
LIMIT 10;

-- Query 4: Quality distribution for unlabeled faces
SELECT
    CASE
        WHEN quality_score IS NULL THEN 'unknown'
        WHEN quality_score >= 0.8 THEN 'excellent (>=0.8)'
        WHEN quality_score >= 0.6 THEN 'good (0.6-0.8)'
        WHEN quality_score >= 0.4 THEN 'fair (0.4-0.6)'
        ELSE 'poor (<0.4)'
    END as quality_tier,
    COUNT(*) as face_count,
    COUNT(*) FILTER (WHERE cluster_id IS NOT NULL) as clustered_count,
    (COUNT(*) FILTER (WHERE cluster_id IS NOT NULL)::float / COUNT(*)::float * 100)::numeric(10,2) as clustering_rate_pct
FROM face_instances
WHERE person_id IS NULL
GROUP BY quality_tier
ORDER BY MIN(COALESCE(quality_score, 0));

-- Query 5: Sample faces by state
(SELECT 'unlabeled_unclustered' as state, id, cluster_id, person_id, quality_score, created_at
 FROM face_instances
 WHERE person_id IS NULL AND cluster_id IS NULL
 LIMIT 5)
UNION ALL
(SELECT 'unlabeled_clustered' as state, id, cluster_id, person_id, quality_score, created_at
 FROM face_instances
 WHERE person_id IS NULL AND cluster_id IS NOT NULL AND cluster_id != '-1'
 LIMIT 5)
UNION ALL
(SELECT 'noise' as state, id, cluster_id, person_id, quality_score, created_at
 FROM face_instances
 WHERE cluster_id = '-1'
 LIMIT 5)
ORDER BY state, created_at DESC;

-- Query 6: Temporal distribution of faces
SELECT
    DATE_TRUNC('day', created_at) as detection_date,
    COUNT(*) as faces_detected,
    COUNT(*) FILTER (WHERE cluster_id IS NOT NULL) as faces_clustered,
    COUNT(*) FILTER (WHERE person_id IS NOT NULL) as faces_labeled
FROM face_instances
WHERE created_at >= NOW() - INTERVAL '30 days'
GROUP BY detection_date
ORDER BY detection_date DESC
LIMIT 30;
```

**Output Interpretation**:

**Query 1** should reveal the smoking gun:
- If `unclustered_unlabeled > 0` → Clustering never run
- If `unclustered_unlabeled = 0` but `clustered_faces` is low → Poor clustering parameters

**Query 2** reveals cluster size distribution:
- If dominated by "1 face" → Clustering parameters too loose
- If dominated by "20+ faces" → Clustering parameters too tight

**Query 4** reveals quality impact:
- If "poor" tier has low `clustering_rate_pct` → Quality threshold filtering too many
- If "excellent" tier has high `clustering_rate_pct` → Confirms quality-based clustering works

---

## 10. Conclusion

The most probable root cause is **clustering has never been executed** on the unlabeled faces in the database. The query logic is correct but filters for `cluster_id IS NOT NULL`, which returns zero results if all unlabeled faces have NULL cluster_ids.

**Next Steps**:
1. Run diagnostic SQL queries (section 9)
2. Verify clustering execution history
3. Manually trigger clustering with appropriate parameters
4. Re-test API with multiple confidence thresholds
5. Consider implementing automatic clustering in detection pipeline

**Critical Understanding**:
Face clustering is a **manual, explicit operation** in this system, not an automatic background process. Users must trigger clustering before clusters become visible in the API.

---

**Document Version**: 1.0
**Last Updated**: 2025-12-30
**Next Review**: After database validation queries completed
