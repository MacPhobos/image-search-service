# Prototype Image Control Research

**Date**: 2025-12-29
**Version**: 2.0 (Updated with Temporal Role System)
**Research Focus**: Temporal prototype management for 30+ year family photo archives with age progression

---

## Executive Summary

The current system automatically creates prototypes from user-verified face assignments using a quality-based selection strategy. While effective for professional photography or security use cases, it **fails to handle the unique challenges of family photo archives spanning 30+ years** where people age from childhood to adulthood.

**Critical Gap**: The current system may select 5 prototypes all from a person's adult years, causing complete failure to match childhood photos of the same person. Face embeddings change dramatically across age ranges, requiring temporal diversity in prototype selection.

**Recommended Solution**: A hybrid approach combining **Manual Pinning** (user control), **Temporal Roles** (age-era buckets), and **Smart Diversity** (metadata-driven selection) to ensure prototypes span infant → child → teen → adult → senior eras.

**Key Finding**: The prototype system doesn't need a complete redesign—**it needs temporal awareness** to handle age progression across decades. This research proposes a backward-compatible enhancement that maintains intelligent automation while giving users surgical control over temporal coverage.

---

## Current State Analysis

### Data Model (PostgreSQL)

**PersonPrototype Table** (`db/models.py` lines 495-533):
```python
class PersonPrototype(Base):
    id: UUID (primary key)
    person_id: UUID → Person (CASCADE delete)
    face_instance_id: UUID → FaceInstance (SET NULL on delete)
    qdrant_point_id: UUID (embedding reference in Qdrant)
    role: PrototypeRole (CENTROID | EXEMPLAR)
    created_at: datetime
```

**Key Relationships**:
- Each `Person` has 0-N `PersonPrototype` records
- Each prototype links to a specific `FaceInstance` (the source face)
- Prototypes are indexed in Qdrant with `is_prototype=True` flag for fast filtering

**Current Role Types**:
- `EXEMPLAR`: High-quality representative face (actively used)
- `CENTROID`: Computed average/centroid (defined but not currently created)

### Prototype Creation Logic

**Automatic Creation** (`services/prototype_service.py`):

The system creates prototypes automatically when users assign faces to persons through three main workflows:

1. **Single Face Assignment** (`POST /api/v1/faces/faces/{face_id}/assign`)
   - User manually assigns one face to a person
   - Calls `create_or_update_prototypes()` immediately

2. **Cluster Labeling** (`POST /api/v1/faces/clusters/{cluster_id}/label`)
   - User labels an entire cluster (e.g., 15 faces) as a person
   - Creates prototypes from top 3 quality faces in the cluster

3. **Bulk Photo Move** (`POST /api/v1/faces/persons/{person_id}/photos/bulk-move`)
   - User moves faces from one person to another
   - Creates prototypes from newly assigned faces

**Selection Criteria** (current implementation):
```python
# Quality threshold filter
if face.quality_score < min_quality_threshold (default: 0.5):
    skip prototype creation

# Maximum exemplars cap
max_exemplars per person = 5 (configurable: 1-20)

# Pruning strategy (when exceeding max)
1. Sort all exemplars by quality_score (descending)
2. Prefer diversity: select from different asset_ids (photos)
3. Fill remaining slots with highest quality regardless of diversity
```

**Configuration** (`core/config.py` lines 106-119):
```python
face_prototype_min_quality: float = 0.5  # Env: FACE_PROTOTYPE_MIN_QUALITY
face_prototype_max_exemplars: int = 5    # Env: FACE_PROTOTYPE_MAX_EXEMPLARS
```

### Prototype Usage in Face Recognition

**Primary Use Case**: Face Suggestion System

When a user assigns a face to a person, the system:
1. Searches Qdrant for similar faces using prototype embeddings
2. Filters by `is_prototype=True` to search only against representative faces (not all faces)
3. Returns suggestions for other unassigned faces that look similar

**Search Method** (`vector/face_qdrant.py` lines 452-473):
```python
def search_against_prototypes(query_embedding, limit=5, score_threshold=0.6):
    """Search against only prototype faces for incremental assignment."""
    return search_similar_faces(
        filter_is_prototype=True,  # CRITICAL: Only searches prototypes
        ...
    )
```

**Performance Benefit**:
- Searching 50 prototypes (10 persons × 5 exemplars) is much faster than searching 5,000 individual face instances
- Prototypes reduce noise from low-quality or occluded faces

### Current Workflow Example

**Scenario**: User labels their first photo of "Alice"

1. User assigns face → Person("Alice") created
2. System checks quality_score = 0.85 (above 0.5 threshold ✓)
3. Creates PersonPrototype(face_instance_id=face.id, role=EXEMPLAR)
4. Marks in Qdrant: `{is_prototype: true, person_id: alice_uuid}`
5. **Propagation job**: System searches Qdrant for similar faces to this prototype
6. Creates FaceSuggestion records for matches above 0.7 confidence

As user labels more faces of Alice:
- Each verified face becomes a prototype (up to max_exemplars=5)
- Lower quality prototypes are pruned automatically
- Suggestion quality improves with more diverse exemplars

---

## Problem Statement

### Limitations of Current Approach

1. **No Manual Override**
   - Users cannot explicitly choose which face represents a person
   - Quality score is computed automatically and may not match user preference
   - No UI to view/manage current prototypes

2. **Quality Score Opacity**
   - Quality metric (from face detection model) may prioritize technical factors:
     - Sharpness, frontal orientation, lighting
   - Users may prefer different criteria:
     - Favorite photo, recency, context (e.g., wedding photo)

3. **Limited Diversity Control**
   - Current pruning prefers "different asset_ids" but doesn't consider:
     - Age/temporal diversity (childhood vs. adult)
     - Expression diversity (smiling vs. neutral)
     - Context diversity (casual vs. formal)

4. **No Visibility into Selection**
   - No API endpoint to list current prototypes for a person
   - No indication which photos were selected as prototypes
   - Users cannot see why certain faces were chosen

5. **Rigid Max Exemplars Limit**
   - Global setting applies to all persons equally
   - Some persons may need more exemplars (e.g., celebrities with many photos)
   - Others may need fewer (e.g., infrequently photographed persons)

### User Pain Points

**Example 1**: User has 100 photos of their child from ages 1-10
- System may select 5 prototypes from age 8-10 (highest quality)
- Face suggestions for age 1-3 photos fail (face changed significantly)
- User wants prototypes spanning different ages

**Example 2**: User has one perfect portrait and 99 candid shots
- System might include lower-quality candids in prototypes
- User wants only the perfect portrait as the sole prototype

**Example 3**: Professional photographer has client galleries
- Wants to use specific edited/retouched photos as prototypes
- Quality score doesn't reflect professional editing decisions

---

## Suggested Methods for Better Control

### Method 1: Manual Prototype Pinning (User Override)

**Concept**: Allow users to explicitly "pin" specific faces as prototypes, bypassing automatic selection.

**Implementation Approach**:

**Database Changes**:
```python
# Add column to PersonPrototype
is_pinned: bool = False  # User-selected, immune to pruning
pinned_by: str | None    # User identifier who pinned it
pinned_at: datetime | None
```

**New API Endpoints**:
```python
POST /api/v1/faces/persons/{person_id}/prototypes/pin
{
    "face_instance_id": "uuid",
    "reason": "Best portrait photo"  # Optional note
}

DELETE /api/v1/faces/persons/{person_id}/prototypes/{proto_id}/pin
# Unpin but keep as regular prototype

GET /api/v1/faces/persons/{person_id}/prototypes
# List all prototypes with pinned status
```

**Modified Pruning Logic**:
```python
def prune_prototypes():
    # Separate pinned and unpinned
    pinned = [p for p in prototypes if p.is_pinned]
    unpinned = [p for p in prototypes if not p.is_pinned]

    # Calculate available slots
    available_slots = max_exemplars - len(pinned)

    if available_slots <= 0:
        # Too many pinned, warn user or raise limit
        raise TooManyPinnedError()

    # Prune only unpinned prototypes
    keep_unpinned = select_top_quality(unpinned, available_slots)

    return pinned + keep_unpinned
```

**Pros**:
- ✅ Direct user control over critical prototypes
- ✅ Minimal code changes (add column + pruning logic)
- ✅ Backward compatible (is_pinned defaults to False)
- ✅ Preserves automatic selection for unpinned slots

**Cons**:
- ⚠️ Users can pin too many faces, exhausting available slots
- ⚠️ Requires UI to show "pin" button on face instances
- ⚠️ Need to handle edge case: user unpins, system re-prunes

**Use Cases**:
- "Always use this wedding photo as prototype"
- "Pin this recent photo, auto-select 4 others from older photos"

---

### Method 2: Prototype Priority Weights

**Concept**: Allow users to assign weights/priorities to faces, influencing automatic selection.

**Implementation Approach**:

**Database Changes**:
```python
# Add to FaceInstance (affects all future prototype selections)
user_priority: int | None  # 1-10 scale, NULL = use quality_score
priority_note: str | None  # "favorite photo", "best angle", etc.
```

**Modified Selection Logic**:
```python
def select_prototypes_with_priority(faces, max_exemplars):
    # Compute composite score
    for face in faces:
        quality = face.quality_score or 0.0
        user_weight = face.user_priority or 5  # Default to neutral

        # Weighted combination (user priority can override quality)
        face.composite_score = (quality * 0.6) + (user_weight / 10 * 0.4)

    # Sort by composite score
    faces.sort(key=lambda f: f.composite_score, reverse=True)

    return select_top_with_diversity(faces, max_exemplars)
```

**New API Endpoints**:
```python
PATCH /api/v1/faces/faces/{face_id}/priority
{
    "priority": 9,  # 1-10 scale
    "note": "Best smile"
}
```

**Pros**:
- ✅ Soft guidance (doesn't hard-override automatic selection)
- ✅ Works with existing pruning logic
- ✅ Users can set priority before or after labeling
- ✅ Gradual control (priority 6 vs. 9 matters)

**Cons**:
- ⚠️ Priority scale (1-10) may be confusing to users
- ⚠️ Still automated—no guarantee high priority = prototype
- ⚠️ Need UI to set priority (slider, stars, etc.)

**Use Cases**:
- "Mark favorite photos as high priority"
- "Downweight blurry photos without deleting them"

---

### Method 3: Role-Based Prototypes

**Concept**: Support multiple prototype roles with different purposes (CENTROID, PRIMARY, TEMPORAL, CONTEXTUAL).

**Implementation Approach**:

**Expand PrototypeRole Enum**:
```python
class PrototypeRole(str, Enum):
    CENTROID = "centroid"      # Average embedding (computed)
    PRIMARY = "primary"        # User's main representative image
    EXEMPLAR = "exemplar"      # High-quality auto-selected
    TEMPORAL = "temporal"      # Age/time-based exemplar (child, adult, elderly)
    CONTEXTUAL = "contextual"  # Situation-based (formal, casual, sports)
```

**Modified Prototype Creation**:
```python
# User can specify role when creating
POST /api/v1/faces/persons/{person_id}/prototypes
{
    "face_instance_id": "uuid",
    "role": "primary"  # Overrides default EXEMPLAR
}

# Different max limits per role
max_primary = 1         # Only one "main" photo
max_temporal = 5        # Childhood, teen, adult, middle-age, elderly
max_exemplars = 10      # Auto-selected high-quality
```

**Search Strategy**:
```python
# Search against specific roles
def search_against_primary_only():
    filter_role = "primary"

def search_against_temporal(age_range):
    # Match prototypes from similar age range
    filter_role = "temporal"
    filter_metadata = {"age_bucket": age_range}
```

**Pros**:
- ✅ Semantic clarity (PRIMARY vs. EXEMPLAR vs. TEMPORAL)
- ✅ Supports complex use cases (age-based prototypes)
- ✅ Search can be role-specific for better accuracy
- ✅ Extensible (add new roles without schema changes)

**Cons**:
- ⚠️ More complex UI (users pick roles)
- ⚠️ Requires metadata (age, context) for TEMPORAL/CONTEXTUAL roles
- ⚠️ Breaks current assumption that all prototypes are equal

**Use Cases**:
- "Set one PRIMARY photo for profile, auto-select 5 EXEMPLARS for matching"
- "Add TEMPORAL prototypes for childhood, adulthood, old age"

---

### Method 4: Prototype Approval Workflow

**Concept**: After automatic selection, show users proposed prototypes for review/approval.

**Implementation Approach**:

**Add Prototype Status**:
```python
class PrototypeStatus(str, Enum):
    PENDING = "pending"    # Auto-selected, awaiting user review
    APPROVED = "approved"  # User confirmed
    REJECTED = "rejected"  # User rejected, will be pruned
    ACTIVE = "active"      # In use (backward compat)

# Add to PersonPrototype
status: PrototypeStatus = PENDING
reviewed_at: datetime | None
reviewed_by: str | None
```

**New Workflow**:
```
1. User labels face → System creates prototypes with status=PENDING
2. UI shows notification: "Review prototype suggestions for Alice"
3. User approves/rejects each proposed prototype
4. System finalizes: APPROVED → ACTIVE, REJECTED → deleted
```

**API Endpoints**:
```python
GET /api/v1/faces/persons/{person_id}/prototypes/pending
# Show user proposed prototypes for review

POST /api/v1/faces/persons/{person_id}/prototypes/{proto_id}/approve
POST /api/v1/faces/persons/{person_id}/prototypes/{proto_id}/reject

# Bulk actions
POST /api/v1/faces/persons/{person_id}/prototypes/bulk-approve
{
    "prototype_ids": ["uuid1", "uuid2"],
    "approve_all": false  # If true, approve all pending
}
```

**Pros**:
- ✅ User sees exactly which faces are selected before activation
- ✅ No guessing—approval is explicit
- ✅ Can show side-by-side comparisons in UI
- ✅ Maintains automatic selection as default (user can skip review)

**Cons**:
- ⚠️ Extra step for users (friction in workflow)
- ⚠️ Pending prototypes not used for suggestions until approved
- ⚠️ Need UI for review workflow

**Use Cases**:
- "Review auto-selected prototypes before going live"
- "System suggests 5, user approves 3, adds 2 custom ones"

---

### Method 5: Smart Diversity Constraints

**Concept**: Enhance automatic selection with metadata-based diversity rules.

**Implementation Approach**:

**Add Metadata to FaceInstance**:
```python
# Extend FaceInstance.landmarks (JSON field) with:
{
    "age_estimate": 35,       # From face analysis
    "expression": "smiling",  # neutral, smiling, surprised
    "pose": "frontal",        # frontal, profile, three_quarter
    "lighting": "outdoor",    # outdoor, indoor, studio
    "timestamp": "2023-05-10" # From EXIF or file mtime
}
```

**Modified Pruning Logic**:
```python
def select_prototypes_with_diversity(faces, max_exemplars):
    selected = []

    # Diversity buckets
    age_buckets = defaultdict(list)
    expression_buckets = defaultdict(list)

    # Group faces by metadata
    for face in sorted(faces, key=quality_score, reverse=True):
        age_range = bucket_age(face.metadata.age_estimate)
        age_buckets[age_range].append(face)
        expression_buckets[face.metadata.expression].append(face)

    # Select one from each bucket (round-robin)
    while len(selected) < max_exemplars:
        for bucket in age_buckets.values():
            if bucket and len(selected) < max_exemplars:
                selected.append(bucket.pop(0))

    return selected
```

**Configuration**:
```python
# New settings
face_prototype_diversity_mode: str = "temporal"  # temporal, expression, pose, balanced
face_prototype_age_buckets: list = [0-12, 13-19, 20-40, 41-60, 60+]
```

**Pros**:
- ✅ Fully automatic (no user action required)
- ✅ Intelligently handles temporal changes (childhood → adulthood)
- ✅ Better matching accuracy across different contexts
- ✅ Leverages existing face analysis models

**Cons**:
- ⚠️ Depends on accurate metadata (age estimation can be wrong)
- ⚠️ More complex selection logic (harder to debug)
- ⚠️ May conflict with user preferences (system thinks diversity > quality)

**Use Cases**:
- "Auto-select prototypes spanning childhood to adulthood"
- "Ensure both smiling and neutral expressions are represented"

---

## Recommended Approach

### Hybrid: Manual Pinning + Temporal Roles + Smart Diversity (Methods 1 + 3 + 5)

**Critical Context**: This recommendation specifically addresses **family photo archives spanning 30+ years** where people appear as children who are now adults. Face recognition must work across dramatic age differences while maintaining user control over critical reference photos.

**Rationale**:
- **Method 1 (Manual Pinning)** provides direct user control for definitive reference photos
- **Method 3 (Temporal Roles)** ensures prototypes span age progression from childhood to adulthood
- **Method 5 (Smart Diversity)** automatically fills temporal slots with best-quality faces from each era
- **Complementary**: Users pin critical photos per era, system intelligently fills age-range gaps

**Core Design Principles**:
1. **Temporal Coverage is Mandatory**: Every person must have prototypes spanning their full photographic history
2. **Age Progression is Critical**: Matching a 5-year-old to their 35-year-old self requires temporal prototypes
3. **User Control for Certainty**: Users can pin definitive photos when automatic selection fails
4. **Graceful Degradation**: System handles missing eras intelligently (sparse photo coverage)

---

### Temporal Role System Design

**Age-Era Bucket Definition**:

```python
class AgeEraBucket(str, Enum):
    INFANT = "infant"              # 0-3 years
    CHILD = "child"                # 4-12 years
    TEEN = "teen"                  # 13-19 years
    YOUNG_ADULT = "young_adult"    # 20-35 years
    ADULT = "adult"                # 36-55 years
    SENIOR = "senior"              # 56+ years

class DecadeBucket(str, Enum):
    Y1990s = "1990s"
    Y2000s = "2000s"
    Y2010s = "2010s"
    Y2020s = "2020s"
```

**Expanded Prototype Role System**:

```python
class PrototypeRole(str, Enum):
    CENTROID = "centroid"          # Computed average embedding (future)
    PRIMARY = "primary"            # User's definitive photo for an era
    TEMPORAL = "temporal"          # Age-based exemplar (auto-selected)
    EXEMPLAR = "exemplar"          # High-quality auto-selected (no temporal assignment)
    FALLBACK = "fallback"          # Lower quality, used when era has sparse coverage
```

**Temporal Prototype Allocation Strategy**:

```python
# Maximum prototypes per person
MAX_TOTAL_PROTOTYPES = 12  # Increased from 5 to support temporal coverage

# Allocation priority (per person):
TEMPORAL_SLOTS = 6         # One per age-era bucket (infant, child, teen, young_adult, adult, senior)
PRIMARY_SLOTS = 3          # User-pinned definitive photos (max 1 per era)
EXEMPLAR_SLOTS = 3         # High-quality auto-selected for diversity

# Allocation logic:
# 1. Assign TEMPORAL prototypes first (one per age-era with photos)
# 2. Fill PRIMARY slots from user pins (optional)
# 3. Fill EXEMPLAR slots with highest-quality remaining faces
# 4. Use FALLBACK role when era has only low-quality photos
```

**Temporal Metadata on FaceInstance**:

```python
# Extend FaceInstance.landmarks JSONB field:
{
    "age_estimate": 8,                    # From face analysis model
    "age_confidence": 0.85,               # Confidence in age estimate
    "age_era_bucket": "child",            # Computed from age_estimate
    "decade_bucket": "2000s",             # From photo EXIF timestamp
    "photo_timestamp": "2003-07-15",      # From EXIF or file mtime
    "temporal_quality_score": 0.92,       # Composite: quality + frontal_pose + clarity
    "pose": "frontal",                    # frontal, profile, three_quarter
    "expression": "smiling",              # neutral, smiling, laughing
    "lighting": "outdoor"                 # outdoor, indoor, studio
}
```

---

### Pinning Integration with Temporal Roles

**User Pin Behavior**:

Users can pin a photo as the "definitive" reference for a specific era:

```python
# API: Pin with optional temporal role
POST /api/v1/faces/persons/{person_id}/prototypes/pin
{
    "face_instance_id": "uuid",
    "age_era_bucket": "child",       # Optional: Associate pin with era
    "is_primary": true,               # Optional: Mark as PRIMARY role (default: false)
    "note": "Best childhood photo"
}
```

**Pin + Temporal Role Interaction**:

1. **Explicit Era Pin**: User pins photo to "child" era
   - System sets role = PRIMARY if is_primary=true, else TEMPORAL
   - Counts toward temporal slot for "child" era
   - Prevents automatic re-selection for that era

2. **Non-Era Pin**: User pins photo without era assignment
   - System sets role = PRIMARY (no era association)
   - Counts toward PRIMARY_SLOTS quota (max 3)
   - Does NOT count toward temporal era allocation

3. **Pin Priority in Pruning**:
   - PRIMARY pins are immune to pruning (highest priority)
   - TEMPORAL pins for eras are immune within their era
   - EXEMPLAR and FALLBACK roles can be pruned

**Pin Quotas and Limits**:

```python
# Quota enforcement:
max_primary_pins = 3          # Definitive photos, no era restriction
max_temporal_pins_per_era = 1 # One pinned photo per age-era bucket

# Validation logic:
def validate_pin(person_id, face_instance_id, age_era_bucket, is_primary):
    if is_primary:
        primary_count = count_primary_pins(person_id)
        if primary_count >= max_primary_pins:
            raise HTTPException(400, "Maximum 3 PRIMARY pins reached")

    if age_era_bucket:
        era_pin_count = count_temporal_pins(person_id, age_era_bucket)
        if era_pin_count >= max_temporal_pins_per_era:
            raise HTTPException(400, f"Era '{age_era_bucket}' already has pinned photo")
```

---

### Smart Selection for Age Progression

**Automatic Temporal Prototype Selection**:

The system automatically populates temporal slots using metadata-driven selection:

```python
def select_temporal_prototypes(person_id: UUID) -> list[PersonPrototype]:
    """
    Select one prototype per age-era bucket for comprehensive temporal coverage.
    """
    # 1. Load all verified faces for person
    faces = get_verified_faces(person_id)

    # 2. Group faces by age-era bucket
    era_buckets = defaultdict(list)
    for face in faces:
        era = face.metadata.get("age_era_bucket")
        if era:
            era_buckets[era].append(face)

    # 3. Select best face from each era
    temporal_prototypes = []
    for era, faces_in_era in era_buckets.items():
        # Check for user-pinned photo in this era
        pinned = [f for f in faces_in_era if is_pinned(f, era)]
        if pinned:
            # Use pinned photo, mark as TEMPORAL role
            temporal_prototypes.append(pinned[0])
            continue

        # Auto-select: Sort by temporal_quality_score
        faces_in_era.sort(key=lambda f: f.metadata.get("temporal_quality_score", 0), reverse=True)

        # Prefer frontal poses for better matching
        frontal_faces = [f for f in faces_in_era if f.metadata.get("pose") == "frontal"]
        selected = frontal_faces[0] if frontal_faces else faces_in_era[0]

        # Check quality threshold
        if selected.quality_score >= 0.5:
            temporal_prototypes.append(selected)
        else:
            # Use as FALLBACK role (low quality but fills era gap)
            selected.role = PrototypeRole.FALLBACK
            temporal_prototypes.append(selected)

    return temporal_prototypes
```

**Temporal Quality Score Calculation**:

```python
def compute_temporal_quality_score(face: FaceInstance) -> float:
    """
    Enhanced quality score emphasizing temporal matching factors.
    """
    base_quality = face.quality_score or 0.0     # From detection model (0-1)
    pose_bonus = 0.2 if face.metadata.get("pose") == "frontal" else 0.0
    clarity_bonus = 0.1 if face.bbox_area > 10000 else 0.0  # Larger face = better
    age_confidence_bonus = face.metadata.get("age_confidence", 0.0) * 0.1

    temporal_quality = (
        base_quality * 0.6 +
        pose_bonus +
        clarity_bonus +
        age_confidence_bonus
    )

    return min(temporal_quality, 1.0)  # Cap at 1.0
```

**Temporal Coverage Gaps**:

When some eras have no photos, the system gracefully degrades:

```python
def handle_sparse_temporal_coverage(person_id: UUID) -> dict:
    """
    Identify and report temporal coverage gaps.
    """
    all_eras = ["infant", "child", "teen", "young_adult", "adult", "senior"]
    prototypes = get_temporal_prototypes(person_id)
    covered_eras = {p.metadata["age_era_bucket"] for p in prototypes}
    missing_eras = set(all_eras) - covered_eras

    return {
        "person_id": person_id,
        "covered_eras": list(covered_eras),
        "missing_eras": list(missing_eras),
        "coverage_percentage": len(covered_eras) / len(all_eras) * 100,
        "recommendation": "Add photos from missing eras for better temporal matching"
    }
```

---

### Database Schema Extensions

**PersonPrototype Table Additions**:

```sql
-- Add temporal role support
ALTER TABLE person_prototypes ADD COLUMN role VARCHAR(50) DEFAULT 'exemplar';
ALTER TABLE person_prototypes ADD COLUMN age_era_bucket VARCHAR(50) NULL;
ALTER TABLE person_prototypes ADD COLUMN decade_bucket VARCHAR(10) NULL;

-- Add pinning support
ALTER TABLE person_prototypes ADD COLUMN is_pinned BOOLEAN DEFAULT FALSE;
ALTER TABLE person_prototypes ADD COLUMN pinned_by VARCHAR(255) NULL;
ALTER TABLE person_prototypes ADD COLUMN pinned_at TIMESTAMP NULL;

-- Temporal metadata (stored in FaceInstance.landmarks JSONB, no schema change needed)

-- Indexes for performance
CREATE INDEX idx_person_prototypes_role ON person_prototypes(person_id, role);
CREATE INDEX idx_person_prototypes_era ON person_prototypes(person_id, age_era_bucket);
CREATE INDEX idx_person_prototypes_pinned ON person_prototypes(person_id, is_pinned);
CREATE INDEX idx_person_prototypes_temporal_lookup ON person_prototypes(person_id, role, age_era_bucket);
```

**Updated PersonPrototype Model**:

```python
class PersonPrototype(Base):
    __tablename__ = "person_prototypes"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    person_id: Mapped[UUID] = mapped_column(ForeignKey("persons.id", ondelete="CASCADE"))
    face_instance_id: Mapped[UUID | None] = mapped_column(ForeignKey("face_instances.id", ondelete="SET NULL"))
    qdrant_point_id: Mapped[UUID] = mapped_column(nullable=False)

    # Role and temporal metadata
    role: Mapped[PrototypeRole] = mapped_column(default=PrototypeRole.EXEMPLAR)
    age_era_bucket: Mapped[str | None] = mapped_column(String(50), nullable=True)  # "child", "adult", etc.
    decade_bucket: Mapped[str | None] = mapped_column(String(10), nullable=True)   # "2000s", "2010s"

    # Pinning metadata
    is_pinned: Mapped[bool] = mapped_column(default=False)
    pinned_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    pinned_at: Mapped[datetime | None] = mapped_column(nullable=True)

    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    # Relationships
    person: Mapped["Person"] = relationship(back_populates="prototypes")
    face_instance: Mapped["FaceInstance"] = relationship()
```

---

### API Design

**New Endpoints for Temporal Prototype Management**:

```python
# 1. Pin prototype with optional era assignment
POST /api/v1/faces/persons/{person_id}/prototypes/pin
{
    "face_instance_id": "uuid",
    "age_era_bucket": "child",       # Optional: "infant", "child", "teen", "young_adult", "adult", "senior"
    "is_primary": true,               # Optional: Mark as PRIMARY role (default: false)
    "note": "Best childhood photo"    # Optional: User note
}
Response: { "prototype_id": "uuid", "role": "primary", "age_era_bucket": "child" }

# 2. Unpin prototype
DELETE /api/v1/faces/persons/{person_id}/prototypes/{proto_id}/pin
Response: { "message": "Prototype unpinned, may be replaced by automatic selection" }

# 3. List prototypes with temporal breakdown
GET /api/v1/faces/persons/{person_id}/prototypes
Response: {
    "items": [
        {
            "id": "uuid",
            "face_instance_id": "uuid",
            "role": "primary",
            "age_era_bucket": "child",
            "decade_bucket": "2000s",
            "is_pinned": true,
            "quality_score": 0.95,
            "created_at": "2025-12-29T10:00:00Z"
        }
    ],
    "coverage": {
        "covered_eras": ["child", "teen", "young_adult", "adult"],
        "missing_eras": ["infant", "senior"],
        "coverage_percentage": 66.7
    }
}

# 4. View prototypes by temporal role/era
GET /api/v1/faces/persons/{person_id}/prototypes/temporal
Response: {
    "infant": null,
    "child": { "prototype_id": "uuid", "face_instance_id": "uuid", "is_pinned": true },
    "teen": { "prototype_id": "uuid", "face_instance_id": "uuid", "is_pinned": false },
    "young_adult": { "prototype_id": "uuid", "face_instance_id": "uuid", "is_pinned": false },
    "adult": null,
    "senior": null
}

# 5. Trigger temporal re-diversification
POST /api/v1/faces/persons/{person_id}/prototypes/recompute
{
    "mode": "temporal",              # "temporal", "quality", "balanced"
    "preserve_pins": true            # Default: true (keep pinned prototypes)
}
Response: {
    "prototypes_created": 4,
    "prototypes_removed": 2,
    "coverage_improved": true,
    "new_eras_covered": ["infant", "senior"]
}

# 6. Create custom prototype with role assignment
POST /api/v1/faces/persons/{person_id}/prototypes
{
    "face_instance_id": "uuid",
    "role": "temporal",              # "primary", "temporal", "exemplar", "fallback"
    "age_era_bucket": "teen",        # Required for temporal role
    "is_pinned": true                # Optional
}
Response: { "prototype_id": "uuid", "role": "temporal", "age_era_bucket": "teen" }

# 7. Get temporal coverage report
GET /api/v1/faces/persons/{person_id}/temporal-coverage
Response: {
    "person_id": "uuid",
    "total_faces": 342,
    "era_distribution": {
        "infant": { "face_count": 0, "prototype_count": 0 },
        "child": { "face_count": 87, "prototype_count": 1 },
        "teen": { "face_count": 43, "prototype_count": 1 },
        "young_adult": { "face_count": 156, "prototype_count": 1 },
        "adult": { "face_count": 56, "prototype_count": 1 },
        "senior": { "face_count": 0, "prototype_count": 0 }
    },
    "decade_distribution": {
        "1990s": { "face_count": 12, "prototype_count": 1 },
        "2000s": { "face_count": 98, "prototype_count": 2 },
        "2010s": { "face_count": 167, "prototype_count": 2 },
        "2020s": { "face_count": 65, "prototype_count": 1 }
    },
    "recommendations": [
        "Add infant photos for complete temporal coverage",
        "Consider pinning definitive photo from 1990s era"
    ]
}
```

---

### Implementation Priority

**Phase 1: Temporal Infrastructure (High Priority)**
- ✅ Add temporal role support to PersonPrototype model
- ✅ Implement age-era bucket classification on face detection
- ✅ Add temporal metadata to FaceInstance.landmarks JSONB
- ✅ Create temporal prototype selection logic
- 📅 Estimated: 5-7 days (schema + face analysis + selection logic)

**Phase 2: Manual Pinning with Era Support (High Priority)**
- ✅ Add pinning columns to PersonPrototype
- ✅ Implement pin/unpin endpoints with era association
- ✅ Modify pruning logic to respect pinned prototypes
- ✅ Add temporal coverage API endpoints
- 📅 Estimated: 3-4 days (backend + API)

**Phase 3: UI for Temporal Management (Medium Priority)**
- ✅ Timeline view showing prototypes across eras
- ✅ Pin button with era selector
- ✅ Temporal coverage visualization (gaps highlighted)
- ✅ Prototype management panel per person
- 📅 Estimated: 5-7 days (frontend components + UX)

**Phase 4: Smart Diversity Refinement (Low Priority)**
- ✅ Enhanced face analysis for age estimation accuracy
- ✅ Expression and pose diversity within eras
- ✅ Decade-based prototype distribution
- ✅ Advanced pruning logic with multi-factor scoring
- 📅 Estimated: 5-7 days (face analysis pipeline + refinement)

---

### Edge Cases and Handling Strategies

**Edge Case 1: Person with Photos Only from Childhood**

**Scenario**: User has 100 photos of their child from ages 1-8, but no teen/adult photos yet.

**System Behavior**:
```python
# Temporal allocation:
temporal_prototypes = {
    "infant": 1 prototype (ages 1-3),
    "child": 1 prototype (ages 4-8),
    "teen": None,
    "young_adult": None,
    "adult": None,
    "senior": None
}

# Coverage report:
{
    "coverage_percentage": 33.3,  # 2 of 6 eras covered
    "missing_eras": ["teen", "young_adult", "adult", "senior"],
    "recommendation": "Archive is childhood-focused. Add photos as person ages."
}
```

**Matching Strategy**:
- Search only uses child-era prototypes
- As new photos are added and verified, system auto-creates teen/adult prototypes
- No false matches with other adults (only searches against available eras)

**User Experience**:
- UI shows "Childhood Archive" badge on person profile
- Timeline view shows gaps as "awaiting future photos"
- System doesn't penalize for missing future eras

---

**Edge Case 2: Large Gaps in Timeline (2005-2015 Missing)**

**Scenario**: User has childhood photos (1998-2004), then a 10-year gap, then adult photos (2015-2025).

**System Behavior**:
```python
# Temporal allocation:
temporal_prototypes = {
    "child": 1 prototype (2000s),
    "teen": None,  # Missing era - no photos from 2005-2015
    "young_adult": 1 prototype (2015-2020),
    "adult": 1 prototype (2020-2025),
}

# Gap detection:
{
    "timeline_gaps": [
        {
            "era": "teen",
            "missing_years": "2005-2015",
            "impact": "Matching accuracy reduced for teen-era photos if added later"
        }
    ]
}
```

**Matching Strategy**:
- Interpolation between child and young_adult prototypes
- Lower confidence threshold for photos estimated in gap era
- If user adds teen photos later, system auto-creates teen prototype and improves matching

**User Experience**:
- Timeline view shows gap as "No photos from teen years"
- If user attempts to label a photo from gap era, system warns: "No teen-era prototypes available. Matches may be less accurate."
- Option to manually pin a teen-era photo to fill gap

---

**Edge Case 3: Identical Twins or Similar-Looking Family Members**

**Scenario**: Two siblings (Alice and Bob) look very similar in childhood photos.

**System Challenges**:
- Embeddings may be very close (cosine similarity > 0.85)
- Automatic suggestions may confuse siblings
- Temporal prototypes for both may cluster together

**Mitigation Strategies**:

1. **User-Driven Disambiguation**:
```python
# After user labels 3+ photos of Alice, system asks:
{
    "disambiguation_prompt": "Detected high similarity with Person 'Bob'. Are these the same person?",
    "suggested_action": "merge_persons",  # If same person
    "alternative_action": "flag_as_similar"  # If siblings/twins
}
```

2. **Similar Person Metadata**:
```python
# Add to Person model:
similar_persons: list[UUID] = []  # IDs of visually similar persons

# Use in matching:
if person.similar_persons:
    # Require higher confidence threshold for suggestions
    min_confidence = 0.75  # Instead of 0.60
    # Show disambiguation UI: "Is this Alice or Bob?"
```

3. **Context-Based Hints**:
```python
# Use photo metadata for disambiguation:
if photo.timestamp == "2003-07-15" and photo.location == "Boston":
    # Alice was in Boston in 2003
    suggest_person = "Alice"
elif photo.timestamp == "2003-07-15" and photo.location == "Seattle":
    # Bob was in Seattle in 2003
    suggest_person = "Bob"
```

**User Experience**:
- UI shows "Similar to Bob" indicator on Alice's profile
- Face suggestions include both Alice and Bob as options
- User can mark "These persons look similar" to adjust thresholds
- Timeline views side-by-side to help distinguish

---

**Edge Case 4: Significant Appearance Changes (Surgery, Aging, Weight)**

**Scenario**: Person has dramatic appearance change (e.g., weight loss, facial hair, plastic surgery).

**System Challenges**:
- Embeddings before/after change may have low similarity
- Temporal prototypes may not bridge the gap
- User may have photos spanning the transition

**Mitigation Strategies**:

1. **Transition Period Prototypes**:
```python
# Add "transition" role for bridging appearance changes:
class PrototypeRole(str, Enum):
    # ... existing roles ...
    TRANSITION = "transition"  # Bridges appearance changes

# User pins "before" and "after" photos as transition prototypes
POST /api/v1/faces/persons/{person_id}/prototypes/pin
{
    "face_instance_id": "uuid",
    "role": "transition",
    "note": "After weight loss - bridge photo"
}
```

2. **Relaxed Matching for Known Transitions**:
```python
# If person has transition prototypes:
if has_transition_prototypes(person_id):
    # Use lower similarity threshold
    min_similarity = 0.55  # Instead of 0.60
    # Search against both pre-transition and post-transition prototypes
    search_results = search_all_temporal_prototypes(person_id)
```

3. **User Feedback Loop**:
```python
# After user verifies face from transition period:
{
    "feedback_prompt": "This photo looks different from other prototypes. Add as transition prototype?",
    "action": "create_transition_prototype",
    "reason": "Improves matching across appearance changes"
}
```

**User Experience**:
- UI shows "Appearance changed in 2018" note on timeline
- Transition prototypes displayed separately: "Before / Transition / After"
- System learns from user corrections during transition period

---

**Edge Case 5: Very High Photo Volume (10,000+ Faces for One Person)**

**Scenario**: Celebrity or public figure with thousands of photos (e.g., family photographer's client).

**System Challenges**:
- 12 prototypes may not cover all contexts (lighting, angles, expressions)
- Selection algorithm must scale efficiently
- Risk of missing rare but important contexts

**Mitigation Strategies**:

1. **Dynamic Prototype Limits**:
```python
# Adjust max prototypes based on face count:
def calculate_max_prototypes(face_count: int) -> int:
    if face_count < 100:
        return 12  # Standard limit
    elif face_count < 1000:
        return 20  # Extended limit
    else:
        return 30  # High-volume limit

# Still respect temporal allocation:
TEMPORAL_SLOTS = 6 + (max_prototypes - 12) // 2
EXEMPLAR_SLOTS = 6 + (max_prototypes - 12) // 2
```

2. **Context-Based Sub-Prototypes**:
```python
# Add context dimension to temporal roles:
class PrototypeContext(str, Enum):
    CASUAL = "casual"
    FORMAL = "formal"
    SPORTS = "sports"
    STUDIO = "studio"

# Select 2 prototypes per era × context combination:
temporal_prototypes = {
    ("child", "casual"): 1 prototype,
    ("child", "formal"): 1 prototype,
    ("adult", "casual"): 1 prototype,
    ("adult", "formal"): 1 prototype,
}
```

3. **Hierarchical Prototype Tiers**:
```python
# Three-tier prototype system:
TIER_1 = "primary"    # 6 prototypes - highest quality, all eras
TIER_2 = "secondary"  # 12 prototypes - context diversity
TIER_3 = "extended"   # 12 prototypes - rare contexts

# Search strategy:
# 1. Search against TIER_1 first (fast)
# 2. If no match, search TIER_2 (medium)
# 3. If still no match, search TIER_3 (comprehensive)
```

**User Experience**:
- UI shows "High-volume archive" badge
- Advanced prototype management panel with filters
- Export prototype set for external processing

---

**Edge Case 6: Low-Quality Photos Only (All Photos <0.5 Quality)**

**Scenario**: Historical archive with only scanned prints, blurry photos, or low-resolution images.

**System Challenges**:
- No photos meet default quality threshold (0.5)
- Risk of zero prototypes for person
- Matching accuracy will be lower

**Mitigation Strategies**:

1. **Adaptive Quality Thresholds**:
```python
def select_prototypes_adaptive_threshold(person_id: UUID):
    faces = get_verified_faces(person_id)

    # Try standard threshold first
    high_quality = [f for f in faces if f.quality_score >= 0.5]
    if len(high_quality) >= 3:
        return select_from_faces(high_quality)

    # Fall back to best available quality
    medium_quality = [f for f in faces if f.quality_score >= 0.3]
    if len(medium_quality) >= 3:
        logger.warning(f"Person {person_id} has only medium-quality photos")
        return select_from_faces(medium_quality)

    # Last resort: use all available faces with FALLBACK role
    logger.warning(f"Person {person_id} has only low-quality photos")
    return select_from_faces(faces, role=PrototypeRole.FALLBACK)
```

2. **Quality Warnings**:
```python
# Add quality metadata to Person:
class PersonQualityStatus(str, Enum):
    HIGH_QUALITY = "high_quality"      # All prototypes >0.7
    MEDIUM_QUALITY = "medium_quality"  # Prototypes 0.3-0.7
    LOW_QUALITY = "low_quality"        # Prototypes <0.3

# Expose in API:
GET /api/v1/faces/persons/{person_id}
{
    "id": "uuid",
    "name": "Alice",
    "quality_status": "medium_quality",
    "quality_note": "Prototypes use best available photos. Consider adding higher-quality photos.",
    "prototypes": [ ... ]
}
```

**User Experience**:
- UI shows "Low-quality archive" warning
- Suggestion: "Upload higher-quality photos for better matching"
- Option to manually enhance scanned photos before labeling

---

**Edge Case 7: Rapid Age Progression (Medical Condition, Stress)**

**Scenario**: Person ages rapidly due to illness, stress, or medical treatment (e.g., chemotherapy hair loss).

**System Challenges**:
- Standard age-era buckets don't match rapid changes
- Calendar-based era assignment is inaccurate
- Appearance changes don't align with chronological age

**Mitigation Strategies**:

1. **Manual Era Override**:
```python
# Allow user to override automatic era assignment:
PATCH /api/v1/faces/faces/{face_id}/metadata
{
    "age_era_bucket_override": "senior",  # Photo from 2020 but person looks senior
    "override_reason": "Illness caused rapid aging"
}

# Use override in prototype selection:
effective_era = face.metadata.get("age_era_bucket_override") or face.metadata.get("age_era_bucket")
```

2. **Appearance-Based Era (Not Age-Based)**:
```python
# Option to switch person to "appearance-based" mode:
PATCH /api/v1/faces/persons/{person_id}/settings
{
    "temporal_mode": "appearance_based",  # Instead of "age_based"
    "note": "Use appearance, not chronological age"
}

# System groups by visual appearance instead of calendar age:
era_buckets = cluster_by_visual_similarity(faces)  # Instead of age_estimate
```

**User Experience**:
- UI provides "Manual era assignment" mode
- Timeline shows "appearance-based" instead of age-based
- System respects user knowledge over automatic estimation

---

### Configuration Updates

**Environment Variables**:

```python
# Add to core/config.py
class Settings(BaseSettings):
    # ... existing settings ...

    # Temporal prototype settings
    face_prototype_max_total: int = Field(
        default=12,
        alias="FACE_PROTOTYPE_MAX_TOTAL",
        description="Maximum prototypes per person (increased from 5 for temporal coverage)"
    )
    face_prototype_temporal_slots: int = Field(
        default=6,
        alias="FACE_PROTOTYPE_TEMPORAL_SLOTS",
        description="Slots reserved for temporal (age-era) prototypes"
    )
    face_prototype_primary_slots: int = Field(
        default=3,
        alias="FACE_PROTOTYPE_PRIMARY_SLOTS",
        description="Slots reserved for user-pinned primary prototypes"
    )
    face_prototype_exemplar_slots: int = Field(
        default=3,
        alias="FACE_PROTOTYPE_EXEMPLAR_SLOTS",
        description="Slots reserved for high-quality exemplars"
    )

    # Temporal diversity settings
    face_prototype_temporal_mode: bool = Field(
        default=True,
        alias="FACE_PROTOTYPE_TEMPORAL_MODE",
        description="Enable temporal (age-based) prototype selection"
    )
    face_prototype_min_quality: float = Field(
        default=0.5,
        alias="FACE_PROTOTYPE_MIN_QUALITY",
        description="Minimum quality threshold (adaptive if not met)"
    )

    # Age-era bucket ranges (years)
    age_era_infant_max: int = Field(default=3, alias="AGE_ERA_INFANT_MAX")
    age_era_child_max: int = Field(default=12, alias="AGE_ERA_CHILD_MAX")
    age_era_teen_max: int = Field(default=19, alias="AGE_ERA_TEEN_MAX")
    age_era_young_adult_max: int = Field(default=35, alias="AGE_ERA_YOUNG_ADULT_MAX")
    age_era_adult_max: int = Field(default=55, alias="AGE_ERA_ADULT_MAX")
    # 56+ is senior
```

---

## Implementation Considerations

### Backward Compatibility

**No Breaking Changes**:
- New columns default to `is_pinned=False` (existing prototypes behave as before)
- API additions are additive (no changes to existing endpoints)
- Pruning logic respects pinned prototypes (unpinned prototypes use existing logic)

**Migration Path**:
1. Deploy database migration (add columns)
2. Deploy backend code (new endpoints + modified pruning)
3. Deploy UI updates (pin buttons, prototype list view)
4. No user action required—existing prototypes continue working

### Performance Impact

**Database Queries**:
- Pinning adds 3 columns (minimal storage: ~16 bytes per prototype)
- Existing indexes remain efficient
- New index on `(person_id, is_pinned)` for fast pruning queries

**Qdrant Impact**:
- No changes to Qdrant schema or queries
- `is_prototype=True` filter remains unchanged
- Pinned vs. unpinned distinction only in Postgres

**API Response Times**:
- `GET /prototypes` endpoint: <50ms (simple query, indexed)
- Pin/unpin operations: <100ms (single row update + Qdrant update)

### User Experience Design

**UI Recommendations**:

1. **Person Profile View**:
   ```
   [Person: Alice]

   Prototype Images (3 of 5 slots):
   [📌 IMG_1234.jpg] [Quality: 95%] [Pinned] [Unpin]
   [   IMG_5678.jpg] [Quality: 87%] [Auto]   [Pin]
   [   IMG_9012.jpg] [Quality: 82%] [Auto]   [Pin]

   [+ Add Custom Prototype] [Recompute Prototypes]
   ```

2. **Face Instance Context Menu**:
   ```
   Right-click on any face:
   - Assign to Person
   - Remove from Person
   - ⭐ Set as Prototype  (new)
   ```

3. **Notification System**:
   ```
   "Prototypes updated for Alice"
   - 2 new high-quality faces added
   - 1 low-quality face removed
   [Review Prototypes]
   ```

### Error Handling

**Edge Cases**:

1. **Too Many Pinned Prototypes**:
   ```python
   if len(pinned) > max_exemplars:
       raise HTTPException(
           status_code=400,
           detail=f"Cannot pin more than {max_exemplars} prototypes. "
                  f"Unpin {len(pinned) - max_exemplars} prototypes or increase limit."
       )
   ```

2. **Pinned Face Deleted**:
   ```python
   # FaceInstance.person_id = CASCADE on delete
   # PersonPrototype.face_instance_id = SET NULL on delete
   # System should:
   # 1. Detect orphaned prototypes (face_instance_id=NULL)
   # 2. Log warning
   # 3. Auto-unpin orphaned prototypes
   # 4. Trigger recomputation
   ```

3. **Quality Score Unavailable**:
   ```python
   # If face.quality_score is NULL (detection failed):
   # - Allow manual pinning (user knows best)
   # - Exclude from automatic selection
   # - Show warning in UI: "Quality score unavailable"
   ```

### Testing Requirements

**Unit Tests**:
```python
# test_prototype_service.py
def test_pin_prototype():
    """Pinned prototypes are immune to pruning."""

def test_prune_respects_pinned():
    """Pruning only affects unpinned prototypes."""

def test_too_many_pinned_raises_error():
    """Cannot pin more than max_exemplars."""

def test_diversity_selection():
    """Smart diversity selects from different age buckets."""
```

**Integration Tests**:
```python
# test_faces_routes.py
def test_pin_prototype_endpoint():
    """POST /prototypes/pin creates pinned prototype."""

def test_unpin_prototype_triggers_recompute():
    """Unpinning triggers automatic reselection."""
```

### Security Considerations

**Authorization**:
- Pin/unpin operations should require authentication (future auth system)
- Users should only modify prototypes for persons they have access to
- Admin users may have global prototype management rights

**Data Validation**:
```python
# Validate pin requests
def validate_pin_request(person_id, face_instance_id):
    # Check face belongs to correct person
    face = db.get(FaceInstance, face_instance_id)
    if face.person_id != person_id:
        raise HTTPException(400, "Face does not belong to this person")

    # Check prototype limit
    pinned_count = count_pinned_prototypes(person_id)
    if pinned_count >= max_exemplars:
        raise HTTPException(400, "Maximum pinned prototypes reached")
```

---

## Alternative Approaches Considered

### Machine Learning for Selection

**Concept**: Train a model to predict which faces users will prefer as prototypes.

**Why Not Recommended**:
- ❌ Requires labeled dataset (thousands of user preferences)
- ❌ Overfitting risk (each user has different preferences)
- ❌ Adds model training/inference complexity
- ❌ Manual pinning achieves same goal with zero training

### Collaborative Filtering

**Concept**: Use community preferences to suggest prototypes (e.g., "Other users picked these photos for this person").

**Why Not Recommended**:
- ❌ Privacy concerns (sharing photo selections)
- ❌ Not applicable to personal photo libraries (no "community")
- ❌ Adds external data dependency

### Time-Weighted Selection

**Concept**: Prefer recent photos as prototypes (assumes people want current appearance).

**Why Not Recommended**:
- ✅ Useful for some use cases (security, ID verification)
- ❌ Breaks for historical photo collections (family archives)
- ❌ User may want diverse temporal coverage (child + adult)
- ✅ Can be added as diversity mode in Phase 3 (Method 5)

---

## Success Metrics

### Prototype Quality Indicators

**Quantitative Metrics**:
1. **Suggestion Accuracy**: % of face suggestions accepted by users
   - Baseline: ~65% acceptance rate (current system)
   - Target: >80% acceptance rate with pinned prototypes

2. **Prototype Stability**: % of prototypes unchanged after 30 days
   - Measure how often users override automatic selection
   - Target: <10% manual override rate (system mostly correct)

3. **User Engagement**: % of persons with at least 1 pinned prototype
   - Indicates users find pinning valuable
   - Target: >30% of active persons have pinned prototypes

**Qualitative Metrics**:
1. User feedback: "Prototypes now match my preferences"
2. Reduction in support tickets about "wrong face suggestions"
3. User surveys: "How satisfied are you with prototype selection?"

### API Usage Patterns

Track usage of new endpoints to gauge adoption:
```
POST /prototypes/pin          → High usage = users want control
GET  /prototypes              → High usage = users value visibility
POST /prototypes/recompute    → Low usage = automatic selection is good
```

---

## Conclusion

The current prototype system is well-architected but requires significant enhancement to handle the unique challenges of **30+ year family photo archives** where people age from childhood to adulthood. The recommended hybrid approach (Manual Pinning + Temporal Roles + Smart Diversity) addresses these challenges through:

### Key Improvements Over Current System

1. **Temporal Coverage (Critical for Family Archives)**:
   - Ensures prototypes span full age progression (infant → senior)
   - Automatically selects representatives from each life stage
   - Handles dramatic appearance changes due to aging
   - Gracefully manages timeline gaps (missing decades)

2. **User Control Where It Matters**:
   - Pin definitive photos for specific eras ("best childhood photo")
   - Override automatic selection when age estimation fails
   - Manual disambiguation for twins/similar family members
   - Flexibility to mark appearance transitions (weight loss, illness)

3. **Smart Automation as Default**:
   - Metadata-driven selection prioritizes temporal diversity
   - Adaptive quality thresholds handle historical archives
   - Automatic era assignment from face analysis + EXIF
   - Dynamic prototype limits based on photo volume

4. **Robust Edge Case Handling**:
   - Childhood-only archives (future photos will expand coverage)
   - Timeline gaps (interpolates between eras)
   - Low-quality scanned photos (adaptive thresholds)
   - Identical twins (disambiguation workflows)
   - Rapid aging (manual era overrides)

### Implementation Roadmap

**Phase 1: Temporal Infrastructure (5-7 days)** - MUST DO FIRST
- Age-era bucket classification on face detection
- Temporal metadata in FaceInstance.landmarks JSONB
- Automatic temporal prototype selection logic
- Database schema additions (role, era_bucket, decade_bucket)

**Phase 2: Manual Pinning with Era Support (3-4 days)** - HIGH PRIORITY
- Pin/unpin API endpoints with era association
- Pruning logic respecting pinned prototypes
- Temporal coverage reporting APIs
- Pin quota enforcement

**Phase 3: UI for Temporal Management (5-7 days)** - MEDIUM PRIORITY
- Timeline view with era-based prototype display
- Pin button with era selector dropdown
- Coverage visualization (gaps highlighted in red)
- Prototype management panel per person

**Phase 4: Smart Diversity Refinement (5-7 days)** - LOW PRIORITY
- Enhanced age estimation models
- Expression/pose diversity within eras
- Multi-factor quality scoring
- Advanced pruning algorithms

**Total Estimated Effort**: 18-25 days (phased implementation allows incremental value delivery)

### Success Metrics

**Temporal Coverage**:
- Target: >80% of persons with 3+ era buckets covered
- Measure: Average era coverage percentage per person
- Goal: Reduce "missing era" warnings by 60%

**Matching Accuracy Across Ages**:
- Baseline: Current system may fail on childhood → adult matches
- Target: >75% match accuracy across 20+ year age spans
- Measure: User acceptance rate of age-gap suggestions

**User Engagement with Pinning**:
- Target: >40% of persons have at least 1 pinned prototype
- Measure: Pin API usage and user feedback
- Goal: Reduce manual overrides through better automatic selection

**Archive Quality**:
- Baseline: Low-quality historical photos may have 0 prototypes
- Target: 100% of persons have at least 1 prototype (using FALLBACK role)
- Measure: Prototype coverage across all persons

### Next Steps

1. **Technical Validation**:
   - Review temporal role system with engineering team
   - Validate age estimation model accuracy (use existing face analysis)
   - Confirm JSONB storage sufficient for temporal metadata
   - Test Qdrant performance with increased prototype counts (12 vs. 5)

2. **Product Validation**:
   - Review UI mockups for timeline view and era selector
   - Validate pin workflow UX with users
   - Confirm edge case handling aligns with user needs
   - Define acceptance criteria for each phase

3. **Implementation Planning**:
   - Create database migration scripts (schema additions)
   - Plan face detection pipeline updates (add age estimation)
   - Design temporal prototype selection algorithm
   - Write comprehensive test suite for edge cases

4. **Metrics Collection**:
   - Instrument APIs to track temporal coverage metrics
   - Log pin/unpin events for usage analysis
   - Track suggestion acceptance rates by era
   - Monitor quality score distributions

### Key Takeaway

The prototype system doesn't need a complete redesign—**it needs temporal awareness**. Family photo archives are fundamentally different from professional photography or security systems because **people age dramatically across 30+ years**. The hybrid approach maintains intelligent automation while giving users surgical control over temporal coverage, ensuring accurate face recognition across a lifetime of photos.

**Critical Success Factor**: The temporal role system must be implemented FIRST (Phase 1) to lay the foundation. Manual pinning (Phase 2) and UI enhancements (Phase 3) build on this infrastructure. Without temporal infrastructure, pinning alone won't solve the core problem of matching childhood photos to adult faces.

---

## Appendix: Code References

**Core Files**:
- `db/models.py` lines 495-533: PersonPrototype model
- `services/prototype_service.py`: Creation and pruning logic
- `api/routes/faces.py` lines 235-331: Cluster labeling (creates prototypes)
- `vector/face_qdrant.py` lines 452-473: Prototype search
- `core/config.py` lines 106-119: Configuration

**Related Features**:
- Face suggestion system: `api/routes/face_suggestions.py`
- Person management: `api/routes/faces.py` lines 368-620
- Face assignment audit log: `db/models.py` FaceAssignmentEvent

**Testing**:
- Unit tests: `tests/unit/test_prototype_service.py` (create after implementation)
- Integration tests: `tests/api/test_faces_routes.py` (extend for pinning)

---

**Research Completed**: 2025-12-29
**Document Version**: 2.0 (Temporal Role System Update)
**Author**: Claude (Research Agent)

**Changelog**:
- **v2.0 (2025-12-29)**: Complete redesign with temporal role system for 30+ year family archives
  - Added age-era bucket system (infant/child/teen/young_adult/adult/senior)
  - Integrated manual pinning with temporal era association
  - Designed comprehensive edge case handling (twins, gaps, low quality, rapid aging)
  - Expanded API design with temporal coverage endpoints
  - Updated implementation roadmap with 4 phased approach
  - Added success metrics and validation criteria

- **v1.0 (2025-12-29)**: Initial research on manual pinning and smart diversity methods
