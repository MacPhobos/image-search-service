# Feature: Auto-Rescan on Prototype Recomputation

## Overview
This feature adds an **optional auto-rescan** capability that triggers suggestion regeneration when prototypes are recomputed.

## Implementation Details

### 1. Configuration Flag
**File**: `src/image_search_service/core/config.py`

New setting:
```python
face_suggestions_auto_rescan_on_recompute: bool = Field(
    default=False,
    alias="FACE_SUGGESTIONS_AUTO_RESCAN_ON_RECOMPUTE",
    description="Automatically regenerate suggestions when prototypes are recomputed",
)
```

**Default**: `False` (no behavior change by default)

**Environment Variable**: `FACE_SUGGESTIONS_AUTO_RESCAN_ON_RECOMPUTE=true`

### 2. Request Schema Update
**File**: `src/image_search_service/api/face_schemas.py`

#### RecomputePrototypesRequest
```python
class RecomputePrototypesRequest(CamelCaseModel):
    preserve_pins: bool = Field(
        default=True,
        description="Whether to preserve manually pinned prototypes"
    )
    trigger_rescan: bool | None = Field(
        default=None,
        description="Trigger suggestion rescan after recompute. If None, uses config default."
    )
```

#### RecomputePrototypesResponse
```python
class RecomputePrototypesResponse(CamelCaseModel):
    prototypes_created: int
    prototypes_removed: int
    coverage: TemporalCoverage
    rescan_triggered: bool = False
    rescan_message: str | None = None
```

### 3. Endpoint Logic
**File**: `src/image_search_service/api/routes/faces.py`

**Endpoint**: `POST /api/v1/faces/persons/{person_id}/prototypes/recompute`

**Behavior**:
1. Accepts optional `trigger_rescan` parameter in request body
2. If `trigger_rescan` is `None`, uses config default `face_suggestions_auto_rescan_on_recompute`
3. If `trigger_rescan` is `True`:
   - After recomputing prototypes, finds highest quality prototype
   - Expires existing pending suggestions for the person
   - Queues propagation job with best prototype
4. Returns `rescan_triggered` and `rescan_message` in response

## Usage Examples

### Example 1: Use Config Default
```bash
# Config: FACE_SUGGESTIONS_AUTO_RESCAN_ON_RECOMPUTE=false
POST /api/v1/faces/persons/123e4567-e89b-12d3-a456-426614174000/prototypes/recompute
{
  "preservePins": true
}

# Response:
{
  "prototypesCreated": 3,
  "prototypesRemoved": 1,
  "coverage": { ... },
  "rescanTriggered": false,
  "rescanMessage": null
}
```

### Example 2: Explicit Override (Enable)
```bash
POST /api/v1/faces/persons/123e4567-e89b-12d3-a456-426614174000/prototypes/recompute
{
  "preservePins": true,
  "triggerRescan": true
}

# Response:
{
  "prototypesCreated": 3,
  "prototypesRemoved": 1,
  "coverage": { ... },
  "rescanTriggered": true,
  "rescanMessage": "Suggestion rescan queued for John Doe. 12 old suggestions expired."
}
```

### Example 3: Explicit Override (Disable)
```bash
# Config: FACE_SUGGESTIONS_AUTO_RESCAN_ON_RECOMPUTE=true
POST /api/v1/faces/persons/123e4567-e89b-12d3-a456-426614174000/prototypes/recompute
{
  "preservePins": true,
  "triggerRescan": false
}

# Response:
{
  "prototypesCreated": 3,
  "prototypesRemoved": 1,
  "coverage": { ... },
  "rescanTriggered": false,
  "rescanMessage": null
}
```

## Decision Logic

```
trigger_rescan parameter | Config default | Result
------------------------|----------------|------------------
None                    | false          | No rescan
None                    | true           | Rescan triggered
true                    | false          | Rescan triggered (override)
true                    | true           | Rescan triggered
false                   | false          | No rescan
false                   | true           | No rescan (override)
```

## Backend Logic Flow

When `trigger_rescan=true`:

1. **Get person** from database
2. **Get all prototypes** for person
3. **Find best quality prototype**:
   - Load face instances for all prototypes
   - Select face with highest `quality_score`
4. **Expire pending suggestions**:
   - Update all `FaceSuggestion` records where:
     - `suggested_person_id = person_id`
     - `status = PENDING`
   - Set `status = EXPIRED`, `reviewed_at = now()`
5. **Queue propagation job**:
   - Job: `propagate_person_label_job`
   - Args: `source_face_id`, `person_id`
   - Confidence threshold: 0.7
   - Max suggestions: 50
6. **Return rescan status**:
   - `rescan_triggered = true`
   - `rescan_message = "Rescan queued for {name}. {count} suggestions expired."`

## Testing

### Unit Tests
**File**: `tests/unit/api/test_face_schemas.py`

- ✅ Default values (preserve_pins=True, trigger_rescan=None)
- ✅ Explicit values (preserve_pins=False, trigger_rescan=True)
- ✅ trigger_rescan=None uses config default
- ✅ CamelCase serialization
- ✅ Response with and without rescan info

### Integration Test
```bash
# Start services
make dev     # Terminal 1
make worker  # Terminal 2

# Test endpoint
curl -X POST http://localhost:8000/api/v1/faces/persons/{person_id}/prototypes/recompute \
  -H "Content-Type: application/json" \
  -d '{
    "preservePins": true,
    "triggerRescan": true
  }'
```

## Acceptance Criteria

- [x] Config flag `FACE_SUGGESTIONS_AUTO_RESCAN_ON_RECOMPUTE` added (default: False)
- [x] Recompute endpoint accepts optional `trigger_rescan` parameter
- [x] When `trigger_rescan=true`, suggestions are regenerated after recompute
- [x] Response includes `rescan_triggered` and `rescan_message`
- [x] Code passes `make lint && make typecheck`
- [x] Unit tests added and passing

## Future Enhancements

1. **Batch Rescan**: Trigger rescan for multiple persons at once
2. **Rescan Progress**: Return job_id to track rescan progress
3. **Rescan Statistics**: Include stats in response (e.g., expected new suggestions)
4. **Conditional Rescan**: Only trigger if coverage improved significantly

## Related Documentation

- Manual regenerate endpoint: `POST /api/v1/faces/persons/{person_id}/suggestions/regenerate`
- Face suggestions: `GET /api/v1/faces/instances/{face_id}/suggestions`
- Prototype management: `/api/v1/faces/persons/{person_id}/prototypes`
