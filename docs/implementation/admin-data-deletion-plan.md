# Admin Data Deletion Implementation Plan

**Created**: 2025-01-XX
**Status**: Planning Complete - Ready for Implementation
**Scope**: image-search-service (backend) + image-search-ui (frontend)

## Executive Summary

This document outlines the implementation plan for adding comprehensive admin functionality to delete all data from both Qdrant (vector database) and PostgreSQL (relational database) while preserving the Alembic migration tracking system. This is essential for rapid development workflows where developers need to reset their entire data state.

## Current State Analysis

### Data Storage Architecture

**Qdrant (Vector Database)**:
- **Collection: `image_assets`** (managed by `vector/qdrant.py`): Stores image embedding vectors (512-dim)
- **Collection: `faces`** (managed by `vector/face_qdrant.py`): Stores face embedding vectors (512-dim)

**PostgreSQL (Relational Database)**:
- **Application Tables**:
  - `image_assets` - Image metadata and file paths
  - `categories` - Category organization
  - `training_sessions` - Training session tracking
  - `training_subdirectories` - Subdirectory selection
  - `training_jobs` - Background job tracking
  - `training_evidence` - Training evidence and debugging
  - `vector_deletion_logs` - Audit log for vector deletions
  - `persons` - Person entities for face recognition
  - `face_instances` - Face detection instances
  - `person_prototypes` - Person prototype vectors
  - `face_assignment_events` - Audit log for face assignments
- **System Tables** (Must Preserve):
  - `alembic_version` - Alembic migration tracking (CRITICAL to preserve)

### Existing Vector Management

- **Location**: `/api/v1/vectors` routes
- **Capabilities**: Directory-based deletion, retrain, orphan cleanup, full Qdrant reset
- **UI Location**: `/vectors` route in UI
- **UI Components**: DirectoryStatsTable, DangerZone, DeletionLogsTable, modals

### Migration System

- **Tool**: Alembic
- **Version Tracking**: `alembic_version` table stores current migration revision
- **Migrations Location**: `src/image_search_service/db/migrations/versions/`
- **Current Migrations**: 001-009 plus hash-named migrations
- **Critical**: Must preserve `alembic_version` table to allow running migrations after deletion

## Requirements

1. **Backend API Endpoints** (`image-search-service`):
   - Delete all vectors from Qdrant (`image_assets` and `faces` collections)
   - Delete all data from PostgreSQL tables (except `alembic_version`)
   - Maintain Alembic migration state
   - Provide operation confirmation and logging

2. **Frontend Admin UI** (`image-search-ui`):
   - New `/admin` route
   - UI for triggering full data deletion
   - Proper warnings and confirmations
   - Extensible structure for future admin tasks

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Postgres Deletion Strategy** | TRUNCATE CASCADE | Fast, preserves table structure, respects foreign keys |
| **Alembic Version Preservation** | Explicit exclusion | Must maintain migration state for re-initialization |
| **Qdrant Deletion** | Delete collections and recreate | Clean state, maintains configuration |
| **Safety Confirmation** | Double confirmation required | Destructive operation needs protection |
| **Order of Operations** | Qdrant first, then Postgres | Vectors reference database records; delete dependencies first |
| **Audit Logging** | Log deletion in vector_deletion_logs (if table exists) | Track admin operations for debugging |
| **UI Placement** | Dedicated `/admin` route | Clear separation, extensible for future admin functions |

## Implementation Plan

### Phase 1: Backend API Endpoints

#### 1.1 Create Admin Routes Module

**File**: `image-search-service/src/image_search_service/api/routes/admin.py`

**Purpose**: Admin-only endpoints for data management

**Endpoints**:
- `POST /api/v1/admin/data/delete-all` - Delete all application data

**Considerations**:
- Should include basic authentication/authorization check (simple API key or admin flag)
- Transaction management for atomicity
- Error handling and rollback capability

#### 1.2 Create Admin Service Layer

**File**: `image-search-service/src/image_search_service/services/admin_service.py`

**Purpose**: Business logic for admin operations

**Methods**:
- `async def delete_all_data(confirm: bool, confirmation_text: str, reason: str | None) -> DeleteAllDataResponse`
  - Validates confirmations
  - Deletes Qdrant collections
  - Truncates PostgreSQL tables (except alembic_version)
  - Returns deletion summary

#### 1.3 Qdrant Deletion Implementation

**Strategy**: 
- Use existing `reset_collection()` function from `vector/qdrant.py` for `image_assets`
- Create similar reset function for `faces` collection in `vector/face_qdrant.py`
- Both operations should delete and recreate empty collections

**Files to Modify**:
- `vector/face_qdrant.py` - Add `reset_collection()` method similar to `qdrant.py`

#### 1.4 PostgreSQL Deletion Implementation

**Strategy**: Use TRUNCATE CASCADE to efficiently delete all data while preserving table structures and migration state.

**Implementation**:
```python
# Get list of all tables excluding alembic_version
async def truncate_all_tables(db: AsyncSession) -> dict[str, int]:
    """Truncate all application tables except alembic_version.
    
    Returns:
        Dictionary mapping table names to deleted row counts
    """
    # Execute raw SQL to get table list
    # TRUNCATE each table with CASCADE
    # Return counts
```

**Tables to Truncate** (in dependency order to avoid FK violations):
1. `face_assignment_events`
2. `person_prototypes`
3. `face_instances`
4. `persons`
5. `vector_deletion_logs`
6. `training_evidence`
7. `training_jobs`
8. `training_subdirectories`
9. `training_sessions`
10. `categories`
11. `image_assets`

**Alternative**: Single TRUNCATE with CASCADE:
```sql
TRUNCATE TABLE 
  face_assignment_events,
  person_prototypes,
  face_instances,
  persons,
  vector_deletion_logs,
  training_evidence,
  training_jobs,
  training_subdirectories,
  training_sessions,
  categories,
  image_assets
CASCADE;
```

#### 1.5 Create Admin Schemas

**File**: `image-search-service/src/image_search_service/api/admin_schemas.py`

**Schemas**:
```python
class DeleteAllDataRequest(BaseModel):
    confirm: bool = Field(description="First confirmation flag")
    confirmation_text: str = Field(description="Must match 'DELETE ALL DATA'")
    reason: str | None = Field(None, description="Optional reason for deletion")

class DeleteAllDataResponse(BaseModel):
    qdrant_collections_deleted: dict[str, int]  # collection_name -> vector_count
    postgres_tables_truncated: dict[str, int]  # table_name -> row_count
    alembic_version_preserved: str  # Current migration version
    message: str
    timestamp: datetime
```

#### 1.6 Register Admin Router

**File**: `image-search-service/src/image_search_service/api/routes/__init__.py`

**Add**:
```python
from image_search_service.api.routes.admin import router as admin_router
api_v1_router.include_router(admin_router)
```

### Phase 2: Frontend Admin UI

#### 2.1 Create Admin Route

**File**: `image-search-ui/src/routes/admin/+page.svelte`

**Structure**:
- Page title: "Admin Panel"
- Section: "Data Management"
- Danger zone with delete all data functionality
- Future sections can be added (e.g., "System Configuration", "User Management")

#### 2.2 Create Admin API Client

**File**: `image-search-ui/src/lib/api/admin.ts`

**Functions**:
```typescript
export interface DeleteAllDataRequest {
  confirm: boolean;
  confirmationText: string;
  reason?: string;
}

export interface DeleteAllDataResponse {
  qdrantCollectionsDeleted: Record<string, number>;
  postgresTablesTruncated: Record<string, number>;
  alembicVersionPreserved: string;
  message: string;
  timestamp: string;
}

export async function deleteAllData(
  request: DeleteAllDataRequest
): Promise<DeleteAllDataResponse>
```

#### 2.3 Create Admin Components

**Files**:
- `image-search-ui/src/lib/components/admin/DeleteAllDataModal.svelte`
  - Double confirmation input
  - Warning messages
  - Reason field (optional)
  - Loading state

- `image-search-ui/src/lib/components/admin/AdminDataManagement.svelte`
  - Section component for data management
  - Button to trigger deletion
  - Results display

#### 2.4 Update Navigation

**File**: `image-search-ui/src/routes/+layout.svelte`

**Add admin link** (possibly with admin-only visibility flag):
```svelte
<a href="/admin">Admin</a>
```

#### 2.5 Admin Page Styling

**Style Considerations**:
- Prominent danger zones with red/warning colors
- Clear visual hierarchy
- Consistent with existing Vector Management UI patterns
- Mobile-responsive

### Phase 3: Safety & Testing

#### 3.1 Confirmation Requirements

**Backend**:
- `confirm: true` required
- `confirmation_text` must exactly match `"DELETE ALL DATA"`
- Both conditions enforced

**Frontend**:
- Text input requiring exact match
- Disabled submit until confirmation matches
- Warning banners explaining consequences

#### 3.2 Error Handling

**Backend**:
- Transaction rollback on failure
- Detailed error messages
- Log all operations

**Frontend**:
- Display API errors clearly
- Handle network failures
- Show operation status

#### 3.3 Testing Strategy

**Unit Tests** (`image-search-service/tests/api/test_admin.py`):
- Test confirmation validation
- Test Qdrant deletion
- Test PostgreSQL truncation (with test DB)
- Test alembic_version preservation

**Integration Tests**:
- End-to-end deletion flow
- Verify migrations still work after deletion
- Verify collections are recreated

**Manual Testing**:
- Test deletion in development environment
- Verify data is truly gone
- Verify migrations can be re-run
- Test error scenarios

## Technical Implementation Details

### Backend: Admin Routes

**Location**: `src/image_search_service/api/routes/admin.py`

**Structure**:
```python
router = APIRouter(prefix="/admin", tags=["admin"])

@router.post("/data/delete-all", response_model=DeleteAllDataResponse)
async def delete_all_data(
    request: DeleteAllDataRequest,
    db: AsyncSession = Depends(get_db),
) -> DeleteAllDataResponse:
    """Delete all application data from Qdrant and PostgreSQL.
    
    DANGER: This is irreversible and deletes ALL data except migrations.
    
    Safety: Requires confirm=true AND confirmation_text="DELETE ALL DATA"
    
    Returns: Deletion summary with counts
    """
    # Validate confirmations
    # Delete Qdrant collections
    # Truncate PostgreSQL tables
    # Return summary
```

### Backend: Admin Service

**Key Implementation Points**:

1. **Qdrant Deletion**:
   ```python
   # Delete image_assets collection
   image_vectors_deleted = qdrant.reset_collection()
   
   # Delete faces collection
   face_client = get_face_qdrant_client()
   face_vectors_deleted = face_client.reset_collection()
   ```

2. **PostgreSQL Truncation**:
   ```python
   # Get current alembic version
   result = await db.execute(text("SELECT version_num FROM alembic_version"))
   alembic_version = result.scalar_one()
   
   # Truncate all tables except alembic_version
   tables = [
       "face_assignment_events", "person_prototypes", "face_instances",
       "persons", "vector_deletion_logs", "training_evidence",
       "training_jobs", "training_subdirectories", "training_sessions",
       "categories", "image_assets"
   ]
   
   await db.execute(text(f"TRUNCATE TABLE {', '.join(tables)} CASCADE"))
   await db.commit()
   ```

### Frontend: Admin Page

**UI Flow**:
1. User navigates to `/admin`
2. Sees "Data Management" section
3. Clicks "Delete All Data" button
4. Modal appears with:
   - Warning message
   - Text input requiring "DELETE ALL DATA"
   - Optional reason field
   - Cancel/Confirm buttons
5. On confirmation, shows loading state
6. Displays results (deletion summary)

## Security Considerations

### Current Implementation
- No authentication/authorization enforced
- Suitable for development environments
- **Production Enhancement Needed**: Add admin API key or role-based access

### Future Enhancements
- Admin API key in environment variables
- Admin role/flag checking
- Rate limiting on destructive operations
- IP whitelist for admin endpoints

## Migration Preservation Strategy

### Critical: Preserve `alembic_version`

**Why**: 
- Alembic tracks current schema version in this table
- After deletion, migrations must still work
- Re-running migrations should apply from current state, not reset to base

**Implementation**:
- Explicitly exclude `alembic_version` from truncation
- Read current version before deletion
- Return version in response for verification
- Document that migrations are preserved

**Verification**:
```python
# Before deletion
current_version = await db.execute(
    text("SELECT version_num FROM alembic_version")
).scalar_one()

# After deletion (verify preserved)
new_version = await db.execute(
    text("SELECT version_num FROM alembic_version")
).scalar_one()

assert current_version == new_version
```

## API Contract

### Request

**Endpoint**: `POST /api/v1/admin/data/delete-all`

**Body**:
```json
{
  "confirm": true,
  "confirmationText": "DELETE ALL DATA",
  "reason": "Development reset before testing new features"
}
```

### Response

**Success (200 OK)**:
```json
{
  "qdrantCollectionsDeleted": {
    "image_assets": 15230,
    "faces": 8743
  },
  "postgresTablesTruncated": {
    "image_assets": 15230,
    "categories": 5,
    "training_sessions": 12,
    "persons": 45,
    "face_instances": 8743
  },
  "alembicVersionPreserved": "009",
  "message": "Successfully deleted all application data",
  "timestamp": "2025-01-XXT10:30:00Z"
}
```

**Error (400 Bad Request)**:
```json
{
  "detail": "confirmationText must exactly match 'DELETE ALL DATA'"
}
```

## File Structure

### Backend Files

```
image-search-service/
├── src/image_search_service/
│   ├── api/
│   │   ├── routes/
│   │   │   └── admin.py                    # NEW: Admin routes
│   │   └── admin_schemas.py                # NEW: Admin Pydantic schemas
│   ├── services/
│   │   └── admin_service.py                # NEW: Admin business logic
│   └── vector/
│       └── face_qdrant.py                  # MODIFY: Add reset_collection()
└── tests/
    └── api/
        └── test_admin.py                   # NEW: Admin route tests
```

### Frontend Files

```
image-search-ui/
├── src/
│   ├── routes/
│   │   └── admin/
│   │       └── +page.svelte                # NEW: Admin page
│   └── lib/
│       ├── api/
│       │   └── admin.ts                    # NEW: Admin API client
│       └── components/
│           └── admin/
│               ├── DeleteAllDataModal.svelte    # NEW
│               └── AdminDataManagement.svelte   # NEW
```

## Testing Checklist

### Backend Tests

- [ ] Test confirmation validation (missing confirm)
- [ ] Test confirmation text validation (wrong text)
- [ ] Test Qdrant collection deletion (both collections)
- [ ] Test PostgreSQL truncation (all tables)
- [ ] Test alembic_version preservation
- [ ] Test error handling and rollback
- [ ] Test transaction atomicity
- [ ] Test logging and audit trail

### Frontend Tests

- [ ] Test admin page renders
- [ ] Test modal opens/closes
- [ ] Test confirmation text input validation
- [ ] Test submit button disabled until confirmation matches
- [ ] Test API call with correct parameters
- [ ] Test error display
- [ ] Test success message display
- [ ] Test loading states

### Integration Tests

- [ ] Test full deletion flow end-to-end
- [ ] Verify all Qdrant vectors deleted
- [ ] Verify all PostgreSQL rows deleted
- [ ] Verify alembic_version preserved
- [ ] Verify migrations can be re-run after deletion
- [ ] Test error scenarios (Qdrant down, DB error)

## Rollout Strategy

### Development Environment

1. Implement backend endpoints
2. Test with local Qdrant and Postgres
3. Implement frontend UI
4. Test full flow
5. Document usage

### Staging Environment

1. Deploy backend changes
2. Test with production-like data volumes
3. Verify performance
4. Test error scenarios

### Production Considerations

**WARNING**: This feature is extremely destructive. Consider:
- Adding additional auth checks
- Rate limiting
- IP whitelisting
- Audit logging to external system
- Requiring multiple admin approvals

## Future Enhancements

The `/admin` route is structured to permit adding other admin tasks:

### Potential Future Admin Functions

1. **System Configuration**:
   - View/edit environment variables
   - Service health checks
   - Performance metrics

2. **Data Management**:
   - Export/import data snapshots
   - Database backups
   - Qdrant backups

3. **User Management** (if adding auth):
   - User roles
   - API keys
   - Access logs

4. **Maintenance**:
   - Reindex Qdrant
   - Rebuild database indexes
   - Cleanup orphaned files

## Dependencies

### Backend
- No new dependencies required
- Uses existing:
  - `qdrant_client` for Qdrant operations
  - `sqlalchemy` for database operations
  - `fastapi` for API routes

### Frontend
- No new dependencies required
- Uses existing SvelteKit patterns

## Success Criteria

1. ✅ Admin can delete all vectors from both Qdrant collections
2. ✅ Admin can delete all data from PostgreSQL tables
3. ✅ Alembic migration state is preserved
4. ✅ Migrations can be re-run after deletion
5. ✅ Proper confirmations prevent accidental deletions
6. ✅ Clear UI/UX for admin operations
7. ✅ Extensible structure for future admin features
8. ✅ Comprehensive error handling
9. ✅ Tests cover critical paths

## Timeline Estimate

- **Phase 1 (Backend)**: 2-3 days
  - Admin routes and schemas: 0.5 day
  - Admin service implementation: 1 day
  - Face Qdrant reset method: 0.5 day
  - Testing: 1 day

- **Phase 2 (Frontend)**: 2 days
  - Admin page and components: 1 day
  - API client and integration: 0.5 day
  - Styling and polish: 0.5 day

- **Phase 3 (Testing & Documentation)**: 1 day
  - Integration testing
  - Documentation updates

**Total**: ~5-6 days

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Accidental deletion in production | Require explicit confirmation text, admin-only access |
| Migration state lost | Explicit check to preserve alembic_version, verify after deletion |
| Transaction failures | Proper error handling, rollback, detailed logging |
| Qdrant collection recreation fails | Check collection config before deletion, handle recreation errors |
| Foreign key violations | Use TRUNCATE CASCADE, proper table ordering |

## References

- Existing vector management: `api/routes/vectors.py`
- Qdrant client: `vector/qdrant.py`, `vector/face_qdrant.py`
- Database models: `db/models.py`
- Alembic migrations: `db/migrations/versions/`
- UI vector management: `image-search-ui/src/routes/vectors/+page.svelte`

---

**Status**: 📋 Planning Complete - Ready for Implementation

**Next Steps**:
1. Review and approve this plan
2. Implement Phase 1 (Backend)
3. Implement Phase 2 (Frontend)
4. Comprehensive testing
5. Documentation updates

