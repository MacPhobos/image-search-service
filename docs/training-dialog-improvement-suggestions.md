# Create Training Session Dialog - Improvement Suggestions

**Date**: 2025-12-30
**Version**: 1.0
**Status**: Awaiting User Approval

---

## 1. Executive Summary

This document outlines comprehensive improvements to the Create Training Session dialog based on detailed research of the codebase. The improvements address user-requested features, critical bugs, and significant usability enhancements.

**Key Findings**:
- Filter functionality **already exists** (text-based)
- Backend **already tracks** trained directory counts via `trainedCount` field
- CategorySelector has a **critical refresh bug** when creating new categories
- No localStorage usage for remembering user preferences
- DirectoryInfo API response lacks training status metadata

**Improvement Categories**:
- **P0 (Critical)**: 1 bug fix
- **P1 (Required)**: 4 user-requested features
- **P2 (Nice-to-have)**: 5 usability enhancements

**Total Estimated Effort**: 2-3 days (1 developer)

---

## 2. Current State Analysis

### 2.1 What Works Well

✅ **Two-Step Workflow**: Clean separation between info entry and subdirectory selection
✅ **Text-Based Filtering**: DirectoryBrowser already has case-insensitive path filtering
✅ **Select All on Filtered Results**: Correctly operates on filtered subdirectories only
✅ **Inline Category Creation**: CategoryCreateModal allows creating categories without leaving dialog
✅ **Backend Training Tracking**: Database tracks `trainedCount` per subdirectory via `TrainingSubdirectory` model
✅ **Responsive Design**: Dialog adapts to viewport (max-width: 700px, max-height: 90vh)

### 2.2 Critical Issues

🐛 **P0: CategorySelector Refresh Bug**
- **Problem**: After creating a new category via CategoryCreateModal, the category dropdown doesn't show the new category until page refresh
- **Root Cause**: CategorySelector loads categories only on mount; no refresh mechanism
- **Impact**: User sees category ID in a disabled input but can't select it from dropdown

⚠️ **Missing Features**
- No localStorage for remembering last-used root path and category
- No visual indicators for already-trained directories
- No filter/checkbox to hide trained directories
- DirectoryInfo API lacks training status metadata

### 2.3 Architecture Overview

```
CreateSessionModal (700px max-width)
├── Step 1: Info Entry
│   ├── Session Name Input
│   ├── Root Path Input
│   ├── CategorySelector ← REFRESH BUG HERE
│   └── CategoryCreateModal (500px, nested)
└── Step 2: Subdirectory Selection
    └── DirectoryBrowser (400px max-height, scrollable)
        ├── Filter Input (text-based) ← ALREADY EXISTS
        ├── Select All / Deselect All
        └── Subdirectory List (checkboxes)
```

**State Management**: Svelte 5 runes (`$state`, `$derived`, `$props`, `$effect`)
**Persistence**: None (no localStorage/cookies)
**API Communication**: Auto-generated TypeScript client from OpenAPI spec

---

## 3. Proposed Improvements

### Priority Matrix

| ID | Feature | Effort | Impact | Priority | Estimate |
|----|---------|--------|--------|----------|----------|
| P0-1 | Fix CategorySelector refresh bug | Low | High | **P0** | 2 hours |
| P1-1 | Remember last root path (localStorage) | Low | Medium | **P1** | 1 hour |
| P1-2 | Remember last category (localStorage) | Low | Medium | **P1** | 1 hour |
| P1-3 | Increase dialog dimensions | Low | Medium | **P1** | 0.5 hours |
| P1-4 | Add training status to DirectoryInfo API | Medium | High | **P1** | 4 hours |
| P1-5 | Visual indicators for trained directories | Medium | High | **P1** | 3 hours |
| P1-6 | Checkbox to hide trained directories | Low | High | **P1** | 2 hours |
| P2-1 | Keyboard shortcuts (Enter, Esc) | Low | Medium | P2 | 2 hours |
| P2-2 | Validation improvements | Medium | Medium | P2 | 3 hours |
| P2-3 | Progress indicators for API calls | Low | Low | P2 | 2 hours |
| P2-4 | Accessibility improvements (ARIA) | Medium | Medium | P2 | 3 hours |
| P2-5 | Error recovery mechanisms | Medium | Medium | P2 | 3 hours |

**Total P0 Effort**: 2 hours
**Total P1 Effort**: 11.5 hours
**Total P2 Effort**: 13 hours

---

## 4. Implementation Details

### P0-1: Fix CategorySelector Refresh Bug

**Problem**: New category created via CategoryCreateModal doesn't appear in dropdown until page refresh.

**Root Cause**: CategorySelector loads categories only once on mount.

**Solution**: Add `refresh()` method to CategorySelector and call it after category creation.

#### Files Affected

**Frontend**:
- `/export/workspace/image-search/image-search-ui/src/lib/components/CategorySelector.svelte`
- `/export/workspace/image-search/image-search-ui/src/lib/components/training/CreateSessionModal.svelte`

#### Implementation

**Step 1: Add refresh method to CategorySelector**

```typescript
// CategorySelector.svelte (add after loadCategories function)

export function refresh() {
  loadCategories(); // Re-fetch categories from API
}
```

**Step 2: Use `bind:this` in CreateSessionModal to get component reference**

```svelte
<!-- CreateSessionModal.svelte (Step 1: Info Entry section) -->

<script>
  let categorySelectorRef: any; // Component reference

  function handleCategoryCreated(category: Category) {
    categoryId = category.id;
    showCategoryModal = false;

    // Refresh the category dropdown
    if (categorySelectorRef?.refresh) {
      categorySelectorRef.refresh();
    }
  }
</script>

<CategorySelector
  bind:this={categorySelectorRef}
  selectedId={categoryId}
  onSelect={(id) => (categoryId = id)}
  onCreateNew={() => (showCategoryModal = true)}
  showCreateOption={true}
/>
```

#### Testing Strategy

1. Open Create Training Session dialog
2. Click "Create New Category" button
3. Create a new category (e.g., "Test Category")
4. Verify new category appears in dropdown immediately
5. Verify new category is pre-selected
6. Verify old categories still appear in dropdown

#### Risk Assessment

- **Breaking Changes**: None
- **Backward Compatibility**: Full (only adds new method)
- **Migration Needed**: No
- **Performance Impact**: Minimal (one additional API call)

---

### P1-1 & P1-2: Remember Last Used Values (localStorage)

**User Request**: "Remember prior selections (Root Path, Category) using cookies/local storage"

**Implementation**: Use localStorage to persist last-used values between sessions.

#### Files Affected

**Frontend**:
- `/export/workspace/image-search/image-search-ui/src/lib/components/training/CreateSessionModal.svelte`

#### Implementation

```typescript
// CreateSessionModal.svelte

const STORAGE_KEYS = {
  LAST_ROOT_PATH: 'training.lastRootPath',
  LAST_CATEGORY_ID: 'training.lastCategoryId',
};

// Load last-used values on mount
onMount(() => {
  try {
    const lastRootPath = localStorage.getItem(STORAGE_KEYS.LAST_ROOT_PATH);
    if (lastRootPath) {
      rootPath = lastRootPath;
    }

    const lastCategoryId = localStorage.getItem(STORAGE_KEYS.LAST_CATEGORY_ID);
    if (lastCategoryId) {
      const categoryIdNum = parseInt(lastCategoryId, 10);
      if (!isNaN(categoryIdNum)) {
        categoryId = categoryIdNum;
      }
    }
  } catch (err) {
    // Ignore localStorage errors (private browsing, quota exceeded)
    console.warn('Failed to load last-used values from localStorage:', err);
  }
});

// Save values when creating session
async function handleCreate() {
  try {
    loading = true;
    error = null;

    // Save to localStorage for next time
    try {
      localStorage.setItem(STORAGE_KEYS.LAST_ROOT_PATH, rootPath);
      if (categoryId !== null) {
        localStorage.setItem(STORAGE_KEYS.LAST_CATEGORY_ID, categoryId.toString());
      }
    } catch (err) {
      // Ignore save errors
      console.warn('Failed to save last-used values to localStorage:', err);
    }

    // Create session via API
    const session = await createTrainingSession({
      name: sessionName,
      rootPath,
      categoryId: categoryId!,
      subdirectories: selectedSubdirs,
    });

    onSessionCreated(session.id);
  } catch (err) {
    error = err instanceof Error ? err.message : 'Failed to create session';
  } finally {
    loading = false;
  }
}
```

#### Additional Enhancement: Clear Button

```svelte
<!-- Optional: Add "Clear Saved Values" button to dialog -->

<button
  type="button"
  class="clear-button"
  onclick={clearSavedValues}
  title="Clear saved root path and category"
>
  Clear Saved Values
</button>

<script>
function clearSavedValues() {
  try {
    localStorage.removeItem(STORAGE_KEYS.LAST_ROOT_PATH);
    localStorage.removeItem(STORAGE_KEYS.LAST_CATEGORY_ID);
    rootPath = '';
    categoryId = null;
  } catch (err) {
    console.warn('Failed to clear saved values:', err);
  }
}
</script>
```

#### Testing Strategy

1. Create a training session with specific root path and category
2. Close and reopen the Create Training Session dialog
3. Verify root path and category are pre-filled
4. Change values and create another session
5. Verify new values are remembered on next open
6. Test in private browsing mode (should gracefully handle localStorage unavailable)
7. Test with localStorage full (quota exceeded)

#### Risk Assessment

- **Breaking Changes**: None
- **Privacy Considerations**: Values stored in plain text in localStorage (not sensitive data)
- **Storage Quota**: Minimal (~100 bytes per user)
- **Browser Compatibility**: All modern browsers support localStorage

---

### P1-3: Increase Dialog Dimensions

**User Request**: "Make dialog wider and taller"

**Current Size**: 700px max-width, 90vh max-height
**Proposed Size**: 900px max-width, 90vh max-height (no change to height)

**Rationale**:
- DirectoryBrowser will display additional training status information (badges, progress bars)
- 900px provides better visibility without being overwhelming
- Height is already optimal (90vh adapts to viewport)

#### Files Affected

**Frontend**:
- `/export/workspace/image-search/image-search-ui/src/lib/components/training/CreateSessionModal.svelte`

#### Implementation

```css
/* CreateSessionModal.svelte */

.modal-content {
  background-color: white;
  border-radius: 8px;
  max-width: 900px;  /* Was 700px */
  width: 90%;
  max-height: 90vh;
  display: flex;
  flex-direction: column;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
```

#### Responsive Breakpoints

```css
/* Optional: Adjust for smaller screens */

@media (max-width: 1024px) {
  .modal-content {
    max-width: 800px;
  }
}

@media (max-width: 768px) {
  .modal-content {
    max-width: 95%;
  }
}
```

#### Testing Strategy

1. Open dialog on various screen sizes (1920x1080, 1366x768, 1024x768)
2. Verify dialog doesn't overflow viewport
3. Verify content is readable and not cramped
4. Test on mobile/tablet viewports
5. Compare with other modals (CategoryCreateModal: 500px, RetrainModal: 550px)

---

### P1-4: Add Training Status to DirectoryInfo API

**User Request**: "Add visual indicators for already-trained directories"

**Current API Response** (DirectoryInfo):
```typescript
{
  path: string;
  name: string;
  imageCount: number;
  selected: boolean; // Always false
}
```

**Enhanced API Response**:
```typescript
{
  path: string;
  name: string;
  imageCount: number;
  selected: boolean;
  trainedCount?: number;        // NEW: Number of trained images
  lastTrainedAt?: string;       // NEW: ISO 8601 timestamp
  trainingStatus?: 'never' | 'partial' | 'complete'; // NEW: Derived status
}
```

#### Files Affected

**Backend**:
- `/export/workspace/image-search/image-search-service/src/image_search_service/api/training_schemas.py`
- `/export/workspace/image-search/image-search-service/src/image_search_service/api/routes/training.py`
- `/export/workspace/image-search/image-search-service/src/image_search_service/services/directory_service.py`
- `/export/workspace/image-search/image-search-service/src/image_search_service/services/training_service.py`

**Frontend**:
- `/export/workspace/image-search/image-search-ui/src/lib/api/generated.ts` (auto-generated)
- `/export/workspace/image-search/image-search-ui/src/lib/types.ts`

#### Implementation

**Step 1: Update Pydantic Schema (Backend)**

```python
# training_schemas.py

class DirectoryInfo(BaseModel):
    """Subdirectory information with optional training status."""

    path: str
    name: str
    image_count: int = Field(alias="imageCount")
    selected: bool = False

    # New fields for training status
    trained_count: int | None = Field(None, alias="trainedCount")
    last_trained_at: datetime | None = Field(None, alias="lastTrainedAt")
    training_status: str | None = Field(None, alias="trainingStatus")

    model_config = ConfigDict(populate_by_name=True)
```

**Step 2: Add Query Parameter to API Endpoint**

```python
# routes/training.py

@router.get("/directories", response_model=list[DirectoryInfo])
async def list_directories(
    path: str = Query(..., description="Root path to scan"),
    include_training_status: bool = Query(
        False,
        description="Include training status metadata (requires DB lookup)"
    ),
    db: AsyncSession = Depends(get_db)
) -> list[DirectoryInfo]:
    """List subdirectories with optional training status."""

    dir_service = DirectoryService()
    subdirs = dir_service.list_subdirectories(path)

    # Optionally enrich with training status
    if include_training_status:
        training_service = TrainingService()
        subdirs = await training_service.enrich_with_training_status(
            db, subdirs, path
        )

    return subdirs
```

**Step 3: Implement Training Status Enrichment Service**

```python
# services/training_service.py

async def enrich_with_training_status(
    self,
    db: AsyncSession,
    subdirs: list[DirectoryInfo],
    root_path: str
) -> list[DirectoryInfo]:
    """Enrich directory list with training status metadata."""

    # Query training_subdirectories table for all subdirectories under this root
    query = (
        select(TrainingSubdirectory)
        .join(TrainingSession)
        .where(TrainingSession.root_path == root_path)
        .where(TrainingSubdirectory.trained_count > 0)
    )

    result = await db.execute(query)
    trained_subdirs = result.scalars().all()

    # Build lookup map: path -> training metadata
    training_map: dict[str, dict] = {}
    for ts in trained_subdirs:
        if ts.path not in training_map:
            training_map[ts.path] = {
                'trained_count': ts.trained_count,
                'last_trained_at': ts.created_at,
                'total_trained': ts.trained_count
            }
        else:
            # If multiple sessions trained this directory, aggregate
            training_map[ts.path]['total_trained'] += ts.trained_count
            if ts.created_at > training_map[ts.path]['last_trained_at']:
                training_map[ts.path]['last_trained_at'] = ts.created_at

    # Enrich subdirectories
    for subdir in subdirs:
        if subdir.path in training_map:
            metadata = training_map[subdir.path]
            subdir.trained_count = metadata['total_trained']
            subdir.last_trained_at = metadata['last_trained_at']

            # Calculate training status
            if subdir.trained_count == 0:
                subdir.training_status = 'never'
            elif subdir.trained_count >= subdir.image_count:
                subdir.training_status = 'complete'
            else:
                subdir.training_status = 'partial'
        else:
            # Never trained
            subdir.trained_count = 0
            subdir.last_trained_at = None
            subdir.training_status = 'never'

    return subdirs
```

**Step 4: Update Frontend API Call**

```typescript
// DirectoryBrowser.svelte

async function loadSubdirectories() {
  try {
    loading = true;
    error = null;

    // Fetch with training status metadata
    const response = await fetch(
      `/api/v1/training/directories?path=${encodeURIComponent(rootPath)}&include_training_status=true`
    );

    if (!response.ok) {
      throw new Error(`Failed to load directories: ${response.statusText}`);
    }

    subdirs = await response.json();
  } catch (err) {
    error = err instanceof Error ? err.message : 'Failed to load directories';
  } finally {
    loading = false;
  }
}
```

**Step 5: Regenerate API Types**

```bash
# In image-search-ui directory
npm run gen:api
```

#### Testing Strategy

**Backend Tests**:
```python
# test_training_routes.py

async def test_list_directories_without_training_status(client):
    """Should not include training status by default."""
    response = await client.get("/api/v1/training/directories?path=/test")
    assert response.status_code == 200
    data = response.json()
    assert all(d.get("trainedCount") is None for d in data)

async def test_list_directories_with_training_status(client, db_session):
    """Should include training status when requested."""
    # Create test training session with trained subdirectories
    session = await create_test_session(db_session)

    response = await client.get(
        "/api/v1/training/directories?path=/test&include_training_status=true"
    )
    assert response.status_code == 200
    data = response.json()

    # Find trained subdirectory
    trained = next(d for d in data if d["path"] == "trained_subdir")
    assert trained["trainedCount"] == 50
    assert trained["trainingStatus"] == "complete"
    assert trained["lastTrainedAt"] is not None

    # Find never-trained subdirectory
    untrained = next(d for d in data if d["path"] == "untrained_subdir")
    assert untrained["trainedCount"] == 0
    assert untrained["trainingStatus"] == "never"
```

**Frontend Tests**:
```typescript
// DirectoryBrowser.test.ts

it('displays training status badges', async () => {
  const subdirs = [
    { path: 'dir1', name: 'dir1', imageCount: 100, trainedCount: 100, trainingStatus: 'complete' },
    { path: 'dir2', name: 'dir2', imageCount: 50, trainedCount: 25, trainingStatus: 'partial' },
    { path: 'dir3', name: 'dir3', imageCount: 30, trainedCount: 0, trainingStatus: 'never' },
  ];

  render(DirectoryBrowser, { rootPath: '/test', subdirs });

  expect(screen.getByText(/100\/100 trained/i)).toBeInTheDocument();
  expect(screen.getByText(/25\/50 trained/i)).toBeInTheDocument();
  expect(screen.queryByText(/dir3.*trained/i)).not.toBeInTheDocument();
});
```

#### Performance Considerations

**Database Query Optimization**:
- Add index on `training_subdirectories.path` for faster lookups
- Consider caching training status for frequently accessed directories
- Limit query to recent training sessions (e.g., last 6 months)

**Migration**:
```python
# Migration: add_training_subdirectories_path_index.py

def upgrade():
    op.create_index(
        'idx_training_subdirectories_path',
        'training_subdirectories',
        ['path']
    )
```

#### Risk Assessment

- **Breaking Changes**: None (new fields are optional)
- **Performance Impact**: Medium (additional DB query per directory list)
- **Mitigation**: Make `include_training_status` opt-in (defaults to False)
- **Backward Compatibility**: Full (new fields ignored by old clients)

---

### P1-5: Visual Indicators for Trained Directories

**User Request**: "Add visual indicators for already-trained directories"

**Design Options**:

#### Option A: Badge + Background Color (Recommended)

```svelte
<!-- DirectoryBrowser.svelte -->

<div
  class="subdir-item"
  class:trained={subdir.trainedCount > 0}
  class:fully-trained={subdir.trainingStatus === 'complete'}
  class:partially-trained={subdir.trainingStatus === 'partial'}
>
  <input
    type="checkbox"
    checked={isSelected}
    onchange={handleToggle}
    id="subdir-{index}"
  />

  <label for="subdir-{index}" class="subdir-info">
    <div class="subdir-header">
      <span class="subdir-path">{subdir.name}</span>

      {#if subdir.trainedCount > 0}
        <span class="training-badge" class:complete={subdir.trainingStatus === 'complete'}>
          {#if subdir.trainingStatus === 'complete'}
            ✓ Fully Trained
          {:else}
            ⚠ {subdir.trainedCount}/{subdir.imageCount} trained
          {/if}
        </span>
      {/if}
    </div>

    <div class="subdir-meta">
      <span class="image-count">{subdir.imageCount} images</span>

      {#if subdir.lastTrainedAt}
        <span class="last-trained">
          Last trained: {formatRelativeTime(subdir.lastTrainedAt)}
        </span>
      {/if}
    </div>
  </label>
</div>

<style>
  .subdir-item {
    padding: 12px;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    margin-bottom: 8px;
    display: flex;
    gap: 12px;
    align-items: flex-start;
    transition: all 0.2s;
  }

  .subdir-item:hover {
    background-color: #f9fafb;
    border-color: #d1d5db;
  }

  /* Trained directory styling */
  .subdir-item.trained {
    background-color: #f0f9ff;
    border-left: 3px solid #3b82f6;
  }

  .subdir-item.fully-trained {
    background-color: #f0fdf4;
    border-left: 3px solid #22c55e;
  }

  .subdir-item.partially-trained {
    background-color: #fffbeb;
    border-left: 3px solid #f59e0b;
  }

  .subdir-info {
    flex: 1;
    cursor: pointer;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .subdir-header {
    display: flex;
    align-items: center;
    gap: 8px;
    justify-content: space-between;
  }

  .subdir-path {
    font-weight: 500;
    color: #1f2937;
  }

  .training-badge {
    font-size: 0.75rem;
    padding: 2px 8px;
    border-radius: 12px;
    background-color: #fef3c7;
    color: #92400e;
    font-weight: 500;
    display: inline-flex;
    align-items: center;
    gap: 4px;
  }

  .training-badge.complete {
    background-color: #d1fae5;
    color: #065f46;
  }

  .subdir-meta {
    display: flex;
    gap: 12px;
    font-size: 0.875rem;
    color: #6b7280;
  }

  .last-trained {
    font-style: italic;
  }
</style>
```

#### Option B: Progress Bar (Alternative)

```svelte
{#if subdir.trainedCount > 0}
  <div class="training-progress">
    <div class="progress-bar-container">
      <div
        class="progress-bar"
        style="width: {(subdir.trainedCount / subdir.imageCount) * 100}%"
        class:complete={subdir.trainingStatus === 'complete'}
      ></div>
    </div>
    <span class="progress-text">
      {subdir.trainedCount}/{subdir.imageCount} trained
    </span>
  </div>
{/if}

<style>
  .training-progress {
    margin-top: 4px;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .progress-bar-container {
    flex: 1;
    height: 6px;
    background-color: #e5e7eb;
    border-radius: 3px;
    overflow: hidden;
  }

  .progress-bar {
    height: 100%;
    background-color: #f59e0b;
    transition: width 0.3s ease;
  }

  .progress-bar.complete {
    background-color: #22c55e;
  }

  .progress-text {
    font-size: 0.75rem;
    color: #6b7280;
    white-space: nowrap;
  }
</style>
```

#### Helper Functions

```typescript
// DirectoryBrowser.svelte

function formatRelativeTime(isoString: string): string {
  const date = new Date(isoString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 60) return `${diffMins} min ago`;
  if (diffHours < 24) return `${diffHours} hours ago`;
  if (diffDays < 7) return `${diffDays} days ago`;

  return date.toLocaleDateString();
}
```

#### Files Affected

**Frontend**:
- `/export/workspace/image-search/image-search-ui/src/lib/components/training/DirectoryBrowser.svelte`

#### Testing Strategy

1. Create training session and train some subdirectories
2. Partially train one subdirectory (50/100 images)
3. Fully train another subdirectory (100/100 images)
4. Open Create Training Session dialog for same root path
5. Verify:
   - Untrained directories have no badge
   - Partially trained directories have yellow badge
   - Fully trained directories have green badge
   - Last trained time displays correctly
   - Background colors match training status

---

### P1-6: Checkbox to Hide Trained Directories

**User Request**: "Add checkbox to hide previously trained directories"

**Implementation**: Add toggle checkbox above directory list to filter out trained directories.

#### Files Affected

**Frontend**:
- `/export/workspace/image-search/image-search-ui/src/lib/components/training/DirectoryBrowser.svelte`

#### Implementation

```svelte
<!-- DirectoryBrowser.svelte -->

<script lang="ts">
  let filterText = $state('');
  let hideTrainedDirs = $state(false); // NEW

  // Enhanced filtering logic
  let filteredSubdirs = $derived(() => {
    let results = subdirs;

    // Text filter
    if (filterText) {
      results = results.filter((d) =>
        d.path.toLowerCase().includes(filterText.toLowerCase())
      );
    }

    // Training status filter
    if (hideTrainedDirs) {
      results = results.filter((d) => {
        // Hide if fully trained
        return d.trainingStatus !== 'complete';
      });
    }

    return results;
  });
</script>

<div class="directory-browser">
  <div class="browser-header">
    <h3>Select Subdirectories</h3>
    <div class="actions">
      <button type="button" onclick={selectAll}>Select All</button>
      <button type="button" onclick={deselectAll}>Deselect All</button>
    </div>
  </div>

  {#if !loading && subdirs.length > 0}
    <div class="filter-container">
      <input
        type="text"
        bind:value={filterText}
        placeholder="Filter directories..."
        class="filter-input"
        aria-label="Filter directories"
      />

      <!-- NEW: Hide trained checkbox -->
      <label class="checkbox-filter">
        <input
          type="checkbox"
          bind:checked={hideTrainedDirs}
        />
        <span>Hide fully trained directories</span>
      </label>

      {#if filterText || hideTrainedDirs}
        <div class="filter-count">
          Showing {filteredSubdirs.length} of {subdirs.length} directories
          {#if hideTrainedDirs}
            ({subdirs.filter(d => d.trainingStatus === 'complete').length} hidden)
          {/if}
        </div>
      {/if}
    </div>
  {/if}

  <!-- Rest of component... -->
</div>

<style>
  .checkbox-filter {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.875rem;
    color: #4b5563;
    cursor: pointer;
    user-select: none;
    padding: 4px 0;
  }

  .checkbox-filter input[type="checkbox"] {
    cursor: pointer;
    width: 16px;
    height: 16px;
  }

  .checkbox-filter:hover {
    color: #1f2937;
  }

  .filter-count {
    font-size: 0.875rem;
    color: #6b7280;
    padding: 4px 0;
  }
</style>
```

#### Enhanced Filter Options (Optional)

```svelte
<!-- Radio buttons for more granular filtering -->

<div class="training-filter">
  <label>Show:</label>
  <label>
    <input type="radio" bind:group={trainingFilter} value="all" />
    All directories
  </label>
  <label>
    <input type="radio" bind:group={trainingFilter} value="untrained" />
    Never trained only
  </label>
  <label>
    <input type="radio" bind:group={trainingFilter} value="partial" />
    Partially trained only
  </label>
  <label>
    <input type="radio" bind:group={trainingFilter} value="complete" />
    Fully trained only
  </label>
</div>

<script>
  let trainingFilter = $state<'all' | 'untrained' | 'partial' | 'complete'>('all');

  let filteredSubdirs = $derived(() => {
    let results = subdirs;

    // Training status filter
    if (trainingFilter !== 'all') {
      results = results.filter((d) => {
        switch (trainingFilter) {
          case 'untrained':
            return d.trainingStatus === 'never';
          case 'partial':
            return d.trainingStatus === 'partial';
          case 'complete':
            return d.trainingStatus === 'complete';
          default:
            return true;
        }
      });
    }

    return results;
  });
</script>
```

#### Testing Strategy

1. Open Create Training Session dialog with mixed trained/untrained directories
2. Check "Hide fully trained directories" checkbox
3. Verify only untrained and partially trained directories appear
4. Verify filter count shows correct numbers
5. Uncheck the checkbox
6. Verify all directories reappear
7. Test combination with text filter
8. Test Select All operates on filtered results only

---

## 5. Additional Usability Improvements (P2)

### P2-1: Keyboard Shortcuts

**Enhancement**: Add keyboard shortcuts for common actions.

#### Implementation

```typescript
// CreateSessionModal.svelte

import { onMount, onDestroy } from 'svelte';

function handleKeyDown(event: KeyboardEvent) {
  // Escape to close modal
  if (event.key === 'Escape') {
    event.preventDefault();
    onClose();
  }

  // Enter to proceed (if valid)
  if (event.key === 'Enter' && !event.shiftKey) {
    if (step === 'info' && canProceedToSubdirs) {
      event.preventDefault();
      handleNext();
    } else if (step === 'subdirs' && canCreate) {
      event.preventDefault();
      handleCreate();
    }
  }

  // Ctrl+A to select all (in subdirs step)
  if (event.ctrlKey && event.key === 'a' && step === 'subdirs') {
    event.preventDefault();
    // Trigger select all in DirectoryBrowser
  }
}

onMount(() => {
  window.addEventListener('keydown', handleKeyDown);
});

onDestroy(() => {
  window.removeEventListener('keydown', handleKeyDown);
});
```

**Keyboard Shortcuts**:
- `Esc`: Close dialog
- `Enter`: Next (step 1) / Create (step 2)
- `Ctrl+A`: Select all directories
- `Ctrl+D`: Deselect all directories

---

### P2-2: Validation Improvements

**Current Validation**: Basic presence checks.

**Enhanced Validation**:

```typescript
// Validation state
let validationErrors = $state<Record<string, string>>({});

function validateSessionName(name: string): string | null {
  if (!name.trim()) {
    return 'Session name is required';
  }
  if (name.length < 3) {
    return 'Session name must be at least 3 characters';
  }
  if (name.length > 100) {
    return 'Session name must be less than 100 characters';
  }
  return null;
}

function validateRootPath(path: string): string | null {
  if (!path.trim()) {
    return 'Root path is required';
  }
  if (!path.startsWith('/')) {
    return 'Root path must be an absolute path';
  }
  // Additional validation via API scan
  return null;
}

function validateCategory(categoryId: number | null): string | null {
  if (categoryId === null) {
    return 'Please select a category';
  }
  return null;
}

// Real-time validation
$effect(() => {
  validationErrors = {
    sessionName: validateSessionName(sessionName),
    rootPath: validateRootPath(rootPath),
    category: validateCategory(categoryId),
  };
});

// Display validation errors
<div class="form-group">
  <label for="session-name">Session Name *</label>
  <input
    id="session-name"
    type="text"
    bind:value={sessionName}
    class:error={validationErrors.sessionName}
    aria-invalid={!!validationErrors.sessionName}
    aria-describedby="session-name-error"
  />
  {#if validationErrors.sessionName}
    <span class="error-message" id="session-name-error">
      {validationErrors.sessionName}
    </span>
  {/if}
</div>
```

---

### P2-3: Progress Indicators

**Enhancement**: Add skeleton loaders and progress feedback.

```svelte
<!-- Loading state for DirectoryBrowser -->

{#if loading}
  <div class="skeleton-loader">
    <div class="skeleton-item" />
    <div class="skeleton-item" />
    <div class="skeleton-item" />
  </div>
{:else}
  <!-- Directory list -->
{/if}

<style>
  .skeleton-loader {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .skeleton-item {
    height: 60px;
    background: linear-gradient(
      90deg,
      #f0f0f0 25%,
      #e0e0e0 50%,
      #f0f0f0 75%
    );
    background-size: 200% 100%;
    animation: loading 1.5s infinite;
    border-radius: 6px;
  }

  @keyframes loading {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
  }
</style>
```

---

### P2-4: Accessibility Improvements

**ARIA Enhancements**:

```svelte
<!-- Modal accessibility -->
<div
  class="modal-overlay"
  role="dialog"
  aria-modal="true"
  aria-labelledby="modal-title"
  aria-describedby="modal-description"
>
  <div class="modal-content">
    <h2 id="modal-title">Create Training Session</h2>
    <p id="modal-description" class="sr-only">
      Create a new training session by selecting a root directory and subdirectories to train.
    </p>

    <!-- Step indicator for screen readers -->
    <div class="step-indicator" aria-live="polite">
      Step {step === 'info' ? '1' : '2'} of 2
    </div>

    <!-- Form fields with proper labels -->
    <label for="root-path">Root Directory Path *</label>
    <input
      id="root-path"
      type="text"
      bind:value={rootPath}
      aria-required="true"
      aria-invalid={!!validationErrors.rootPath}
    />

    <!-- Announce errors to screen readers -->
    <div role="alert" aria-live="assertive">
      {#if error}
        <span class="error-message">{error}</span>
      {/if}
    </div>
  </div>
</div>

<style>
  .sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border-width: 0;
  }
</style>
```

**Focus Management**:

```typescript
import { tick } from 'svelte';

let firstInputRef: HTMLInputElement;

onMount(async () => {
  await tick();
  firstInputRef?.focus(); // Auto-focus first input
});

// Trap focus within modal
function handleKeyDown(event: KeyboardEvent) {
  if (event.key === 'Tab') {
    // Implement focus trap logic
  }
}
```

---

### P2-5: Error Recovery Mechanisms

**Enhancement**: Better error handling with retry and recovery options.

```svelte
<script>
  let retryCount = $state(0);
  const MAX_RETRIES = 3;

  async function handleCreate() {
    try {
      loading = true;
      error = null;

      const session = await createTrainingSession({
        name: sessionName,
        rootPath,
        categoryId: categoryId!,
        subdirectories: selectedSubdirs,
      });

      retryCount = 0; // Reset on success
      onSessionCreated(session.id);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to create session';

      // Classify error type
      if (errorMessage.includes('network') || errorMessage.includes('fetch')) {
        error = 'Network error. Please check your connection.';
      } else if (errorMessage.includes('path') || errorMessage.includes('directory')) {
        error = 'Invalid directory path. Please verify the path exists.';
      } else {
        error = errorMessage;
      }

      retryCount++;
    } finally {
      loading = false;
    }
  }

  function handleRetry() {
    if (retryCount < MAX_RETRIES) {
      handleCreate();
    } else {
      error = 'Maximum retry attempts reached. Please contact support.';
    }
  }
</script>

{#if error}
  <div class="error-banner" role="alert">
    <span class="error-icon">⚠</span>
    <div class="error-content">
      <strong>Error:</strong> {error}

      {#if retryCount < MAX_RETRIES}
        <button type="button" class="retry-button" onclick={handleRetry}>
          Retry ({MAX_RETRIES - retryCount} attempts remaining)
        </button>
      {/if}
    </div>
  </div>
{/if}
```

---

## 6. Implementation Roadmap

### Phase 1: Critical Fixes (Day 1, 2-3 hours)

**Priority**: Address blocking bugs

1. **P0-1**: Fix CategorySelector refresh bug (2 hours)
   - Add `refresh()` method
   - Update CreateSessionModal to call refresh after category creation
   - Test end-to-end flow

**Deliverable**: Category dropdown updates immediately after creating new category

---

### Phase 2: User-Requested Features (Day 1-2, 8-9 hours)

**Priority**: Implement features explicitly requested by user

1. **P1-1 & P1-2**: localStorage for last used values (2 hours)
   - Add onMount logic to load saved values
   - Save values in handleCreate
   - Add error handling for localStorage unavailable

2. **P1-3**: Increase dialog width (0.5 hours)
   - Update CSS max-width to 900px
   - Test responsive behavior

3. **P1-4**: Add training status to DirectoryInfo API (4 hours)
   - Update Pydantic schemas
   - Add `include_training_status` query parameter
   - Implement `enrich_with_training_status` service method
   - Add database index for performance
   - Write backend tests
   - Regenerate frontend types

4. **P1-5**: Visual indicators for trained directories (3 hours)
   - Update DirectoryBrowser component
   - Add CSS for badges and background colors
   - Implement formatRelativeTime helper
   - Write frontend tests

5. **P1-6**: Checkbox to hide trained directories (2 hours)
   - Add hideTrainedDirs state
   - Update filtering logic
   - Add checkbox UI
   - Test filtering combinations

**Deliverable**: All user-requested features implemented and tested

---

### Phase 3: Usability Enhancements (Day 3, Optional, 13 hours)

**Priority**: Nice-to-have improvements for better UX

1. **P2-1**: Keyboard shortcuts (2 hours)
2. **P2-2**: Validation improvements (3 hours)
3. **P2-3**: Progress indicators (2 hours)
4. **P2-4**: Accessibility improvements (3 hours)
5. **P2-5**: Error recovery mechanisms (3 hours)

**Deliverable**: Enhanced user experience with accessibility and polish

---

### Phase 4: Documentation & Testing (Day 3, 3 hours)

1. Update API contract documentation
2. Write end-to-end tests
3. Update user-facing documentation
4. Create migration guide (if needed)

---

## 7. Risk Assessment

### Breaking Changes

**None identified** - All changes are additive or bug fixes.

### Migration Requirements

**Database Migration**:
```python
# Migration: add_training_subdirectories_path_index.py
# Estimated time: <1 second for <1M records

def upgrade():
    op.create_index(
        'idx_training_subdirectories_path',
        'training_subdirectories',
        ['path']
    )

def downgrade():
    op.drop_index('idx_training_subdirectories_path', 'training_subdirectories')
```

**API Contract Update**:
- Backend version: Bump from 1.0.0 to 1.1.0 (minor version)
- Frontend types: Regenerate with `npm run gen:api`
- Backward compatible: New fields are optional

### Performance Impact

**Low Impact**:
- localStorage operations: <1ms
- Additional DB query: ~10-50ms (with index)
- Frontend rendering: No significant change

**Mitigation**:
- Make `include_training_status` opt-in (default: false)
- Add database index on `training_subdirectories.path`
- Consider caching for frequently accessed directories

### Browser Compatibility

**localStorage**: Supported in all modern browsers (IE8+)
**CSS Features**: Standard flexbox and transitions (IE11+)
**JavaScript**: ES2020 features (needs transpilation for older browsers)

---

## 8. Testing Strategy

### Unit Tests

**Backend** (`pytest`):
- `test_directory_info_training_status()`: Verify API response includes training metadata
- `test_enrich_training_status()`: Test enrichment service logic
- `test_training_status_calculation()`: Verify never/partial/complete logic
- `test_directory_list_without_status()`: Ensure backward compatibility

**Frontend** (`vitest`):
- `test_category_selector_refresh()`: Verify refresh method works
- `test_localStorage_persistence()`: Test saving/loading last values
- `test_training_badge_display()`: Verify badges appear correctly
- `test_hide_trained_filter()`: Test filtering logic

### Integration Tests

1. **Full Session Creation Flow**:
   - Create category → Verify appears in dropdown immediately
   - Enter root path → Verify directories load with training status
   - Filter directories → Verify text and status filters work
   - Hide trained dirs → Verify only untrained appear
   - Create session → Verify localStorage saves values
   - Reopen dialog → Verify values restored

2. **Training Status Accuracy**:
   - Create training session and train subdirectories
   - Verify trained directories show correct status
   - Partially train directory → Verify shows partial status
   - Fully train directory → Verify shows complete status

### Manual QA Checklist

- [ ] CategorySelector refresh works after creating category
- [ ] Last root path persists across dialog opens
- [ ] Last category persists across dialog opens
- [ ] Dialog displays at 900px width on desktop
- [ ] Dialog is responsive on tablet/mobile
- [ ] Trained directories show badges with correct counts
- [ ] Background colors match training status
- [ ] Last trained time displays correctly
- [ ] Hide trained checkbox filters correctly
- [ ] Text filter + hide trained work together
- [ ] Select All operates on filtered results
- [ ] Keyboard shortcuts work (Esc, Enter)
- [ ] Validation errors display clearly
- [ ] Screen reader announces errors
- [ ] Focus management works correctly

---

## 9. Success Metrics

### Functional Requirements

✅ **P0-1**: Category dropdown refreshes after creating new category
✅ **P1-1**: Root path remembered in localStorage
✅ **P1-2**: Category remembered in localStorage
✅ **P1-3**: Dialog width increased to 900px
✅ **P1-4**: DirectoryInfo API includes training status metadata
✅ **P1-5**: Visual badges and colors indicate training status
✅ **P1-6**: Checkbox to hide fully trained directories

### Non-Functional Requirements

- **Performance**: Directory list loads in <500ms (with training status)
- **Accessibility**: WCAG 2.1 AA compliance (keyboard navigation, screen reader support)
- **Browser Support**: Chrome, Firefox, Safari, Edge (latest 2 versions)
- **Mobile Responsive**: Works on 768px and 1024px breakpoints
- **Error Handling**: Graceful degradation for localStorage unavailable

---

## 10. Open Questions for User

1. **Dialog Width**: Is 900px acceptable, or would you prefer a different size (800px, 1000px)? answer: 90% of screen width

2. **Visual Indicators**: Preference for Option A (badges + background) or Option B (progress bars)? answer: Option A

3. **Hide Trained Checkbox**: Should it hide "fully trained" only, or also "partially trained"? answer: hide "fully trained"

4. **Training Status Definition**:
   - **Never trained**: `trainedCount = 0`
   - **Partially trained**: `0 < trainedCount < imageCount`
   - **Fully trained**: `trainedCount >= imageCount`
   - Is this definition correct?  answer: yes, it is correct

5. **Phase 3 (P2) Features**: Should these be implemented now or deferred to future iteration? Answer: create a ticket for future implementation. include enough context to be able to achive that.

6. **Keyboard Shortcuts**: Are the proposed shortcuts (Esc, Enter, Ctrl+A) acceptable? answer: yes

7. **LocalStorage Scope**: Should saved values be global or per-user (if authentication is added later)? answer: per user

---

## 11. Appendix A: File Reference

### Frontend Files to Modify

| File Path | Changes | Estimated LOC |
|-----------|---------|---------------|
| `/export/workspace/image-search/image-search-ui/src/lib/components/CategorySelector.svelte` | Add refresh() method | +5 |
| `/export/workspace/image-search/image-search-ui/src/lib/components/training/CreateSessionModal.svelte` | localStorage, bind:this, increased width | +40 |
| `/export/workspace/image-search/image-search-ui/src/lib/components/training/DirectoryBrowser.svelte` | Visual indicators, hide checkbox, enhanced filtering | +100 |
| `/export/workspace/image-search/image-search-ui/src/lib/types.ts` | Update DirectoryInfo type (if not using generated) | +3 |

### Backend Files to Modify

| File Path | Changes | Estimated LOC |
|-----------|---------|---------------|
| `/export/workspace/image-search/image-search-service/src/image_search_service/api/training_schemas.py` | Add training status fields to DirectoryInfo | +10 |
| `/export/workspace/image-search/image-search-service/src/image_search_service/api/routes/training.py` | Add include_training_status query param | +5 |
| `/export/workspace/image-search/image-search-service/src/image_search_service/services/training_service.py` | Add enrich_with_training_status method | +50 |
| `/export/workspace/image-search/image-search-service/src/image_search_service/db/migrations/versions/XXX_add_path_index.py` | Add database index | +10 |

### Files to Generate/Update

| File Path | Action |
|-----------|--------|
| `/export/workspace/image-search/image-search-ui/src/lib/api/generated.ts` | Regenerate with `npm run gen:api` |
| `/export/workspace/image-search/docs/api-contract.md` | Update with new DirectoryInfo fields (both repos) |

---

## 12. Appendix B: Code Snippets

### LocalStorage Utility (Optional)

```typescript
// src/lib/utils/storage.ts

const STORAGE_PREFIX = 'image-search.';

export function getStorageItem<T>(key: string, defaultValue: T): T {
  try {
    const item = localStorage.getItem(STORAGE_PREFIX + key);
    return item ? JSON.parse(item) : defaultValue;
  } catch (err) {
    console.warn(`Failed to load ${key} from localStorage:`, err);
    return defaultValue;
  }
}

export function setStorageItem<T>(key: string, value: T): void {
  try {
    localStorage.setItem(STORAGE_PREFIX + key, JSON.stringify(value));
  } catch (err) {
    console.warn(`Failed to save ${key} to localStorage:`, err);
  }
}

export function removeStorageItem(key: string): void {
  try {
    localStorage.removeItem(STORAGE_PREFIX + key);
  } catch (err) {
    console.warn(`Failed to remove ${key} from localStorage:`, err);
  }
}

// Usage in CreateSessionModal.svelte
import { getStorageItem, setStorageItem } from '$lib/utils/storage';

onMount(() => {
  rootPath = getStorageItem('lastRootPath', '');
  categoryId = getStorageItem('lastCategoryId', null);
});

async function handleCreate() {
  setStorageItem('lastRootPath', rootPath);
  setStorageItem('lastCategoryId', categoryId);
  // ... create session
}
```

---

## 13. Conclusion

This document provides a comprehensive roadmap for improving the Create Training Session dialog. The proposed changes address all user-requested features while maintaining backward compatibility and code quality.

**Next Steps**:
1. User reviews and approves improvement plan
2. Clarifies open questions (dialog width, visual design preferences)
3. Prioritizes P2 features (implement now vs. defer)
4. Engineering team implements in phases
5. QA verifies all acceptance criteria
6. Deploy to production

**Estimated Timeline**:
- Phase 1 (P0): 2-3 hours
- Phase 2 (P1): 8-9 hours
- Phase 3 (P2): 13 hours (optional)
- Total: 1-3 days (1 developer)

---

**Document Version**: 1.0
**Last Updated**: 2025-12-30
**Author**: Documentation Agent
**Reviewed By**: Pending User Approval
