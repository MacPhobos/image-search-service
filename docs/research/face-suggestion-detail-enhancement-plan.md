# Face Suggestion Detail Enhancement Plan

**Date**: 2025-12-29
**Status**: Ready for Review
**Scope**: image-search-ui (frontend changes only)

---

## Executive Summary

Enhance the "Face Suggestion Details" dialog to display bounding boxes for **all faces** detected in the image (not just the suggested face), and add an "Assign" button workflow in a side panel allowing users to label additional faces. This mirrors the existing functionality in the "Photo Preview" dialog.

### Key Deliverables

1. Show bounding boxes for ALL faces in the suggestion detail view
2. Add side panel with "Assign" button workflow (exactly like PhotoPreviewModal)
3. Maintain existing Accept/Reject functionality for the primary suggestion
4. Comprehensive test coverage

---

## Current State Analysis

### SuggestionDetailModal (Current)

**File**: `src/lib/components/faces/SuggestionDetailModal.svelte`

**Current Behavior**:
- Displays full image with bounding box for **only the suggested face**
- Shows suggestion details (person name, confidence, status)
- Provides Accept/Reject buttons for pending suggestions
- Uses `ImageWithFaceBoundingBoxes` component with single face

**Limitation**: Cannot see or interact with other faces in the same image.

### PhotoPreviewModal (Reference)

**File**: `src/lib/components/faces/PhotoPreviewModal.svelte`

**Features to Replicate**:
- Displays ALL faces in the image
- Side panel listing all faces with metadata
- "Assign" button for unknown faces
- Assignment panel with person search + create new
- Suggestion hints with quick "Accept" button
- Proper state management for face suggestions loading

---

## Technical Approach

### Architecture Decision

**Option A**: Extend SuggestionDetailModal directly
**Option B**: Create new EnhancedSuggestionDetailModal component
**Option C**: Refactor shared code into composable utilities

**Recommendation**: **Option A** - Extend SuggestionDetailModal

**Rationale**:
- Maintains single component responsibility
- Avoids duplication
- PhotoPreviewModal patterns can be extracted as utilities
- Backward compatible (same props interface)

---

## Implementation Plan

### Phase 1: Data Loading Enhancement

#### 1.1 Add API Client Function (if not exists)

**File**: `src/lib/api/faces.ts`

```typescript
/**
 * Get all faces detected in an image
 * @param assetId - The image asset ID
 */
export async function getFacesForAsset(assetId: number): Promise<FaceInstanceListResponse> {
  const response = await fetch(`${API_BASE_URL}/api/v1/faces/assets/${assetId}`);
  if (!response.ok) throw new ApiError(response);
  return response.json();
}
```

**API Used**: `GET /api/v1/faces/assets/{asset_id}`

**Response Type**:
```typescript
interface FaceInstanceListResponse {
  items: FaceInstanceResponse[];
  total: number;
  page: number;
  pageSize: number;
}

interface FaceInstanceResponse {
  id: string;  // UUID
  assetId: number;
  bboxX: number;
  bboxY: number;
  bboxW: number;
  bboxH: number;
  detectionConfidence: number;
  qualityScore: number | null;
  clusterId: string | null;
  personId: string | null;
  personName: string | null;
  createdAt: string;
}
```

#### 1.2 Extract Asset ID from Suggestion

**Challenge**: `FaceSuggestion` contains `fullImageUrl` but not `assetId` directly.

**Solution**: Parse asset ID from URL pattern `/api/v1/images/{assetId}/full`

```typescript
function extractAssetIdFromUrl(fullImageUrl: string | null): number | null {
  if (!fullImageUrl) return null;
  const match = fullImageUrl.match(/\/images\/(\d+)\//);
  return match ? parseInt(match[1], 10) : null;
}
```

**Alternative**: Verify if `FaceSuggestion` type already has `assetId` field in generated types.

---

### Phase 2: Component State Enhancement

#### 2.1 Add New State Variables

**File**: `src/lib/components/faces/SuggestionDetailModal.svelte`

```typescript
// All faces in the image (including the suggested one)
let allFaces = $state<FaceInstanceResponse[]>([]);
let allFacesLoading = $state(false);
let allFacesError = $state<string | null>(null);

// Face assignment state (from PhotoPreviewModal pattern)
let assigningFaceId = $state<string | null>(null);
let personSearchQuery = $state('');
let persons = $state<Person[]>([]);
let personsLoading = $state(false);

// Face suggestions for each unknown face
interface FaceSuggestionsState {
  suggestions: FaceSuggestionItem[];
  loading: boolean;
  error: string | null;
}
let faceSuggestions = $state.raw<Map<string, FaceSuggestionsState>>(new Map());

// Highlighted face (for hover interactions)
let highlightedFaceId = $state<string | null>(null);
```

#### 2.2 Add Data Loading in onMount

```typescript
onMount(() => {
  const controller = new AbortController();

  // Load all faces for this image
  async function loadAllFaces() {
    const assetId = extractAssetIdFromUrl(suggestion?.fullImageUrl);
    if (!assetId) return;

    allFacesLoading = true;
    allFacesError = null;

    try {
      const response = await getFacesForAsset(assetId);
      allFaces = response.items;

      // Load suggestions for unknown faces
      const unknownFaces = allFaces.filter(f => !f.personId);
      await loadSuggestionsForFaces(unknownFaces, controller.signal);
    } catch (err) {
      if (err.name !== 'AbortError') {
        allFacesError = err.message;
      }
    } finally {
      allFacesLoading = false;
    }
  }

  // Load persons list for assignment dropdown
  async function loadPersons() {
    personsLoading = true;
    try {
      const response = await listPersons({ pageSize: 100 });
      persons = response.items;
    } catch (err) {
      console.error('Failed to load persons:', err);
    } finally {
      personsLoading = false;
    }
  }

  loadAllFaces();
  loadPersons();

  return () => controller.abort();
});
```

#### 2.3 Face Suggestions Loading (from PhotoPreviewModal)

```typescript
async function loadSuggestionsForFaces(faces: FaceInstanceResponse[], signal: AbortSignal) {
  for (const face of faces) {
    updateFaceSuggestion(face.id, {
      suggestions: [],
      loading: true,
      error: null
    });

    try {
      const response = await getFaceSuggestions(face.id, {
        minConfidence: 0.7,
        limit: 5,
        signal
      });
      updateFaceSuggestion(face.id, {
        suggestions: response.suggestions,
        loading: false,
        error: null
      });
    } catch (err) {
      if (err.name !== 'AbortError') {
        updateFaceSuggestion(face.id, {
          suggestions: [],
          loading: false,
          error: err.message
        });
      }
    }
  }
}

function updateFaceSuggestion(faceId: string, state: FaceSuggestionsState) {
  const newMap = new Map(faceSuggestions);
  newMap.set(faceId, state);
  faceSuggestions = newMap;  // Trigger reactivity
}
```

---

### Phase 3: UI Layout Enhancement

#### 3.1 Update Modal Layout (Two-Column)

**Current Layout**:
```
┌─────────────────────────────────┐
│ Header                          │
├─────────────────────────────────┤
│                                 │
│    Image with single bbox       │
│                                 │
├─────────────────────────────────┤
│ Suggestion Details              │
├─────────────────────────────────┤
│ Accept | Reject                 │
└─────────────────────────────────┘
```

**New Layout**:
```
┌─────────────────────────────────────────────────┐
│ Header                                          │
├─────────────────────────────────┬───────────────┤
│                                 │ Faces Panel   │
│    Image with ALL bboxes        │               │
│    (primary face highlighted)   │ • Face 1 ✓    │
│                                 │ • Face 2 [?]  │
│                                 │   [Assign]    │
│                                 │   💡 Suggested│
│                                 │ • Face 3 [?]  │
│                                 │   [Assign]    │
├─────────────────────────────────┴───────────────┤
│ Primary Suggestion Details                      │
├─────────────────────────────────────────────────┤
│ Accept | Reject (for primary suggestion)        │
└─────────────────────────────────────────────────┘
```

#### 3.2 Updated Template Structure

```svelte
{#if suggestion}
  <div class="modal-backdrop" onclick={handleBackdropClick}>
    <div class="modal enhanced-modal">
      <header class="modal-header">
        <h2>Face Suggestion Details</h2>
        <button class="close-btn" onclick={onClose}>×</button>
      </header>

      <div class="modal-body two-column">
        <!-- Left: Image with all faces -->
        <div class="image-section">
          <ImageWithFaceBoundingBoxes
            imageUrl={fullImageUrl}
            faces={allFaceBoxes}
            primaryFaceId={suggestion.faceInstanceId}
            highlightedFaceId={highlightedFaceId}
            onFaceClick={handleFaceClick}
            maxHeight="70vh"
          />
        </div>

        <!-- Right: Faces sidebar -->
        <aside class="faces-sidebar">
          <h3>Faces ({allFaces.length})</h3>

          {#if allFacesLoading}
            <div class="loading">Loading faces...</div>
          {:else if allFacesError}
            <div class="error">{allFacesError}</div>
          {:else}
            <ul class="face-list">
              {#each allFaces as face, index}
                <li
                  class="face-item"
                  class:primary={face.id === suggestion.faceInstanceId}
                  class:highlighted={face.id === highlightedFaceId}
                  onmouseenter={() => highlightedFaceId = face.id}
                  onmouseleave={() => highlightedFaceId = null}
                >
                  <!-- Face indicator + info -->
                  <div class="face-info">
                    <span
                      class="face-color-indicator"
                      style="background-color: {getFaceColor(index)}"
                    ></span>
                    <div class="face-details">
                      <span class="face-name">
                        {#if face.id === suggestion.faceInstanceId}
                          <strong>{suggestion.personName || 'Suggested'}</strong>
                          <span class="primary-badge">Primary</span>
                        {:else if face.personName}
                          {face.personName}
                        {:else}
                          Unknown
                        {/if}
                      </span>
                      <span class="face-meta">
                        Conf: {Math.round(face.detectionConfidence * 100)}%
                        {#if face.qualityScore}
                          | Q: {face.qualityScore.toFixed(1)}
                        {/if}
                      </span>
                    </div>
                  </div>

                  <!-- Assignment controls (for non-primary unknown faces) -->
                  {#if face.id !== suggestion.faceInstanceId && !face.personId}
                    {#if assigningFaceId !== face.id}
                      <button
                        class="assign-btn"
                        onclick={() => startAssignment(face.id)}
                      >
                        Assign
                      </button>

                      <!-- Suggestion hint -->
                      {@const suggestionState = faceSuggestions.get(face.id)}
                      {@const topSuggestion = suggestionState?.suggestions?.[0]}
                      {#if topSuggestion}
                        <div class="suggestion-hint">
                          <span>
                            💡 {topSuggestion.personName}
                            ({Math.round(topSuggestion.confidence * 100)}%)
                          </span>
                          <button
                            class="accept-suggestion-btn"
                            onclick={() => handleAssignFace(
                              face.id,
                              topSuggestion.personId,
                              topSuggestion.personName
                            )}
                          >
                            ✓
                          </button>
                        </div>
                      {:else if suggestionState?.loading}
                        <div class="suggestion-loading">Loading...</div>
                      {/if}
                    {:else}
                      <!-- Assignment panel -->
                      <div class="assignment-panel">
                        <div class="assignment-header">
                          <span>Assign Face</span>
                          <button onclick={cancelAssignment}>×</button>
                        </div>
                        <input
                          type="text"
                          placeholder="Search or create..."
                          bind:value={personSearchQuery}
                          class="person-search-input"
                        />
                        <div class="person-options">
                          {#if showCreateOption}
                            <button
                              class="person-option create-new"
                              onclick={handleCreateAndAssign}
                            >
                              <span class="create-icon">+</span>
                              Create "{personSearchQuery.trim()}"
                            </button>
                          {/if}
                          {#each filteredPersons as person}
                            <button
                              class="person-option"
                              onclick={() => handleAssignToExisting(person)}
                            >
                              <span class="person-avatar">
                                {person.name.charAt(0).toUpperCase()}
                              </span>
                              <span>{person.name}</span>
                            </button>
                          {/each}
                        </div>
                      </div>
                    {/if}
                  {/if}

                  <!-- Already assigned indicator (for non-primary assigned faces) -->
                  {#if face.id !== suggestion.faceInstanceId && face.personName}
                    <span class="assigned-indicator">✓ {face.personName}</span>
                  {/if}
                </li>
              {/each}
            </ul>
          {/if}
        </aside>
      </div>

      <!-- Primary suggestion details (existing functionality) -->
      <div class="suggestion-details">
        <div class="detail-row">
          <span>Suggested Person:</span>
          <span>{suggestion.personName || 'Unknown'}</span>
        </div>
        <div class="detail-row">
          <span>Confidence:</span>
          <span class="confidence" style="color: {confidenceColor}">
            {confidencePercent}%
          </span>
        </div>
        <div class="detail-row">
          <span>Status:</span>
          <span class="status-badge {suggestion.status}">{suggestion.status}</span>
        </div>
      </div>

      <!-- Footer with primary suggestion actions -->
      {#if suggestion.status === 'pending'}
        <footer class="modal-footer">
          <button class="btn-accept" onclick={handleAccept}>
            ✓ Accept Suggestion
          </button>
          <button class="btn-reject" onclick={handleReject}>
            ✗ Reject Suggestion
          </button>
        </footer>
      {/if}
    </div>
  </div>
{/if}
```

---

### Phase 4: Event Handlers

#### 4.1 Assignment Handlers

```typescript
function startAssignment(faceId: string) {
  assigningFaceId = faceId;
  personSearchQuery = '';
}

function cancelAssignment() {
  assigningFaceId = null;
  personSearchQuery = '';
}

async function handleAssignFace(
  faceId: string,
  personId: string,
  personName: string
) {
  try {
    await assignFaceToPerson(faceId, personId);

    // Update local state
    const faceIndex = allFaces.findIndex(f => f.id === faceId);
    if (faceIndex !== -1) {
      allFaces[faceIndex] = {
        ...allFaces[faceIndex],
        personId,
        personName
      };
      allFaces = [...allFaces];  // Trigger reactivity
    }

    // Clear assignment UI
    cancelAssignment();

    // Notify parent (optional callback)
    onFaceAssigned?.(faceId, personId, personName);
  } catch (err) {
    console.error('Failed to assign face:', err);
    // Show error toast/notification
  }
}

async function handleCreateAndAssign() {
  const name = personSearchQuery.trim();
  if (!name || !assigningFaceId) return;

  try {
    const newPerson = await createPerson(name);
    await handleAssignFace(assigningFaceId, newPerson.id, newPerson.name);

    // Add to persons list
    persons = [...persons, newPerson];
  } catch (err) {
    console.error('Failed to create person:', err);
  }
}

function handleAssignToExisting(person: Person) {
  if (!assigningFaceId) return;
  handleAssignFace(assigningFaceId, person.id, person.name);
}
```

#### 4.2 Derived State

```typescript
// Filter persons by search query
const filteredPersons = $derived(
  persons.filter(p =>
    p.name.toLowerCase().includes(personSearchQuery.toLowerCase())
  ).slice(0, 10)
);

// Show create option when search has text and no exact match
const showCreateOption = $derived(
  personSearchQuery.trim().length > 0 &&
  !persons.some(p =>
    p.name.toLowerCase() === personSearchQuery.trim().toLowerCase()
  )
);

// Transform all faces to FaceBox format for ImageWithFaceBoundingBoxes
const allFaceBoxes = $derived<FaceBox[]>(
  allFaces.map((face, index) => {
    const suggestionState = faceSuggestions.get(face.id);
    const topSuggestion = suggestionState?.suggestions?.[0];
    const isPrimary = face.id === suggestion?.faceInstanceId;

    let labelStyle: FaceBox['labelStyle'];
    let label: string;
    let suggestionConfidence: number | undefined;

    if (isPrimary) {
      // Primary suggestion face
      labelStyle = 'suggested';
      label = suggestion?.personName || 'Suggested';
      suggestionConfidence = suggestion?.confidence;
    } else if (face.personName) {
      labelStyle = 'assigned';
      label = face.personName;
    } else if (suggestionState?.loading) {
      labelStyle = 'loading';
      label = 'Loading...';
    } else if (topSuggestion) {
      labelStyle = 'suggested';
      label = `Suggested: ${topSuggestion.personName}`;
      suggestionConfidence = topSuggestion.confidence;
    } else {
      labelStyle = 'unknown';
      label = 'Unknown';
    }

    return {
      id: face.id,
      bboxX: face.bboxX,
      bboxY: face.bboxY,
      bboxW: face.bboxW,
      bboxH: face.bboxH,
      label,
      labelStyle,
      color: getFaceColor(index),
      suggestionConfidence
    };
  })
);
```

---

### Phase 5: Styling

#### 5.1 New CSS Classes

```css
.enhanced-modal {
  max-width: 1200px;
  width: 95vw;
}

.modal-body.two-column {
  display: grid;
  grid-template-columns: 1fr 320px;
  gap: 1rem;
  padding: 1rem;
  max-height: 70vh;
  overflow: hidden;
}

.image-section {
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
}

.faces-sidebar {
  overflow-y: auto;
  border-left: 1px solid #e0e0e0;
  padding-left: 1rem;
}

.faces-sidebar h3 {
  margin: 0 0 0.75rem 0;
  font-size: 1rem;
  font-weight: 600;
  color: #374151;
}

.face-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

.face-item {
  padding: 0.75rem;
  border-radius: 8px;
  margin-bottom: 0.5rem;
  background-color: #f9fafb;
  transition: background-color 0.2s;
}

.face-item:hover,
.face-item.highlighted {
  background-color: #f3f4f6;
}

.face-item.primary {
  background-color: #fef3c7;
  border: 2px solid #f59e0b;
}

.face-info {
  display: flex;
  align-items: flex-start;
  gap: 0.5rem;
}

.face-color-indicator {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  flex-shrink: 0;
  margin-top: 4px;
}

.face-details {
  flex: 1;
}

.face-name {
  display: block;
  font-weight: 500;
  color: #111827;
}

.face-meta {
  display: block;
  font-size: 0.75rem;
  color: #6b7280;
  margin-top: 2px;
}

.primary-badge {
  display: inline-block;
  background-color: #f59e0b;
  color: white;
  font-size: 0.625rem;
  padding: 1px 6px;
  border-radius: 4px;
  margin-left: 0.5rem;
  font-weight: 600;
  text-transform: uppercase;
}

.assign-btn {
  display: block;
  width: 100%;
  margin-top: 0.5rem;
  padding: 0.375rem 0.75rem;
  background-color: #3b82f6;
  color: white;
  border: none;
  border-radius: 6px;
  font-size: 0.875rem;
  font-weight: 500;
  cursor: pointer;
  transition: background-color 0.2s;
}

.assign-btn:hover {
  background-color: #2563eb;
}

.suggestion-hint {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 0.5rem;
  padding: 0.5rem;
  background-color: #fffbeb;
  border: 1px solid #fef3c7;
  border-radius: 6px;
  font-size: 0.75rem;
  color: #92400e;
}

.accept-suggestion-btn {
  padding: 0.25rem 0.5rem;
  background-color: #22c55e;
  color: white;
  border: none;
  border-radius: 4px;
  font-size: 0.75rem;
  font-weight: 600;
  cursor: pointer;
}

.assignment-panel {
  margin-top: 0.5rem;
  padding: 0.75rem;
  background-color: white;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
}

.assignment-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
  font-weight: 500;
}

.person-search-input {
  width: 100%;
  padding: 0.5rem;
  border: 1px solid #d1d5db;
  border-radius: 6px;
  font-size: 0.875rem;
  margin-bottom: 0.5rem;
}

.person-options {
  max-height: 200px;
  overflow-y: auto;
}

.person-option {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  width: 100%;
  padding: 0.5rem;
  border: none;
  background: none;
  cursor: pointer;
  border-radius: 6px;
  transition: background-color 0.15s;
}

.person-option:hover {
  background-color: #f3f4f6;
}

.person-option.create-new {
  color: #3b82f6;
  font-weight: 500;
}

.person-avatar {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  background-color: #e5e7eb;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.75rem;
  font-weight: 600;
}

.create-icon {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  background-color: #dbeafe;
  color: #3b82f6;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.25rem;
  font-weight: 300;
}

.assigned-indicator {
  display: block;
  margin-top: 0.5rem;
  font-size: 0.75rem;
  color: #22c55e;
  font-weight: 500;
}

.suggestion-loading {
  margin-top: 0.5rem;
  font-size: 0.75rem;
  color: #6b7280;
  font-style: italic;
}

/* Responsive adjustments */
@media (max-width: 900px) {
  .modal-body.two-column {
    grid-template-columns: 1fr;
    grid-template-rows: auto auto;
  }

  .faces-sidebar {
    border-left: none;
    border-top: 1px solid #e0e0e0;
    padding-left: 0;
    padding-top: 1rem;
    max-height: 200px;
  }
}
```

---

### Phase 6: Props Enhancement

#### 6.1 Updated Props Interface

```typescript
interface Props {
  suggestion: FaceSuggestion | null;
  onClose: () => void;
  onAccept: (suggestion: FaceSuggestion) => void;
  onReject: (suggestion: FaceSuggestion) => void;
  // NEW: Optional callback when additional face is assigned
  onFaceAssigned?: (faceId: string, personId: string, personName: string) => void;
}
```

---

## Testing Plan

### Unit Tests

#### Test File: `src/tests/components/SuggestionDetailModal.test.ts`

```typescript
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import SuggestionDetailModal from '$lib/components/faces/SuggestionDetailModal.svelte';
import * as facesApi from '$lib/api/faces';

describe('SuggestionDetailModal', () => {
  const mockSuggestion = {
    id: 1,
    faceInstanceId: 'face-uuid-1',
    suggestedPersonId: 'person-uuid-1',
    confidence: 0.92,
    status: 'pending',
    personName: 'John Doe',
    fullImageUrl: '/api/v1/images/123/full',
    bboxX: 100,
    bboxY: 100,
    bboxW: 200,
    bboxH: 200,
    detectionConfidence: 0.98,
    qualityScore: 0.85
  };

  const mockAllFaces = [
    {
      id: 'face-uuid-1',  // Primary suggestion face
      assetId: 123,
      bboxX: 100, bboxY: 100, bboxW: 200, bboxH: 200,
      detectionConfidence: 0.98,
      qualityScore: 0.85,
      personId: null,
      personName: null
    },
    {
      id: 'face-uuid-2',  // Another face (unknown)
      assetId: 123,
      bboxX: 400, bboxY: 150, bboxW: 180, bboxH: 180,
      detectionConfidence: 0.95,
      qualityScore: 0.72,
      personId: null,
      personName: null
    },
    {
      id: 'face-uuid-3',  // Already assigned face
      assetId: 123,
      bboxX: 700, bboxY: 200, bboxW: 150, bboxH: 150,
      detectionConfidence: 0.88,
      qualityScore: 0.65,
      personId: 'person-uuid-2',
      personName: 'Jane Smith'
    }
  ];

  beforeEach(() => {
    vi.resetAllMocks();
    vi.spyOn(facesApi, 'getFacesForAsset').mockResolvedValue({
      items: mockAllFaces,
      total: 3,
      page: 1,
      pageSize: 20
    });
    vi.spyOn(facesApi, 'listPersons').mockResolvedValue({
      items: [
        { id: 'person-uuid-1', name: 'John Doe', status: 'active' },
        { id: 'person-uuid-2', name: 'Jane Smith', status: 'active' }
      ],
      total: 2,
      page: 1,
      pageSize: 100
    });
    vi.spyOn(facesApi, 'getFaceSuggestions').mockResolvedValue({
      faceId: 'face-uuid-2',
      suggestions: [
        { personId: 'person-uuid-3', personName: 'Bob Wilson', confidence: 0.78 }
      ],
      thresholdUsed: 0.7
    });
  });

  describe('Rendering', () => {
    it('should render modal with suggestion data', async () => {
      render(SuggestionDetailModal, {
        suggestion: mockSuggestion,
        onClose: vi.fn(),
        onAccept: vi.fn(),
        onReject: vi.fn()
      });

      expect(screen.getByText('Face Suggestion Details')).toBeInTheDocument();
      expect(screen.getByText('John Doe')).toBeInTheDocument();
      expect(screen.getByText('92%')).toBeInTheDocument();
    });

    it('should not render when suggestion is null', () => {
      const { container } = render(SuggestionDetailModal, {
        suggestion: null,
        onClose: vi.fn(),
        onAccept: vi.fn(),
        onReject: vi.fn()
      });

      expect(container.querySelector('.modal')).not.toBeInTheDocument();
    });

    it('should load and display all faces in sidebar', async () => {
      render(SuggestionDetailModal, {
        suggestion: mockSuggestion,
        onClose: vi.fn(),
        onAccept: vi.fn(),
        onReject: vi.fn()
      });

      await waitFor(() => {
        expect(screen.getByText('Faces (3)')).toBeInTheDocument();
      });

      expect(facesApi.getFacesForAsset).toHaveBeenCalledWith(123);
    });

    it('should mark primary face with badge', async () => {
      render(SuggestionDetailModal, {
        suggestion: mockSuggestion,
        onClose: vi.fn(),
        onAccept: vi.fn(),
        onReject: vi.fn()
      });

      await waitFor(() => {
        expect(screen.getByText('Primary')).toBeInTheDocument();
      });
    });
  });

  describe('All Faces Display', () => {
    it('should show Assign button for unknown non-primary faces', async () => {
      render(SuggestionDetailModal, {
        suggestion: mockSuggestion,
        onClose: vi.fn(),
        onAccept: vi.fn(),
        onReject: vi.fn()
      });

      await waitFor(() => {
        // Should have one Assign button (for face-uuid-2)
        const assignButtons = screen.getAllByText('Assign');
        expect(assignButtons).toHaveLength(1);
      });
    });

    it('should show assigned indicator for already assigned faces', async () => {
      render(SuggestionDetailModal, {
        suggestion: mockSuggestion,
        onClose: vi.fn(),
        onAccept: vi.fn(),
        onReject: vi.fn()
      });

      await waitFor(() => {
        expect(screen.getByText('✓ Jane Smith')).toBeInTheDocument();
      });
    });

    it('should NOT show Assign button for primary suggestion face', async () => {
      render(SuggestionDetailModal, {
        suggestion: mockSuggestion,
        onClose: vi.fn(),
        onAccept: vi.fn(),
        onReject: vi.fn()
      });

      await waitFor(() => {
        const primaryFaceItem = screen.getByText('Primary').closest('.face-item');
        expect(primaryFaceItem?.querySelector('.assign-btn')).not.toBeInTheDocument();
      });
    });
  });

  describe('Face Assignment', () => {
    it('should show assignment panel when Assign clicked', async () => {
      render(SuggestionDetailModal, {
        suggestion: mockSuggestion,
        onClose: vi.fn(),
        onAccept: vi.fn(),
        onReject: vi.fn()
      });

      await waitFor(() => {
        expect(screen.getByText('Assign')).toBeInTheDocument();
      });

      await fireEvent.click(screen.getByText('Assign'));

      expect(screen.getByPlaceholderText('Search or create...')).toBeInTheDocument();
    });

    it('should filter persons by search query', async () => {
      render(SuggestionDetailModal, {
        suggestion: mockSuggestion,
        onClose: vi.fn(),
        onAccept: vi.fn(),
        onReject: vi.fn()
      });

      await waitFor(() => screen.getByText('Assign'));
      await fireEvent.click(screen.getByText('Assign'));

      const searchInput = screen.getByPlaceholderText('Search or create...');
      await fireEvent.input(searchInput, { target: { value: 'John' } });

      expect(screen.getByText('John Doe')).toBeInTheDocument();
      expect(screen.queryByText('Jane Smith')).not.toBeInTheDocument();
    });

    it('should show create option for non-matching search', async () => {
      render(SuggestionDetailModal, {
        suggestion: mockSuggestion,
        onClose: vi.fn(),
        onAccept: vi.fn(),
        onReject: vi.fn()
      });

      await waitFor(() => screen.getByText('Assign'));
      await fireEvent.click(screen.getByText('Assign'));

      const searchInput = screen.getByPlaceholderText('Search or create...');
      await fireEvent.input(searchInput, { target: { value: 'New Person' } });

      expect(screen.getByText('Create "New Person"')).toBeInTheDocument();
    });

    it('should call assignFaceToPerson when person selected', async () => {
      vi.spyOn(facesApi, 'assignFaceToPerson').mockResolvedValue({
        faceId: 'face-uuid-2',
        personId: 'person-uuid-1',
        personName: 'John Doe'
      });

      render(SuggestionDetailModal, {
        suggestion: mockSuggestion,
        onClose: vi.fn(),
        onAccept: vi.fn(),
        onReject: vi.fn()
      });

      await waitFor(() => screen.getByText('Assign'));
      await fireEvent.click(screen.getByText('Assign'));
      await fireEvent.click(screen.getByText('John Doe'));

      await waitFor(() => {
        expect(facesApi.assignFaceToPerson).toHaveBeenCalledWith(
          'face-uuid-2',
          'person-uuid-1'
        );
      });
    });

    it('should call onFaceAssigned callback after assignment', async () => {
      vi.spyOn(facesApi, 'assignFaceToPerson').mockResolvedValue({
        faceId: 'face-uuid-2',
        personId: 'person-uuid-1',
        personName: 'John Doe'
      });

      const onFaceAssigned = vi.fn();

      render(SuggestionDetailModal, {
        suggestion: mockSuggestion,
        onClose: vi.fn(),
        onAccept: vi.fn(),
        onReject: vi.fn(),
        onFaceAssigned
      });

      await waitFor(() => screen.getByText('Assign'));
      await fireEvent.click(screen.getByText('Assign'));
      await fireEvent.click(screen.getByText('John Doe'));

      await waitFor(() => {
        expect(onFaceAssigned).toHaveBeenCalledWith(
          'face-uuid-2',
          'person-uuid-1',
          'John Doe'
        );
      });
    });
  });

  describe('Suggestion Hints', () => {
    it('should show suggestion hint for unknown faces', async () => {
      render(SuggestionDetailModal, {
        suggestion: mockSuggestion,
        onClose: vi.fn(),
        onAccept: vi.fn(),
        onReject: vi.fn()
      });

      await waitFor(() => {
        expect(screen.getByText(/Bob Wilson/)).toBeInTheDocument();
        expect(screen.getByText(/78%/)).toBeInTheDocument();
      });
    });

    it('should quick-assign when suggestion hint accepted', async () => {
      vi.spyOn(facesApi, 'assignFaceToPerson').mockResolvedValue({
        faceId: 'face-uuid-2',
        personId: 'person-uuid-3',
        personName: 'Bob Wilson'
      });

      render(SuggestionDetailModal, {
        suggestion: mockSuggestion,
        onClose: vi.fn(),
        onAccept: vi.fn(),
        onReject: vi.fn()
      });

      await waitFor(() => {
        expect(screen.getByText(/Bob Wilson/)).toBeInTheDocument();
      });

      // Click the accept button on suggestion hint
      const acceptBtn = screen.getByRole('button', { name: '✓' });
      await fireEvent.click(acceptBtn);

      await waitFor(() => {
        expect(facesApi.assignFaceToPerson).toHaveBeenCalledWith(
          'face-uuid-2',
          'person-uuid-3'
        );
      });
    });
  });

  describe('Primary Suggestion Actions', () => {
    it('should show Accept/Reject buttons for pending suggestions', () => {
      render(SuggestionDetailModal, {
        suggestion: mockSuggestion,
        onClose: vi.fn(),
        onAccept: vi.fn(),
        onReject: vi.fn()
      });

      expect(screen.getByText('✓ Accept Suggestion')).toBeInTheDocument();
      expect(screen.getByText('✗ Reject Suggestion')).toBeInTheDocument();
    });

    it('should call onAccept when Accept clicked', async () => {
      const onAccept = vi.fn();

      render(SuggestionDetailModal, {
        suggestion: mockSuggestion,
        onClose: vi.fn(),
        onAccept,
        onReject: vi.fn()
      });

      await fireEvent.click(screen.getByText('✓ Accept Suggestion'));

      expect(onAccept).toHaveBeenCalledWith(mockSuggestion);
    });

    it('should call onReject when Reject clicked', async () => {
      const onReject = vi.fn();

      render(SuggestionDetailModal, {
        suggestion: mockSuggestion,
        onClose: vi.fn(),
        onAccept: vi.fn(),
        onReject
      });

      await fireEvent.click(screen.getByText('✗ Reject Suggestion'));

      expect(onReject).toHaveBeenCalledWith(mockSuggestion);
    });

    it('should hide Accept/Reject for non-pending suggestions', () => {
      render(SuggestionDetailModal, {
        suggestion: { ...mockSuggestion, status: 'accepted' },
        onClose: vi.fn(),
        onAccept: vi.fn(),
        onReject: vi.fn()
      });

      expect(screen.queryByText('✓ Accept Suggestion')).not.toBeInTheDocument();
      expect(screen.queryByText('✗ Reject Suggestion')).not.toBeInTheDocument();
    });
  });

  describe('Modal Interactions', () => {
    it('should close on ESC key', async () => {
      const onClose = vi.fn();

      render(SuggestionDetailModal, {
        suggestion: mockSuggestion,
        onClose,
        onAccept: vi.fn(),
        onReject: vi.fn()
      });

      await fireEvent.keyDown(window, { key: 'Escape' });

      expect(onClose).toHaveBeenCalled();
    });

    it('should close on backdrop click', async () => {
      const onClose = vi.fn();

      render(SuggestionDetailModal, {
        suggestion: mockSuggestion,
        onClose,
        onAccept: vi.fn(),
        onReject: vi.fn()
      });

      const backdrop = screen.getByRole('dialog').parentElement;
      await fireEvent.click(backdrop!);

      expect(onClose).toHaveBeenCalled();
    });

    it('should highlight face on sidebar hover', async () => {
      render(SuggestionDetailModal, {
        suggestion: mockSuggestion,
        onClose: vi.fn(),
        onAccept: vi.fn(),
        onReject: vi.fn()
      });

      await waitFor(() => {
        expect(screen.getByText('Faces (3)')).toBeInTheDocument();
      });

      const faceItems = document.querySelectorAll('.face-item');
      expect(faceItems.length).toBe(3);

      await fireEvent.mouseEnter(faceItems[1]);
      expect(faceItems[1]).toHaveClass('highlighted');
    });
  });

  describe('Error Handling', () => {
    it('should show error when faces fail to load', async () => {
      vi.spyOn(facesApi, 'getFacesForAsset').mockRejectedValue(
        new Error('Network error')
      );

      render(SuggestionDetailModal, {
        suggestion: mockSuggestion,
        onClose: vi.fn(),
        onAccept: vi.fn(),
        onReject: vi.fn()
      });

      await waitFor(() => {
        expect(screen.getByText('Network error')).toBeInTheDocument();
      });
    });

    it('should handle missing fullImageUrl gracefully', async () => {
      render(SuggestionDetailModal, {
        suggestion: { ...mockSuggestion, fullImageUrl: null },
        onClose: vi.fn(),
        onAccept: vi.fn(),
        onReject: vi.fn()
      });

      // Should render without crashing, but not load faces
      expect(facesApi.getFacesForAsset).not.toHaveBeenCalled();
    });
  });

  describe('Accessibility', () => {
    it('should have proper ARIA attributes', () => {
      render(SuggestionDetailModal, {
        suggestion: mockSuggestion,
        onClose: vi.fn(),
        onAccept: vi.fn(),
        onReject: vi.fn()
      });

      const modal = screen.getByRole('dialog');
      expect(modal).toHaveAttribute('aria-modal', 'true');
    });

    it('should focus trap within modal', async () => {
      render(SuggestionDetailModal, {
        suggestion: mockSuggestion,
        onClose: vi.fn(),
        onAccept: vi.fn(),
        onReject: vi.fn()
      });

      const closeButton = screen.getByRole('button', { name: '×' });
      closeButton.focus();
      expect(document.activeElement).toBe(closeButton);
    });
  });
});
```

### Integration Tests

#### Test File: `src/tests/integration/face-suggestion-workflow.test.ts`

```typescript
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import FaceSuggestionsPage from '../../routes/faces/suggestions/+page.svelte';
import * as facesApi from '$lib/api/faces';

describe('Face Suggestion Workflow Integration', () => {
  // ... integration tests for full workflow

  it('should open detail modal and allow assigning additional faces', async () => {
    // Setup mocks
    // Render suggestions page
    // Click thumbnail to open modal
    // Verify all faces shown
    // Click Assign on secondary face
    // Select person
    // Verify assignment
    // Verify modal state updated
  });
});
```

---

## Implementation Sequence

### Step 1: API Client (if needed)
- [ ] Verify `getFacesForAsset` exists in `src/lib/api/faces.ts`
- [ ] Add if missing, following existing patterns

### Step 2: Component State
- [ ] Add new state variables to SuggestionDetailModal
- [ ] Add `onMount` data loading logic
- [ ] Add face suggestions loading for unknown faces

### Step 3: UI Layout
- [ ] Update template to two-column layout
- [ ] Add faces sidebar structure
- [ ] Add responsive CSS

### Step 4: Event Handlers
- [ ] Add assignment workflow handlers
- [ ] Add derived state computations
- [ ] Add keyboard navigation (ESC)

### Step 5: Styling
- [ ] Add new CSS classes
- [ ] Test responsive behavior
- [ ] Match existing design patterns

### Step 6: Props Update
- [ ] Add optional `onFaceAssigned` callback
- [ ] Update TypeScript interface

### Step 7: Testing
- [ ] Add unit tests for all new functionality
- [ ] Add integration tests for workflow
- [ ] Test edge cases (no faces, API errors, etc.)

### Step 8: Parent Integration
- [ ] Update suggestions page to handle `onFaceAssigned` callback
- [ ] Consider refreshing data after face assignment

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| API response format mismatch | High | Verify `getFacesForAsset` response matches expected type |
| Performance with many faces | Medium | Limit face suggestions loading to first 10 unknown faces |
| CSS conflicts | Low | Use scoped styles with specific class prefixes |
| State sync issues | Medium | Use reactive Map pattern from PhotoPreviewModal |

---

## Acceptance Criteria

1. ✅ Modal displays bounding boxes for ALL faces in the image
2. ✅ Primary suggestion face is visually distinguished (highlighted, badge)
3. ✅ Side panel shows list of all faces with metadata
4. ✅ Unknown non-primary faces have "Assign" button
5. ✅ Assignment panel with person search and create new
6. ✅ Suggestion hints shown for unknown faces
7. ✅ Quick accept button on suggestion hints
8. ✅ Existing Accept/Reject functionality preserved
9. ✅ Local state updates after assignment (no full refresh needed)
10. ✅ Optional callback `onFaceAssigned` notifies parent
11. ✅ Responsive layout for smaller screens
12. ✅ Keyboard navigation (ESC to close)
13. ✅ All tests passing
14. ✅ Accessibility compliance (ARIA attributes)

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/lib/api/faces.ts` | Verify/add `getFacesForAsset` function |
| `src/lib/components/faces/SuggestionDetailModal.svelte` | Major enhancement (state, UI, handlers) |
| `src/tests/components/SuggestionDetailModal.test.ts` | New test file |
| `src/routes/faces/suggestions/+page.svelte` | Add `onFaceAssigned` handler (optional) |

---

## Estimated Effort

| Phase | Effort |
|-------|--------|
| API Client | 0.5 hours |
| Component State | 1.5 hours |
| UI Layout | 2 hours |
| Event Handlers | 1.5 hours |
| Styling | 1 hour |
| Testing | 2.5 hours |
| Integration | 1 hour |
| **Total** | **10 hours** |

---

**Document Status**: Ready for Review
**Next Step**: Approval from user before implementation begins
