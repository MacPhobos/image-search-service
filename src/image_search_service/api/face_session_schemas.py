"""Pydantic schemas for face detection session APIs."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, computed_field


def to_camel(string: str) -> str:
    """Convert snake_case to camelCase."""
    words = string.split("_")
    return words[0] + "".join(word.capitalize() for word in words[1:])


class CamelCaseModel(BaseModel):
    """Base model with camelCase aliases for JSON serialization."""

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
    )


class FaceDetectionSessionStatus(str):
    """Face detection session status enum matching model."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class CreateFaceDetectionSessionRequest(CamelCaseModel):
    """Request to create a new face detection session."""

    training_session_id: int | None = None
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    min_face_size: int = Field(default=20, ge=10)
    batch_size: int = Field(default=16, ge=1, le=100)


class FaceDetectionSessionResponse(CamelCaseModel):
    """Response schema for a face detection session."""

    id: str  # UUID as string
    training_session_id: int | None
    status: str
    total_images: int
    processed_images: int
    failed_images: int
    faces_detected: int
    faces_assigned: int
    min_confidence: float
    min_face_size: int
    batch_size: int
    last_error: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    job_id: str | None

    # Detailed assignment breakdown
    faces_assigned_to_persons: int = 0
    clusters_created: int = 0
    suggestions_created: int = 0

    # Batch progress tracking
    current_batch: int = 0
    total_batches: int = 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        if self.total_images == 0:
            return 0.0
        return (self.processed_images / self.total_images) * 100.0


class FaceDetectionSessionListResponse(CamelCaseModel):
    """Paginated list of face detection sessions."""

    items: list[FaceDetectionSessionResponse]
    total: int
    page: int
    page_size: int


class FaceSuggestionResponse(CamelCaseModel):
    """Response schema for a face suggestion."""

    id: int
    face_instance_id: str  # UUID as string
    suggested_person_id: str  # UUID as string
    confidence: float
    source_face_id: str  # UUID as string
    status: str
    created_at: datetime
    reviewed_at: datetime | None
    # Include face thumbnail URL for UI
    face_thumbnail_url: str | None = None
    person_name: str | None = None
    # Full image and bounding box information for overlay display
    full_image_url: str | None = None
    path: str = ""  # filesystem path to the image
    bbox_x: int | None = None
    bbox_y: int | None = None
    bbox_w: int | None = None
    bbox_h: int | None = None
    detection_confidence: float | None = None
    quality_score: float | None = None

    # Multi-prototype scoring fields (for temporal/multi-era matching)
    matching_prototype_ids: list[str] | None = None
    prototype_scores: dict[str, float] | None = None
    aggregate_confidence: float | None = None
    prototype_match_count: int | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_multi_prototype_match(self) -> bool:
        """True if multiple prototypes matched this face."""
        return (self.prototype_match_count or 0) > 1


class FaceSuggestionListResponse(CamelCaseModel):
    """Paginated list of face suggestions (legacy flat pagination)."""

    items: list[FaceSuggestionResponse]
    total: int
    page: int
    page_size: int


class SuggestionGroup(CamelCaseModel):
    """Group of suggestions for a single person."""

    person_id: str  # UUID as string
    person_name: str | None
    suggestion_count: int  # Total suggestions for this person
    max_confidence: float
    suggestions: list[FaceSuggestionResponse]  # Limited by suggestionsPerGroup


class FaceSuggestionsGroupedResponse(CamelCaseModel):
    """Group-based paginated response for face suggestions."""

    groups: list[SuggestionGroup]
    total_groups: int  # Total number of person groups
    total_suggestions: int  # Total number of suggestions across all groups
    page: int
    groups_per_page: int
    suggestions_per_group: int


class AcceptSuggestionRequest(CamelCaseModel):
    """Request to accept a suggestion."""

    pass  # No additional fields needed


class RejectSuggestionRequest(CamelCaseModel):
    """Request to reject a suggestion."""

    pass  # No additional fields needed


class BulkSuggestionActionRequest(CamelCaseModel):
    """Request for bulk accept/reject of suggestions."""

    suggestion_ids: list[int]
    action: str = Field(..., pattern="^(accept|reject)$")
    auto_find_more: bool = Field(
        default=False,
        description="Auto-trigger find-more job after accepting",
    )
    find_more_prototype_count: int = Field(
        default=50,
        ge=10,
        le=1000,
        description="Prototype count for auto-triggered find-more jobs",
    )


class FindMoreJobInfo(CamelCaseModel):
    """Info about an auto-triggered find-more job."""

    person_id: str
    job_id: str
    progress_key: str


class BulkSuggestionActionResponse(CamelCaseModel):
    """Response for bulk suggestion action."""

    processed: int
    failed: int
    errors: list[str] = []
    find_more_jobs: list[FindMoreJobInfo] | None = Field(
        default=None,
        description="List of auto-triggered find-more jobs with progress keys",
    )


# Find More Suggestions Schemas


class FindMoreSuggestionsRequest(CamelCaseModel):
    """Request to find more suggestions using random face sampling."""

    prototype_count: int = Field(
        default=50,
        ge=10,
        le=1000,
        description="Number of labeled faces to sample as temporary prototypes",
    )
    max_suggestions: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Maximum number of new suggestions to create",
    )


class FindMoreJobResponse(CamelCaseModel):
    """Response when starting a find-more job."""

    job_id: str
    person_id: str
    person_name: str
    prototype_count: int
    labeled_face_count: int  # Total available for sampling
    status: str  # "queued"
    progress_key: str  # Redis key for progress


class JobProgress(CamelCaseModel):
    """Progress update for any background job (SSE payload)."""

    phase: str
    current: int
    total: int
    message: str
    timestamp: str  # ISO format


class JobResult(CamelCaseModel):
    """Final result when job completes."""

    status: str  # "completed" or "failed"
    suggestions_created: int | None = None
    prototypes_used: int | None = None
    candidates_found: int | None = None
    duplicates_skipped: int | None = None
    error: str | None = None
