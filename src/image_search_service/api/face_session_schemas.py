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


class FaceSuggestionListResponse(CamelCaseModel):
    """Paginated list of face suggestions."""

    items: list[FaceSuggestionResponse]
    total: int
    page: int
    page_size: int


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


class BulkSuggestionActionResponse(CamelCaseModel):
    """Response for bulk suggestion action."""

    processed: int
    failed: int
    errors: list[str] = []
