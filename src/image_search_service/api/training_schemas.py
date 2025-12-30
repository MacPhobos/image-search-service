"""Pydantic schemas for training system API."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from image_search_service.api.category_schemas import CategoryResponse

# ============================================================================
# Training Session Schemas
# ============================================================================


class TrainingSessionConfig(BaseModel):
    """Configuration for training session."""

    recursive: bool = True
    extensions: list[str] = Field(default=["jpg", "jpeg", "png", "webp"])
    batch_size: int = 32


class TrainingSessionCreate(BaseModel):
    """Request to create a new training session."""

    model_config = ConfigDict(populate_by_name=True)

    name: str
    root_path: str = Field(alias="rootPath")
    category_id: int = Field(alias="categoryId", description="Category ID for this session")
    subdirectories: list[str] = Field(default_factory=list)
    config: TrainingSessionConfig | None = None


class TrainingSessionUpdate(BaseModel):
    """Request to update training session metadata."""

    model_config = ConfigDict(populate_by_name=True)

    name: str | None = None
    config: TrainingSessionConfig | None = None


class TrainingSessionResponse(BaseModel):
    """Response schema for training session."""

    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    id: int
    name: str
    status: str
    root_path: str = Field(alias="rootPath")
    category_id: int | None = Field(None, alias="categoryId", serialization_alias="categoryId")
    category: CategoryResponse | None = None
    config: dict[str, object] | None = None
    total_images: int = Field(alias="totalImages")
    processed_images: int = Field(alias="processedImages")
    failed_images: int = Field(alias="failedImages")
    created_at: datetime = Field(alias="createdAt")
    started_at: datetime | None = Field(None, alias="startedAt")
    completed_at: datetime | None = Field(None, alias="completedAt")
    paused_at: datetime | None = Field(None, alias="pausedAt")


# ============================================================================
# Training Subdirectory Schemas
# ============================================================================


class SubdirectorySelectionUpdate(BaseModel):
    """Request to update subdirectory selection status."""

    model_config = ConfigDict(populate_by_name=True)

    id: int
    selected: bool


class TrainingSubdirectoryResponse(BaseModel):
    """Response schema for training subdirectory."""

    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    id: int
    session_id: int = Field(alias="sessionId")
    path: str
    name: str
    selected: bool
    image_count: int = Field(alias="imageCount")
    trained_count: int = Field(alias="trainedCount")
    status: str
    created_at: datetime = Field(alias="createdAt")


# ============================================================================
# Training Job Schemas
# ============================================================================


class TrainingJobResponse(BaseModel):
    """Response schema for training job."""

    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    id: int
    session_id: int = Field(alias="sessionId")
    asset_id: int = Field(alias="assetId")
    status: str
    rq_job_id: str | None = Field(None, alias="rqJobId")
    progress: int
    error_message: str | None = Field(None, alias="errorMessage")
    processing_time_ms: int | None = Field(None, alias="processingTimeMs")
    created_at: datetime = Field(alias="createdAt")
    started_at: datetime | None = Field(None, alias="startedAt")
    completed_at: datetime | None = Field(None, alias="completedAt")


# ============================================================================
# Training Evidence Schemas
# ============================================================================


class TrainingEvidenceResponse(BaseModel):
    """Response schema for training evidence."""

    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    id: int
    asset_id: int = Field(alias="assetId")
    session_id: int = Field(alias="sessionId")
    model_name: str = Field(alias="modelName")
    model_version: str = Field(alias="modelVersion")
    embedding_checksum: str | None = Field(None, alias="embeddingChecksum")
    device: str
    processing_time_ms: int = Field(alias="processingTimeMs")
    error_message: str | None = Field(None, alias="errorMessage")
    metadata_json: dict[str, object] | None = Field(None, alias="metadataJson")
    created_at: datetime = Field(alias="createdAt")


# ============================================================================
# Progress and Statistics Schemas
# ============================================================================


class ProgressStats(BaseModel):
    """Progress statistics for training session."""

    model_config = ConfigDict(populate_by_name=True)

    current: int
    total: int
    percentage: float
    eta_seconds: int | None = Field(None, alias="etaSeconds")
    images_per_minute: float | None = Field(None, alias="imagesPerMinute")


class JobsSummary(BaseModel):
    """Summary of jobs for training session."""

    model_config = ConfigDict(populate_by_name=True)

    pending: int
    running: int
    completed: int
    failed: int
    cancelled: int


class TrainingProgressResponse(BaseModel):
    """Response schema for training progress."""

    model_config = ConfigDict(populate_by_name=True)

    session_id: int = Field(alias="sessionId")
    status: str
    progress: ProgressStats
    jobs_summary: JobsSummary = Field(alias="jobsSummary")


class TrainingSessionDetailResponse(TrainingSessionResponse):
    """Detailed training session response with subdirectories and progress."""

    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    category_id: int | None = Field(None, alias="categoryId", serialization_alias="categoryId")
    category: CategoryResponse | None = None
    subdirectories: list[TrainingSubdirectoryResponse] = Field(default_factory=list)
    progress: ProgressStats | None = None
    jobs_summary: JobsSummary | None = Field(None, alias="jobsSummary")


# ============================================================================
# Directory Scanning Schemas
# ============================================================================


class DirectoryScanRequest(BaseModel):
    """Request to scan directory for images."""

    model_config = ConfigDict(populate_by_name=True)

    root_path: str = Field(alias="rootPath")
    recursive: bool = True
    extensions: list[str] = Field(default=["jpg", "jpeg", "png", "webp"])


class DirectoryInfo(BaseModel):
    """Information about a directory."""

    model_config = ConfigDict(populate_by_name=True)

    path: str
    name: str
    image_count: int = Field(alias="imageCount")
    selected: bool = False

    # Training status metadata (optional, populated when include_training_status=true)
    trained_count: int | None = Field(None, alias="trainedCount")
    last_trained_at: datetime | None = Field(None, alias="lastTrainedAt")
    training_status: str | None = Field(None, alias="trainingStatus")


class DirectoryScanResponse(BaseModel):
    """Response from directory scan operation."""

    model_config = ConfigDict(populate_by_name=True)

    root_path: str = Field(alias="rootPath")
    subdirectories: list[DirectoryInfo]
    total_subdirectories: int = Field(alias="totalSubdirectories")
    total_images: int = Field(alias="totalImages")


# ============================================================================
# Control Operation Schemas
# ============================================================================


class RestartRequest(BaseModel):
    """Request to restart training."""

    model_config = ConfigDict(populate_by_name=True)

    failed_only: bool = Field(True, alias="failedOnly")


class ControlResponse(BaseModel):
    """Response from control operation."""

    model_config = ConfigDict(populate_by_name=True)

    session_id: int = Field(alias="sessionId")
    status: str
    message: str


# ============================================================================
# Evidence Statistics Schemas
# ============================================================================


class EvidenceStatsResponse(BaseModel):
    """Aggregate statistics for session's training evidence."""

    model_config = ConfigDict(populate_by_name=True)

    total: int
    successful: int
    failed: int
    avg_processing_time_ms: float = Field(alias="avgProcessingTimeMs")
    min_processing_time_ms: int = Field(alias="minProcessingTimeMs")
    max_processing_time_ms: int = Field(alias="maxProcessingTimeMs")
    devices_used: list[str] = Field(alias="devicesUsed")
    model_versions: list[str] = Field(alias="modelVersions")


class TrainingEvidenceDetailResponse(BaseModel):
    """Extended evidence response with asset information."""

    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    id: int
    asset_id: int = Field(alias="assetId")
    asset_path: str = Field(alias="assetPath")
    session_id: int = Field(alias="sessionId")
    model_name: str = Field(alias="modelName")
    model_version: str = Field(alias="modelVersion")
    embedding_checksum: str | None = Field(None, alias="embeddingChecksum")
    device: str
    processing_time_ms: int = Field(alias="processingTimeMs")
    error_message: str | None = Field(None, alias="errorMessage")
    metadata_json: dict[str, object] | None = Field(None, alias="metadataJson")
    created_at: datetime = Field(alias="createdAt")
