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
    skipped_images: int = Field(0, alias="skippedImages")
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
    image_path: str | None = Field(None, alias="imagePath")
    skip_reason: str | None = Field(None, alias="skipReason")
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
    unique: int | None = None
    skipped: int | None = None


class JobsSummary(BaseModel):
    """Summary of jobs for training session."""

    model_config = ConfigDict(populate_by_name=True)

    pending: int
    running: int
    completed: int
    failed: int
    cancelled: int
    skipped: int = 0


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


# ============================================================================
# Unified Multi-Phase Progress Schemas
# ============================================================================


class PhaseProgress(BaseModel):
    """Progress information for a single training phase."""

    model_config = ConfigDict(populate_by_name=True)

    name: str  # "training" | "face_detection" | "clustering"
    status: str  # "pending" | "running" | "completed" | "failed" | "paused"
    progress: ProgressStats
    started_at: str | None = Field(None, alias="startedAt")
    completed_at: str | None = Field(None, alias="completedAt")


class OverallProgress(BaseModel):
    """Overall progress across all phases."""

    model_config = ConfigDict(populate_by_name=True)

    percentage: float = Field(..., ge=0, le=100)
    eta_seconds: int | None = Field(None, alias="etaSeconds")
    # "training" | "face_detection" | "clustering" | "completed"
    current_phase: str = Field(alias="currentPhase")


class UnifiedProgressResponse(BaseModel):
    """Unified progress response combining all training phases."""

    model_config = ConfigDict(populate_by_name=True)

    session_id: int = Field(alias="sessionId")
    # "pending" | "running" | "completed" | "failed" | "paused"
    overall_status: str = Field(alias="overallStatus")
    overall_progress: OverallProgress = Field(alias="overallProgress")
    phases: dict[str, PhaseProgress]  # Keys: "training", "faceDetection", "clustering"


# ============================================================================
# Restart Operation Schemas
# ============================================================================


class RestartResponseBase(BaseModel):
    """Base response for all restart operations."""

    model_config = ConfigDict(populate_by_name=True)

    session_id: int = Field(alias="sessionId")
    status: str
    message: str
    cleanup_stats: dict[str, object] = Field(alias="cleanupStats")


class TrainingRestartResponse(RestartResponseBase):
    """Response for training restart (Phase 1: CLIP embeddings)."""

    pass


class FaceDetectionRestartResponse(RestartResponseBase):
    """Response for face detection restart (Phase 2: InsightFace)."""

    pass


class ClusteringRestartResponse(RestartResponseBase):
    """Response for clustering restart (Phase 3: HDBSCAN)."""

    pass


# ============================================================================
# Directory Image Preview Schemas (for non-ingested images)
# ============================================================================


class DirectoryImageInfo(BaseModel):
    """Information about a single image file in directory preview."""

    model_config = ConfigDict(populate_by_name=True)

    filename: str
    full_path: str = Field(alias="fullPath")  # For thumbnail URL construction
    size_bytes: int = Field(alias="sizeBytes")
    modified_at: str = Field(alias="modifiedAt")  # ISO datetime


class DirectoryPreviewResponse(BaseModel):
    """Response for directory image preview (before ingestion)."""

    model_config = ConfigDict(populate_by_name=True)

    directory: str
    image_count: int = Field(alias="imageCount")
    images: list[DirectoryImageInfo]


# ============================================================================
# Directory Ignore Schemas
# ============================================================================


class IgnoreDirectoryRequest(BaseModel):
    """Request to mark a directory as ignored."""

    model_config = ConfigDict(populate_by_name=True)

    path: str
    reason: str | None = None


class IgnoreDirectoryResponse(BaseModel):
    """Response from directory ignore operation."""

    model_config = ConfigDict(populate_by_name=True)

    status: str
    path: str
    ignored_at: datetime | None = Field(None, alias="ignoredAt")


class IgnoredDirectoryInfo(BaseModel):
    """Information about an ignored directory."""

    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    path: str
    reason: str | None
    ignored_at: datetime = Field(alias="ignoredAt")
    ignored_by: str | None = Field(None, alias="ignoredBy")


class IgnoredDirectoriesResponse(BaseModel):
    """Response with list of ignored directories."""

    model_config = ConfigDict(populate_by_name=True)

    directories: list[IgnoredDirectoryInfo]
