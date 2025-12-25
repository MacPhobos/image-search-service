"""Pydantic schemas for face detection and recognition APIs."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class BoundingBox(CamelCaseModel):
    """Face bounding box coordinates."""

    x: int
    y: int
    width: int
    height: int


class Landmarks(CamelCaseModel):
    """5-point facial landmarks."""

    left_eye: list[float]
    right_eye: list[float]
    nose: list[float]
    mouth_left: list[float]
    mouth_right: list[float]


class FaceInstanceResponse(CamelCaseModel):
    """Response schema for a face instance."""

    id: UUID
    asset_id: int
    bbox: BoundingBox
    detection_confidence: float
    quality_score: float | None = None
    cluster_id: str | None = None
    person_id: UUID | None = None
    person_name: str | None = None
    created_at: datetime


class FaceInstanceListResponse(CamelCaseModel):
    """Paginated list of face instances."""

    items: list[FaceInstanceResponse]
    total: int
    page: int
    page_size: int


class ClusterSummary(CamelCaseModel):
    """Summary of a face cluster."""

    cluster_id: str
    face_count: int
    sample_face_ids: list[UUID]
    avg_quality: float | None = None
    person_id: UUID | None = None
    person_name: str | None = None


class ClusterListResponse(CamelCaseModel):
    """Paginated list of clusters."""

    items: list[ClusterSummary]
    total: int
    page: int
    page_size: int


class ClusterDetailResponse(CamelCaseModel):
    """Detailed cluster info with all faces."""

    cluster_id: str
    faces: list[FaceInstanceResponse]
    person_id: UUID | None = None
    person_name: str | None = None


class PersonResponse(CamelCaseModel):
    """Response schema for a person."""

    id: UUID
    name: str
    status: str
    face_count: int
    prototype_count: int
    created_at: datetime
    updated_at: datetime


class PersonListResponse(CamelCaseModel):
    """Paginated list of persons."""

    items: list[PersonResponse]
    total: int
    page: int
    page_size: int


class LabelClusterRequest(CamelCaseModel):
    """Request to label a cluster with a person name."""

    name: str = Field(min_length=1, max_length=255)


class LabelClusterResponse(CamelCaseModel):
    """Response from labeling a cluster."""

    person_id: UUID
    person_name: str
    faces_labeled: int
    prototypes_created: int


class MergePersonsRequest(CamelCaseModel):
    """Request to merge one person into another."""

    into_person_id: UUID


class MergePersonsResponse(CamelCaseModel):
    """Response from merging persons."""

    source_person_id: UUID
    target_person_id: UUID
    faces_moved: int


class SplitClusterRequest(CamelCaseModel):
    """Request to split a cluster."""

    min_cluster_size: int = Field(default=3, ge=2)


class SplitClusterResponse(CamelCaseModel):
    """Response from splitting a cluster."""

    original_cluster_id: str
    new_clusters: list[str]
    status: str


class TriggerClusteringRequest(CamelCaseModel):
    """Request to trigger clustering."""

    quality_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_faces: int = Field(default=50000, ge=100)
    min_cluster_size: int = Field(default=5, ge=2)


class ClusteringResultResponse(CamelCaseModel):
    """Response from clustering operation."""

    total_faces: int
    clusters_found: int
    noise_count: int
    status: str = "completed"


class DetectFacesRequest(CamelCaseModel):
    """Request to detect faces in an asset."""

    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    min_face_size: int = Field(default=20, ge=10)


class DetectFacesResponse(CamelCaseModel):
    """Response from face detection."""

    asset_id: int
    faces_detected: int
    face_ids: list[UUID]


class FaceInPhoto(CamelCaseModel):
    """Face instance info for photo grouping."""

    face_instance_id: UUID
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    detection_confidence: float
    quality_score: float | None = None
    person_id: UUID | None = None
    person_name: str | None = None
    cluster_id: str | None = None


class PersonPhotoGroup(CamelCaseModel):
    """A photo with its faces, grouped for person review."""

    photo_id: int  # asset_id
    taken_at: datetime | None = None  # from EXIF if available
    thumbnail_url: str
    full_url: str
    faces: list[FaceInPhoto]
    face_count: int
    has_non_person_faces: bool  # True if any face in photo has different/no person_id


class PersonPhotosResponse(CamelCaseModel):
    """Paginated response for person's photos."""

    items: list[PersonPhotoGroup]
    total: int
    page: int
    page_size: int
    person_id: UUID
    person_name: str


class BulkRemoveRequest(CamelCaseModel):
    """Request to remove person assignment from faces in selected photos."""

    photo_ids: list[int]  # asset_ids


class BulkRemoveResponse(CamelCaseModel):
    """Response from bulk remove operation."""

    updated_faces: int
    updated_photos: int
    skipped_faces: int  # faces that didn't match (already unassigned)


class BulkMoveRequest(CamelCaseModel):
    """Request to move faces from one person to another."""

    photo_ids: list[int]  # asset_ids
    to_person_id: UUID | None = None  # existing person
    to_person_name: str | None = None  # create new person if provided

    @model_validator(mode='after')
    def validate_destination(self) -> 'BulkMoveRequest':
        """Validate that either to_person_id or to_person_name is provided."""
        if not self.to_person_id and not self.to_person_name:
            raise ValueError("Either to_person_id or to_person_name must be provided")
        if self.to_person_id and self.to_person_name:
            raise ValueError("Provide either to_person_id or to_person_name, not both")
        return self


class BulkMoveResponse(CamelCaseModel):
    """Response from bulk move operation."""

    to_person_id: UUID
    to_person_name: str
    updated_faces: int
    updated_photos: int
    skipped_faces: int
    person_created: bool  # True if new person was created


class CreatePersonRequest(CamelCaseModel):
    """Request to create a new person."""

    name: str = Field(min_length=1, max_length=255)


class CreatePersonResponse(CamelCaseModel):
    """Response from creating a person."""

    id: UUID
    name: str
    status: str
    created_at: datetime


class AssignFaceRequest(CamelCaseModel):
    """Request to assign a face to a person."""

    person_id: UUID


class AssignFaceResponse(CamelCaseModel):
    """Response from assigning a face."""

    face_id: UUID
    person_id: UUID
    person_name: str


# ============ Dual-Mode Clustering Schemas ============


class ClusterDualRequest(CamelCaseModel):
    """Request to run dual-mode clustering (supervised + unsupervised)."""

    person_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    unknown_method: str = Field(default="hdbscan", pattern="^(hdbscan|dbscan|agglomerative)$")
    unknown_min_size: int = Field(default=3, ge=1)
    unknown_eps: float = Field(default=0.5, ge=0.0, le=2.0)
    max_faces: int | None = Field(default=None, ge=1)
    queue: bool = Field(default=True, description="Run as background job")


class ClusterDualResponse(CamelCaseModel):
    """Response from dual-mode clustering."""

    job_id: str | None = None
    status: str
    result: dict[str, int] | None = None


# ============ Face Training Schemas ============


class TrainMatchingRequest(CamelCaseModel):
    """Request to train face matching model using triplet loss."""

    epochs: int = Field(default=20, ge=1, le=1000)
    margin: float = Field(default=0.2, ge=0.0, le=1.0)
    batch_size: int = Field(default=32, ge=1, le=256)
    learning_rate: float = Field(default=0.0001, ge=0.00001, le=0.1)
    min_faces: int = Field(default=5, ge=1)
    checkpoint_path: str | None = Field(default=None)
    queue: bool = Field(default=True, description="Run as background job")


class TrainMatchingResponse(CamelCaseModel):
    """Response from face training."""

    job_id: str | None = None
    status: str
    result: dict[str, int | float] | None = None
