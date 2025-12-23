"""Pydantic schemas for face detection and recognition APIs."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


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
    quality_score: Optional[float] = None
    cluster_id: Optional[str] = None
    person_id: Optional[UUID] = None
    person_name: Optional[str] = None
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
    avg_quality: Optional[float] = None
    person_id: Optional[UUID] = None
    person_name: Optional[str] = None


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
    person_id: Optional[UUID] = None
    person_name: Optional[str] = None


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
