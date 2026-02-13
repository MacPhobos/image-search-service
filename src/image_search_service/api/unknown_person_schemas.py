"""Pydantic schemas for unknown person discovery and management."""

from __future__ import annotations

from datetime import datetime
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


# ============ Request Schemas ============


class DiscoverUnknownPersonsRequest(CamelCaseModel):
    """Request to discover unknown person candidates via clustering."""

    clustering_method: str = Field(
        default="hdbscan",
        pattern="^(hdbscan|dbscan)$",
        description="Clustering algorithm to use (currently only hdbscan implemented)",
    )
    min_cluster_size: int = Field(
        default=5,
        ge=2,
        le=50,
        description="Minimum faces per cluster",
    )
    min_quality: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum quality score for faces to include",
    )
    max_faces: int = Field(
        default=50000,
        ge=100,
        le=100000,
        description="Maximum unassigned faces to process",
    )
    min_cluster_confidence: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Minimum intra-cluster confidence to display group",
    )
    eps: float = Field(
        default=0.5,
        ge=0.0,
        le=2.0,
        description="DBSCAN epsilon parameter (distance threshold)",
    )


class AcceptUnknownPersonRequest(CamelCaseModel):
    """Request to accept an unknown person group as a labeled person."""

    name: str = Field(
        min_length=1,
        max_length=255,
        description="Name for the new person",
    )
    face_ids_to_exclude: list[UUID] | None = Field(
        default=None,
        description="Face IDs to exclude from assignment (partial acceptance)",
    )
    trigger_reclustering: bool = Field(
        default=True,
        description="Trigger re-clustering after acceptance to update remaining unknowns",
    )


class DismissUnknownPersonRequest(CamelCaseModel):
    """Request to dismiss an unknown person group."""

    reason: str | None = Field(
        default=None,
        description="Optional reason for dismissal",
    )
    mark_as_noise: bool = Field(
        default=False,
        description="Mark faces as noise (cluster_id = '-1') to exclude from future clustering",
    )


# ============ Response Schemas ============


class DiscoverJobResponse(CamelCaseModel):
    """Response from discovery job enqueue."""

    job_id: str
    status: str
    progress_key: str
    params: dict[str, object]


class FaceInGroupResponse(CamelCaseModel):
    """Single face instance in an unknown person group."""

    face_instance_id: UUID
    asset_id: int
    quality_score: float
    detection_confidence: float
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    thumbnail_url: str | None = None


class UnknownPersonCandidateGroup(CamelCaseModel):
    """A candidate unknown person group (cluster of unassigned faces)."""

    group_id: str
    membership_hash: str
    face_count: int
    cluster_confidence: float
    avg_quality: float
    representative_face: FaceInGroupResponse
    sample_faces: list[FaceInGroupResponse]
    is_dismissed: bool = False
    dismissed_at: datetime | None = None


class UnknownPersonCandidatesResponse(CamelCaseModel):
    """Paginated list of unknown person candidate groups."""

    groups: list[UnknownPersonCandidateGroup]
    total_groups: int
    total_unassigned_faces: int
    total_noise_faces: int
    total_dismissed_groups: int
    page: int
    groups_per_page: int
    faces_per_group: int
    last_discovery_at: datetime | None = None
    min_group_size_setting: int
    min_confidence_setting: float
    discovery_min_confidence: float = Field(default=0.50)
    filtered_by_confidence: int = Field(default=0)
    filtered_by_size: int = Field(default=0)
    filtered_by_dismissed: int = Field(default=0)
    total_before_filtering: int = Field(default=0)


class UnknownPersonCandidateDetail(CamelCaseModel):
    """Detailed view of a single unknown person group with all faces."""

    group_id: str
    membership_hash: str
    face_count: int
    cluster_confidence: float
    avg_quality: float
    faces: list[FaceInGroupResponse]


class AcceptUnknownPersonResponse(CamelCaseModel):
    """Response from accepting an unknown person group."""

    person_id: UUID
    person_name: str
    faces_assigned: int
    faces_excluded: int
    prototypes_created: int
    find_more_job_id: str | None = None
    reclustering_job_id: str | None = None


class DismissUnknownPersonResponse(CamelCaseModel):
    """Response from dismissing an unknown person group."""

    group_id: str
    membership_hash: str
    faces_affected: int
    marked_as_noise: bool


class UnknownPersonsStatsResponse(CamelCaseModel):
    """Statistics about unknown persons and unassigned faces."""

    total_unassigned_faces: int
    total_clustered_faces: int
    total_noise_faces: int
    total_unclustered_faces: int
    candidate_groups: int
    avg_group_size: float
    avg_group_confidence: float
    total_dismissed_groups: int
    last_discovery_at: datetime | None = None


# ============ Merge Schemas ============


class MergeSuggestion(CamelCaseModel):
    """Suggestion to merge two candidate groups based on centroid similarity."""

    group_a_id: str
    group_b_id: str
    similarity: float = Field(ge=0.0, le=1.0, description="Cosine similarity between centroids")
    group_a_face_count: int
    group_b_face_count: int


class MergeSuggestionsResponse(CamelCaseModel):
    """Response containing merge suggestions."""

    suggestions: list[MergeSuggestion]
    total_groups_compared: int


class MergeGroupsRequest(CamelCaseModel):
    """Request to merge two candidate groups."""

    group_a_id: str = Field(description="Target group to merge into")
    group_b_id: str = Field(description="Source group to merge from")


class MergeGroupsResponse(CamelCaseModel):
    """Response from merging two groups."""

    merged_group_id: str = Field(description="Resulting group ID (group_a_id)")
    total_faces: int = Field(description="Total faces in merged group")
    faces_moved: int = Field(description="Number of faces moved from group_b")
