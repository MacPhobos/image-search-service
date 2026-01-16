"""Pydantic schemas for centroid management API endpoints."""

from datetime import datetime
from typing import Literal
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


# Request schemas


class ComputeCentroidsRequest(CamelCaseModel):
    """Request to compute/recompute centroids for a person.

    Args:
        force_rebuild: Recompute even if centroid is fresh (default: False)
        enable_clustering: Enable cluster-based centroids (v2 feature, default: False)
        min_faces: Minimum faces required (default: 2, min: 2)
    """

    force_rebuild: bool = False
    enable_clustering: bool = False  # v2 feature, default off
    min_faces: int = Field(default=2, ge=2)


class CentroidSuggestionRequest(CamelCaseModel):
    """Request to get face suggestions using person centroids.

    Args:
        min_similarity: Minimum similarity score (0.5-0.95, default: 0.65)
        max_results: Maximum number of results (1-500, default: 200)
        unassigned_only: Only return unassigned faces (default: True)
        exclude_prototypes: Exclude faces marked as prototypes (default: True)
        auto_rebuild: Auto-rebuild stale centroids before search (default: True)
    """

    min_similarity: float = Field(default=0.65, ge=0.5, le=0.95)
    max_results: int = Field(default=200, ge=1, le=500)
    unassigned_only: bool = True
    exclude_prototypes: bool = True
    auto_rebuild: bool = True


# Response schemas


class CentroidInfo(CamelCaseModel):
    """Information about a single centroid.

    Includes metadata about centroid type, version, face count, and staleness.
    """

    centroid_id: UUID
    centroid_type: Literal["global", "cluster"]
    cluster_label: str
    n_faces: int
    model_version: str
    centroid_version: int
    created_at: datetime
    is_stale: bool = False


class ComputeCentroidsResponse(CamelCaseModel):
    """Response after computing centroids for a person.

    Returns:
        person_id: Person UUID
        centroids: List of computed centroids
        rebuilt: Whether centroids were recomputed
        stale_reason: Why rebuild was needed (if applicable)
    """

    person_id: UUID
    centroids: list[CentroidInfo]
    rebuilt: bool
    stale_reason: str | None = None


class GetCentroidsResponse(CamelCaseModel):
    """Response when fetching existing centroids for a person.

    Returns:
        person_id: Person UUID
        centroids: List of active centroids
        is_stale: Whether any centroid is stale
        stale_reason: Why rebuild is recommended (if applicable)
    """

    person_id: UUID
    centroids: list[CentroidInfo]
    is_stale: bool
    stale_reason: str | None = None


class CentroidSuggestion(CamelCaseModel):
    """Single face suggestion from centroid search.

    Includes face metadata, similarity score, and image URLs.
    """

    face_instance_id: UUID
    asset_id: str
    score: float
    matched_centroid: str  # Centroid UUID or label that matched
    thumbnail_url: str | None = None


class CentroidSuggestionResponse(CamelCaseModel):
    """Response with face suggestions from centroid search.

    Returns:
        person_id: Person UUID
        centroids_used: UUIDs of centroids used for search
        suggestions: List of face suggestions sorted by score
        total_found: Total number of matches found
        rebuilt_centroids: Whether centroids were rebuilt before search
    """

    person_id: UUID
    centroids_used: list[UUID]
    suggestions: list[CentroidSuggestion]
    total_found: int
    rebuilt_centroids: bool


class DeleteCentroidsResponse(CamelCaseModel):
    """Response after deleting centroids for a person.

    Returns:
        person_id: Person UUID
        deleted_count: Number of centroids deleted
    """

    person_id: UUID
    deleted_count: int
