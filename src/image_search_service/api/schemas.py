"""Pydantic schemas for API requests and responses."""

from datetime import datetime
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

T = TypeVar("T")


class LocationMetadata(BaseModel):
    """GPS location from EXIF."""

    model_config = ConfigDict(populate_by_name=True)

    latitude: float = Field(alias="lat")
    longitude: float = Field(alias="lng")


class CameraMetadata(BaseModel):
    """Camera info from EXIF."""

    model_config = ConfigDict(populate_by_name=True)

    make: str | None = None
    model: str | None = None


class Asset(BaseModel):
    """Asset response schema."""

    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    id: int
    path: str
    created_at: datetime = Field(alias="createdAt")
    indexed_at: datetime | None = Field(None, alias="indexedAt")
    taken_at: datetime | None = Field(None, alias="takenAt")
    camera: CameraMetadata | None = None
    location: LocationMetadata | None = None

    @model_validator(mode="before")
    @classmethod
    def build_nested_metadata(cls, data: Any) -> dict[str, Any]:
        """Transform flat DB fields into nested EXIF metadata objects.

        Constructs camera and location objects from flat fields in ImageAsset model.
        Only creates nested objects when underlying data is present.

        Args:
            data: SQLAlchemy model instance or dict

        Returns:
            Dict with nested camera and location objects (or None)
        """
        # If already a dict (e.g., from JSON), pass through
        if isinstance(data, dict):
            return data

        # Convert SQLAlchemy model to dict
        values: dict[str, Any] = {}
        for field in ["id", "path", "created_at", "indexed_at", "taken_at"]:
            values[field] = getattr(data, field, None)

        # Build camera metadata if any camera field is present
        camera_make = getattr(data, "camera_make", None)
        camera_model = getattr(data, "camera_model", None)
        if camera_make is not None or camera_model is not None:
            values["camera"] = {
                "make": camera_make,
                "model": camera_model,
            }
        else:
            values["camera"] = None

        # Build location metadata only if both lat and lng are present
        gps_latitude = getattr(data, "gps_latitude", None)
        gps_longitude = getattr(data, "gps_longitude", None)
        if gps_latitude is not None and gps_longitude is not None:
            values["location"] = {
                "lat": gps_latitude,
                "lng": gps_longitude,
            }
        else:
            values["location"] = None

        return values

    @computed_field(alias="url")
    def url(self) -> str:
        """Full-size image URL."""
        return f"/api/v1/images/{self.id}/full"

    @computed_field(alias="thumbnailUrl")
    def thumbnail_url(self) -> str:
        """Thumbnail image URL."""
        return f"/api/v1/images/{self.id}/thumbnail"

    @computed_field(alias="filename")
    def filename(self) -> str:
        """Extracted filename from path."""
        return self.path.split("/")[-1]


class IngestRequest(BaseModel):
    """Request to ingest images from filesystem."""

    model_config = ConfigDict(populate_by_name=True)

    root_path: str = Field(alias="rootPath")
    recursive: bool = True
    extensions: list[str] = ["jpg", "jpeg", "png", "webp"]
    dry_run: bool = Field(False, alias="dryRun")


class IngestResponse(BaseModel):
    """Response from ingest operation."""

    discovered: int
    enqueued: int
    skipped: int


class SearchRequest(BaseModel):
    """Request to search for assets."""

    model_config = ConfigDict(populate_by_name=True)

    query: str
    limit: int = 50
    offset: int = 0
    filters: dict[str, str] | None = None  # from_date, to_date
    category_id: int | None = Field(None, alias="categoryId", description="Filter by category ID")


class SearchResult(BaseModel):
    """Single search result with asset and score."""

    asset: Asset
    score: float
    highlights: list[str] = []


class SearchResponse(BaseModel):
    """Response from search operation."""

    results: list[SearchResult]
    total: int
    query: str


class PaginatedResponse[T](BaseModel):
    """Generic paginated response."""

    model_config = ConfigDict(populate_by_name=True)

    items: list[T]
    total: int
    page: int
    page_size: int = Field(alias="pageSize")
    has_more: bool = Field(alias="hasMore")


class BatchThumbnailRequest(BaseModel):
    """Request for batch thumbnail retrieval."""

    model_config = ConfigDict(populate_by_name=True)

    asset_ids: list[int] = Field(
        alias="assetIds",
        min_length=1,
        max_length=100,
        description="Array of asset IDs to fetch thumbnails for",
    )


class BatchThumbnailResponse(BaseModel):
    """Response containing batch thumbnails as data URIs."""

    model_config = ConfigDict(populate_by_name=True)

    thumbnails: dict[str, str | None] = Field(
        description="Map of asset ID to base64 data URI (or null if not found)"
    )
    found: int = Field(description="Count of successfully retrieved thumbnails")
    not_found: list[int] = Field(
        alias="notFound",
        default_factory=list,
        description="Array of asset IDs that were not found",
    )


class ImageSearchRequest(BaseModel):
    """Request to search by image with optional filters."""

    model_config = ConfigDict(populate_by_name=True)

    limit: int = 50
    offset: int = 0
    filters: dict[str, str | int] | None = None
    category_id: int | None = Field(None, alias="categoryId", description="Filter by category ID")


class SimilarSearchRequest(BaseModel):
    """Request to find similar images."""

    model_config = ConfigDict(populate_by_name=True)

    limit: int = 50
    offset: int = 0
    exclude_self: bool = Field(True, alias="excludeSelf", description="Exclude source image from results")
    filters: dict[str, str | int] | None = None
    category_id: int | None = Field(None, alias="categoryId", description="Filter by category ID")


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str
    message: str
    details: dict[str, str] | None = None
