"""Pydantic schemas for API requests and responses."""

from datetime import datetime
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, Field, computed_field

T = TypeVar("T")


class Asset(BaseModel):
    """Asset response schema."""

    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    id: int
    path: str
    created_at: datetime = Field(alias="createdAt")
    indexed_at: datetime | None = Field(None, alias="indexedAt")

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


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str
    message: str
    details: dict[str, str] | None = None
