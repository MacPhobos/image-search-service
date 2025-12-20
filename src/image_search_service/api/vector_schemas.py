"""Pydantic schemas for vector management API."""

from datetime import datetime

from pydantic import BaseModel, Field


class DirectoryDeleteRequest(BaseModel):
    """Request schema for deleting vectors by directory prefix."""

    path_prefix: str = Field(
        ..., alias="pathPrefix", min_length=1, description="Directory path prefix to delete"
    )
    deletion_reason: str | None = Field(
        None, alias="deletionReason", description="Optional reason for deletion"
    )
    confirm: bool = Field(False, description="Must be True to proceed with deletion")

    model_config = {"populate_by_name": True}


class DirectoryDeleteResponse(BaseModel):
    """Response schema for directory deletion."""

    path_prefix: str = Field(..., alias="pathPrefix")
    vectors_deleted: int = Field(..., alias="vectorsDeleted")
    message: str

    model_config = {"populate_by_name": True}


class RetrainRequest(BaseModel):
    """Request schema for retraining a directory."""

    path_prefix: str = Field(..., alias="pathPrefix", min_length=1)
    category_id: int = Field(..., alias="categoryId")
    deletion_reason: str | None = Field(None, alias="deletionReason")

    model_config = {"populate_by_name": True}


class RetrainResponse(BaseModel):
    """Response schema for retrain operation."""

    path_prefix: str = Field(..., alias="pathPrefix")
    vectors_deleted: int = Field(..., alias="vectorsDeleted")
    new_session_id: int = Field(..., alias="newSessionId")
    message: str

    model_config = {"populate_by_name": True}


class DirectoryStats(BaseModel):
    """Statistics for a single directory."""

    path_prefix: str = Field(..., alias="pathPrefix")
    vector_count: int = Field(..., alias="vectorCount")
    last_indexed: datetime | None = Field(None, alias="lastIndexed")

    model_config = {"populate_by_name": True}


class DirectoryStatsResponse(BaseModel):
    """Response schema for directory statistics."""

    directories: list[DirectoryStats]
    total_vectors: int = Field(..., alias="totalVectors")

    model_config = {"populate_by_name": True}


class AssetDeleteResponse(BaseModel):
    """Response schema for asset deletion."""

    asset_id: int = Field(..., alias="assetId")
    vectors_deleted: int = Field(..., alias="vectorsDeleted")
    message: str

    model_config = {"populate_by_name": True}


class SessionDeleteResponse(BaseModel):
    """Response schema for session deletion."""

    session_id: int = Field(..., alias="sessionId")
    vectors_deleted: int = Field(..., alias="vectorsDeleted")
    message: str

    model_config = {"populate_by_name": True}


class CategoryDeleteResponse(BaseModel):
    """Response schema for category deletion."""

    category_id: int = Field(..., alias="categoryId")
    vectors_deleted: int = Field(..., alias="vectorsDeleted")
    message: str

    model_config = {"populate_by_name": True}


class OrphanCleanupRequest(BaseModel):
    """Request schema for orphan cleanup."""

    confirm: bool = Field(False, description="Must be True to proceed")
    deletion_reason: str | None = Field(None, alias="deletionReason")

    model_config = {"populate_by_name": True}


class OrphanCleanupResponse(BaseModel):
    """Response schema for orphan cleanup."""

    orphans_deleted: int = Field(..., alias="orphansDeleted")
    message: str

    model_config = {"populate_by_name": True}


class ResetRequest(BaseModel):
    """Request schema for collection reset."""

    confirm: bool = Field(False, description="Must be True to proceed")
    confirmation_text: str = Field(
        ..., alias="confirmationText", description="Must be 'DELETE ALL VECTORS'"
    )
    deletion_reason: str | None = Field(None, alias="deletionReason")

    model_config = {"populate_by_name": True}


class ResetResponse(BaseModel):
    """Response schema for collection reset."""

    vectors_deleted: int = Field(..., alias="vectorsDeleted")
    message: str

    model_config = {"populate_by_name": True}


class DeletionLogEntry(BaseModel):
    """Schema for a deletion log entry."""

    id: int
    deletion_type: str = Field(..., alias="deletionType")
    deletion_target: str = Field(..., alias="deletionTarget")
    vector_count: int = Field(..., alias="vectorCount")
    deletion_reason: str | None = Field(None, alias="deletionReason")
    created_at: datetime = Field(..., alias="createdAt")

    model_config = {"populate_by_name": True}


class DeletionLogsResponse(BaseModel):
    """Response schema for deletion logs."""

    logs: list[DeletionLogEntry]
    total: int
    page: int
    page_size: int = Field(..., alias="pageSize")

    model_config = {"populate_by_name": True}
