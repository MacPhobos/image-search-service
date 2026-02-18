"""Pydantic schemas for Google Drive storage API.

Definition order matters: UploadOptions MUST be defined before StartUploadRequest
to avoid a NameError (forward reference would not resolve at class-body evaluation time).
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

# ============================================================================
# Upload Schemas
# ============================================================================


class UploadOptions(BaseModel):
    """Optional configuration for an upload batch."""

    model_config = ConfigDict(populate_by_name=True)

    person_name_override: str | None = Field(
        None,
        alias="personNameOverride",
        description="Override subfolder name (default: person's name from DB)",
    )
    create_person_subfolder: bool = Field(
        True,
        alias="createPersonSubfolder",
        description="Create a subfolder named after the person inside folder_id",
    )


class StartUploadRequest(BaseModel):
    """Request to start a batch upload of photos to Google Drive."""

    model_config = ConfigDict(populate_by_name=True)

    person_id: str = Field(
        alias="personId",
        description="UUID of the person whose photos to upload",
    )
    photo_ids: list[int] = Field(
        alias="photoIds",
        description=(
            "List of ImageAsset IDs to upload. "
            "If empty, uploads all photos for the person."
        ),
    )
    folder_id: str = Field(
        alias="folderId",
        description="Google Drive folder ID to upload into",
    )
    options: UploadOptions | None = None


class StartUploadResponse(BaseModel):
    """Response after successfully enqueuing a batch upload."""

    model_config = ConfigDict(populate_by_name=True)

    batch_id: str = Field(alias="batchId")
    job_ids: list[str] = Field(
        alias="jobIds",
        description="RQ job IDs for the upload chunks",
    )
    total_photos: int = Field(alias="totalPhotos")
    estimated_time_seconds: int | None = Field(
        None,
        alias="estimatedTimeSeconds",
        description="Rough ETA based on photo count and rate limits (~3 uploads/sec)",
    )
    message: str


class UploadFileStatus(BaseModel):
    """Status of a single file within a batch upload."""

    model_config = ConfigDict(populate_by_name=True)

    asset_id: int = Field(alias="assetId")
    filename: str
    status: str  # "pending" | "uploading" | "completed" | "failed" | "cancelled"
    remote_file_id: str | None = Field(None, alias="remoteFileId")
    error_message: str | None = Field(None, alias="errorMessage")
    completed_at: datetime | None = Field(None, alias="completedAt")


class UploadStatusResponse(BaseModel):
    """Polling response for batch upload progress.

    Field names (canonical per INTERFACES.md §6.2):
        batch_id     (NOT job_id)
        total        (NOT total_files)
        completed    (NOT completed_files)
        failed       (NOT failed_files)
        in_progress  (computed: pending + uploading)
        percentage
        eta_seconds
        files        (list[UploadFileStatus])
        started_at
        completed_at
    """

    model_config = ConfigDict(populate_by_name=True)

    batch_id: str = Field(alias="batchId")
    status: str = Field(
        description=(
            'Overall batch status: "pending" | "in_progress" | '
            '"completed" | "partial_failure" | "cancelled" | "failed"'
        ),
    )
    total: int
    completed: int
    failed: int
    in_progress: int = Field(alias="inProgress")
    percentage: float = Field(ge=0, le=100)
    eta_seconds: int | None = Field(None, alias="etaSeconds")
    files: list[UploadFileStatus]
    started_at: datetime | None = Field(None, alias="startedAt")
    completed_at: datetime | None = Field(None, alias="completedAt")


class CancelUploadResponse(BaseModel):
    """Response after cancelling an upload batch."""

    model_config = ConfigDict(populate_by_name=True)

    batch_id: str = Field(alias="batchId")
    status: str  # "cancelled"
    completed_before_cancel: int = Field(alias="completedBeforeCancel")
    cancelled_count: int = Field(alias="cancelledCount")
    message: str


# ============================================================================
# Folder Schemas
# ============================================================================


class DriveFolderInfo(BaseModel):
    """A single Google Drive folder entry."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    name: str
    parent_id: str | None = Field(None, alias="parentId")
    has_children: bool = Field(alias="hasChildren")
    created_at: datetime | None = Field(None, alias="createdAt")


class FolderListResponse(BaseModel):
    """Response listing folders in Google Drive."""

    model_config = ConfigDict(populate_by_name=True)

    parent_id: str | None = Field(None, alias="parentId")
    folders: list[DriveFolderInfo]
    total: int


class CreateFolderRequest(BaseModel):
    """Request to create a new folder in Google Drive."""

    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(min_length=1, max_length=255)
    parent_id: str | None = Field(
        None,
        alias="parentId",
        description="Parent folder ID. Null = root folder.",
    )


class CreateFolderResponse(BaseModel):
    """Response after creating a folder."""

    model_config = ConfigDict(populate_by_name=True)

    folder_id: str = Field(alias="folderId")
    name: str
    parent_id: str | None = Field(None, alias="parentId")
    path: str = Field(description="Human-readable path for display")


# ============================================================================
# Health / Status Schemas
# ============================================================================


class DriveHealthResponse(BaseModel):
    """Connection health and quota information for Google Drive.

    Always returned with HTTP 200 — the frontend uses `connected` and `enabled`
    fields to decide whether to show upload buttons.
    """

    model_config = ConfigDict(populate_by_name=True)

    connected: bool
    enabled: bool
    service_account_email: str | None = Field(None, alias="serviceAccountEmail")
    root_folder_id: str | None = Field(None, alias="rootFolderId")
    root_folder_name: str | None = Field(None, alias="rootFolderName")
    storage_used_bytes: int | None = Field(None, alias="storageUsedBytes")
    storage_total_bytes: int | None = Field(None, alias="storageTotalBytes")
    storage_usage_percentage: float | None = Field(None, alias="storageUsagePercentage")
    last_upload_at: datetime | None = Field(None, alias="lastUploadAt")
    error: str | None = Field(
        None,
        description="Error message if connection test failed",
    )


# ============================================================================
# Error Schemas
# ============================================================================


class StorageErrorResponse(BaseModel):
    """Standard error response for storage endpoints."""

    detail: str
    error_code: str | None = Field(None, alias="errorCode")
    retry_after_seconds: int | None = Field(None, alias="retryAfterSeconds")
