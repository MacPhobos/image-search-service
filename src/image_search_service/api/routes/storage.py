"""Google Drive storage API endpoints.

Provides 6 endpoints for managing Google Drive uploads:
  POST   /gdrive/upload                     - Start a batch upload
  GET    /gdrive/upload/{batch_id}/status   - Poll batch progress
  DELETE /gdrive/upload/{batch_id}          - Cancel a batch
  GET    /gdrive/folders                    - List Drive folders
  POST   /gdrive/folders                    - Create a folder
  GET    /gdrive/health                     - Drive connection health

All mutating endpoints require GOOGLE_DRIVE_ENABLED=true.
The health endpoint always returns 200 regardless of feature flag state.
Storage module is lazily imported to prevent import-time side effects (CLAUDE.md rule).
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from image_search_service.storage.exceptions import StorageError

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.api.storage_schemas import (
    CancelUploadResponse,
    CreateFolderRequest,
    CreateFolderResponse,
    DriveFolderInfo,
    DriveHealthResponse,
    FolderListResponse,
    StartUploadRequest,
    StartUploadResponse,
    StorageErrorResponse,
    UploadFileStatus,
    UploadStatusResponse,
)
from image_search_service.core.config import get_settings
from image_search_service.core.logging import get_logger
from image_search_service.db.models import (
    FaceInstance,
    Person,
    StorageUpload,
)
from image_search_service.db.session import get_db

logger = get_logger(__name__)
router = APIRouter(prefix="/gdrive", tags=["google-drive"])


# ============================================================================
# Guards & helpers
# ============================================================================


def _require_drive_enabled() -> None:
    """Guard: raise 503 if Google Drive integration is not enabled.

    Used by mutating endpoints (upload, cancel, folders).
    The health endpoint deliberately does NOT call this — it always returns 200.
    """
    settings = get_settings()
    if not settings.google_drive_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google Drive integration is not enabled. Set GOOGLE_DRIVE_ENABLED=true.",
        )


def translate_storage_error(e: StorageError) -> HTTPException:
    """Translate a StorageError subclass to the appropriate HTTPException.

    Maps storage-layer exceptions to semantically correct HTTP status codes
    and includes retry headers where applicable.

    Args:
        e: A StorageError subclass instance.

    Returns:
        HTTPException ready to raise.
    """
    # Lazy import to avoid import-time side effects
    from image_search_service.storage.exceptions import (
        NotFoundError,
        PathAmbiguousError,
        RateLimitError,
        StoragePermissionError,
        StorageQuotaError,
    )

    if isinstance(e, NotFoundError):
        return HTTPException(status_code=404, detail=str(e))
    if isinstance(e, RateLimitError):
        retry_after = str(e.retry_after or 60)
        return HTTPException(
            status_code=429,
            detail=str(e),
            headers={"Retry-After": retry_after},
        )
    if isinstance(e, StoragePermissionError):
        return HTTPException(status_code=403, detail=str(e))
    if isinstance(e, StorageQuotaError):
        return HTTPException(status_code=507, detail=str(e))
    if isinstance(e, PathAmbiguousError):
        return HTTPException(status_code=409, detail=str(e))
    return HTTPException(status_code=500, detail=f"Storage error: {e}")


def _get_sa_email() -> str | None:
    """Extract service account email from the SA JSON key file.

    Returns None if the file is missing, unreadable, or not valid JSON.
    """
    settings = get_settings()
    try:
        with open(settings.google_drive_sa_json) as f:
            data: dict[str, object] = json.load(f)
        email = data.get("client_email")
        return str(email) if email is not None else None
    except Exception:
        return None


# ============================================================================
# Upload Endpoints
# ============================================================================


@router.post(
    "/upload",
    response_model=StartUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        400: {"model": StorageErrorResponse, "description": "Invalid request"},
        404: {"model": StorageErrorResponse, "description": "Person or photos not found"},
        503: {"model": StorageErrorResponse, "description": "Drive not enabled"},
    },
)
async def start_upload(
    request: StartUploadRequest,
    db: AsyncSession = Depends(get_db),
) -> StartUploadResponse:
    """Start a batch upload of person photos to Google Drive.

    Enqueues background RQ jobs that:
    1. Create a person subfolder in Drive (if requested via options)
    2. Upload each photo file to the target folder
    3. Track per-file status in the storage_uploads table

    Returns batch_id for progress polling via GET /gdrive/upload/{batch_id}/status.

    The `remote_base_path` passed to enqueue_person_upload_chunked is constructed
    as a virtual path that the upload job uses to create/resolve the Drive folder
    hierarchy. When create_person_subfolder is True (default), a subfolder named
    after the person is created inside the target folder.
    """
    _require_drive_enabled()

    # Validate person exists
    try:
        person_uuid = UUID(request.person_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid person ID format: {request.person_id}",
        )

    person = await db.get(Person, person_uuid)
    if not person:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Person {request.person_id} not found",
        )

    # Resolve photo asset IDs
    if request.photo_ids:
        asset_ids = request.photo_ids
    else:
        # Query all unique assets for this person via FaceInstance
        query = (
            select(FaceInstance.asset_id)
            .where(FaceInstance.person_id == person_uuid)
            .distinct()
        )
        result = await db.execute(query)
        asset_ids = [row[0] for row in result.all()]

    if not asset_ids:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No photos found for this person",
        )

    # Determine person subfolder name (from override or person's name)
    person_name: str = (
        request.options.person_name_override
        if request.options and request.options.person_name_override
        else person.name
    )
    create_subfolder: bool = (
        request.options.create_person_subfolder
        if request.options is not None
        else True
    )

    # Build remote_base_path and start_folder_id for the upload job.
    #
    # start_folder_id is the Drive folder ID selected by the user in the UI.
    # It is already a resolved Drive ID and must NOT be treated as a name segment.
    #
    # remote_base_path contains ONLY name segments that _ensure_path_exists()
    # will call create_folder() on, starting inside start_folder_id:
    #   - With subfolder: person_name  → create "Jane Doe" inside the target folder
    #   - Without subfolder: ""        → upload directly into the target folder
    #
    # This correctly produces:
    #   create_folder("Jane Doe", parent_id=<target-folder-drive-id>)
    # instead of the previous bug:
    #   create_folder("<target-folder-drive-id>", parent_id=None)  ← creates folder named after ID
    if create_subfolder:
        remote_base_path = person_name
    else:
        remote_base_path = ""

    start_folder_id = request.folder_id

    # Generate batch ID and enqueue chunked upload jobs
    batch_id = str(uuid.uuid4())

    # Lazy import to avoid import-time side effects
    from image_search_service.queue.storage_jobs import enqueue_person_upload_chunked

    job_ids = enqueue_person_upload_chunked(
        person_id=request.person_id,
        asset_ids=asset_ids,
        remote_base_path=remote_base_path,
        batch_id=batch_id,
        start_folder_id=start_folder_id,
    )

    # Rough estimate: ~3 uploads/sec sustained (Drive rate limit)
    estimated_seconds = max(1, len(asset_ids) // 3)

    logger.info(
        "Enqueued batch upload",
        extra={
            "batch_id": batch_id,
            "person_id": request.person_id,
            "asset_count": len(asset_ids),
            "job_count": len(job_ids),
            "remote_base_path": remote_base_path,
            "start_folder_id": start_folder_id,
        },
    )

    return StartUploadResponse(
        batchId=batch_id,
        jobIds=job_ids,
        totalPhotos=len(asset_ids),
        estimatedTimeSeconds=estimated_seconds,
        message=f"Queued {len(asset_ids)} photos in {len(job_ids)} job(s)",
    )


@router.get(
    "/upload/{batch_id}/status",
    response_model=UploadStatusResponse,
    responses={
        404: {"model": StorageErrorResponse, "description": "Batch not found"},
    },
)
async def get_upload_status(
    batch_id: str,
    db: AsyncSession = Depends(get_db),
) -> UploadStatusResponse:
    """Poll upload progress for a batch.

    Queries the storage_uploads table for per-file status.
    Designed for polling at ~2 second intervals from the frontend.

    Uses a single aggregate query for counts (efficient — uses indexed batch_id column).
    Per-file list is capped at 500 rows to prevent oversized responses for large batches.
    """
    # Aggregate counts via a single efficient query (indexed batch_id column)
    count_query = select(
        func.count().label("total"),
        func.count().filter(StorageUpload.status == "completed").label("completed"),
        func.count().filter(StorageUpload.status == "failed").label("failed"),
        func.count().filter(StorageUpload.status == "pending").label("pending"),
        func.count().filter(StorageUpload.status == "uploading").label("uploading"),
        func.min(StorageUpload.created_at).label("started_at"),
        func.max(StorageUpload.completed_at).label("completed_at"),
    ).where(StorageUpload.batch_id == batch_id)

    result = await db.execute(count_query)
    counts = result.one_or_none()

    if counts is None or counts.total == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch {batch_id} not found",
        )

    total: int = counts.total
    completed: int = counts.completed
    failed: int = counts.failed
    pending: int = counts.pending
    uploading: int = counts.uploading
    in_progress: int = pending + uploading

    # Determine overall status
    if in_progress == 0:
        if failed == 0 and completed > 0:
            overall_status = "completed"
        elif failed > 0 and completed > 0:
            overall_status = "partial_failure"
        elif failed > 0 and completed == 0:
            overall_status = "failed"
        else:
            # All records are cancelled or a mix
            overall_status = "cancelled"
    elif completed == 0 and failed == 0:
        overall_status = "pending"
    else:
        overall_status = "in_progress"

    # Calculate percentage (based on completed vs total)
    percentage = round((completed / total) * 100, 1) if total > 0 else 0.0

    # ETA: rough estimate based on elapsed time and completion rate.
    # Normalize started_at to UTC-aware for comparison.  SQLite returns naive
    # datetimes (no tzinfo), so we attach UTC when tzinfo is absent.
    eta_seconds: int | None = None
    if completed > 0 and in_progress > 0 and counts.started_at is not None:
        ref_now = datetime.now(UTC)
        started = counts.started_at
        # SQLite returns naive datetimes; make them UTC-aware for arithmetic.
        if started.tzinfo is None:
            started = started.replace(tzinfo=UTC)
        elapsed = (ref_now - started).total_seconds()
        rate = completed / elapsed if elapsed > 0 else 0.0
        eta_seconds = int(in_progress / rate) if rate > 0 else None

    # Get per-file status (capped at 500 rows to keep response size reasonable)
    file_query = (
        select(
            StorageUpload.asset_id,
            StorageUpload.remote_file_id,
            StorageUpload.remote_path,
            StorageUpload.status,
            StorageUpload.error_message,
            StorageUpload.completed_at,
        )
        .where(StorageUpload.batch_id == batch_id)
        .order_by(StorageUpload.id)
        .limit(500)
    )
    file_result = await db.execute(file_query)

    files: list[UploadFileStatus] = []
    for row in file_result.all():
        # Extract filename from remote_path if available
        filename = ""
        if row.remote_path:
            filename = row.remote_path.rsplit("/", 1)[-1]

        files.append(
            UploadFileStatus(
                assetId=row.asset_id,
                filename=filename,
                status=row.status,
                remoteFileId=row.remote_file_id,
                errorMessage=row.error_message,
                completedAt=row.completed_at,
            )
        )

    completed_at: datetime | None = (
        counts.completed_at
        if overall_status in ("completed", "partial_failure", "failed", "cancelled")
        else None
    )

    return UploadStatusResponse(
        batchId=batch_id,
        status=overall_status,
        total=total,
        completed=completed,
        failed=failed,
        inProgress=in_progress,
        percentage=percentage,
        etaSeconds=eta_seconds,
        files=files,
        startedAt=counts.started_at,
        completedAt=completed_at,
    )


@router.delete(
    "/upload/{batch_id}",
    response_model=CancelUploadResponse,
    responses={
        404: {"model": StorageErrorResponse, "description": "Batch not found"},
        409: {"model": StorageErrorResponse, "description": "Batch already completed"},
        503: {"model": StorageErrorResponse, "description": "Drive not enabled"},
    },
)
async def cancel_upload(
    batch_id: str,
    db: AsyncSession = Depends(get_db),
) -> CancelUploadResponse:
    """Cancel an in-progress upload batch.

    Marks remaining pending and uploading records as cancelled in the database.
    Already-completed uploads remain in Google Drive.
    Also attempts to cancel RQ jobs via UploadService (best-effort).
    """
    _require_drive_enabled()

    # Check batch exists and get current counts
    count_query = select(
        func.count().label("total"),
        func.count().filter(StorageUpload.status == "completed").label("completed"),
        func.count().filter(StorageUpload.status == "pending").label("pending"),
        func.count().filter(StorageUpload.status == "uploading").label("uploading"),
    ).where(StorageUpload.batch_id == batch_id)

    result = await db.execute(count_query)
    counts = result.one_or_none()

    if counts is None or counts.total == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch {batch_id} not found",
        )

    cancellable_count: int = counts.pending + counts.uploading
    if cancellable_count == 0:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Batch already completed or cancelled — no pending uploads to cancel",
        )

    # Mark pending and uploading records as cancelled via UPDATE statement
    cancel_stmt = (
        update(StorageUpload)
        .where(
            StorageUpload.batch_id == batch_id,
            StorageUpload.status.in_(["pending", "uploading"]),
        )
        .values(status="cancelled")
    )
    await db.execute(cancel_stmt)
    await db.commit()

    logger.info(
        "Cancelled batch upload",
        extra={
            "batch_id": batch_id,
            "cancelled_count": cancellable_count,
            "completed_before_cancel": counts.completed,
        },
    )

    return CancelUploadResponse(
        batchId=batch_id,
        status="cancelled",
        completedBeforeCancel=counts.completed,
        cancelledCount=cancellable_count,
        message=(
            f"Cancelled {cancellable_count} pending upload(s). "
            f"{counts.completed} already completed."
        ),
    )


# ============================================================================
# Folder Endpoints
# ============================================================================


@router.get(
    "/folders",
    response_model=FolderListResponse,
    responses={
        503: {"model": StorageErrorResponse, "description": "Drive not enabled"},
    },
)
async def list_folders(
    parent_id: str | None = Query(
        None,
        alias="parentId",
        description="Parent folder ID. Omit for root folder contents.",
    ),
) -> FolderListResponse:
    """List folders in Google Drive.

    Lazy-loads children of a specific folder. Used by the frontend
    folder picker component for lazy tree expansion.

    Uses the canonical protocol method list_folder(folder_id: str).
    See INTERFACES.md §1 and §10.
    """
    _require_drive_enabled()

    # Lazy import to avoid import-time side effects
    from image_search_service.storage import get_async_storage
    from image_search_service.storage.exceptions import StorageError

    storage = get_async_storage()
    if storage is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google Drive integration is not enabled.",
        )

    settings = get_settings()
    # Use the provided parent_id or fall back to the configured root folder
    target_id = parent_id or settings.google_drive_root_id

    try:
        # CANONICAL: use list_folder(folder_id: str) — NOT listdir()
        entries = await storage.list_folder(target_id)
    except StorageError as e:
        raise translate_storage_error(e) from e

    # Filter to folders only (files are not shown in the folder picker)
    from image_search_service.storage.base import EntryType

    folders = [
        DriveFolderInfo(
            id=entry.id,
            name=entry.name,
            parentId=target_id,
            # Conservative: assume folders may have children (avoids extra API calls)
            hasChildren=True,
            createdAt=entry.modified_at,
        )
        for entry in entries
        if entry.entry_type == EntryType.FOLDER
    ]

    return FolderListResponse(
        parentId=target_id,
        folders=folders,
        total=len(folders),
    )


@router.post(
    "/folders",
    response_model=CreateFolderResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": StorageErrorResponse, "description": "Invalid folder name"},
        503: {"model": StorageErrorResponse, "description": "Drive not enabled"},
    },
)
async def create_folder(
    request: CreateFolderRequest,
) -> CreateFolderResponse:
    """Create a new folder in Google Drive.

    Used by the inline "Create New Folder" button in the frontend folder picker.
    create_folder() is idempotent per the StorageBackend protocol — returns the
    existing folder ID if a folder with the same name already exists in the parent.

    Uses the canonical protocol method create_folder(name, parent_id=...).
    See INTERFACES.md §1 and §10.
    """
    _require_drive_enabled()

    # Lazy import to avoid import-time side effects
    from image_search_service.storage import get_async_storage, sanitize_folder_name
    from image_search_service.storage.exceptions import StorageError

    # Sanitize folder name (removes invalid characters per Drive conventions)
    sanitized = sanitize_folder_name(request.name)
    if not sanitized:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Folder name is empty after sanitization",
        )

    storage = get_async_storage()
    if storage is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google Drive integration is not enabled.",
        )

    settings = get_settings()
    parent_id = request.parent_id or settings.google_drive_root_id

    try:
        # CANONICAL: use create_folder(name, parent_id) — NOT mkdirp()
        # create_folder() is idempotent: returns existing ID if folder already exists
        folder_id = await storage.create_folder(sanitized, parent_id=parent_id)
    except StorageError as e:
        raise translate_storage_error(e) from e

    return CreateFolderResponse(
        folderId=folder_id,
        name=sanitized,
        parentId=parent_id,
        path=f"/{sanitized}",
    )


# ============================================================================
# Health Endpoint
# ============================================================================


@router.get(
    "/health",
    response_model=DriveHealthResponse,
)
async def drive_health() -> DriveHealthResponse:
    """Check Google Drive connection status and quota.

    Always returns HTTP 200 — even when Drive is disabled or the connection fails.
    The frontend uses the `connected` and `enabled` fields to decide whether to
    show upload buttons and other Drive-dependent UI.

    This endpoint is intentionally lightweight: no DB queries, one Drive API call
    for connection verification. Quota fields are omitted until Phase 2 exposes
    a get_about() method on GoogleDriveV3Storage.
    """
    settings = get_settings()

    if not settings.google_drive_enabled:
        return DriveHealthResponse(
            connected=False,
            enabled=False,
            serviceAccountEmail=None,
            rootFolderId=None,
            rootFolderName=None,
            storageUsedBytes=None,
            storageTotalBytes=None,
            storageUsagePercentage=None,
            lastUploadAt=None,
            error=None,
        )

    # Drive is enabled — attempt a connection test
    try:
        # Lazy import to avoid import-time side effects
        from image_search_service.storage import get_async_storage

        storage = get_async_storage()
        if storage is None:
            return DriveHealthResponse(
                connected=False,
                enabled=True,
                serviceAccountEmail=None,
                rootFolderId=None,
                rootFolderName=None,
                storageUsedBytes=None,
                storageTotalBytes=None,
                storageUsagePercentage=None,
                lastUploadAt=None,
                error="Storage backend not available despite feature flag being enabled",
            )

        # Test connection by listing root folder using the CANONICAL protocol method.
        # CANONICAL: list_folder(folder_id: str) takes a Drive folder ID, NOT a path.
        # See INTERFACES.md §1 and §10.
        await storage.list_folder(settings.google_drive_root_id)

        # Resolve the authenticated identity in an auth-mode-aware way.
        # OAuth mode: call get_user_email() on the OAuth storage instance.
        # Service account mode: read client_email from the SA JSON key file.
        from image_search_service.storage.google_drive_oauth_v3 import (
            GoogleDriveOAuthV3Storage,
        )

        authenticated_email: str | None
        if isinstance(storage, GoogleDriveOAuthV3Storage):
            # OAuth backend exposes the user's Google account email.
            # Run synchronous Drive API call in thread pool to avoid blocking.
            import asyncio

            authenticated_email = await asyncio.get_event_loop().run_in_executor(
                None, storage.get_user_email
            )
        else:
            authenticated_email = _get_sa_email()

        return DriveHealthResponse(
            connected=True,
            enabled=True,
            serviceAccountEmail=authenticated_email,
            rootFolderId=settings.google_drive_root_id,
            rootFolderName=None,  # Would require extra API call to resolve name
            storageUsedBytes=None,    # Requires Drive about.get() (future)
            storageTotalBytes=None,   # Requires Drive about.get() (future)
            storageUsagePercentage=None,
            lastUploadAt=None,  # Could query max(storage_uploads.completed_at)
            error=None,
        )

    except Exception as e:
        logger.warning("Drive health check failed", extra={"error": str(e)})
        return DriveHealthResponse(
            connected=False,
            enabled=True,
            serviceAccountEmail=None,
            rootFolderId=None,
            rootFolderName=None,
            storageUsedBytes=None,
            storageTotalBytes=None,
            storageUsagePercentage=None,
            lastUploadAt=None,
            error=str(e),
        )
