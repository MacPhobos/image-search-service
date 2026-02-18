"""Upload service for cloud storage batch operations.

Manages the lifecycle of batch uploads: creation, processing, progress tracking,
resumption after failure, and cancellation.  The service is sync-only because it
runs inside RQ background jobs (fork-safety requirement).

Usage (in an RQ job)::

    from image_search_service.services.upload_service import UploadService

    service = UploadService(db=session, storage=storage_backend)
    batch_id = service.create_batch(
        person_id=person_uuid,
        asset_ids=[1, 2, 3],
        remote_base_path="people/Jane Doe",
    )
    result = service.process_batch(batch_id, remote_base_path="people/Jane Doe")
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from image_search_service.core.logging import get_logger
from image_search_service.db.models import (
    FaceInstance,
    ImageAsset,
    StorageUpload,
    StorageUploadStatus,
)
from image_search_service.storage.base import StorageBackend

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _ensure_path_exists(
    storage: StorageBackend,
    path: str,
    start_parent_id: str | None = None,
) -> str:
    """Create nested folders one segment at a time and return the leaf folder ID.

    Each path segment is created via ``storage.create_folder()``, which is
    idempotent — if the folder already exists the provider returns its ID.

    Args:
        storage: Sync storage backend instance.
        path: Forward-slash separated path of *name* segments,
              e.g. ``"Jane Doe"`` or ``"subdir/Jane Doe"``.
              Leading/trailing slashes are stripped before splitting.
              Must NOT contain Drive folder IDs — use ``start_parent_id`` for that.
        start_parent_id: Provider-specific folder ID to start the hierarchy from.
              When provided, the first segment is created inside this folder
              (i.e. it is used as the initial ``parent_id``).
              When ``None`` (default), folders are created under the Drive root.

    Returns:
        Provider-specific folder ID of the deepest (leaf) folder.
        Returns ``start_parent_id`` (or ``""``) when ``path`` is empty or
        root-only (``"/"``).  The returned ID should be passed directly to
        ``upload_file(folder_id=...)``.
    """
    segments = [s for s in path.strip("/").split("/") if s]
    if not segments:
        # No name segments to create — use the provided starting folder as-is.
        return start_parent_id or ""
    parent_id: str | None = start_parent_id
    folder_id: str = start_parent_id or ""
    for segment in segments:
        folder_id = storage.create_folder(segment, parent_id=parent_id)
        parent_id = folder_id
    return folder_id


# ---------------------------------------------------------------------------
# BatchProgress
# ---------------------------------------------------------------------------


class BatchProgress:
    """Snapshot of a batch's upload progress.

    Plain class (not a dataclass) to allow computed properties.
    """

    def __init__(
        self,
        batch_id: str,
        total: int,
        pending: int,
        uploading: int,
        completed: int,
        failed: int,
        cancelled: int,
    ) -> None:
        self.batch_id = batch_id
        self.total = total
        self.pending = pending
        self.uploading = uploading
        self.completed = completed
        self.failed = failed
        self.cancelled = cancelled

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def in_progress(self) -> int:
        """Number of records actively uploading or still pending."""
        return self.pending + self.uploading

    @property
    def percentage(self) -> float:
        """Completion percentage (0-100).  Returns 0 when total is 0."""
        if self.total == 0:
            return 0.0
        return round((self.completed / self.total) * 100, 2)

    @property
    def is_complete(self) -> bool:
        """True when every record has reached a terminal state (no in-progress records)."""
        return self.in_progress == 0

    @property
    def status(self) -> str:
        """Canonical batch status string per INTERFACES.md §2.4.

        Valid values: ``"in_progress"``, ``"partial_failure"``, ``"failed"``,
        ``"cancelled"``, ``"completed"``.
        """
        if self.in_progress > 0:
            return "in_progress"
        if self.failed > 0 and self.completed > 0:
            return "partial_failure"
        if self.failed > 0 and self.completed == 0:
            return "failed"
        if self.total > 0 and self.cancelled == self.total:
            return "cancelled"
        return "completed"

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<BatchProgress(batch_id={self.batch_id!r}, total={self.total}, "
            f"completed={self.completed}, failed={self.failed}, "
            f"status={self.status!r})>"
        )


# ---------------------------------------------------------------------------
# UploadService
# ---------------------------------------------------------------------------


class UploadService:
    """Orchestrates cloud-storage batch upload operations.

    All methods are synchronous (designed for RQ background jobs).

    Args:
        db: SQLAlchemy sync session.
        storage: Storage backend implementing the ``StorageBackend`` protocol.
    """

    def __init__(self, db: Session, storage: StorageBackend) -> None:
        self._db = db
        self._storage = storage

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_batch(
        self,
        person_id: uuid.UUID,
        asset_ids: list[int],
        remote_base_path: str,
        batch_id: str | None = None,
    ) -> str:
        """Create upload records for each asset in the batch.

        Idempotent: existing (batch_id, asset_id) pairs are skipped.

        Args:
            person_id: UUID of the person whose photos are being uploaded.
            asset_ids: List of ImageAsset primary keys to include.
            remote_base_path: Virtual path prefix for the remote destination,
                              e.g. ``"people/Jane Doe"``.
            batch_id: Caller-supplied batch identifier.  If omitted, a UUID4
                      hex string is generated automatically.

        Returns:
            The batch_id (generated or caller-supplied).
        """
        if batch_id is None:
            batch_id = uuid.uuid4().hex

        if not asset_ids:
            logger.info(
                "create_batch called with empty asset list",
                extra={"batch_id": batch_id, "person_id": str(person_id)},
            )
            return batch_id

        # Find which assets already have records for this batch (idempotency)
        existing_stmt = select(StorageUpload.asset_id).where(
            StorageUpload.batch_id == batch_id,
            StorageUpload.asset_id.in_(asset_ids),
        )
        existing_asset_ids: set[int] = {
            row[0] for row in self._db.execute(existing_stmt).fetchall()
        }

        new_records = [
            StorageUpload(
                batch_id=batch_id,
                asset_id=asset_id,
                person_id=person_id,
                remote_path=remote_base_path,
                status=StorageUploadStatus.PENDING.value,
                provider="google_drive",
            )
            for asset_id in asset_ids
            if asset_id not in existing_asset_ids
        ]

        if new_records:
            self._db.add_all(new_records)
            self._db.commit()
            logger.info(
                "Batch created",
                extra={
                    "batch_id": batch_id,
                    "person_id": str(person_id),
                    "new_records": len(new_records),
                    "skipped_existing": len(existing_asset_ids),
                    "total_requested": len(asset_ids),
                },
            )
        else:
            logger.info(
                "Batch creation no-op (all assets already present)",
                extra={
                    "batch_id": batch_id,
                    "person_id": str(person_id),
                    "total_requested": len(asset_ids),
                },
            )

        return batch_id

    def process_batch(
        self,
        batch_id: str,
        remote_base_path: str,
        start_folder_id: str | None = None,
    ) -> dict[str, object]:
        """Process all pending/failed records in a batch.

        Ensures the remote folder hierarchy exists once before iterating
        individual files.  Per-file failures do NOT abort the batch —
        the error is recorded in the row and processing continues.

        Args:
            batch_id: Batch to process.
            remote_base_path: Path of *name* segments to create beneath
                              ``start_folder_id``, e.g. ``"Jane Doe"`` or ``""``.
                              Must NOT contain Drive folder IDs.
            start_folder_id: Drive folder ID to use as the root of the path
                             hierarchy.  When provided, folder creation starts
                             inside this folder rather than at the Drive root.
                             Must match the value used in ``create_batch()``.

        Returns:
            Dict with keys: ``uploaded``, ``skipped``, ``failed``, ``errors``.
        """
        # Ensure the remote folder tree exists once for the whole batch
        try:
            folder_id = _ensure_path_exists(
                self._storage, remote_base_path, start_parent_id=start_folder_id
            )
        except Exception as e:
            logger.error(
                "Failed to create remote directory",
                extra={
                    "batch_id": batch_id,
                    "remote_base_path": remote_base_path,
                    "error": str(e),
                },
            )
            raise

        # Load all processable records
        stmt = select(StorageUpload).where(
            StorageUpload.batch_id == batch_id,
            StorageUpload.status.in_(
                [StorageUploadStatus.PENDING.value, StorageUploadStatus.FAILED.value]
            ),
        )
        records = list(self._db.execute(stmt).scalars())

        uploaded = 0
        skipped = 0
        failed = 0
        errors: list[str] = []

        for record in records:
            try:
                self._process_single_upload(record, folder_id)
                uploaded += 1
            except Exception as exc:
                failed += 1
                errors.append(f"asset_id={record.asset_id}: {exc}")
                logger.warning(
                    "Upload failed for asset",
                    extra={
                        "batch_id": batch_id,
                        "asset_id": record.asset_id,
                        "error": str(exc),
                    },
                )

        logger.info(
            "Batch processing complete",
            extra={
                "batch_id": batch_id,
                "uploaded": uploaded,
                "skipped": skipped,
                "failed": failed,
            },
        )
        return {
            "uploaded": uploaded,
            "skipped": skipped,
            "failed": failed,
            "errors": errors,
        }

    def get_batch_status(self, batch_id: str) -> BatchProgress:
        """Return a BatchProgress snapshot for the given batch.

        Uses a GROUP BY query for efficient counting rather than loading all rows.

        Args:
            batch_id: Batch to inspect.

        Returns:
            BatchProgress instance (total=0 for unknown batches).
        """
        stmt = (
            select(StorageUpload.status, func.count(StorageUpload.id).label("cnt"))
            .where(StorageUpload.batch_id == batch_id)
            .group_by(StorageUpload.status)
        )
        rows = self._db.execute(stmt).fetchall()

        counts: dict[str, int] = {row[0]: row[1] for row in rows}
        total = sum(counts.values())

        return BatchProgress(
            batch_id=batch_id,
            total=total,
            pending=counts.get(StorageUploadStatus.PENDING.value, 0),
            uploading=counts.get(StorageUploadStatus.UPLOADING.value, 0),
            completed=counts.get(StorageUploadStatus.COMPLETED.value, 0),
            failed=counts.get(StorageUploadStatus.FAILED.value, 0),
            cancelled=counts.get(StorageUploadStatus.CANCELLED.value, 0),
        )

    def resume_batch(
        self,
        batch_id: str,
        remote_base_path: str,
        start_folder_id: str | None = None,
    ) -> dict[str, object]:
        """Resume a partially-failed batch by resetting FAILED records to PENDING.

        Args:
            batch_id: Batch to resume.
            remote_base_path: Path of name segments (same as original).
            start_folder_id: Drive folder ID root (same as original).

        Returns:
            Same dict as ``process_batch()``.
        """
        # Reset failed records so process_batch picks them up
        failed_stmt = select(StorageUpload).where(
            StorageUpload.batch_id == batch_id,
            StorageUpload.status == StorageUploadStatus.FAILED.value,
        )
        failed_records = list(self._db.execute(failed_stmt).scalars())

        reset_count = 0
        for record in failed_records:
            record.status = StorageUploadStatus.PENDING.value
            record.error_message = None
            reset_count += 1

        if reset_count:
            self._db.commit()
            logger.info(
                "Reset failed records for resume",
                extra={"batch_id": batch_id, "reset_count": reset_count},
            )

        return self.process_batch(batch_id, remote_base_path, start_folder_id=start_folder_id)

    def cancel_batch(self, batch_id: str) -> int:
        """Mark all pending/uploading records as CANCELLED.

        Completed records are intentionally left untouched.

        Args:
            batch_id: Batch to cancel.

        Returns:
            Number of records cancelled.
        """
        stmt = select(StorageUpload).where(
            StorageUpload.batch_id == batch_id,
            StorageUpload.status.in_(
                [
                    StorageUploadStatus.PENDING.value,
                    StorageUploadStatus.UPLOADING.value,
                ]
            ),
        )
        records = list(self._db.execute(stmt).scalars())

        for record in records:
            record.status = StorageUploadStatus.CANCELLED.value

        if records:
            self._db.commit()

        logger.info(
            "Batch cancelled",
            extra={"batch_id": batch_id, "cancelled_count": len(records)},
        )
        return len(records)

    def get_person_asset_ids(self, person_id: uuid.UUID) -> list[int]:
        """Return distinct asset IDs for all faces assigned to a person.

        Joins FaceInstance → ImageAsset and returns DISTINCT asset_id values
        ordered by id for deterministic results.

        Args:
            person_id: Person UUID.

        Returns:
            Sorted list of distinct asset primary keys.
        """
        stmt = (
            select(FaceInstance.asset_id)
            .where(FaceInstance.person_id == person_id)
            .distinct()
            .order_by(FaceInstance.asset_id)
        )
        return [row[0] for row in self._db.execute(stmt).fetchall()]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_single_upload(self, record: StorageUpload, folder_id: str) -> None:
        """Upload one file to the remote storage provider.

        Updates the record to UPLOADING before starting, then COMPLETED on
        success.  On any error, sets status to FAILED and records the message.
        Commits the session after each successful upload so progress is
        durable even if a later file fails.

        Args:
            record: StorageUpload ORM instance (must be attached to self._db).
            folder_id: Pre-resolved provider folder ID returned by
                       ``_ensure_path_exists()``.

        Raises:
            FileNotFoundError: If the local file is missing.
            Exception: Any storage backend error (upload failure, quota, etc.).
        """
        record.status = StorageUploadStatus.UPLOADING.value
        self._db.commit()

        try:
            # Load the associated asset to get the local file path
            asset = self._db.get(ImageAsset, record.asset_id)
            if asset is None:
                raise ValueError(f"ImageAsset {record.asset_id} not found in database")

            local_path = asset.path
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"Local file not found: {local_path}")

            with open(local_path, "rb") as fh:
                content = fh.read()

            filename = os.path.basename(local_path)
            mime_type = asset.mime_type or "application/octet-stream"

            result = self._storage.upload_file(
                content=content,
                filename=filename,
                mime_type=mime_type,
                folder_id=folder_id,
            )

            # Mark success and store remote metadata
            record.status = StorageUploadStatus.COMPLETED.value
            record.remote_file_id = result.file_id
            record.file_size_bytes = result.size
            record.completed_at = datetime.now(UTC)
            self._db.commit()

            logger.debug(
                "File uploaded successfully",
                extra={
                    "batch_id": record.batch_id,
                    "asset_id": record.asset_id,
                    "remote_file_id": result.file_id,
                    "size_bytes": result.size,
                },
            )

        except Exception as exc:
            record.status = StorageUploadStatus.FAILED.value
            record.error_message = str(exc)
            self._db.commit()
            raise
