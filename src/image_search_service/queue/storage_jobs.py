"""RQ background jobs for cloud storage upload operations.

FORK-SAFETY REQUIREMENTS FOR MACOS
==================================
All job functions in this module are executed by RQ work-horse subprocesses.
On macOS, these subprocesses are created via spawn() (not fork()), requiring:

1. SYNCHRONOUS database operations ONLY
   ✓ Use: get_sync_session() for all database access
   ✗ Don't use: async/await, asyncio, greenlets

2. No fork-unsafe libraries in the job hot-path.

3. Verified job functions:
   ✓ upload_person_photos_job() - Sync operations only

See worker.py for complete macOS fork-safety architecture.
"""

from __future__ import annotations

import logging
from typing import Any

from rq import get_current_job

from image_search_service.db.sync_operations import get_sync_session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_CHUNK_SIZE = 50
"""Number of assets per RQ job when chunking large batches."""


# ---------------------------------------------------------------------------
# Job function
# ---------------------------------------------------------------------------


def upload_person_photos_job(
    person_id: str,
    asset_ids: list[int],
    remote_base_path: str,
    batch_id: str,
    start_folder_id: str | None = None,
) -> dict[str, Any]:
    """Upload a person's photos to cloud storage (sync, RQ-safe).

    Creates upload records for the given assets, then processes all pending
    records in the batch.  The batch is idempotent: if some records already
    exist for this batch_id + asset_id pair they are silently skipped.

    Args:
        person_id: UUID string of the person whose photos are being uploaded.
        asset_ids: List of ImageAsset primary keys to upload.
        remote_base_path: Path of *name* segments beneath ``start_folder_id``,
                          e.g. ``"Jane Doe"`` or ``""`` (upload directly into
                          ``start_folder_id``).  Must NOT contain Drive folder IDs.
        batch_id: Caller-supplied batch identifier used for grouping and
                  resumption.
        start_folder_id: Drive folder ID to use as the root of the upload
                         hierarchy.  Folder creation for ``remote_base_path``
                         segments starts inside this folder.

    Returns:
        Dict with ``uploaded``, ``skipped``, ``failed``, ``errors`` counts
        plus ``batch_id`` for reference.
    """
    import uuid as _uuid

    from image_search_service.services.upload_service import UploadService
    from image_search_service.storage import get_storage

    rq_job = get_current_job()
    job_id = rq_job.id if rq_job else "local"

    logger.info(
        "Starting upload_person_photos_job",
        extra={
            "job_id": job_id,
            "batch_id": batch_id,
            "person_id": person_id,
            "asset_count": len(asset_ids),
            "remote_base_path": remote_base_path,
            "start_folder_id": start_folder_id,
        },
    )

    db = get_sync_session()
    try:
        storage = get_storage()
        if storage is None:
            logger.warning(
                "Google Drive not enabled — upload_person_photos_job is a no-op",
                extra={"batch_id": batch_id, "job_id": job_id},
            )
            return {
                "batch_id": batch_id,
                "uploaded": 0,
                "skipped": len(asset_ids),
                "failed": 0,
                "errors": ["Google Drive not enabled"],
            }

        service = UploadService(db=db, storage=storage)

        person_uuid = _uuid.UUID(person_id)

        # Idempotent: create_batch skips existing (batch_id, asset_id) pairs
        service.create_batch(
            person_id=person_uuid,
            asset_ids=asset_ids,
            remote_base_path=remote_base_path,
            batch_id=batch_id,
        )

        result = service.process_batch(
            batch_id=batch_id,
            remote_base_path=remote_base_path,
            start_folder_id=start_folder_id,
        )

        logger.info(
            "upload_person_photos_job complete",
            extra={
                "job_id": job_id,
                "batch_id": batch_id,
                "person_id": person_id,
                **{k: v for k, v in result.items() if k != "errors"},
            },
        )

        return {"batch_id": batch_id, **result}

    except Exception:
        logger.exception(
            "upload_person_photos_job failed",
            extra={"batch_id": batch_id, "person_id": person_id, "job_id": job_id},
        )
        raise
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Enqueue helpers
# ---------------------------------------------------------------------------


def enqueue_person_upload(
    person_id: str,
    asset_ids: list[int],
    remote_base_path: str,
    batch_id: str,
    start_folder_id: str | None = None,
) -> str:
    """Enqueue a single upload job for all assets in one call.

    Args:
        person_id: UUID string of the person.
        asset_ids: All asset IDs to upload.
        remote_base_path: Path of name segments to create beneath
                          ``start_folder_id``.
        batch_id: Batch identifier (must be unique per logical batch).
        start_folder_id: Drive folder ID to use as the upload root.

    Returns:
        RQ job ID string.
    """
    from image_search_service.queue.worker import QUEUE_LOW, get_queue

    queue = get_queue(QUEUE_LOW)
    job = queue.enqueue(
        "image_search_service.queue.storage_jobs.upload_person_photos_job",
        person_id=person_id,
        asset_ids=asset_ids,
        remote_base_path=remote_base_path,
        batch_id=batch_id,
        start_folder_id=start_folder_id,
        job_timeout="30m",
        result_ttl=3600,
    )
    logger.info(
        "Enqueued upload job",
        extra={
            "rq_job_id": job.id,
            "batch_id": batch_id,
            "person_id": person_id,
            "asset_count": len(asset_ids),
            "start_folder_id": start_folder_id,
        },
    )
    return str(job.id)


def enqueue_person_upload_chunked(
    person_id: str,
    asset_ids: list[int],
    remote_base_path: str,
    batch_id: str,
    chunk_size: int = BATCH_CHUNK_SIZE,
    start_folder_id: str | None = None,
) -> list[str]:
    """Enqueue multiple upload jobs by splitting assets into chunks.

    All chunks share the same ``batch_id`` so progress can be tracked
    holistically via ``UploadService.get_batch_status(batch_id)``.

    Args:
        person_id: UUID string of the person.
        asset_ids: All asset IDs to upload (may be very large).
        remote_base_path: Path of name segments to create beneath
                          ``start_folder_id``.
        batch_id: Shared batch identifier for all chunk jobs.
        chunk_size: Number of assets per chunk (default: ``BATCH_CHUNK_SIZE``).
        start_folder_id: Drive folder ID to use as the upload root.

    Returns:
        List of RQ job ID strings, one per chunk.
    """
    if not asset_ids:
        return []

    job_ids: list[str] = []
    chunks = [asset_ids[i : i + chunk_size] for i in range(0, len(asset_ids), chunk_size)]

    for chunk in chunks:
        job_id = enqueue_person_upload(
            person_id=person_id,
            asset_ids=chunk,
            remote_base_path=remote_base_path,
            batch_id=batch_id,
            start_folder_id=start_folder_id,
        )
        job_ids.append(job_id)

    logger.info(
        "Enqueued chunked upload jobs",
        extra={
            "batch_id": batch_id,
            "person_id": person_id,
            "total_assets": len(asset_ids),
            "chunks": len(chunks),
            "chunk_size": chunk_size,
            "start_folder_id": start_folder_id,
        },
    )
    return job_ids
