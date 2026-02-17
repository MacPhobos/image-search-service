"""Unit tests for UploadService.

All tests use in-memory SQLite (no external dependencies) and
FakeStorageBackend (no real Google API calls).

Test categories:
  1. Batch creation (3)
  2. Partial failure & resume (2)
  3. Progress tracking (4)
  4. Cancel operations (2)
  5. Fake storage integration (2)
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from image_search_service.db.models import (
    Base,
    FaceInstance,
    ImageAsset,
    Person,
    PersonStatus,
    StorageUpload,
    StorageUploadStatus,
    TrainingStatus,
)
from image_search_service.services.upload_service import BatchProgress, UploadService
from image_search_service.storage.base import UploadResult

# ---------------------------------------------------------------------------
# FakeStorageBackend
# ---------------------------------------------------------------------------


class FakeStorageBackend:
    """In-memory storage backend for tests.

    Implements the StorageBackend protocol without any real I/O.
    Folders and files are stored in plain dicts.
    """

    def __init__(self) -> None:
        # Folder tree: parent_id -> {name -> folder_id}
        self._folders: dict[str | None, dict[str, str]] = defaultdict(dict)
        # Uploaded files: file_id -> (filename, content, mime_type, folder_id)
        self._files: dict[str, tuple[str, bytes, str, str | None]] = {}
        self._next_id = 1

    def _new_id(self) -> str:
        fid = f"fake_{self._next_id}"
        self._next_id += 1
        return fid

    def upload_file(
        self,
        content: bytes,
        filename: str,
        mime_type: str,
        folder_id: str | None = None,
    ) -> UploadResult:
        fid = self._new_id()
        self._files[fid] = (filename, content, mime_type, folder_id)
        return UploadResult(
            file_id=fid,
            name=filename,
            size=len(content),
            mime_type=mime_type,
        )

    def create_folder(self, name: str, parent_id: str | None = None) -> str:
        # Idempotent: return existing folder ID if already present
        existing = self._folders[parent_id].get(name)
        if existing:
            return existing
        fid = self._new_id()
        self._folders[parent_id][name] = fid
        return fid

    def file_exists(self, file_id: str) -> bool:
        return file_id in self._files or any(
            file_id in children.values() for children in self._folders.values()
        )

    def list_folder(self, folder_id: str) -> list[Any]:
        return []

    def delete_file(self, file_id: str, *, trash: bool = True) -> None:
        self._files.pop(file_id, None)

    # Test helpers
    def file_count(self) -> int:
        return len(self._files)

    def folder_ids_for(self, parent_id: str | None) -> dict[str, str]:
        return dict(self._folders[parent_id])


# ---------------------------------------------------------------------------
# FailAfterNUploads — injects failures for storage tests
# ---------------------------------------------------------------------------


class FailAfterNUploads:
    """Wraps a FakeStorageBackend and fails upload_file after N successes."""

    def __init__(self, inner: FakeStorageBackend, succeed_count: int) -> None:
        self._inner = inner
        self._succeed_count = succeed_count
        self._call_count = 0

    def upload_file(
        self,
        content: bytes,
        filename: str,
        mime_type: str,
        folder_id: str | None = None,
    ) -> UploadResult:
        self._call_count += 1
        if self._call_count > self._succeed_count:
            raise RuntimeError(f"Simulated upload failure on call {self._call_count}")
        return self._inner.upload_file(
            content=content, filename=filename, mime_type=mime_type, folder_id=folder_id
        )

    # Delegate non-upload methods to inner backend
    def create_folder(self, name: str, parent_id: str | None = None) -> str:
        return self._inner.create_folder(name, parent_id)

    def file_exists(self, file_id: str) -> bool:
        return self._inner.file_exists(file_id)

    def list_folder(self, folder_id: str) -> list[Any]:
        return self._inner.list_folder(folder_id)

    def delete_file(self, file_id: str, *, trash: bool = True) -> None:
        self._inner.delete_file(file_id, trash=trash)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_engine():
    """In-memory SQLite engine with all tables created."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture
def db_session(db_engine) -> Session:
    """Synchronous SQLite session, rolled back after each test."""
    session = Session(db_engine)
    yield session
    session.rollback()
    session.close()


@pytest.fixture
def fake_storage() -> FakeStorageBackend:
    return FakeStorageBackend()


@pytest.fixture
def person(db_session: Session) -> Person:
    """A Person with no faces (base fixture)."""
    p = Person(
        id=uuid.uuid4(),
        name="Jane Doe",
        status=PersonStatus.ACTIVE,
    )
    db_session.add(p)
    db_session.commit()
    return p


@pytest.fixture
def real_image_files(tmp_path: Path) -> list[Path]:
    """Create a handful of real files on disk for upload tests."""
    files: list[Path] = []
    for i in range(3):
        p = tmp_path / f"photo_{i}.jpg"
        p.write_bytes(b"JFIF" + bytes([i]) * 100)
        files.append(p)
    return files


@pytest.fixture
def assets_with_files(db_session: Session, real_image_files: list[Path]) -> list[ImageAsset]:
    """ImageAssets that point to actual files on disk."""
    assets = []
    for path in real_image_files:
        a = ImageAsset(
            path=str(path),
            training_status=TrainingStatus.PENDING.value,
            mime_type="image/jpeg",
            file_size=path.stat().st_size,
        )
        db_session.add(a)
        assets.append(a)
    db_session.commit()
    for a in assets:
        db_session.refresh(a)
    return assets


@pytest.fixture
def assets_no_files(db_session: Session) -> list[ImageAsset]:
    """ImageAssets whose paths do NOT exist on disk (for missing-file tests)."""
    assets = []
    for i in range(2):
        a = ImageAsset(
            path=f"/nonexistent/path/photo_{i}.jpg",
            training_status=TrainingStatus.PENDING.value,
            mime_type="image/jpeg",
        )
        db_session.add(a)
        assets.append(a)
    db_session.commit()
    for a in assets:
        db_session.refresh(a)
    return assets


@pytest.fixture
def person_with_faces(db_session: Session, assets_with_files: list[ImageAsset]) -> Person:
    """A Person with FaceInstances linking to real-file assets."""
    p = Person(
        id=uuid.uuid4(),
        name="John Smith",
        status=PersonStatus.ACTIVE,
    )
    db_session.add(p)
    db_session.flush()

    for i, asset in enumerate(assets_with_files):
        face = FaceInstance(
            id=uuid.uuid4(),
            asset_id=asset.id,
            person_id=p.id,
            bbox_x=10 * i,
            bbox_y=10 * i,
            bbox_w=50,
            bbox_h=50,
            detection_confidence=0.9,
            qdrant_point_id=uuid.uuid4(),
        )
        db_session.add(face)

    db_session.commit()
    db_session.refresh(p)
    return p


def _make_service(
    db_session: Session, storage: FakeStorageBackend | FailAfterNUploads
) -> UploadService:
    return UploadService(db=db_session, storage=storage)  # type: ignore[arg-type]


# ===========================================================================
# 1. BATCH CREATION
# ===========================================================================


class TestBatchCreation:
    def test_create_batch_creates_pending_records(
        self,
        db_session: Session,
        fake_storage: FakeStorageBackend,
        person: Person,
        assets_with_files: list[ImageAsset],
    ) -> None:
        """create_batch should insert one PENDING row per asset."""
        service = _make_service(db_session, fake_storage)
        asset_ids = [a.id for a in assets_with_files]

        batch_id = service.create_batch(
            person_id=person.id,
            asset_ids=asset_ids,
            remote_base_path="people/Jane Doe",
        )

        rows = db_session.query(StorageUpload).filter_by(batch_id=batch_id).all()
        assert len(rows) == len(asset_ids)
        assert all(r.status == StorageUploadStatus.PENDING.value for r in rows)
        assert all(r.person_id == person.id for r in rows)

    def test_create_batch_is_idempotent_when_called_twice(
        self,
        db_session: Session,
        fake_storage: FakeStorageBackend,
        person: Person,
        assets_with_files: list[ImageAsset],
    ) -> None:
        """Calling create_batch twice with the same batch_id must not create duplicates."""
        service = _make_service(db_session, fake_storage)
        asset_ids = [a.id for a in assets_with_files]
        batch_id = "idempotent-batch-001"

        service.create_batch(
            person_id=person.id,
            asset_ids=asset_ids,
            remote_base_path="people/Jane Doe",
            batch_id=batch_id,
        )
        service.create_batch(
            person_id=person.id,
            asset_ids=asset_ids,
            remote_base_path="people/Jane Doe",
            batch_id=batch_id,
        )

        rows = db_session.query(StorageUpload).filter_by(batch_id=batch_id).all()
        assert len(rows) == len(asset_ids), "Duplicate rows created on second call"

    def test_create_batch_auto_generates_batch_id(
        self,
        db_session: Session,
        fake_storage: FakeStorageBackend,
        person: Person,
        assets_with_files: list[ImageAsset],
    ) -> None:
        """When batch_id is omitted, a non-empty string is generated automatically."""
        service = _make_service(db_session, fake_storage)
        asset_ids = [a.id for a in assets_with_files]

        batch_id = service.create_batch(
            person_id=person.id,
            asset_ids=asset_ids,
            remote_base_path="people/Jane Doe",
        )

        assert isinstance(batch_id, str)
        assert len(batch_id) > 0
        rows = db_session.query(StorageUpload).filter_by(batch_id=batch_id).all()
        assert len(rows) == len(asset_ids)


# ===========================================================================
# 2. PARTIAL FAILURE & RESUME
# ===========================================================================


class TestPartialFailureAndResume:
    def test_partial_failure_records_completed_files(
        self,
        db_session: Session,
        fake_storage: FakeStorageBackend,
        person: Person,
        assets_with_files: list[ImageAsset],
    ) -> None:
        """When one upload fails, earlier successful uploads remain COMPLETED."""
        # Allow only the first upload to succeed
        failing_storage = FailAfterNUploads(inner=fake_storage, succeed_count=1)
        service = _make_service(db_session, failing_storage)
        asset_ids = [a.id for a in assets_with_files]
        batch_id = "partial-fail-batch"

        service.create_batch(
            person_id=person.id,
            asset_ids=asset_ids,
            remote_base_path="people/Jane Doe",
            batch_id=batch_id,
        )
        result = service.process_batch(batch_id, remote_base_path="people/Jane Doe")

        assert result["uploaded"] == 1
        assert result["failed"] >= 1

        rows = db_session.query(StorageUpload).filter_by(batch_id=batch_id).all()
        statuses = {r.status for r in rows}
        assert StorageUploadStatus.COMPLETED.value in statuses
        assert StorageUploadStatus.FAILED.value in statuses

    def test_resume_batch_only_processes_incomplete_records(
        self,
        db_session: Session,
        fake_storage: FakeStorageBackend,
        person: Person,
        assets_with_files: list[ImageAsset],
    ) -> None:
        """resume_batch should retry only FAILED records, leaving COMPLETED untouched."""
        # First pass: fail all uploads
        always_failing = FailAfterNUploads(inner=fake_storage, succeed_count=0)
        service_fail = _make_service(db_session, always_failing)
        asset_ids = [a.id for a in assets_with_files]
        batch_id = "resume-test-batch"

        service_fail.create_batch(
            person_id=person.id,
            asset_ids=asset_ids,
            remote_base_path="people/Jane Doe",
            batch_id=batch_id,
        )
        service_fail.process_batch(batch_id, remote_base_path="people/Jane Doe")

        # Confirm all are FAILED
        fail_rows = db_session.query(StorageUpload).filter_by(batch_id=batch_id).all()
        assert all(r.status == StorageUploadStatus.FAILED.value for r in fail_rows)

        # Resume with working storage
        service_ok = _make_service(db_session, fake_storage)
        resume_result = service_ok.resume_batch(batch_id, remote_base_path="people/Jane Doe")

        assert resume_result["uploaded"] == len(asset_ids)
        assert resume_result["failed"] == 0

        # All records should now be COMPLETED
        rows = db_session.query(StorageUpload).filter_by(batch_id=batch_id).all()
        assert all(r.status == StorageUploadStatus.COMPLETED.value for r in rows)


# ===========================================================================
# 3. PROGRESS TRACKING
# ===========================================================================


class TestProgressTracking:
    def test_get_batch_status_returns_all_pending_when_not_processed(
        self,
        db_session: Session,
        fake_storage: FakeStorageBackend,
        person: Person,
        assets_with_files: list[ImageAsset],
    ) -> None:
        """Before processing, every record should be PENDING."""
        service = _make_service(db_session, fake_storage)
        asset_ids = [a.id for a in assets_with_files]
        batch_id = service.create_batch(
            person_id=person.id,
            asset_ids=asset_ids,
            remote_base_path="people/Jane Doe",
        )

        progress = service.get_batch_status(batch_id)

        assert isinstance(progress, BatchProgress)
        assert progress.total == len(asset_ids)
        assert progress.pending == len(asset_ids)
        assert progress.completed == 0
        assert progress.failed == 0
        assert progress.percentage == 0.0
        assert not progress.is_complete
        assert progress.status == "in_progress"

    def test_get_batch_status_reflects_partial_completion(
        self,
        db_session: Session,
        fake_storage: FakeStorageBackend,
        person: Person,
        assets_with_files: list[ImageAsset],
    ) -> None:
        """After 1-of-3 uploads succeed, progress should reflect partial state."""
        failing_storage = FailAfterNUploads(inner=fake_storage, succeed_count=1)
        service = _make_service(db_session, failing_storage)
        asset_ids = [a.id for a in assets_with_files]
        batch_id = "progress-partial-batch"

        service.create_batch(
            person_id=person.id,
            asset_ids=asset_ids,
            remote_base_path="people/Jane Doe",
            batch_id=batch_id,
        )
        service.process_batch(batch_id, remote_base_path="people/Jane Doe")

        progress = service.get_batch_status(batch_id)

        assert progress.completed == 1
        assert progress.failed >= 1
        assert progress.total == len(asset_ids)
        assert 0 < progress.percentage < 100

    def test_get_batch_status_shows_complete_when_all_uploaded(
        self,
        db_session: Session,
        fake_storage: FakeStorageBackend,
        person: Person,
        assets_with_files: list[ImageAsset],
    ) -> None:
        """After all uploads succeed, BatchProgress.is_complete should be True."""
        service = _make_service(db_session, fake_storage)
        asset_ids = [a.id for a in assets_with_files]
        batch_id = service.create_batch(
            person_id=person.id,
            asset_ids=asset_ids,
            remote_base_path="people/Jane Doe",
        )
        service.process_batch(batch_id, remote_base_path="people/Jane Doe")

        progress = service.get_batch_status(batch_id)

        assert progress.is_complete
        assert progress.completed == len(asset_ids)
        assert progress.percentage == 100.0
        assert progress.status == "completed"

    def test_get_batch_status_returns_zero_total_for_unknown_batch(
        self,
        db_session: Session,
        fake_storage: FakeStorageBackend,
    ) -> None:
        """Querying a non-existent batch should return an empty BatchProgress."""
        service = _make_service(db_session, fake_storage)
        progress = service.get_batch_status("nonexistent-batch-xyz")

        assert progress.total == 0
        assert progress.percentage == 0.0
        assert progress.is_complete  # Empty batch: in_progress==0 → is_complete is True
        assert progress.status == "completed"  # Empty batch falls through to "completed"


# ===========================================================================
# 4. CANCEL OPERATIONS
# ===========================================================================


class TestCancelOperations:
    def test_cancel_batch_marks_pending_as_cancelled(
        self,
        db_session: Session,
        fake_storage: FakeStorageBackend,
        person: Person,
        assets_with_files: list[ImageAsset],
    ) -> None:
        """cancel_batch should transition all PENDING records to CANCELLED."""
        service = _make_service(db_session, fake_storage)
        asset_ids = [a.id for a in assets_with_files]
        batch_id = service.create_batch(
            person_id=person.id,
            asset_ids=asset_ids,
            remote_base_path="people/Jane Doe",
        )

        cancelled_count = service.cancel_batch(batch_id)

        assert cancelled_count == len(asset_ids)
        rows = db_session.query(StorageUpload).filter_by(batch_id=batch_id).all()
        assert all(r.status == StorageUploadStatus.CANCELLED.value for r in rows)

    def test_cancel_batch_does_not_affect_completed_records(
        self,
        db_session: Session,
        fake_storage: FakeStorageBackend,
        person: Person,
        assets_with_files: list[ImageAsset],
    ) -> None:
        """cancel_batch must leave already-COMPLETED records untouched."""
        # Allow exactly 1 upload to succeed before failing the rest
        failing_storage = FailAfterNUploads(inner=fake_storage, succeed_count=1)
        service = _make_service(db_session, failing_storage)
        asset_ids = [a.id for a in assets_with_files]
        batch_id = "cancel-partial-batch"

        service.create_batch(
            person_id=person.id,
            asset_ids=asset_ids,
            remote_base_path="people/Jane Doe",
            batch_id=batch_id,
        )
        service.process_batch(batch_id, remote_base_path="people/Jane Doe")

        # Verify we have 1 COMPLETED before cancellation
        completed_before = (
            db_session.query(StorageUpload)
            .filter_by(batch_id=batch_id, status=StorageUploadStatus.COMPLETED.value)
            .count()
        )
        assert completed_before == 1

        # Cancel remaining (FAILED records need explicit resume; PENDING is what we cancel)
        service.cancel_batch(batch_id)

        completed_after = (
            db_session.query(StorageUpload)
            .filter_by(batch_id=batch_id, status=StorageUploadStatus.COMPLETED.value)
            .count()
        )
        assert completed_after == 1, "cancel_batch must not affect COMPLETED records"


# ===========================================================================
# 5. FAKE STORAGE INTEGRATION
# ===========================================================================


class TestFakeStorageIntegration:
    def test_uploaded_files_appear_in_db_records(
        self,
        db_session: Session,
        fake_storage: FakeStorageBackend,
        person: Person,
        assets_with_files: list[ImageAsset],
    ) -> None:
        """After a successful process_batch, DB records should have remote_file_id set."""
        service = _make_service(db_session, fake_storage)
        asset_ids = [a.id for a in assets_with_files]
        batch_id = service.create_batch(
            person_id=person.id,
            asset_ids=asset_ids,
            remote_base_path="people/Jane Doe",
        )
        result = service.process_batch(batch_id, remote_base_path="people/Jane Doe")

        assert result["uploaded"] == len(asset_ids)
        assert result["failed"] == 0

        rows = db_session.query(StorageUpload).filter_by(batch_id=batch_id).all()
        for row in rows:
            assert row.status == StorageUploadStatus.COMPLETED.value
            assert row.remote_file_id is not None
            assert row.file_size_bytes is not None and row.file_size_bytes > 0
            assert row.completed_at is not None

        # Verify files actually landed in FakeStorageBackend
        assert fake_storage.file_count() == len(asset_ids)

    def test_error_messages_recorded_on_failure(
        self,
        db_session: Session,
        fake_storage: FakeStorageBackend,
        person: Person,
        assets_no_files: list[ImageAsset],
    ) -> None:
        """When local files are missing, the error_message column should be populated."""
        service = _make_service(db_session, fake_storage)
        asset_ids = [a.id for a in assets_no_files]
        batch_id = service.create_batch(
            person_id=person.id,
            asset_ids=asset_ids,
            remote_base_path="people/Jane Doe",
        )
        result = service.process_batch(batch_id, remote_base_path="people/Jane Doe")

        assert result["failed"] == len(asset_ids)
        assert result["uploaded"] == 0

        rows = db_session.query(StorageUpload).filter_by(batch_id=batch_id).all()
        for row in rows:
            assert row.status == StorageUploadStatus.FAILED.value
            assert row.error_message is not None
            assert len(row.error_message) > 0


# ===========================================================================
# 6. MISSING ASSET EDGE CASE
# ===========================================================================


class TestMissingAssetEdgeCase:
    def test_process_batch_marks_deleted_asset_as_failed(
        self,
        db_session: Session,
        fake_storage: FakeStorageBackend,
        person: Person,
        assets_with_files: list[ImageAsset],
    ) -> None:
        """When an ImageAsset row is deleted after batch creation, process_batch
        must mark that record FAILED (with an error message) and continue
        processing remaining assets normally.

        Scenario:
          - Batch created with 3 assets
          - Middle asset's DB row is deleted (simulates hard delete)
          - process_batch runs: first and third should COMPLETE, middle FAIL
        """
        service = _make_service(db_session, fake_storage)
        asset_ids = [a.id for a in assets_with_files]
        assert len(asset_ids) >= 2, "Fixture must provide at least 2 assets"

        batch_id = service.create_batch(
            person_id=person.id,
            asset_ids=asset_ids,
            remote_base_path="people/Jane Doe",
        )

        # Delete the first asset from the DB to simulate a hard delete
        deleted_asset = assets_with_files[0]
        deleted_asset_id = deleted_asset.id
        db_session.delete(deleted_asset)
        db_session.commit()

        # process_batch should handle the missing row gracefully
        result = service.process_batch(batch_id, remote_base_path="people/Jane Doe")

        # One asset was deleted → should count as failed
        assert result["failed"] == 1
        # Remaining assets (the ones still in the DB) should succeed
        expected_uploaded = len(asset_ids) - 1
        assert result["uploaded"] == expected_uploaded

        # The deleted asset's upload record should be FAILED with an error message
        deleted_upload_row = (
            db_session.query(StorageUpload)
            .filter_by(batch_id=batch_id, asset_id=deleted_asset_id)
            .one()
        )
        assert deleted_upload_row.status == StorageUploadStatus.FAILED.value
        assert deleted_upload_row.error_message is not None
        assert "not found" in deleted_upload_row.error_message.lower()

        # Other assets' upload records should all be COMPLETED
        other_rows = (
            db_session.query(StorageUpload)
            .filter(
                StorageUpload.batch_id == batch_id,
                StorageUpload.asset_id != deleted_asset_id,
            )
            .all()
        )
        assert len(other_rows) == expected_uploaded
        for row in other_rows:
            assert row.status == StorageUploadStatus.COMPLETED.value
            assert row.remote_file_id is not None
