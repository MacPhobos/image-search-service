"""Comprehensive tests for queue/training_jobs.py.

Tests for training job functions: train_session, train_single_asset, and _build_evidence_metadata.
Covers success cases, error handling, cancellation, and metadata generation.

Target: 40%+ coverage for training_jobs.py module.
"""

import hashlib
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from PIL import Image
from sqlalchemy.orm import Session

from image_search_service.db.models import (
    ImageAsset,
    JobStatus,
    SessionStatus,
    TrainingEvidence,
    TrainingJob,
    TrainingSession,
)
from image_search_service.queue.training_jobs import (
    _build_evidence_metadata,
    train_single_asset,
    train_session,
)


@pytest.fixture
def training_job_fixtures(sync_db_session: Session, tmp_path: Path, monkeypatch):
    """Set up dependencies for training_jobs testing.

    Provides:
    - DB session with test data (assets, session, jobs)
    - Mock embedding service (512-dim vectors)
    - Mock Qdrant client
    - Test images on disk
    - Mock ProgressTracker
    - Mock settings
    """
    # Mock get_sync_session to return test session
    monkeypatch.setattr(
        "image_search_service.queue.training_jobs.get_sync_session",
        lambda: sync_db_session,
    )

    # Mock embedding service (768-dim for image search)
    mock_embed = MagicMock()
    mock_embed.embed_image.return_value = [0.1] * 768
    mock_embed.embed_images_batch.return_value = [[0.1] * 768, [0.2] * 768, [0.3] * 768]
    mock_embed.embed_image_from_pil.return_value = [0.1] * 768
    mock_embed.embedding_dim = 768
    mock_embed.device = "cpu"
    monkeypatch.setattr(
        "image_search_service.queue.training_jobs.get_embedding_service",
        lambda: mock_embed,
    )

    # Mock Qdrant client (no-op)
    mock_qdrant = MagicMock()
    monkeypatch.setattr(
        "image_search_service.queue.training_jobs.ensure_collection",
        lambda dim: None,
    )
    monkeypatch.setattr(
        "image_search_service.queue.training_jobs.upsert_vector",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "image_search_service.queue.training_jobs.upsert_vectors_batch",
        lambda vectors: None,
    )

    # Mock ProgressTracker
    mock_tracker_class = MagicMock()
    mock_tracker_instance = MagicMock()
    mock_tracker_instance.should_stop.return_value = False
    mock_tracker_instance.check_cancelled.return_value = False
    mock_tracker_instance.get_current_progress.return_value = {"percentage": 50}
    mock_tracker_instance.calculate_rate.return_value = 10.5
    mock_tracker_class.return_value = mock_tracker_instance
    monkeypatch.setattr(
        "image_search_service.queue.training_jobs.ProgressTracker",
        mock_tracker_class,
    )

    # Mock settings
    mock_settings = MagicMock()
    mock_settings.training_batch_size = 3
    mock_settings.gpu_batch_size = 2
    mock_settings.clip_model_name = "ViT-B-32"
    mock_settings.gpu_memory_cleanup_enabled = False
    mock_settings.gpu_memory_cleanup_interval = 100
    monkeypatch.setattr(
        "image_search_service.queue.training_jobs.get_settings",
        lambda: mock_settings,
    )

    # Mock get_device_info for metadata tests
    mock_device_info = {
        "selected_device": "cpu",
        "cuda_available": False,
        "mps_available": False,
        "platform": "Linux",
        "machine": "x86_64",
        "pytorch_version": "2.0.0",
    }
    monkeypatch.setattr(
        "image_search_service.queue.training_jobs.get_device_info",
        lambda: mock_device_info,
    )

    # Mock sync operations (these are already imported in training_jobs.py)
    monkeypatch.setattr(
        "image_search_service.queue.training_jobs.update_asset_indexed_at_sync",
        lambda session, asset_id: None,
    )
    monkeypatch.setattr(
        "image_search_service.queue.training_jobs.update_job_progress_sync",
        lambda session, job_id, progress, time_ms: None,
    )
    monkeypatch.setattr(
        "image_search_service.queue.training_jobs.increment_subdirectory_trained_count_sync",
        lambda session, session_id, path: None,
    )

    # Create test images on disk
    test_images = {}
    for i, name in enumerate(["photo1.jpg", "photo2.jpg", "photo3.jpg", "photo4.jpg"]):
        img_path = tmp_path / name
        img = Image.new("RGB", (100, 100), color=(i * 50, 100, 150))
        img.save(img_path)
        test_images[name] = str(img_path)

    # Create test database records
    session = TrainingSession(
        id=1,
        name="Test Session",
        status=SessionStatus.RUNNING.value,
        root_path=str(tmp_path),
        total_images=3,
        processed_images=0,
        failed_images=0,
    )
    sync_db_session.add(session)

    assets = []
    jobs = []
    for i, (name, path) in enumerate(test_images.items(), start=1):
        asset = ImageAsset(
            id=i,
            path=path,
            width=100,
            height=100,
            file_size=5000,
            mime_type="image/jpeg",
        )
        sync_db_session.add(asset)
        assets.append(asset)

        # Create jobs for first 3 assets only (4th is for testing missing asset)
        if i <= 3:
            job = TrainingJob(
                id=i,
                session_id=1,
                asset_id=i,
                status=JobStatus.PENDING.value,
                image_path=path,
            )
            sync_db_session.add(job)
            jobs.append(job)

    sync_db_session.commit()

    return {
        "db_session": sync_db_session,
        "mock_embed": mock_embed,
        "mock_tracker": mock_tracker_instance,
        "mock_settings": mock_settings,
        "test_images": test_images,
        "tmp_path": tmp_path,
        "session": session,
        "assets": assets,
        "jobs": jobs,
    }


# ==================== _build_evidence_metadata Tests ====================


def test_build_evidence_metadata_basic(training_job_fixtures):
    """Test _build_evidence_metadata returns dict with expected keys."""
    fixtures = training_job_fixtures
    asset = fixtures["assets"][0]
    vector = [0.1] * 768  # 768-dim for image search
    mock_embed = fixtures["mock_embed"]

    metadata = _build_evidence_metadata(
        asset=asset,
        vector=vector,
        embedding_time_ms=100,
        total_time_ms=150,
        embedding_service=mock_embed,
    )

    # Check top-level keys
    assert "image" in metadata
    assert "embedding" in metadata
    assert "environment" in metadata
    assert "timing" in metadata

    # Check timing metadata
    assert metadata["timing"]["embedding_time_ms"] == 100
    assert metadata["timing"]["total_time_ms"] == 150
    assert metadata["timing"]["overhead_ms"] == 50


def test_build_evidence_metadata_with_image_metadata(training_job_fixtures):
    """Test metadata includes image dimensions and file info."""
    fixtures = training_job_fixtures
    asset = fixtures["assets"][0]
    vector = [0.1] * 768  # 768-dim for image search
    mock_embed = fixtures["mock_embed"]

    metadata = _build_evidence_metadata(
        asset=asset,
        vector=vector,
        embedding_time_ms=100,
        total_time_ms=150,
        embedding_service=mock_embed,
    )

    # Check image metadata
    image_meta = metadata["image"]
    assert image_meta["width"] == 100
    assert image_meta["height"] == 100
    assert image_meta["file_size"] == 5000
    assert image_meta["mime_type"] == "image/jpeg"


def test_build_evidence_metadata_with_embedding_stats(training_job_fixtures):
    """Test metadata includes embedding dimension and norm."""
    fixtures = training_job_fixtures
    asset = fixtures["assets"][0]
    vector = [0.1] * 768  # 768-dim for image search
    mock_embed = fixtures["mock_embed"]

    metadata = _build_evidence_metadata(
        asset=asset,
        vector=vector,
        embedding_time_ms=100,
        total_time_ms=150,
        embedding_service=mock_embed,
    )

    # Check embedding metadata
    embed_meta = metadata["embedding"]
    assert embed_meta["dimension"] == 768
    assert "norm" in embed_meta
    assert embed_meta["norm"] > 0  # L2 norm should be positive
    assert embed_meta["generation_time_ms"] == 100


def test_build_evidence_metadata_null_vector(training_job_fixtures):
    """Test metadata handles None vector (failed embedding)."""
    fixtures = training_job_fixtures
    asset = fixtures["assets"][0]
    mock_embed = fixtures["mock_embed"]

    metadata = _build_evidence_metadata(
        asset=asset,
        vector=None,
        embedding_time_ms=50,
        total_time_ms=100,
        embedding_service=mock_embed,
    )

    # Embedding metadata should be empty when vector is None
    embed_meta = metadata["embedding"]
    assert "dimension" not in embed_meta
    assert "norm" not in embed_meta


def test_build_evidence_metadata_environment_info(training_job_fixtures):
    """Test metadata includes environment details."""
    fixtures = training_job_fixtures
    asset = fixtures["assets"][0]
    vector = [0.1] * 768  # 768-dim for image search
    mock_embed = fixtures["mock_embed"]

    metadata = _build_evidence_metadata(
        asset=asset,
        vector=vector,
        embedding_time_ms=100,
        total_time_ms=150,
        embedding_service=mock_embed,
    )

    # Check environment metadata
    env_meta = metadata["environment"]
    assert "python_version" in env_meta
    assert "device" in env_meta
    assert env_meta["device"] == "cpu"
    assert env_meta["cuda_available"] is False
    assert env_meta["platform"] == "Linux"


# ==================== train_single_asset Tests ====================


def test_train_single_asset_success(training_job_fixtures, monkeypatch):
    """Test train_single_asset successfully embeds image and creates evidence."""
    fixtures = training_job_fixtures
    job = fixtures["jobs"][0]
    asset = fixtures["assets"][0]

    # Track evidence creation
    evidence_created = []

    def mock_create_evidence(session, data):
        evidence_created.append(data)

    monkeypatch.setattr(
        "image_search_service.queue.training_jobs.create_evidence_sync",
        mock_create_evidence,
    )

    # Track job status updates
    job_updates = []

    def mock_update_job(session, job_id, status, error=None):
        job_updates.append({"job_id": job_id, "status": status, "error": error})

    monkeypatch.setattr(
        "image_search_service.queue.training_jobs.update_training_job_sync",
        mock_update_job,
    )

    result = train_single_asset(job.id, asset.id, session_id=1)

    # Check result
    assert result["status"] == "success"
    assert result["asset_id"] == asset.id
    assert "processing_time_ms" in result

    # Check evidence was created
    assert len(evidence_created) == 1
    evidence = evidence_created[0]
    assert evidence["asset_id"] == asset.id
    assert evidence["session_id"] == 1
    assert evidence["model_name"] == "OpenCLIP"
    assert "embedding_checksum" in evidence
    assert "metadata_json" in evidence

    # Check job was marked running then completed
    assert len(job_updates) == 2
    assert job_updates[0]["status"] == JobStatus.RUNNING.value
    assert job_updates[1]["status"] == JobStatus.COMPLETED.value


def test_train_single_asset_file_not_found(training_job_fixtures, monkeypatch):
    """Test train_single_asset handles missing image file gracefully."""
    fixtures = training_job_fixtures
    job = fixtures["jobs"][0]

    # Create asset with non-existent path
    asset = ImageAsset(
        id=999,
        path="/nonexistent/image.jpg",
        width=100,
        height=100,
    )
    fixtures["db_session"].add(asset)
    fixtures["db_session"].commit()

    # Mock embedding to raise FileNotFoundError
    def mock_embed_error(path):
        raise FileNotFoundError(f"Image not found: {path}")

    fixtures["mock_embed"].embed_image.side_effect = mock_embed_error

    # Track job updates
    job_updates = []

    def mock_update_job(session, job_id, status, error=None):
        job_updates.append({"job_id": job_id, "status": status, "error": error})

    monkeypatch.setattr(
        "image_search_service.queue.training_jobs.update_training_job_sync",
        mock_update_job,
    )

    # Track evidence creation (for failure)
    evidence_created = []

    def mock_create_evidence(session, data):
        evidence_created.append(data)

    monkeypatch.setattr(
        "image_search_service.queue.training_jobs.create_evidence_sync",
        mock_create_evidence,
    )

    result = train_single_asset(job.id, asset.id, session_id=1)

    # Check result indicates error
    assert result["status"] == "error"
    assert "not found" in result["message"].lower()

    # Check job was marked as failed
    failed_updates = [u for u in job_updates if u["status"] == JobStatus.FAILED.value]
    assert len(failed_updates) >= 1
    assert "not found" in failed_updates[0]["error"].lower()


def test_train_single_asset_corrupt_image(training_job_fixtures, monkeypatch, tmp_path):
    """Test train_single_asset handles corrupt image (PIL cannot open)."""
    fixtures = training_job_fixtures

    # Create corrupt image file (not a valid image)
    corrupt_path = tmp_path / "corrupt.jpg"
    corrupt_path.write_text("not an image")

    asset = ImageAsset(
        id=998,
        path=str(corrupt_path),
        width=100,
        height=100,
    )
    fixtures["db_session"].add(asset)

    job = TrainingJob(
        id=998,
        session_id=1,
        asset_id=998,
        status=JobStatus.PENDING.value,
    )
    fixtures["db_session"].add(job)
    fixtures["db_session"].commit()

    # Mock embedding to raise PIL error
    def mock_embed_error(path):
        raise OSError("cannot identify image file")

    fixtures["mock_embed"].embed_image.side_effect = mock_embed_error

    # Track job updates
    job_updates = []

    def mock_update_job(session, job_id, status, error=None):
        job_updates.append({"job_id": job_id, "status": status, "error": error})

    monkeypatch.setattr(
        "image_search_service.queue.training_jobs.update_training_job_sync",
        mock_update_job,
    )

    result = train_single_asset(job.id, asset.id, session_id=1)

    # Check result indicates error
    assert result["status"] == "error"

    # Check job was marked as failed
    failed_updates = [u for u in job_updates if u["status"] == JobStatus.FAILED.value]
    assert len(failed_updates) >= 1


def test_train_single_asset_embedding_error(training_job_fixtures, monkeypatch):
    """Test train_single_asset handles EmbeddingService exceptions."""
    fixtures = training_job_fixtures
    job = fixtures["jobs"][0]
    asset = fixtures["assets"][0]

    # Mock embedding to raise exception
    def mock_embed_error(path):
        raise RuntimeError("GPU out of memory")

    fixtures["mock_embed"].embed_image.side_effect = mock_embed_error

    # Track job updates
    job_updates = []

    def mock_update_job(session, job_id, status, error=None):
        job_updates.append({"job_id": job_id, "status": status, "error": error})

    monkeypatch.setattr(
        "image_search_service.queue.training_jobs.update_training_job_sync",
        mock_update_job,
    )

    result = train_single_asset(job.id, asset.id, session_id=1)

    # Check result indicates error
    assert result["status"] == "error"
    assert "GPU out of memory" in result["message"]

    # Check job was marked as failed
    failed_updates = [u for u in job_updates if u["status"] == JobStatus.FAILED.value]
    assert len(failed_updates) >= 1
    assert "GPU out of memory" in failed_updates[0]["error"]


def test_train_single_asset_evidence_metadata(training_job_fixtures, monkeypatch):
    """Test train_single_asset creates TrainingEvidence with comprehensive metadata."""
    fixtures = training_job_fixtures
    job = fixtures["jobs"][0]
    asset = fixtures["assets"][0]

    # Track evidence creation
    evidence_created = []

    def mock_create_evidence(session, data):
        evidence_created.append(data)

    monkeypatch.setattr(
        "image_search_service.queue.training_jobs.create_evidence_sync",
        mock_create_evidence,
    )

    # Mock job updates to avoid errors
    monkeypatch.setattr(
        "image_search_service.queue.training_jobs.update_training_job_sync",
        lambda *args, **kwargs: None,
    )

    result = train_single_asset(job.id, asset.id, session_id=1)

    # Check evidence metadata structure
    assert len(evidence_created) == 1
    evidence = evidence_created[0]

    metadata = evidence["metadata_json"]
    assert "image" in metadata
    assert "embedding" in metadata
    assert "environment" in metadata
    assert "timing" in metadata

    # Check embedding checksum
    assert len(evidence["embedding_checksum"]) == 64  # SHA256 hex digest


# ==================== train_session Tests ====================


def test_train_session_not_found(training_job_fixtures, monkeypatch):
    """Test train_session returns error for invalid session_id."""
    # Mock get_session_by_id_sync to return None
    monkeypatch.setattr(
        "image_search_service.queue.training_jobs.get_session_by_id_sync",
        lambda session, session_id: None,
    )

    # This will return early since no pending jobs
    result = train_session(session_id=999)

    # Should complete with no jobs processed
    assert result["status"] == "completed"
    assert result["processed"] == 0
    assert result["message"] == "No pending jobs"


def test_train_session_no_pending_jobs(training_job_fixtures, monkeypatch):
    """Test train_session completes gracefully when no pending TrainingJobs exist."""
    fixtures = training_job_fixtures

    # Mark all jobs as completed
    for job in fixtures["jobs"]:
        job.status = JobStatus.COMPLETED.value
    fixtures["db_session"].commit()

    result = train_session(session_id=1)

    # Should complete with message
    assert result["status"] == "completed"
    assert result["session_id"] == 1
    assert result["processed"] == 0
    assert result["failed"] == 0
    assert result["message"] == "No pending jobs"


def test_train_session_basic_flow(training_job_fixtures, monkeypatch):
    """Test train_session processes 3 pending jobs successfully."""
    fixtures = training_job_fixtures

    # Ensure jobs are in PENDING status (they should be, but make explicit)
    for job in fixtures["jobs"]:
        job.status = JobStatus.PENDING.value
    fixtures["db_session"].commit()

    # Mock train_batch to simulate successful processing
    def mock_train_batch(session_id, asset_ids, batch_num, **kwargs):
        # Mark jobs as processed
        return {
            "processed": len(asset_ids),
            "failed": 0,
            "io_time": 0.5,
            "gpu_time": 1.0,
        }

    monkeypatch.setattr(
        "image_search_service.queue.training_jobs.train_batch",
        mock_train_batch,
    )

    # Mock get_session_by_id_sync for status update
    def mock_get_session(session, session_id):
        return fixtures["session"]

    monkeypatch.setattr(
        "image_search_service.queue.training_jobs.get_session_by_id_sync",
        mock_get_session,
    )

    # Mock face detection auto-trigger (patch at import location)
    def mock_get_queue(name):
        mock_queue = MagicMock()
        mock_job = MagicMock()
        mock_job.id = "test-job-id"
        mock_queue.enqueue.return_value = mock_job
        return mock_queue

    monkeypatch.setattr(
        "image_search_service.queue.worker.get_queue",
        mock_get_queue,
    )

    result = train_session(session_id=1)

    # Check result
    assert result["status"] == "completed"
    assert result["session_id"] == 1
    assert result["processed"] == 3
    assert result["failed"] == 0
    assert "elapsed_seconds" in result
    assert "rate_per_minute" in result


def test_train_session_cancellation(training_job_fixtures, monkeypatch):
    """Test train_session stops when session is cancelled mid-batch."""
    fixtures = training_job_fixtures

    # Ensure jobs are in PENDING status
    for job in fixtures["jobs"]:
        job.status = JobStatus.PENDING.value
    fixtures["db_session"].commit()

    # Mock ProgressTracker to indicate cancellation immediately
    def mock_should_stop(session):
        return True  # Cancel immediately on first check

    def mock_check_cancelled(session):
        return True  # Indicate cancelled (not paused)

    fixtures["mock_tracker"].should_stop.side_effect = mock_should_stop
    fixtures["mock_tracker"].check_cancelled.side_effect = mock_check_cancelled

    # Mock train_batch (won't be called since cancelled before batch)
    def mock_train_batch(session_id, asset_ids, batch_num, **kwargs):
        return {"processed": len(asset_ids), "failed": 0}

    monkeypatch.setattr(
        "image_search_service.queue.training_jobs.train_batch",
        mock_train_batch,
    )

    result = train_session(session_id=1)

    # Should stop due to cancellation
    assert result["status"] == "cancelled"
    assert result["session_id"] == 1
    assert "Processing cancelled" in result["message"]


def test_train_session_pause(training_job_fixtures, monkeypatch):
    """Test train_session stops when session is paused."""
    fixtures = training_job_fixtures

    # Ensure jobs are in PENDING status
    for job in fixtures["jobs"]:
        job.status = JobStatus.PENDING.value
    fixtures["db_session"].commit()

    # Mock ProgressTracker to indicate pause immediately
    def mock_should_stop(session):
        return True  # Pause immediately on first check

    def mock_check_cancelled(session):
        return False  # Not cancelled, so must be paused

    fixtures["mock_tracker"].should_stop.side_effect = mock_should_stop
    fixtures["mock_tracker"].check_cancelled.side_effect = mock_check_cancelled

    # Mock train_batch (won't be called since paused before batch)
    def mock_train_batch(session_id, asset_ids, batch_num, **kwargs):
        return {"processed": len(asset_ids), "failed": 0}

    monkeypatch.setattr(
        "image_search_service.queue.training_jobs.train_batch",
        mock_train_batch,
    )

    result = train_session(session_id=1)

    # Should stop due to pause
    assert result["status"] == "paused"
    assert result["session_id"] == 1
    assert "Processing paused" in result["message"]


def test_train_session_partial_failure(training_job_fixtures, monkeypatch):
    """Test train_session handles case where 1 of 3 jobs fails."""
    fixtures = training_job_fixtures

    # Ensure jobs are in PENDING status
    for job in fixtures["jobs"]:
        job.status = JobStatus.PENDING.value
    fixtures["db_session"].commit()

    # Mock train_batch to simulate partial failure
    def mock_train_batch(session_id, asset_ids, batch_num, **kwargs):
        # Fail first asset, succeed others
        processed = len(asset_ids) - 1
        failed = 1
        return {"processed": processed, "failed": failed}

    monkeypatch.setattr(
        "image_search_service.queue.training_jobs.train_batch",
        mock_train_batch,
    )

    # Mock get_session_by_id_sync
    def mock_get_session(session, session_id):
        return fixtures["session"]

    monkeypatch.setattr(
        "image_search_service.queue.training_jobs.get_session_by_id_sync",
        mock_get_session,
    )

    # Mock face detection auto-trigger
    def mock_get_queue(name):
        mock_queue = MagicMock()
        mock_job = MagicMock()
        mock_job.id = "test-job-id"
        mock_queue.enqueue.return_value = mock_job
        return mock_queue

    monkeypatch.setattr(
        "image_search_service.queue.worker.get_queue",
        mock_get_queue,
    )

    result = train_session(session_id=1)

    # Should complete with partial failure
    assert result["status"] == "completed"
    assert result["processed"] == 2  # 3 assets - 1 failed
    assert result["failed"] == 1


def test_train_session_exception_handling(training_job_fixtures, monkeypatch):
    """Test train_session handles unexpected exceptions gracefully."""
    # Mock train_batch to raise exception
    def mock_train_batch_error(*args, **kwargs):
        raise RuntimeError("Unexpected error in batch processing")

    monkeypatch.setattr(
        "image_search_service.queue.training_jobs.train_batch",
        mock_train_batch_error,
    )

    result = train_session(session_id=1)

    # Should return failed status
    assert result["status"] == "failed"
    assert result["session_id"] == 1
    assert "error" in result
    assert "Unexpected error" in result["error"]
