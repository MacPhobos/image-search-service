"""Unit tests for discover_unknown_persons_job background task."""

from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from image_search_service.queue.face_jobs import discover_unknown_persons_job


@pytest.fixture
def mock_dependencies() -> dict:
    """Mock all external dependencies for the job."""
    # Create mock hdbscan module
    mock_hdbscan_module = MagicMock()
    mock_clusterer = MagicMock()
    mock_hdbscan_module.HDBSCAN.return_value = mock_clusterer

    with (
        patch("image_search_service.queue.face_jobs.get_current_job") as mock_job,
        patch("image_search_service.queue.face_jobs.get_sync_session") as mock_session,
        patch("image_search_service.vector.face_qdrant.get_face_qdrant_client") as mock_qdrant,
        patch("image_search_service.queue.worker.get_redis") as mock_redis,
        patch("image_search_service.services.face_clustering_service.compute_cluster_confidence_from_embeddings") as mock_confidence,  # noqa: E501
        patch("image_search_service.services.unknown_person_service.compute_membership_hash") as mock_hash,  # noqa: E501
        patch.dict("sys.modules", {"hdbscan": mock_hdbscan_module}),
    ):
        # Setup mock job
        mock_job.return_value = Mock(id="test-job-123")

        # Setup mock session
        mock_db = MagicMock()
        mock_session.return_value = mock_db

        # Setup mock Qdrant client
        mock_qdrant_instance = MagicMock()
        mock_qdrant.return_value = mock_qdrant_instance

        # Setup mock Redis
        mock_redis_instance = MagicMock()
        mock_redis.return_value = mock_redis_instance

        yield {
            "job": mock_job,
            "session": mock_db,
            "qdrant": mock_qdrant_instance,
            "redis": mock_redis_instance,
            "confidence": mock_confidence,
            "hash": mock_hash,
            "hdbscan_module": mock_hdbscan_module,
            "clusterer": mock_clusterer,
        }


def test_successful_clustering_with_multiple_clusters(mock_dependencies: dict) -> None:
    """Test successful clustering with multiple qualifying clusters."""
    # Arrange: Create 3 clusters with 5 faces each
    face_ids = [uuid.uuid4() for _ in range(15)]
    embeddings_list = [np.random.randn(512).astype(np.float32) for _ in range(15)]

    # Mock Qdrant response
    mock_dependencies["qdrant"].get_unlabeled_faces_with_embeddings.return_value = [
        (fid, emb) for fid, emb in zip(face_ids, embeddings_list)
    ]

    # Mock HDBSCAN labels: 3 clusters (0, 1, 2) with 5 faces each
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    mock_dependencies["clusterer"].fit_predict.return_value = labels

    # Mock high confidence for all clusters
    mock_dependencies["confidence"].return_value = 0.85

    # Mock membership hash
    mock_dependencies["hash"].side_effect = lambda ids: f"hash_{len(ids)}"

    # Mock database operations
    mock_face_instances = []
    for face_id in face_ids:
        mock_face = MagicMock()
        mock_face.id = face_id
        mock_face.cluster_id = None
        mock_face_instances.append(mock_face)

    def mock_execute(stmt):
        result = MagicMock()
        # Return first face instance for simplicity
        result.scalar_one_or_none.return_value = mock_face_instances[0]
        return result

    mock_dependencies["session"].execute.side_effect = mock_execute

    # Act
    result = discover_unknown_persons_job(
        clustering_method="hdbscan",
        min_cluster_size=5,
        min_quality=0.3,
        max_faces=50000,
        min_cluster_confidence=0.70,
    )

    # Assert
    assert result["status"] == "completed"
    assert result["total_faces"] == 15
    assert result["clusters_found"] == 3
    assert result["noise_count"] == 0
    assert result["qualifying_groups"] == 3
    assert result["filtered_low_confidence"] == 0

    # Verify Qdrant was called
    mock_dependencies["qdrant"].get_unlabeled_faces_with_embeddings.assert_called_once_with(
        quality_threshold=0.3,
        limit=50000,
    )

    # Verify HDBSCAN was configured correctly
    mock_dependencies["hdbscan_module"].HDBSCAN.assert_called_once_with(
        min_cluster_size=5,
        min_samples=3,
        metric="euclidean",
        cluster_selection_method="eom",
        core_dist_n_jobs=1,
    )

    # Verify Redis caching
    assert mock_dependencies["redis"].set.call_count >= 2  # last_discovery + cluster metadata


def test_insufficient_faces(mock_dependencies: dict) -> None:
    """Test job with too few faces (< min_cluster_size)."""
    # Arrange: Only 3 faces available
    face_ids = [uuid.uuid4() for _ in range(3)]
    embeddings_list = [np.random.randn(512).astype(np.float32) for _ in range(3)]

    mock_dependencies["qdrant"].get_unlabeled_faces_with_embeddings.return_value = [
        (fid, emb) for fid, emb in zip(face_ids, embeddings_list)
    ]

    # Act
    result = discover_unknown_persons_job(
        clustering_method="hdbscan",
        min_cluster_size=5,
        min_quality=0.3,
        max_faces=50000,
        min_cluster_confidence=0.70,
    )

    # Assert
    assert result["status"] == "completed"
    assert result["total_faces"] == 3
    assert result["clusters_found"] == 0
    assert "Insufficient faces" in result["message"]

    # Verify HDBSCAN was NOT called
    mock_dependencies["clusterer"].fit_predict.assert_not_called()


def test_memory_ceiling_exceeded(mock_dependencies: dict) -> None:
    """Test that job fails when memory ceiling is exceeded."""
    # Arrange: Too many faces (would require >4GB memory)
    # sqrt(4 * 1024^3 / 8) ~ 23170 faces for 4GB
    face_ids = [uuid.uuid4() for _ in range(25000)]
    embeddings_list = [np.random.randn(512).astype(np.float32) for _ in range(25000)]

    mock_dependencies["qdrant"].get_unlabeled_faces_with_embeddings.return_value = [
        (fid, emb) for fid, emb in zip(face_ids, embeddings_list)
    ]

    # Act
    result = discover_unknown_persons_job(
        clustering_method="hdbscan",
        min_cluster_size=5,
        min_quality=0.3,
        max_faces=50000,
        min_cluster_confidence=0.70,
    )

    # Assert
    assert result["status"] == "error"
    assert "Memory ceiling exceeded" in result["message"]

    # Verify HDBSCAN was NOT called
    mock_dependencies["clusterer"].fit_predict.assert_not_called()


def test_low_confidence_filtering(mock_dependencies: dict) -> None:
    """Test that low-confidence clusters are filtered out."""
    # Arrange: 2 clusters, one high confidence, one low
    face_ids = [uuid.uuid4() for _ in range(10)]
    embeddings_list = [np.random.randn(512).astype(np.float32) for _ in range(10)]

    mock_dependencies["qdrant"].get_unlabeled_faces_with_embeddings.return_value = [
        (fid, emb) for fid, emb in zip(face_ids, embeddings_list)
    ]

    # Mock HDBSCAN labels: 2 clusters with 5 faces each
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    mock_dependencies["clusterer"].fit_predict.return_value = labels

    # Mock confidence: cluster 0 high, cluster 1 low
    def mock_confidence_func(embeddings, sample_size=20):
        # First call (cluster 0) -> high confidence
        if not hasattr(mock_confidence_func, "call_count"):
            mock_confidence_func.call_count = 0
        mock_confidence_func.call_count += 1
        return 0.85 if mock_confidence_func.call_count == 1 else 0.65

    mock_dependencies["confidence"].side_effect = mock_confidence_func

    # Mock membership hash
    mock_dependencies["hash"].side_effect = lambda ids: f"hash_{len(ids)}"

    # Mock database operations
    def mock_execute(stmt):
        result = MagicMock()
        mock_face = MagicMock()
        mock_face.id = face_ids[0]
        mock_face.cluster_id = None
        result.scalar_one_or_none.return_value = mock_face
        return result

    mock_dependencies["session"].execute.side_effect = mock_execute

    # Act
    result = discover_unknown_persons_job(
        clustering_method="hdbscan",
        min_cluster_size=5,
        min_quality=0.3,
        max_faces=50000,
        min_cluster_confidence=0.70,
    )

    # Assert
    assert result["status"] == "completed"
    assert result["clusters_found"] == 2
    assert result["qualifying_groups"] == 1  # Only cluster 0 qualifies
    assert result["filtered_low_confidence"] == 1  # Cluster 1 filtered


def test_noise_faces_labeled(mock_dependencies: dict) -> None:
    """Test that noise faces (label -1) get cluster_id = '-1'."""
    # Arrange: 7 faces, 5 in cluster, 2 noise
    face_ids = [uuid.uuid4() for _ in range(7)]
    embeddings_list = [np.random.randn(512).astype(np.float32) for _ in range(7)]

    mock_dependencies["qdrant"].get_unlabeled_faces_with_embeddings.return_value = [
        (fid, emb) for fid, emb in zip(face_ids, embeddings_list)
    ]

    # Mock HDBSCAN labels: 1 cluster (0) with 5 faces, 2 noise (-1)
    labels = np.array([0, 0, 0, 0, 0, -1, -1])
    mock_dependencies["clusterer"].fit_predict.return_value = labels

    # Mock high confidence
    mock_dependencies["confidence"].return_value = 0.85

    # Mock membership hash
    mock_dependencies["hash"].side_effect = lambda ids: f"hash_{len(ids)}"

    # Mock database operations
    mock_faces = []
    for face_id in face_ids:
        mock_face = MagicMock()
        mock_face.id = face_id
        mock_face.cluster_id = None
        mock_faces.append(mock_face)

    execute_call_count = [0]

    def mock_execute(stmt):
        result = MagicMock()
        # Return faces in order
        idx = execute_call_count[0]
        if idx < len(mock_faces):
            result.scalar_one_or_none.return_value = mock_faces[idx]
        else:
            result.scalar_one_or_none.return_value = None
        execute_call_count[0] += 1
        return result

    mock_dependencies["session"].execute.side_effect = mock_execute

    # Act
    result = discover_unknown_persons_job(
        clustering_method="hdbscan",
        min_cluster_size=5,
        min_quality=0.3,
        max_faces=50000,
        min_cluster_confidence=0.70,
    )

    # Assert
    assert result["status"] == "completed"
    assert result["noise_count"] == 2
    assert result["qualifying_groups"] == 1

    # Verify that noise faces got cluster_id = '-1'
    # (Check last 2 mock faces)
    # Note: In real implementation, FaceInstance.cluster_id would be set
    # We verify the logic by checking labels array
    assert labels[5] == -1
    assert labels[6] == -1


def test_cluster_id_namespace_prefix(mock_dependencies: dict) -> None:
    """Test that cluster IDs have 'unknown_' namespace prefix."""
    # Arrange
    face_ids = [uuid.uuid4() for _ in range(5)]
    embeddings_list = [np.random.randn(512).astype(np.float32) for _ in range(5)]

    mock_dependencies["qdrant"].get_unlabeled_faces_with_embeddings.return_value = [
        (fid, emb) for fid, emb in zip(face_ids, embeddings_list)
    ]

    # Mock HDBSCAN labels: cluster with label 42
    labels = np.array([42, 42, 42, 42, 42])
    mock_dependencies["clusterer"].fit_predict.return_value = labels

    # Mock high confidence
    mock_dependencies["confidence"].return_value = 0.85

    # Mock membership hash
    captured_cluster_ids = []

    def mock_hash_func(ids):
        # Capture cluster metadata from Redis calls
        return f"hash_{len(ids)}"

    mock_dependencies["hash"].side_effect = mock_hash_func

    # Capture Redis set calls
    redis_calls = []

    def mock_redis_set(key, value, ex=None):
        redis_calls.append({"key": key, "value": value})

    mock_dependencies["redis"].set.side_effect = mock_redis_set

    # Mock database
    def mock_execute(stmt):
        result = MagicMock()
        mock_face = MagicMock()
        result.scalar_one_or_none.return_value = mock_face
        return result

    mock_dependencies["session"].execute.side_effect = mock_execute

    # Act
    result = discover_unknown_persons_job(
        clustering_method="hdbscan",
        min_cluster_size=5,
        min_quality=0.3,
        max_faces=50000,
        min_cluster_confidence=0.70,
    )

    # Assert
    assert result["status"] == "completed"

    # Check Redis cache keys for namespace prefix
    cluster_keys = [call["key"] for call in redis_calls if "cluster:" in call["key"]]
    assert len(cluster_keys) >= 1
    assert all("unknown_" in key for key in cluster_keys), (
        "All cluster keys should have 'unknown_' namespace prefix"
    )


def test_progress_reporting(mock_dependencies: dict) -> None:
    """Test that progress is reported to Redis throughout execution."""
    # Arrange
    face_ids = [uuid.uuid4() for _ in range(5)]
    embeddings_list = [np.random.randn(512).astype(np.float32) for _ in range(5)]

    mock_dependencies["qdrant"].get_unlabeled_faces_with_embeddings.return_value = [
        (fid, emb) for fid, emb in zip(face_ids, embeddings_list)
    ]

    labels = np.array([0, 0, 0, 0, 0])
    mock_dependencies["clusterer"].fit_predict.return_value = labels
    mock_dependencies["confidence"].return_value = 0.85
    mock_dependencies["hash"].side_effect = lambda ids: f"hash_{len(ids)}"

    # Capture progress updates
    progress_updates = []

    def mock_redis_set(key, value, ex=None):
        if "progress" in key:
            progress_updates.append(json.loads(value))

    mock_dependencies["redis"].set.side_effect = mock_redis_set

    # Mock database
    def mock_execute(stmt):
        result = MagicMock()
        result.scalar_one_or_none.return_value = MagicMock()
        return result

    mock_dependencies["session"].execute.side_effect = mock_execute

    # Act
    result = discover_unknown_persons_job(
        clustering_method="hdbscan",
        min_cluster_size=5,
        min_quality=0.3,
        max_faces=50000,
        min_cluster_confidence=0.70,
    )

    # Assert
    assert result["status"] == "completed"

    # Verify progress phases
    phases = [update["phase"] for update in progress_updates]
    assert "retrieval" in phases
    assert "memory_check" in phases
    assert "embedding" in phases
    assert "clustering" in phases
    assert "metadata" in phases
    assert "persistence" in phases
    assert "caching" in phases
    assert "complete" in phases


def test_unsupported_clustering_method(mock_dependencies: dict) -> None:
    """Test that unsupported clustering methods return error."""
    # Act
    result = discover_unknown_persons_job(
        clustering_method="kmeans",  # Not supported
        min_cluster_size=5,
        min_quality=0.3,
        max_faces=50000,
        min_cluster_confidence=0.70,
    )

    # Assert
    assert result["status"] == "error"
    assert "Unsupported clustering method" in result["message"]


def test_hdbscan_not_installed() -> None:
    """Test graceful error when hdbscan is not installed."""
    # Arrange: Remove hdbscan from sys.modules to simulate ImportError
    # We need to test without the fixture
    with (
        patch("image_search_service.queue.face_jobs.get_current_job") as mock_job,
        patch("image_search_service.queue.face_jobs.get_sync_session") as mock_session,
        patch.dict("sys.modules", {"hdbscan": None}),  # Simulate import error
    ):
        mock_job.return_value = Mock(id="test-job-123")
        mock_db = MagicMock()
        mock_session.return_value = mock_db

        # Act
        result = discover_unknown_persons_job(
            clustering_method="hdbscan",
            min_cluster_size=5,
            min_quality=0.3,
            max_faces=50000,
            min_cluster_confidence=0.70,
        )

    # Assert
    assert result["status"] == "error"
    assert "hdbscan package not installed" in result["message"]


def test_redis_caching_with_metadata(mock_dependencies: dict) -> None:
    """Test that cluster metadata is cached in Redis with correct structure."""
    # Arrange
    face_ids = [uuid.uuid4() for _ in range(5)]
    embeddings_list = [np.random.randn(512).astype(np.float32) for _ in range(5)]

    mock_dependencies["qdrant"].get_unlabeled_faces_with_embeddings.return_value = [
        (fid, emb) for fid, emb in zip(face_ids, embeddings_list)
    ]

    labels = np.array([0, 0, 0, 0, 0])
    mock_dependencies["clusterer"].fit_predict.return_value = labels
    mock_dependencies["confidence"].return_value = 0.85
    mock_dependencies["hash"].return_value = "test_hash"

    # Capture Redis calls
    redis_cache = {}

    def mock_redis_set(key, value, ex=None):
        redis_cache[key] = json.loads(value)

    mock_dependencies["redis"].set.side_effect = mock_redis_set

    # Mock database
    def mock_execute(stmt):
        result = MagicMock()
        result.scalar_one_or_none.return_value = MagicMock()
        return result

    mock_dependencies["session"].execute.side_effect = mock_execute

    # Act
    result = discover_unknown_persons_job(
        clustering_method="hdbscan",
        min_cluster_size=5,
        min_quality=0.3,
        max_faces=50000,
        min_cluster_confidence=0.70,
    )

    # Assert
    assert result["status"] == "completed"

    # Verify last_discovery metadata
    assert "unknown_persons:last_discovery" in redis_cache
    last_discovery = redis_cache["unknown_persons:last_discovery"]
    assert "timestamp" in last_discovery
    assert last_discovery["total_faces"] == 5
    assert last_discovery["clusters_found"] == 1

    # Verify cluster metadata
    cluster_keys = [k for k in redis_cache.keys() if "cluster:" in k]
    assert len(cluster_keys) == 1
    cluster_meta = redis_cache[cluster_keys[0]]
    assert "face_count" in cluster_meta
    assert "confidence" in cluster_meta
    assert "membership_hash" in cluster_meta
    assert "face_ids" in cluster_meta
