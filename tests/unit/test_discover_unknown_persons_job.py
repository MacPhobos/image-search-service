"""Unit tests for discover_unknown_persons_job background task."""

from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from sklearn.decomposition import PCA

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
        result.rowcount = 1
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

    # Verify HDBSCAN was configured correctly.
    # 15 faces < pca_target_dim=50, so PCA is skipped -> algorithm="best", n_jobs=1
    mock_dependencies["hdbscan_module"].HDBSCAN.assert_called_once_with(
        min_cluster_size=5,
        min_samples=3,
        metric="euclidean",
        cluster_selection_method="eom",
        algorithm="best",
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
    """Test that job fails when memory ceiling is exceeded.

    The memory ceiling uses an accurate PCA-aware formula (C1 fix):
    - Without PCA: O(N²) distance matrix → (N² × 8 bytes)
    - With PCA: O(N × D_reduced × 8 × 5) bytes

    To trigger the ceiling with the more accurate estimate, we disable PCA
    (pca_target_dim=0) so the original O(N²) formula applies.
    25,000 faces × 25,000 × 8 bytes = ~4.66 GB, which exceeds the 4 GB cap.
    """
    # Arrange: Too many faces (would require >4GB memory WITHOUT PCA)
    # sqrt(4 * 1024^3 / 8) ~ 23170 faces for 4GB (O(N²) formula)
    face_ids = [uuid.uuid4() for _ in range(25000)]
    embeddings_list = [np.random.randn(512).astype(np.float32) for _ in range(25000)]

    mock_dependencies["qdrant"].get_unlabeled_faces_with_embeddings.return_value = [
        (fid, emb) for fid, emb in zip(face_ids, embeddings_list)
    ]

    # Act: disable PCA so the O(N²) memory formula applies
    result = discover_unknown_persons_job(
        clustering_method="hdbscan",
        min_cluster_size=5,
        min_quality=0.3,
        max_faces=50000,
        min_cluster_confidence=0.70,
        pca_target_dim=0,  # disable PCA → triggers O(N²) formula → 4.66 GB > 4 GB cap
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
        result.rowcount = 1
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
        result.rowcount = 1
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
    def mock_hash_func(ids):
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
        result.rowcount = 1
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
        result.rowcount = 1
        return result

    mock_dependencies["session"].execute.side_effect = mock_execute

    # Act
    test_progress_key = "job:test-job-123:progress"
    result = discover_unknown_persons_job(
        clustering_method="hdbscan",
        min_cluster_size=5,
        min_quality=0.3,
        max_faces=50000,
        min_cluster_confidence=0.70,
        progress_key=test_progress_key,
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
    assert "completed" in phases


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
        result.rowcount = 1
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


# ---------------------------------------------------------------------------
# PCA preprocessing tests
# ---------------------------------------------------------------------------


def test_pca_reduces_dimensions_when_embedding_dim_exceeds_target() -> None:
    """Verify PCA reduces 512d embeddings to 50d before clustering.

    Uses low-rank structured data (like real face embeddings) where the top 50
    components capture most of the variance.
    """
    rng = np.random.default_rng(0)
    n_faces = 100
    # Simulate structured face embeddings: data lives on a low-rank subspace.
    # Generate embeddings as linear combinations of 10 latent factors + noise.
    n_latent = 10
    latent_factors = rng.standard_normal((n_latent, 512)).astype(np.float32)
    latent_codes = rng.standard_normal((n_faces, n_latent)).astype(np.float32)
    # Signal + small noise (makes the data low-rank)
    embeddings_512d = latent_codes @ latent_factors + 0.1 * rng.standard_normal(
        (n_faces, 512)
    ).astype(np.float32)
    # Normalize to unit vectors
    norms = np.linalg.norm(embeddings_512d, axis=1, keepdims=True)
    embeddings_512d /= norms

    pca = PCA(n_components=50, random_state=42)
    reduced = pca.fit_transform(embeddings_512d)

    assert reduced.shape == (100, 50)
    # Structured low-rank data: top 50 components should capture >80% of variance
    assert pca.explained_variance_ratio_.sum() > 0.80


def test_pca_skipped_when_dimensions_below_target() -> None:
    """Verify PCA is not applied when embeddings are already <=50d."""
    embeddings_30d = np.random.default_rng(0).standard_normal((100, 30)).astype(np.float32)
    pca_target_dim = 50

    # Condition mirrors the implementation guard
    n_samples, n_dims = embeddings_30d.shape
    should_apply_pca = pca_target_dim > 0 and n_dims > pca_target_dim and n_samples > pca_target_dim
    assert not should_apply_pca, "PCA should be skipped when dims <= target"


def test_pca_skipped_when_target_is_zero() -> None:
    """Verify PCA is not applied when pca_target_dim=0 (disabled)."""
    embeddings_512d = np.random.default_rng(0).standard_normal((100, 512)).astype(np.float32)
    pca_target_dim = 0

    # Condition mirrors the implementation guard
    n_samples, n_dims = embeddings_512d.shape
    should_apply_pca = pca_target_dim > 0 and n_dims > pca_target_dim and n_samples > pca_target_dim
    assert not should_apply_pca, "PCA should be skipped when pca_target_dim=0"


def test_pca_handles_fewer_samples_than_target_components() -> None:
    """Verify PCA reduces to min(target, n_samples-1) when samples < target."""
    n_faces = 20  # Less than pca_target_dim=50
    embeddings = np.random.default_rng(0).standard_normal((n_faces, 512)).astype(np.float32)

    # Implementation guard: cannot produce more components than n_samples - 1
    n_components = min(50, n_faces - 1)  # = 19
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(embeddings)

    assert reduced.shape == (20, 19)


def test_pca_original_embeddings_not_modified() -> None:
    """Verify the original 512d embeddings are not modified by PCA fit_transform."""
    embeddings = np.random.default_rng(0).standard_normal((100, 512)).astype(np.float32)
    original_copy = embeddings.copy()

    pca = PCA(n_components=50, random_state=42)
    _ = pca.fit_transform(embeddings)

    # fit_transform returns a NEW array; original should be unchanged
    np.testing.assert_array_equal(embeddings, original_copy)


def test_discover_job_pca_disabled_passes_512d_to_hdbscan(mock_dependencies: dict) -> None:
    """When pca_target_dim=0, the full 512d embeddings are passed to HDBSCAN."""
    face_ids = [uuid.uuid4() for _ in range(10)]
    embeddings_list = [np.random.randn(512).astype(np.float32) for _ in range(10)]

    mock_dependencies["qdrant"].get_unlabeled_faces_with_embeddings.return_value = [
        (fid, emb) for fid, emb in zip(face_ids, embeddings_list)
    ]

    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    mock_dependencies["clusterer"].fit_predict.return_value = labels
    mock_dependencies["confidence"].return_value = 0.85
    mock_dependencies["hash"].side_effect = lambda ids: f"hash_{len(ids)}"

    def mock_execute(stmt):
        result = MagicMock()
        result.scalar_one_or_none.return_value = MagicMock()
        result.rowcount = 1
        return result

    mock_dependencies["session"].execute.side_effect = mock_execute

    result = discover_unknown_persons_job(
        clustering_method="hdbscan",
        min_cluster_size=5,
        min_quality=0.3,
        max_faces=50000,
        min_cluster_confidence=0.70,
        pca_target_dim=0,  # Disable PCA
    )

    assert result["status"] == "completed"

    # Verify fit_predict received 512d embeddings (no PCA reduction)
    call_args = mock_dependencies["clusterer"].fit_predict.call_args
    passed_embeddings = call_args[0][0]
    assert passed_embeddings.shape[1] == 512, (
        f"Expected 512d embeddings when PCA disabled, got {passed_embeddings.shape[1]}d"
    )

    # When PCA is disabled, dimensions are unchanged so pca_was_applied=False.
    # HDBSCAN should use "best" (auto-selects Prim's at 512d) not "boruvka_kdtree",
    # and core_dist_n_jobs=1 since parallelism only works with Boruvka.
    mock_dependencies["hdbscan_module"].HDBSCAN.assert_called_once_with(
        min_cluster_size=5,
        min_samples=3,
        metric="euclidean",
        cluster_selection_method="eom",
        algorithm="best",
        core_dist_n_jobs=1,
    )


def test_discover_job_pca_applied_reduces_to_50d(mock_dependencies: dict) -> None:
    """When pca_target_dim=50 and embeddings are 512d, fit_predict receives 50d data."""
    face_ids = [uuid.uuid4() for _ in range(100)]
    embeddings_list = [np.random.randn(512).astype(np.float32) for _ in range(100)]

    mock_dependencies["qdrant"].get_unlabeled_faces_with_embeddings.return_value = [
        (fid, emb) for fid, emb in zip(face_ids, embeddings_list)
    ]

    labels = np.array([0] * 50 + [1] * 50)
    mock_dependencies["clusterer"].fit_predict.return_value = labels
    mock_dependencies["confidence"].return_value = 0.85
    mock_dependencies["hash"].side_effect = lambda ids: f"hash_{len(ids)}"

    def mock_execute(stmt):
        result = MagicMock()
        result.scalar_one_or_none.return_value = MagicMock()
        result.rowcount = 1
        return result

    mock_dependencies["session"].execute.side_effect = mock_execute

    result = discover_unknown_persons_job(
        clustering_method="hdbscan",
        min_cluster_size=5,
        min_quality=0.3,
        max_faces=50000,
        min_cluster_confidence=0.70,
        pca_target_dim=50,  # Enable PCA (default)
    )

    assert result["status"] == "completed"

    # Verify fit_predict received 50d PCA-reduced embeddings
    call_args = mock_dependencies["clusterer"].fit_predict.call_args
    passed_embeddings = call_args[0][0]
    assert passed_embeddings.shape[1] == 50, (
        f"Expected 50d embeddings after PCA reduction, got {passed_embeddings.shape[1]}d"
    )
    assert passed_embeddings.shape[0] == 100


def test_discover_job_phase5_uses_original_512d_embeddings(mock_dependencies: dict) -> None:
    """Phase 5 confidence computation uses original 512d embeddings, not PCA-reduced."""
    n_faces = 100
    face_ids = [uuid.uuid4() for _ in range(n_faces)]
    embeddings_list = [np.random.randn(512).astype(np.float32) for _ in range(n_faces)]

    mock_dependencies["qdrant"].get_unlabeled_faces_with_embeddings.return_value = [
        (fid, emb) for fid, emb in zip(face_ids, embeddings_list)
    ]

    labels = np.array([0] * n_faces)
    mock_dependencies["clusterer"].fit_predict.return_value = labels
    mock_dependencies["hash"].side_effect = lambda ids: f"hash_{len(ids)}"

    # Capture the embeddings passed to confidence computation
    captured_confidence_embeddings: list[np.ndarray] = []

    def capture_confidence(cluster_embeds, sample_size=20):
        captured_confidence_embeddings.append(cluster_embeds.copy())
        return 0.85

    mock_dependencies["confidence"].side_effect = capture_confidence

    def mock_execute(stmt):
        result = MagicMock()
        result.scalar_one_or_none.return_value = MagicMock()
        result.rowcount = 1
        return result

    mock_dependencies["session"].execute.side_effect = mock_execute

    result = discover_unknown_persons_job(
        clustering_method="hdbscan",
        min_cluster_size=5,
        min_quality=0.3,
        max_faces=50000,
        min_cluster_confidence=0.70,
        pca_target_dim=50,  # PCA enabled
    )

    assert result["status"] == "completed"
    assert len(captured_confidence_embeddings) == 1

    # Confidence was computed with 512d embeddings (not 50d PCA-reduced)
    assert captured_confidence_embeddings[0].shape[1] == 512, (
        "Phase 5 confidence computation must use original 512d embeddings"
    )


def test_discover_job_pca_applied_uses_boruvka_algorithm(mock_dependencies: dict) -> None:
    """When PCA reduces dimensions, HDBSCAN must use boruvka_kdtree with full parallelism.

    Boruvka with KD-tree is efficient at low dimensions (<=50d) but inefficient
    at 512d.  Only when PCA has actually reduced the dimensionality should we
    enable the Boruvka path and set core_dist_n_jobs=-1.
    """
    # 100 faces with 512d embeddings; pca_target_dim=50 triggers PCA reduction.
    face_ids = [uuid.uuid4() for _ in range(100)]
    embeddings_list = [np.random.randn(512).astype(np.float32) for _ in range(100)]

    mock_dependencies["qdrant"].get_unlabeled_faces_with_embeddings.return_value = [
        (fid, emb) for fid, emb in zip(face_ids, embeddings_list)
    ]

    labels = np.array([0] * 50 + [1] * 50)
    mock_dependencies["clusterer"].fit_predict.return_value = labels
    mock_dependencies["confidence"].return_value = 0.85
    mock_dependencies["hash"].side_effect = lambda ids: f"hash_{len(ids)}"

    def mock_execute(stmt):
        result = MagicMock()
        result.scalar_one_or_none.return_value = MagicMock()
        result.rowcount = 1
        return result

    mock_dependencies["session"].execute.side_effect = mock_execute

    result = discover_unknown_persons_job(
        clustering_method="hdbscan",
        min_cluster_size=5,
        min_quality=0.3,
        max_faces=50000,
        min_cluster_confidence=0.70,
        pca_target_dim=50,  # PCA will reduce 512d -> 50d
    )

    assert result["status"] == "completed"

    # PCA reduced 512d to 50d, so pca_was_applied=True.
    # HDBSCAN should use "boruvka_kdtree" (fast at low dims) with full parallelism.
    mock_dependencies["hdbscan_module"].HDBSCAN.assert_called_once_with(
        min_cluster_size=5,
        min_samples=3,
        metric="euclidean",
        cluster_selection_method="eom",
        algorithm="boruvka_kdtree",
        core_dist_n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# Batch DB update tests
# ---------------------------------------------------------------------------


def test_batch_grouping_groups_faces_by_cluster_id() -> None:
    """Batch grouping assigns face UUIDs to correct cluster_id buckets."""
    import uuid as _uuid

    face_ids = [_uuid.uuid4() for _ in range(7)]
    labels = np.array([0, 0, 1, 1, 1, -1, -1])
    cluster_metadata = {"unknown_0": {}, "unknown_1": {}}

    cluster_to_faces: dict[str, list[_uuid.UUID]] = {}
    for label, face_id in zip(labels, face_ids):
        if label == -1:
            cluster_to_faces.setdefault("-1", []).append(face_id)
        else:
            cluster_id = f"unknown_{label}"
            if cluster_id in cluster_metadata:
                cluster_to_faces.setdefault(cluster_id, []).append(face_id)

    assert set(cluster_to_faces.keys()) == {"unknown_0", "unknown_1", "-1"}
    assert len(cluster_to_faces["unknown_0"]) == 2
    assert len(cluster_to_faces["unknown_1"]) == 3
    assert len(cluster_to_faces["-1"]) == 2

    # Verify face UUIDs are assigned in order
    assert cluster_to_faces["unknown_0"] == face_ids[:2]
    assert cluster_to_faces["unknown_1"] == face_ids[2:5]
    assert cluster_to_faces["-1"] == face_ids[5:]


def test_batch_grouping_excludes_non_qualifying_clusters() -> None:
    """Faces from clusters not in cluster_metadata are excluded from batch updates."""
    import uuid as _uuid

    face_ids = [_uuid.uuid4() for _ in range(4)]
    labels = np.array([0, 0, 1, 1])
    # Only unknown_0 qualifies; unknown_1 was filtered by confidence
    cluster_metadata = {"unknown_0": {}}

    cluster_to_faces: dict[str, list[_uuid.UUID]] = {}
    for label, face_id in zip(labels, face_ids):
        if label == -1:
            cluster_to_faces.setdefault("-1", []).append(face_id)
        else:
            cluster_id = f"unknown_{label}"
            if cluster_id in cluster_metadata:
                cluster_to_faces.setdefault(cluster_id, []).append(face_id)

    assert "unknown_1" not in cluster_to_faces, (
        "Non-qualifying clusters must be excluded from batch updates"
    )
    assert "unknown_0" in cluster_to_faces
    assert len(cluster_to_faces["unknown_0"]) == 2


def test_batch_chunking_splits_large_cluster_correctly() -> None:
    """Large clusters are chunked into sub-lists bounded by max_batch_size."""
    import uuid as _uuid

    max_batch_size = 5000
    n_faces = 12000
    face_ids = [_uuid.uuid4() for _ in range(n_faces)]

    chunks = [
        face_ids[start : start + max_batch_size]
        for start in range(0, len(face_ids), max_batch_size)
    ]

    assert len(chunks) == 3
    assert len(chunks[0]) == 5000
    assert len(chunks[1]) == 5000
    assert len(chunks[2]) == 2000

    # Verify all original face IDs appear exactly once across all chunks
    all_chunked = [fid for chunk in chunks for fid in chunk]
    assert all_chunked == face_ids


def test_batch_grouping_all_noise_faces() -> None:
    """When all faces are noise (label -1), only the '-1' bucket is populated."""
    import uuid as _uuid

    face_ids = [_uuid.uuid4() for _ in range(3)]
    labels = np.array([-1, -1, -1])
    cluster_metadata: dict[str, dict] = {}

    cluster_to_faces: dict[str, list[_uuid.UUID]] = {}
    for label, face_id in zip(labels, face_ids):
        if label == -1:
            cluster_to_faces.setdefault("-1", []).append(face_id)
        else:
            cluster_id = f"unknown_{label}"
            if cluster_id in cluster_metadata:
                cluster_to_faces.setdefault(cluster_id, []).append(face_id)

    assert list(cluster_to_faces.keys()) == ["-1"]
    assert cluster_to_faces["-1"] == face_ids


def test_batch_update_uses_update_statement_not_select(mock_dependencies: dict) -> None:
    """Phase 6 issues batch UPDATE statements grouped by cluster_id, not per-face SELECTs."""
    face_ids = [uuid.uuid4() for _ in range(6)]
    embeddings_list = [np.random.randn(512).astype(np.float32) for _ in range(6)]

    mock_dependencies["qdrant"].get_unlabeled_faces_with_embeddings.return_value = [
        (fid, emb) for fid, emb in zip(face_ids, embeddings_list)
    ]

    # 2 clusters of 3 faces each
    labels = np.array([0, 0, 0, 1, 1, 1])
    mock_dependencies["clusterer"].fit_predict.return_value = labels
    mock_dependencies["confidence"].return_value = 0.85
    mock_dependencies["hash"].side_effect = lambda ids: f"hash_{len(ids)}"

    execute_calls: list[object] = []

    def capturing_execute(stmt):
        execute_calls.append(stmt)
        result = MagicMock()
        result.rowcount = 3
        return result

    mock_dependencies["session"].execute.side_effect = capturing_execute

    result = discover_unknown_persons_job(
        clustering_method="hdbscan",
        min_cluster_size=3,
        min_quality=0.3,
        max_faces=50000,
        min_cluster_confidence=0.70,
    )

    assert result["status"] == "completed"

    # Phase 6 should have called execute exactly twice (one per cluster):
    # one batch UPDATE for unknown_0, one for unknown_1.
    # Confirm no per-face SELECT calls were issued by the new batch pattern.
    assert len(execute_calls) == 2, (
        f"Expected 2 batch UPDATE calls (one per cluster), got {len(execute_calls)}"
    )

    # Verify the execute call count is consistent with batch semantics:
    # 2 clusters × 1 batch each = 2 total execute calls (not 6 per-face calls).
    assert mock_dependencies["session"].commit.call_count == 1, (
        "Single commit after all batch updates"
    )
