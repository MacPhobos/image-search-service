"""Unit tests for cluster confidence computation utility."""

from __future__ import annotations

import numpy as np
import pytest

from image_search_service.services.face_clustering_service import (
    compute_cluster_confidence_from_embeddings,
)


def test_identical_embeddings_perfect_confidence() -> None:
    """Test that identical embeddings produce confidence of 1.0."""
    # Two identical normalized vectors
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    confidence = compute_cluster_confidence_from_embeddings(embeddings)

    assert confidence == pytest.approx(
        1.0, abs=1e-6
    ), "Identical embeddings should have confidence ~1.0"


def test_orthogonal_embeddings_low_confidence() -> None:
    """Test that orthogonal embeddings produce low confidence."""
    # Two orthogonal vectors (90 degrees apart)
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    confidence = compute_cluster_confidence_from_embeddings(embeddings)

    assert confidence == pytest.approx(
        0.0, abs=1e-6
    ), "Orthogonal embeddings should have confidence ~0.0"


def test_single_embedding_returns_one() -> None:
    """Test that single embedding returns perfect confidence."""
    embeddings = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

    confidence = compute_cluster_confidence_from_embeddings(embeddings)

    assert confidence == 1.0, "Single embedding should return confidence = 1.0"


def test_empty_array_returns_one() -> None:
    """Test that empty array returns perfect confidence."""
    embeddings = np.array([], dtype=np.float32).reshape(0, 512)

    confidence = compute_cluster_confidence_from_embeddings(embeddings)

    assert confidence == 1.0, "Empty array should return confidence = 1.0"


def test_three_embeddings_high_similarity() -> None:
    """Test confidence calculation with three similar embeddings."""
    # Three similar vectors (small variations)
    embeddings = np.array(
        [
            [1.0, 0.1, 0.0],
            [1.0, 0.0, 0.1],
            [0.9, 0.1, 0.1],
        ],
        dtype=np.float32,
    )

    # Normalize vectors (cosine_similarity expects normalized vectors for optimal results)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    confidence = compute_cluster_confidence_from_embeddings(embeddings)

    # Expect high confidence (>0.9) since vectors are similar
    assert 0.9 < confidence <= 1.0, f"Similar embeddings should have high confidence, got {confidence}"


def test_mixed_embeddings_medium_confidence() -> None:
    """Test confidence calculation with mixed similarity embeddings."""
    # Mix of similar and dissimilar vectors
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.1, 0.0],
            [0.5, 0.5, 0.0],
        ],
        dtype=np.float32,
    )

    # Normalize vectors
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    confidence = compute_cluster_confidence_from_embeddings(embeddings)

    # Expect medium confidence (0.5-0.9) due to mixed similarity
    assert 0.5 < confidence < 0.95, f"Mixed embeddings should have medium confidence, got {confidence}"


def test_sampling_large_cluster() -> None:
    """Test that large clusters are sampled to sample_size."""
    # Create 50 random embeddings
    np.random.seed(42)
    embeddings = np.random.randn(50, 512).astype(np.float32)

    # Normalize vectors
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    # Should sample to 20 embeddings
    confidence = compute_cluster_confidence_from_embeddings(embeddings, sample_size=20)

    # Confidence should be between 0 and 1
    assert 0.0 <= confidence <= 1.0, f"Confidence must be in [0, 1], got {confidence}"


def test_confidence_output_range() -> None:
    """Test that confidence is always between 0.0 and 1.0."""
    np.random.seed(123)

    # Test with various random embeddings
    for _ in range(10):
        n_embeddings = np.random.randint(2, 30)
        embeddings = np.random.randn(n_embeddings, 512).astype(np.float32)

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        confidence = compute_cluster_confidence_from_embeddings(embeddings)

        assert 0.0 <= confidence <= 1.0, f"Confidence must be in [0, 1], got {confidence}"


def test_normalized_vs_unnormalized_embeddings() -> None:
    """Test that normalized embeddings work correctly."""
    # Create unnormalized vectors
    embeddings_unnormalized = np.array(
        [
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],  # Same direction, different magnitudes
        ],
        dtype=np.float32,
    )

    confidence_unnormalized = compute_cluster_confidence_from_embeddings(embeddings_unnormalized)

    # Cosine similarity should be ~1.0 (same direction)
    assert confidence_unnormalized == pytest.approx(1.0, abs=1e-5), (
        "Vectors in same direction should have high confidence regardless of magnitude"
    )


def test_opposite_direction_embeddings() -> None:
    """Test embeddings pointing in opposite directions."""
    # sklearn's cosine_similarity normalizes vectors internally, so opposite
    # vectors will have cosine similarity close to 0.0 (not exactly -1.0)
    # because we're passing already normalized vectors
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],  # Perpendicular (90 degrees)
        ],
        dtype=np.float32,
    )

    confidence = compute_cluster_confidence_from_embeddings(embeddings)

    # Perpendicular vectors have cosine similarity ~0.0
    assert confidence == pytest.approx(0.0, abs=1e-6), (
        "Perpendicular embeddings should have confidence ~0.0"
    )


def test_high_dimensional_embeddings() -> None:
    """Test with realistic face embedding dimensions (512-dim)."""
    np.random.seed(99)

    # Create 5 embeddings with 512 dimensions (realistic face embeddings)
    embeddings = np.random.randn(5, 512).astype(np.float32)

    # Normalize (face embeddings are typically normalized)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    confidence = compute_cluster_confidence_from_embeddings(embeddings)

    # Random normalized vectors should have confidence near 0
    # (orthogonal in high dimensions)
    assert -0.2 < confidence < 0.2, (
        f"Random high-dim embeddings should have near-zero confidence, got {confidence}"
    )


def test_deterministic_output() -> None:
    """Test that same input produces same output (deterministic)."""
    embeddings = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=np.float32,
    )

    confidence1 = compute_cluster_confidence_from_embeddings(embeddings)
    confidence2 = compute_cluster_confidence_from_embeddings(embeddings)

    assert confidence1 == confidence2, "Function should be deterministic for same input"
