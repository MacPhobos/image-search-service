"""Shared embedding preprocessing for face clustering pipelines.

Extracts PCA dimensionality reduction and HDBSCAN parameter selection logic
into a reusable module shared by FaceClusterer (clusterer.py) and the discover
job (face_jobs.py). Maintaining a single implementation prevents the two HDBSCAN
call sites from diverging when threshold or variance parameters need tuning.
"""
from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def reduce_dimensions_pca(
    embeddings: NDArray[np.float32],
    target_dim: int = 50,
    random_state: int = 42,
    variance_warning_threshold: float = 0.90,
    job_id: str = "no-job",
) -> tuple[NDArray[np.float32], dict[str, float]]:
    """Reduce embedding dimensionality via PCA for faster HDBSCAN clustering.

    At 512 dimensions, HDBSCAN selects Prim's MST algorithm (the ``dimensions > 60``
    threshold in hdbscan 0.8.x), which is single-threaded and has O(N² × D)
    complexity. Reducing to ≤50 dimensions triggers Boruvka KD-tree algorithm
    selection, which is parallel and uses O(N log²N) complexity.

    The original high-dimensional embeddings are NOT returned; callers that
    need the original vectors for downstream tasks (e.g., cosine similarity
    scoring) must retain them independently.

    PCA is skipped when any of these conditions holds:
    - ``target_dim <= 0`` (disabled by caller)
    - Input dimensionality is already ≤ ``target_dim``
    - Number of samples ≤ ``target_dim`` (PCA constraint: n_components < n_samples)

    Args:
        embeddings: Float32 array of shape ``(n_samples, n_dims)``.
        target_dim: Target number of principal components. Pass ``0`` to disable
            PCA entirely (useful for testing without dimensionality reduction).
            Defaults to 50, which is below hdbscan's 60-dimension threshold for
            Boruvka algorithm selection.
        random_state: Random seed for deterministic PCA. Defaults to 42.
        variance_warning_threshold: Log a warning when retained variance falls
            below this fraction. Defaults to 0.90 (90%).
        job_id: Job identifier included in log messages for traceability.
            Defaults to ``"no-job"`` for calls outside an RQ context.

    Returns:
        A two-tuple ``(reduced_embeddings, stats)`` where:

        - ``reduced_embeddings``: Float32 array of shape ``(n_samples, reduced_dims)``.
          Identical to ``embeddings`` (same object) when PCA is skipped.
        - ``stats``: Dict with the following float keys:

          - ``"original_dims"``: Input dimensionality (before PCA).
          - ``"reduced_dims"``: Output dimensionality (after PCA, or same as input
            when skipped).
          - ``"explained_variance"``: Fraction of variance retained (1.0 when skipped).
          - ``"applied"``: ``1.0`` if PCA was applied, ``0.0`` if skipped.
    """
    from sklearn.decomposition import PCA  # lazy import -- sklearn is a transitive dep

    n_samples, n_dims = embeddings.shape
    stats: dict[str, float] = {
        "original_dims": float(n_dims),
        "reduced_dims": float(n_dims),
        "explained_variance": 1.0,
        "applied": 0.0,
    }

    if target_dim <= 0 or n_dims <= target_dim or n_samples <= target_dim:
        logger.info(
            "[%s] PCA skipped: %d samples × %dd (target=%dd)",
            job_id,
            n_samples,
            n_dims,
            target_dim,
        )
        return embeddings, stats

    # Guard: PCA cannot produce more components than min(n_samples - 1, n_features)
    n_components = min(target_dim, n_samples - 1)
    pca = PCA(n_components=n_components, random_state=random_state)
    reduced = pca.fit_transform(embeddings)

    explained_variance = float(pca.explained_variance_ratio_.sum())
    stats.update(
        {
            "reduced_dims": float(n_components),
            "explained_variance": explained_variance,
            "applied": 1.0,
        }
    )

    logger.info(
        "[%s] PCA: %dd -> %dd (%s variance retained)",
        job_id,
        n_dims,
        n_components,
        f"{explained_variance:.1%}",
    )

    if explained_variance < variance_warning_threshold:
        logger.warning(
            "[%s] PCA retained only %s variance (threshold: %s). "
            "Clustering quality may be degraded.",
            job_id,
            f"{explained_variance:.1%}",
            f"{variance_warning_threshold:.0%}",
        )

    return reduced.astype(np.float32), stats


def select_hdbscan_params(
    n_dims: int,
    pca_applied: bool,
) -> dict[str, object]:
    """Select optimal HDBSCAN algorithm and parallelism based on dimensionality.

    The hdbscan 0.8.x library auto-selects the MST construction algorithm based
    on a ``dimensions > 60`` internal threshold:

    - **≤60 dimensions**: Boruvka KD-tree, O(N log²N), parallelised via
      ``core_dist_n_jobs``.
    - **>60 dimensions**: Prim's MST, O(N² × D), single-threaded (``core_dist_n_jobs``
      is silently ignored at high dimensions).

    When PCA has reduced dimensions to ≤60, this function returns an explicit
    ``"boruvka_kdtree"`` override with ``core_dist_n_jobs=-1``. The explicit
    algorithm name is required because the auto-selection threshold evaluates the
    number of dimensions *before* any internal transformations, and callers may
    pass arrays that land exactly on the boundary.

    .. note::
        When ``algorithm="boruvka_kdtree"`` is set explicitly, hdbscan validates
        the metric against ``BALLTREE_VALID_METRICS`` rather than
        ``KDTREE_VALID_METRICS`` (a library quirk). Since ``BALLTREE_VALID_METRICS``
        is a superset, ``"euclidean"`` works correctly with this explicit override.

    Args:
        n_dims: Number of embedding dimensions *after* any preprocessing (i.e.,
            after PCA reduction, or the original dimension if PCA was skipped).
        pca_applied: Whether PCA was applied to reduce dimensions. When ``True``
            and ``n_dims <= 60``, Boruvka KD-tree is selected explicitly.

    Returns:
        Dict of keyword arguments intended to be unpacked directly into the
        ``hdbscan.HDBSCAN`` constructor via ``**hdbscan_params``. Contains:

        - ``"algorithm"``: Either ``"boruvka_kdtree"`` (fast, parallel) or
          ``"best"`` (HDBSCAN default, Prim's at high dimensions).
        - ``"core_dist_n_jobs"``: ``-1`` (all cores) for Boruvka, ``1`` for Prim's.
          Prim's silently ignores this parameter; setting it to 1 makes the
          intentional single-threading explicit rather than accidental.
    """
    if pca_applied and n_dims <= 60:
        return {
            "algorithm": "boruvka_kdtree",
            "core_dist_n_jobs": -1,  # Boruvka uses this; Prim's ignores it
        }
    else:
        return {
            "algorithm": "best",
            "core_dist_n_jobs": 1,  # Prim's ignores this anyway; explicit for clarity
        }
