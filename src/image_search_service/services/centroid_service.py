"""Centroid management service for person face recognition.

This service handles computation and maintenance of person centroid embeddings,
which are robust average face vectors used for improved face→person matching.

Key features:
- Global centroid computation with outlier trimming
- Versioning and staleness detection
- Multi-centroid support (future: cluster-based centroids)
- Efficient centroid-based face similarity search
"""

import hashlib
import logging
import uuid
from typing import Any

import numpy as np
import numpy.typing as npt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.core.config import get_settings
from image_search_service.db.models import (
    CentroidStatus,
    CentroidType,
    FaceInstance,
    PersonCentroid,
)
from image_search_service.vector.centroid_qdrant import CentroidQdrantClient
from image_search_service.vector.face_qdrant import FaceQdrantClient

logger = logging.getLogger(__name__)


def compute_global_centroid(
    embeddings: npt.NDArray[np.float32],
    trim_outliers: bool = True,
    trim_threshold_small: float = 0.05,
    trim_threshold_large: float = 0.10,
) -> npt.NDArray[np.float32]:
    """Compute robust global centroid from face embeddings.

    Uses mean with optional outlier trimming based on cosine similarity.

    Trimming rules:
    - n < 50: no trimming (insufficient data)
    - 50 <= n <= 300: trim bottom `trim_threshold_small` (default 5%)
    - n > 300: trim bottom `trim_threshold_large` (default 10%)

    Args:
        embeddings: Array of shape (n_faces, 512)
        trim_outliers: Whether to remove outliers before computing mean
        trim_threshold_small: Trim percentage for 50-300 faces
        trim_threshold_large: Trim percentage for 300+ faces

    Returns:
        Centroid vector of shape (512,)
    """
    n_faces = embeddings.shape[0]

    # No trimming for small sample sizes
    if not trim_outliers or n_faces < 50:
        centroid_mean = np.mean(embeddings, axis=0)
        # Normalize to unit vector
        norm = np.linalg.norm(centroid_mean)
        normalized = centroid_mean / norm
        result_early: npt.NDArray[np.float32] = normalized.astype(np.float32)
        return result_early

    # Compute initial mean
    initial_mean = np.mean(embeddings, axis=0)
    initial_mean_norm = initial_mean / np.linalg.norm(initial_mean)

    # Compute cosine similarities to initial mean
    # cosine_sim = (A · B) / (||A|| * ||B||)
    # Since embeddings are already normalized, this simplifies to dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normalized = embeddings / norms
    similarities = embeddings_normalized @ initial_mean_norm

    # Determine trim percentage based on sample size
    if n_faces <= 300:
        trim_pct = trim_threshold_small
    else:
        trim_pct = trim_threshold_large

    # Sort by similarity and trim bottom percentile
    n_trim = int(n_faces * trim_pct)
    if n_trim > 0:
        # Get indices of top (1 - trim_pct) most similar faces
        sorted_indices = np.argsort(similarities)[::-1]  # Descending order
        keep_indices = sorted_indices[: n_faces - n_trim]
        trimmed_embeddings = embeddings[keep_indices]

        logger.debug(
            f"Trimmed {n_trim}/{n_faces} outliers ({trim_pct * 100:.1f}%) "
            f"based on similarity to initial mean"
        )
    else:
        trimmed_embeddings = embeddings

    # Compute final centroid from trimmed set
    final_mean = np.mean(trimmed_embeddings, axis=0)
    # Normalize to unit vector
    norm = np.linalg.norm(final_mean)
    normalized = final_mean / norm
    result: npt.NDArray[np.float32] = normalized.astype(np.float32)
    return result


def compute_source_hash(face_ids: list[uuid.UUID]) -> str:
    """Compute deterministic hash of face IDs for staleness detection.

    Args:
        face_ids: List of face instance UUIDs

    Returns:
        16-character hex hash string
    """
    # Sort face IDs for deterministic hash
    sorted_ids = sorted(str(fid) for fid in face_ids)
    hash_input = ":".join(sorted_ids)
    return hashlib.sha256(hash_input.encode()).hexdigest()[:16]


def is_centroid_stale(
    centroid: PersonCentroid,
    current_face_ids: list[uuid.UUID],
    current_model_version: str,
    current_algorithm_version: int,
) -> bool:
    """Check if centroid needs recomputation.

    Staleness triggers:
    1. Model version mismatch (embeddings incompatible)
    2. Algorithm version mismatch (computation method changed)
    3. Source face IDs changed (new/removed assignments)

    Args:
        centroid: Existing PersonCentroid record
        current_face_ids: Current face instance IDs for this person
        current_model_version: Current embedding model version
        current_algorithm_version: Current centroid algorithm version

    Returns:
        True if centroid should be recomputed
    """
    # Check model version
    if centroid.model_version != current_model_version:
        logger.debug(
            f"Centroid {centroid.centroid_id} stale: model version mismatch "
            f"({centroid.model_version} != {current_model_version})"
        )
        return True

    # Check algorithm version
    if centroid.centroid_version != current_algorithm_version:
        logger.debug(
            f"Centroid {centroid.centroid_id} stale: algorithm version mismatch "
            f"({centroid.centroid_version} != {current_algorithm_version})"
        )
        return True

    # Check source face IDs hash
    current_hash = compute_source_hash(current_face_ids)
    if centroid.source_face_ids_hash != current_hash:
        logger.debug(
            f"Centroid {centroid.centroid_id} stale: face assignments changed "
            f"({centroid.source_face_ids_hash} != {current_hash})"
        )
        return True

    return False


async def get_person_face_embeddings(
    db: AsyncSession,
    qdrant: FaceQdrantClient,
    person_id: uuid.UUID,
) -> tuple[list[uuid.UUID], list[list[float]]]:
    """Retrieve face embeddings for a person from Qdrant.

    Args:
        db: Database session
        qdrant: Qdrant face client
        person_id: Person UUID

    Returns:
        Tuple of (face_instance_ids, embeddings)
    """
    # Get face instances for this person from database
    query = select(FaceInstance).where(FaceInstance.person_id == person_id)
    result = await db.execute(query)
    faces = result.scalars().all()

    if not faces:
        logger.warning(f"No face instances found for person {person_id}")
        return [], []

    # Collect all Qdrant point IDs
    point_ids = [face.qdrant_point_id for face in faces]

    # Batch retrieve embeddings from Qdrant
    embeddings_map = qdrant.get_embeddings_batch(point_ids)

    # Build result lists, maintaining order
    face_ids = []
    embeddings = []

    for face in faces:
        embedding = embeddings_map.get(face.qdrant_point_id)
        if embedding is not None:
            face_ids.append(face.id)
            embeddings.append(embedding)
        else:
            logger.warning(
                f"Embedding not found in Qdrant for face {face.id} "
                f"(point_id={face.qdrant_point_id})"
            )

    logger.info(f"Retrieved {len(embeddings)} embeddings for person {person_id}")
    return face_ids, embeddings


async def deprecate_centroids(
    db: AsyncSession,
    centroid_qdrant: CentroidQdrantClient,
    person_id: uuid.UUID,
) -> int:
    """Mark all active centroids for a person as deprecated.

    Does NOT delete from Qdrant, just marks as deprecated in DB.
    Qdrant centroids will be overwritten with new versions.

    Args:
        db: Database session
        centroid_qdrant: Centroid Qdrant client
        person_id: Person UUID

    Returns:
        Number of centroids deprecated
    """
    # Find all active centroids for this person
    query = select(PersonCentroid).where(
        PersonCentroid.person_id == person_id, PersonCentroid.status == CentroidStatus.ACTIVE
    )
    result = await db.execute(query)
    centroids = result.scalars().all()

    # Mark as deprecated
    deprecated_count = 0
    for centroid in centroids:
        centroid.status = CentroidStatus.DEPRECATED
        deprecated_count += 1

    if deprecated_count > 0:
        await db.flush()
        logger.info(f"Deprecated {deprecated_count} centroids for person {person_id}")

    return deprecated_count


async def compute_centroids_for_person(
    db: AsyncSession,
    face_qdrant: FaceQdrantClient,
    centroid_qdrant: CentroidQdrantClient,
    person_id: uuid.UUID,
    force_rebuild: bool = False,
) -> PersonCentroid | None:
    """Compute and store global centroid for a person.

    This is the main entry point for centroid computation. It:
    1. Checks if existing centroid is stale (if not forcing rebuild)
    2. Retrieves face embeddings from Qdrant
    3. Deprecates old centroids BEFORE creating new one (avoids unique constraint violation)
    4. Computes robust centroid with outlier trimming
    5. Stores centroid in DB and Qdrant

    Args:
        db: Database session
        face_qdrant: Face Qdrant client
        centroid_qdrant: Centroid Qdrant client
        person_id: Person UUID
        force_rebuild: If True, recompute even if centroid is fresh

    Returns:
        PersonCentroid record if successful, None if insufficient faces
    """
    settings = get_settings()

    # Get current face embeddings
    face_ids, embeddings = await get_person_face_embeddings(db, face_qdrant, person_id)

    # Check minimum face requirement
    if len(face_ids) < settings.centroid_min_faces:
        logger.info(
            f"Insufficient faces for centroid computation: "
            f"{len(face_ids)} < {settings.centroid_min_faces}"
        )
        return None

    # Check if existing centroid is fresh (unless forcing rebuild)
    if not force_rebuild:
        existing_query = (
            select(PersonCentroid)
            .where(
                PersonCentroid.person_id == person_id,
                PersonCentroid.status == CentroidStatus.ACTIVE,
                PersonCentroid.centroid_type == CentroidType.GLOBAL,
            )
            .order_by(PersonCentroid.created_at.desc())
            .limit(1)
        )
        existing_result = await db.execute(existing_query)
        existing_centroid = existing_result.scalar_one_or_none()

        if existing_centroid:
            if not is_centroid_stale(
                existing_centroid,
                face_ids,
                settings.centroid_model_version,
                settings.centroid_algorithm_version,
            ):
                logger.debug(f"Existing centroid {existing_centroid.centroid_id} is fresh")
                return existing_centroid

    # Deprecate old centroids BEFORE creating new one to avoid unique constraint violation
    # The unique index is partial (status = 'active'), so we can't have two active centroids
    # with the same (person_id, model_version, centroid_version, centroid_type, cluster_label)
    await deprecate_centroids(db, centroid_qdrant, person_id)

    # Compute centroid
    embeddings_array = np.array(embeddings, dtype=np.float32)
    centroid_vector = compute_global_centroid(
        embeddings_array,
        trim_outliers=True,
        trim_threshold_small=settings.centroid_trim_threshold_small,
        trim_threshold_large=settings.centroid_trim_threshold_large,
    )

    # Create PersonCentroid record
    centroid_id = uuid.uuid4()
    qdrant_point_id = centroid_id  # Use same UUID for 1:1 mapping

    source_hash = compute_source_hash(face_ids)

    build_params: dict[str, Any] = {
        "trim_outliers": True,
        "trim_threshold_small": settings.centroid_trim_threshold_small,
        "trim_threshold_large": settings.centroid_trim_threshold_large,
        "n_faces_used": len(face_ids),
    }

    centroid = PersonCentroid(
        centroid_id=centroid_id,
        person_id=person_id,
        qdrant_point_id=qdrant_point_id,
        model_version=settings.centroid_model_version,
        centroid_version=settings.centroid_algorithm_version,
        centroid_type=CentroidType.GLOBAL,
        cluster_label="global",
        n_faces=len(face_ids),
        status=CentroidStatus.ACTIVE,
        source_face_ids_hash=source_hash,
        build_params=build_params,
    )

    db.add(centroid)
    await db.flush()

    # Store centroid in Qdrant
    qdrant_payload: dict[str, Any] = {
        "person_id": person_id,
        "centroid_id": centroid_id,
        "model_version": settings.centroid_model_version,
        "centroid_version": settings.centroid_algorithm_version,
        "centroid_type": "global",
        "cluster_label": "global",
        "n_faces": len(face_ids),
        "created_at": centroid.created_at.isoformat(),
        "source_hash": source_hash,
        "build_params": build_params,
    }

    try:
        centroid_qdrant.upsert_centroid(
            centroid_id=centroid_id,
            vector=centroid_vector.tolist(),
            payload=qdrant_payload,
        )
        logger.info(
            f"Created centroid {centroid_id} for person {person_id} from {len(face_ids)} faces"
        )
    except Exception as e:
        logger.error(f"Failed to upsert centroid to Qdrant: {e}")
        # Mark centroid as failed in DB
        centroid.status = CentroidStatus.FAILED
        await db.flush()
        raise

    return centroid


async def get_active_centroid(
    db: AsyncSession,
    person_id: uuid.UUID,
    centroid_type: CentroidType = CentroidType.GLOBAL,
) -> PersonCentroid | None:
    """Get active centroid for a person.

    Args:
        db: Database session
        person_id: Person UUID
        centroid_type: Type of centroid (default: GLOBAL)

    Returns:
        Active PersonCentroid or None if not found
    """
    query = (
        select(PersonCentroid)
        .where(
            PersonCentroid.person_id == person_id,
            PersonCentroid.status == CentroidStatus.ACTIVE,
            PersonCentroid.centroid_type == centroid_type,
        )
        .order_by(PersonCentroid.created_at.desc())
        .limit(1)
    )
    result = await db.execute(query)
    return result.scalar_one_or_none()
