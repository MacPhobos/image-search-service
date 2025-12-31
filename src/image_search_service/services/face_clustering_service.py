"""Face clustering confidence calculation and filtering service."""

import logging
from uuid import UUID

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.db.models import FaceInstance
from image_search_service.vector.face_qdrant import FaceQdrantClient

logger = logging.getLogger(__name__)


class FaceClusteringService:
    """Service for calculating cluster confidence and quality metrics."""

    def __init__(self, db: AsyncSession, qdrant: FaceQdrantClient):
        """Initialize face clustering service.

        Args:
            db: Async database session
            qdrant: Face Qdrant client for retrieving embeddings
        """
        self.db = db
        self.qdrant = qdrant

    async def calculate_cluster_confidence(
        self,
        cluster_id: str,
        qdrant_point_ids: list[UUID] | None = None,
        max_faces_for_calculation: int = 20,
    ) -> float:
        """Calculate average pairwise cosine similarity for faces in cluster.

        This provides a confidence score (0.0-1.0) indicating how similar faces
        in the cluster are to each other. Higher values indicate tighter, more
        confident clusters.

        Algorithm:
        1. Retrieve face embeddings from Qdrant
        2. Calculate pairwise cosine similarities for all face pairs
        3. Return average similarity as confidence score

        Performance optimization:
        - If cluster > max_faces_for_calculation, sample random subset
        - Cache results in memory for repeated calls

        Args:
            cluster_id: Cluster ID to calculate confidence for
            qdrant_point_ids: Optional pre-fetched Qdrant point IDs (optimization)
            max_faces_for_calculation: Maximum faces to use for calculation (sampling)

        Returns:
            Confidence score (0.0-1.0), where 1.0 is perfect similarity

        Raises:
            ValueError: If cluster has no faces or embeddings not found
        """
        # Get Qdrant point IDs if not provided
        if qdrant_point_ids is None:
            query = select(FaceInstance.qdrant_point_id).where(
                FaceInstance.cluster_id == cluster_id
            )
            result = await self.db.execute(query)
            qdrant_point_ids = [row[0] for row in result.all()]

        if not qdrant_point_ids:
            raise ValueError(f"No faces found for cluster {cluster_id}")

        # Single face = perfect confidence
        if len(qdrant_point_ids) == 1:
            return 1.0

        # Sample if cluster too large for performance
        if len(qdrant_point_ids) > max_faces_for_calculation:
            import random

            qdrant_point_ids = random.sample(qdrant_point_ids, max_faces_for_calculation)
            logger.info(
                f"Sampled {max_faces_for_calculation} faces from cluster {cluster_id} "
                f"for confidence calculation"
            )

        # Retrieve embeddings from Qdrant
        embeddings = []
        for point_id in qdrant_point_ids:
            try:
                embedding = self.qdrant.get_embedding_by_point_id(point_id)
                if embedding is not None:
                    embeddings.append(np.array(embedding))
            except Exception as e:
                logger.warning(
                    f"Failed to retrieve embedding for point {point_id} in cluster {cluster_id}: {e}"
                )
                continue

        if len(embeddings) < 2:
            # Not enough embeddings to calculate similarity
            logger.warning(
                f"Cluster {cluster_id} has {len(embeddings)} valid embeddings, "
                "cannot calculate confidence"
            )
            return 0.0

        # Calculate pairwise cosine similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                # Cosine similarity = dot product of normalized vectors
                similarity = self._cosine_similarity(embeddings[i], embeddings[j])
                similarities.append(similarity)

        # Return average similarity as confidence
        confidence = float(np.mean(similarities))
        logger.debug(
            f"Cluster {cluster_id}: confidence={confidence:.3f} "
            f"(from {len(similarities)} pairwise comparisons of {len(embeddings)} faces)"
        )

        return confidence

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector (numpy array)
            vec2: Second vector (numpy array)

        Returns:
            Cosine similarity (-1.0 to 1.0), typically 0.0-1.0 for face embeddings
        """
        # Normalize vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Cosine similarity = dot(v1, v2) / (||v1|| * ||v2||)
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)

        # Clamp to valid range (floating point errors can cause slight overflow)
        return float(np.clip(similarity, -1.0, 1.0))

    async def select_representative_face(
        self,
        cluster_id: str,
        face_ids: list[UUID] | None = None,
    ) -> UUID | None:
        """Select highest quality face from cluster as representative.

        Quality score is based on:
        - quality_score from FaceInstance (if available)
        - Bbox area (width * height) as fallback
        - Detection confidence as tie-breaker

        Args:
            cluster_id: Cluster ID to select representative from
            face_ids: Optional pre-fetched face IDs (optimization)

        Returns:
            Face instance ID of highest quality face, or None if cluster empty
        """
        # Build query for faces in cluster
        query = select(FaceInstance).where(FaceInstance.cluster_id == cluster_id)

        if face_ids is not None:
            query = query.where(FaceInstance.id.in_(face_ids))

        result = await self.db.execute(query)
        faces = result.scalars().all()

        if not faces:
            return None

        # Calculate composite quality score for each face
        def calculate_quality(face: FaceInstance) -> float:
            # Primary: quality_score if available
            quality = face.quality_score if face.quality_score is not None else 0.5

            # Secondary: bbox area (larger faces are usually clearer)
            bbox_area = (face.bbox_w or 0) * (face.bbox_h or 0)
            area_bonus = min(bbox_area / 10000.0, 0.2)  # Up to 0.2 bonus

            # Tertiary: detection confidence
            conf_bonus = (face.detection_confidence or 0.0) * 0.1  # Up to 0.1 bonus

            return quality + area_bonus + conf_bonus

        # Select face with highest composite quality
        best_face = max(faces, key=calculate_quality)

        logger.debug(
            f"Selected face {best_face.id} as representative for cluster {cluster_id} "
            f"(quality={best_face.quality_score}, "
            f"bbox={best_face.bbox_w}x{best_face.bbox_h}, "
            f"confidence={best_face.detection_confidence})"
        )

        return best_face.id
