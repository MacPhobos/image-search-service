"""Face clustering using HDBSCAN for grouping similar faces into identity clusters."""

import logging
import uuid
from typing import Optional

import numpy as np
from qdrant_client.models import FieldCondition, Filter, IsEmptyCondition, PayloadField, Range
from sqlalchemy import update
from sqlalchemy.orm import Session as SyncSession

logger = logging.getLogger(__name__)


class FaceClusterer:
    """Clusters unlabeled faces using HDBSCAN algorithm."""

    def __init__(
        self,
        db_session: SyncSession,
        min_cluster_size: int = 5,
        min_samples: int = 3,
        cluster_selection_epsilon: float = 0.0,
        metric: str = "euclidean",  # HDBSCAN works well with normalized vectors + euclidean
    ):
        """Initialize face clusterer.

        Args:
            db_session: Synchronous SQLAlchemy session
            min_cluster_size: Minimum cluster size for HDBSCAN
            min_samples: Minimum samples for core point
            cluster_selection_epsilon: Cluster selection epsilon (0 = default behavior)
            metric: Distance metric (euclidean or cosine)
        """
        self.db = db_session
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.metric = metric

    def cluster_unlabeled_faces(
        self,
        quality_threshold: float = 0.5,
        max_faces: int = 50000,
        time_bucket: Optional[str] = None,  # e.g., "2024-01" for filtering by year-month
    ) -> dict:
        """Cluster all unlabeled faces into identity clusters.

        Args:
            quality_threshold: Minimum quality score for faces to include
            max_faces: Maximum number of faces to cluster at once
            time_bucket: Optional YYYY-MM filter for time-bounded clustering

        Returns:
            Summary dict with cluster counts and statistics
        """
        from image_search_service.db.models import FaceInstance
        from image_search_service.vector.face_qdrant import get_face_qdrant_client

        logger.info(
            f"Starting face clustering (quality>={quality_threshold}, max={max_faces})"
        )

        # Get unlabeled faces with embeddings from Qdrant
        qdrant = get_face_qdrant_client()

        # Scroll through unlabeled faces and collect embeddings
        face_ids: list[uuid.UUID] = []
        embeddings: list[list[float]] = []
        qdrant_point_ids: list[uuid.UUID] = []

        offset = None
        while len(face_ids) < max_faces:
            # Build filter for unlabeled faces
            must_conditions = [
                IsEmptyCondition(is_empty=PayloadField(key="person_id")),
                FieldCondition(key="quality_score", range=Range(gte=quality_threshold)),
            ]

            # Add time bucket filter if specified
            # TODO: Add taken_at range filter based on time_bucket when needed
            # if time_bucket:
            #     pass

            scroll_filter = Filter(must=must_conditions)  # type: ignore[arg-type]

            # Scroll with filter
            records, next_offset = qdrant.client.scroll(
                collection_name="faces",
                scroll_filter=scroll_filter,
                limit=min(1000, max_faces - len(face_ids)),
                offset=offset,
                with_vectors=True,
                with_payload=True,
            )

            if not records:
                break

            for record in records:
                if record.vector is not None and record.payload is not None:
                    face_instance_id_str = record.payload.get("face_instance_id")
                    if face_instance_id_str:
                        face_ids.append(uuid.UUID(face_instance_id_str))
                        # Handle both dict and list vector formats
                        if isinstance(record.vector, dict):
                            embeddings.append(list(record.vector.values())[0])
                        else:
                            embeddings.append(record.vector)
                        qdrant_point_ids.append(uuid.UUID(str(record.id)))

            offset = next_offset
            if offset is None:
                break

        if len(embeddings) < self.min_cluster_size:
            logger.info(
                f"Not enough faces for clustering: {len(embeddings)} < {self.min_cluster_size}"
            )
            return {
                "total_faces": len(embeddings),
                "clusters_found": 0,
                "noise_count": len(embeddings),
                "cluster_sizes": {},
            }

        logger.info(f"Clustering {len(embeddings)} unlabeled faces")

        # Convert to numpy array
        X = np.array(embeddings)

        # Run HDBSCAN clustering
        cluster_labels = self._run_hdbscan(X)

        # Process cluster assignments
        cluster_assignments: dict[str, list[tuple[uuid.UUID, uuid.UUID]]] = {}
        noise_count = 0

        # Map HDBSCAN labels to cluster_ids
        label_to_cluster_id: dict[int, str] = {}
        for i, label in enumerate(cluster_labels):
            if label == -1:
                noise_count += 1
                continue

            if label not in label_to_cluster_id:
                label_to_cluster_id[label] = f"clu_{uuid.uuid4().hex[:12]}"

            cluster_id = label_to_cluster_id[label]
            if cluster_id not in cluster_assignments:
                cluster_assignments[cluster_id] = []
            cluster_assignments[cluster_id].append((face_ids[i], qdrant_point_ids[i]))

        # Update database and Qdrant
        for cluster_id, face_data in cluster_assignments.items():
            face_instance_ids = [f[0] for f in face_data]
            point_ids = [f[1] for f in face_data]

            # Update FaceInstance.cluster_id in database
            stmt = (
                update(FaceInstance)
                .where(FaceInstance.id.in_(face_instance_ids))
                .values(cluster_id=cluster_id)
            )
            self.db.execute(stmt)

            # Update Qdrant payloads
            qdrant.update_cluster_ids(point_ids, cluster_id)

        self.db.commit()

        cluster_sizes = {cid: len(faces) for cid, faces in cluster_assignments.items()}

        logger.info(
            f"Clustering complete: {len(cluster_assignments)} clusters, "
            f"{sum(cluster_sizes.values())} assigned, {noise_count} noise"
        )

        return {
            "total_faces": len(embeddings),
            "clusters_found": len(cluster_assignments),
            "noise_count": noise_count,
            "cluster_sizes": cluster_sizes,
        }

    def _run_hdbscan(self, X: np.ndarray) -> np.ndarray:
        """Run HDBSCAN clustering on embedding matrix.

        Args:
            X: Numpy array of shape (n_samples, embedding_dim)

        Returns:
            Cluster labels array of shape (n_samples,)
        """
        try:
            import hdbscan
        except ImportError:
            logger.error("hdbscan not installed. Run: pip install hdbscan")
            raise

        # For normalized face embeddings, euclidean distance works well
        # (it's proportional to cosine distance for unit vectors)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            metric=self.metric,
            core_dist_n_jobs=-1,  # Use all CPU cores
        )

        cluster_labels = clusterer.fit_predict(X)

        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        logger.debug(f"HDBSCAN found {n_clusters} clusters")

        return cluster_labels

    def recluster_within_cluster(
        self,
        cluster_id: str,
        min_cluster_size: int = 3,
    ) -> dict:
        """Re-cluster faces within an existing cluster (for splitting).

        Args:
            cluster_id: The cluster to split
            min_cluster_size: Minimum size for sub-clusters

        Returns:
            Summary of sub-clusters created
        """
        from image_search_service.db.models import FaceInstance
        from image_search_service.vector.face_qdrant import get_face_qdrant_client

        qdrant = get_face_qdrant_client()

        # Get all faces in this cluster
        face_ids: list[uuid.UUID] = []
        embeddings: list[list[float]] = []
        qdrant_point_ids: list[uuid.UUID] = []

        offset = None
        while True:
            records, next_offset = qdrant.scroll_faces(
                limit=1000,
                offset=offset,
                filter_cluster_id=cluster_id,
                include_vectors=True,
            )

            if not records:
                break

            for record in records:
                if record.vector is not None and record.payload is not None:
                    face_instance_id_str = record.payload.get("face_instance_id")
                    if face_instance_id_str:
                        face_ids.append(uuid.UUID(face_instance_id_str))
                        # Handle both dict and list vector formats
                        if isinstance(record.vector, dict):
                            embeddings.append(list(record.vector.values())[0])
                        else:
                            embeddings.append(record.vector)
                        qdrant_point_ids.append(uuid.UUID(str(record.id)))

            offset = next_offset
            if offset is None:
                break

        if len(embeddings) < min_cluster_size * 2:
            return {"status": "too_small", "count": len(embeddings)}

        # Run tighter clustering
        original_min = self.min_cluster_size
        self.min_cluster_size = min_cluster_size

        X = np.array(embeddings)
        cluster_labels = self._run_hdbscan(X)

        self.min_cluster_size = original_min

        # Assign new sub-cluster IDs
        label_to_new_cluster: dict[int, str] = {}
        for i, label in enumerate(cluster_labels):
            if label == -1:
                continue

            if label not in label_to_new_cluster:
                label_to_new_cluster[label] = f"{cluster_id}_sub_{uuid.uuid4().hex[:8]}"

            new_cluster_id = label_to_new_cluster[label]

            # Update database
            stmt = (
                update(FaceInstance)
                .where(FaceInstance.id == face_ids[i])
                .values(cluster_id=new_cluster_id)
            )
            self.db.execute(stmt)

            # Update Qdrant
            qdrant.update_cluster_ids([qdrant_point_ids[i]], new_cluster_id)

        self.db.commit()

        return {
            "status": "split",
            "original_size": len(embeddings),
            "sub_clusters": len(label_to_new_cluster),
            "new_cluster_ids": list(label_to_new_cluster.values()),
        }


def get_face_clusterer(
    db_session: SyncSession,
    min_cluster_size: int = 5,
    min_samples: int = 3,
) -> FaceClusterer:
    """Factory function for FaceClusterer.

    Args:
        db_session: Synchronous SQLAlchemy session
        min_cluster_size: Minimum cluster size for HDBSCAN
        min_samples: Minimum samples for core point

    Returns:
        Configured FaceClusterer instance
    """
    return FaceClusterer(
        db_session=db_session,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )
