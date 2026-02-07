"""Dual-mode face clustering: supervised + unsupervised."""

import logging
import uuid
from collections import defaultdict
from typing import Any

import numpy as np
import numpy.typing as npt
from sqlalchemy import select, update
from sqlalchemy.orm import Session as SyncSession

from image_search_service.db.models import FaceInstance

logger = logging.getLogger(__name__)


class DualModeClusterer:
    """Clusters faces in dual mode: supervised (known people) + unsupervised (unknown faces)."""

    def __init__(
        self,
        db_session: SyncSession,
        person_match_threshold: float = 0.7,
        unknown_min_cluster_size: int = 3,
        unknown_method: str = "hdbscan",
        unknown_eps: float = 0.5,
    ):
        """Initialize dual-mode clusterer.

        Args:
            db_session: Synchronous SQLAlchemy session
            person_match_threshold: Minimum similarity for assignment to person (0-1)
            unknown_min_cluster_size: Minimum cluster size for unknown faces
            unknown_method: Clustering method (hdbscan, dbscan, agglomerative)
            unknown_eps: Distance threshold for DBSCAN/Agglomerative
        """
        self.db = db_session
        self.person_match_threshold = person_match_threshold
        self.unknown_min_cluster_size = unknown_min_cluster_size
        self.unknown_method = unknown_method
        self.unknown_eps = unknown_eps

    def cluster_all_faces(self, max_faces: int | None = None) -> dict[str, int]:
        """Run dual-mode clustering on all faces.

        Process:
        1. Get all faces (labeled and unlabeled)
        2. Separate by person_id
        3. Assign unlabeled faces to known people (supervised)
        4. Cluster remaining unknown faces (unsupervised)
        5. Save results

        Args:
            max_faces: Optional limit on number of faces to process

        Returns:
            Summary dict with counts
        """
        logger.info("Starting dual-mode clustering")

        # Get all faces with embeddings
        query = select(FaceInstance).where(FaceInstance.qdrant_point_id.isnot(None))

        if max_faces:
            query = query.limit(max_faces)

        all_faces = self.db.execute(query).scalars().all()

        if not all_faces:
            logger.info("No faces found for clustering")
            return {
                "total_processed": 0,
                "assigned_to_people": 0,
                "unknown_clusters": 0,
                "still_unlabeled": 0,
            }

        logger.info(f"Processing {len(all_faces)} faces")

        # Separate labeled from unlabeled
        labeled_faces = []
        unlabeled_faces = []

        for face in all_faces:
            face_dict = {
                "id": face.id,
                "qdrant_point_id": face.qdrant_point_id,
                "person_id": face.person_id,
                "cluster_id": face.cluster_id,
            }

            if face.person_id:
                labeled_faces.append(face_dict)
            else:
                unlabeled_faces.append(face_dict)

        logger.info(f"Labeled: {len(labeled_faces)}, Unlabeled: {len(unlabeled_faces)}")

        # Phase 1: Assign unlabeled faces to known people (supervised)
        assigned_faces, still_unknown = self.assign_to_known_people(
            unlabeled_faces=unlabeled_faces,
            labeled_faces=labeled_faces,
        )

        # Phase 2: Cluster remaining unknown faces (unsupervised)
        unknown_clusters = self.cluster_unknown_faces(still_unknown)

        # Phase 3: Save results
        self.save_dual_mode_results(
            assigned_faces=assigned_faces,
            unknown_clusters=unknown_clusters,
        )

        logger.info(
            f"Dual-mode clustering complete: "
            f"{len(assigned_faces)} assigned to people, "
            f"{len(unknown_clusters)} in unknown clusters"
        )

        return {
            "total_processed": len(all_faces),
            "assigned_to_people": len(assigned_faces),
            "unknown_clusters": len(set(unknown_clusters.values())),
            "still_unlabeled": len(still_unknown) - len(unknown_clusters),
        }

    def assign_to_known_people(
        self,
        unlabeled_faces: list[dict[str, Any]],
        labeled_faces: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Assign unlabeled faces to known Person entities (supervised).

        Args:
            unlabeled_faces: Faces without person_id
            labeled_faces: Faces with person_id

        Returns:
            (assigned_faces, still_unknown)
        """
        if not labeled_faces:
            logger.info("No labeled faces for supervised assignment")
            return [], unlabeled_faces

        # Build person centroids from labeled faces
        person_embeddings = defaultdict(list)

        for face in labeled_faces:
            embedding = self._get_face_embedding(face["qdrant_point_id"])
            if embedding is not None:
                person_embeddings[face["person_id"]].append(embedding)

        if not person_embeddings:
            logger.warning("Could not retrieve embeddings for labeled faces")
            return [], unlabeled_faces

        # Calculate centroid for each person
        person_centroids = {}
        for person_id, embeddings in person_embeddings.items():
            centroid = np.mean(embeddings, axis=0)
            # Re-normalize
            centroid = centroid / np.linalg.norm(centroid)
            person_centroids[person_id] = centroid

        logger.info(f"Computed centroids for {len(person_centroids)} people")

        assigned = []
        still_unknown = []

        # Match each unlabeled face to nearest person
        for face in unlabeled_faces:
            embedding = self._get_face_embedding(face["qdrant_point_id"])
            if embedding is None:
                still_unknown.append(face)
                continue

            best_person = None
            best_similarity = -1

            for person_id, centroid in person_centroids.items():
                # Cosine similarity
                similarity = np.dot(embedding, centroid) / (
                    np.linalg.norm(embedding) * np.linalg.norm(centroid)
                )

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_person = person_id

            # Assign if above threshold
            if best_similarity >= self.person_match_threshold:
                assigned.append(
                    {
                        "face_id": face["id"],
                        "person_id": best_person,
                        "similarity": best_similarity,
                        "qdrant_point_id": face["qdrant_point_id"],
                    }
                )
            else:
                still_unknown.append(face)

        logger.info(
            f"Supervised assignment: {len(assigned)} assigned, "
            f"{len(still_unknown)} still unknown"
        )

        return assigned, still_unknown

    def cluster_unknown_faces(
        self, unknown_faces: list[dict[str, Any]]
    ) -> dict[uuid.UUID, str]:
        """Cluster unknown faces using unsupervised learning.

        Args:
            unknown_faces: Faces still unassigned after supervised phase

        Returns:
            {face_id: cluster_label}
        """
        if not unknown_faces:
            return {}

        # Get embeddings
        embeddings = []
        face_ids = []

        for face in unknown_faces:
            embedding = self._get_face_embedding(face["qdrant_point_id"])
            if embedding is not None:
                embeddings.append(embedding)
                face_ids.append(face["id"])

        if len(embeddings) < self.unknown_min_cluster_size:
            # Not enough for clustering
            logger.info(
                f"Not enough faces for clustering: {len(embeddings)} < "
                f"{self.unknown_min_cluster_size}"
            )
            return {fid: f"unknown_noise_{fid}" for fid in face_ids}

        embeddings_array = np.array(embeddings)

        # Choose clustering method
        if self.unknown_method == "hdbscan":
            cluster_labels = self._cluster_hdbscan(embeddings_array)
        elif self.unknown_method == "dbscan":
            cluster_labels = self._cluster_dbscan(embeddings_array)
        elif self.unknown_method == "agglomerative":
            cluster_labels = self._cluster_agglomerative(embeddings_array)
        else:
            raise ValueError(f"Unknown clustering method: {self.unknown_method}")

        # Map face_id to cluster_label
        clusters = {}
        for face_id, label in zip(face_ids, cluster_labels):
            if label == -1:  # Noise
                clusters[face_id] = f"unknown_noise_{face_id}"
            else:
                clusters[face_id] = f"unknown_cluster_{label}"

        unique_clusters = len(set(clusters.values()))
        noise_count = sum(1 for v in clusters.values() if "noise" in v)

        logger.info(
            f"Unsupervised clustering: {unique_clusters} clusters, {noise_count} noise"
        )

        return clusters

    def save_dual_mode_results(
        self,
        assigned_faces: list[dict[str, Any]],
        unknown_clusters: dict[uuid.UUID, str],
    ) -> None:
        """Save clustering results to database and Qdrant.

        Args:
            assigned_faces: List of dicts with face_id, person_id, similarity
            unknown_clusters: Dict mapping face_id to cluster_label
        """
        from image_search_service.vector.face_qdrant import get_face_qdrant_client

        qdrant = get_face_qdrant_client()

        # Update assigned faces (person_id)
        if assigned_faces:
            # Group by person_id for batch updates
            person_groups = defaultdict(list)
            for face in assigned_faces:
                person_groups[face["person_id"]].append(face)

            for person_id, faces in person_groups.items():
                face_ids = [f["face_id"] for f in faces]
                qdrant_point_ids = [f["qdrant_point_id"] for f in faces]

                # Update database
                stmt = (
                    update(FaceInstance)
                    .where(FaceInstance.id.in_(face_ids))
                    .values(person_id=person_id)
                )
                self.db.execute(stmt)

                # Update Qdrant
                qdrant.update_person_ids(qdrant_point_ids, person_id)

            logger.info(f"Updated person_id for {len(assigned_faces)} faces")

        # Update unknown clusters (cluster_id)
        if unknown_clusters:
            # Group by cluster_id for batch updates
            cluster_groups = defaultdict(list)
            for face_id, cluster_id in unknown_clusters.items():
                cluster_groups[cluster_id].append(face_id)

            for cluster_id, face_ids in cluster_groups.items():
                # Get qdrant_point_ids from database
                faces_query = select(FaceInstance).where(FaceInstance.id.in_(face_ids))
                faces_result = self.db.execute(faces_query).scalars().all()

                qdrant_point_ids = [
                    f.qdrant_point_id for f in faces_result if f.qdrant_point_id
                ]

                # Update database
                stmt = (
                    update(FaceInstance)
                    .where(FaceInstance.id.in_(face_ids))
                    .values(cluster_id=cluster_id)
                )
                self.db.execute(stmt)

                # Update Qdrant
                qdrant.update_cluster_ids(qdrant_point_ids, cluster_id)

            logger.info(f"Updated cluster_id for {len(unknown_clusters)} faces")

        # Commit transaction
        self.db.commit()
        logger.info("Saved dual-mode clustering results")

    def _get_face_embedding(self, qdrant_point_id: uuid.UUID) -> npt.NDArray[np.float64] | None:
        """Get face embedding from Qdrant by point ID."""
        from image_search_service.vector.face_qdrant import (
            _get_face_collection_name,
            get_face_qdrant_client,
        )

        qdrant = get_face_qdrant_client()

        try:
            points = qdrant.client.retrieve(
                collection_name=_get_face_collection_name(),
                ids=[str(qdrant_point_id)],
                with_vectors=True,
            )

            if points and points[0].vector:
                vector = points[0].vector
                # Handle both dict and list formats
                if isinstance(vector, dict):
                    return np.array(list(vector.values())[0])
                return np.array(vector)
            return None
        except Exception as e:
            logger.error(f"Error retrieving embedding for {qdrant_point_id}: {e}")
            return None

    def _cluster_hdbscan(self, embeddings: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
        """Cluster using HDBSCAN with cosine distance."""
        try:
            import hdbscan
        except ImportError:
            logger.error("hdbscan not installed. Run: pip install hdbscan")
            raise

        from sklearn.metrics.pairwise import cosine_distances

        # Compute pairwise cosine distances (precomputed metric)
        distance_matrix = cosine_distances(embeddings)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.unknown_min_cluster_size,
            metric="precomputed",
            cluster_selection_method="eom",
        )
        return clusterer.fit_predict(distance_matrix)  # type: ignore[no-any-return]

    def _cluster_dbscan(self, embeddings: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
        """Cluster using DBSCAN."""
        from sklearn.cluster import DBSCAN

        clusterer = DBSCAN(
            eps=self.unknown_eps,
            min_samples=self.unknown_min_cluster_size,
            metric="cosine",
        )
        return clusterer.fit_predict(embeddings)  # type: ignore[no-any-return]

    def _cluster_agglomerative(self, embeddings: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
        """Cluster using Agglomerative Clustering."""
        from sklearn.cluster import AgglomerativeClustering

        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.unknown_eps,
            linkage="average",
            metric="cosine",
        )
        return clusterer.fit_predict(embeddings)  # type: ignore[no-any-return]


def get_dual_mode_clusterer(
    db_session: SyncSession,
    person_match_threshold: float = 0.7,
    unknown_min_cluster_size: int = 3,
    unknown_method: str = "hdbscan",
    unknown_eps: float = 0.5,
) -> DualModeClusterer:
    """Factory function for DualModeClusterer."""
    return DualModeClusterer(
        db_session=db_session,
        person_match_threshold=person_match_threshold,
        unknown_min_cluster_size=unknown_min_cluster_size,
        unknown_method=unknown_method,
        unknown_eps=unknown_eps,
    )
