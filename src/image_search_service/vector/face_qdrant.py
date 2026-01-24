"""Face-specific Qdrant vector database client for face embeddings and person clustering.

This module manages a separate 'faces' collection in Qdrant optimized for:
- Face embeddings (512-dim ArcFace/InsightFace vectors)
- Person clustering and incremental assignment
- Face instance tracking and metadata
- Efficient similarity search with payload filtering
"""

import uuid
from datetime import datetime
from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointIdsList,
    PointStruct,
    Record,
    ScoredPoint,
    VectorParams,
)

from image_search_service.core.config import get_settings
from image_search_service.core.logging import get_logger

logger = get_logger(__name__)

# Collection configuration
FACE_VECTOR_DIM = 512  # ArcFace/InsightFace embedding dimension


def _get_face_collection_name() -> str:
    """Get face collection name from settings (allows test override).

    This function reads from environment-configurable settings instead of
    using a hardcoded constant, preventing tests from accidentally deleting
    production data.

    Returns:
        Face collection name from settings (e.g., "faces" or "test_faces")
    """
    return get_settings().qdrant_face_collection


class FaceQdrantClient:
    """Singleton client for face embeddings in Qdrant.

    Manages a separate 'faces' collection with payload schema:
    - asset_id: UUID (string format for Qdrant)
    - face_instance_id: UUID
    - person_id: Optional UUID
    - cluster_id: Optional string
    - detection_confidence: float (0.0-1.0)
    - quality_score: Optional float (0.0-1.0)
    - taken_at: Optional ISO datetime string
    - bbox: dict with x, y, w, h (pixel coordinates)
    - is_prototype: bool (for incremental assignment queries)
    """

    _instance: Optional["FaceQdrantClient"] = None
    _client: QdrantClient | None = None

    def __init__(self) -> None:
        """Initialize client with lazy loading."""
        pass

    @classmethod
    def get_instance(cls) -> "FaceQdrantClient":
        """Get singleton instance of FaceQdrantClient.

        Returns:
            Singleton FaceQdrantClient instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def client(self) -> QdrantClient:
        """Get or create Qdrant client (lazy initialization).

        Returns:
            QdrantClient instance
        """
        if self._client is None:
            settings = get_settings()
            self._client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key if settings.qdrant_api_key else None,
            )
            logger.info("Face Qdrant client initialized")

        return self._client

    def ensure_collection(self) -> None:
        """Create faces collection if it doesn't exist.

        Sets up:
        - 512-dim vectors with cosine distance
        - Payload indexes for person_id, cluster_id, is_prototype
        """
        collection_name = _get_face_collection_name()
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]

            if collection_name not in collection_names:
                # Create collection with cosine distance
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=FACE_VECTOR_DIM, distance=Distance.COSINE),
                )
                logger.info(
                    f"Created Qdrant collection '{collection_name}' with dim={FACE_VECTOR_DIM}"
                )

                # Create payload indexes for efficient filtering
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name="person_id",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
                logger.info("Created payload index on 'person_id'")

                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name="cluster_id",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
                logger.info("Created payload index on 'cluster_id'")

                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name="is_prototype",
                    field_schema=PayloadSchemaType.BOOL,
                )
                logger.info("Created payload index on 'is_prototype'")

                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name="asset_id",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
                logger.info("Created payload index on 'asset_id'")

                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name="face_instance_id",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
                logger.info("Created payload index on 'face_instance_id'")

            else:
                logger.debug(f"Collection '{collection_name}' already exists")

        except Exception as e:
            logger.error(f"Failed to ensure collection '{collection_name}': {e}")
            raise

    def upsert_face(
        self,
        point_id: uuid.UUID,
        embedding: list[float],
        asset_id: uuid.UUID,
        face_instance_id: uuid.UUID,
        detection_confidence: float,
        quality_score: float | None = None,
        bbox: dict[str, float] | None = None,
        person_id: uuid.UUID | None = None,
        cluster_id: str | None = None,
        taken_at: datetime | None = None,
        is_prototype: bool = False,
    ) -> None:
        """Upsert a single face embedding with metadata payload.

        Args:
            point_id: Unique ID for this face point (typically face_instance_id)
            embedding: 512-dim face embedding vector
            asset_id: UUID of the image asset
            face_instance_id: UUID of the face instance record
            detection_confidence: Face detection confidence (0.0-1.0)
            quality_score: Face quality score (0.0-1.0)
            bbox: Bounding box dict with keys: x, y, w, h
            person_id: UUID of associated person (if labeled)
            cluster_id: Cluster ID from clustering algorithm
            taken_at: Image taken timestamp
            is_prototype: Whether this face is a cluster prototype
        """
        payload: dict[str, Any] = {
            "asset_id": str(asset_id),
            "face_instance_id": str(face_instance_id),
            "detection_confidence": detection_confidence,
            "is_prototype": is_prototype,
        }

        # Add optional fields
        if quality_score is not None:
            payload["quality_score"] = quality_score
        if bbox is not None:
            payload["bbox"] = bbox
        if person_id is not None:
            payload["person_id"] = str(person_id)
        if cluster_id is not None:
            payload["cluster_id"] = cluster_id
        if taken_at is not None:
            payload["taken_at"] = taken_at.isoformat()

        try:
            self.client.upsert(
                collection_name=_get_face_collection_name(),
                points=[
                    PointStruct(
                        id=str(point_id),
                        vector=embedding,
                        payload=payload,
                    )
                ],
            )
            logger.debug(f"Upserted face {point_id} for asset {asset_id}")

        except Exception as e:
            logger.error(f"Failed to upsert face {point_id}: {e}")
            raise

    def upsert_faces_batch(
        self,
        faces: list[dict[str, Any]],
    ) -> None:
        """Batch upsert multiple face embeddings.

        Args:
            faces: List of dicts, each containing:
                - point_id: uuid.UUID
                - embedding: list[float]
                - asset_id: uuid.UUID
                - face_instance_id: uuid.UUID
                - detection_confidence: float
                - quality_score: Optional[float]
                - bbox: Optional[dict]
                - person_id: Optional[uuid.UUID]
                - cluster_id: Optional[str]
                - taken_at: Optional[datetime]
                - is_prototype: bool
        """
        if not faces:
            logger.debug("No faces to upsert")
            return

        try:
            points: list[PointStruct] = []

            for face in faces:
                payload: dict[str, Any] = {
                    "asset_id": str(face["asset_id"]),
                    "face_instance_id": str(face["face_instance_id"]),
                    "detection_confidence": face["detection_confidence"],
                    "is_prototype": face.get("is_prototype", False),
                }

                # Add optional fields
                if face.get("quality_score") is not None:
                    payload["quality_score"] = face["quality_score"]
                if face.get("bbox") is not None:
                    payload["bbox"] = face["bbox"]
                if face.get("person_id") is not None:
                    payload["person_id"] = str(face["person_id"])
                if face.get("cluster_id") is not None:
                    payload["cluster_id"] = face["cluster_id"]
                if face.get("taken_at") is not None:
                    taken_at = face["taken_at"]
                    payload["taken_at"] = (
                        taken_at.isoformat() if isinstance(taken_at, datetime) else taken_at
                    )

                points.append(
                    PointStruct(
                        id=str(face["point_id"]),
                        vector=face["embedding"],
                        payload=payload,
                    )
                )

            self.client.upsert(
                collection_name=_get_face_collection_name(),
                points=points,
            )
            logger.info(f"Batch upserted {len(points)} faces")

        except Exception as e:
            logger.error(f"Failed to batch upsert {len(faces)} faces: {e}")
            raise

    def update_payload(
        self,
        point_id: uuid.UUID,
        payload_updates: dict[str, Any],
    ) -> None:
        """Update payload fields without re-uploading vector.

        Args:
            point_id: Face point ID to update
            payload_updates: Dict of payload fields to update
        """
        try:
            # Convert UUIDs to strings in payload
            processed_payload = {}
            for key, value in payload_updates.items():
                if isinstance(value, uuid.UUID):
                    processed_payload[key] = str(value)
                elif isinstance(value, datetime):
                    processed_payload[key] = value.isoformat()
                else:
                    processed_payload[key] = value

            self.client.set_payload(
                collection_name=_get_face_collection_name(),
                payload=processed_payload,
                points=[str(point_id)],
            )
            logger.debug(f"Updated payload for face {point_id}: {list(payload_updates.keys())}")

        except Exception as e:
            logger.error(f"Failed to update payload for face {point_id}: {e}")
            raise

    def update_cluster_ids(
        self,
        point_ids: list[uuid.UUID],
        cluster_id: str,
    ) -> None:
        """Bulk update cluster_id for multiple faces.

        Args:
            point_ids: List of face point IDs
            cluster_id: Cluster ID to assign
        """
        if not point_ids:
            return

        try:
            self.client.set_payload(
                collection_name=_get_face_collection_name(),
                payload={"cluster_id": cluster_id},
                points=[str(point_id) for point_id in point_ids],
            )
            logger.info(f"Updated cluster_id to '{cluster_id}' for {len(point_ids)} faces")

        except Exception as e:
            logger.error(f"Failed to update cluster_ids: {e}")
            raise

    def update_person_ids(
        self,
        point_ids: list[uuid.UUID],
        person_id: uuid.UUID | None,
    ) -> None:
        """Bulk update person_id for multiple faces.

        Args:
            point_ids: List of face point IDs
            person_id: Person UUID to assign, or None to remove assignment
        """
        if not point_ids:
            return

        try:
            # Use delete_payload to remove person_id if None
            if person_id is None:
                self.client.delete_payload(
                    collection_name=_get_face_collection_name(),
                    keys=["person_id"],
                    points=[str(point_id) for point_id in point_ids],
                )
                logger.info(f"Removed person_id from {len(point_ids)} faces")
            else:
                self.client.set_payload(
                    collection_name=_get_face_collection_name(),
                    payload={"person_id": str(person_id)},
                    points=[str(point_id) for point_id in point_ids],
                )
                logger.info(f"Updated person_id to {person_id} for {len(point_ids)} faces")

        except Exception as e:
            logger.error(f"Failed to update person_ids: {e}")
            raise

    def search_similar_faces(
        self,
        query_embedding: list[float],
        limit: int = 10,
        score_threshold: float = 0.5,
        filter_person_id: uuid.UUID | None = None,
        filter_cluster_id: str | None = None,
        filter_is_prototype: bool | None = None,
    ) -> list[ScoredPoint]:
        """Search for similar faces with optional filters.

        Args:
            query_embedding: 512-dim query vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score (0.0-1.0)
            filter_person_id: Filter by person_id
            filter_cluster_id: Filter by cluster_id
            filter_is_prototype: Filter by is_prototype flag

        Returns:
            List of ScoredPoint objects with face metadata
        """
        try:
            # Build filter conditions
            conditions: list[FieldCondition] = []

            if filter_person_id is not None:
                conditions.append(
                    FieldCondition(key="person_id", match=MatchValue(value=str(filter_person_id)))
                )

            if filter_cluster_id is not None:
                conditions.append(
                    FieldCondition(key="cluster_id", match=MatchValue(value=filter_cluster_id))
                )

            if filter_is_prototype is not None:
                conditions.append(
                    FieldCondition(key="is_prototype", match=MatchValue(value=filter_is_prototype))
                )

            query_filter = Filter(must=conditions) if conditions else None  # type: ignore[arg-type]

            # Execute search using query_points
            results = self.client.query_points(
                collection_name=_get_face_collection_name(),
                query=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter,
                with_payload=True,
            )

            logger.debug(f"Found {len(results.points)} similar faces")
            return results.points

        except Exception as e:
            logger.error(f"Failed to search similar faces: {e}")
            raise

    def search_against_prototypes(
        self,
        query_embedding: list[float],
        limit: int = 5,
        score_threshold: float = 0.6,
    ) -> list[ScoredPoint]:
        """Search against only prototype faces for incremental assignment.

        Args:
            query_embedding: 512-dim query vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score (0.0-1.0)

        Returns:
            List of ScoredPoint objects for prototype faces
        """
        return self.search_similar_faces(
            query_embedding=query_embedding,
            limit=limit,
            score_threshold=score_threshold,
            filter_is_prototype=True,
        )

    def get_embedding_by_point_id(self, point_id: uuid.UUID) -> list[float] | None:
        """Retrieve the embedding vector for a specific face point.

        Args:
            point_id: Face point ID (qdrant_point_id from FaceInstance)

        Returns:
            512-dim embedding vector, or None if not found
        """
        try:
            points = self.client.retrieve(
                collection_name=_get_face_collection_name(),
                ids=[str(point_id)],
                with_payload=False,
                with_vectors=True,
            )

            if not points:
                logger.warning(f"Face point {point_id} not found in Qdrant")
                return None

            # Handle both dict and list vector formats
            vector = points[0].vector
            if vector is None:
                return None

            if isinstance(vector, dict):
                # Named vector case - get the first vector
                first_value = next(iter(vector.values()))
                # Ensure it's a list[float], not a SparseVector or nested list
                if isinstance(first_value, list) and all(isinstance(x, float) for x in first_value):
                    return first_value
                return None
            elif isinstance(vector, list):
                # Direct list case - verify it's list[float]
                if all(isinstance(x, float) for x in vector):
                    return vector
                return None
            else:
                # Unsupported vector type
                return None

        except Exception as e:
            logger.error(f"Failed to retrieve embedding for point {point_id}: {e}")
            return None

    def scroll_faces(
        self,
        limit: int = 100,
        offset: str | None = None,
        filter_person_id: uuid.UUID | None = None,
        filter_cluster_id: str | None = None,
        include_vectors: bool = False,
    ) -> tuple[list[Record], str | None]:
        """Scroll through faces with optional filters.

        Args:
            limit: Maximum number of records per page
            offset: Scroll offset from previous call
            filter_person_id: Filter by person_id
            filter_cluster_id: Filter by cluster_id
            include_vectors: Whether to include embeddings in response

        Returns:
            Tuple of (records, next_offset)
        """
        try:
            # Build filter conditions
            conditions: list[FieldCondition] = []

            if filter_person_id is not None:
                conditions.append(
                    FieldCondition(key="person_id", match=MatchValue(value=str(filter_person_id)))
                )

            if filter_cluster_id is not None:
                conditions.append(
                    FieldCondition(key="cluster_id", match=MatchValue(value=filter_cluster_id))
                )

            scroll_filter = Filter(must=conditions) if conditions else None  # type: ignore[arg-type]

            records, next_offset = self.client.scroll(
                collection_name=_get_face_collection_name(),
                scroll_filter=scroll_filter,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=include_vectors,
            )

            logger.debug(f"Scrolled {len(records)} faces, next_offset={next_offset}")
            return records, next_offset

        except Exception as e:
            logger.error(f"Failed to scroll faces: {e}")
            raise

    def get_unlabeled_faces_with_embeddings(
        self,
        quality_threshold: float = 0.5,
        limit: int = 10000,
    ) -> list[tuple[uuid.UUID, list[float]]]:
        """Get embeddings for faces without person_id for clustering.

        Args:
            quality_threshold: Minimum quality_score (0.0-1.0)
            limit: Maximum number of faces to retrieve

        Returns:
            List of (face_instance_id, embedding) tuples
        """
        try:
            # Scroll with filter: person_id is None, quality >= threshold
            # Note: Qdrant doesn't have a native "is null" filter, so we scroll
            # all records and filter in Python
            face_embeddings: list[tuple[uuid.UUID, list[float]]] = []
            offset = None

            while len(face_embeddings) < limit:
                records, next_offset = self.client.scroll(
                    collection_name=_get_face_collection_name(),
                    limit=min(100, limit - len(face_embeddings)),
                    offset=offset,
                    with_vectors=True,
                    with_payload=True,
                )

                if not records:
                    break

                # Filter faces without person_id and above quality threshold
                for record in records:
                    if record.payload is None or record.vector is None:
                        continue

                    # Check if person_id is missing (unlabeled)
                    if "person_id" in record.payload:
                        continue

                    # Check quality threshold
                    quality = record.payload.get("quality_score")
                    if quality is not None and quality < quality_threshold:
                        continue

                    # Extract face_instance_id and embedding
                    face_instance_id_str = record.payload.get("face_instance_id")
                    if face_instance_id_str:
                        face_instance_id = uuid.UUID(face_instance_id_str)
                        # Handle both dict and list vector formats
                        if isinstance(record.vector, dict):
                            embedding = list(record.vector.values())[0]  # Named vector case
                        else:
                            embedding = record.vector

                        face_embeddings.append((face_instance_id, embedding))

                # Check if we've scrolled through all points
                if next_offset is None:
                    break

                offset = next_offset

            logger.info(
                f"Retrieved {len(face_embeddings)} unlabeled faces "
                f"with quality >= {quality_threshold}"
            )
            return face_embeddings[:limit]

        except Exception as e:
            logger.error(f"Failed to get unlabeled faces with embeddings: {e}")
            raise

    def delete_by_asset(self, asset_id: uuid.UUID) -> int:
        """Delete all faces for an asset.

        Args:
            asset_id: Asset UUID

        Returns:
            Number of faces deleted
        """
        try:
            deleted_count = 0
            offset = None

            # Scroll with asset_id filter
            asset_filter = Filter(
                must=[FieldCondition(key="asset_id", match=MatchValue(value=str(asset_id)))]
            )

            while True:
                records, next_offset = self.client.scroll(
                    collection_name=_get_face_collection_name(),
                    scroll_filter=asset_filter,
                    limit=100,
                    offset=offset,
                    with_payload=False,  # We only need IDs
                )

                if not records:
                    break

                # Extract point IDs
                point_ids = [record.id for record in records]

                # Delete batch
                if point_ids:
                    self.client.delete(
                        collection_name=_get_face_collection_name(),
                        points_selector=PointIdsList(points=point_ids),
                    )
                    deleted_count += len(point_ids)

                # Check if we've scrolled through all matching points
                if next_offset is None:
                    break

                offset = next_offset

            if deleted_count > 0:
                logger.info(f"Deleted {deleted_count} faces for asset {asset_id}")

            return deleted_count

        except Exception as e:
            logger.error(f"Failed to delete faces for asset {asset_id}: {e}")
            raise

    def delete_by_face_instance(self, face_instance_id: uuid.UUID) -> int:
        """Delete a specific face by face_instance_id.

        Args:
            face_instance_id: Face instance UUID

        Returns:
            Number of faces deleted (0 or 1)
        """
        try:
            deleted_count = 0
            offset = None

            # Scroll with face_instance_id filter
            face_filter = Filter(
                must=[
                    FieldCondition(
                        key="face_instance_id", match=MatchValue(value=str(face_instance_id))
                    )
                ]
            )

            records, _ = self.client.scroll(
                collection_name=_get_face_collection_name(),
                scroll_filter=face_filter,
                limit=10,  # Should only be 1
                offset=offset,
                with_payload=False,
            )

            # Extract point IDs
            point_ids = [record.id for record in records]

            # Delete
            if point_ids:
                self.client.delete(
                    collection_name=_get_face_collection_name(),
                    points_selector=PointIdsList(points=point_ids),
                )
                deleted_count = len(point_ids)
                logger.info(f"Deleted face instance {face_instance_id}")

            return deleted_count

        except Exception as e:
            logger.error(f"Failed to delete face instance {face_instance_id}: {e}")
            raise

    def point_exists(self, point_id: uuid.UUID) -> bool:
        """Check if a point exists in the collection.

        Args:
            point_id: Face point ID to check

        Returns:
            True if point exists, False otherwise
        """
        try:
            # Retrieve the specific point by ID
            points = self.client.retrieve(
                collection_name=_get_face_collection_name(),
                ids=[str(point_id)],
                with_payload=False,
                with_vectors=False,
            )
            return len(points) > 0

        except Exception as e:
            logger.error(f"Failed to check if point {point_id} exists: {e}")
            return False

    def get_collection_info(self) -> dict[str, Any] | None:
        """Get collection statistics and configuration.

        Returns:
            Dict with collection info or None if collection doesn't exist
        """
        try:
            collection_info = self.client.get_collection(collection_name=_get_face_collection_name())  # noqa: E501

            return {
                "name": _get_face_collection_name(),
                "points_count": collection_info.points_count,
                # Use points_count (vectors_count removed in qdrant-client 1.12+)
                "vectors_count": collection_info.points_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "vector_dim": FACE_VECTOR_DIM,
                "distance": "cosine",
            }

        except Exception as e:
            logger.warning(f"Failed to get collection info for '{_get_face_collection_name()}': {e}")  # noqa: E501
            return None

    def reset_collection(self) -> int:
        """Delete all vectors in the faces collection and recreate it.

        WARNING: This is destructive and cannot be undone.

        Returns:
            Total number of vectors deleted

        Raises:
            RuntimeError: If attempting to delete production collection during tests
        """
        import os

        collection_name = _get_face_collection_name()

        # CRITICAL SAFETY GUARD: Prevent deletion of production collections during tests
        if os.getenv("PYTEST_CURRENT_TEST"):
            if collection_name == "faces":
                raise RuntimeError(
                    "SAFETY GUARD: Refusing to delete production 'faces' collection during tests. "
                    "Ensure QDRANT_FACE_COLLECTION is set to 'test_faces' in test fixtures."
                )

        # Additional safety guard for non-test destructive operations
        if collection_name == "faces" and not os.getenv("ALLOW_PRODUCTION_RESET"):
            logger.warning(
                "Resetting production 'faces' collection - set ALLOW_PRODUCTION_RESET=1 to suppress"
            )

        try:
            # Get count before deletion
            vector_count = 0
            try:
                collection_info = self.client.get_collection(collection_name=collection_name)
                vector_count = collection_info.points_count or 0
            except Exception:
                # Collection might not exist yet
                logger.info(f"Collection '{collection_name}' does not exist, nothing to reset")
                return 0

            # Delete the entire collection
            self.client.delete_collection(collection_name=collection_name)
            logger.warning(f"Deleted collection '{collection_name}'")

            # Recreate empty collection with same configuration
            self.ensure_collection()
            logger.info(f"Recreated collection '{collection_name}' with dim={FACE_VECTOR_DIM}")

            logger.warning(f"Reset face collection: deleted {vector_count} vectors")
            return vector_count

        except Exception as e:
            logger.error(f"Failed to reset face collection: {e}")
            raise

    def close(self) -> None:
        """Close Qdrant client and cleanup resources."""
        if self._client is not None:
            self._client.close()
            self._client = None
            logger.info("Face Qdrant client closed")


def get_face_qdrant_client() -> FaceQdrantClient:
    """Get singleton instance of FaceQdrantClient.

    Returns:
        Singleton FaceQdrantClient instance
    """
    return FaceQdrantClient.get_instance()
