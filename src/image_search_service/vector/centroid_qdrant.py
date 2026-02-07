"""Centroid-specific Qdrant vector database client for person centroids.

This module manages the 'person_centroids' collection in Qdrant optimized for:
- Person centroid embeddings (512-dim ArcFace vectors)
- Versioning and staleness detection
- Multi-centroid support (global + cluster modes)
- Efficient similarity search for face→person matching
"""

import uuid
from typing import Any

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
CENTROID_VECTOR_DIM = 512  # ArcFace embedding dimension


class CentroidQdrantClient:
    """Singleton client for person centroids in Qdrant.

    Manages a separate 'person_centroids' collection with payload schema:
    - person_id: UUID (string format for Qdrant)
    - centroid_id: UUID (matches DB primary key)
    - model_version: str (embedding model identifier)
    - centroid_version: int (algorithm version)
    - centroid_type: str (global, cluster)
    - cluster_label: str (global, k2_0, k2_1, etc.)
    - n_faces: int (number of faces used)
    - created_at: ISO datetime string
    - source_hash: str (hash of source face IDs)
    - build_params: dict (algorithm parameters)
    """

    _instance: "CentroidQdrantClient | None" = None
    _client: QdrantClient | None = None

    def __init__(self) -> None:
        """Initialize client with lazy loading."""
        settings = get_settings()
        self.collection_name = settings.qdrant_centroid_collection

    @classmethod
    def get_instance(cls) -> "CentroidQdrantClient":
        """Get singleton instance of CentroidQdrantClient.

        Returns:
            Singleton CentroidQdrantClient instance
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
            logger.info("Centroid Qdrant client initialized")

        return self._client

    def ensure_collection(self) -> None:
        """Create person_centroids collection if it doesn't exist.

        Sets up:
        - 512-dim vectors with cosine distance
        - Payload indexes for person_id, model_version, centroid_type
        """
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.collection_name not in collection_names:
                # Create collection with cosine distance
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=CENTROID_VECTOR_DIM, distance=Distance.COSINE),
                )
                logger.info(
                    f"Created Qdrant collection '{self.collection_name}' "
                    f"with dim={CENTROID_VECTOR_DIM}"
                )

                # Create payload indexes for efficient filtering
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="person_id",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
                logger.info("Created payload index on 'person_id'")

                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="centroid_id",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
                logger.info("Created payload index on 'centroid_id'")

                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="model_version",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
                logger.info("Created payload index on 'model_version'")

                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="centroid_version",
                    field_schema=PayloadSchemaType.INTEGER,
                )
                logger.info("Created payload index on 'centroid_version'")

                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="centroid_type",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
                logger.info("Created payload index on 'centroid_type'")

            else:
                logger.debug(f"Collection '{self.collection_name}' already exists")

        except Exception as e:
            logger.error(f"Failed to ensure collection '{self.collection_name}': {e}")
            raise

    def upsert_centroid(
        self,
        centroid_id: uuid.UUID,
        vector: list[float],
        payload: dict[str, Any],
    ) -> None:
        """Upsert a centroid embedding with metadata.

        Args:
            centroid_id: UUID of the centroid (used as Qdrant point ID)
            vector: 512-dim centroid embedding
            payload: Metadata dict with keys:
                - person_id: UUID (required)
                - model_version: str (required)
                - centroid_version: int (required)
                - centroid_type: str (required)
                - cluster_label: str (optional)
                - n_faces: int (required)
                - created_at: str ISO datetime (optional)
                - source_hash: str (optional)
                - build_params: dict (optional)
        """
        # Ensure collection exists before upserting (lazy initialization)
        self.ensure_collection()

        # Convert UUIDs to strings in payload
        processed_payload = {}
        for key, value in payload.items():
            if isinstance(value, uuid.UUID):
                processed_payload[key] = str(value)
            else:
                processed_payload[key] = value

        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=str(centroid_id),
                        vector=vector,
                        payload=processed_payload,
                    )
                ],
            )
            logger.debug(f"Upserted centroid {centroid_id} for person {payload.get('person_id')}")

        except Exception as e:
            logger.error(f"Failed to upsert centroid {centroid_id}: {e}")
            raise

    def delete_centroid(self, centroid_id: uuid.UUID) -> None:
        """Delete a centroid by ID.

        Args:
            centroid_id: UUID of the centroid to delete
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=[str(centroid_id)]),
            )
            logger.info(f"Deleted centroid {centroid_id}")

        except Exception as e:
            logger.error(f"Failed to delete centroid {centroid_id}: {e}")
            raise

    def get_centroid_vector(self, centroid_id: uuid.UUID) -> list[float] | None:
        """Retrieve the centroid vector by ID.

        Args:
            centroid_id: UUID of the centroid

        Returns:
            512-dim embedding vector, or None if not found
        """
        try:
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[str(centroid_id)],
                with_payload=False,
                with_vectors=True,
            )

            if not points:
                logger.warning(f"Centroid {centroid_id} not found in Qdrant")
                return None

            # Handle both dict and list vector formats
            vector = points[0].vector
            if vector is None:
                return None

            if isinstance(vector, dict):
                # Named vector case - get the first vector
                first_value = next(iter(vector.values()))
                if isinstance(first_value, list):
                    # Ensure all elements are numeric types
                    result: list[float] = []
                    for x in first_value:
                        if isinstance(x, (float, int)):
                            result.append(float(x))
                        else:
                            return None
                    return result
                return None
            elif isinstance(vector, list):
                # Direct list case - verify it's list[float]
                result = []
                for x in vector:
                    if isinstance(x, (float, int)):
                        result.append(float(x))
                    else:
                        return None
                return result
            else:
                return None

        except Exception as e:
            logger.error(f"Failed to retrieve centroid vector {centroid_id}: {e}")
            return None

    def search_faces_with_centroid(
        self,
        centroid_vector: list[float],
        limit: int = 100,
        score_threshold: float = 0.7,
        exclude_person_id: uuid.UUID | None = None,
    ) -> list[ScoredPoint]:
        """Search faces collection using a centroid vector.

        This method searches the 'faces' collection (not centroids) using a
        centroid as the query vector, to find candidate faces for suggestions.

        Args:
            centroid_vector: 512-dim centroid vector to use as query
            limit: Maximum number of results
            score_threshold: Minimum similarity score (0.0-1.0)
            exclude_person_id: Exclude faces already assigned to this person

        Returns:
            List of ScoredPoint objects from faces collection
        """
        try:
            from image_search_service.vector.face_qdrant import _get_face_collection_name

            # Get face collection name
            face_collection = _get_face_collection_name()

            # Build filter: exclude faces with person_id if specified
            conditions = []
            if exclude_person_id is not None:
                # Use must_not to exclude this person's faces
                conditions = [
                    FieldCondition(key="person_id", match=MatchValue(value=str(exclude_person_id)))
                ]

            query_filter = (
                Filter(must_not=conditions) if conditions else None  # type: ignore[arg-type]
            )

            # Search faces collection using centroid vector
            results = self.client.query_points(
                collection_name=face_collection,
                query=centroid_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter,
                with_payload=True,
            )

            logger.debug(
                f"Found {len(results.points)} candidate faces using centroid "
                f"(threshold={score_threshold})"
            )
            return results.points

        except Exception as e:
            logger.error(f"Failed to search faces with centroid: {e}")
            raise

    def scroll_centroids(
        self,
        limit: int = 100,
        offset: str | None = None,
        filter_person_id: uuid.UUID | None = None,
        filter_model_version: str | None = None,
        filter_centroid_type: str | None = None,
        include_vectors: bool = False,
    ) -> tuple[list[Record], str | None]:
        """Scroll through centroids with optional filters.

        Args:
            limit: Maximum number of records per page
            offset: Scroll offset from previous call
            filter_person_id: Filter by person_id
            filter_model_version: Filter by model_version
            filter_centroid_type: Filter by centroid_type
            include_vectors: Whether to include embeddings in response

        Returns:
            Tuple of (records, next_offset)
        """
        try:
            # Build filter conditions
            conditions = []

            if filter_person_id is not None:
                conditions.append(
                    FieldCondition(key="person_id", match=MatchValue(value=str(filter_person_id)))
                )

            if filter_model_version is not None:
                conditions.append(
                    FieldCondition(
                        key="model_version", match=MatchValue(value=filter_model_version)
                    )
                )

            if filter_centroid_type is not None:
                conditions.append(
                    FieldCondition(
                        key="centroid_type", match=MatchValue(value=filter_centroid_type)
                    )
                )

            scroll_filter = Filter(must=conditions) if conditions else None  # type: ignore[arg-type]

            records, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=include_vectors,
            )

            logger.debug(f"Scrolled {len(records)} centroids, next_offset={next_offset}")
            # Convert next_offset to string or None for type safety
            next_offset_str = str(next_offset) if next_offset is not None else None
            return records, next_offset_str

        except Exception as e:
            logger.error(f"Failed to scroll centroids: {e}")
            raise

    def delete_centroids_by_person(self, person_id: uuid.UUID) -> int:
        """Delete all centroids for a person.

        Args:
            person_id: Person UUID

        Returns:
            Number of centroids deleted
        """
        try:
            deleted_count = 0
            offset = None

            # Scroll with person_id filter
            person_filter = Filter(
                must=[FieldCondition(key="person_id", match=MatchValue(value=str(person_id)))]
            )

            while True:
                records, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=person_filter,
                    limit=100,
                    offset=offset,
                    with_payload=False,
                )

                if not records:
                    break

                # Extract point IDs
                point_ids = [record.id for record in records]

                # Delete batch
                if point_ids:
                    self.client.delete(
                        collection_name=self.collection_name,
                        points_selector=PointIdsList(points=point_ids),
                    )
                    deleted_count += len(point_ids)

                # Check if we've scrolled through all matching points
                if next_offset is None:
                    break

                offset = next_offset

            if deleted_count > 0:
                logger.info(f"Deleted {deleted_count} centroids for person {person_id}")

            return deleted_count

        except Exception as e:
            logger.error(f"Failed to delete centroids for person {person_id}: {e}")
            raise

    def get_collection_info(self) -> dict[str, Any] | None:
        """Get collection statistics and configuration.

        Returns:
            Dict with collection info or None if collection doesn't exist
        """
        try:
            collection_info = self.client.get_collection(collection_name=self.collection_name)

            return {
                "name": self.collection_name,
                "points_count": collection_info.points_count,
                "vectors_count": collection_info.points_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "vector_dim": CENTROID_VECTOR_DIM,
                "distance": "cosine",
            }

        except Exception as e:
            logger.warning(f"Failed to get collection info for '{self.collection_name}': {e}")
            return None

    def reset_collection(self) -> int:
        """Delete all centroids and recreate the collection.

        WARNING: This is destructive and cannot be undone.

        Returns:
            Total number of centroids deleted
        """
        try:
            # Get count before deletion
            centroid_count = 0
            try:
                collection_info = self.client.get_collection(
                    collection_name=self.collection_name
                )
                centroid_count = collection_info.points_count or 0
            except Exception:
                logger.info(
                    f"Collection '{self.collection_name}' does not exist, nothing to reset"
                )
                return 0

            # Delete the entire collection
            self.client.delete_collection(collection_name=self.collection_name)
            logger.warning(f"Deleted collection '{self.collection_name}'")

            # Recreate empty collection with same configuration
            self.ensure_collection()
            logger.info(
                f"Recreated collection '{self.collection_name}' with dim={CENTROID_VECTOR_DIM}"
            )

            logger.warning(f"Reset centroid collection: deleted {centroid_count} centroids")
            return centroid_count

        except Exception as e:
            logger.error(f"Failed to reset centroid collection: {e}")
            raise

    def close(self) -> None:
        """Close Qdrant client and cleanup resources."""
        if self._client is not None:
            self._client.close()
            self._client = None
            logger.info("Centroid Qdrant client closed")


def get_centroid_qdrant_client() -> CentroidQdrantClient:
    """Get singleton instance of CentroidQdrantClient.

    Returns:
        Singleton CentroidQdrantClient instance
    """
    return CentroidQdrantClient.get_instance()
