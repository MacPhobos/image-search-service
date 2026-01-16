"""Unit tests for CentroidQdrantClient with mocked Qdrant.

These tests verify centroid vector operations without requiring a running Qdrant instance.
All Qdrant client operations are mocked to ensure fast, isolated unit tests.
"""

import uuid
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from qdrant_client.models import CollectionInfo

from image_search_service.vector.centroid_qdrant import (
    CENTROID_COLLECTION_NAME,
    CENTROID_VECTOR_DIM,
    CentroidQdrantClient,
)


class TestCentroidQdrantClient:
    """Tests for CentroidQdrantClient with mocked Qdrant."""

    @pytest.fixture
    def mock_qdrant_client(self) -> MagicMock:
        """Create a mock Qdrant client instance."""
        mock_client = MagicMock()

        # Mock get_collections to return empty list by default
        mock_collections_response = MagicMock()
        mock_collections_response.collections = []
        mock_client.get_collections.return_value = mock_collections_response

        return mock_client

    @pytest.fixture
    def centroid_qdrant(self, mock_qdrant_client: MagicMock) -> CentroidQdrantClient:
        """Create CentroidQdrantClient with mocked Qdrant client.

        This fixture injects the mock client directly into the singleton instance.
        """
        # Clear singleton instance to ensure clean state
        CentroidQdrantClient._instance = None
        CentroidQdrantClient._client = None

        # Create new instance and inject mock client
        client = CentroidQdrantClient.get_instance()
        client._client = mock_qdrant_client

        return client

    def test_ensure_collection_creates_when_not_exists(
        self, centroid_qdrant: CentroidQdrantClient, mock_qdrant_client: MagicMock
    ) -> None:
        """Test that ensure_collection creates collection when it doesn't exist."""
        # Mock: collection doesn't exist
        mock_collections_response = MagicMock()
        mock_collections_response.collections = []
        mock_qdrant_client.get_collections.return_value = mock_collections_response

        centroid_qdrant.ensure_collection()

        # Verify create_collection was called with correct parameters
        mock_qdrant_client.create_collection.assert_called_once()
        call_kwargs = mock_qdrant_client.create_collection.call_args.kwargs
        assert call_kwargs["collection_name"] == CENTROID_COLLECTION_NAME
        assert call_kwargs["vectors_config"].size == CENTROID_VECTOR_DIM

        # Verify payload indexes were created
        assert mock_qdrant_client.create_payload_index.call_count >= 3
        # Check for person_id, centroid_id, model_version indexes
        index_calls = mock_qdrant_client.create_payload_index.call_args_list
        indexed_fields = {call_args.kwargs["field_name"] for call_args in index_calls}
        assert "person_id" in indexed_fields
        assert "centroid_id" in indexed_fields
        assert "model_version" in indexed_fields

    def test_ensure_collection_skips_when_exists(
        self, centroid_qdrant: CentroidQdrantClient, mock_qdrant_client: MagicMock
    ) -> None:
        """Test that ensure_collection skips creation when collection exists."""
        # Mock: collection already exists
        mock_collection = MagicMock()
        mock_collection.name = CENTROID_COLLECTION_NAME
        mock_collections_response = MagicMock()
        mock_collections_response.collections = [mock_collection]
        mock_qdrant_client.get_collections.return_value = mock_collections_response

        centroid_qdrant.ensure_collection()

        # Verify create_collection was NOT called
        mock_qdrant_client.create_collection.assert_not_called()
        mock_qdrant_client.create_payload_index.assert_not_called()

    def test_upsert_centroid_creates_collection_first(
        self, centroid_qdrant: CentroidQdrantClient, mock_qdrant_client: MagicMock
    ) -> None:
        """Test that upsert_centroid creates collection if it doesn't exist (lazy init)."""
        # Mock: collection doesn't exist initially
        mock_collections_response = MagicMock()
        mock_collections_response.collections = []
        mock_qdrant_client.get_collections.return_value = mock_collections_response

        # Perform upsert
        centroid_id = uuid.uuid4()
        vector = np.random.rand(512).astype(np.float32).tolist()
        payload = {
            "person_id": str(uuid.uuid4()),
            "centroid_type": "global",
            "model_version": "buffalo_l",
            "centroid_version": 1,
            "n_faces": 10,
        }

        centroid_qdrant.upsert_centroid(centroid_id, vector, payload)

        # Verify collection was created before upsert
        mock_qdrant_client.create_collection.assert_called_once()
        mock_qdrant_client.upsert.assert_called_once()

    def test_upsert_centroid_success(
        self, centroid_qdrant: CentroidQdrantClient, mock_qdrant_client: MagicMock
    ) -> None:
        """Test upserting a centroid."""
        # Mock: collection exists
        mock_collection = MagicMock()
        mock_collection.name = CENTROID_COLLECTION_NAME
        mock_collections_response = MagicMock()
        mock_collections_response.collections = [mock_collection]
        mock_qdrant_client.get_collections.return_value = mock_collections_response

        centroid_id = uuid.uuid4()
        person_id = uuid.uuid4()
        vector = np.random.rand(512).astype(np.float32).tolist()
        payload: dict[str, Any] = {
            "person_id": person_id,
            "centroid_type": "global",
            "model_version": "buffalo_l",
            "centroid_version": 1,
            "n_faces": 10,
        }

        centroid_qdrant.upsert_centroid(centroid_id, vector, payload)

        # Verify upsert was called with correct parameters
        mock_qdrant_client.upsert.assert_called_once()
        upsert_call = mock_qdrant_client.upsert.call_args

        # Check collection name
        assert upsert_call.kwargs["collection_name"] == CENTROID_COLLECTION_NAME

        # Check point structure
        points = upsert_call.kwargs["points"]
        assert len(points) == 1
        point = points[0]
        assert point.id == str(centroid_id)
        assert point.vector == vector

        # Check payload (UUIDs should be converted to strings)
        assert point.payload["person_id"] == str(person_id)
        assert point.payload["centroid_type"] == "global"

    def test_upsert_centroid_converts_uuid_in_payload(
        self, centroid_qdrant: CentroidQdrantClient, mock_qdrant_client: MagicMock
    ) -> None:
        """Test that UUIDs in payload are converted to strings."""
        # Mock: collection exists
        mock_collection = MagicMock()
        mock_collection.name = CENTROID_COLLECTION_NAME
        mock_collections_response = MagicMock()
        mock_collections_response.collections = [mock_collection]
        mock_qdrant_client.get_collections.return_value = mock_collections_response

        centroid_id = uuid.uuid4()
        person_id = uuid.uuid4()
        vector = [0.1] * 512
        payload: dict[str, Any] = {
            "person_id": person_id,  # UUID object, not string
            "centroid_type": "global",
            "model_version": "buffalo_l",
            "centroid_version": 1,
            "n_faces": 10,
        }

        centroid_qdrant.upsert_centroid(centroid_id, vector, payload)

        # Verify UUID was converted to string in upsert call
        upsert_call = mock_qdrant_client.upsert.call_args
        point = upsert_call.kwargs["points"][0]
        assert isinstance(point.payload["person_id"], str)
        assert point.payload["person_id"] == str(person_id)

    def test_get_centroid_vector_success(
        self, centroid_qdrant: CentroidQdrantClient, mock_qdrant_client: MagicMock
    ) -> None:
        """Test retrieving a centroid vector."""
        centroid_id = uuid.uuid4()
        expected_vector = [0.1] * 512

        # Mock the retrieve response
        mock_point = MagicMock()
        mock_point.vector = expected_vector
        mock_qdrant_client.retrieve.return_value = [mock_point]

        result = centroid_qdrant.get_centroid_vector(centroid_id)

        assert result == expected_vector
        mock_qdrant_client.retrieve.assert_called_once_with(
            collection_name=CENTROID_COLLECTION_NAME,
            ids=[str(centroid_id)],
            with_payload=False,
            with_vectors=True,
        )

    def test_get_centroid_vector_not_found(
        self, centroid_qdrant: CentroidQdrantClient, mock_qdrant_client: MagicMock
    ) -> None:
        """Test retrieving a centroid that doesn't exist."""
        centroid_id = uuid.uuid4()

        # Mock: no points returned
        mock_qdrant_client.retrieve.return_value = []

        result = centroid_qdrant.get_centroid_vector(centroid_id)

        assert result is None

    def test_get_centroid_vector_handles_dict_format(
        self, centroid_qdrant: CentroidQdrantClient, mock_qdrant_client: MagicMock
    ) -> None:
        """Test handling named vector format (dict)."""
        centroid_id = uuid.uuid4()
        expected_vector = [0.1] * 512

        # Mock: named vector response (dict format)
        mock_point = MagicMock()
        mock_point.vector = {"default": expected_vector}
        mock_qdrant_client.retrieve.return_value = [mock_point]

        result = centroid_qdrant.get_centroid_vector(centroid_id)

        assert result == expected_vector

    def test_delete_centroid(
        self, centroid_qdrant: CentroidQdrantClient, mock_qdrant_client: MagicMock
    ) -> None:
        """Test deleting a centroid."""
        centroid_id = uuid.uuid4()

        centroid_qdrant.delete_centroid(centroid_id)

        # Verify delete was called
        mock_qdrant_client.delete.assert_called_once()
        delete_call = mock_qdrant_client.delete.call_args
        assert delete_call.kwargs["collection_name"] == CENTROID_COLLECTION_NAME
        assert str(centroid_id) in delete_call.kwargs["points_selector"].points

    def test_delete_centroids_by_person(
        self, centroid_qdrant: CentroidQdrantClient, mock_qdrant_client: MagicMock
    ) -> None:
        """Test deleting all centroids for a person."""
        person_id = uuid.uuid4()

        # Mock: scroll returns 2 batches of centroids
        mock_record1 = MagicMock()
        mock_record1.id = str(uuid.uuid4())
        mock_record2 = MagicMock()
        mock_record2.id = str(uuid.uuid4())
        mock_record3 = MagicMock()
        mock_record3.id = str(uuid.uuid4())

        # First scroll call returns 2 records, second returns 1, third returns empty
        mock_qdrant_client.scroll.side_effect = [
            ([mock_record1, mock_record2], "offset_1"),
            ([mock_record3], None),
        ]

        deleted_count = centroid_qdrant.delete_centroids_by_person(person_id)

        # Verify 3 centroids deleted in 2 batches
        assert deleted_count == 3
        assert mock_qdrant_client.delete.call_count == 2

    def test_get_collection_info_success(
        self, centroid_qdrant: CentroidQdrantClient, mock_qdrant_client: MagicMock
    ) -> None:
        """Test getting collection statistics."""
        # Mock collection info
        mock_collection_info = MagicMock(spec=CollectionInfo)
        mock_collection_info.points_count = 42
        mock_collection_info.indexed_vectors_count = 42
        mock_qdrant_client.get_collection.return_value = mock_collection_info

        result = centroid_qdrant.get_collection_info()

        assert result is not None
        assert result["name"] == CENTROID_COLLECTION_NAME
        assert result["points_count"] == 42
        assert result["vector_dim"] == 512
        assert result["distance"] == "cosine"

    def test_get_collection_info_not_found(
        self, centroid_qdrant: CentroidQdrantClient, mock_qdrant_client: MagicMock
    ) -> None:
        """Test getting collection info when collection doesn't exist."""
        # Mock: collection not found
        mock_qdrant_client.get_collection.side_effect = Exception("Collection not found")

        result = centroid_qdrant.get_collection_info()

        assert result is None

    def test_reset_collection(
        self, centroid_qdrant: CentroidQdrantClient, mock_qdrant_client: MagicMock
    ) -> None:
        """Test resetting the collection (delete + recreate)."""
        # Mock: collection exists with 10 centroids
        mock_collection_info = MagicMock(spec=CollectionInfo)
        mock_collection_info.points_count = 10
        mock_qdrant_client.get_collection.return_value = mock_collection_info

        # Mock: collection doesn't exist after deletion (for ensure_collection check)
        mock_collections_response = MagicMock()
        mock_collections_response.collections = []
        mock_qdrant_client.get_collections.return_value = mock_collections_response

        deleted_count = centroid_qdrant.reset_collection()

        assert deleted_count == 10
        # Verify delete and recreate were called
        mock_qdrant_client.delete_collection.assert_called_once_with(
            collection_name=CENTROID_COLLECTION_NAME
        )
        mock_qdrant_client.create_collection.assert_called_once()

    def test_singleton_pattern(self) -> None:
        """Test that CentroidQdrantClient follows singleton pattern."""
        # Clear singleton
        CentroidQdrantClient._instance = None

        instance1 = CentroidQdrantClient.get_instance()
        instance2 = CentroidQdrantClient.get_instance()

        assert instance1 is instance2

    def test_search_faces_with_centroid(
        self, centroid_qdrant: CentroidQdrantClient, mock_qdrant_client: MagicMock
    ) -> None:
        """Test searching faces collection using a centroid vector."""
        centroid_vector = [0.1] * 512

        # Mock query_points response
        mock_result = MagicMock()
        mock_scored_point = MagicMock()
        mock_scored_point.id = str(uuid.uuid4())
        mock_scored_point.score = 0.85
        mock_result.points = [mock_scored_point]
        mock_qdrant_client.query_points.return_value = mock_result

        results = centroid_qdrant.search_faces_with_centroid(
            centroid_vector=centroid_vector,
            limit=100,
            score_threshold=0.7,
        )

        assert len(results) == 1
        assert results[0].score == 0.85
        mock_qdrant_client.query_points.assert_called_once()

    def test_scroll_centroids_with_filters(
        self, centroid_qdrant: CentroidQdrantClient, mock_qdrant_client: MagicMock
    ) -> None:
        """Test scrolling centroids with person_id filter."""
        person_id = uuid.uuid4()

        # Mock scroll response
        mock_record = MagicMock()
        mock_record.id = str(uuid.uuid4())
        mock_record.payload = {"person_id": str(person_id)}
        mock_qdrant_client.scroll.return_value = ([mock_record], None)

        records, next_offset = centroid_qdrant.scroll_centroids(
            limit=100, filter_person_id=person_id
        )

        assert len(records) == 1
        assert next_offset is None
        # Verify filter was applied
        scroll_call = mock_qdrant_client.scroll.call_args
        assert scroll_call.kwargs["scroll_filter"] is not None
