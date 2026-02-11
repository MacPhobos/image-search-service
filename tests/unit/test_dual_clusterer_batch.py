"""Tests for batch embedding retrieval optimization in dual_clusterer.py."""

import uuid

import numpy as np
import pytest
from qdrant_client.models import PointStruct

from image_search_service.db.models import FaceInstance, ImageAsset, Person, PersonStatus
from image_search_service.faces.dual_clusterer import DualModeClusterer


@pytest.fixture
def mock_qdrant_for_batch(monkeypatch, use_test_settings, qdrant_client):
    """Inject in-memory Qdrant into dual_clusterer module.

    This ensures batch retrieval methods in dual_clusterer.py use our test
    Qdrant client with test data.
    """
    from image_search_service.core.config import get_settings
    from image_search_service.vector.face_qdrant import FaceQdrantClient

    settings = get_settings()

    # Reset singleton instance and set up test client
    FaceQdrantClient._instance = None
    FaceQdrantClient._client = None

    # Get singleton instance and inject test client
    face_qdrant = FaceQdrantClient.get_instance()
    face_qdrant._client = qdrant_client

    return qdrant_client, face_qdrant, settings


def make_test_embeddings(n: int, dim: int = 512, seed: int = 42) -> list[np.ndarray]:
    """Generate random normalized embeddings for testing.

    Args:
        n: Number of embeddings to generate
        dim: Embedding dimension
        seed: Random seed for reproducibility

    Returns:
        List of normalized embeddings
    """
    rng = np.random.RandomState(seed)
    embeddings = []
    for _ in range(n):
        emb = rng.randn(dim)
        emb = emb / np.linalg.norm(emb)
        embeddings.append(emb)
    return embeddings


class TestBatchEmbeddingRetrieval:
    """Tests for _get_face_embeddings_batch method."""

    def test_batch_retrieval_empty_input(self, sync_db_session, mock_qdrant_for_batch):
        """Test batch retrieval with empty input returns empty dict."""
        clusterer = DualModeClusterer(db_session=sync_db_session)

        result = clusterer._get_face_embeddings_batch([])

        assert result == {}

    def test_batch_retrieval_single_embedding(
        self, sync_db_session, mock_qdrant_for_batch
    ):
        """Test batch retrieval with single embedding works correctly."""
        qdrant_client, face_qdrant, settings = mock_qdrant_for_batch

        # Create single embedding in Qdrant
        point_id = uuid.uuid4()
        embedding = np.random.randn(512)
        embedding = embedding / np.linalg.norm(embedding)

        qdrant_client.upsert(
            collection_name=settings.qdrant_face_collection,
            points=[
                PointStruct(
                    id=str(point_id),
                    vector=embedding.tolist(),
                    payload={},
                )
            ],
        )

        clusterer = DualModeClusterer(db_session=sync_db_session)

        result = clusterer._get_face_embeddings_batch([point_id])

        assert len(result) == 1
        assert point_id in result
        # Check embedding is close to original (allow floating point error)
        np.testing.assert_allclose(result[point_id], embedding, rtol=1e-5)

    def test_batch_retrieval_multiple_embeddings(
        self, sync_db_session, mock_qdrant_for_batch
    ):
        """Test batch retrieval with multiple embeddings (less than batch size)."""
        qdrant_client, face_qdrant, settings = mock_qdrant_for_batch

        # Create 10 embeddings (less than default batch size of 100)
        embeddings = make_test_embeddings(10)
        point_ids = [uuid.uuid4() for _ in range(10)]

        # Insert into Qdrant
        points = []
        for point_id, embedding in zip(point_ids, embeddings):
            points.append(
                PointStruct(
                    id=str(point_id),
                    vector=embedding.tolist(),
                    payload={},
                )
            )

        qdrant_client.upsert(
            collection_name=settings.qdrant_face_collection,
            points=points,
        )

        clusterer = DualModeClusterer(db_session=sync_db_session)

        result = clusterer._get_face_embeddings_batch(point_ids)

        assert len(result) == 10
        for point_id, expected_embedding in zip(point_ids, embeddings):
            assert point_id in result
            np.testing.assert_allclose(result[point_id], expected_embedding, rtol=1e-5)

    def test_batch_retrieval_exact_batch_size(
        self, sync_db_session, mock_qdrant_for_batch
    ):
        """Test batch retrieval when input equals batch size."""
        qdrant_client, face_qdrant, settings = mock_qdrant_for_batch

        # Create exactly 100 embeddings (default batch size)
        embeddings = make_test_embeddings(100)
        point_ids = [uuid.uuid4() for _ in range(100)]

        # Insert into Qdrant
        points = []
        for point_id, embedding in zip(point_ids, embeddings):
            points.append(
                PointStruct(
                    id=str(point_id),
                    vector=embedding.tolist(),
                    payload={},
                )
            )

        qdrant_client.upsert(
            collection_name=settings.qdrant_face_collection,
            points=points,
        )

        clusterer = DualModeClusterer(db_session=sync_db_session)

        result = clusterer._get_face_embeddings_batch(point_ids, batch_size=100)

        assert len(result) == 100
        for point_id, expected_embedding in zip(point_ids, embeddings):
            assert point_id in result
            np.testing.assert_allclose(result[point_id], expected_embedding, rtol=1e-5)

    def test_batch_retrieval_multiple_batches(
        self, sync_db_session, mock_qdrant_for_batch
    ):
        """Test batch retrieval when input requires multiple batches."""
        qdrant_client, face_qdrant, settings = mock_qdrant_for_batch

        # Create 250 embeddings (requires 3 batches at batch_size=100)
        embeddings = make_test_embeddings(250)
        point_ids = [uuid.uuid4() for _ in range(250)]

        # Insert into Qdrant
        points = []
        for point_id, embedding in zip(point_ids, embeddings):
            points.append(
                PointStruct(
                    id=str(point_id),
                    vector=embedding.tolist(),
                    payload={},
                )
            )

        qdrant_client.upsert(
            collection_name=settings.qdrant_face_collection,
            points=points,
        )

        clusterer = DualModeClusterer(db_session=sync_db_session)

        result = clusterer._get_face_embeddings_batch(point_ids, batch_size=100)

        assert len(result) == 250
        for point_id, expected_embedding in zip(point_ids, embeddings):
            assert point_id in result
            np.testing.assert_allclose(result[point_id], expected_embedding, rtol=1e-5)

    def test_batch_retrieval_custom_batch_size(
        self, sync_db_session, mock_qdrant_for_batch
    ):
        """Test batch retrieval with custom batch size."""
        qdrant_client, face_qdrant, settings = mock_qdrant_for_batch

        # Create 50 embeddings, use batch_size=10
        embeddings = make_test_embeddings(50)
        point_ids = [uuid.uuid4() for _ in range(50)]

        # Insert into Qdrant
        points = []
        for point_id, embedding in zip(point_ids, embeddings):
            points.append(
                PointStruct(
                    id=str(point_id),
                    vector=embedding.tolist(),
                    payload={},
                )
            )

        qdrant_client.upsert(
            collection_name=settings.qdrant_face_collection,
            points=points,
        )

        clusterer = DualModeClusterer(db_session=sync_db_session)

        result = clusterer._get_face_embeddings_batch(point_ids, batch_size=10)

        # Should require 5 batches and retrieve all 50
        assert len(result) == 50
        for point_id, expected_embedding in zip(point_ids, embeddings):
            assert point_id in result
            np.testing.assert_allclose(result[point_id], expected_embedding, rtol=1e-5)

    def test_batch_retrieval_missing_ids(
        self, sync_db_session, mock_qdrant_for_batch
    ):
        """Test batch retrieval when some IDs don't exist in Qdrant."""
        qdrant_client, face_qdrant, settings = mock_qdrant_for_batch

        # Create 5 embeddings
        embeddings = make_test_embeddings(5)
        point_ids = [uuid.uuid4() for _ in range(5)]

        # Insert only 3 into Qdrant
        points = []
        for point_id, embedding in zip(point_ids[:3], embeddings[:3]):
            points.append(
                PointStruct(
                    id=str(point_id),
                    vector=embedding.tolist(),
                    payload={},
                )
            )

        qdrant_client.upsert(
            collection_name=settings.qdrant_face_collection,
            points=points,
        )

        clusterer = DualModeClusterer(db_session=sync_db_session)

        # Request all 5, but only 3 exist
        result = clusterer._get_face_embeddings_batch(point_ids)

        assert len(result) == 3
        for point_id, expected_embedding in zip(point_ids[:3], embeddings[:3]):
            assert point_id in result
            np.testing.assert_allclose(result[point_id], expected_embedding, rtol=1e-5)
        # Missing IDs should not be in result
        assert point_ids[3] not in result
        assert point_ids[4] not in result

    def test_batch_retrieval_preserves_order(
        self, sync_db_session, mock_qdrant_for_batch
    ):
        """Test batch retrieval returns correct embeddings even if order differs."""
        qdrant_client, face_qdrant, settings = mock_qdrant_for_batch

        # Create 10 embeddings
        embeddings = make_test_embeddings(10)
        point_ids = [uuid.uuid4() for _ in range(10)]

        # Insert into Qdrant
        points = []
        for point_id, embedding in zip(point_ids, embeddings):
            points.append(
                PointStruct(
                    id=str(point_id),
                    vector=embedding.tolist(),
                    payload={},
                )
            )

        qdrant_client.upsert(
            collection_name=settings.qdrant_face_collection,
            points=points,
        )

        clusterer = DualModeClusterer(db_session=sync_db_session)

        # Request in shuffled order
        import random

        shuffled_ids = point_ids.copy()
        random.seed(123)
        random.shuffle(shuffled_ids)

        result = clusterer._get_face_embeddings_batch(shuffled_ids)

        assert len(result) == 10
        # Verify each ID maps to its correct embedding (not shuffled embedding)
        for point_id, expected_embedding in zip(point_ids, embeddings):
            assert point_id in result
            np.testing.assert_allclose(result[point_id], expected_embedding, rtol=1e-5)


class TestBatchIntegrationWithClustering:
    """Integration tests verifying batch retrieval works in clustering workflow."""

    def test_assign_to_known_people_uses_batch(
        self, sync_db_session, sync_db_engine, mock_qdrant_for_batch
    ):
        """Test that assign_to_known_people uses batch retrieval correctly."""
        qdrant_client, face_qdrant, settings = mock_qdrant_for_batch

        # Create image asset
        asset = ImageAsset(
            path="/test/image.jpg",
            training_status="completed",
            width=640,
            height=480,
            file_size=1024,
            mime_type="image/jpeg",
        )
        sync_db_session.add(asset)
        sync_db_session.commit()

        # Create person
        person_id = uuid.uuid4()
        person = Person(id=person_id, name="Test Person", status=PersonStatus.ACTIVE)
        sync_db_session.add(person)
        sync_db_session.commit()

        # Create 50 labeled faces (to test batch retrieval)
        labeled_faces = []
        base_embedding = np.random.randn(512)
        base_embedding = base_embedding / np.linalg.norm(base_embedding)

        for i in range(50):
            point_id = uuid.uuid4()

            # Small noise around base embedding
            embedding = base_embedding + np.random.randn(512) * 0.01
            embedding = embedding / np.linalg.norm(embedding)

            # Add to Qdrant
            qdrant_client.upsert(
                collection_name=settings.qdrant_face_collection,
                points=[
                    PointStruct(
                        id=str(point_id),
                        vector=embedding.tolist(),
                        payload={"person_id": str(person_id)},
                    )
                ],
            )

            labeled_faces.append(
                {
                    "id": uuid.uuid4(),
                    "qdrant_point_id": point_id,
                    "person_id": person_id,
                    "cluster_id": None,
                }
            )

        # Create 10 unlabeled faces similar to person
        unlabeled_faces = []
        for i in range(10):
            point_id = uuid.uuid4()
            embedding = base_embedding + np.random.randn(512) * 0.01
            embedding = embedding / np.linalg.norm(embedding)

            qdrant_client.upsert(
                collection_name=settings.qdrant_face_collection,
                points=[
                    PointStruct(
                        id=str(point_id),
                        vector=embedding.tolist(),
                        payload={},
                    )
                ],
            )

            unlabeled_faces.append(
                {
                    "id": uuid.uuid4(),
                    "qdrant_point_id": point_id,
                    "person_id": None,
                    "cluster_id": None,
                }
            )

        clusterer = DualModeClusterer(
            db_session=sync_db_session, person_match_threshold=0.7
        )

        assigned, still_unknown = clusterer.assign_to_known_people(
            unlabeled_faces=unlabeled_faces, labeled_faces=labeled_faces
        )

        # All unlabeled faces should be assigned (high similarity to person)
        assert len(assigned) == 10
        assert len(still_unknown) == 0
        for assignment in assigned:
            assert assignment["person_id"] == person_id
            assert assignment["similarity"] >= 0.7

    def test_cluster_unknown_faces_uses_batch(
        self, sync_db_session, mock_qdrant_for_batch
    ):
        """Test that cluster_unknown_faces uses batch retrieval correctly."""
        qdrant_client, face_qdrant, settings = mock_qdrant_for_batch

        # Create 100 embeddings with 2 clear clusters (50 each)
        cluster1_center = np.random.randn(512)
        cluster1_center = cluster1_center / np.linalg.norm(cluster1_center)

        cluster2_center = -cluster1_center  # Opposite direction

        unknown_faces = []

        # Cluster 1: 50 faces
        for i in range(50):
            point_id = uuid.uuid4()
            embedding = cluster1_center + np.random.randn(512) * 0.05
            embedding = embedding / np.linalg.norm(embedding)

            qdrant_client.upsert(
                collection_name=settings.qdrant_face_collection,
                points=[
                    PointStruct(
                        id=str(point_id),
                        vector=embedding.tolist(),
                        payload={},
                    )
                ],
            )

            unknown_faces.append(
                {
                    "id": uuid.uuid4(),
                    "qdrant_point_id": point_id,
                    "person_id": None,
                    "cluster_id": None,
                }
            )

        # Cluster 2: 50 faces
        for i in range(50):
            point_id = uuid.uuid4()
            embedding = cluster2_center + np.random.randn(512) * 0.05
            embedding = embedding / np.linalg.norm(embedding)

            qdrant_client.upsert(
                collection_name=settings.qdrant_face_collection,
                points=[
                    PointStruct(
                        id=str(point_id),
                        vector=embedding.tolist(),
                        payload={},
                    )
                ],
            )

            unknown_faces.append(
                {
                    "id": uuid.uuid4(),
                    "qdrant_point_id": point_id,
                    "person_id": None,
                    "cluster_id": None,
                }
            )

        clusterer = DualModeClusterer(
            db_session=sync_db_session,
            unknown_method="hdbscan",
            unknown_min_cluster_size=10,
        )

        clusters = clusterer.cluster_unknown_faces(unknown_faces)

        # Should cluster all 100 faces
        assert len(clusters) == 100

        # Should find at least 2 distinct clusters
        unique_clusters = set(clusters.values())
        non_noise_clusters = [c for c in unique_clusters if "noise" not in c]
        assert len(non_noise_clusters) >= 2

    def test_full_pipeline_with_batch_retrieval(
        self, sync_db_session, sync_db_engine, mock_qdrant_for_batch
    ):
        """Test full clustering pipeline uses batch retrieval efficiently."""
        qdrant_client, face_qdrant, settings = mock_qdrant_for_batch

        # Create image asset
        asset = ImageAsset(
            path="/test/image.jpg",
            training_status="completed",
            width=640,
            height=480,
            file_size=1024,
            mime_type="image/jpeg",
        )
        sync_db_session.add(asset)
        sync_db_session.commit()

        # Create person
        person_id = uuid.uuid4()
        person = Person(id=person_id, name="Test Person", status=PersonStatus.ACTIVE)
        sync_db_session.add(person)
        sync_db_session.commit()

        # Create 30 labeled faces for person
        person_embedding = np.random.randn(512)
        person_embedding = person_embedding / np.linalg.norm(person_embedding)

        for i in range(30):
            point_id = uuid.uuid4()
            embedding = person_embedding + np.random.randn(512) * 0.01
            embedding = embedding / np.linalg.norm(embedding)

            face = FaceInstance(
                id=uuid.uuid4(),
                asset_id=asset.id,
                bbox_x=100 + i * 10,
                bbox_y=100 + (i % 10) * 10,
                bbox_w=50,
                bbox_h=50,
                detection_confidence=0.9,
                qdrant_point_id=point_id,
                person_id=person_id,
            )
            sync_db_session.add(face)

            qdrant_client.upsert(
                collection_name=settings.qdrant_face_collection,
                points=[
                    PointStruct(
                        id=str(point_id),
                        vector=embedding.tolist(),
                        payload={"person_id": str(person_id)},
                    )
                ],
            )

        # Create 20 unlabeled faces similar to person (should be assigned)
        similar_face_ids = []
        for i in range(20):
            point_id = uuid.uuid4()
            face_id = uuid.uuid4()
            embedding = person_embedding + np.random.randn(512) * 0.01
            embedding = embedding / np.linalg.norm(embedding)

            face = FaceInstance(
                id=face_id,
                asset_id=asset.id,
                bbox_x=500 + i * 10,
                bbox_y=100 + (i % 10) * 10,
                bbox_w=50,
                bbox_h=50,
                detection_confidence=0.9,
                qdrant_point_id=point_id,
                person_id=None,
            )
            sync_db_session.add(face)

            qdrant_client.upsert(
                collection_name=settings.qdrant_face_collection,
                points=[
                    PointStruct(
                        id=str(point_id),
                        vector=embedding.tolist(),
                        payload={},
                    )
                ],
            )

            similar_face_ids.append(face_id)

        # Create 50 unknown faces (different cluster, should cluster together)
        unknown_center = -person_embedding  # Opposite direction
        unknown_face_ids = []
        for i in range(50):
            point_id = uuid.uuid4()
            face_id = uuid.uuid4()
            embedding = unknown_center + np.random.randn(512) * 0.05
            embedding = embedding / np.linalg.norm(embedding)

            face = FaceInstance(
                id=face_id,
                asset_id=asset.id,
                bbox_x=1000 + i * 10,
                bbox_y=100 + (i % 10) * 10,
                bbox_w=50,
                bbox_h=50,
                detection_confidence=0.9,
                qdrant_point_id=point_id,
                person_id=None,
            )
            sync_db_session.add(face)

            qdrant_client.upsert(
                collection_name=settings.qdrant_face_collection,
                points=[
                    PointStruct(
                        id=str(point_id),
                        vector=embedding.tolist(),
                        payload={},
                    )
                ],
            )

            unknown_face_ids.append(face_id)

        sync_db_session.commit()

        # Run full clustering pipeline
        clusterer = DualModeClusterer(
            db_session=sync_db_session,
            person_match_threshold=0.7,
            unknown_min_cluster_size=10,
            unknown_method="hdbscan",
        )

        result = clusterer.cluster_all_faces()

        # Verify results
        assert result["total_processed"] == 100  # 30 labeled + 20 similar + 50 unknown
        assert result["assigned_to_people"] == 20  # Similar faces assigned

        # Verify similar faces were assigned to person
        for face_id in similar_face_ids:
            face = sync_db_session.get(FaceInstance, face_id)
            assert face is not None
            assert face.person_id == person_id

        # Verify unknown faces got clustered
        unknown_clusters = set()
        for face_id in unknown_face_ids:
            face = sync_db_session.get(FaceInstance, face_id)
            assert face is not None
            if face.cluster_id:
                unknown_clusters.add(face.cluster_id)

        # Should have at least 1 unknown cluster
        assert len(unknown_clusters) >= 1
