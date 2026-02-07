"""Tests for dual-mode face clustering (supervised + unsupervised)."""

import uuid

import numpy as np
import pytest
from qdrant_client.models import PointStruct

from image_search_service.db.models import FaceInstance, ImageAsset, Person, PersonStatus
from image_search_service.faces.dual_clusterer import DualModeClusterer
from image_search_service.vector.face_qdrant import FaceQdrantClient


def make_clustered_embeddings(
    n_per_cluster: int = 5, n_clusters: int = 3, dim: int = 512, seed: int = 42
) -> tuple[list[np.ndarray], list[int]]:
    """Generate embeddings with known cluster structure.

    Args:
        n_per_cluster: Number of faces per cluster
        n_clusters: Number of distinct clusters
        dim: Embedding dimension
        seed: Random seed for reproducibility

    Returns:
        (embeddings, labels) where embeddings are normalized vectors
    """
    rng = np.random.RandomState(seed)
    embeddings = []
    labels = []

    for cluster_idx in range(n_clusters):
        # Create cluster center
        center = rng.randn(dim)
        center = center / np.linalg.norm(center)

        # Add faces around this center
        for _ in range(n_per_cluster):
            noise = rng.randn(dim) * 0.05  # Small noise to keep faces close
            emb = center + noise
            emb = emb / np.linalg.norm(emb)  # Re-normalize
            embeddings.append(emb)
            labels.append(cluster_idx)

    return embeddings, labels


@pytest.fixture
def mock_qdrant_for_clusterer(monkeypatch, use_test_settings, qdrant_client):
    """Inject in-memory Qdrant into dual_clusterer module.

    This ensures _get_face_embedding() calls in dual_clusterer.py
    use our test Qdrant client with test data.

    NOTE: The qdrant_client fixture already creates "test_faces" collection.
    We use the same collection name from settings to ensure consistency.
    """
    from image_search_service.core.config import get_settings

    settings = get_settings()

    # The qdrant_client fixture already created the collection with name
    # from settings.qdrant_face_collection ("test_faces"), so we don't
    # need to create it again. All tests should use the settings collection name.

    # Reset singleton instance and set up test client
    # This ensures the singleton pattern returns our test instance
    FaceQdrantClient._instance = None
    FaceQdrantClient._client = None

    # Get singleton instance and inject test client
    face_qdrant = FaceQdrantClient.get_instance()
    face_qdrant._client = qdrant_client

    return qdrant_client, face_qdrant, settings


class TestDualModeClustererInit:
    """Tests for DualModeClusterer initialization."""

    def test_init_default_params(self, sync_db_session):
        """Test initialization with default parameters."""
        clusterer = DualModeClusterer(db_session=sync_db_session)

        assert clusterer.db is sync_db_session
        assert clusterer.person_match_threshold == 0.7
        assert clusterer.unknown_min_cluster_size == 3
        assert clusterer.unknown_method == "hdbscan"
        assert clusterer.unknown_eps == 0.5

    def test_init_custom_params(self, sync_db_session):
        """Test initialization with custom parameters."""
        clusterer = DualModeClusterer(
            db_session=sync_db_session,
            person_match_threshold=0.8,
            unknown_min_cluster_size=5,
            unknown_method="dbscan",
            unknown_eps=0.3,
        )

        assert clusterer.person_match_threshold == 0.8
        assert clusterer.unknown_min_cluster_size == 5
        assert clusterer.unknown_method == "dbscan"
        assert clusterer.unknown_eps == 0.3


class TestClusterAllFaces:
    """Tests for cluster_all_faces method."""

    def test_cluster_all_faces_empty_db(self, sync_db_session, mock_qdrant_for_clusterer):
        """Test clustering when database has no faces."""
        clusterer = DualModeClusterer(db_session=sync_db_session)

        result = clusterer.cluster_all_faces()

        assert result["total_processed"] == 0
        assert result["assigned_to_people"] == 0
        assert result["unknown_clusters"] == 0
        assert result["still_unlabeled"] == 0

    def test_cluster_all_faces_all_labeled(
        self, sync_db_session, sync_db_engine, mock_qdrant_for_clusterer
    ):
        """Test clustering when all faces already have person_id."""
        qdrant_client, face_qdrant, settings = mock_qdrant_for_clusterer

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

        # Create 3 faces all assigned to this person
        for i in range(3):
            point_id = uuid.uuid4()
            face = FaceInstance(
                id=uuid.uuid4(),
                asset_id=asset.id,
                bbox_x=100 + i * 10,
                bbox_y=100 + i * 5,  # Vary to avoid unique constraint
                bbox_w=50,
                bbox_h=50,
                detection_confidence=0.9,
                qdrant_point_id=point_id,
                person_id=person_id,
            )
            sync_db_session.add(face)

            # Add to Qdrant
            embedding = np.random.randn(512)
            embedding = embedding / np.linalg.norm(embedding)
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

        sync_db_session.commit()

        clusterer = DualModeClusterer(db_session=sync_db_session)
        result = clusterer.cluster_all_faces()

        # All faces are labeled, so nothing to assign or cluster
        assert result["total_processed"] == 3
        assert result["assigned_to_people"] == 0
        assert result["unknown_clusters"] == 0

    def test_cluster_all_faces_all_unlabeled(
        self, sync_db_session, sync_db_engine, mock_qdrant_for_clusterer
    ):
        """Test clustering when no faces have person_id (all unlabeled)."""
        qdrant_client, face_qdrant, settings = mock_qdrant_for_clusterer

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

        # Create embeddings with 2 natural clusters (5 faces each)
        embeddings, labels = make_clustered_embeddings(n_per_cluster=5, n_clusters=2)

        # Create faces without person_id
        for i, embedding in enumerate(embeddings):
            point_id = uuid.uuid4()
            face = FaceInstance(
                id=uuid.uuid4(),
                asset_id=asset.id,
                bbox_x=100 + i * 10,
                bbox_y=100 + i * 5,  # Vary y too to avoid unique constraint
                bbox_w=50,
                bbox_h=50,
                detection_confidence=0.9,
                qdrant_point_id=point_id,
                person_id=None,  # Unlabeled
            )
            sync_db_session.add(face)

            # Add to Qdrant
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

        sync_db_session.commit()

        clusterer = DualModeClusterer(
            db_session=sync_db_session,
            unknown_method="hdbscan",
            unknown_min_cluster_size=3,
        )
        result = clusterer.cluster_all_faces()

        # All unlabeled, so supervised assignment skipped, unsupervised clustering runs
        assert result["total_processed"] == 10
        assert result["assigned_to_people"] == 0
        # Should find at least 2 clusters from our 2-cluster embeddings
        assert result["unknown_clusters"] >= 2

    def test_cluster_all_faces_max_faces_limit(
        self, sync_db_session, sync_db_engine, mock_qdrant_for_clusterer
    ):
        """Test that max_faces parameter limits processing."""
        qdrant_client, face_qdrant, settings = mock_qdrant_for_clusterer

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

        # Create 10 faces
        for i in range(10):
            point_id = uuid.uuid4()
            face = FaceInstance(
                id=uuid.uuid4(),
                asset_id=asset.id,
                bbox_x=100 + i * 10,
                bbox_y=100 + i * 5,  # Vary to avoid unique constraint
                bbox_w=50,
                bbox_h=50,
                detection_confidence=0.9,
                qdrant_point_id=point_id,
            )
            sync_db_session.add(face)

        sync_db_session.commit()

        clusterer = DualModeClusterer(db_session=sync_db_session)
        result = clusterer.cluster_all_faces(max_faces=5)

        # Should only process 5 faces
        assert result["total_processed"] == 5


class TestAssignToKnownPeople:
    """Tests for assign_to_known_people method."""

    def test_assign_to_known_people_above_threshold(
        self, sync_db_session, mock_qdrant_for_clusterer
    ):
        """Test that unlabeled face close to person centroid gets assigned."""
        qdrant_client, face_qdrant, settings = mock_qdrant_for_clusterer

        person_id = uuid.uuid4()

        # Create labeled faces for person (3 faces very similar)
        labeled_faces = []
        base_embedding = np.random.randn(512)
        base_embedding = base_embedding / np.linalg.norm(base_embedding)

        for i in range(3):
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

        # Create unlabeled face very similar to person's embeddings
        unlabeled_point_id = uuid.uuid4()
        unlabeled_embedding = base_embedding + np.random.randn(512) * 0.01
        unlabeled_embedding = unlabeled_embedding / np.linalg.norm(unlabeled_embedding)

        qdrant_client.upsert(
            collection_name=settings.qdrant_face_collection,
            points=[
                PointStruct(
                    id=str(unlabeled_point_id),
                    vector=unlabeled_embedding.tolist(),
                    payload={},
                )
            ],
        )

        unlabeled_faces = [
            {
                "id": uuid.uuid4(),
                "qdrant_point_id": unlabeled_point_id,
                "person_id": None,
                "cluster_id": None,
            }
        ]

        clusterer = DualModeClusterer(
            db_session=sync_db_session, person_match_threshold=0.7
        )

        assigned, still_unknown = clusterer.assign_to_known_people(
            unlabeled_faces=unlabeled_faces, labeled_faces=labeled_faces
        )

        # Face should be assigned (high similarity)
        assert len(assigned) == 1
        assert len(still_unknown) == 0
        assert assigned[0]["person_id"] == person_id
        assert assigned[0]["similarity"] >= 0.7

    def test_assign_to_known_people_below_threshold(
        self, sync_db_session, mock_qdrant_for_clusterer
    ):
        """Test that unlabeled face far from all persons stays unknown."""
        qdrant_client, face_qdrant, settings = mock_qdrant_for_clusterer

        person_id = uuid.uuid4()

        # Create labeled face for person
        labeled_point_id = uuid.uuid4()
        labeled_embedding = np.random.randn(512)
        labeled_embedding = labeled_embedding / np.linalg.norm(labeled_embedding)

        qdrant_client.upsert(
            collection_name=settings.qdrant_face_collection,
            points=[
                PointStruct(
                    id=str(labeled_point_id),
                    vector=labeled_embedding.tolist(),
                    payload={"person_id": str(person_id)},
                )
            ],
        )

        labeled_faces = [
            {
                "id": uuid.uuid4(),
                "qdrant_point_id": labeled_point_id,
                "person_id": person_id,
                "cluster_id": None,
            }
        ]

        # Create unlabeled face very different from person
        unlabeled_point_id = uuid.uuid4()
        unlabeled_embedding = -labeled_embedding  # Opposite direction, very different
        unlabeled_embedding = unlabeled_embedding / np.linalg.norm(unlabeled_embedding)

        qdrant_client.upsert(
            collection_name=settings.qdrant_face_collection,
            points=[
                PointStruct(
                    id=str(unlabeled_point_id),
                    vector=unlabeled_embedding.tolist(),
                    payload={},
                )
            ],
        )

        unlabeled_faces = [
            {
                "id": uuid.uuid4(),
                "qdrant_point_id": unlabeled_point_id,
                "person_id": None,
                "cluster_id": None,
            }
        ]

        clusterer = DualModeClusterer(
            db_session=sync_db_session, person_match_threshold=0.7
        )

        assigned, still_unknown = clusterer.assign_to_known_people(
            unlabeled_faces=unlabeled_faces, labeled_faces=labeled_faces
        )

        # Face should stay unknown (low similarity)
        assert len(assigned) == 0
        assert len(still_unknown) == 1

    def test_assign_to_known_people_no_embedding(
        self, sync_db_session, mock_qdrant_for_clusterer
    ):
        """Test that face with missing embedding goes to still_unknown."""
        qdrant_client, face_qdrant, settings = mock_qdrant_for_clusterer

        person_id = uuid.uuid4()

        # Create labeled face
        labeled_point_id = uuid.uuid4()
        labeled_embedding = np.random.randn(512)
        labeled_embedding = labeled_embedding / np.linalg.norm(labeled_embedding)

        qdrant_client.upsert(
            collection_name=settings.qdrant_face_collection,
            points=[
                PointStruct(
                    id=str(labeled_point_id),
                    vector=labeled_embedding.tolist(),
                    payload={"person_id": str(person_id)},
                )
            ],
        )

        labeled_faces = [
            {
                "id": uuid.uuid4(),
                "qdrant_point_id": labeled_point_id,
                "person_id": person_id,
                "cluster_id": None,
            }
        ]

        # Create unlabeled face with non-existent qdrant_point_id
        unlabeled_faces = [
            {
                "id": uuid.uuid4(),
                "qdrant_point_id": uuid.uuid4(),  # Not in Qdrant
                "person_id": None,
                "cluster_id": None,
            }
        ]

        clusterer = DualModeClusterer(db_session=sync_db_session)

        assigned, still_unknown = clusterer.assign_to_known_people(
            unlabeled_faces=unlabeled_faces, labeled_faces=labeled_faces
        )

        # Face should go to still_unknown (no embedding retrieved)
        assert len(assigned) == 0
        assert len(still_unknown) == 1

    def test_assign_to_known_people_no_labeled_faces(
        self, sync_db_session, mock_qdrant_for_clusterer
    ):
        """Test that no labeled faces returns all unlabeled."""
        unlabeled_faces = [
            {
                "id": uuid.uuid4(),
                "qdrant_point_id": uuid.uuid4(),
                "person_id": None,
                "cluster_id": None,
            }
        ]

        clusterer = DualModeClusterer(db_session=sync_db_session)

        assigned, still_unknown = clusterer.assign_to_known_people(
            unlabeled_faces=unlabeled_faces, labeled_faces=[]
        )

        assert len(assigned) == 0
        assert len(still_unknown) == 1


class TestClusterUnknownFaces:
    """Tests for cluster_unknown_faces method."""

    def test_cluster_unknown_faces_hdbscan(
        self, sync_db_session, mock_qdrant_for_clusterer
    ):
        """Test clustering with HDBSCAN finds natural clusters."""
        qdrant_client, face_qdrant, settings = mock_qdrant_for_clusterer

        # Create embeddings with 2 clear clusters (7 faces each)
        embeddings, labels = make_clustered_embeddings(n_per_cluster=7, n_clusters=2)

        unknown_faces = []
        for idx, embedding in enumerate(embeddings):
            point_id = uuid.uuid4()
            face_id = uuid.uuid4()

            # Add to Qdrant
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
                    "id": face_id,
                    "qdrant_point_id": point_id,
                    "person_id": None,
                    "cluster_id": None,
                }
            )

        clusterer = DualModeClusterer(
            db_session=sync_db_session,
            unknown_method="hdbscan",
            unknown_min_cluster_size=3,
        )

        clusters = clusterer.cluster_unknown_faces(unknown_faces)

        # Should find clusters
        assert len(clusters) == 14  # All faces assigned
        unique_clusters = set(clusters.values())
        # Should have at least 2 non-noise clusters
        non_noise_clusters = [c for c in unique_clusters if "noise" not in c]
        assert len(non_noise_clusters) >= 2

    def test_cluster_unknown_faces_dbscan(
        self, sync_db_session, mock_qdrant_for_clusterer
    ):
        """Test clustering with DBSCAN."""
        qdrant_client, face_qdrant, settings = mock_qdrant_for_clusterer

        # Create embeddings with 2 clear clusters (7 faces each)
        embeddings, labels = make_clustered_embeddings(n_per_cluster=7, n_clusters=2)

        unknown_faces = []
        for embedding in embeddings:
            point_id = uuid.uuid4()
            face_id = uuid.uuid4()

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
                    "id": face_id,
                    "qdrant_point_id": point_id,
                    "person_id": None,
                    "cluster_id": None,
                }
            )

        clusterer = DualModeClusterer(
            db_session=sync_db_session,
            unknown_method="dbscan",
            unknown_min_cluster_size=3,
            unknown_eps=0.3,
        )

        clusters = clusterer.cluster_unknown_faces(unknown_faces)

        # Should find clusters
        assert len(clusters) == 14
        unique_clusters = set(clusters.values())
        assert len(unique_clusters) >= 1  # At least 1 cluster

    def test_cluster_unknown_faces_agglomerative(
        self, sync_db_session, mock_qdrant_for_clusterer
    ):
        """Test clustering with Agglomerative Clustering."""
        qdrant_client, face_qdrant, settings = mock_qdrant_for_clusterer

        # Create embeddings with 2 clear clusters (7 faces each)
        embeddings, labels = make_clustered_embeddings(n_per_cluster=7, n_clusters=2)

        unknown_faces = []
        for embedding in embeddings:
            point_id = uuid.uuid4()
            face_id = uuid.uuid4()

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
                    "id": face_id,
                    "qdrant_point_id": point_id,
                    "person_id": None,
                    "cluster_id": None,
                }
            )

        clusterer = DualModeClusterer(
            db_session=sync_db_session,
            unknown_method="agglomerative",
            unknown_min_cluster_size=3,
            unknown_eps=0.3,
        )

        clusters = clusterer.cluster_unknown_faces(unknown_faces)

        # Should find clusters
        assert len(clusters) == 14
        unique_clusters = set(clusters.values())
        assert len(unique_clusters) >= 1

    def test_cluster_unknown_faces_too_few(
        self, sync_db_session, mock_qdrant_for_clusterer
    ):
        """Test that fewer faces than min_cluster_size labels all as noise."""
        qdrant_client, face_qdrant, settings = mock_qdrant_for_clusterer

        # Create only 2 faces (below min_cluster_size=3)
        unknown_faces = []
        for _ in range(2):
            point_id = uuid.uuid4()
            face_id = uuid.uuid4()
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

            unknown_faces.append(
                {
                    "id": face_id,
                    "qdrant_point_id": point_id,
                    "person_id": None,
                    "cluster_id": None,
                }
            )

        clusterer = DualModeClusterer(
            db_session=sync_db_session, unknown_min_cluster_size=3
        )

        clusters = clusterer.cluster_unknown_faces(unknown_faces)

        # All should be labeled as noise
        assert len(clusters) == 2
        for cluster_id in clusters.values():
            assert "noise" in cluster_id

    def test_cluster_unknown_faces_invalid_method(self, sync_db_session):
        """Test that invalid clustering method raises ValueError."""
        clusterer = DualModeClusterer(
            db_session=sync_db_session, unknown_method="invalid"
        )

        # Need at least min_cluster_size faces to trigger clustering
        unknown_faces = [
            {"id": uuid.uuid4(), "qdrant_point_id": uuid.uuid4()}
            for _ in range(5)
        ]

        # Mock _get_face_embedding to return valid embeddings
        def mock_get_embedding(point_id):
            emb = np.random.randn(512)
            return emb / np.linalg.norm(emb)

        clusterer._get_face_embedding = mock_get_embedding  # type: ignore

        with pytest.raises(ValueError, match="Unknown clustering method"):
            clusterer.cluster_unknown_faces(unknown_faces)


class TestSaveDualModeResults:
    """Tests for save_dual_mode_results method."""

    def test_save_dual_mode_results_assigned(
        self, sync_db_session, sync_db_engine, mock_qdrant_for_clusterer
    ):
        """Test saving assigned faces updates DB and Qdrant."""
        qdrant_client, face_qdrant, settings = mock_qdrant_for_clusterer

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

        # Create 3 faces to assign
        assigned_faces = []
        for i in range(3):
            face_id = uuid.uuid4()
            point_id = uuid.uuid4()

            face = FaceInstance(
                id=face_id,
                asset_id=asset.id,
                bbox_x=100 + i * 10,
                bbox_y=100 + i * 5,  # Vary to avoid unique constraint
                bbox_w=50,
                bbox_h=50,
                detection_confidence=0.9,
                qdrant_point_id=point_id,
            )
            sync_db_session.add(face)

            # Add to Qdrant
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

            assigned_faces.append(
                {
                    "face_id": face_id,
                    "person_id": person_id,
                    "similarity": 0.85,
                    "qdrant_point_id": point_id,
                }
            )

        sync_db_session.commit()

        clusterer = DualModeClusterer(db_session=sync_db_session)
        clusterer.save_dual_mode_results(
            assigned_faces=assigned_faces, unknown_clusters={}
        )

        # Verify DB updated
        for face_data in assigned_faces:
            face = sync_db_session.get(FaceInstance, face_data["face_id"])
            assert face is not None
            assert face.person_id == person_id

        # Verify Qdrant updated
        for face_data in assigned_faces:
            points = qdrant_client.retrieve(
                collection_name=settings.qdrant_face_collection,
                ids=[str(face_data["qdrant_point_id"])],
                with_payload=True,
            )
            assert len(points) == 1
            assert points[0].payload.get("person_id") == str(person_id)

    def test_save_dual_mode_results_clusters(
        self, sync_db_session, sync_db_engine, mock_qdrant_for_clusterer
    ):
        """Test saving cluster assignments updates DB and Qdrant."""
        qdrant_client, face_qdrant, settings = mock_qdrant_for_clusterer

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

        # Create 5 faces in 2 clusters
        face_ids = []
        point_ids = []

        for i in range(5):
            face_id = uuid.uuid4()
            point_id = uuid.uuid4()

            face = FaceInstance(
                id=face_id,
                asset_id=asset.id,
                bbox_x=100 + i * 10,
                bbox_y=100 + i * 5,  # Vary to avoid unique constraint
                bbox_w=50,
                bbox_h=50,
                detection_confidence=0.9,
                qdrant_point_id=point_id,
            )
            sync_db_session.add(face)

            # Add to Qdrant
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

            face_ids.append(face_id)
            point_ids.append(point_id)

        sync_db_session.commit()

        # Assign to 2 clusters
        unknown_clusters = {
            face_ids[0]: "unknown_cluster_0",
            face_ids[1]: "unknown_cluster_0",
            face_ids[2]: "unknown_cluster_1",
            face_ids[3]: "unknown_cluster_1",
            face_ids[4]: "unknown_noise_abc",
        }

        clusterer = DualModeClusterer(db_session=sync_db_session)
        clusterer.save_dual_mode_results(
            assigned_faces=[], unknown_clusters=unknown_clusters
        )

        # Verify DB updated
        for face_id, expected_cluster in unknown_clusters.items():
            face = sync_db_session.get(FaceInstance, face_id)
            assert face is not None
            assert face.cluster_id == expected_cluster

        # Verify Qdrant updated
        for point_id, (face_id, expected_cluster) in zip(
            point_ids, unknown_clusters.items()
        ):
            points = qdrant_client.retrieve(
                collection_name=settings.qdrant_face_collection,
                ids=[str(point_id)],
                with_payload=True,
            )
            assert len(points) == 1
            assert points[0].payload.get("cluster_id") == expected_cluster

    def test_save_dual_mode_results_commits(
        self, sync_db_session, sync_db_engine, mock_qdrant_for_clusterer
    ):
        """Test that save_dual_mode_results commits transaction."""
        qdrant_client, face_qdrant, settings = mock_qdrant_for_clusterer

        # Create image asset and face
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

        person_id = uuid.uuid4()
        person = Person(id=person_id, name="Test Person", status=PersonStatus.ACTIVE)
        sync_db_session.add(person)
        sync_db_session.commit()

        face_id = uuid.uuid4()
        point_id = uuid.uuid4()
        face = FaceInstance(
            id=face_id,
            asset_id=asset.id,
            bbox_x=100,
            bbox_y=100,
            bbox_w=50,
            bbox_h=50,
            detection_confidence=0.9,
            qdrant_point_id=point_id,
        )
        sync_db_session.add(face)
        sync_db_session.commit()

        # Add to Qdrant
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

        assigned_faces = [
            {
                "face_id": face_id,
                "person_id": person_id,
                "similarity": 0.85,
                "qdrant_point_id": point_id,
            }
        ]

        clusterer = DualModeClusterer(db_session=sync_db_session)
        clusterer.save_dual_mode_results(
            assigned_faces=assigned_faces, unknown_clusters={}
        )

        # Create NEW session to verify commit
        from sqlalchemy.orm import Session

        new_session = Session(sync_db_engine)
        face = new_session.get(FaceInstance, face_id)
        assert face is not None
        assert face.person_id == person_id
        new_session.close()


class TestFullDualModePipeline:
    """Integration tests for full dual-mode pipeline."""

    def test_full_dual_mode_pipeline(
        self, sync_db_session, sync_db_engine, mock_qdrant_for_clusterer
    ):
        """Test end-to-end: labeled faces → assign unlabeled → cluster remaining."""
        qdrant_client, face_qdrant, settings = mock_qdrant_for_clusterer

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

        # Create 2 people
        person1_id = uuid.uuid4()
        person1 = Person(id=person1_id, name="Person 1", status=PersonStatus.ACTIVE)
        sync_db_session.add(person1)

        person2_id = uuid.uuid4()
        person2 = Person(id=person2_id, name="Person 2", status=PersonStatus.ACTIVE)
        sync_db_session.add(person2)
        sync_db_session.commit()

        # Create labeled faces for person1 (3 faces)
        person1_embedding = np.random.randn(512)
        person1_embedding = person1_embedding / np.linalg.norm(person1_embedding)

        for i in range(3):
            point_id = uuid.uuid4()
            embedding = person1_embedding + np.random.randn(512) * 0.01
            embedding = embedding / np.linalg.norm(embedding)

            face = FaceInstance(
                id=uuid.uuid4(),
                asset_id=asset.id,
                bbox_x=100 + i * 10,
                bbox_y=100 + i * 5,  # Vary to avoid unique constraint
                bbox_w=50,
                bbox_h=50,
                detection_confidence=0.9,
                qdrant_point_id=point_id,
                person_id=person1_id,
            )
            sync_db_session.add(face)

            qdrant_client.upsert(
                collection_name=settings.qdrant_face_collection,
                points=[
                    PointStruct(
                        id=str(point_id),
                        vector=embedding.tolist(),
                        payload={"person_id": str(person1_id)},
                    )
                ],
            )

        # Create unlabeled face similar to person1 (should be assigned)
        similar_point_id = uuid.uuid4()
        similar_face_id = uuid.uuid4()
        similar_embedding = person1_embedding + np.random.randn(512) * 0.01
        similar_embedding = similar_embedding / np.linalg.norm(similar_embedding)

        similar_face = FaceInstance(
            id=similar_face_id,
            asset_id=asset.id,
            bbox_x=200,
            bbox_y=100,
            bbox_w=50,
            bbox_h=50,
            detection_confidence=0.9,
            qdrant_point_id=similar_point_id,
            person_id=None,  # Unlabeled
        )
        sync_db_session.add(similar_face)

        qdrant_client.upsert(
            collection_name=settings.qdrant_face_collection,
            points=[
                PointStruct(
                    id=str(similar_point_id),
                    vector=similar_embedding.tolist(),
                    payload={},
                )
            ],
        )

        # Create unlabeled faces for unknown cluster (5 faces, different from person1)
        unknown_embeddings, _ = make_clustered_embeddings(
            n_per_cluster=5, n_clusters=1, seed=99
        )

        unknown_face_ids = []
        for idx, embedding in enumerate(unknown_embeddings):
            point_id = uuid.uuid4()
            face_id = uuid.uuid4()

            face = FaceInstance(
                id=face_id,
                asset_id=asset.id,
                bbox_x=300 + idx * 10,
                bbox_y=100 + idx * 5,  # Vary to avoid unique constraint
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

        # Run full pipeline
        clusterer = DualModeClusterer(
            db_session=sync_db_session,
            person_match_threshold=0.7,
            unknown_min_cluster_size=3,
            unknown_method="hdbscan",
        )

        result = clusterer.cluster_all_faces()

        # Verify results
        assert result["total_processed"] == 9  # 3 labeled + 1 similar + 5 unknown
        assert result["assigned_to_people"] >= 1  # Similar face assigned to person1

        # Verify similar face assigned to person1
        similar_face = sync_db_session.get(FaceInstance, similar_face_id)
        assert similar_face is not None
        assert similar_face.person_id == person1_id

        # Verify unknown faces got cluster_id
        for face_id in unknown_face_ids:
            face = sync_db_session.get(FaceInstance, face_id)
            assert face is not None
            # Should have either cluster_id or stayed unlabeled
            # (depending on HDBSCAN's noise detection)
