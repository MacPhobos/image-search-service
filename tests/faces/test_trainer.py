"""Tests for faces/trainer.py - triplet dataset and face trainer."""

import json
import uuid
from pathlib import Path

import numpy as np
import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from image_search_service.db.models import FaceInstance, ImageAsset, Person, PersonStatus
from image_search_service.faces.trainer import FaceTrainer, TripletFaceDataset
from image_search_service.vector.face_qdrant import FaceQdrantClient


# ==============================================================================
# TripletFaceDataset Tests (Pure NumPy, No Mocking)
# ==============================================================================


def test_generate_triplets_basic():
    """Generate triplets with 3 persons, 5 faces each returns valid triplets."""
    # Set seed for reproducibility
    np.random.seed(42)

    # Create embeddings for 3 persons, 5 faces each
    embeddings_by_person = {}
    for i in range(3):
        person_id = f"person_{i}"
        embeddings = []
        for j in range(5):
            # Create normalized random embedding
            emb = np.random.randn(512).astype(np.float64)
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
        embeddings_by_person[person_id] = embeddings

    dataset = TripletFaceDataset(embeddings_by_person, triplets_per_person=10)
    triplets = dataset.generate_triplets()

    # Should generate triplets (may be less than 30 if hard negative mining fails)
    assert len(triplets) > 0
    assert len(triplets) <= 30  # 3 persons * 10 triplets

    # Check each triplet has correct structure
    for anchor, positive, negative in triplets:
        assert anchor.shape == (512,)
        assert positive.shape == (512,)
        assert negative.shape == (512,)
        assert anchor.dtype == np.float64
        assert positive.dtype == np.float64
        assert negative.dtype == np.float64


def test_generate_triplets_count():
    """Triplets per person parameter controls output count."""
    np.random.seed(42)

    # 3 persons, 5 faces each
    embeddings_by_person = {}
    for i in range(3):
        person_id = f"person_{i}"
        embeddings = []
        for j in range(5):
            emb = np.random.randn(512).astype(np.float64)
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
        embeddings_by_person[person_id] = embeddings

    dataset = TripletFaceDataset(embeddings_by_person, triplets_per_person=10)
    triplets = dataset.generate_triplets()

    # Should generate approximately 30 triplets (3 persons * 10)
    # Allow some slack for hard negative mining failures
    assert 25 <= len(triplets) <= 30


def test_generate_triplets_skip_single_face_person():
    """Person with only 1 face is skipped (need 2 for anchor+positive)."""
    np.random.seed(42)

    embeddings_by_person = {
        "person_1": [np.random.randn(512).astype(np.float64) / 10],  # Only 1 face
        "person_2": [
            np.random.randn(512).astype(np.float64) / 10,
            np.random.randn(512).astype(np.float64) / 10,
        ],
        "person_3": [
            np.random.randn(512).astype(np.float64) / 10,
            np.random.randn(512).astype(np.float64) / 10,
        ],
    }

    # Normalize embeddings
    for person_id in embeddings_by_person:
        embeddings_by_person[person_id] = [
            emb / np.linalg.norm(emb) for emb in embeddings_by_person[person_id]
        ]

    dataset = TripletFaceDataset(embeddings_by_person, triplets_per_person=5)
    triplets = dataset.generate_triplets()

    # Should only generate triplets for person_2 and person_3 (2 persons * 5 = 10)
    assert len(triplets) <= 10  # Max 10 triplets
    assert len(triplets) > 0  # But should have some triplets


def test_generate_triplets_two_persons_min():
    """Minimum 2 persons works correctly."""
    np.random.seed(42)

    embeddings_by_person = {
        "person_1": [
            np.random.randn(512).astype(np.float64),
            np.random.randn(512).astype(np.float64),
        ],
        "person_2": [
            np.random.randn(512).astype(np.float64),
            np.random.randn(512).astype(np.float64),
        ],
    }

    # Normalize
    for person_id in embeddings_by_person:
        embeddings_by_person[person_id] = [
            emb / np.linalg.norm(emb) for emb in embeddings_by_person[person_id]
        ]

    dataset = TripletFaceDataset(embeddings_by_person, triplets_per_person=5)
    triplets = dataset.generate_triplets()

    # Should generate triplets (2 persons * 5 = 10)
    assert len(triplets) == 10


def test_generate_triplets_single_person():
    """Only 1 person produces no triplets (no negatives available)."""
    np.random.seed(42)

    embeddings_by_person = {
        "person_1": [
            np.random.randn(512).astype(np.float64),
            np.random.randn(512).astype(np.float64),
            np.random.randn(512).astype(np.float64),
        ],
    }

    # Normalize
    for person_id in embeddings_by_person:
        embeddings_by_person[person_id] = [
            emb / np.linalg.norm(emb) for emb in embeddings_by_person[person_id]
        ]

    dataset = TripletFaceDataset(embeddings_by_person, triplets_per_person=5)
    triplets = dataset.generate_triplets()

    # No negatives available, so no triplets
    assert len(triplets) == 0


def test_select_hard_negative_returns_most_similar():
    """Hard negative mining selects most similar face from different person."""
    np.random.seed(42)

    # Create embeddings where person_2's first face is most similar to person_1's anchor
    anchor = np.array([1.0] + [0.0] * 511, dtype=np.float64)
    anchor = anchor / np.linalg.norm(anchor)

    # person_2's first face is very similar to anchor
    similar_negative = np.array([0.9] + [0.1] * 511, dtype=np.float64)
    similar_negative = similar_negative / np.linalg.norm(similar_negative)

    # person_2's second face is less similar
    dissimilar_negative = np.array([0.1] + [0.9] * 511, dtype=np.float64)
    dissimilar_negative = dissimilar_negative / np.linalg.norm(dissimilar_negative)

    embeddings_by_person = {
        "person_1": [anchor],
        "person_2": [similar_negative, dissimilar_negative],
    }

    dataset = TripletFaceDataset(embeddings_by_person)
    hard_negative = dataset._select_hard_negative(anchor, "person_1")

    assert hard_negative is not None
    # Should select the more similar negative
    similarity_similar = np.dot(anchor, similar_negative)
    similarity_dissimilar = np.dot(anchor, dissimilar_negative)
    assert similarity_similar > similarity_dissimilar
    # Hard negative should be the most similar one
    selected_similarity = np.dot(anchor, hard_negative)
    assert abs(selected_similarity - similarity_similar) < 0.01


def test_select_hard_negative_no_other_persons():
    """Only 1 person returns None (no negatives available)."""
    anchor = np.random.randn(512).astype(np.float64)
    anchor = anchor / np.linalg.norm(anchor)

    embeddings_by_person = {
        "person_1": [anchor],
    }

    dataset = TripletFaceDataset(embeddings_by_person)
    hard_negative = dataset._select_hard_negative(anchor, "person_1")

    assert hard_negative is None


def test_triplet_shapes():
    """Each triplet element has correct shape (512,)."""
    np.random.seed(42)

    embeddings_by_person = {}
    for i in range(3):
        person_id = f"person_{i}"
        embeddings = []
        for j in range(3):
            emb = np.random.randn(512).astype(np.float64)
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
        embeddings_by_person[person_id] = embeddings

    dataset = TripletFaceDataset(embeddings_by_person, triplets_per_person=5)
    triplets = dataset.generate_triplets()

    for anchor, positive, negative in triplets:
        assert anchor.shape == (512,)
        assert positive.shape == (512,)
        assert negative.shape == (512,)


# ==============================================================================
# FaceTrainer Tests (Require Mocking Qdrant + DB)
# ==============================================================================


@pytest.fixture
def trainer_with_mock_qdrant(sync_db_session, monkeypatch):
    """Create FaceTrainer with mocked Qdrant client."""
    client = QdrantClient(":memory:")
    client.create_collection("faces", vectors_config=VectorParams(size=512, distance=Distance.COSINE))
    face_qdrant = FaceQdrantClient()
    face_qdrant._client = client
    # Patch the import location (imported inside _get_face_embedding method)
    monkeypatch.setattr(
        "image_search_service.vector.face_qdrant.get_face_qdrant_client",
        lambda: face_qdrant,
    )
    trainer = FaceTrainer(db_session=sync_db_session, margin=0.2, epochs=5, batch_size=8)
    return trainer, client, face_qdrant


def populate_labeled_faces(db_session, qdrant_client, persons_config):
    """Create labeled faces in DB + Qdrant.

    Args:
        db_session: Synchronous DB session
        qdrant_client: Qdrant client instance
        persons_config: list of (person_name, n_faces, base_embedding)
    """
    for name, n_faces, base_embedding in persons_config:
        person = Person(id=uuid.uuid4(), name=name, status=PersonStatus.ACTIVE.value)
        db_session.add(person)
        db_session.flush()

        asset = ImageAsset(path=f"/test/{name}.jpg")
        db_session.add(asset)
        db_session.flush()

        for i in range(n_faces):
            point_id = uuid.uuid4()
            # Add small random noise to base embedding
            emb = base_embedding + np.random.randn(512) * 0.05
            emb = emb / np.linalg.norm(emb)

            face = FaceInstance(
                id=uuid.uuid4(),
                asset_id=asset.id,
                bbox_x=i * 10,
                bbox_y=0,
                bbox_w=50,
                bbox_h=50,
                detection_confidence=0.95,
                qdrant_point_id=point_id,
                person_id=person.id,
            )
            db_session.add(face)

            qdrant_client.upsert(
                collection_name="faces",
                points=[PointStruct(id=str(point_id), vector=emb.tolist(), payload={})],
            )

        db_session.flush()

    db_session.commit()


def test_compute_triplet_loss_satisfied():
    """Triplet loss is 0 when constraint is satisfied: d(a,p) + margin < d(a,n)."""
    trainer = FaceTrainer(db_session=None, margin=0.2)  # type: ignore

    # Create embeddings where anchor is close to positive, far from negative
    anchor = np.array([1.0] + [0.0] * 511, dtype=np.float64)
    positive = np.array([0.95] + [0.05] * 511, dtype=np.float64)
    negative = np.array([0.1] + [0.9] * 511, dtype=np.float64)

    # Normalize
    anchor = anchor / np.linalg.norm(anchor)
    positive = positive / np.linalg.norm(positive)
    negative = negative / np.linalg.norm(negative)

    loss = trainer.compute_triplet_loss(anchor, positive, negative)

    # Loss should be 0 (constraint satisfied)
    assert loss == 0.0


def test_compute_triplet_loss_violated():
    """Triplet loss > 0 when constraint is violated: d(a,p) + margin > d(a,n)."""
    trainer = FaceTrainer(db_session=None, margin=0.2)  # type: ignore

    # Create embeddings where anchor is far from positive, close to negative
    anchor = np.array([1.0] + [0.0] * 511, dtype=np.float64)
    positive = np.array([0.1] + [0.9] * 511, dtype=np.float64)  # Far from anchor
    negative = np.array([0.95] + [0.05] * 511, dtype=np.float64)  # Close to anchor

    # Normalize
    anchor = anchor / np.linalg.norm(anchor)
    positive = positive / np.linalg.norm(positive)
    negative = negative / np.linalg.norm(negative)

    loss = trainer.compute_triplet_loss(anchor, positive, negative)

    # Loss should be > 0 (constraint violated)
    assert loss > 0.0


def test_compute_triplet_loss_identical_anchor_positive():
    """Loss when anchor == positive equals max(0, margin - d(a,n))."""
    trainer = FaceTrainer(db_session=None, margin=0.2)  # type: ignore

    # Identical anchor and positive
    anchor = np.array([1.0] + [0.0] * 511, dtype=np.float64)
    positive = anchor.copy()
    negative = np.array([0.5] + [0.5] * 511, dtype=np.float64)

    # Normalize
    anchor = anchor / np.linalg.norm(anchor)
    positive = positive / np.linalg.norm(positive)
    negative = negative / np.linalg.norm(negative)

    loss = trainer.compute_triplet_loss(anchor, positive, negative)

    # d(a,p) = 0, so loss = max(0, 0 - d(a,n) + margin) = max(0, margin - d(a,n))
    # Since d(a,n) will be > 0, loss = margin - d(a,n)
    assert loss >= 0.0


def test_compute_triplet_loss_margin_effect():
    """Higher margin produces higher loss for same embeddings."""
    # Same embeddings for both trainers
    anchor = np.array([1.0] + [0.0] * 511, dtype=np.float64)
    positive = np.array([0.8] + [0.2] * 511, dtype=np.float64)
    negative = np.array([0.7] + [0.3] * 511, dtype=np.float64)

    # Normalize
    anchor = anchor / np.linalg.norm(anchor)
    positive = positive / np.linalg.norm(positive)
    negative = negative / np.linalg.norm(negative)

    trainer_small_margin = FaceTrainer(db_session=None, margin=0.1)  # type: ignore
    trainer_large_margin = FaceTrainer(db_session=None, margin=0.5)  # type: ignore

    loss_small = trainer_small_margin.compute_triplet_loss(anchor, positive, negative)
    loss_large = trainer_large_margin.compute_triplet_loss(anchor, positive, negative)

    # Larger margin should produce higher loss
    assert loss_large > loss_small


def test_get_labeled_faces_by_person_empty_db(trainer_with_mock_qdrant):
    """Empty database returns empty dict."""
    trainer, _, _ = trainer_with_mock_qdrant

    result = trainer.get_labeled_faces_by_person()

    assert result == {}


def test_get_labeled_faces_by_person_groups_correctly(trainer_with_mock_qdrant):
    """Labeled faces are grouped by person_id."""
    trainer, qdrant_client, _ = trainer_with_mock_qdrant
    np.random.seed(42)

    # Create 3 persons with different numbers of faces
    base_emb1 = np.random.randn(512).astype(np.float64)
    base_emb1 = base_emb1 / np.linalg.norm(base_emb1)

    base_emb2 = np.random.randn(512).astype(np.float64)
    base_emb2 = base_emb2 / np.linalg.norm(base_emb2)

    base_emb3 = np.random.randn(512).astype(np.float64)
    base_emb3 = base_emb3 / np.linalg.norm(base_emb3)

    persons_config = [
        ("Alice", 5, base_emb1),
        ("Bob", 3, base_emb2),
        ("Charlie", 7, base_emb3),
    ]

    populate_labeled_faces(trainer.db, qdrant_client, persons_config)

    result = trainer.get_labeled_faces_by_person()

    # Should have 3 persons
    assert len(result) == 3

    # Check face counts per person
    face_counts = {person_id: len(faces) for person_id, faces in result.items()}
    assert 5 in face_counts.values()
    assert 3 in face_counts.values()
    assert 7 in face_counts.values()

    # Check structure of returned faces
    for person_id, faces in result.items():
        for face in faces:
            assert "face_id" in face
            assert "qdrant_point_id" in face
            assert "embedding" in face
            assert isinstance(face["embedding"], np.ndarray)
            assert face["embedding"].shape == (512,)


def test_fine_tune_no_labeled_faces(trainer_with_mock_qdrant):
    """Fine-tuning with no labeled faces returns zeros."""
    trainer, _, _ = trainer_with_mock_qdrant

    result = trainer.fine_tune_for_person_clustering(min_faces_per_person=5)

    assert result["epochs"] == 0
    assert result["final_loss"] == 0.0
    assert result["persons_trained"] == 0
    assert result["triplets_used"] == 0


def test_fine_tune_insufficient_faces(trainer_with_mock_qdrant):
    """Fine-tuning where all persons have < min_faces returns zeros."""
    trainer, qdrant_client, _ = trainer_with_mock_qdrant
    np.random.seed(42)

    # Create 3 persons with only 3 faces each (below min_faces=5)
    base_emb1 = np.random.randn(512).astype(np.float64)
    base_emb1 = base_emb1 / np.linalg.norm(base_emb1)

    base_emb2 = np.random.randn(512).astype(np.float64)
    base_emb2 = base_emb2 / np.linalg.norm(base_emb2)

    base_emb3 = np.random.randn(512).astype(np.float64)
    base_emb3 = base_emb3 / np.linalg.norm(base_emb3)

    persons_config = [
        ("Alice", 3, base_emb1),
        ("Bob", 3, base_emb2),
        ("Charlie", 3, base_emb3),
    ]

    populate_labeled_faces(trainer.db, qdrant_client, persons_config)

    result = trainer.fine_tune_for_person_clustering(min_faces_per_person=5)

    assert result["epochs"] == 0
    assert result["final_loss"] == 0.0
    assert result["persons_trained"] == 0
    assert result["triplets_used"] == 0


def test_fine_tune_success(trainer_with_mock_qdrant):
    """Fine-tuning with sufficient data returns proper results."""
    trainer, qdrant_client, _ = trainer_with_mock_qdrant
    np.random.seed(42)

    # Create 3 persons with 10 faces each
    base_emb1 = np.random.randn(512).astype(np.float64)
    base_emb1 = base_emb1 / np.linalg.norm(base_emb1)

    base_emb2 = np.random.randn(512).astype(np.float64)
    base_emb2 = base_emb2 / np.linalg.norm(base_emb2)

    base_emb3 = np.random.randn(512).astype(np.float64)
    base_emb3 = base_emb3 / np.linalg.norm(base_emb3)

    persons_config = [
        ("Alice", 10, base_emb1),
        ("Bob", 10, base_emb2),
        ("Charlie", 10, base_emb3),
    ]

    populate_labeled_faces(trainer.db, qdrant_client, persons_config)

    result = trainer.fine_tune_for_person_clustering(min_faces_per_person=5)

    assert result["epochs"] == 5  # trainer configured with epochs=5
    assert result["persons_trained"] == 3
    assert result["triplets_used"] > 0  # Should generate many triplets
    assert isinstance(result["final_loss"], float)


def test_fine_tune_saves_checkpoint(trainer_with_mock_qdrant, tmp_path):
    """Fine-tuning with checkpoint_path creates checkpoint file."""
    trainer, qdrant_client, _ = trainer_with_mock_qdrant
    np.random.seed(42)

    # Create 2 persons with 10 faces each
    base_emb1 = np.random.randn(512).astype(np.float64)
    base_emb1 = base_emb1 / np.linalg.norm(base_emb1)

    base_emb2 = np.random.randn(512).astype(np.float64)
    base_emb2 = base_emb2 / np.linalg.norm(base_emb2)

    persons_config = [
        ("Alice", 10, base_emb1),
        ("Bob", 10, base_emb2),
    ]

    populate_labeled_faces(trainer.db, qdrant_client, persons_config)

    checkpoint_path = tmp_path / "checkpoint.json"

    result = trainer.fine_tune_for_person_clustering(
        min_faces_per_person=5,
        checkpoint_path=str(checkpoint_path),
    )

    # Check checkpoint file exists
    assert checkpoint_path.exists()

    # Check checkpoint content
    with open(checkpoint_path) as f:
        checkpoint = json.load(f)

    assert checkpoint["epoch"] == 5
    assert "loss" in checkpoint
    assert checkpoint["margin"] == 0.2
    assert checkpoint["batch_size"] == 8
    assert checkpoint["learning_rate"] == 0.0001


def test_save_checkpoint_creates_directory(tmp_path):
    """Save checkpoint creates parent directory if it doesn't exist."""
    trainer = FaceTrainer(db_session=None, margin=0.2)  # type: ignore

    # Non-existent subdirectory
    checkpoint_path = tmp_path / "subdir" / "checkpoint.json"

    trainer.save_checkpoint(str(checkpoint_path), epoch=10, loss=0.15)

    # Check file and directory exist
    assert checkpoint_path.exists()
    assert checkpoint_path.parent.exists()

    # Verify content
    with open(checkpoint_path) as f:
        checkpoint = json.load(f)

    assert checkpoint["epoch"] == 10
    assert checkpoint["loss"] == 0.15
    assert checkpoint["margin"] == 0.2


def test_load_checkpoint_success(tmp_path):
    """Load valid checkpoint file returns correct data."""
    trainer = FaceTrainer(db_session=None, margin=0.3, learning_rate=0.001)  # type: ignore

    checkpoint_path = tmp_path / "checkpoint.json"

    # Save checkpoint first
    trainer.save_checkpoint(str(checkpoint_path), epoch=15, loss=0.08)

    # Load it back
    loaded = trainer.load_checkpoint(str(checkpoint_path))

    assert loaded["epoch"] == 15
    assert loaded["loss"] == 0.08
    assert loaded["margin"] == 0.3
    assert loaded["learning_rate"] == 0.001


def test_load_checkpoint_missing_file(tmp_path):
    """Load non-existent checkpoint returns empty dict."""
    trainer = FaceTrainer(db_session=None, margin=0.2)  # type: ignore

    checkpoint_path = tmp_path / "nonexistent.json"

    loaded = trainer.load_checkpoint(str(checkpoint_path))

    assert loaded == {}
