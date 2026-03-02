"""Integration tests for end-to-end workflows.

Tests cover cross-service interactions that exercise the full data path
through multiple layers (DB -> embedding -> Qdrant) without requiring
external services.

Key workflows tested:
1. Ingest-to-Search: Create asset -> index_asset() -> Qdrant upsert -> text search
2. Face Pipeline: Create assets with faces -> assign to persons -> update Qdrant payload
3. Asset Lifecycle: Create -> index -> update -> re-index -> delete -> verify cleanup

Architecture:
- Uses in-memory SQLite for database operations
- Uses in-memory Qdrant client for vector operations
- Mocks embedding service via conftest autouse fixture (SemanticMockEmbeddingService)
- Monkeypatches get_sync_engine() and get_qdrant_client() so that sync job
  functions (index_asset, update_asset_person_ids_job) use test fixtures
  instead of real external services
"""

import uuid
from unittest.mock import patch

import pytest
from qdrant_client import QdrantClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from image_search_service.db.models import (
    Base,
    FaceInstance,
    ImageAsset,
    Person,
    PersonStatus,
    TrainingStatus,
)
from image_search_service.queue.jobs import index_asset, update_asset_person_ids_job
from image_search_service.vector.qdrant import (
    delete_vectors_by_asset,
    search_vectors,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sync_engine():
    """Create a fresh synchronous SQLite engine with all tables.

    Each test gets an isolated in-memory database that is torn down
    after the test completes.
    """
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture
def sync_session(sync_engine):
    """Create a synchronous session bound to the test engine."""
    session = Session(sync_engine)
    yield session
    session.rollback()
    session.close()


@pytest.fixture
def patch_sync_engine(sync_engine):
    """Monkeypatch get_sync_engine to return the test SQLite engine.

    This ensures that sync job functions (index_asset, update_asset_person_ids_job)
    use the test database instead of trying to connect to PostgreSQL.
    """
    with patch(
        "image_search_service.queue.jobs.get_sync_engine",
        return_value=sync_engine,
    ):
        yield sync_engine


@pytest.fixture
def patch_qdrant(qdrant_client: QdrantClient):
    """Monkeypatch get_qdrant_client to return the in-memory test client.

    This ensures that Qdrant operations in jobs.py and qdrant.py module
    functions use the test client instead of connecting to a real server.

    Also clears the _collection_ensured cache so ensure_collection() actually
    creates collections in the test client.
    """
    import image_search_service.vector.qdrant as qdrant_module

    # Clear the collection-ensured cache so ensure_collection works with test client
    original_ensured = qdrant_module._collection_ensured.copy()
    qdrant_module._collection_ensured.clear()

    with patch(
        "image_search_service.vector.qdrant.get_qdrant_client",
        return_value=qdrant_client,
    ):
        yield qdrant_client

    # Restore original cache
    qdrant_module._collection_ensured = original_ensured


# ============================================================================
# Workflow 1: Ingest-to-Search
# ============================================================================


class TestIngestToSearchWorkflow:
    """End-to-end workflow: create asset -> index -> search -> find asset.

    This tests the full data path from database insertion through embedding
    generation to Qdrant vector storage and similarity search retrieval.
    """

    @pytest.mark.asyncio
    async def test_ingest_index_and_search_finds_asset(
        self,
        sync_engine,
        patch_sync_engine,
        patch_qdrant,
        qdrant_client: QdrantClient,
        mock_embedding_service,
    ) -> None:
        """Full ingest-to-search workflow returns the indexed asset.

        Steps:
        1. Insert an ImageAsset into the database
        2. Call index_asset() to generate embedding and upsert to Qdrant
        3. Generate a text query embedding
        4. Search Qdrant and verify the asset appears in results
        """
        # Step 1: Create an asset in the database
        with Session(sync_engine) as session:
            asset = ImageAsset(
                path="/test/images/sunset_beach.jpg",
                training_status=TrainingStatus.PENDING.value,
                width=1920,
                height=1080,
                file_size=2048000,
                mime_type="image/jpeg",
            )
            session.add(asset)
            session.commit()
            session.refresh(asset)
            asset_id = asset.id
            asset_path = asset.path

        # Step 2: Call index_asset() to embed and upsert
        result = index_asset(str(asset_id))

        assert result["status"] == "success"
        assert result["asset_id"] == str(asset_id)

        # Verify indexed_at was set in the database
        with Session(sync_engine) as session:
            refreshed = session.execute(
                select(ImageAsset).where(ImageAsset.id == asset_id)
            ).scalar_one()
            assert refreshed.indexed_at is not None

        # Step 3: Generate a text query embedding for "sunset"
        query_vector = mock_embedding_service.embed_text("sunset beach")

        # Step 4: Search Qdrant and verify asset is found
        results = search_vectors(
            query_vector=query_vector,
            limit=10,
            client=qdrant_client,
        )

        assert len(results) >= 1
        found_asset_ids = [r["asset_id"] for r in results]
        assert str(asset_id) in found_asset_ids

        # Verify payload contains expected data
        matching = [r for r in results if r["asset_id"] == str(asset_id)]
        assert len(matching) == 1
        assert matching[0]["payload"]["path"] == asset_path

    @pytest.mark.asyncio
    async def test_multiple_assets_ranked_by_relevance(
        self,
        sync_engine,
        patch_sync_engine,
        patch_qdrant,
        qdrant_client: QdrantClient,
        mock_embedding_service,
    ) -> None:
        """Multiple indexed assets are ranked by semantic relevance.

        The SemanticMockEmbeddingService clusters "sunset" and "beach" in the
        "nature" concept cluster.  An asset with a nature-related filename
        should score higher than one with an unrelated filename when searching
        for a nature query.
        """
        # Create two assets: one nature-related, one food-related
        with Session(sync_engine) as session:
            nature_asset = ImageAsset(
                path="/test/images/sunset_landscape.jpg",
                training_status=TrainingStatus.PENDING.value,
            )
            food_asset = ImageAsset(
                path="/test/images/pizza_dinner.jpg",
                training_status=TrainingStatus.PENDING.value,
            )
            session.add_all([nature_asset, food_asset])
            session.commit()
            session.refresh(nature_asset)
            session.refresh(food_asset)
            nature_id = nature_asset.id
            food_id = food_asset.id

        # Index both assets
        result1 = index_asset(str(nature_id))
        result2 = index_asset(str(food_id))
        assert result1["status"] == "success"
        assert result2["status"] == "success"

        # Search for nature-related query
        query_vector = mock_embedding_service.embed_text("beautiful sunset")
        results = search_vectors(
            query_vector=query_vector,
            limit=10,
            client=qdrant_client,
        )

        assert len(results) == 2

        # Nature asset should rank higher (closer embedding)
        assert results[0]["asset_id"] == str(nature_id)
        assert results[1]["asset_id"] == str(food_id)

        # Nature asset should have higher similarity score
        assert results[0]["score"] > results[1]["score"]

    @pytest.mark.asyncio
    async def test_index_nonexistent_asset_returns_error(
        self,
        patch_sync_engine,
        patch_qdrant,
    ) -> None:
        """Indexing a non-existent asset returns an error status."""
        result = index_asset("99999")

        assert result["status"] == "error"
        assert "not found" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_index_sets_indexed_at_timestamp(
        self,
        sync_engine,
        patch_sync_engine,
        patch_qdrant,
    ) -> None:
        """index_asset() sets the indexed_at timestamp on the database record."""
        with Session(sync_engine) as session:
            asset = ImageAsset(
                path="/test/images/mountain_view.jpg",
                training_status=TrainingStatus.PENDING.value,
            )
            session.add(asset)
            session.commit()
            session.refresh(asset)
            asset_id = asset.id

            # Verify indexed_at is initially None
            assert asset.indexed_at is None

        # Index the asset
        result = index_asset(str(asset_id))
        assert result["status"] == "success"

        # Verify indexed_at is now set
        with Session(sync_engine) as session:
            refreshed = session.execute(
                select(ImageAsset).where(ImageAsset.id == asset_id)
            ).scalar_one()
            assert refreshed.indexed_at is not None


# ============================================================================
# Workflow 2: Face Pipeline
# ============================================================================


class TestFacePipelineWorkflow:
    """End-to-end workflow: create assets with faces -> assign to persons -> update Qdrant.

    This tests the face-to-person association pipeline and Qdrant payload sync.
    """

    @pytest.mark.asyncio
    async def test_face_assignment_updates_qdrant_payload(
        self,
        sync_engine,
        patch_sync_engine,
        patch_qdrant,
        qdrant_client: QdrantClient,
        mock_embedding_service,
    ) -> None:
        """Assigning faces to persons updates Qdrant vector payload with person_ids.

        Steps:
        1. Create ImageAsset and index it (puts vector in Qdrant)
        2. Create Person and FaceInstance records, assign face to person
        3. Call update_asset_person_ids_job() to sync Qdrant payload
        4. Verify Qdrant payload contains the correct person_ids
        """
        # Step 1: Create and index an asset
        person_id = uuid.uuid4()

        with Session(sync_engine) as session:
            asset = ImageAsset(
                path="/test/images/family_portrait.jpg",
                training_status=TrainingStatus.PENDING.value,
            )
            session.add(asset)
            session.commit()
            session.refresh(asset)
            asset_id = asset.id

        # Index the asset first (creates vector in Qdrant)
        result = index_asset(str(asset_id))
        assert result["status"] == "success"

        # Step 2: Create Person and FaceInstance, assign face to person
        with Session(sync_engine) as session:
            person = Person(
                id=person_id,
                name="Alice",
                status=PersonStatus.ACTIVE.value,
            )
            session.add(person)
            session.flush()

            face = FaceInstance(
                id=uuid.uuid4(),
                asset_id=asset_id,
                bbox_x=100,
                bbox_y=150,
                bbox_w=80,
                bbox_h=80,
                detection_confidence=0.95,
                quality_score=0.85,
                qdrant_point_id=uuid.uuid4(),
                person_id=person_id,
            )
            session.add(face)
            session.commit()

        # Step 3: Call update_asset_person_ids_job() to sync Qdrant
        update_result = update_asset_person_ids_job(asset_id)

        assert update_result["status"] == "success"
        assert update_result["person_count"] == 1

        # Step 4: Verify Qdrant payload contains person_ids
        # Retrieve the point directly from Qdrant
        from image_search_service.core.config import get_settings

        settings = get_settings()
        points = qdrant_client.retrieve(
            collection_name=settings.qdrant_collection,
            ids=[asset_id],
            with_payload=True,
        )

        assert len(points) == 1
        payload = points[0].payload
        assert payload is not None
        assert "person_ids" in payload
        assert str(person_id) in payload["person_ids"]

    @pytest.mark.asyncio
    async def test_multiple_faces_multiple_persons(
        self,
        sync_engine,
        patch_sync_engine,
        patch_qdrant,
        qdrant_client: QdrantClient,
        mock_embedding_service,
    ) -> None:
        """Asset with multiple faces assigned to different persons gets all person_ids.

        Verifies that when an asset has faces belonging to different persons,
        all distinct person_ids appear in the Qdrant payload.
        """
        person_id_1 = uuid.uuid4()
        person_id_2 = uuid.uuid4()

        # Create and index asset
        with Session(sync_engine) as session:
            asset = ImageAsset(
                path="/test/images/group_photo.jpg",
                training_status=TrainingStatus.PENDING.value,
            )
            session.add(asset)
            session.commit()
            session.refresh(asset)
            asset_id = asset.id

        result = index_asset(str(asset_id))
        assert result["status"] == "success"

        # Create two persons and two face instances
        with Session(sync_engine) as session:
            person1 = Person(
                id=person_id_1,
                name="Alice",
                status=PersonStatus.ACTIVE.value,
            )
            person2 = Person(
                id=person_id_2,
                name="Bob",
                status=PersonStatus.ACTIVE.value,
            )
            session.add_all([person1, person2])
            session.flush()

            face1 = FaceInstance(
                id=uuid.uuid4(),
                asset_id=asset_id,
                bbox_x=100,
                bbox_y=100,
                bbox_w=60,
                bbox_h=60,
                detection_confidence=0.92,
                quality_score=0.80,
                qdrant_point_id=uuid.uuid4(),
                person_id=person_id_1,
            )
            face2 = FaceInstance(
                id=uuid.uuid4(),
                asset_id=asset_id,
                bbox_x=300,
                bbox_y=100,
                bbox_w=60,
                bbox_h=60,
                detection_confidence=0.88,
                quality_score=0.75,
                qdrant_point_id=uuid.uuid4(),
                person_id=person_id_2,
            )
            session.add_all([face1, face2])
            session.commit()

        # Update Qdrant payload
        update_result = update_asset_person_ids_job(asset_id)

        assert update_result["status"] == "success"
        assert update_result["person_count"] == 2

        # Verify both person_ids in Qdrant payload
        from image_search_service.core.config import get_settings

        settings = get_settings()
        points = qdrant_client.retrieve(
            collection_name=settings.qdrant_collection,
            ids=[asset_id],
            with_payload=True,
        )

        assert len(points) == 1
        person_ids_in_qdrant = set(points[0].payload["person_ids"])
        assert str(person_id_1) in person_ids_in_qdrant
        assert str(person_id_2) in person_ids_in_qdrant

    @pytest.mark.asyncio
    async def test_unassigned_faces_excluded_from_person_ids(
        self,
        sync_engine,
        patch_sync_engine,
        patch_qdrant,
        qdrant_client: QdrantClient,
        mock_embedding_service,
    ) -> None:
        """Faces without person assignments are excluded from the Qdrant person_ids.

        Only faces with non-NULL person_id should contribute to the payload.
        """
        person_id = uuid.uuid4()

        # Create and index asset
        with Session(sync_engine) as session:
            asset = ImageAsset(
                path="/test/images/crowd_scene.jpg",
                training_status=TrainingStatus.PENDING.value,
            )
            session.add(asset)
            session.commit()
            session.refresh(asset)
            asset_id = asset.id

        index_asset(str(asset_id))

        # Create one assigned face and one unassigned face
        with Session(sync_engine) as session:
            person = Person(
                id=person_id,
                name="Charlie",
                status=PersonStatus.ACTIVE.value,
            )
            session.add(person)
            session.flush()

            assigned_face = FaceInstance(
                id=uuid.uuid4(),
                asset_id=asset_id,
                bbox_x=100,
                bbox_y=100,
                bbox_w=60,
                bbox_h=60,
                detection_confidence=0.95,
                quality_score=0.90,
                qdrant_point_id=uuid.uuid4(),
                person_id=person_id,
            )
            unassigned_face = FaceInstance(
                id=uuid.uuid4(),
                asset_id=asset_id,
                bbox_x=400,
                bbox_y=100,
                bbox_w=50,
                bbox_h=50,
                detection_confidence=0.80,
                quality_score=0.60,
                qdrant_point_id=uuid.uuid4(),
                person_id=None,  # Unassigned
            )
            session.add_all([assigned_face, unassigned_face])
            session.commit()

        # Update Qdrant payload
        update_result = update_asset_person_ids_job(asset_id)

        assert update_result["status"] == "success"
        assert update_result["person_count"] == 1  # Only the assigned face

        # Verify only assigned person in payload
        from image_search_service.core.config import get_settings

        settings = get_settings()
        points = qdrant_client.retrieve(
            collection_name=settings.qdrant_collection,
            ids=[asset_id],
            with_payload=True,
        )

        assert len(points) == 1
        person_ids = points[0].payload["person_ids"]
        assert len(person_ids) == 1
        assert str(person_id) in person_ids

    @pytest.mark.asyncio
    async def test_deleted_asset_skips_person_ids_update(
        self,
        sync_engine,
        patch_sync_engine,
        patch_qdrant,
    ) -> None:
        """update_asset_person_ids_job skips gracefully if asset was deleted."""
        result = update_asset_person_ids_job(99999)

        assert result["status"] == "skipped"
        assert "not found" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_search_by_person_id_filter(
        self,
        sync_engine,
        patch_sync_engine,
        patch_qdrant,
        qdrant_client: QdrantClient,
        mock_embedding_service,
    ) -> None:
        """Searching with person_id filter returns only assets containing that person.

        This verifies the full pipeline: index assets, update person_ids,
        then search with a person_id filter.
        """
        person_id = uuid.uuid4()

        # Create two assets - one with the person, one without
        with Session(sync_engine) as session:
            asset_with_person = ImageAsset(
                path="/test/images/alice_portrait.jpg",
                training_status=TrainingStatus.PENDING.value,
            )
            asset_without_person = ImageAsset(
                path="/test/images/empty_room.jpg",
                training_status=TrainingStatus.PENDING.value,
            )
            session.add_all([asset_with_person, asset_without_person])
            session.commit()
            session.refresh(asset_with_person)
            session.refresh(asset_without_person)
            id_with = asset_with_person.id
            id_without = asset_without_person.id

        # Index both assets
        index_asset(str(id_with))
        index_asset(str(id_without))

        # Assign person to a face on the first asset only
        with Session(sync_engine) as session:
            person = Person(
                id=person_id,
                name="Alice",
                status=PersonStatus.ACTIVE.value,
            )
            session.add(person)
            session.flush()

            face = FaceInstance(
                id=uuid.uuid4(),
                asset_id=id_with,
                bbox_x=100,
                bbox_y=100,
                bbox_w=60,
                bbox_h=60,
                detection_confidence=0.95,
                quality_score=0.85,
                qdrant_point_id=uuid.uuid4(),
                person_id=person_id,
            )
            session.add(face)
            session.commit()

        # Update person_ids for both assets
        update_asset_person_ids_job(id_with)
        update_asset_person_ids_job(id_without)

        # Search with person_id filter
        query_vector = mock_embedding_service.embed_text("portrait")
        results = search_vectors(
            query_vector=query_vector,
            limit=10,
            filters={"person_id": str(person_id)},
            client=qdrant_client,
        )

        # Only the asset with the person should be returned
        found_ids = [r["asset_id"] for r in results]
        assert str(id_with) in found_ids
        assert str(id_without) not in found_ids


# ============================================================================
# Workflow 3: Asset Lifecycle
# ============================================================================


class TestAssetLifecycleWorkflow:
    """End-to-end workflow: create -> index -> update -> re-index -> delete -> verify.

    Tests the complete lifecycle of an asset through database and vector store,
    ensuring consistency at every stage.
    """

    @pytest.mark.asyncio
    async def test_full_asset_lifecycle(
        self,
        sync_engine,
        patch_sync_engine,
        patch_qdrant,
        qdrant_client: QdrantClient,
        mock_embedding_service,
    ) -> None:
        """Complete asset lifecycle: create -> index -> re-index -> delete.

        Steps:
        1. Create asset and verify initial state
        2. Index asset and verify Qdrant vector exists
        3. Update asset path in DB, re-index, verify Qdrant payload updated
        4. Delete vector from Qdrant and verify removal
        """
        from image_search_service.core.config import get_settings

        settings = get_settings()

        # Step 1: Create asset
        with Session(sync_engine) as session:
            asset = ImageAsset(
                path="/test/images/original_photo.jpg",
                training_status=TrainingStatus.PENDING.value,
                width=800,
                height=600,
            )
            session.add(asset)
            session.commit()
            session.refresh(asset)
            asset_id = asset.id

        # Step 2: Index asset
        result = index_asset(str(asset_id))
        assert result["status"] == "success"

        # Verify vector exists in Qdrant
        points = qdrant_client.retrieve(
            collection_name=settings.qdrant_collection,
            ids=[asset_id],
            with_payload=True,
        )
        assert len(points) == 1
        assert points[0].payload["path"] == "/test/images/original_photo.jpg"

        # Step 3: Update asset path and re-index
        with Session(sync_engine) as session:
            asset = session.execute(
                select(ImageAsset).where(ImageAsset.id == asset_id)
            ).scalar_one()
            asset.path = "/test/images/renamed_photo.jpg"
            asset.indexed_at = None  # Reset for re-indexing
            session.commit()

        result = index_asset(str(asset_id))
        assert result["status"] == "success"

        # Verify updated payload in Qdrant
        points = qdrant_client.retrieve(
            collection_name=settings.qdrant_collection,
            ids=[asset_id],
            with_payload=True,
        )
        assert len(points) == 1
        assert points[0].payload["path"] == "/test/images/renamed_photo.jpg"

        # Step 4: Delete vector and verify removal
        deleted = delete_vectors_by_asset(asset_id, client=qdrant_client)
        assert deleted == 1

        # Verify vector is gone
        points = qdrant_client.retrieve(
            collection_name=settings.qdrant_collection,
            ids=[asset_id],
            with_payload=True,
        )
        assert len(points) == 0

    @pytest.mark.asyncio
    async def test_re_index_preserves_person_ids(
        self,
        sync_engine,
        patch_sync_engine,
        patch_qdrant,
        qdrant_client: QdrantClient,
        mock_embedding_service,
    ) -> None:
        """Re-indexing an asset does NOT preserve person_ids (requires separate sync).

        When index_asset() is called, it creates a fresh Qdrant payload from the
        database record. person_ids are managed separately by
        update_asset_person_ids_job(). This test documents that re-indexing
        resets person_ids to empty, and a subsequent person_ids update restores them.
        """
        from image_search_service.core.config import get_settings

        settings = get_settings()
        person_id = uuid.uuid4()

        # Create and index asset
        with Session(sync_engine) as session:
            asset = ImageAsset(
                path="/test/images/person_photo.jpg",
                training_status=TrainingStatus.PENDING.value,
            )
            session.add(asset)
            session.commit()
            session.refresh(asset)
            asset_id = asset.id

        index_asset(str(asset_id))

        # Assign face to person
        with Session(sync_engine) as session:
            person = Person(
                id=person_id,
                name="Diana",
                status=PersonStatus.ACTIVE.value,
            )
            session.add(person)
            session.flush()

            face = FaceInstance(
                id=uuid.uuid4(),
                asset_id=asset_id,
                bbox_x=100,
                bbox_y=100,
                bbox_w=60,
                bbox_h=60,
                detection_confidence=0.95,
                quality_score=0.85,
                qdrant_point_id=uuid.uuid4(),
                person_id=person_id,
            )
            session.add(face)
            session.commit()

        # Update person_ids in Qdrant
        update_asset_person_ids_job(asset_id)

        # Verify person_ids are set
        points = qdrant_client.retrieve(
            collection_name=settings.qdrant_collection,
            ids=[asset_id],
            with_payload=True,
        )
        assert str(person_id) in points[0].payload["person_ids"]

        # Re-index the asset (simulating a re-embedding)
        with Session(sync_engine) as session:
            asset = session.execute(
                select(ImageAsset).where(ImageAsset.id == asset_id)
            ).scalar_one()
            asset.indexed_at = None
            session.commit()

        index_asset(str(asset_id))

        # After re-index, person_ids are reset to empty (index_asset doesn't
        # set person_ids in the payload)
        points = qdrant_client.retrieve(
            collection_name=settings.qdrant_collection,
            ids=[asset_id],
            with_payload=True,
        )
        assert points[0].payload.get("person_ids", []) == []

        # Re-sync person_ids to restore them
        update_asset_person_ids_job(asset_id)

        points = qdrant_client.retrieve(
            collection_name=settings.qdrant_collection,
            ids=[asset_id],
            with_payload=True,
        )
        assert str(person_id) in points[0].payload["person_ids"]

    @pytest.mark.asyncio
    async def test_concurrent_assets_isolated(
        self,
        sync_engine,
        patch_sync_engine,
        patch_qdrant,
        qdrant_client: QdrantClient,
        mock_embedding_service,
    ) -> None:
        """Multiple assets can be independently created, indexed, and searched.

        Verifies that operations on one asset do not interfere with another.
        """
        from image_search_service.core.config import get_settings

        settings = get_settings()

        # Create three independent assets
        asset_ids = []
        with Session(sync_engine) as session:
            for i, name in enumerate(["dog_park", "cat_sleeping", "bird_flying"]):
                asset = ImageAsset(
                    path=f"/test/images/{name}.jpg",
                    training_status=TrainingStatus.PENDING.value,
                )
                session.add(asset)
                session.flush()
                asset_ids.append(asset.id)
            session.commit()

        # Index all three
        for aid in asset_ids:
            result = index_asset(str(aid))
            assert result["status"] == "success"

        # Verify each has its own vector
        for aid in asset_ids:
            points = qdrant_client.retrieve(
                collection_name=settings.qdrant_collection,
                ids=[aid],
                with_payload=True,
            )
            assert len(points) == 1

        # Delete one asset's vector
        delete_vectors_by_asset(asset_ids[1], client=qdrant_client)

        # Verify only that asset's vector is gone
        for i, aid in enumerate(asset_ids):
            points = qdrant_client.retrieve(
                collection_name=settings.qdrant_collection,
                ids=[aid],
                with_payload=True,
            )
            if i == 1:
                assert len(points) == 0, "Deleted asset should have no vector"
            else:
                assert len(points) == 1, "Other assets should still have vectors"

    @pytest.mark.asyncio
    async def test_index_idempotent(
        self,
        sync_engine,
        patch_sync_engine,
        patch_qdrant,
        qdrant_client: QdrantClient,
        mock_embedding_service,
    ) -> None:
        """Calling index_asset() multiple times produces the same result.

        The upsert operation is idempotent: re-indexing should simply
        overwrite the existing vector and payload.
        """
        from image_search_service.core.config import get_settings

        settings = get_settings()

        with Session(sync_engine) as session:
            asset = ImageAsset(
                path="/test/images/stable_photo.jpg",
                training_status=TrainingStatus.PENDING.value,
            )
            session.add(asset)
            session.commit()
            session.refresh(asset)
            asset_id = asset.id

        # Index twice
        result1 = index_asset(str(asset_id))
        result2 = index_asset(str(asset_id))

        assert result1["status"] == "success"
        assert result2["status"] == "success"

        # Should still have exactly one point
        points = qdrant_client.retrieve(
            collection_name=settings.qdrant_collection,
            ids=[asset_id],
            with_payload=True,
        )
        assert len(points) == 1

        # Search should return it exactly once
        query_vector = mock_embedding_service.embed_text("stable photo")
        results = search_vectors(
            query_vector=query_vector,
            limit=10,
            client=qdrant_client,
        )

        matching = [r for r in results if r["asset_id"] == str(asset_id)]
        assert len(matching) == 1
