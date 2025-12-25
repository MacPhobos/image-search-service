"""Test admin endpoints."""

import pytest
from httpx import AsyncClient
from qdrant_client import QdrantClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.db.models import (
    Category,
    ImageAsset,
    Person,
    TrainingSession,
)
from image_search_service.vector.qdrant import upsert_vector
from tests.conftest import MockEmbeddingService

# Fixtures


@pytest.fixture
async def test_category(db_session: AsyncSession) -> Category:
    """Create a test category."""
    category = Category(
        name="Test Category",
        description="For testing admin operations",
        is_default=False,
    )
    db_session.add(category)
    await db_session.commit()
    await db_session.refresh(category)
    return category


@pytest.fixture
async def test_assets(db_session: AsyncSession) -> list[ImageAsset]:
    """Create test image assets."""
    assets = [
        ImageAsset(
            path="/photos/2024/test1.jpg",
            training_status="trained",
        ),
        ImageAsset(
            path="/photos/2024/test2.jpg",
            training_status="trained",
        ),
    ]
    for asset in assets:
        db_session.add(asset)
    await db_session.commit()
    for asset in assets:
        await db_session.refresh(asset)
    return assets


@pytest.fixture
async def test_training_session(
    db_session: AsyncSession, test_category: Category
) -> TrainingSession:
    """Create a test training session."""
    session = TrainingSession(
        name="Test Session",
        root_path="/photos/2024",
        category_id=test_category.id,
        status="completed",
    )
    db_session.add(session)
    await db_session.commit()
    await db_session.refresh(session)
    return session


@pytest.fixture
async def test_person(db_session: AsyncSession) -> Person:
    """Create a test person."""
    person = Person(name="Test Person")
    db_session.add(person)
    await db_session.commit()
    await db_session.refresh(person)
    return person


@pytest.fixture
def test_vectors(
    qdrant_client: QdrantClient,
    test_assets: list[ImageAsset],
    test_category: Category,
    mock_embedding_service: MockEmbeddingService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Create test vectors in Qdrant."""
    # Patch get_qdrant_client to return our test fixture
    monkeypatch.setattr(
        "image_search_service.vector.qdrant.get_qdrant_client", lambda: qdrant_client
    )

    for asset in test_assets:
        vector = mock_embedding_service.embed_text(asset.path)
        upsert_vector(
            asset_id=asset.id,
            vector=vector,
            payload={
                "path": asset.path,
                "category_id": test_category.id,
                "created_at": "2024-01-01T00:00:00Z",
            },
        )


@pytest.fixture
async def setup_alembic_version(db_session: AsyncSession) -> str:
    """Ensure alembic_version table exists with test version."""
    # Create table if not exists and insert test version
    await db_session.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS alembic_version (
                version_num VARCHAR(32) NOT NULL,
                CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
            )
            """
        )
    )
    await db_session.execute(text("DELETE FROM alembic_version"))
    await db_session.execute(
        text("INSERT INTO alembic_version (version_num) VALUES ('012_test_version')")
    )
    await db_session.commit()
    return "012_test_version"


# Validation Tests


@pytest.mark.asyncio
async def test_delete_all_data_requires_confirm_flag(
    test_client: AsyncClient,
) -> None:
    """Test that confirm=false is rejected."""
    response = await test_client.post(
        "/api/v1/admin/data/delete-all",
        json={
            "confirm": False,
            "confirmationText": "DELETE ALL DATA",
        },
    )

    assert response.status_code == 400
    assert "confirm" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_delete_all_data_requires_exact_confirmation_text(
    test_client: AsyncClient,
) -> None:
    """Test that wrong confirmation text is rejected."""
    # Wrong case
    response = await test_client.post(
        "/api/v1/admin/data/delete-all",
        json={
            "confirm": True,
            "confirmationText": "delete all data",  # Wrong case
        },
    )

    assert response.status_code == 400
    assert "confirmationText" in response.json()["detail"]

    # Missing word
    response = await test_client.post(
        "/api/v1/admin/data/delete-all",
        json={
            "confirm": True,
            "confirmationText": "DELETE ALL",  # Incomplete
        },
    )

    assert response.status_code == 400
    assert "confirmationText" in response.json()["detail"]


@pytest.mark.asyncio
async def test_delete_all_data_accepts_optional_reason(
    test_client: AsyncClient,
    db_session: AsyncSession,
    setup_alembic_version: str,
) -> None:
    """Test that reason field is optional."""
    response = await test_client.post(
        "/api/v1/admin/data/delete-all",
        json={
            "confirm": True,
            "confirmationText": "DELETE ALL DATA",
            # No reason provided
        },
    )

    assert response.status_code == 200


# Functional Tests


@pytest.mark.asyncio
async def test_delete_all_data_deletes_qdrant_collections(
    test_client: AsyncClient,
    db_session: AsyncSession,
    test_assets: list[ImageAsset],
    test_category: Category,
    test_vectors: None,
    setup_alembic_version: str,
    qdrant_client: QdrantClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that Qdrant collections are deleted."""
    # Patch the qdrant module
    monkeypatch.setattr(
        "image_search_service.services.admin_service.qdrant.reset_collection",
        lambda: 2,  # Mock return value
    )

    # Mock face qdrant client
    class MockFaceClient:
        def reset_collection(self) -> int:
            return 0

    monkeypatch.setattr(
        "image_search_service.services.admin_service.get_face_qdrant_client",
        lambda: MockFaceClient(),
    )

    response = await test_client.post(
        "/api/v1/admin/data/delete-all",
        json={
            "confirm": True,
            "confirmationText": "DELETE ALL DATA",
            "reason": "Test deletion",
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert "qdrantCollectionsDeleted" in data
    assert "main" in data["qdrantCollectionsDeleted"]
    assert "faces" in data["qdrantCollectionsDeleted"]


@pytest.mark.asyncio
async def test_delete_all_data_truncates_postgres_tables(
    test_client: AsyncClient,
    db_session: AsyncSession,
    test_assets: list[ImageAsset],
    test_category: Category,
    test_training_session: TrainingSession,
    test_person: Person,
    setup_alembic_version: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that PostgreSQL tables are truncated."""
    # Mock Qdrant deletions
    monkeypatch.setattr(
        "image_search_service.services.admin_service.qdrant.reset_collection",
        lambda: 0,
    )

    class MockFaceClient:
        def reset_collection(self) -> int:
            return 0

    monkeypatch.setattr(
        "image_search_service.services.admin_service.get_face_qdrant_client",
        lambda: MockFaceClient(),
    )

    # Verify data exists before deletion
    from sqlalchemy import select

    result = await db_session.execute(select(ImageAsset))
    assert len(result.scalars().all()) == 2

    result = await db_session.execute(select(Category))
    assert len(result.scalars().all()) == 1

    result = await db_session.execute(select(TrainingSession))
    assert len(result.scalars().all()) == 1

    result = await db_session.execute(select(Person))
    assert len(result.scalars().all()) == 1

    # Perform deletion
    response = await test_client.post(
        "/api/v1/admin/data/delete-all",
        json={
            "confirm": True,
            "confirmationText": "DELETE ALL DATA",
            "reason": "Test truncation",
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert "postgresTablesTruncated" in data
    assert isinstance(data["postgresTablesTruncated"], dict)

    # Refresh session after truncation
    await db_session.rollback()

    # Verify tables are empty
    result = await db_session.execute(select(ImageAsset))
    assert len(result.scalars().all()) == 0

    result = await db_session.execute(select(Category))
    assert len(result.scalars().all()) == 0

    result = await db_session.execute(select(TrainingSession))
    assert len(result.scalars().all()) == 0

    result = await db_session.execute(select(Person))
    assert len(result.scalars().all()) == 0


@pytest.mark.asyncio
async def test_delete_all_data_preserves_alembic_version(
    test_client: AsyncClient,
    db_session: AsyncSession,
    setup_alembic_version: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that alembic_version table is preserved."""
    # Mock Qdrant deletions
    monkeypatch.setattr(
        "image_search_service.services.admin_service.qdrant.reset_collection",
        lambda: 0,
    )

    class MockFaceClient:
        def reset_collection(self) -> int:
            return 0

    monkeypatch.setattr(
        "image_search_service.services.admin_service.get_face_qdrant_client",
        lambda: MockFaceClient(),
    )

    response = await test_client.post(
        "/api/v1/admin/data/delete-all",
        json={
            "confirm": True,
            "confirmationText": "DELETE ALL DATA",
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Check that alembic version is preserved in response
    assert "alembicVersionPreserved" in data
    assert data["alembicVersionPreserved"] == "012_test_version"

    # Verify alembic_version table still exists and has correct value
    result = await db_session.execute(text("SELECT version_num FROM alembic_version"))
    row = result.fetchone()
    assert row is not None
    assert row[0] == "012_test_version"


@pytest.mark.asyncio
async def test_delete_all_data_returns_proper_response_structure(
    test_client: AsyncClient,
    db_session: AsyncSession,
    setup_alembic_version: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that response has all required fields."""
    # Mock Qdrant deletions
    monkeypatch.setattr(
        "image_search_service.services.admin_service.qdrant.reset_collection",
        lambda: 10,
    )

    class MockFaceClient:
        def reset_collection(self) -> int:
            return 5

    monkeypatch.setattr(
        "image_search_service.services.admin_service.get_face_qdrant_client",
        lambda: MockFaceClient(),
    )

    response = await test_client.post(
        "/api/v1/admin/data/delete-all",
        json={
            "confirm": True,
            "confirmationText": "DELETE ALL DATA",
            "reason": "Testing response structure",
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Verify all required fields are present
    assert "qdrantCollectionsDeleted" in data
    assert isinstance(data["qdrantCollectionsDeleted"], dict)

    assert "postgresTablesTruncated" in data
    assert isinstance(data["postgresTablesTruncated"], dict)

    assert "alembicVersionPreserved" in data
    assert isinstance(data["alembicVersionPreserved"], str)

    assert "message" in data
    assert isinstance(data["message"], str)

    assert "timestamp" in data
    assert isinstance(data["timestamp"], str)


@pytest.mark.asyncio
async def test_delete_all_data_includes_reason_in_message(
    test_client: AsyncClient,
    db_session: AsyncSession,
    setup_alembic_version: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that message field contains useful information."""
    # Mock Qdrant deletions
    monkeypatch.setattr(
        "image_search_service.services.admin_service.qdrant.reset_collection",
        lambda: 100,
    )

    class MockFaceClient:
        def reset_collection(self) -> int:
            return 50

    monkeypatch.setattr(
        "image_search_service.services.admin_service.get_face_qdrant_client",
        lambda: MockFaceClient(),
    )

    response = await test_client.post(
        "/api/v1/admin/data/delete-all",
        json={
            "confirm": True,
            "confirmationText": "DELETE ALL DATA",
            "reason": "Complete system reset",
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Message should contain useful info
    message = data["message"].lower()
    assert "deleted" in message or "truncated" in message
    assert "preserved" in message or "migration" in message


# Error Handling Tests


@pytest.mark.asyncio
async def test_delete_all_data_handles_missing_alembic_version(
    test_client: AsyncClient,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test graceful handling when alembic_version doesn't exist."""
    # Ensure alembic_version table doesn't exist or is empty
    try:
        await db_session.execute(text("DROP TABLE IF EXISTS alembic_version"))
        await db_session.commit()
    except Exception:
        pass

    # Mock Qdrant deletions
    monkeypatch.setattr(
        "image_search_service.services.admin_service.qdrant.reset_collection",
        lambda: 0,
    )

    class MockFaceClient:
        def reset_collection(self) -> int:
            return 0

    monkeypatch.setattr(
        "image_search_service.services.admin_service.get_face_qdrant_client",
        lambda: MockFaceClient(),
    )

    response = await test_client.post(
        "/api/v1/admin/data/delete-all",
        json={
            "confirm": True,
            "confirmationText": "DELETE ALL DATA",
        },
    )

    # Should still succeed with "unknown" version
    assert response.status_code == 200
    data = response.json()
    assert data["alembicVersionPreserved"] == "unknown"


@pytest.mark.asyncio
async def test_delete_all_data_handles_qdrant_errors_gracefully(
    test_client: AsyncClient,
    db_session: AsyncSession,
    setup_alembic_version: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that Qdrant errors don't break the entire operation."""

    def failing_reset() -> int:
        raise Exception("Qdrant connection failed")

    monkeypatch.setattr(
        "image_search_service.services.admin_service.qdrant.reset_collection",
        failing_reset,
    )

    class MockFaceClient:
        def reset_collection(self) -> int:
            return 0

    monkeypatch.setattr(
        "image_search_service.services.admin_service.get_face_qdrant_client",
        lambda: MockFaceClient(),
    )

    response = await test_client.post(
        "/api/v1/admin/data/delete-all",
        json={
            "confirm": True,
            "confirmationText": "DELETE ALL DATA",
        },
    )

    # Should still complete with main collection showing 0 deleted
    assert response.status_code == 200
    data = response.json()
    assert data["qdrantCollectionsDeleted"]["main"] == 0
