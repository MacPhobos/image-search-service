"""Test vector management endpoints."""

import pytest
from httpx import AsyncClient
from qdrant_client import QdrantClient
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.db.models import (
    Category,
    ImageAsset,
    TrainingJob,
    TrainingSession,
    VectorDeletionLog,
)
from image_search_service.vector.qdrant import upsert_vector
from tests.conftest import MockEmbeddingService


# Fixtures


@pytest.fixture
async def test_category(db_session: AsyncSession) -> Category:
    """Create a test category."""
    category = Category(
        name="Test Category",
        description="For testing",
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
            path="/photos/2024/vacation/beach.jpg",
            training_status="trained",
        ),
        ImageAsset(
            path="/photos/2024/vacation/sunset.jpg",
            training_status="trained",
        ),
        ImageAsset(
            path="/photos/2024/work/office.jpg",
            training_status="trained",
        ),
        ImageAsset(
            path="/photos/2023/family/reunion.jpg",
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
async def test_session(
    db_session: AsyncSession, test_category: Category, test_assets: list[ImageAsset]
) -> TrainingSession:
    """Create a test training session with jobs."""
    session = TrainingSession(
        name="Test Session",
        root_path="/photos/2024",
        category_id=test_category.id,
        status="completed",
    )
    db_session.add(session)
    await db_session.commit()
    await db_session.refresh(session)

    # Add training jobs for first two assets
    for asset in test_assets[:2]:
        job = TrainingJob(
            session_id=session.id,
            asset_id=asset.id,
            status="completed",
        )
        db_session.add(job)
    await db_session.commit()

    return session


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


# Unit Tests: Qdrant Client Deletion Methods


def test_delete_vectors_by_directory_with_prefix_matching(
    qdrant_client: QdrantClient,
    test_assets: list[ImageAsset],
    test_vectors: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test delete_vectors_by_directory with path prefix matching."""
    from image_search_service.vector.qdrant import delete_vectors_by_directory

    # Patch get_qdrant_client
    monkeypatch.setattr(
        "image_search_service.vector.qdrant.get_qdrant_client", lambda: qdrant_client
    )

    # Delete vectors with path prefix /photos/2024/vacation/
    deleted_count = delete_vectors_by_directory(
        path_prefix="/photos/2024/vacation/", client=qdrant_client
    )

    # Should delete 2 vectors (beach.jpg and sunset.jpg)
    assert deleted_count == 2


def test_delete_vectors_by_asset_for_single_asset(
    qdrant_client: QdrantClient,
    test_assets: list[ImageAsset],
    test_vectors: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test delete_vectors_by_asset for single asset."""
    from image_search_service.vector.qdrant import delete_vectors_by_asset

    # Patch get_qdrant_client
    monkeypatch.setattr(
        "image_search_service.vector.qdrant.get_qdrant_client", lambda: qdrant_client
    )

    # Delete vector for first asset
    asset_id = test_assets[0].id
    deleted_count = delete_vectors_by_asset(asset_id=asset_id, client=qdrant_client)

    # Should delete 1 vector
    assert deleted_count == 1


def test_delete_vectors_by_session_with_multiple_assets(
    qdrant_client: QdrantClient,
    test_assets: list[ImageAsset],
    test_vectors: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test delete_vectors_by_session with multiple assets."""
    from image_search_service.vector.qdrant import delete_vectors_by_session

    # Patch get_qdrant_client
    monkeypatch.setattr(
        "image_search_service.vector.qdrant.get_qdrant_client", lambda: qdrant_client
    )

    # Delete vectors for first two assets (simulating a session)
    asset_ids = [test_assets[0].id, test_assets[1].id]
    deleted_count = delete_vectors_by_session(
        session_id=1, asset_ids=asset_ids, client=qdrant_client
    )

    # Should delete 2 vectors
    assert deleted_count == 2


def test_delete_vectors_by_category_with_filter(
    qdrant_client: QdrantClient,
    test_assets: list[ImageAsset],
    test_category: Category,
    test_vectors: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test delete_vectors_by_category with filter."""
    from image_search_service.vector.qdrant import delete_vectors_by_category

    # Patch get_qdrant_client
    monkeypatch.setattr(
        "image_search_service.vector.qdrant.get_qdrant_client", lambda: qdrant_client
    )

    # Delete all vectors for category
    deleted_count = delete_vectors_by_category(
        category_id=test_category.id, client=qdrant_client
    )

    # Should delete all 4 vectors
    assert deleted_count == 4


def test_get_directory_stats_returns_proper_grouping(
    qdrant_client: QdrantClient,
    test_assets: list[ImageAsset],
    test_vectors: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test get_directory_stats returns proper grouping."""
    from image_search_service.vector.qdrant import get_directory_stats

    # Patch get_qdrant_client
    monkeypatch.setattr(
        "image_search_service.vector.qdrant.get_qdrant_client", lambda: qdrant_client
    )

    # Get directory stats
    stats = get_directory_stats(client=qdrant_client)

    # Should have 3 directories: /photos/2024/vacation, /photos/2024/work, /photos/2023/family
    assert len(stats) == 3

    # Find vacation directory stats
    vacation_stats = next((s for s in stats if s["path_prefix"] == "/photos/2024/vacation"), None)
    assert vacation_stats is not None
    assert vacation_stats["vector_count"] == 2

    # Find work directory stats
    work_stats = next((s for s in stats if s["path_prefix"] == "/photos/2024/work"), None)
    assert work_stats is not None
    assert work_stats["vector_count"] == 1


def test_delete_orphan_vectors_identifies_orphans_correctly(
    qdrant_client: QdrantClient,
    test_assets: list[ImageAsset],
    test_vectors: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test delete_orphan_vectors identifies orphans correctly."""
    from image_search_service.vector.qdrant import delete_orphan_vectors

    # Patch get_qdrant_client
    monkeypatch.setattr(
        "image_search_service.vector.qdrant.get_qdrant_client", lambda: qdrant_client
    )

    # Create set of valid asset IDs (only first two assets)
    valid_asset_ids = {test_assets[0].id, test_assets[1].id}

    # Delete orphan vectors
    deleted_count = delete_orphan_vectors(valid_asset_ids=valid_asset_ids, client=qdrant_client)

    # Should delete 2 orphans (last two assets)
    assert deleted_count == 2


def test_reset_collection_clears_all_vectors(
    qdrant_client: QdrantClient,
    test_assets: list[ImageAsset],
    test_vectors: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test reset_collection clears all vectors."""
    from image_search_service.core.config import get_settings
    from image_search_service.vector.qdrant import reset_collection

    # Patch get_qdrant_client
    monkeypatch.setattr(
        "image_search_service.vector.qdrant.get_qdrant_client", lambda: qdrant_client
    )

    # Verify vectors exist
    settings = get_settings()
    collection_info = qdrant_client.get_collection(collection_name=settings.qdrant_collection)
    assert collection_info.points_count == 4

    # Reset collection
    deleted_count = reset_collection(client=qdrant_client)

    # Should delete all 4 vectors
    assert deleted_count == 4

    # Verify collection is empty
    collection_info = qdrant_client.get_collection(collection_name=settings.qdrant_collection)
    assert collection_info.points_count == 0


# Integration Tests: API Endpoints


@pytest.mark.asyncio
async def test_get_directory_stats_returns_statistics(
    test_client: AsyncClient,
    test_assets: list[ImageAsset],
    test_vectors: None,
) -> None:
    """Test GET /vectors/directories/stats returns directory statistics."""
    response = await test_client.get("/api/v1/vectors/directories/stats")

    assert response.status_code == 200
    data = response.json()

    assert "directories" in data
    assert "totalVectors" in data
    assert data["totalVectors"] == 4
    assert len(data["directories"]) == 3


@pytest.mark.asyncio
async def test_delete_by_directory_requires_confirmation(
    test_client: AsyncClient,
    test_assets: list[ImageAsset],
    test_vectors: None,
) -> None:
    """Test DELETE /vectors/by-directory requires confirmation."""
    response = await test_client.request(
        "DELETE",
        "/api/v1/vectors/by-directory",
        json={
            "pathPrefix": "/photos/2024/vacation/",
            "confirm": False,
        },
    )

    assert response.status_code == 400
    assert "confirm" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_delete_by_directory_deletes_vectors_and_creates_log(
    test_client: AsyncClient,
    db_session: AsyncSession,
    test_assets: list[ImageAsset],
    test_vectors: None,
) -> None:
    """Test DELETE /vectors/by-directory deletes vectors and creates log."""
    response = await test_client.request(
        "DELETE",
        "/api/v1/vectors/by-directory",
        json={
            "pathPrefix": "/photos/2024/vacation/",
            "deletionReason": "Test deletion",
            "confirm": True,
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert data["pathPrefix"] == "/photos/2024/vacation/"
    assert data["vectorsDeleted"] == 2
    assert "Successfully deleted" in data["message"]

    # Verify deletion log was created
    from sqlalchemy import select

    log_query = select(VectorDeletionLog).where(
        VectorDeletionLog.deletion_type == "DIRECTORY"
    )
    result = await db_session.execute(log_query)
    log = result.scalar_one()

    assert log.deletion_target == "/photos/2024/vacation/"
    assert log.vector_count == 2
    assert log.deletion_reason == "Test deletion"


@pytest.mark.asyncio
async def test_retrain_creates_new_session_after_deletion(
    test_client: AsyncClient,
    db_session: AsyncSession,
    test_category: Category,
    test_assets: list[ImageAsset],
    test_vectors: None,
) -> None:
    """Test POST /vectors/retrain creates new session after deletion."""
    response = await test_client.post(
        "/api/v1/vectors/retrain",
        json={
            "pathPrefix": "/photos/2024/vacation/",
            "categoryId": test_category.id,
            "deletionReason": "Retrain operation",
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert data["pathPrefix"] == "/photos/2024/vacation/"
    assert data["vectorsDeleted"] == 2
    assert "newSessionId" in data
    assert "created new training session" in data["message"]

    # Verify new training session was created
    from sqlalchemy import select

    session_query = select(TrainingSession).where(TrainingSession.id == data["newSessionId"])
    result = await db_session.execute(session_query)
    session = result.scalar_one()

    assert session.root_path == "/photos/2024/vacation/"
    assert session.category_id == test_category.id
    assert session.status == "PENDING"


@pytest.mark.asyncio
async def test_delete_by_asset_deletes_single_asset_vector(
    test_client: AsyncClient,
    db_session: AsyncSession,
    test_assets: list[ImageAsset],
    test_vectors: None,
) -> None:
    """Test DELETE /vectors/by-asset/{id} deletes single asset vector."""
    asset_id = test_assets[0].id

    response = await test_client.delete(f"/api/v1/vectors/by-asset/{asset_id}")

    assert response.status_code == 200
    data = response.json()

    assert data["assetId"] == asset_id
    assert data["vectorsDeleted"] == 1
    assert "Successfully deleted" in data["message"]


@pytest.mark.asyncio
async def test_delete_by_session_marks_session_as_reset(
    test_client: AsyncClient,
    db_session: AsyncSession,
    test_session: TrainingSession,
    test_vectors: None,
) -> None:
    """Test DELETE /vectors/by-session/{id} marks session as reset."""
    session_id = test_session.id

    response = await test_client.delete(f"/api/v1/vectors/by-session/{session_id}")

    assert response.status_code == 200
    data = response.json()

    assert data["sessionId"] == session_id
    assert data["vectorsDeleted"] == 2  # Session has 2 assets

    # Verify session was marked as reset
    await db_session.refresh(test_session)
    assert test_session.reset_at is not None
    assert test_session.reset_reason == "Manual session vector deletion"


@pytest.mark.asyncio
async def test_delete_by_category_deletes_category_vectors(
    test_client: AsyncClient,
    db_session: AsyncSession,
    test_category: Category,
    test_assets: list[ImageAsset],
    test_vectors: None,
) -> None:
    """Test DELETE /vectors/by-category/{id} deletes category vectors."""
    category_id = test_category.id

    response = await test_client.delete(f"/api/v1/vectors/by-category/{category_id}")

    assert response.status_code == 200
    data = response.json()

    assert data["categoryId"] == category_id
    assert data["vectorsDeleted"] == 4
    assert "Successfully deleted" in data["message"]


@pytest.mark.asyncio
async def test_cleanup_orphans_requires_confirmation(
    test_client: AsyncClient,
    test_assets: list[ImageAsset],
    test_vectors: None,
) -> None:
    """Test POST /vectors/cleanup-orphans requires confirmation."""
    response = await test_client.post(
        "/api/v1/vectors/cleanup-orphans",
        json={
            "confirm": False,
        },
    )

    assert response.status_code == 400
    assert "confirm" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_cleanup_orphans_deletes_orphaned_vectors(
    test_client: AsyncClient,
    db_session: AsyncSession,
    test_assets: list[ImageAsset],
    test_vectors: None,
) -> None:
    """Test POST /vectors/cleanup-orphans deletes orphaned vectors."""
    # Delete last two assets from database (but vectors remain)
    await db_session.delete(test_assets[2])
    await db_session.delete(test_assets[3])
    await db_session.commit()

    response = await test_client.post(
        "/api/v1/vectors/cleanup-orphans",
        json={
            "confirm": True,
            "deletionReason": "Cleanup test",
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert data["orphansDeleted"] == 2
    assert "Successfully deleted" in data["message"]


@pytest.mark.asyncio
async def test_reset_requires_double_confirmation(
    test_client: AsyncClient,
    test_assets: list[ImageAsset],
    test_vectors: None,
) -> None:
    """Test POST /vectors/reset requires double confirmation."""
    # Missing confirm flag
    response = await test_client.post(
        "/api/v1/vectors/reset",
        json={
            "confirm": False,
            "confirmationText": "DELETE ALL VECTORS",
        },
    )

    assert response.status_code == 400
    assert "confirm" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_reset_with_invalid_confirmation_text_returns_400(
    test_client: AsyncClient,
    test_assets: list[ImageAsset],
    test_vectors: None,
) -> None:
    """Test POST /vectors/reset with invalid confirmation text returns 400."""
    response = await test_client.post(
        "/api/v1/vectors/reset",
        json={
            "confirm": True,
            "confirmationText": "delete all vectors",  # Wrong case
        },
    )

    assert response.status_code == 400
    assert "confirmationText" in response.json()["detail"]


@pytest.mark.asyncio
async def test_reset_deletes_all_vectors(
    test_client: AsyncClient,
    db_session: AsyncSession,
    test_assets: list[ImageAsset],
    test_vectors: None,
) -> None:
    """Test POST /vectors/reset deletes all vectors."""
    response = await test_client.post(
        "/api/v1/vectors/reset",
        json={
            "confirm": True,
            "confirmationText": "DELETE ALL VECTORS",
            "deletionReason": "Full reset test",
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert data["vectorsDeleted"] == 4
    assert "reset collection" in data["message"].lower()

    # Verify deletion log was created
    from sqlalchemy import select

    log_query = select(VectorDeletionLog).where(
        VectorDeletionLog.deletion_type == "FULL_RESET"
    )
    result = await db_session.execute(log_query)
    log = result.scalar_one()

    assert log.vector_count == 4
    assert log.deletion_reason == "Full reset test"


@pytest.mark.asyncio
async def test_get_deletion_logs_returns_paginated_logs(
    test_client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Test GET /vectors/deletion-logs returns paginated logs."""
    # Create some deletion logs
    logs = [
        VectorDeletionLog(
            deletion_type="ASSET",
            deletion_target="1",
            vector_count=1,
        ),
        VectorDeletionLog(
            deletion_type="DIRECTORY",
            deletion_target="/test/path/",
            vector_count=10,
            deletion_reason="Test deletion",
        ),
        VectorDeletionLog(
            deletion_type="FULL_RESET",
            deletion_target="all_vectors",
            vector_count=100,
        ),
    ]
    for log in logs:
        db_session.add(log)
    await db_session.commit()

    # Test page 1
    response = await test_client.get("/api/v1/vectors/deletion-logs?page=1&page_size=2")

    assert response.status_code == 200
    data = response.json()

    assert "logs" in data
    assert data["total"] == 3
    assert data["page"] == 1
    assert data["pageSize"] == 2
    assert len(data["logs"]) == 2

    # Test page 2
    response = await test_client.get("/api/v1/vectors/deletion-logs?page=2&page_size=2")

    assert response.status_code == 200
    data = response.json()

    assert len(data["logs"]) == 1


# Edge Cases


@pytest.mark.asyncio
async def test_delete_non_existent_asset_returns_404(
    test_client: AsyncClient,
) -> None:
    """Test delete non-existent asset returns 404."""
    response = await test_client.delete("/api/v1/vectors/by-asset/99999")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_delete_non_existent_session_returns_404(
    test_client: AsyncClient,
) -> None:
    """Test delete non-existent session returns 404."""
    response = await test_client.delete("/api/v1/vectors/by-session/99999")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_delete_non_existent_category_returns_404(
    test_client: AsyncClient,
) -> None:
    """Test delete non-existent category returns 404."""
    response = await test_client.delete("/api/v1/vectors/by-category/99999")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_retrain_with_non_existent_category_returns_404(
    test_client: AsyncClient,
    test_assets: list[ImageAsset],
    test_vectors: None,
) -> None:
    """Test retrain with non-existent category returns 404."""
    response = await test_client.post(
        "/api/v1/vectors/retrain",
        json={
            "pathPrefix": "/photos/2024/vacation/",
            "categoryId": 99999,
        },
    )

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
