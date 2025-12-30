"""Test training system endpoints."""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.db.models import (
    Category,
    SessionStatus,
    TrainingSession,
    TrainingSubdirectory,
)


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
async def training_session_with_subdirs(
    db_session: AsyncSession, test_category: Category
) -> TrainingSession:
    """Create a training session with subdirectories."""
    session = TrainingSession(
        name="Test Session",
        root_path="/test/images",
        category_id=test_category.id,
        status=SessionStatus.COMPLETED.value,
        total_images=150,
        processed_images=150,
        failed_images=0,
    )
    db_session.add(session)
    await db_session.flush()

    # Add subdirectories with different training statuses
    subdirs = [
        # Fully trained subdirectory
        TrainingSubdirectory(
            session_id=session.id,
            path="/test/images/trained_dir",
            name="trained_dir",
            selected=True,
            image_count=50,
            trained_count=50,
            status="trained",
        ),
        # Partially trained subdirectory
        TrainingSubdirectory(
            session_id=session.id,
            path="/test/images/partial_dir",
            name="partial_dir",
            selected=True,
            image_count=100,
            trained_count=50,
            status="training",
        ),
        # Never trained subdirectory
        TrainingSubdirectory(
            session_id=session.id,
            path="/test/images/untrained_dir",
            name="untrained_dir",
            selected=False,
            image_count=30,
            trained_count=0,
            status="pending",
        ),
    ]

    for subdir in subdirs:
        db_session.add(subdir)

    await db_session.commit()
    await db_session.refresh(session)

    return session


# Directory Listing Tests


@pytest.mark.asyncio
async def test_list_directories_without_training_status(
    test_client: AsyncClient,
    training_session_with_subdirs: TrainingSession,
    monkeypatch,
) -> None:
    """Test that directories endpoint returns basic info without training status by default."""
    # Mock DirectoryService to return fixed subdirectories
    from image_search_service.api.training_schemas import DirectoryInfo

    mock_subdirs = [
        DirectoryInfo(
            path="/test/images/trained_dir",
            name="trained_dir",
            imageCount=50,
            selected=False,
        ),
        DirectoryInfo(
            path="/test/images/partial_dir",
            name="partial_dir",
            imageCount=100,
            selected=False,
        ),
        DirectoryInfo(
            path="/test/images/untrained_dir",
            name="untrained_dir",
            imageCount=30,
            selected=False,
        ),
    ]

    def mock_list_subdirectories(self, path: str):
        return mock_subdirs

    from image_search_service.services import directory_service

    monkeypatch.setattr(
        directory_service.DirectoryService,
        "list_subdirectories",
        mock_list_subdirectories,
    )

    response = await test_client.get("/api/v1/training/directories?path=/test/images")

    assert response.status_code == 200
    data = response.json()

    assert len(data) == 3

    # Verify training status fields are None (not included by default)
    for item in data:
        assert item["trainedCount"] is None
        assert item["lastTrainedAt"] is None
        assert item["trainingStatus"] is None


@pytest.mark.asyncio
async def test_list_directories_with_training_status(
    test_client: AsyncClient,
    training_session_with_subdirs: TrainingSession,
    monkeypatch,
) -> None:
    """Test that directories endpoint includes training status when requested."""
    # Mock DirectoryService to return fixed subdirectories
    from image_search_service.api.training_schemas import DirectoryInfo

    mock_subdirs = [
        DirectoryInfo(
            path="/test/images/trained_dir",
            name="trained_dir",
            imageCount=50,
            selected=False,
        ),
        DirectoryInfo(
            path="/test/images/partial_dir",
            name="partial_dir",
            imageCount=100,
            selected=False,
        ),
        DirectoryInfo(
            path="/test/images/untrained_dir",
            name="untrained_dir",
            imageCount=30,
            selected=False,
        ),
    ]

    def mock_list_subdirectories(self, path: str):
        return mock_subdirs

    from image_search_service.services import directory_service

    monkeypatch.setattr(
        directory_service.DirectoryService,
        "list_subdirectories",
        mock_list_subdirectories,
    )

    response = await test_client.get(
        "/api/v1/training/directories?path=/test/images&include_training_status=true"
    )

    assert response.status_code == 200
    data = response.json()

    assert len(data) == 3

    # Find each directory and verify training status
    trained_dir = next(d for d in data if d["name"] == "trained_dir")
    assert trained_dir["trainedCount"] == 50
    assert trained_dir["trainingStatus"] == "complete"
    assert trained_dir["lastTrainedAt"] is not None

    partial_dir = next(d for d in data if d["name"] == "partial_dir")
    assert partial_dir["trainedCount"] == 50
    assert partial_dir["trainingStatus"] == "partial"
    assert partial_dir["lastTrainedAt"] is not None

    untrained_dir = next(d for d in data if d["name"] == "untrained_dir")
    assert untrained_dir["trainedCount"] == 0
    assert untrained_dir["trainingStatus"] == "never"
    assert untrained_dir["lastTrainedAt"] is None


@pytest.mark.asyncio
async def test_training_status_calculation_never(
    db_session: AsyncSession,
) -> None:
    """Test training status calculation for never trained directory."""
    from image_search_service.api.training_schemas import DirectoryInfo
    from image_search_service.services.training_service import TrainingService

    service = TrainingService()

    subdirs = [
        DirectoryInfo(
            path="/test/dir1",
            name="dir1",
            imageCount=100,
            selected=False,
        ),
    ]

    # No training session exists, so no enrichment should happen
    result = await service.enrich_with_training_status(db_session, subdirs, "/test")

    assert len(result) == 1
    assert result[0].trained_count == 0
    assert result[0].training_status == "never"
    assert result[0].last_trained_at is None


@pytest.mark.asyncio
async def test_training_status_calculation_partial(
    db_session: AsyncSession,
    test_category: Category,
) -> None:
    """Test training status calculation for partially trained directory."""
    from image_search_service.api.training_schemas import DirectoryInfo
    from image_search_service.services.training_service import TrainingService

    # Create session with partially trained subdirectory
    session = TrainingSession(
        name="Test Session",
        root_path="/test",
        category_id=test_category.id,
        status=SessionStatus.RUNNING.value,
        total_images=100,
        processed_images=50,
        failed_images=0,
    )
    db_session.add(session)
    await db_session.flush()

    subdir = TrainingSubdirectory(
        session_id=session.id,
        path="/test/dir1",
        name="dir1",
        selected=True,
        image_count=100,
        trained_count=50,
        status="training",
    )
    db_session.add(subdir)
    await db_session.commit()

    service = TrainingService()

    subdirs = [
        DirectoryInfo(
            path="/test/dir1",
            name="dir1",
            imageCount=100,
            selected=False,
        ),
    ]

    result = await service.enrich_with_training_status(db_session, subdirs, "/test")

    assert len(result) == 1
    assert result[0].trained_count == 50
    assert result[0].training_status == "partial"
    assert result[0].last_trained_at is not None


@pytest.mark.asyncio
async def test_training_status_calculation_complete(
    db_session: AsyncSession,
    test_category: Category,
) -> None:
    """Test training status calculation for fully trained directory."""
    from image_search_service.api.training_schemas import DirectoryInfo
    from image_search_service.services.training_service import TrainingService

    # Create session with fully trained subdirectory
    session = TrainingSession(
        name="Test Session",
        root_path="/test",
        category_id=test_category.id,
        status=SessionStatus.COMPLETED.value,
        total_images=100,
        processed_images=100,
        failed_images=0,
    )
    db_session.add(session)
    await db_session.flush()

    subdir = TrainingSubdirectory(
        session_id=session.id,
        path="/test/dir1",
        name="dir1",
        selected=True,
        image_count=100,
        trained_count=100,
        status="trained",
    )
    db_session.add(subdir)
    await db_session.commit()

    service = TrainingService()

    subdirs = [
        DirectoryInfo(
            path="/test/dir1",
            name="dir1",
            imageCount=100,
            selected=False,
        ),
    ]

    result = await service.enrich_with_training_status(db_session, subdirs, "/test")

    assert len(result) == 1
    assert result[0].trained_count == 100
    assert result[0].training_status == "complete"
    assert result[0].last_trained_at is not None


@pytest.mark.asyncio
async def test_training_status_aggregates_across_sessions(
    db_session: AsyncSession,
    test_category: Category,
) -> None:
    """Test that training status aggregates counts across multiple sessions."""
    from image_search_service.api.training_schemas import DirectoryInfo
    from image_search_service.services.training_service import TrainingService

    # Create first session
    session1 = TrainingSession(
        name="Session 1",
        root_path="/test",
        category_id=test_category.id,
        status=SessionStatus.COMPLETED.value,
        total_images=50,
        processed_images=50,
        failed_images=0,
    )
    db_session.add(session1)
    await db_session.flush()

    subdir1 = TrainingSubdirectory(
        session_id=session1.id,
        path="/test/dir1",
        name="dir1",
        selected=True,
        image_count=100,
        trained_count=30,
        status="trained",
    )
    db_session.add(subdir1)

    # Create second session training same directory
    session2 = TrainingSession(
        name="Session 2",
        root_path="/test",
        category_id=test_category.id,
        status=SessionStatus.COMPLETED.value,
        total_images=50,
        processed_images=50,
        failed_images=0,
    )
    db_session.add(session2)
    await db_session.flush()

    subdir2 = TrainingSubdirectory(
        session_id=session2.id,
        path="/test/dir1",
        name="dir1",
        selected=True,
        image_count=100,
        trained_count=40,
        status="trained",
    )
    db_session.add(subdir2)

    await db_session.commit()

    service = TrainingService()

    subdirs = [
        DirectoryInfo(
            path="/test/dir1",
            name="dir1",
            imageCount=100,
            selected=False,
        ),
    ]

    result = await service.enrich_with_training_status(db_session, subdirs, "/test")

    assert len(result) == 1
    # Should aggregate: 30 + 40 = 70
    assert result[0].trained_count == 70
    assert result[0].training_status == "partial"
    assert result[0].last_trained_at is not None
