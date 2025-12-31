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

    # Create session with partially trained subdirectory (COMPLETED, not RUNNING)
    session = TrainingSession(
        name="Test Session",
        root_path="/test",
        category_id=test_category.id,
        status=SessionStatus.COMPLETED.value,  # Changed from RUNNING to test "partial" status
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
        status="trained",  # Changed from "training" since session is completed
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
async def test_training_status_calculation_in_progress(
    db_session: AsyncSession,
    test_category: Category,
) -> None:
    """Test training status calculation for in-progress training."""
    from image_search_service.api.training_schemas import DirectoryInfo
    from image_search_service.services.training_service import TrainingService

    # Create session with RUNNING status
    session = TrainingSession(
        name="Test Session",
        root_path="/test",
        category_id=test_category.id,
        status=SessionStatus.RUNNING.value,  # Currently running
        total_images=100,
        processed_images=25,
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
        trained_count=25,
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
    assert result[0].trained_count == 25
    assert result[0].training_status == "in_progress"
    assert result[0].last_trained_at is not None


@pytest.mark.asyncio
async def test_training_status_in_progress_zero_count(
    db_session: AsyncSession,
    test_category: Category,
) -> None:
    """Test in_progress status even when trained_count is 0."""
    from image_search_service.api.training_schemas import DirectoryInfo
    from image_search_service.services.training_service import TrainingService

    # Create RUNNING session with no images trained yet
    session = TrainingSession(
        name="Test Session",
        root_path="/test",
        category_id=test_category.id,
        status=SessionStatus.RUNNING.value,
        total_images=100,
        processed_images=0,
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
        trained_count=0,  # Not trained yet but session is running
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
    assert result[0].trained_count == 0
    # Should still show in_progress because session is RUNNING
    assert result[0].training_status == "in_progress"
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


@pytest.mark.asyncio
async def test_training_status_aggregation_reaches_complete(
    db_session: AsyncSession,
    test_category: Category,
) -> None:
    """Test that aggregated training across sessions reaches complete status."""
    from image_search_service.api.training_schemas import DirectoryInfo
    from image_search_service.services.training_service import TrainingService

    # Create first session (50/100 images)
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
        trained_count=50,
        status="trained",
    )
    db_session.add(subdir1)

    # Create second session (remaining 50/100 images)
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
        trained_count=50,
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
    # Should aggregate: 50 + 50 = 100 (complete)
    assert result[0].trained_count == 100
    assert result[0].training_status == "complete"
    assert result[0].last_trained_at is not None


@pytest.mark.asyncio
async def test_training_status_mixed_batch_request(
    db_session: AsyncSession,
    test_category: Category,
) -> None:
    """Test enrichment with multiple subdirectories in different training states."""
    from image_search_service.api.training_schemas import DirectoryInfo
    from image_search_service.services.training_service import TrainingService

    # Create completed session
    completed_session = TrainingSession(
        name="Completed Session",
        root_path="/test",
        category_id=test_category.id,
        status=SessionStatus.COMPLETED.value,
        total_images=100,
        processed_images=100,
        failed_images=0,
    )
    db_session.add(completed_session)
    await db_session.flush()

    # Add fully trained directory
    subdir_complete = TrainingSubdirectory(
        session_id=completed_session.id,
        path="/test/complete_dir",
        name="complete_dir",
        selected=True,
        image_count=50,
        trained_count=50,
        status="trained",
    )
    db_session.add(subdir_complete)

    # Add partially trained directory
    subdir_partial = TrainingSubdirectory(
        session_id=completed_session.id,
        path="/test/partial_dir",
        name="partial_dir",
        selected=True,
        image_count=100,
        trained_count=30,
        status="trained",
    )
    db_session.add(subdir_partial)

    # Create running session for in_progress directory
    running_session = TrainingSession(
        name="Running Session",
        root_path="/test",
        category_id=test_category.id,
        status=SessionStatus.RUNNING.value,
        total_images=50,
        processed_images=10,
        failed_images=0,
    )
    db_session.add(running_session)
    await db_session.flush()

    # Add in-progress directory
    subdir_in_progress = TrainingSubdirectory(
        session_id=running_session.id,
        path="/test/in_progress_dir",
        name="in_progress_dir",
        selected=True,
        image_count=50,
        trained_count=10,
        status="training",
    )
    db_session.add(subdir_in_progress)

    await db_session.commit()

    service = TrainingService()

    # Request all directories at once, including never-trained one
    subdirs = [
        DirectoryInfo(
            path="/test/complete_dir",
            name="complete_dir",
            imageCount=50,
            selected=False,
        ),
        DirectoryInfo(
            path="/test/partial_dir",
            name="partial_dir",
            imageCount=100,
            selected=False,
        ),
        DirectoryInfo(
            path="/test/in_progress_dir",
            name="in_progress_dir",
            imageCount=50,
            selected=False,
        ),
        DirectoryInfo(
            path="/test/never_trained_dir",
            name="never_trained_dir",
            imageCount=25,
            selected=False,
        ),
    ]

    result = await service.enrich_with_training_status(db_session, subdirs, "/test")

    assert len(result) == 4

    # Find each directory and verify status
    complete_dir = next(d for d in result if d.name == "complete_dir")
    assert complete_dir.trained_count == 50
    assert complete_dir.training_status == "complete"
    assert complete_dir.last_trained_at is not None

    partial_dir = next(d for d in result if d.name == "partial_dir")
    assert partial_dir.trained_count == 30
    assert partial_dir.training_status == "partial"
    assert partial_dir.last_trained_at is not None

    in_progress_dir = next(d for d in result if d.name == "in_progress_dir")
    assert in_progress_dir.trained_count == 10
    assert in_progress_dir.training_status == "in_progress"
    assert in_progress_dir.last_trained_at is not None

    never_trained_dir = next(d for d in result if d.name == "never_trained_dir")
    assert never_trained_dir.trained_count == 0
    assert never_trained_dir.training_status == "never"
    assert never_trained_dir.last_trained_at is None


@pytest.mark.asyncio
async def test_training_status_cross_session_with_in_progress(
    db_session: AsyncSession,
    test_category: Category,
) -> None:
    """Test that in_progress status takes priority even with completed sessions."""
    from image_search_service.api.training_schemas import DirectoryInfo
    from image_search_service.services.training_service import TrainingService

    # Create completed session with 50 trained images
    completed_session = TrainingSession(
        name="Completed Session",
        root_path="/test",
        category_id=test_category.id,
        status=SessionStatus.COMPLETED.value,
        total_images=50,
        processed_images=50,
        failed_images=0,
    )
    db_session.add(completed_session)
    await db_session.flush()

    subdir1 = TrainingSubdirectory(
        session_id=completed_session.id,
        path="/test/dir1",
        name="dir1",
        selected=True,
        image_count=100,
        trained_count=50,
        status="trained",
    )
    db_session.add(subdir1)

    # Create running session training more images
    running_session = TrainingSession(
        name="Running Session",
        root_path="/test",
        category_id=test_category.id,
        status=SessionStatus.RUNNING.value,  # Currently running
        total_images=50,
        processed_images=20,
        failed_images=0,
    )
    db_session.add(running_session)
    await db_session.flush()

    subdir2 = TrainingSubdirectory(
        session_id=running_session.id,
        path="/test/dir1",
        name="dir1",
        selected=True,
        image_count=100,
        trained_count=20,
        status="training",
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
    # Should aggregate: 50 + 20 = 70
    assert result[0].trained_count == 70
    # Should be in_progress because one session is RUNNING
    assert result[0].training_status == "in_progress"
    assert result[0].last_trained_at is not None


@pytest.mark.asyncio
async def test_training_status_trained_count_exceeds_image_count(
    db_session: AsyncSession,
    test_category: Category,
) -> None:
    """Test handling when trained_count exceeds current image_count (images deleted)."""
    from image_search_service.api.training_schemas import DirectoryInfo
    from image_search_service.services.training_service import TrainingService

    # Create session that trained 100 images
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

    # Directory now only has 50 images (some were deleted)
    subdirs = [
        DirectoryInfo(
            path="/test/dir1",
            name="dir1",
            imageCount=50,  # Fewer images than trained
            selected=False,
        ),
    ]

    result = await service.enrich_with_training_status(db_session, subdirs, "/test")

    assert len(result) == 1
    assert result[0].trained_count == 100
    # Should still be "complete" because trained_count >= image_count
    assert result[0].training_status == "complete"
    assert result[0].last_trained_at is not None


@pytest.mark.asyncio
async def test_training_status_empty_subdirectories_list(
    db_session: AsyncSession,
) -> None:
    """Test enrichment with empty subdirectories list."""
    from image_search_service.services.training_service import TrainingService

    service = TrainingService()

    result = await service.enrich_with_training_status(db_session, [], "/test")

    assert result == []
