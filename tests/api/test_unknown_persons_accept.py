"""Tests for POST /api/v1/faces/unknown-persons/candidates/{group_id}/accept endpoint."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.db.models import FaceInstance, ImageAsset, Person


@pytest.fixture
async def sample_asset(db_session: AsyncSession) -> ImageAsset:
    """Create sample image asset."""
    asset = ImageAsset(path="/test/image.jpg", training_status="pending")
    db_session.add(asset)
    await db_session.commit()
    await db_session.refresh(asset)
    return asset


@pytest.fixture
async def test_cluster(db_session: AsyncSession, sample_asset: ImageAsset) -> list[FaceInstance]:
    """Create test cluster with 5 faces."""
    faces = []
    for i in range(5):
        face = FaceInstance(
            id=uuid4(),
            asset_id=sample_asset.id,
            bbox_x=100 + i * 50,
            bbox_y=100,
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.95,
            quality_score=0.9 - (i * 0.05),  # Varying quality
            qdrant_point_id=uuid4(),
            cluster_id="unknown_accept_test",
            person_id=None,
        )
        db_session.add(face)
        faces.append(face)

    await db_session.commit()
    return faces


@pytest.mark.asyncio
async def test_accept_full_group(
    test_client: AsyncClient,
    db_session: AsyncSession,
    test_cluster: list[FaceInstance],
) -> None:
    """Test accepting entire group creates person and assigns all faces."""
    with patch(
        "image_search_service.api.routes.unknown_persons.get_face_qdrant_client"
    ) as mock_qdrant_factory:
        # Mock Qdrant client
        mock_qdrant = MagicMock()
        mock_qdrant_factory.return_value = mock_qdrant

        response = await test_client.post(
            "/api/v1/faces/unknown-persons/candidates/unknown_accept_test/accept",
            json={"name": "John Doe"},
        )

        assert response.status_code == 200
        data = response.json()

        assert data["personName"] == "John Doe"
        assert data["facesAssigned"] == 5
        assert data["facesExcluded"] == 0
        assert data["prototypesCreated"] == 3  # Top 3 quality faces
        assert "personId" in data
        assert "findMoreJobId" in data  # Always triggered

        # Verify person created in database
        person_result = await db_session.execute(select(Person).where(Person.name == "John Doe"))
        person = person_result.scalar_one()
        assert person is not None

        # Verify faces assigned
        faces_result = await db_session.execute(
            select(FaceInstance).where(FaceInstance.cluster_id == "unknown_accept_test")
        )
        faces = list(faces_result.scalars().all())
        assert all(f.person_id == person.id for f in faces)


@pytest.mark.asyncio
async def test_accept_partial_group_with_exclusions(
    test_client: AsyncClient,
    db_session: AsyncSession,
    test_cluster: list[FaceInstance],
) -> None:
    """Test accepting group with face exclusions (partial acceptance)."""
    # Exclude first 2 faces
    exclude_ids = [str(test_cluster[0].id), str(test_cluster[1].id)]

    with patch(
        "image_search_service.api.routes.unknown_persons.get_face_qdrant_client"
    ) as mock_qdrant_factory:
        mock_qdrant = MagicMock()
        mock_qdrant_factory.return_value = mock_qdrant

        response = await test_client.post(
            "/api/v1/faces/unknown-persons/candidates/unknown_accept_test/accept",
            json={"name": "Jane Smith", "faceIdsToExclude": exclude_ids},
        )

        assert response.status_code == 200
        data = response.json()

        assert data["facesAssigned"] == 3  # Only 3 faces assigned
        assert data["facesExcluded"] == 2

        # Verify excluded faces NOT assigned
        excluded_faces_result = await db_session.execute(
            select(FaceInstance).where(
                FaceInstance.id.in_([test_cluster[0].id, test_cluster[1].id])
            )
        )
        excluded_faces = list(excluded_faces_result.scalars().all())
        assert all(f.person_id is None for f in excluded_faces)


@pytest.mark.asyncio
async def test_accept_group_not_found(test_client: AsyncClient) -> None:
    """Test accepting non-existent group returns 404."""
    with patch(
        "image_search_service.api.routes.unknown_persons.get_face_qdrant_client"
    ) as mock_qdrant_factory:
        mock_qdrant = MagicMock()
        mock_qdrant_factory.return_value = mock_qdrant

        response = await test_client.post(
            "/api/v1/faces/unknown-persons/candidates/unknown_nonexistent/accept",
            json={"name": "Test Person"},
        )

        assert response.status_code == 404
        assert "No unassigned faces found" in response.json()["detail"]


@pytest.mark.asyncio
async def test_accept_all_faces_excluded(
    test_client: AsyncClient,
    test_cluster: list[FaceInstance],
) -> None:
    """Test accepting group with all faces excluded returns error."""
    # Exclude all faces
    exclude_ids = [str(f.id) for f in test_cluster]

    with patch(
        "image_search_service.api.routes.unknown_persons.get_face_qdrant_client"
    ) as mock_qdrant_factory:
        mock_qdrant = MagicMock()
        mock_qdrant_factory.return_value = mock_qdrant

        response = await test_client.post(
            "/api/v1/faces/unknown-persons/candidates/unknown_accept_test/accept",
            json={"name": "Test Person", "faceIdsToExclude": exclude_ids},
        )

        assert response.status_code == 400
        assert "at least one face" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_accept_with_reclustering(
    test_client: AsyncClient,
    test_cluster: list[FaceInstance],
) -> None:
    """Test accepting group triggers re-clustering when requested."""
    with patch(
        "image_search_service.api.routes.unknown_persons.get_face_qdrant_client"
    ) as mock_qdrant_factory:
        with patch("image_search_service.api.routes.unknown_persons.get_queue") as mock_get_queue:
            mock_qdrant = MagicMock()
            mock_qdrant_factory.return_value = mock_qdrant

            # Mock reclustering job
            mock_queue = MagicMock()
            mock_job = MagicMock()
            mock_job.id = "recluster-job-123"
            mock_queue.enqueue.return_value = mock_job
            mock_get_queue.return_value = mock_queue

            response = await test_client.post(
                "/api/v1/faces/unknown-persons/candidates/unknown_accept_test/accept",
                json={"name": "Test Person", "triggerReclustering": True},
            )

            assert response.status_code == 200
            data = response.json()

            assert data["reclusteringJobId"] == "recluster-job-123"


@pytest.mark.asyncio
async def test_accept_without_reclustering(
    test_client: AsyncClient,
    test_cluster: list[FaceInstance],
) -> None:
    """Test accepting group without re-clustering."""
    with patch(
        "image_search_service.api.routes.unknown_persons.get_face_qdrant_client"
    ) as mock_qdrant_factory:
        mock_qdrant = MagicMock()
        mock_qdrant_factory.return_value = mock_qdrant

        response = await test_client.post(
            "/api/v1/faces/unknown-persons/candidates/unknown_accept_test/accept",
            json={"name": "Test Person", "triggerReclustering": False},
        )

        assert response.status_code == 200
        data = response.json()

        assert data["reclusteringJobId"] is None


@pytest.mark.asyncio
async def test_accept_name_validation(test_client: AsyncClient) -> None:
    """Test name validation (min_length=1, max_length=255)."""
    with patch(
        "image_search_service.api.routes.unknown_persons.get_face_qdrant_client"
    ) as mock_qdrant_factory:
        mock_qdrant = MagicMock()
        mock_qdrant_factory.return_value = mock_qdrant

        # Empty name
        response = await test_client.post(
            "/api/v1/faces/unknown-persons/candidates/unknown_test/accept",
            json={"name": ""},
        )
        assert response.status_code == 422

        # Too long name (256 chars)
        response = await test_client.post(
            "/api/v1/faces/unknown-persons/candidates/unknown_test/accept",
            json={"name": "a" * 256},
        )
        assert response.status_code == 422


@pytest.mark.asyncio
async def test_accept_find_more_always_triggered(
    test_client: AsyncClient,
    test_cluster: list[FaceInstance],
) -> None:
    """Test find-more job is ALWAYS triggered (required for propagation)."""
    with patch(
        "image_search_service.api.routes.unknown_persons.get_face_qdrant_client"
    ) as mock_qdrant_factory:
        mock_qdrant = MagicMock()
        mock_qdrant_factory.return_value = mock_qdrant

        response = await test_client.post(
            "/api/v1/faces/unknown-persons/candidates/unknown_accept_test/accept",
            json={"name": "Test Person"},
        )

        assert response.status_code == 200
        data = response.json()

        # Find-more should be triggered (may be None if enqueue failed, but field exists)
        assert "findMoreJobId" in data


@pytest.mark.asyncio
async def test_accept_existing_person_name(
    test_client: AsyncClient,
    db_session: AsyncSession,
    test_cluster: list[FaceInstance],
) -> None:
    """Test accepting group with existing person name reuses person."""
    # Create existing person
    existing_person = Person(name="Existing Person")
    db_session.add(existing_person)
    await db_session.commit()
    await db_session.refresh(existing_person)

    with patch(
        "image_search_service.api.routes.unknown_persons.get_face_qdrant_client"
    ) as mock_qdrant_factory:
        mock_qdrant = MagicMock()
        mock_qdrant_factory.return_value = mock_qdrant

        response = await test_client.post(
            "/api/v1/faces/unknown-persons/candidates/unknown_accept_test/accept",
            json={"name": "Existing Person"},
        )

        assert response.status_code == 200
        data = response.json()

        # Should reuse existing person
        assert data["personId"] == str(existing_person.id)
        assert data["personName"] == "Existing Person"


@pytest.mark.asyncio
async def test_accept_already_assigned_faces(
    test_client: AsyncClient,
    db_session: AsyncSession,
    sample_asset: ImageAsset,
) -> None:
    """Test accepting group where all faces already assigned returns 404.

    Note: Faces with person_id are filtered out by the query,
    so the group appears not to exist (404), not as a conflict (409).
    """
    # Create person
    person = Person(name="Already Assigned")
    db_session.add(person)
    await db_session.commit()
    await db_session.refresh(person)

    # Create faces already assigned (use unique y coords)
    for i in range(3):
        face = FaceInstance(
            id=uuid4(),
            asset_id=sample_asset.id,
            bbox_x=100,
            bbox_y=500 + i,  # Unique y coordinate per face
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.95,
            quality_score=0.9,
            qdrant_point_id=uuid4(),
            cluster_id="unknown_assigned",
            person_id=person.id,  # Already assigned!
        )
        db_session.add(face)

    await db_session.commit()

    with patch(
        "image_search_service.api.routes.unknown_persons.get_face_qdrant_client"
    ) as mock_qdrant_factory:
        mock_qdrant = MagicMock()
        mock_qdrant_factory.return_value = mock_qdrant

        response = await test_client.post(
            "/api/v1/faces/unknown-persons/candidates/unknown_assigned/accept",
            json={"name": "New Person"},
        )

        # 404 because query filters for person_id IS NULL
        assert response.status_code == 404
        assert "no unassigned faces" in response.json()["detail"].lower()
