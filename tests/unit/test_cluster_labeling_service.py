"""Unit tests for ClusterLabelingService.

Tests the core logic of labeling face clusters as persons, including:
- Person creation and lookup
- Face assignment (full and partial with exclusions)
- Prototype creation
- Qdrant synchronization
- Find-more job triggering
"""

import uuid
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.db.models import (
    FaceInstance,
    Person,
    PersonPrototype,
    PrototypeRole,
)
from image_search_service.services.cluster_labeling_service import ClusterLabelingService


@pytest.fixture
def mock_qdrant() -> MagicMock:
    """Mock FaceQdrantClient for testing."""
    qdrant = MagicMock()
    qdrant.update_person_ids = MagicMock()
    qdrant.update_payload = MagicMock()
    return qdrant


@pytest.fixture
async def sample_faces(db_session: AsyncSession) -> list[FaceInstance]:
    """Create sample face instances for testing.

    Returns:
        List of 5 FaceInstance objects with varying quality scores
    """
    faces = []
    for i in range(5):
        face = FaceInstance(
            id=uuid.uuid4(),
            asset_id=1 + i,  # Different assets
            bbox_x=10 * i,
            bbox_y=10 * i,
            bbox_w=100,
            bbox_h=100,
            detection_confidence=0.9,
            quality_score=0.5 + (i * 0.1),  # 0.5, 0.6, 0.7, 0.8, 0.9
            qdrant_point_id=uuid.uuid4(),
            cluster_id="test_cluster",
        )
        db_session.add(face)
        faces.append(face)

    await db_session.commit()
    return faces


class TestClusterLabelingService:
    """Tests for ClusterLabelingService."""

    async def test_label_cluster_creates_person(
        self,
        db_session: AsyncSession,
        mock_qdrant: MagicMock,
        sample_faces: list[FaceInstance],
    ) -> None:
        """Test that labeling creates a new Person record with correct name."""
        service = ClusterLabelingService(db_session, mock_qdrant)
        face_ids = [f.id for f in sample_faces]

        result = await service.label_cluster_as_person(
            face_ids=face_ids,
            person_name="John Doe",
            trigger_find_more=False,
        )

        # Check Person was created
        assert result["person_name"] == "John Doe"
        assert result["person_id"] is not None

        # Verify in database
        person_query = select(Person).where(Person.id == result["person_id"])
        person_result = await db_session.execute(person_query)
        person = person_result.scalar_one()

        assert person.name == "John Doe"

    async def test_label_cluster_assigns_faces(
        self,
        db_session: AsyncSession,
        mock_qdrant: MagicMock,
        sample_faces: list[FaceInstance],
    ) -> None:
        """Test that all face IDs get person_id set correctly."""
        service = ClusterLabelingService(db_session, mock_qdrant)
        face_ids = [f.id for f in sample_faces]

        result = await service.label_cluster_as_person(
            face_ids=face_ids,
            person_name="Jane Smith",
            trigger_find_more=False,
        )

        person_id = result["person_id"]
        assert result["faces_assigned"] == 5
        assert result["faces_excluded"] == 0

        # Verify all faces have person_id set
        for face_id in face_ids:
            face_query = select(FaceInstance).where(FaceInstance.id == face_id)
            face_result = await db_session.execute(face_query)
            face = face_result.scalar_one()
            assert face.person_id == person_id

    async def test_label_cluster_with_exclusions(
        self,
        db_session: AsyncSession,
        mock_qdrant: MagicMock,
        sample_faces: list[FaceInstance],
    ) -> None:
        """Test that excluded faces are NOT assigned to person."""
        service = ClusterLabelingService(db_session, mock_qdrant)
        face_ids = [f.id for f in sample_faces]

        # Exclude the first 2 faces
        exclude_ids = face_ids[:2]
        expected_assigned = face_ids[2:]

        result = await service.label_cluster_as_person(
            face_ids=face_ids,
            person_name="Partial Person",
            exclude_face_ids=exclude_ids,
            trigger_find_more=False,
        )

        person_id = result["person_id"]
        assert result["faces_assigned"] == 3
        assert result["faces_excluded"] == 2

        # Verify assigned faces have person_id
        for face_id in expected_assigned:
            face_query = select(FaceInstance).where(FaceInstance.id == face_id)
            face_result = await db_session.execute(face_query)
            face = face_result.scalar_one()
            assert face.person_id == person_id

        # Verify excluded faces do NOT have person_id
        for face_id in exclude_ids:
            face_query = select(FaceInstance).where(FaceInstance.id == face_id)
            face_result = await db_session.execute(face_query)
            face = face_result.scalar_one()
            assert face.person_id is None

    async def test_label_cluster_creates_prototypes(
        self,
        db_session: AsyncSession,
        mock_qdrant: MagicMock,
        sample_faces: list[FaceInstance],
    ) -> None:
        """Test that prototype records are created for top 3 quality faces."""
        service = ClusterLabelingService(db_session, mock_qdrant)
        face_ids = [f.id for f in sample_faces]

        result = await service.label_cluster_as_person(
            face_ids=face_ids,
            person_name="Prototype Person",
            trigger_find_more=False,
        )

        person_id = result["person_id"]
        assert result["prototypes_created"] == 3

        # Verify prototypes in database
        proto_query = select(PersonPrototype).where(PersonPrototype.person_id == person_id)
        proto_result = await db_session.execute(proto_query)
        prototypes = list(proto_result.scalars().all())

        assert len(prototypes) == 3

        # All should be EXEMPLAR role
        for proto in prototypes:
            assert proto.role == PrototypeRole.EXEMPLAR
            assert proto.person_id == person_id

        # Prototypes should be the 3 highest quality faces
        # sample_faces quality: 0.5, 0.6, 0.7, 0.8, 0.9
        # Top 3: index 4, 3, 2 (quality 0.9, 0.8, 0.7)
        expected_face_ids = {sample_faces[4].id, sample_faces[3].id, sample_faces[2].id}
        actual_face_ids = {proto.face_instance_id for proto in prototypes}
        assert actual_face_ids == expected_face_ids

    async def test_label_cluster_syncs_qdrant(
        self,
        db_session: AsyncSession,
        mock_qdrant: MagicMock,
        sample_faces: list[FaceInstance],
    ) -> None:
        """Test that Qdrant is updated with person_id and is_prototype flags."""
        service = ClusterLabelingService(db_session, mock_qdrant)
        face_ids = [f.id for f in sample_faces]

        result = await service.label_cluster_as_person(
            face_ids=face_ids,
            person_name="Qdrant Person",
            trigger_find_more=False,
        )

        person_id = result["person_id"]

        # Check update_person_ids was called with all face point IDs
        expected_point_ids = set(f.qdrant_point_id for f in sample_faces)
        mock_qdrant.update_person_ids.assert_called_once()
        call_args = mock_qdrant.update_person_ids.call_args
        actual_point_ids = set(call_args[0][0])
        assert actual_point_ids == expected_point_ids
        assert call_args[0][1] == person_id

        # Check update_payload was called 3 times (for prototypes)
        # to set is_prototype=True
        assert mock_qdrant.update_payload.call_count == 3
        for call in mock_qdrant.update_payload.call_args_list:
            point_id, payload = call[0]
            assert payload == {"is_prototype": True}

    async def test_label_cluster_triggers_find_more(
        self,
        db_session: AsyncSession,
        mock_qdrant: MagicMock,
        sample_faces: list[FaceInstance],
    ) -> None:
        """Test that find-more job is enqueued when trigger_find_more=True."""
        service = ClusterLabelingService(db_session, mock_qdrant)
        face_ids = [f.id for f in sample_faces]

        # Mock Redis and RQ at import point
        with patch("redis.Redis"), \
             patch("rq.Queue") as mock_queue:

            mock_job = MagicMock()
            mock_job.id = "test_job_123"
            mock_queue_instance = MagicMock()
            mock_queue_instance.enqueue = MagicMock(return_value=mock_job)
            mock_queue.return_value = mock_queue_instance

            result = await service.label_cluster_as_person(
                face_ids=face_ids,
                person_name="Find More Person",
                trigger_find_more=True,
            )

            # Job should be enqueued
            assert result["find_more_job_id"] == "test_job_123"
            mock_queue_instance.enqueue.assert_called_once()

            # Verify job args
            call_kwargs = mock_queue_instance.enqueue.call_args[1]
            assert str(sample_faces[4].id) == call_kwargs["source_face_id"]  # Best quality face
            assert str(result["person_id"]) == call_kwargs["person_id"]

    async def test_label_cluster_skips_find_more(
        self,
        db_session: AsyncSession,
        mock_qdrant: MagicMock,
        sample_faces: list[FaceInstance],
    ) -> None:
        """Test that no job is enqueued when trigger_find_more=False."""
        service = ClusterLabelingService(db_session, mock_qdrant)
        face_ids = [f.id for f in sample_faces]

        with patch("redis.Redis") as mock_redis:
            result = await service.label_cluster_as_person(
                face_ids=face_ids,
                person_name="No Job Person",
                trigger_find_more=False,
            )

            # No job enqueued
            assert result["find_more_job_id"] is None
            mock_redis.assert_not_called()

    async def test_label_cluster_reuses_existing_person(
        self,
        db_session: AsyncSession,
        mock_qdrant: MagicMock,
        sample_faces: list[FaceInstance],
    ) -> None:
        """Test that existing Person is reused (case-insensitive match)."""
        # Create existing person
        existing_person = Person(name="Alice Test")
        db_session.add(existing_person)
        await db_session.commit()

        service = ClusterLabelingService(db_session, mock_qdrant)
        face_ids = [f.id for f in sample_faces]

        # Label with same name (different case)
        result = await service.label_cluster_as_person(
            face_ids=face_ids,
            person_name="alice test",  # Lowercase
            trigger_find_more=False,
        )

        # Should reuse existing person
        assert result["person_id"] == existing_person.id
        assert result["person_name"] == "Alice Test"  # Original case preserved

        # Verify only 1 person exists
        count_query = select(Person)
        count_result = await db_session.execute(count_query)
        all_persons = list(count_result.scalars().all())
        assert len(all_persons) == 1

    async def test_label_cluster_validates_face_ids(
        self,
        db_session: AsyncSession,
        mock_qdrant: MagicMock,
    ) -> None:
        """Test that ValueError is raised if no valid faces are found."""
        service = ClusterLabelingService(db_session, mock_qdrant)

        # Non-existent face IDs
        fake_ids = [uuid.uuid4(), uuid.uuid4()]

        with pytest.raises(ValueError, match="No faces found"):
            await service.label_cluster_as_person(
                face_ids=fake_ids,
                person_name="Invalid Person",
                trigger_find_more=False,
            )

    async def test_label_cluster_validates_exclusions(
        self,
        db_session: AsyncSession,
        mock_qdrant: MagicMock,
        sample_faces: list[FaceInstance],
    ) -> None:
        """Test that ValueError is raised if all faces are excluded."""
        service = ClusterLabelingService(db_session, mock_qdrant)
        face_ids = [f.id for f in sample_faces]

        # Exclude all faces
        with pytest.raises(ValueError, match="All faces excluded"):
            await service.label_cluster_as_person(
                face_ids=face_ids,
                person_name="Empty Person",
                exclude_face_ids=face_ids,  # Exclude everything
                trigger_find_more=False,
            )

    async def test_label_cluster_handles_job_failure_gracefully(
        self,
        db_session: AsyncSession,
        mock_qdrant: MagicMock,
        sample_faces: list[FaceInstance],
    ) -> None:
        """Test that job enqueue failure doesn't crash the labeling process."""
        service = ClusterLabelingService(db_session, mock_qdrant)
        face_ids = [f.id for f in sample_faces]

        # Mock Redis to raise exception
        with patch("redis.Redis") as mock_redis:
            mock_redis.from_url.side_effect = Exception("Redis connection failed")

            # Should not raise, just log warning
            result = await service.label_cluster_as_person(
                face_ids=face_ids,
                person_name="Resilient Person",
                trigger_find_more=True,  # Attempt to enqueue
            )

            # Labeling should succeed
            assert result["person_id"] is not None
            assert result["faces_assigned"] == 5
            # Job ID is None due to failure
            assert result["find_more_job_id"] is None
