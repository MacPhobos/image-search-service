"""Unit tests for PersonService.

Tests use real async DB sessions (SQLite in-memory) instead of mocking
internal methods (_get_identified_people, _get_unidentified_clusters,
_get_noise_faces). This catches SQL query bugs that mocked tests miss.

The _generate_display_name tests remain unchanged (pure logic, no DB needed).
"""

import uuid

import pytest

from image_search_service.api.face_schemas import PersonType
from image_search_service.db.models import (
    FaceInstance,
    ImageAsset,
    Person,
    PersonStatus,
    TrainingStatus,
)
from image_search_service.services.person_service import PersonService

# ============ Async factory fixtures ============


@pytest.fixture
async def create_image_asset(db_session):
    """Factory for creating ImageAsset records in async session."""

    async def _create(path: str | None = None) -> ImageAsset:
        if path is None:
            path = f"/test/images/photo_{uuid.uuid4().hex[:8]}.jpg"
        asset = ImageAsset(
            path=path,
            training_status=TrainingStatus.PENDING.value,
            width=640,
            height=480,
            file_size=102400,
            mime_type="image/jpeg",
        )
        db_session.add(asset)
        await db_session.flush()
        await db_session.refresh(asset)
        return asset

    return _create


@pytest.fixture
async def create_person(db_session):
    """Factory for creating Person records in async session."""
    _counter = [0]

    async def _create(
        name: str | None = None,
        status: PersonStatus = PersonStatus.ACTIVE,
    ) -> Person:
        if name is None:
            _counter[0] += 1
            name = f"Test Person {_counter[0]}"
        person = Person(
            id=uuid.uuid4(),
            name=name,
            status=status.value,
        )
        db_session.add(person)
        await db_session.flush()
        await db_session.refresh(person)
        return person

    return _create


@pytest.fixture
async def create_face_instance(db_session, create_image_asset):
    """Factory for creating FaceInstance records in async session."""

    async def _create(
        person_id: uuid.UUID | None = None,
        cluster_id: str | None = None,
        quality_score: float = 0.75,
        asset_id: int | None = None,
    ) -> FaceInstance:
        if asset_id is None:
            asset = await create_image_asset()
            asset_id = asset.id
        face = FaceInstance(
            id=uuid.uuid4(),
            asset_id=asset_id,
            person_id=person_id,
            cluster_id=cluster_id,
            bbox_x=100,
            bbox_y=150,
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.95,
            quality_score=quality_score,
            qdrant_point_id=uuid.uuid4(),
        )
        db_session.add(face)
        await db_session.flush()
        await db_session.refresh(face)
        return face

    return _create


class TestGenerateDisplayName:
    """Test suite for _generate_display_name (pure logic, no DB needed)."""

    @pytest.fixture
    def service(self):
        """Create PersonService with a dummy db (not used by these tests)."""
        from unittest.mock import AsyncMock

        return PersonService(AsyncMock())

    def test_generate_display_name_when_regular_cluster_then_numbered(self, service):
        """Should generate 'Unidentified Person N' for regular clusters."""
        assert service._generate_display_name("clu_abc123", 0) == "Unidentified Person 1"
        assert service._generate_display_name("cluster_5", 4) == "Unidentified Person 5"
        assert service._generate_display_name("some_cluster", 9) == "Unidentified Person 10"

    def test_generate_display_name_when_noise_cluster_then_unknown_faces(self, service):
        """Should return 'Unknown Faces' for noise clusters."""
        assert service._generate_display_name("-1", 0) == "Unknown Faces"
        assert service._generate_display_name("noise", 0) == "Unknown Faces"
        assert service._generate_display_name(None, 0) == "Unknown Faces"

    def test_generate_display_name_when_sequential_indices_then_sequential_names(self, service):
        """Should generate sequential numbers based on index."""
        names = [service._generate_display_name(f"cluster_{i}", i) for i in range(5)]
        expected = [
            "Unidentified Person 1",
            "Unidentified Person 2",
            "Unidentified Person 3",
            "Unidentified Person 4",
            "Unidentified Person 5",
        ]
        assert names == expected


class TestGetAllPeopleWithRealDB:
    """Test get_all_people using real DB records instead of mocked internals."""

    # ============ Filtering tests ============

    @pytest.mark.asyncio
    async def test_get_all_people_when_only_identified_then_returns_identified(
        self, db_session, create_person, create_face_instance
    ):
        """Should return only identified people when include_unidentified=False."""
        person = await create_person(name="John Doe")
        await create_face_instance(person_id=person.id)
        await create_face_instance(person_id=person.id)

        # Also create an unidentified cluster to prove it is excluded
        await create_face_instance(cluster_id="cluster_99")

        service = PersonService(db_session)
        result = await service.get_all_people(
            include_identified=True,
            include_unidentified=False,
            include_noise=False,
        )

        assert result.identified_count == 1
        assert result.unidentified_count == 0
        assert result.noise_count == 0
        assert len(result.people) == 1
        assert result.people[0].type == PersonType.IDENTIFIED
        assert result.people[0].name == "John Doe"
        assert result.people[0].face_count == 2

    @pytest.mark.asyncio
    async def test_get_all_people_when_only_unidentified_then_returns_clusters(
        self, db_session, create_person, create_face_instance
    ):
        """Should return only unidentified clusters when include_identified=False."""
        # Create an identified person to prove it is excluded
        person = await create_person(name="Jane Doe")
        await create_face_instance(person_id=person.id)

        # Create unidentified cluster faces (no person_id, with cluster_id != -1)
        await create_face_instance(cluster_id="cluster_abc")
        await create_face_instance(cluster_id="cluster_abc")
        await create_face_instance(cluster_id="cluster_abc")

        service = PersonService(db_session)
        result = await service.get_all_people(
            include_identified=False,
            include_unidentified=True,
            include_noise=False,
        )

        assert result.identified_count == 0
        assert result.unidentified_count == 1
        assert result.noise_count == 0
        assert len(result.people) == 1
        assert result.people[0].type == PersonType.UNIDENTIFIED
        assert result.people[0].face_count == 3

    @pytest.mark.asyncio
    async def test_get_all_people_when_include_noise_then_returns_noise(
        self, db_session, create_face_instance
    ):
        """Should include noise faces when requested."""
        # Create noise faces (cluster_id = '-1', no person_id)
        await create_face_instance(cluster_id="-1")
        await create_face_instance(cluster_id="-1")
        # Also add a face with NULL cluster (also treated as noise)
        await create_face_instance(cluster_id=None, person_id=None)

        service = PersonService(db_session)
        result = await service.get_all_people(
            include_identified=False,
            include_unidentified=False,
            include_noise=True,
        )

        assert result.noise_count == 1  # Aggregated into single entry
        assert result.identified_count == 0
        assert result.unidentified_count == 0
        assert len(result.people) == 1
        assert result.people[0].type == PersonType.NOISE
        assert result.people[0].name == "Unknown Faces"
        assert result.people[0].face_count == 3

    @pytest.mark.asyncio
    async def test_get_all_people_when_all_types_then_returns_mixed(
        self, db_session, create_person, create_face_instance
    ):
        """Should return all types when all filters enabled."""
        # Identified person
        person = await create_person(name="Alice")
        await create_face_instance(person_id=person.id)

        # Unidentified cluster
        await create_face_instance(cluster_id="cluster_x")
        await create_face_instance(cluster_id="cluster_x")

        # Noise
        await create_face_instance(cluster_id="-1")

        service = PersonService(db_session)
        result = await service.get_all_people(
            include_identified=True,
            include_unidentified=True,
            include_noise=True,
        )

        assert result.total == 3
        assert result.identified_count == 1
        assert result.unidentified_count == 1
        assert result.noise_count == 1

        types = {p.type for p in result.people}
        assert types == {PersonType.IDENTIFIED, PersonType.UNIDENTIFIED, PersonType.NOISE}

    @pytest.mark.asyncio
    async def test_get_all_people_when_no_noise_faces_then_noise_not_included(self, db_session):
        """Should not include noise entry when there are no noise faces."""
        service = PersonService(db_session)
        result = await service.get_all_people(
            include_identified=True,
            include_unidentified=True,
            include_noise=True,
        )

        assert len(result.people) == 0
        assert result.noise_count == 0

    # ============ Sorting tests ============

    @pytest.mark.asyncio
    async def test_get_all_people_when_sort_by_face_count_desc_then_highest_first(
        self, db_session, create_person, create_face_instance
    ):
        """Should sort by face count descending by default."""
        # Person A: 1 face
        person_a = await create_person(name="Person A")
        await create_face_instance(person_id=person_a.id)

        # Person B: 3 faces
        person_b = await create_person(name="Person B")
        for _ in range(3):
            await create_face_instance(person_id=person_b.id)

        # Person C: 2 faces
        person_c = await create_person(name="Person C")
        for _ in range(2):
            await create_face_instance(person_id=person_c.id)

        service = PersonService(db_session)
        result = await service.get_all_people(
            include_unidentified=False,
            sort_by="face_count",
            sort_order="desc",
        )

        face_counts = [p.face_count for p in result.people]
        assert face_counts == [3, 2, 1]

    @pytest.mark.asyncio
    async def test_get_all_people_when_sort_by_face_count_asc_then_lowest_first(
        self, db_session, create_person, create_face_instance
    ):
        """Should sort by face count ascending when requested."""
        person_a = await create_person(name="Person A")
        await create_face_instance(person_id=person_a.id)

        person_b = await create_person(name="Person B")
        for _ in range(3):
            await create_face_instance(person_id=person_b.id)

        service = PersonService(db_session)
        result = await service.get_all_people(
            include_unidentified=False,
            sort_by="face_count",
            sort_order="asc",
        )

        face_counts = [p.face_count for p in result.people]
        assert face_counts == [1, 3]

    @pytest.mark.asyncio
    async def test_get_all_people_when_sort_by_name_asc_then_alphabetical(
        self, db_session, create_person, create_face_instance
    ):
        """Should sort by name ascending (case-insensitive)."""
        for name in ["Charlie", "Alice", "Bob"]:
            person = await create_person(name=name)
            await create_face_instance(person_id=person.id)

        service = PersonService(db_session)
        result = await service.get_all_people(
            include_unidentified=False,
            sort_by="name",
            sort_order="asc",
        )

        names = [p.name for p in result.people]
        assert names == ["Alice", "Bob", "Charlie"]

    @pytest.mark.asyncio
    async def test_get_all_people_when_sort_by_name_desc_then_reverse_alphabetical(
        self, db_session, create_person, create_face_instance
    ):
        """Should sort by name descending."""
        for name in ["Charlie", "Alice", "Bob"]:
            person = await create_person(name=name)
            await create_face_instance(person_id=person.id)

        service = PersonService(db_session)
        result = await service.get_all_people(
            include_unidentified=False,
            sort_by="name",
            sort_order="desc",
        )

        names = [p.name for p in result.people]
        assert names == ["Charlie", "Bob", "Alice"]

    @pytest.mark.asyncio
    async def test_get_all_people_when_sort_mixed_types_then_sorts_across_types(
        self, db_session, create_person, create_face_instance
    ):
        """Should sort correctly across different person types."""
        # Identified: 2 faces
        person = await create_person(name="Alice")
        for _ in range(2):
            await create_face_instance(person_id=person.id)

        # Unidentified cluster: 5 faces
        for _ in range(5):
            await create_face_instance(cluster_id="cluster_big")

        # Noise: 1 face
        await create_face_instance(cluster_id="-1")

        service = PersonService(db_session)
        result = await service.get_all_people(
            include_identified=True,
            include_unidentified=True,
            include_noise=True,
            sort_by="face_count",
            sort_order="desc",
        )

        face_counts = [p.face_count for p in result.people]
        assert face_counts == [5, 2, 1]
        assert result.people[0].type == PersonType.UNIDENTIFIED
        assert result.people[1].type == PersonType.IDENTIFIED
        assert result.people[2].type == PersonType.NOISE

    # ============ Count accuracy tests ============

    @pytest.mark.asyncio
    async def test_get_all_people_when_multiple_of_each_type_then_counts_accurate(
        self, db_session, create_person, create_face_instance
    ):
        """Should return accurate counts for each type."""
        # 3 identified people
        for i in range(3):
            person = await create_person(name=f"Person {i}")
            await create_face_instance(person_id=person.id)

        # 2 unidentified clusters
        for cluster_id in ["cluster_a", "cluster_b"]:
            await create_face_instance(cluster_id=cluster_id)

        # Noise
        await create_face_instance(cluster_id="-1")

        service = PersonService(db_session)
        result = await service.get_all_people(
            include_identified=True,
            include_unidentified=True,
            include_noise=True,
        )

        assert result.total == 6  # 3 identified + 2 unidentified + 1 noise
        assert result.identified_count == 3
        assert result.unidentified_count == 2
        assert result.noise_count == 1

    @pytest.mark.asyncio
    async def test_get_all_people_when_empty_db_then_returns_empty(self, db_session):
        """Should handle empty results gracefully."""
        service = PersonService(db_session)
        result = await service.get_all_people()

        assert result.people == []
        assert result.total == 0
        assert result.identified_count == 0
        assert result.unidentified_count == 0
        assert result.noise_count == 0

    # ============ Edge cases ============

    @pytest.mark.asyncio
    async def test_get_all_people_when_person_has_zero_faces_then_still_returned(
        self, db_session, create_person
    ):
        """A person with no face instances should still appear with face_count=0."""
        await create_person(name="No Faces Person")

        service = PersonService(db_session)
        result = await service.get_all_people(
            include_identified=True,
            include_unidentified=False,
            include_noise=False,
        )

        assert result.identified_count == 1
        assert len(result.people) == 1
        assert result.people[0].name == "No Faces Person"
        assert result.people[0].face_count == 0

    @pytest.mark.asyncio
    async def test_get_all_people_when_hidden_person_then_excluded(
        self, db_session, create_person, create_face_instance
    ):
        """Persons with non-ACTIVE status should be excluded."""
        # Active person should appear
        active = await create_person(name="Active Person")
        await create_face_instance(person_id=active.id)

        # Hidden person should not appear
        hidden = await create_person(name="Hidden Person", status=PersonStatus.HIDDEN)
        await create_face_instance(person_id=hidden.id)

        # Merged person should not appear
        merged = await create_person(name="Merged Person", status=PersonStatus.MERGED)
        await create_face_instance(person_id=merged.id)

        service = PersonService(db_session)
        result = await service.get_all_people(
            include_identified=True,
            include_unidentified=False,
            include_noise=False,
        )

        assert result.identified_count == 1
        assert result.people[0].name == "Active Person"

    @pytest.mark.asyncio
    async def test_get_all_people_when_faces_assigned_to_person_then_not_in_unidentified(
        self, db_session, create_person, create_face_instance
    ):
        """Faces assigned to a person should not appear in unidentified clusters."""
        person = await create_person(name="Named Person")
        # Face with both person_id AND cluster_id should count as identified, not unidentified
        await create_face_instance(person_id=person.id, cluster_id="cluster_old")

        service = PersonService(db_session)
        result = await service.get_all_people(
            include_identified=True,
            include_unidentified=True,
            include_noise=False,
        )

        assert result.identified_count == 1
        assert result.unidentified_count == 0
        assert result.people[0].name == "Named Person"
        assert result.people[0].face_count == 1

    @pytest.mark.asyncio
    async def test_get_all_people_when_multiple_clusters_then_separate_entries(
        self, db_session, create_face_instance
    ):
        """Each cluster_id should produce a separate unidentified entry."""
        for _ in range(3):
            await create_face_instance(cluster_id="cluster_alpha")
        for _ in range(2):
            await create_face_instance(cluster_id="cluster_beta")

        service = PersonService(db_session)
        result = await service.get_all_people(
            include_identified=False,
            include_unidentified=True,
            include_noise=False,
        )

        assert result.unidentified_count == 2
        face_counts = sorted([p.face_count for p in result.people])
        assert face_counts == [2, 3]

    @pytest.mark.asyncio
    async def test_get_all_people_when_cluster_minus_one_excluded_from_unidentified(
        self, db_session, create_face_instance
    ):
        """Cluster '-1' (noise marker) should not appear in unidentified clusters."""
        await create_face_instance(cluster_id="-1")
        await create_face_instance(cluster_id="cluster_real")

        service = PersonService(db_session)
        result = await service.get_all_people(
            include_identified=False,
            include_unidentified=True,
            include_noise=False,
        )

        assert result.unidentified_count == 1
        assert result.people[0].face_count == 1

    @pytest.mark.asyncio
    async def test_get_all_people_when_large_cluster_then_accurate_count(
        self, db_session, create_face_instance, create_person
    ):
        """Should handle a person with many faces accurately."""
        person = await create_person(name="Popular Person")

        # Create 20 face instances for one person
        for _ in range(20):
            await create_face_instance(person_id=person.id)

        service = PersonService(db_session)
        result = await service.get_all_people(
            include_identified=True,
            include_unidentified=False,
            include_noise=False,
        )

        assert result.identified_count == 1
        assert result.people[0].face_count == 20
        assert result.people[0].name == "Popular Person"

    # ============ Thumbnail URL tests ============

    @pytest.mark.asyncio
    async def test_get_all_people_when_identified_person_has_faces_then_thumbnail_set(
        self, db_session, create_person, create_face_instance
    ):
        """Identified persons with faces should have a thumbnail URL."""
        person = await create_person(name="Photo Person")
        await create_face_instance(person_id=person.id, quality_score=0.9)

        service = PersonService(db_session)
        result = await service.get_all_people(
            include_identified=True,
            include_unidentified=False,
            include_noise=False,
        )

        assert len(result.people) == 1
        thumbnail = result.people[0].thumbnail_url
        assert thumbnail is not None
        assert "/api/v1/images/" in thumbnail
        assert "/thumbnail" in thumbnail

    @pytest.mark.asyncio
    async def test_get_all_people_when_person_has_no_faces_then_no_thumbnail(
        self, db_session, create_person
    ):
        """Identified persons with zero faces should have thumbnail_url=None."""
        await create_person(name="No Photo Person")

        service = PersonService(db_session)
        result = await service.get_all_people(
            include_identified=True,
            include_unidentified=False,
            include_noise=False,
        )

        assert len(result.people) == 1
        assert result.people[0].thumbnail_url is None

    @pytest.mark.asyncio
    async def test_get_all_people_when_unidentified_cluster_then_thumbnail_set(
        self, db_session, create_face_instance
    ):
        """Unidentified clusters should have thumbnail URLs from their faces."""
        await create_face_instance(cluster_id="cluster_with_thumb", quality_score=0.8)

        service = PersonService(db_session)
        result = await service.get_all_people(
            include_identified=False,
            include_unidentified=True,
            include_noise=False,
        )

        assert len(result.people) == 1
        assert result.people[0].thumbnail_url is not None
        assert "/api/v1/images/" in result.people[0].thumbnail_url

    @pytest.mark.asyncio
    async def test_get_all_people_when_noise_faces_exist_then_noise_has_thumbnail(
        self, db_session, create_face_instance
    ):
        """Noise entry should have a thumbnail from one of its faces."""
        await create_face_instance(cluster_id="-1")

        service = PersonService(db_session)
        result = await service.get_all_people(
            include_identified=False,
            include_unidentified=False,
            include_noise=True,
        )

        assert len(result.people) == 1
        assert result.people[0].thumbnail_url is not None
