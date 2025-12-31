"""Unit tests for PersonService."""

import uuid
from unittest.mock import AsyncMock

import pytest

from image_search_service.api.face_schemas import PersonType, UnifiedPersonResponse
from image_search_service.services.person_service import PersonService


class TestPersonService:
    """Test suite for PersonService."""

    @pytest.fixture
    def mock_db(self):
        """Create mock database session."""
        return AsyncMock()

    @pytest.fixture
    def service(self, mock_db):
        """Create PersonService instance with mock db."""
        return PersonService(mock_db)

    # ============ Test _generate_display_name ============

    def test_generate_display_name_regular_cluster(self, service):
        """Should generate 'Unidentified Person N' for regular clusters."""
        assert service._generate_display_name("clu_abc123", 0) == "Unidentified Person 1"
        assert service._generate_display_name("cluster_5", 4) == "Unidentified Person 5"
        assert service._generate_display_name("some_cluster", 9) == "Unidentified Person 10"

    def test_generate_display_name_noise_cluster(self, service):
        """Should return 'Unknown Faces' for noise clusters."""
        assert service._generate_display_name("-1", 0) == "Unknown Faces"
        assert service._generate_display_name("noise", 0) == "Unknown Faces"
        assert service._generate_display_name(None, 0) == "Unknown Faces"

    def test_generate_display_name_sequential_numbering(self, service):
        """Should generate sequential numbers based on index."""
        # Simulating multiple clusters being processed
        names = [service._generate_display_name(f"cluster_{i}", i) for i in range(5)]
        expected = [
            "Unidentified Person 1",
            "Unidentified Person 2",
            "Unidentified Person 3",
            "Unidentified Person 4",
            "Unidentified Person 5",
        ]
        assert names == expected

    # ============ Test filtering behavior ============

    @pytest.mark.asyncio
    async def test_get_all_people_only_identified(self, service, mock_db):
        """Should return only identified people when filters set."""
        # Mock the internal methods
        mock_identified = [
            UnifiedPersonResponse(
                id=str(uuid.uuid4()),
                name="John Doe",
                type=PersonType.IDENTIFIED,
                face_count=10,
            )
        ]
        service._get_identified_people = AsyncMock(return_value=mock_identified)
        service._get_unidentified_clusters = AsyncMock(return_value=[])
        service._get_noise_faces = AsyncMock(return_value=None)

        result = await service.get_all_people(
            include_identified=True,
            include_unidentified=False,
            include_noise=False,
        )

        # Verify only identified method was called
        service._get_identified_people.assert_called_once()
        service._get_unidentified_clusters.assert_not_called()
        service._get_noise_faces.assert_not_called()

        # Verify result
        assert len(result.people) == 1
        assert result.people[0].type == PersonType.IDENTIFIED
        assert result.identified_count == 1
        assert result.unidentified_count == 0
        assert result.noise_count == 0

    @pytest.mark.asyncio
    async def test_get_all_people_only_unidentified(self, service, mock_db):
        """Should return only unidentified clusters when filters set."""
        mock_unidentified = [
            UnifiedPersonResponse(
                id="cluster_123",
                name="Unidentified Person 1",
                type=PersonType.UNIDENTIFIED,
                face_count=15,
            )
        ]
        service._get_identified_people = AsyncMock(return_value=[])
        service._get_unidentified_clusters = AsyncMock(return_value=mock_unidentified)
        service._get_noise_faces = AsyncMock(return_value=None)

        result = await service.get_all_people(
            include_identified=False,
            include_unidentified=True,
            include_noise=False,
        )

        # Verify only unidentified method was called
        service._get_identified_people.assert_not_called()
        service._get_unidentified_clusters.assert_called_once()
        service._get_noise_faces.assert_not_called()

        # Verify result
        assert len(result.people) == 1
        assert result.people[0].type == PersonType.UNIDENTIFIED
        assert result.identified_count == 0
        assert result.unidentified_count == 1
        assert result.noise_count == 0

    @pytest.mark.asyncio
    async def test_get_all_people_include_noise(self, service, mock_db):
        """Should include noise faces when requested."""
        mock_noise = UnifiedPersonResponse(
            id="-1",
            name="Unknown Faces",
            type=PersonType.NOISE,
            face_count=25,
        )
        service._get_identified_people = AsyncMock(return_value=[])
        service._get_unidentified_clusters = AsyncMock(return_value=[])
        service._get_noise_faces = AsyncMock(return_value=mock_noise)

        result = await service.get_all_people(
            include_identified=False,
            include_unidentified=False,
            include_noise=True,
        )

        # Verify noise method was called
        service._get_noise_faces.assert_called_once()

        # Verify result
        assert len(result.people) == 1
        assert result.people[0].type == PersonType.NOISE
        assert result.people[0].name == "Unknown Faces"
        assert result.noise_count == 1

    @pytest.mark.asyncio
    async def test_get_all_people_all_types(self, service, mock_db):
        """Should return all types when all filters enabled."""
        mock_identified = [
            UnifiedPersonResponse(
                id=str(uuid.uuid4()),
                name="Jane Doe",
                type=PersonType.IDENTIFIED,
                face_count=20,
            )
        ]
        mock_unidentified = [
            UnifiedPersonResponse(
                id="cluster_456",
                name="Unidentified Person 1",
                type=PersonType.UNIDENTIFIED,
                face_count=10,
            )
        ]
        mock_noise = UnifiedPersonResponse(
            id="-1",
            name="Unknown Faces",
            type=PersonType.NOISE,
            face_count=5,
        )

        service._get_identified_people = AsyncMock(return_value=mock_identified)
        service._get_unidentified_clusters = AsyncMock(return_value=mock_unidentified)
        service._get_noise_faces = AsyncMock(return_value=mock_noise)

        result = await service.get_all_people(
            include_identified=True,
            include_unidentified=True,
            include_noise=True,
        )

        # Verify all methods were called
        service._get_identified_people.assert_called_once()
        service._get_unidentified_clusters.assert_called_once()
        service._get_noise_faces.assert_called_once()

        # Verify result contains all types
        assert len(result.people) == 3
        assert result.total == 3
        assert result.identified_count == 1
        assert result.unidentified_count == 1
        assert result.noise_count == 1

    @pytest.mark.asyncio
    async def test_get_all_people_no_noise_when_zero(self, service, mock_db):
        """Should not include noise entry when count is zero."""
        service._get_identified_people = AsyncMock(return_value=[])
        service._get_unidentified_clusters = AsyncMock(return_value=[])
        service._get_noise_faces = AsyncMock(return_value=None)  # Returns None when no noise

        result = await service.get_all_people(
            include_identified=True,
            include_unidentified=True,
            include_noise=True,
        )

        # Verify result does not include noise
        assert len(result.people) == 0
        assert result.noise_count == 0

    # ============ Test sorting behavior ============

    @pytest.mark.asyncio
    async def test_get_all_people_sort_by_face_count_desc(self, service, mock_db):
        """Should sort by face count descending by default."""
        mock_people = [
            UnifiedPersonResponse(
                id="1", name="Person A", type=PersonType.IDENTIFIED, face_count=5
            ),
            UnifiedPersonResponse(
                id="2", name="Person B", type=PersonType.IDENTIFIED, face_count=15
            ),
            UnifiedPersonResponse(
                id="3", name="Person C", type=PersonType.IDENTIFIED, face_count=10
            ),
        ]

        service._get_identified_people = AsyncMock(return_value=mock_people)
        service._get_unidentified_clusters = AsyncMock(return_value=[])
        service._get_noise_faces = AsyncMock(return_value=None)

        result = await service.get_all_people(sort_by="face_count", sort_order="desc")

        # Verify sorting
        face_counts = [p.face_count for p in result.people]
        assert face_counts == [15, 10, 5]

    @pytest.mark.asyncio
    async def test_get_all_people_sort_by_face_count_asc(self, service, mock_db):
        """Should sort by face count ascending when requested."""
        mock_people = [
            UnifiedPersonResponse(
                id="1", name="Person A", type=PersonType.IDENTIFIED, face_count=5
            ),
            UnifiedPersonResponse(
                id="2", name="Person B", type=PersonType.IDENTIFIED, face_count=15
            ),
            UnifiedPersonResponse(
                id="3", name="Person C", type=PersonType.IDENTIFIED, face_count=10
            ),
        ]

        service._get_identified_people = AsyncMock(return_value=mock_people)
        service._get_unidentified_clusters = AsyncMock(return_value=[])
        service._get_noise_faces = AsyncMock(return_value=None)

        result = await service.get_all_people(sort_by="face_count", sort_order="asc")

        # Verify sorting
        face_counts = [p.face_count for p in result.people]
        assert face_counts == [5, 10, 15]

    @pytest.mark.asyncio
    async def test_get_all_people_sort_by_name_asc(self, service, mock_db):
        """Should sort by name ascending when requested."""
        mock_people = [
            UnifiedPersonResponse(
                id="1", name="Charlie", type=PersonType.IDENTIFIED, face_count=5
            ),
            UnifiedPersonResponse(
                id="2", name="Alice", type=PersonType.IDENTIFIED, face_count=15
            ),
            UnifiedPersonResponse(
                id="3", name="Bob", type=PersonType.IDENTIFIED, face_count=10
            ),
        ]

        service._get_identified_people = AsyncMock(return_value=mock_people)
        service._get_unidentified_clusters = AsyncMock(return_value=[])
        service._get_noise_faces = AsyncMock(return_value=None)

        result = await service.get_all_people(sort_by="name", sort_order="asc")

        # Verify sorting (case-insensitive)
        names = [p.name for p in result.people]
        assert names == ["Alice", "Bob", "Charlie"]

    @pytest.mark.asyncio
    async def test_get_all_people_sort_by_name_desc(self, service, mock_db):
        """Should sort by name descending when requested."""
        mock_people = [
            UnifiedPersonResponse(
                id="1", name="Charlie", type=PersonType.IDENTIFIED, face_count=5
            ),
            UnifiedPersonResponse(
                id="2", name="Alice", type=PersonType.IDENTIFIED, face_count=15
            ),
            UnifiedPersonResponse(
                id="3", name="Bob", type=PersonType.IDENTIFIED, face_count=10
            ),
        ]

        service._get_identified_people = AsyncMock(return_value=mock_people)
        service._get_unidentified_clusters = AsyncMock(return_value=[])
        service._get_noise_faces = AsyncMock(return_value=None)

        result = await service.get_all_people(sort_by="name", sort_order="desc")

        # Verify sorting
        names = [p.name for p in result.people]
        assert names == ["Charlie", "Bob", "Alice"]

    @pytest.mark.asyncio
    async def test_get_all_people_sort_mixed_types(self, service, mock_db):
        """Should sort correctly across different person types."""
        mock_identified = [
            UnifiedPersonResponse(
                id="1", name="Alice", type=PersonType.IDENTIFIED, face_count=10
            ),
        ]
        mock_unidentified = [
            UnifiedPersonResponse(
                id="cluster_1",
                name="Unidentified Person 1",
                type=PersonType.UNIDENTIFIED,
                face_count=20,
            ),
        ]
        mock_noise = UnifiedPersonResponse(
            id="-1", name="Unknown Faces", type=PersonType.NOISE, face_count=5
        )

        service._get_identified_people = AsyncMock(return_value=mock_identified)
        service._get_unidentified_clusters = AsyncMock(return_value=mock_unidentified)
        service._get_noise_faces = AsyncMock(return_value=mock_noise)

        result = await service.get_all_people(
            include_identified=True,
            include_unidentified=True,
            include_noise=True,
            sort_by="face_count",
            sort_order="desc",
        )

        # Verify sorting by face count
        face_counts = [p.face_count for p in result.people]
        assert face_counts == [20, 10, 5]

        # Verify types are mixed
        types = [p.type for p in result.people]
        assert PersonType.IDENTIFIED in types
        assert PersonType.UNIDENTIFIED in types
        assert PersonType.NOISE in types

    # ============ Test count accuracy ============

    @pytest.mark.asyncio
    async def test_get_all_people_counts_are_accurate(self, service, mock_db):
        """Should return accurate counts for each type."""
        mock_identified = [
            UnifiedPersonResponse(
                id=str(uuid.uuid4()),
                name=f"Person {i}",
                type=PersonType.IDENTIFIED,
                face_count=10,
            )
            for i in range(3)
        ]
        mock_unidentified = [
            UnifiedPersonResponse(
                id=f"cluster_{i}",
                name=f"Unidentified Person {i+1}",
                type=PersonType.UNIDENTIFIED,
                face_count=5,
            )
            for i in range(2)
        ]
        mock_noise = UnifiedPersonResponse(
            id="-1", name="Unknown Faces", type=PersonType.NOISE, face_count=8
        )

        service._get_identified_people = AsyncMock(return_value=mock_identified)
        service._get_unidentified_clusters = AsyncMock(return_value=mock_unidentified)
        service._get_noise_faces = AsyncMock(return_value=mock_noise)

        result = await service.get_all_people(
            include_identified=True,
            include_unidentified=True,
            include_noise=True,
        )

        # Verify counts
        assert result.total == 6  # 3 + 2 + 1
        assert result.identified_count == 3
        assert result.unidentified_count == 2
        assert result.noise_count == 1

    @pytest.mark.asyncio
    async def test_get_all_people_empty_result(self, service, mock_db):
        """Should handle empty results gracefully."""
        service._get_identified_people = AsyncMock(return_value=[])
        service._get_unidentified_clusters = AsyncMock(return_value=[])
        service._get_noise_faces = AsyncMock(return_value=None)

        result = await service.get_all_people()

        assert result.people == []
        assert result.total == 0
        assert result.identified_count == 0
        assert result.unidentified_count == 0
        assert result.noise_count == 0
