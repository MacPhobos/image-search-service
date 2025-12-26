"""Tests for face suggestions API endpoint."""

import uuid
from unittest.mock import MagicMock, patch

import pytest
from qdrant_client.models import ScoredPoint


# Fixtures for face suggestion tests
@pytest.fixture
async def mock_image_asset(db_session):
    """Create a mock ImageAsset in the database."""
    from image_search_service.db.models import ImageAsset, TrainingStatus

    asset = ImageAsset(
        path="/test/images/photo.jpg",
        training_status=TrainingStatus.PENDING.value,
        width=640,
        height=480,
        file_size=102400,
        mime_type="image/jpeg",
    )
    db_session.add(asset)
    await db_session.commit()
    await db_session.refresh(asset)
    return asset


@pytest.fixture
async def mock_face_instance(db_session, mock_image_asset):
    """Create a mock FaceInstance in the database."""
    from image_search_service.db.models import FaceInstance

    face = FaceInstance(
        id=uuid.uuid4(),
        asset_id=mock_image_asset.id,
        bbox_x=100,
        bbox_y=150,
        bbox_w=80,
        bbox_h=80,
        detection_confidence=0.95,
        quality_score=0.75,
        qdrant_point_id=uuid.uuid4(),
    )
    db_session.add(face)
    await db_session.commit()
    await db_session.refresh(face)
    return face


@pytest.fixture
async def mock_person(db_session):
    """Create a mock Person in the database."""
    from image_search_service.db.models import Person, PersonStatus

    person = Person(
        id=uuid.uuid4(),
        name="Test Person",
        status=PersonStatus.ACTIVE.value,
    )
    db_session.add(person)
    await db_session.commit()
    await db_session.refresh(person)
    return person


@pytest.fixture
async def mock_inactive_person(db_session):
    """Create a mock inactive Person in the database."""
    from image_search_service.db.models import Person, PersonStatus

    person = Person(
        id=uuid.uuid4(),
        name="Inactive Person",
        status=PersonStatus.MERGED.value,
    )
    db_session.add(person)
    await db_session.commit()
    await db_session.refresh(person)
    return person


class TestGetFaceSuggestions:
    """Tests for GET /api/v1/faces/faces/{face_id}/suggestions endpoint."""

    @pytest.mark.asyncio
    async def test_get_face_suggestions_success(
        self, test_client, db_session, mock_face_instance, mock_person
    ):
        """Test successful face suggestions retrieval with valid face ID and similar persons."""
        # Mock Qdrant client methods
        with patch(
            "image_search_service.vector.face_qdrant.get_face_qdrant_client"
        ) as mock_qdrant:
            # Setup mock embedding
            mock_embedding = [0.1] * 512  # 512-dim embedding vector

            # Setup mock Qdrant client
            qdrant_client = MagicMock()
            qdrant_client.get_embedding_by_point_id.return_value = mock_embedding

            # Setup mock similarity search results (prototypes)
            scored_point = ScoredPoint(
                id=str(uuid.uuid4()),
                score=0.85,
                payload={"person_id": str(mock_person.id)},
                version=1,
                vector=None,
            )
            qdrant_client.search_against_prototypes.return_value = [scored_point]

            mock_qdrant.return_value = qdrant_client

            # Make request
            response = await test_client.get(
                f"/api/v1/faces/faces/{mock_face_instance.id}/suggestions",
                params={"min_confidence": 0.7, "limit": 5},
            )

            # Verify response
            assert response.status_code == 200
            data = response.json()

            # Check response structure
            assert "faceId" in data
            assert "suggestions" in data
            assert "thresholdUsed" in data

            # Verify data values
            assert data["faceId"] == str(mock_face_instance.id)
            assert data["thresholdUsed"] == 0.7
            assert isinstance(data["suggestions"], list)
            assert len(data["suggestions"]) == 1

            # Check suggestion structure
            suggestion = data["suggestions"][0]
            assert "personId" in suggestion
            assert "personName" in suggestion
            assert "confidence" in suggestion
            assert suggestion["personId"] == str(mock_person.id)
            assert suggestion["personName"] == mock_person.name
            assert suggestion["confidence"] == 0.85

            # Verify Qdrant methods were called correctly
            qdrant_client.get_embedding_by_point_id.assert_called_once_with(
                mock_face_instance.qdrant_point_id
            )
            qdrant_client.search_against_prototypes.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_face_suggestions_face_not_found(self, test_client, db_session):
        """Test 404 when face ID doesn't exist in database."""
        fake_face_id = uuid.uuid4()

        response = await test_client.get(
            f"/api/v1/faces/faces/{fake_face_id}/suggestions"
        )

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()
        assert str(fake_face_id) in data["detail"]

    @pytest.mark.asyncio
    async def test_get_face_suggestions_invalid_uuid(self, test_client):
        """Test 422 validation error for invalid UUID format."""
        invalid_uuid = "not-a-valid-uuid"

        response = await test_client.get(
            f"/api/v1/faces/faces/{invalid_uuid}/suggestions"
        )

        # FastAPI validation should return 422 for invalid UUID
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_get_face_suggestions_embedding_not_found(
        self, test_client, db_session, mock_face_instance
    ):
        """Test 404 when face exists but embedding is missing in Qdrant."""
        with patch(
            "image_search_service.vector.face_qdrant.get_face_qdrant_client"
        ) as mock_qdrant:
            # Setup mock to return None for embedding (not found)
            qdrant_client = MagicMock()
            qdrant_client.get_embedding_by_point_id.return_value = None
            mock_qdrant.return_value = qdrant_client

            response = await test_client.get(
                f"/api/v1/faces/faces/{mock_face_instance.id}/suggestions"
            )

            assert response.status_code == 404
            data = response.json()
            assert "embedding not found" in data["detail"].lower()

    @pytest.mark.asyncio
    async def test_get_face_suggestions_empty_results(
        self, test_client, db_session, mock_face_instance
    ):
        """Test empty suggestions list when no matches above threshold (not an error)."""
        with patch(
            "image_search_service.vector.face_qdrant.get_face_qdrant_client"
        ) as mock_qdrant:
            # Setup mock embedding
            mock_embedding = [0.1] * 512

            # Setup mock with no results above threshold
            qdrant_client = MagicMock()
            qdrant_client.get_embedding_by_point_id.return_value = mock_embedding
            qdrant_client.search_against_prototypes.return_value = []  # No matches

            mock_qdrant.return_value = qdrant_client

            response = await test_client.get(
                f"/api/v1/faces/faces/{mock_face_instance.id}/suggestions",
                params={"min_confidence": 0.9},  # High threshold
            )

            assert response.status_code == 200
            data = response.json()
            assert data["suggestions"] == []
            assert data["thresholdUsed"] == 0.9

    @pytest.mark.asyncio
    async def test_get_face_suggestions_respects_min_confidence(
        self, test_client, db_session, mock_face_instance, mock_person
    ):
        """Test that min_confidence query param filters out low-confidence results."""
        with patch(
            "image_search_service.vector.face_qdrant.get_face_qdrant_client"
        ) as mock_qdrant:
            mock_embedding = [0.1] * 512

            qdrant_client = MagicMock()
            qdrant_client.get_embedding_by_point_id.return_value = mock_embedding

            # Return high confidence result
            high_confidence_point = ScoredPoint(
                id=str(uuid.uuid4()),
                score=0.92,
                payload={"person_id": str(mock_person.id)},
                version=1,
                vector=None,
            )
            qdrant_client.search_against_prototypes.return_value = [
                high_confidence_point
            ]

            mock_qdrant.return_value = qdrant_client

            # Request with high threshold
            response = await test_client.get(
                f"/api/v1/faces/faces/{mock_face_instance.id}/suggestions",
                params={"min_confidence": 0.9, "limit": 5},
            )

            assert response.status_code == 200
            data = response.json()
            assert len(data["suggestions"]) == 1
            assert data["suggestions"][0]["confidence"] >= 0.9

            # Verify threshold was passed to Qdrant
            call_kwargs = qdrant_client.search_against_prototypes.call_args[1]
            assert call_kwargs["score_threshold"] == 0.9

    @pytest.mark.asyncio
    async def test_get_face_suggestions_respects_limit(
        self, test_client, db_session, mock_face_instance
    ):
        """Test that limit query param restricts number of results returned."""
        # Create multiple persons
        from image_search_service.db.models import Person, PersonStatus

        persons = []
        for i in range(5):
            person = Person(
                id=uuid.uuid4(),
                name=f"Person {i}",
                status=PersonStatus.ACTIVE.value,
            )
            db_session.add(person)
            persons.append(person)

        await db_session.commit()

        with patch(
            "image_search_service.vector.face_qdrant.get_face_qdrant_client"
        ) as mock_qdrant:
            mock_embedding = [0.1] * 512

            qdrant_client = MagicMock()
            qdrant_client.get_embedding_by_point_id.return_value = mock_embedding

            # Return multiple results (one per person)
            scored_points = [
                ScoredPoint(
                    id=str(uuid.uuid4()),
                    score=0.9 - (i * 0.05),  # Decreasing confidence
                    payload={"person_id": str(person.id)},
                    version=1,
                    vector=None,
                )
                for i, person in enumerate(persons)
            ]
            qdrant_client.search_against_prototypes.return_value = scored_points

            mock_qdrant.return_value = qdrant_client

            # Request with limit=2
            response = await test_client.get(
                f"/api/v1/faces/faces/{mock_face_instance.id}/suggestions",
                params={"min_confidence": 0.7, "limit": 2},
            )

            assert response.status_code == 200
            data = response.json()

            # Should only return 2 results despite having 5 matches
            assert len(data["suggestions"]) == 2

            # Results should be sorted by confidence (highest first)
            confidences = [s["confidence"] for s in data["suggestions"]]
            assert confidences == sorted(confidences, reverse=True)

    @pytest.mark.asyncio
    async def test_get_face_suggestions_filters_inactive_persons(
        self, test_client, db_session, mock_face_instance, mock_person, mock_inactive_person
    ):
        """Test that merged/hidden persons are filtered out from suggestions."""
        with patch(
            "image_search_service.vector.face_qdrant.get_face_qdrant_client"
        ) as mock_qdrant:
            mock_embedding = [0.1] * 512

            qdrant_client = MagicMock()
            qdrant_client.get_embedding_by_point_id.return_value = mock_embedding

            # Return both active and inactive person
            scored_points = [
                ScoredPoint(
                    id=str(uuid.uuid4()),
                    score=0.85,
                    payload={"person_id": str(mock_person.id)},
                    version=1,
                    vector=None,
                ),
                ScoredPoint(
                    id=str(uuid.uuid4()),
                    score=0.82,
                    payload={"person_id": str(mock_inactive_person.id)},
                    version=1,
                    vector=None,
                ),
            ]
            qdrant_client.search_against_prototypes.return_value = scored_points

            mock_qdrant.return_value = qdrant_client

            response = await test_client.get(
                f"/api/v1/faces/faces/{mock_face_instance.id}/suggestions"
            )

            assert response.status_code == 200
            data = response.json()

            # Should only return active person
            assert len(data["suggestions"]) == 1
            assert data["suggestions"][0]["personId"] == str(mock_person.id)
            assert data["suggestions"][0]["personName"] == "Test Person"

    @pytest.mark.asyncio
    async def test_get_face_suggestions_deduplicates_by_person(
        self, test_client, db_session, mock_face_instance, mock_person
    ):
        """Test that multiple prototypes for same person are deduplicated with highest confidence kept."""
        with patch(
            "image_search_service.vector.face_qdrant.get_face_qdrant_client"
        ) as mock_qdrant:
            mock_embedding = [0.1] * 512

            qdrant_client = MagicMock()
            qdrant_client.get_embedding_by_point_id.return_value = mock_embedding

            # Return multiple prototypes for same person with different scores
            scored_points = [
                ScoredPoint(
                    id=str(uuid.uuid4()),
                    score=0.75,  # Lower confidence
                    payload={"person_id": str(mock_person.id)},
                    version=1,
                    vector=None,
                ),
                ScoredPoint(
                    id=str(uuid.uuid4()),
                    score=0.92,  # Higher confidence (should be kept)
                    payload={"person_id": str(mock_person.id)},
                    version=1,
                    vector=None,
                ),
                ScoredPoint(
                    id=str(uuid.uuid4()),
                    score=0.80,  # Medium confidence
                    payload={"person_id": str(mock_person.id)},
                    version=1,
                    vector=None,
                ),
            ]
            qdrant_client.search_against_prototypes.return_value = scored_points

            mock_qdrant.return_value = qdrant_client

            response = await test_client.get(
                f"/api/v1/faces/faces/{mock_face_instance.id}/suggestions"
            )

            assert response.status_code == 200
            data = response.json()

            # Should return only one suggestion for the person
            assert len(data["suggestions"]) == 1

            # Should have the highest confidence score (0.92)
            assert data["suggestions"][0]["confidence"] == 0.92
            assert data["suggestions"][0]["personId"] == str(mock_person.id)

    @pytest.mark.asyncio
    async def test_get_face_suggestions_handles_missing_person_gracefully(
        self, test_client, db_session, mock_face_instance
    ):
        """Test that suggestions with person_id not in database are skipped gracefully."""
        with patch(
            "image_search_service.vector.face_qdrant.get_face_qdrant_client"
        ) as mock_qdrant:
            mock_embedding = [0.1] * 512

            qdrant_client = MagicMock()
            qdrant_client.get_embedding_by_point_id.return_value = mock_embedding

            # Return prototype for non-existent person
            fake_person_id = uuid.uuid4()
            scored_point = ScoredPoint(
                id=str(uuid.uuid4()),
                score=0.85,
                payload={"person_id": str(fake_person_id)},
                version=1,
                vector=None,
            )
            qdrant_client.search_against_prototypes.return_value = [scored_point]

            mock_qdrant.return_value = qdrant_client

            response = await test_client.get(
                f"/api/v1/faces/faces/{mock_face_instance.id}/suggestions"
            )

            # Should succeed but with empty suggestions
            assert response.status_code == 200
            data = response.json()
            assert data["suggestions"] == []

    @pytest.mark.asyncio
    async def test_get_face_suggestions_default_params(
        self, test_client, db_session, mock_face_instance, mock_person
    ):
        """Test endpoint uses correct default values when params not provided."""
        with patch(
            "image_search_service.vector.face_qdrant.get_face_qdrant_client"
        ) as mock_qdrant:
            mock_embedding = [0.1] * 512

            qdrant_client = MagicMock()
            qdrant_client.get_embedding_by_point_id.return_value = mock_embedding

            scored_point = ScoredPoint(
                id=str(uuid.uuid4()),
                score=0.85,
                payload={"person_id": str(mock_person.id)},
                version=1,
                vector=None,
            )
            qdrant_client.search_against_prototypes.return_value = [scored_point]

            mock_qdrant.return_value = qdrant_client

            # Request without params
            response = await test_client.get(
                f"/api/v1/faces/faces/{mock_face_instance.id}/suggestions"
            )

            assert response.status_code == 200
            data = response.json()

            # Default threshold should be 0.7
            assert data["thresholdUsed"] == 0.7

            # Verify default params passed to Qdrant
            call_kwargs = qdrant_client.search_against_prototypes.call_args[1]
            assert call_kwargs["score_threshold"] == 0.7
            assert call_kwargs["limit"] == 15  # limit * 3 for deduplication

    @pytest.mark.asyncio
    async def test_get_face_suggestions_handles_qdrant_error(
        self, test_client, db_session, mock_face_instance
    ):
        """Test 500 error when Qdrant search fails."""
        with patch(
            "image_search_service.vector.face_qdrant.get_face_qdrant_client"
        ) as mock_qdrant:
            mock_embedding = [0.1] * 512

            qdrant_client = MagicMock()
            qdrant_client.get_embedding_by_point_id.return_value = mock_embedding

            # Simulate Qdrant search failure
            qdrant_client.search_against_prototypes.side_effect = Exception(
                "Qdrant connection timeout"
            )

            mock_qdrant.return_value = qdrant_client

            response = await test_client.get(
                f"/api/v1/faces/faces/{mock_face_instance.id}/suggestions"
            )

            assert response.status_code == 500
            data = response.json()
            assert "failed to search" in data["detail"].lower()

    @pytest.mark.asyncio
    async def test_get_face_suggestions_validation_min_confidence(self, test_client, mock_face_instance):
        """Test validation for min_confidence parameter boundaries."""
        # Test below minimum (< 0.0)
        response = await test_client.get(
            f"/api/v1/faces/faces/{mock_face_instance.id}/suggestions",
            params={"min_confidence": -0.1},
        )
        assert response.status_code == 422

        # Test above maximum (> 1.0)
        response = await test_client.get(
            f"/api/v1/faces/faces/{mock_face_instance.id}/suggestions",
            params={"min_confidence": 1.5},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_get_face_suggestions_validation_limit(self, test_client, mock_face_instance):
        """Test validation for limit parameter boundaries."""
        # Test below minimum (< 1)
        response = await test_client.get(
            f"/api/v1/faces/faces/{mock_face_instance.id}/suggestions",
            params={"limit": 0},
        )
        assert response.status_code == 422

        # Test above maximum (> 10)
        response = await test_client.get(
            f"/api/v1/faces/faces/{mock_face_instance.id}/suggestions",
            params={"limit": 11},
        )
        assert response.status_code == 422
