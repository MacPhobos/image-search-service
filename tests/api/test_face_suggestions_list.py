"""Tests for face suggestions list API endpoint with group-based pagination."""

import uuid
from datetime import UTC, datetime

import pytest


# Fixtures for creating test data
@pytest.fixture
async def setup_test_data(db_session):
    """Create comprehensive test data for group-based pagination tests."""
    from image_search_service.db.models import (
        FaceInstance,
        FaceSuggestion,
        FaceSuggestionStatus,
        ImageAsset,
        Person,
        PersonStatus,
        TrainingStatus,
    )

    # Create image assets
    assets = []
    for i in range(10):
        asset = ImageAsset(
            path=f"/test/images/photo{i}.jpg",
            training_status=TrainingStatus.PENDING.value,
            width=640,
            height=480,
            file_size=102400,
            mime_type="image/jpeg",
        )
        db_session.add(asset)
        assets.append(asset)

    # Create persons
    persons = []
    for i in range(5):
        person = Person(
            id=uuid.uuid4(),
            name=f"Person {i}",
            status=PersonStatus.ACTIVE,
        )
        db_session.add(person)
        persons.append(person)

    await db_session.commit()

    # Create face instances and suggestions
    suggestions = []
    suggestion_id_counter = 1

    # Person 0: 5 suggestions (confidence 0.95-0.91)
    for i in range(5):
        face = FaceInstance(
            id=uuid.uuid4(),
            asset_id=assets[i].id,
            bbox_x=100,
            bbox_y=150,
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.95,
            quality_score=0.75,
            qdrant_point_id=uuid.uuid4(),
        )
        db_session.add(face)
        await db_session.flush()

        suggestion = FaceSuggestion(
            id=suggestion_id_counter,
            face_instance_id=face.id,
            suggested_person_id=persons[0].id,
            confidence=0.95 - (i * 0.01),
            source_face_id=uuid.uuid4(),
            status=FaceSuggestionStatus.PENDING.value,
            created_at=datetime.now(UTC),
        )
        db_session.add(suggestion)
        suggestions.append(suggestion)
        suggestion_id_counter += 1

    # Person 1: 3 suggestions (confidence 0.90-0.88)
    for i in range(3):
        face = FaceInstance(
            id=uuid.uuid4(),
            asset_id=assets[5 + i].id,
            bbox_x=100,
            bbox_y=150,
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.95,
            quality_score=0.75,
            qdrant_point_id=uuid.uuid4(),
        )
        db_session.add(face)
        await db_session.flush()

        suggestion = FaceSuggestion(
            id=suggestion_id_counter,
            face_instance_id=face.id,
            suggested_person_id=persons[1].id,
            confidence=0.90 - (i * 0.01),
            source_face_id=uuid.uuid4(),
            status=FaceSuggestionStatus.PENDING.value,
            created_at=datetime.now(UTC),
        )
        db_session.add(suggestion)
        suggestions.append(suggestion)
        suggestion_id_counter += 1

    # Person 2: 2 suggestions (confidence 0.85, 0.84)
    for i in range(2):
        face = FaceInstance(
            id=uuid.uuid4(),
            asset_id=assets[8 + i].id,
            bbox_x=100,
            bbox_y=150,
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.95,
            quality_score=0.75,
            qdrant_point_id=uuid.uuid4(),
        )
        db_session.add(face)
        await db_session.flush()

        suggestion = FaceSuggestion(
            id=suggestion_id_counter,
            face_instance_id=face.id,
            suggested_person_id=persons[2].id,
            confidence=0.85 - (i * 0.01),
            source_face_id=uuid.uuid4(),
            status=FaceSuggestionStatus.PENDING.value,
            created_at=datetime.now(UTC),
        )
        db_session.add(suggestion)
        suggestions.append(suggestion)
        suggestion_id_counter += 1

    await db_session.commit()

    return {
        "persons": persons,
        "assets": assets,
        "suggestions": suggestions,
    }


@pytest.fixture
async def setup_config_keys(db_session):
    """Create configuration keys in database."""
    from image_search_service.db.models import ConfigDataType, SystemConfig

    configs = [
        SystemConfig(
            key="face_suggestion_groups_per_page",
            value="10",
            data_type=ConfigDataType.INT.value,
            description="Number of person groups per page",
            min_value="1",
            max_value="50",
            category="face_suggestions",
        ),
        SystemConfig(
            key="face_suggestion_items_per_group",
            value="20",
            data_type=ConfigDataType.INT.value,
            description="Number of suggestions per group",
            min_value="1",
            max_value="50",
            category="face_suggestions",
        ),
    ]

    for config in configs:
        db_session.add(config)

    await db_session.commit()


class TestListSuggestionsGrouped:
    """Tests for GET /api/v1/faces/suggestions with group-based pagination."""

    @pytest.mark.asyncio
    async def test_list_suggestions_grouped_default_behavior(
        self, test_client, db_session, setup_test_data, setup_config_keys
    ):
        """Test grouped pagination is default behavior."""
        response = await test_client.get("/api/v1/faces/suggestions")

        assert response.status_code == 200
        data = response.json()

        # Should have grouped response structure
        assert "groups" in data
        assert "totalGroups" in data
        assert "totalSuggestions" in data
        assert "page" in data
        assert "groupsPerPage" in data
        assert "suggestionsPerGroup" in data

        # Should have 3 groups (Person 0, 1, 2)
        assert len(data["groups"]) == 3
        assert data["totalGroups"] == 3
        assert data["totalSuggestions"] == 10

    @pytest.mark.asyncio
    async def test_list_suggestions_grouped_ordering_by_confidence(
        self, test_client, db_session, setup_test_data, setup_config_keys
    ):
        """Test groups are ordered by max confidence descending."""
        response = await test_client.get("/api/v1/faces/suggestions")

        assert response.status_code == 200
        data = response.json()

        # Groups should be ordered by max confidence
        max_confidences = [group["maxConfidence"] for group in data["groups"]]
        assert max_confidences == sorted(max_confidences, reverse=True)

        # First group should be Person 0 (confidence 0.95)
        assert data["groups"][0]["maxConfidence"] == 0.95
        assert data["groups"][0]["personName"] == "Person 0"

        # Second group should be Person 1 (confidence 0.90)
        assert data["groups"][1]["maxConfidence"] == 0.90
        assert data["groups"][1]["personName"] == "Person 1"

        # Third group should be Person 2 (confidence 0.85)
        assert data["groups"][2]["maxConfidence"] == 0.85
        assert data["groups"][2]["personName"] == "Person 2"

    @pytest.mark.asyncio
    async def test_list_suggestions_grouped_limits_suggestions_per_group(
        self, test_client, db_session, setup_test_data, setup_config_keys
    ):
        """Test suggestionsPerGroup parameter limits items in each group."""
        response = await test_client.get(
            "/api/v1/faces/suggestions",
            params={"suggestionsPerGroup": 2},
        )

        assert response.status_code == 200
        data = response.json()

        # Each group should have at most 2 suggestions
        for group in data["groups"]:
            assert len(group["suggestions"]) <= 2

        # Person 0 has 5 total but should show only 2
        person_0_group = next(g for g in data["groups"] if g["personName"] == "Person 0")
        assert person_0_group["suggestionCount"] == 5
        assert len(person_0_group["suggestions"]) == 2

        # Suggestions should be highest confidence
        assert person_0_group["suggestions"][0]["confidence"] == 0.95
        assert person_0_group["suggestions"][1]["confidence"] == 0.94

    @pytest.mark.asyncio
    async def test_list_suggestions_grouped_pagination(
        self, test_client, db_session, setup_test_data, setup_config_keys
    ):
        """Test pagination with groupsPerPage parameter."""
        # Page 1: Get first 2 groups
        response = await test_client.get(
            "/api/v1/faces/suggestions",
            params={"groupsPerPage": 2, "page": 1},
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data["groups"]) == 2
        assert data["totalGroups"] == 3
        assert data["page"] == 1
        assert data["groupsPerPage"] == 2

        # First page should have Person 0 and Person 1
        assert data["groups"][0]["personName"] == "Person 0"
        assert data["groups"][1]["personName"] == "Person 1"

        # Page 2: Get remaining group
        response = await test_client.get(
            "/api/v1/faces/suggestions",
            params={"groupsPerPage": 2, "page": 2},
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data["groups"]) == 1
        assert data["page"] == 2

        # Second page should have Person 2
        assert data["groups"][0]["personName"] == "Person 2"

    @pytest.mark.asyncio
    async def test_list_suggestions_grouped_empty_page(
        self, test_client, db_session, setup_test_data, setup_config_keys
    ):
        """Test requesting page beyond available data returns empty groups."""
        response = await test_client.get(
            "/api/v1/faces/suggestions",
            params={"page": 10},
        )

        assert response.status_code == 200
        data = response.json()

        assert data["groups"] == []
        assert data["totalGroups"] == 3  # Still reports total
        assert data["page"] == 10

    @pytest.mark.asyncio
    async def test_list_suggestions_grouped_filter_by_status(
        self, test_client, db_session, setup_test_data, setup_config_keys
    ):
        """Test filtering by status parameter."""
        # Mark one suggestion as accepted
        from image_search_service.db.models import FaceSuggestion, FaceSuggestionStatus

        suggestion = await db_session.get(FaceSuggestion, 1)
        suggestion.status = FaceSuggestionStatus.ACCEPTED.value
        await db_session.commit()

        # Query for pending only
        response = await test_client.get(
            "/api/v1/faces/suggestions",
            params={"status": "pending"},
        )

        assert response.status_code == 200
        data = response.json()

        # Should still have 3 groups but fewer total suggestions
        assert data["totalSuggestions"] == 9  # One was accepted

        # Query for accepted only
        response = await test_client.get(
            "/api/v1/faces/suggestions",
            params={"status": "accepted"},
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data["groups"]) == 1  # Only Person 0 has accepted
        assert data["totalSuggestions"] == 1

    @pytest.mark.asyncio
    async def test_list_suggestions_grouped_filter_by_person(
        self, test_client, db_session, setup_test_data, setup_config_keys
    ):
        """Test filtering by personId parameter."""
        persons = setup_test_data["persons"]

        response = await test_client.get(
            "/api/v1/faces/suggestions",
            params={"personId": str(persons[1].id)},
        )

        assert response.status_code == 200
        data = response.json()

        # Should only have Person 1's group
        assert len(data["groups"]) == 1
        assert data["totalGroups"] == 1
        assert data["groups"][0]["personName"] == "Person 1"
        assert data["groups"][0]["suggestionCount"] == 3

    @pytest.mark.asyncio
    async def test_list_suggestions_grouped_suggestion_metadata(
        self, test_client, db_session, setup_test_data, setup_config_keys
    ):
        """Test that suggestion objects contain all required metadata."""
        response = await test_client.get(
            "/api/v1/faces/suggestions",
            params={"suggestionsPerGroup": 1},
        )

        assert response.status_code == 200
        data = response.json()

        # Check first suggestion has all fields
        suggestion = data["groups"][0]["suggestions"][0]

        assert "id" in suggestion
        assert "faceInstanceId" in suggestion
        assert "suggestedPersonId" in suggestion
        assert "confidence" in suggestion
        assert "sourceFaceId" in suggestion
        assert "status" in suggestion
        assert "createdAt" in suggestion
        assert "faceThumbnailUrl" in suggestion
        assert "personName" in suggestion
        assert "fullImageUrl" in suggestion
        assert "bboxX" in suggestion
        assert "bboxY" in suggestion
        assert "bboxW" in suggestion
        assert "bboxH" in suggestion
        assert "detectionConfidence" in suggestion
        assert "qualityScore" in suggestion

    @pytest.mark.asyncio
    async def test_list_suggestions_flat_mode_legacy(
        self, test_client, db_session, setup_test_data, setup_config_keys
    ):
        """Test legacy flat pagination mode when grouped=false."""
        response = await test_client.get(
            "/api/v1/faces/suggestions",
            params={"grouped": False, "pageSize": 5},
        )

        assert response.status_code == 200
        data = response.json()

        # Should have flat response structure
        assert "items" in data
        assert "total" in data
        assert "page" in data
        assert "pageSize" in data

        # Should NOT have grouped fields
        assert "groups" not in data
        assert "totalGroups" not in data

        # Should have 5 items (pageSize)
        assert len(data["items"]) == 5
        assert data["total"] == 10

    @pytest.mark.asyncio
    async def test_list_suggestions_uses_config_defaults(
        self, test_client, db_session, setup_test_data, setup_config_keys
    ):
        """Test that config defaults are used when params not provided."""
        response = await test_client.get("/api/v1/faces/suggestions")

        assert response.status_code == 200
        data = response.json()

        # Should use config defaults
        assert data["groupsPerPage"] == 10  # From config
        assert data["suggestionsPerGroup"] == 20  # From config

    @pytest.mark.asyncio
    async def test_list_suggestions_param_overrides_config(
        self, test_client, db_session, setup_test_data, setup_config_keys
    ):
        """Test that query params override config defaults."""
        response = await test_client.get(
            "/api/v1/faces/suggestions",
            params={"groupsPerPage": 5, "suggestionsPerGroup": 3},
        )

        assert response.status_code == 200
        data = response.json()

        # Should use query params
        assert data["groupsPerPage"] == 5
        assert data["suggestionsPerGroup"] == 3

    @pytest.mark.asyncio
    async def test_list_suggestions_validation_groups_per_page(
        self, test_client, setup_config_keys
    ):
        """Test validation for groupsPerPage parameter."""
        # Below minimum
        response = await test_client.get(
            "/api/v1/faces/suggestions",
            params={"groupsPerPage": 0},
        )
        assert response.status_code == 422

        # Above maximum
        response = await test_client.get(
            "/api/v1/faces/suggestions",
            params={"groupsPerPage": 51},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_list_suggestions_validation_suggestions_per_group(
        self, test_client, setup_config_keys
    ):
        """Test validation for suggestionsPerGroup parameter."""
        # Below minimum
        response = await test_client.get(
            "/api/v1/faces/suggestions",
            params={"suggestionsPerGroup": 0},
        )
        assert response.status_code == 422

        # Above maximum
        response = await test_client.get(
            "/api/v1/faces/suggestions",
            params={"suggestionsPerGroup": 51},
        )
        assert response.status_code == 422


class TestFaceSuggestionSettings:
    """Tests for /api/v1/config/face-suggestions endpoint."""

    @pytest.mark.asyncio
    async def test_get_face_suggestion_settings(
        self, test_client, db_session, setup_config_keys
    ):
        """Test retrieving face suggestion settings."""
        response = await test_client.get("/api/v1/config/face-suggestions")

        assert response.status_code == 200
        data = response.json()

        assert "groupsPerPage" in data
        assert "itemsPerGroup" in data
        assert data["groupsPerPage"] == 10
        assert data["itemsPerGroup"] == 20

    @pytest.mark.asyncio
    async def test_update_face_suggestion_settings(
        self, test_client, db_session, setup_config_keys
    ):
        """Test updating face suggestion settings."""
        response = await test_client.put(
            "/api/v1/config/face-suggestions",
            json={"groupsPerPage": 15, "itemsPerGroup": 25},
        )

        assert response.status_code == 200
        data = response.json()

        assert data["groupsPerPage"] == 15
        assert data["itemsPerGroup"] == 25

        # Verify persistence
        response = await test_client.get("/api/v1/config/face-suggestions")
        assert response.status_code == 200
        data = response.json()
        assert data["groupsPerPage"] == 15
        assert data["itemsPerGroup"] == 25

    @pytest.mark.asyncio
    async def test_update_face_suggestion_settings_validation(
        self, test_client, db_session, setup_config_keys
    ):
        """Test validation for update face suggestion settings."""
        # Below minimum
        response = await test_client.put(
            "/api/v1/config/face-suggestions",
            json={"groupsPerPage": 0, "itemsPerGroup": 20},
        )
        assert response.status_code == 422

        # Above maximum
        response = await test_client.put(
            "/api/v1/config/face-suggestions",
            json={"groupsPerPage": 10, "itemsPerGroup": 51},
        )
        assert response.status_code == 422
