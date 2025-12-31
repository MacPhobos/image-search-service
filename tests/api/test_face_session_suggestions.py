"""Tests for face session suggestions API endpoints.

These tests verify the /api/v1/faces/suggestions endpoints that manage
FaceSuggestion records created during face detection sessions.
"""

import uuid
from datetime import UTC, datetime

import pytest


@pytest.fixture
async def mock_image_asset(db_session):
    """Create a mock ImageAsset in the database."""
    from image_search_service.db.models import ImageAsset, TrainingStatus

    asset = ImageAsset(
        path="/test/images/photo.jpg",
        training_status=TrainingStatus.PENDING.value,
        width=1920,
        height=1080,
        file_size=204800,
        mime_type="image/jpeg",
    )
    db_session.add(asset)
    await db_session.commit()
    await db_session.refresh(asset)
    return asset


@pytest.fixture
async def mock_face_instance_with_bbox(db_session, mock_image_asset):
    """Create a FaceInstance with complete bounding box data."""
    from image_search_service.db.models import FaceInstance

    face = FaceInstance(
        id=uuid.uuid4(),
        asset_id=mock_image_asset.id,
        bbox_x=250,
        bbox_y=180,
        bbox_w=120,
        bbox_h=120,
        detection_confidence=0.98,
        quality_score=0.85,
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
        name="Jane Doe",
        status=PersonStatus.ACTIVE.value,
    )
    db_session.add(person)
    await db_session.commit()
    await db_session.refresh(person)
    return person


@pytest.fixture
async def mock_face_suggestion_with_bbox(
    db_session, mock_face_instance_with_bbox, mock_person
):
    """Create a FaceSuggestion with a face that has bbox data."""
    from image_search_service.db.models import FaceSuggestion, FaceSuggestionStatus

    suggestion = FaceSuggestion(
        face_instance_id=mock_face_instance_with_bbox.id,
        suggested_person_id=mock_person.id,
        confidence=0.92,
        source_face_id=uuid.uuid4(),
        status=FaceSuggestionStatus.PENDING.value,
        created_at=datetime.now(UTC),
    )
    db_session.add(suggestion)
    await db_session.commit()
    await db_session.refresh(suggestion)
    return suggestion


class TestGetSuggestion:
    """Tests for GET /api/v1/faces/suggestions/{suggestion_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_suggestion_returns_bounding_box_fields(
        self,
        test_client,
        db_session,
        mock_face_suggestion_with_bbox,
        mock_face_instance_with_bbox,
        mock_image_asset,
    ):
        """Test that bounding box fields are returned when face has bbox data."""
        response = await test_client.get(
            f"/api/v1/faces/suggestions/{mock_face_suggestion_with_bbox.id}"
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "id" in data
        assert "faceInstanceId" in data
        assert "suggestedPersonId" in data
        assert "confidence" in data

        # Verify bounding box fields are returned
        assert "fullImageUrl" in data
        assert "bboxX" in data
        assert "bboxY" in data
        assert "bboxW" in data
        assert "bboxH" in data
        assert "detectionConfidence" in data
        assert "qualityScore" in data

        # Verify bounding box values match the face instance
        assert data["bboxX"] == mock_face_instance_with_bbox.bbox_x
        assert data["bboxY"] == mock_face_instance_with_bbox.bbox_y
        assert data["bboxW"] == mock_face_instance_with_bbox.bbox_w
        assert data["bboxH"] == mock_face_instance_with_bbox.bbox_h
        assert data["detectionConfidence"] == mock_face_instance_with_bbox.detection_confidence
        assert data["qualityScore"] == mock_face_instance_with_bbox.quality_score

        # Verify full image URL construction
        expected_url = f"/api/v1/images/{mock_image_asset.id}/full"
        assert data["fullImageUrl"] == expected_url

    @pytest.mark.asyncio
    async def test_get_suggestion_not_found(self, test_client):
        """Test 404 when suggestion ID doesn't exist."""
        fake_id = 99999
        response = await test_client.get(f"/api/v1/faces/suggestions/{fake_id}")

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()


class TestListSuggestions:
    """Tests for GET /api/v1/faces/suggestions endpoint."""

    @pytest.mark.asyncio
    async def test_list_suggestions_includes_bbox_fields(
        self,
        test_client,
        db_session,
        mock_face_suggestion_with_bbox,
        mock_face_instance_with_bbox,
        mock_image_asset,
    ):
        """Test that list endpoint includes bounding box fields in all items (flat mode)."""
        # Use flat pagination mode to test legacy format
        response = await test_client.get(
            "/api/v1/faces/suggestions",
            params={"grouped": False}
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure (flat mode)
        assert "items" in data
        assert "total" in data
        assert len(data["items"]) >= 1

        # Find our suggestion in the list
        suggestion_item = next(
            (
                item
                for item in data["items"]
                if item["id"] == mock_face_suggestion_with_bbox.id
            ),
            None,
        )
        assert suggestion_item is not None

        # Verify bounding box fields are present and correct
        assert suggestion_item["bboxX"] == mock_face_instance_with_bbox.bbox_x
        assert suggestion_item["bboxY"] == mock_face_instance_with_bbox.bbox_y
        assert suggestion_item["bboxW"] == mock_face_instance_with_bbox.bbox_w
        assert suggestion_item["bboxH"] == mock_face_instance_with_bbox.bbox_h
        assert (
            suggestion_item["detectionConfidence"]
            == mock_face_instance_with_bbox.detection_confidence
        )
        assert suggestion_item["qualityScore"] == mock_face_instance_with_bbox.quality_score

        # Verify full image URL
        expected_url = f"/api/v1/images/{mock_image_asset.id}/full"
        assert suggestion_item["fullImageUrl"] == expected_url

class TestAcceptSuggestion:
    """Tests for POST /api/v1/faces/suggestions/{suggestion_id}/accept endpoint."""

    @pytest.mark.asyncio
    async def test_accept_suggestion_returns_bbox_fields(
        self,
        test_client,
        db_session,
        mock_face_suggestion_with_bbox,
        mock_face_instance_with_bbox,
        mock_image_asset,
    ):
        """Test that accept endpoint returns bbox fields in response."""
        from image_search_service.api.face_session_schemas import AcceptSuggestionRequest

        request_body = AcceptSuggestionRequest().model_dump(mode="json")

        response = await test_client.post(
            f"/api/v1/faces/suggestions/{mock_face_suggestion_with_bbox.id}/accept",
            json=request_body,
        )

        assert response.status_code == 200
        data = response.json()

        # Verify status changed
        assert data["status"] == "accepted"

        # Verify bounding box fields are present
        assert data["bboxX"] == mock_face_instance_with_bbox.bbox_x
        assert data["bboxY"] == mock_face_instance_with_bbox.bbox_y
        assert data["bboxW"] == mock_face_instance_with_bbox.bbox_w
        assert data["bboxH"] == mock_face_instance_with_bbox.bbox_h
        assert data["detectionConfidence"] == mock_face_instance_with_bbox.detection_confidence
        assert data["qualityScore"] == mock_face_instance_with_bbox.quality_score

        # Verify full image URL
        expected_url = f"/api/v1/images/{mock_image_asset.id}/full"
        assert data["fullImageUrl"] == expected_url


class TestRejectSuggestion:
    """Tests for POST /api/v1/faces/suggestions/{suggestion_id}/reject endpoint."""

    @pytest.mark.asyncio
    async def test_reject_suggestion_returns_bbox_fields(
        self,
        test_client,
        db_session,
        mock_face_suggestion_with_bbox,
        mock_face_instance_with_bbox,
        mock_image_asset,
    ):
        """Test that reject endpoint returns bbox fields in response."""
        from image_search_service.api.face_session_schemas import RejectSuggestionRequest

        request_body = RejectSuggestionRequest().model_dump(mode="json")

        response = await test_client.post(
            f"/api/v1/faces/suggestions/{mock_face_suggestion_with_bbox.id}/reject",
            json=request_body,
        )

        assert response.status_code == 200
        data = response.json()

        # Verify status changed
        assert data["status"] == "rejected"

        # Verify bounding box fields are present
        assert data["bboxX"] == mock_face_instance_with_bbox.bbox_x
        assert data["bboxY"] == mock_face_instance_with_bbox.bbox_y
        assert data["bboxW"] == mock_face_instance_with_bbox.bbox_w
        assert data["bboxH"] == mock_face_instance_with_bbox.bbox_h
        assert data["detectionConfidence"] == mock_face_instance_with_bbox.detection_confidence
        assert data["qualityScore"] == mock_face_instance_with_bbox.quality_score

        # Verify full image URL
        expected_url = f"/api/v1/images/{mock_image_asset.id}/full"
        assert data["fullImageUrl"] == expected_url

class TestBoundingBoxFieldValidation:
    """Tests for bounding box field value validation and edge cases."""

    @pytest.mark.asyncio
    async def test_bbox_fields_match_exact_values(
        self,
        test_client,
        db_session,
        mock_image_asset,
        mock_person,
    ):
        """Test that bbox fields return exact integer values from database."""
        from image_search_service.db.models import (
            FaceInstance,
            FaceSuggestion,
            FaceSuggestionStatus,
        )

        # Create face with specific bbox values
        face = FaceInstance(
            id=uuid.uuid4(),
            asset_id=mock_image_asset.id,
            bbox_x=42,
            bbox_y=137,
            bbox_w=256,
            bbox_h=256,
            detection_confidence=0.9876543,
            quality_score=0.123456,
            qdrant_point_id=uuid.uuid4(),
        )
        db_session.add(face)

        suggestion = FaceSuggestion(
            face_instance_id=face.id,
            suggested_person_id=mock_person.id,
            confidence=0.95,
            source_face_id=uuid.uuid4(),
            status=FaceSuggestionStatus.PENDING.value,
            created_at=datetime.now(UTC),
        )
        db_session.add(suggestion)
        await db_session.commit()
        await db_session.refresh(suggestion)

        response = await test_client.get(f"/api/v1/faces/suggestions/{suggestion.id}")

        assert response.status_code == 200
        data = response.json()

        # Verify exact integer values
        assert data["bboxX"] == 42
        assert data["bboxY"] == 137
        assert data["bboxW"] == 256
        assert data["bboxH"] == 256

        # Verify float precision preserved
        assert abs(data["detectionConfidence"] - 0.9876543) < 0.000001
        assert abs(data["qualityScore"] - 0.123456) < 0.000001

    @pytest.mark.asyncio
    async def test_full_image_url_pattern(
        self,
        test_client,
        db_session,
        mock_face_suggestion_with_bbox,
        mock_image_asset,
    ):
        """Test that fullImageUrl follows the correct pattern."""
        response = await test_client.get(
            f"/api/v1/faces/suggestions/{mock_face_suggestion_with_bbox.id}"
        )

        assert response.status_code == 200
        data = response.json()

        # Verify URL pattern: /api/v1/images/{asset_id}/full
        assert data["fullImageUrl"].startswith("/api/v1/images/")
        assert data["fullImageUrl"].endswith("/full")
        assert str(mock_image_asset.id) in data["fullImageUrl"]

        # Exact match
        expected = f"/api/v1/images/{mock_image_asset.id}/full"
        assert data["fullImageUrl"] == expected
