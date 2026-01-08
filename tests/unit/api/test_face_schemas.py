"""Unit tests for face API schemas."""

import uuid

import pytest
from pydantic import ValidationError

from image_search_service.api.face_schemas import (
    PersonType,
    RecomputePrototypesRequest,
    RecomputePrototypesResponse,
    TemporalCoverage,
    UnifiedPeopleListResponse,
    UnifiedPersonResponse,
)


class TestPersonType:
    """Tests for PersonType enum."""

    def test_enum_values(self):
        """Should have correct enum values."""
        assert PersonType.IDENTIFIED.value == "identified"
        assert PersonType.UNIDENTIFIED.value == "unidentified"
        assert PersonType.NOISE.value == "noise"

    def test_enum_from_string(self):
        """Should create enum from string value."""
        assert PersonType("identified") == PersonType.IDENTIFIED
        assert PersonType("unidentified") == PersonType.UNIDENTIFIED
        assert PersonType("noise") == PersonType.NOISE

    def test_enum_invalid_value(self):
        """Should raise ValueError for invalid enum value."""
        with pytest.raises(ValueError):
            PersonType("invalid_type")


class TestUnifiedPersonResponse:
    """Tests for UnifiedPersonResponse schema."""

    def test_identified_person_serialization(self):
        """Should serialize identified person correctly."""
        person_id = uuid.uuid4()
        person = UnifiedPersonResponse(
            id=str(person_id),
            name="John Doe",
            type=PersonType.IDENTIFIED,
            face_count=10,
            thumbnail_url="http://example.com/thumb.jpg",
        )

        # Serialize with aliases (camelCase)
        data = person.model_dump(by_alias=True)

        assert data["id"] == str(person_id)
        assert data["name"] == "John Doe"
        assert data["type"] == "identified"
        assert data["faceCount"] == 10  # camelCase
        assert data["thumbnailUrl"] == "http://example.com/thumb.jpg"
        assert data["confidence"] is None

    def test_unidentified_person_serialization(self):
        """Should serialize unidentified cluster correctly."""
        person = UnifiedPersonResponse(
            id="cluster_abc123",
            name="Unidentified Person 1",
            type=PersonType.UNIDENTIFIED,
            face_count=25,
            confidence=0.85,
        )

        data = person.model_dump(by_alias=True)

        assert data["id"] == "cluster_abc123"
        assert data["name"] == "Unidentified Person 1"
        assert data["type"] == "unidentified"
        assert data["faceCount"] == 25
        assert data["confidence"] == 0.85
        assert data["thumbnailUrl"] is None

    def test_noise_person_serialization(self):
        """Should serialize noise faces correctly."""
        person = UnifiedPersonResponse(
            id="-1",
            name="Unknown Faces",
            type=PersonType.NOISE,
            face_count=5,
            thumbnail_url="/api/v1/images/123/thumbnail",
        )

        data = person.model_dump(by_alias=True)

        assert data["id"] == "-1"
        assert data["name"] == "Unknown Faces"
        assert data["type"] == "noise"
        assert data["faceCount"] == 5
        assert data["thumbnailUrl"] == "/api/v1/images/123/thumbnail"
        assert data["confidence"] is None

    def test_optional_fields(self):
        """Should handle optional fields correctly."""
        person = UnifiedPersonResponse(
            id="test",
            name="Test",
            type=PersonType.IDENTIFIED,
            face_count=1,
        )

        data = person.model_dump(by_alias=True)

        assert data["thumbnailUrl"] is None
        assert data["confidence"] is None

    def test_camel_case_serialization(self):
        """Should use camelCase for JSON serialization."""
        person = UnifiedPersonResponse(
            id="test",
            name="Test",
            type=PersonType.IDENTIFIED,
            face_count=10,
            thumbnail_url="/test/url",
        )

        # by_alias=True should use camelCase
        data = person.model_dump(by_alias=True)

        assert "faceCount" in data
        assert "thumbnailUrl" in data
        # snake_case should not exist
        assert "face_count" not in data
        assert "thumbnail_url" not in data

    def test_snake_case_internal(self):
        """Should use snake_case internally."""
        person = UnifiedPersonResponse(
            id="test",
            name="Test",
            type=PersonType.IDENTIFIED,
            face_count=10,
        )

        # by_alias=False should use snake_case (default)
        data = person.model_dump(by_alias=False)

        assert "face_count" in data
        assert "thumbnail_url" in data
        assert "faceCount" not in data
        assert "thumbnailUrl" not in data

    def test_required_fields(self):
        """Should require id, name, type, and face_count."""
        # Missing id
        with pytest.raises(ValidationError) as exc_info:
            UnifiedPersonResponse(
                name="Test",
                type=PersonType.IDENTIFIED,
                face_count=10,
            )
        assert "id" in str(exc_info.value)

        # Missing name
        with pytest.raises(ValidationError) as exc_info:
            UnifiedPersonResponse(
                id="test",
                type=PersonType.IDENTIFIED,
                face_count=10,
            )
        assert "name" in str(exc_info.value)

        # Missing type
        with pytest.raises(ValidationError) as exc_info:
            UnifiedPersonResponse(
                id="test",
                name="Test",
                face_count=10,
            )
        assert "type" in str(exc_info.value)

        # Missing face_count (error message uses camelCase alias)
        with pytest.raises(ValidationError) as exc_info:
            UnifiedPersonResponse(
                id="test",
                name="Test",
                type=PersonType.IDENTIFIED,
            )
        # Pydantic uses the alias (camelCase) in error messages
        assert "faceCount" in str(exc_info.value)

    def test_type_validation(self):
        """Should validate field types."""
        # Invalid face_count type
        with pytest.raises(ValidationError):
            UnifiedPersonResponse(
                id="test",
                name="Test",
                type=PersonType.IDENTIFIED,
                face_count="not_a_number",
            )

        # Invalid confidence type
        with pytest.raises(ValidationError):
            UnifiedPersonResponse(
                id="test",
                name="Test",
                type=PersonType.IDENTIFIED,
                face_count=10,
                confidence="not_a_float",
            )


class TestUnifiedPeopleListResponse:
    """Tests for UnifiedPeopleListResponse schema."""

    def test_list_response_serialization(self):
        """Should serialize list response correctly."""
        people = [
            UnifiedPersonResponse(
                id=str(uuid.uuid4()),
                name="Person 1",
                type=PersonType.IDENTIFIED,
                face_count=10,
            ),
            UnifiedPersonResponse(
                id="cluster_1",
                name="Unidentified Person 1",
                type=PersonType.UNIDENTIFIED,
                face_count=5,
            ),
        ]

        response = UnifiedPeopleListResponse(
            people=people,
            total=2,
            identified_count=1,
            unidentified_count=1,
            noise_count=0,
        )

        data = response.model_dump(by_alias=True)

        assert len(data["people"]) == 2
        assert data["total"] == 2
        assert data["identifiedCount"] == 1  # camelCase
        assert data["unidentifiedCount"] == 1
        assert data["noiseCount"] == 0

    def test_empty_list_response(self):
        """Should handle empty people list."""
        response = UnifiedPeopleListResponse(
            people=[],
            total=0,
            identified_count=0,
            unidentified_count=0,
            noise_count=0,
        )

        data = response.model_dump(by_alias=True)

        assert data["people"] == []
        assert data["total"] == 0
        assert data["identifiedCount"] == 0
        assert data["unidentifiedCount"] == 0
        assert data["noiseCount"] == 0

    def test_camel_case_count_fields(self):
        """Should use camelCase for count fields."""
        response = UnifiedPeopleListResponse(
            people=[],
            total=10,
            identified_count=5,
            unidentified_count=3,
            noise_count=2,
        )

        data = response.model_dump(by_alias=True)

        # Check camelCase
        assert "identifiedCount" in data
        assert "unidentifiedCount" in data
        assert "noiseCount" in data

        # Check values
        assert data["identifiedCount"] == 5
        assert data["unidentifiedCount"] == 3
        assert data["noiseCount"] == 2

    def test_required_fields_list_response(self):
        """Should require all fields."""
        # Missing total
        with pytest.raises(ValidationError) as exc_info:
            UnifiedPeopleListResponse(
                people=[],
                identified_count=0,
                unidentified_count=0,
                noise_count=0,
            )
        assert "total" in str(exc_info.value)

        # Missing identified_count (error uses camelCase alias)
        with pytest.raises(ValidationError) as exc_info:
            UnifiedPeopleListResponse(
                people=[],
                total=0,
                unidentified_count=0,
                noise_count=0,
            )
        # Pydantic uses the alias (camelCase) in error messages
        assert "identifiedCount" in str(exc_info.value)

    def test_count_consistency(self):
        """Counts should reflect actual people list."""
        people = [
            UnifiedPersonResponse(
                id=str(uuid.uuid4()),
                name="Person 1",
                type=PersonType.IDENTIFIED,
                face_count=10,
            ),
            UnifiedPersonResponse(
                id=str(uuid.uuid4()),
                name="Person 2",
                type=PersonType.IDENTIFIED,
                face_count=8,
            ),
            UnifiedPersonResponse(
                id="cluster_1",
                name="Unidentified Person 1",
                type=PersonType.UNIDENTIFIED,
                face_count=5,
            ),
        ]

        response = UnifiedPeopleListResponse(
            people=people,
            total=3,
            identified_count=2,
            unidentified_count=1,
            noise_count=0,
        )

        # Validate that counts match the actual list
        actual_identified = sum(1 for p in people if p.type == PersonType.IDENTIFIED)
        actual_unidentified = sum(1 for p in people if p.type == PersonType.UNIDENTIFIED)
        actual_noise = sum(1 for p in people if p.type == PersonType.NOISE)

        assert response.identified_count == actual_identified
        assert response.unidentified_count == actual_unidentified
        assert response.noise_count == actual_noise
        assert response.total == len(people)

    def test_json_serialization_round_trip(self):
        """Should serialize and deserialize correctly."""
        original = UnifiedPeopleListResponse(
            people=[
                UnifiedPersonResponse(
                    id=str(uuid.uuid4()),
                    name="Test Person",
                    type=PersonType.IDENTIFIED,
                    face_count=15,
                    thumbnail_url="/api/v1/images/1/thumbnail",
                    confidence=None,
                )
            ],
            total=1,
            identified_count=1,
            unidentified_count=0,
            noise_count=0,
        )

        # Serialize to JSON
        json_data = original.model_dump(by_alias=True)

        # Deserialize from JSON (using snake_case field names)
        restored = UnifiedPeopleListResponse.model_validate(
            {
                "people": json_data["people"],
                "total": json_data["total"],
                "identified_count": json_data["identifiedCount"],
                "unidentified_count": json_data["unidentifiedCount"],
                "noise_count": json_data["noiseCount"],
            }
        )

        assert restored.total == original.total
        assert restored.identified_count == original.identified_count
        assert len(restored.people) == len(original.people)


class TestRecomputePrototypesRequest:
    """Tests for RecomputePrototypesRequest schema."""

    def test_default_values(self):
        """Should use default values when not provided."""
        request = RecomputePrototypesRequest()

        assert request.preserve_pins is True
        assert request.trigger_rescan is None

    def test_explicit_values(self):
        """Should accept explicit values."""
        request = RecomputePrototypesRequest(preserve_pins=False, trigger_rescan=True)

        assert request.preserve_pins is False
        assert request.trigger_rescan is True

    def test_trigger_rescan_none(self):
        """Should allow trigger_rescan to be None (uses config default)."""
        request = RecomputePrototypesRequest(preserve_pins=True, trigger_rescan=None)

        assert request.preserve_pins is True
        assert request.trigger_rescan is None

    def test_camel_case_serialization(self):
        """Should use camelCase for JSON serialization."""
        request = RecomputePrototypesRequest(preserve_pins=False, trigger_rescan=True)
        data = request.model_dump(by_alias=True)

        assert "preservePins" in data
        assert "triggerRescan" in data
        assert data["preservePins"] is False
        assert data["triggerRescan"] is True


class TestRecomputePrototypesResponse:
    """Tests for RecomputePrototypesResponse schema."""

    def test_minimal_response(self):
        """Should work with minimal required fields."""
        coverage = TemporalCoverage(
            covered_eras=["infant", "child"],
            missing_eras=["teen", "young_adult"],
            coverage_percentage=0.4,
            total_prototypes=2,
        )
        response = RecomputePrototypesResponse(
            prototypes_created=2,
            prototypes_removed=1,
            coverage=coverage,
        )

        assert response.prototypes_created == 2
        assert response.prototypes_removed == 1
        assert response.rescan_triggered is False
        assert response.rescan_message is None

    def test_with_rescan_info(self):
        """Should include rescan information when provided."""
        coverage = TemporalCoverage(
            covered_eras=["infant"],
            missing_eras=["child", "teen"],
            coverage_percentage=0.2,
            total_prototypes=1,
        )
        response = RecomputePrototypesResponse(
            prototypes_created=1,
            prototypes_removed=0,
            coverage=coverage,
            rescan_triggered=True,
            rescan_message="Rescan queued successfully",
        )

        assert response.rescan_triggered is True
        assert response.rescan_message == "Rescan queued successfully"

    def test_camel_case_serialization(self):
        """Should use camelCase for JSON serialization."""
        coverage = TemporalCoverage(
            covered_eras=["infant"],
            missing_eras=[],
            coverage_percentage=1.0,
            total_prototypes=1,
        )
        response = RecomputePrototypesResponse(
            prototypes_created=1,
            prototypes_removed=0,
            coverage=coverage,
            rescan_triggered=True,
            rescan_message="Test message",
        )
        data = response.model_dump(by_alias=True)

        assert "prototypesCreated" in data
        assert "prototypesRemoved" in data
        assert "rescanTriggered" in data
        assert "rescanMessage" in data
        assert data["rescanTriggered"] is True
        assert data["rescanMessage"] == "Test message"
