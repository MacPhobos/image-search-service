"""Tests for prototype management endpoints."""

import uuid
from unittest.mock import MagicMock

import pytest

# mock_image_asset from root conftest.py
# mock_person and mock_face_instance kept local (variants with person_id assignment)


@pytest.fixture
async def mock_person(db_session):
    """Create a mock Person in the database."""
    from image_search_service.db.models import Person, PersonStatus

    person = Person(
        id=uuid.uuid4(),
        name="Test Person",
        status=PersonStatus.ACTIVE,
    )
    db_session.add(person)
    await db_session.commit()
    await db_session.refresh(person)
    return person


@pytest.fixture
async def mock_face_instance(db_session, mock_image_asset, mock_person):
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
        person_id=mock_person.id,
    )
    db_session.add(face)
    await db_session.commit()
    await db_session.refresh(face)
    return face


@pytest.fixture
def mock_qdrant_client(monkeypatch):
    """Create a mock FaceQdrantClient."""
    mock_client = MagicMock()
    mock_client.ensure_collection.return_value = None
    mock_client.upsert_face.return_value = None
    mock_client.update_payload.return_value = None

    def get_mock_client():
        return mock_client

    monkeypatch.setattr(
        "image_search_service.vector.face_qdrant.get_face_qdrant_client",
        get_mock_client,
    )
    return mock_client


class TestPinPrototype:
    """Tests for PIN /persons/{id}/prototypes/pin endpoint."""

    @pytest.mark.asyncio
    async def test_pin_prototype_success(
        self, test_client, db_session, mock_person, mock_face_instance, mock_qdrant_client
    ):
        """POST /persons/{id}/prototypes/pin creates pinned prototype."""
        request_body = {
            "faceInstanceId": str(mock_face_instance.id),
            "ageEraBucket": "adult",
            "role": "temporal",
        }

        response = await test_client.post(
            f"/api/v1/faces/persons/{mock_person.id}/prototypes/pin",
            json=request_body,
        )

        assert response.status_code == 200
        data = response.json()
        assert "prototypeId" in data
        assert data["role"] == "temporal"
        assert data["ageEraBucket"] == "adult"
        assert data["isPinned"] is True

        # Verify Qdrant was updated
        mock_qdrant_client.update_payload.assert_called()

    @pytest.mark.asyncio
    async def test_pin_quota_exceeded(
        self, test_client, db_session, mock_person, mock_face_instance, mock_qdrant_client
    ):
        """Returns 400 when pin quota exceeded."""
        from image_search_service.db.models import FaceInstance, PersonPrototype, PrototypeRole

        # Create 3 PRIMARY prototypes (max quota)
        for i in range(3):
            face = FaceInstance(
                id=uuid.uuid4(),
                asset_id=mock_face_instance.asset_id,
                bbox_x=200 + i * 100,  # Start at 200 to avoid conflict with mock_face_instance
                bbox_y=200 + i * 100,  # Start at 200 to avoid conflict
                bbox_w=80,
                bbox_h=80,
                detection_confidence=0.95,
                quality_score=0.75,
                qdrant_point_id=uuid.uuid4(),
                person_id=mock_person.id,
            )
            db_session.add(face)
            await db_session.flush()

            prototype = PersonPrototype(
                person_id=mock_person.id,
                face_instance_id=face.id,
                qdrant_point_id=face.qdrant_point_id,
                role=PrototypeRole.PRIMARY,
                is_pinned=True,
            )
            db_session.add(prototype)

        await db_session.commit()

        # Try to pin a 4th PRIMARY
        request_body = {
            "faceInstanceId": str(mock_face_instance.id),
            "role": "primary",
        }

        response = await test_client.post(
            f"/api/v1/faces/persons/{mock_person.id}/prototypes/pin",
            json=request_body,
        )

        assert response.status_code == 400
        # The actual message is: "Maximum 2 PRIMARY prototypes already pinned (based on max_exemplars=5)"
        # Check for the key parts of the error message
        detail = response.json()["detail"]
        assert "PRIMARY prototypes already pinned" in detail
        assert "max_exemplars" in detail

    @pytest.mark.asyncio
    async def test_pin_era_already_pinned(
        self, test_client, db_session, mock_person, mock_face_instance, mock_qdrant_client
    ):
        """Returns 400 when era already has pinned prototype."""
        from image_search_service.db.models import FaceInstance, PersonPrototype, PrototypeRole

        # Create a TEMPORAL prototype for "adult" era
        existing_face = FaceInstance(
            id=uuid.uuid4(),
            asset_id=mock_face_instance.asset_id,
            bbox_x=200,
            bbox_y=150,
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.95,
            quality_score=0.75,
            qdrant_point_id=uuid.uuid4(),
            person_id=mock_person.id,
        )
        db_session.add(existing_face)
        await db_session.flush()

        prototype = PersonPrototype(
            person_id=mock_person.id,
            face_instance_id=existing_face.id,
            qdrant_point_id=existing_face.qdrant_point_id,
            role=PrototypeRole.TEMPORAL,
            age_era_bucket="adult",
            is_pinned=True,
        )
        db_session.add(prototype)
        await db_session.commit()

        # Try to pin another TEMPORAL for "adult" era
        request_body = {
            "faceInstanceId": str(mock_face_instance.id),
            "ageEraBucket": "adult",
            "role": "temporal",
        }

        response = await test_client.post(
            f"/api/v1/faces/persons/{mock_person.id}/prototypes/pin",
            json=request_body,
        )

        assert response.status_code == 400
        assert "already has a pinned temporal prototype" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_pin_face_not_found(self, test_client, db_session, mock_person):
        """Returns 404 for non-existent face."""
        fake_face_id = uuid.uuid4()
        request_body = {
            "faceInstanceId": str(fake_face_id),
            "role": "temporal",
            "ageEraBucket": "adult",
        }

        response = await test_client.post(
            f"/api/v1/faces/persons/{mock_person.id}/prototypes/pin",
            json=request_body,
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_pin_face_wrong_person(
        self, test_client, db_session, mock_person, mock_face_instance, mock_qdrant_client
    ):
        """Returns 400 when face belongs to different person."""
        from image_search_service.db.models import Person, PersonStatus

        # Create another person
        other_person = Person(
            id=uuid.uuid4(),
            name="Other Person",
            status=PersonStatus.ACTIVE,
        )
        db_session.add(other_person)
        await db_session.commit()

        # Try to pin mock_face_instance (which belongs to mock_person) to other_person
        request_body = {
            "faceInstanceId": str(mock_face_instance.id),
            "role": "temporal",
            "ageEraBucket": "adult",
        }

        response = await test_client.post(
            f"/api/v1/faces/persons/{other_person.id}/prototypes/pin",
            json=request_body,
        )

        assert response.status_code == 400
        assert "does not belong to person" in response.json()["detail"]


class TestUnpinPrototype:
    """Tests for DELETE /persons/{id}/prototypes/{proto_id}/pin endpoint."""

    @pytest.mark.asyncio
    async def test_unpin_success(
        self, test_client, db_session, mock_person, mock_face_instance, mock_qdrant_client
    ):
        """DELETE /prototypes/{id}/pin removes pin."""
        from image_search_service.db.models import PersonPrototype, PrototypeRole

        # Create a pinned prototype
        prototype = PersonPrototype(
            person_id=mock_person.id,
            face_instance_id=mock_face_instance.id,
            qdrant_point_id=mock_face_instance.qdrant_point_id,
            role=PrototypeRole.PRIMARY,
            is_pinned=True,
        )
        db_session.add(prototype)
        await db_session.commit()
        await db_session.refresh(prototype)

        response = await test_client.delete(
            f"/api/v1/faces/persons/{mock_person.id}/prototypes/{prototype.id}/pin"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unpinned"

        # Verify prototype is unpinned in database
        await db_session.refresh(prototype)
        assert prototype.is_pinned is False

    @pytest.mark.asyncio
    async def test_unpin_not_found(self, test_client, db_session, mock_person):
        """Returns 404 for non-existent prototype."""
        fake_proto_id = uuid.uuid4()
        response = await test_client.delete(
            f"/api/v1/faces/persons/{mock_person.id}/prototypes/{fake_proto_id}/pin"
        )

        assert response.status_code == 404


class TestListPrototypes:
    """Tests for GET /persons/{id}/prototypes endpoint."""

    @pytest.mark.asyncio
    async def test_list_includes_coverage(
        self, test_client, db_session, mock_person, mock_face_instance, mock_qdrant_client
    ):
        """GET /prototypes returns coverage information."""
        from image_search_service.db.models import PersonPrototype, PrototypeRole

        # Create a TEMPORAL prototype
        prototype = PersonPrototype(
            person_id=mock_person.id,
            face_instance_id=mock_face_instance.id,
            qdrant_point_id=mock_face_instance.qdrant_point_id,
            role=PrototypeRole.TEMPORAL,
            age_era_bucket="adult",
            is_pinned=True,
        )
        db_session.add(prototype)
        await db_session.commit()

        response = await test_client.get(f"/api/v1/faces/persons/{mock_person.id}/prototypes")

        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "coverage" in data
        assert len(data["items"]) == 1
        assert data["items"][0]["role"] == "temporal"
        assert data["items"][0]["ageEraBucket"] == "adult"

        # Check coverage
        coverage = data["coverage"]
        assert "coveredEras" in coverage
        assert "missingEras" in coverage
        assert "coveragePercentage" in coverage
        assert "totalPrototypes" in coverage
        assert "adult" in coverage["coveredEras"]

    @pytest.mark.asyncio
    async def test_list_empty(self, test_client, db_session, mock_person):
        """GET /prototypes returns empty list when no prototypes."""
        response = await test_client.get(f"/api/v1/faces/persons/{mock_person.id}/prototypes")

        assert response.status_code == 200
        data = response.json()
        assert data["items"] == []
        assert data["coverage"]["totalPrototypes"] == 0


class TestTemporalCoverage:
    """Tests for GET /persons/{id}/temporal-coverage endpoint."""

    @pytest.mark.asyncio
    async def test_coverage_calculation(
        self, test_client, db_session, mock_person, mock_face_instance, mock_qdrant_client
    ):
        """GET /temporal-coverage returns accurate stats."""
        from image_search_service.db.models import FaceInstance, PersonPrototype, PrototypeRole

        # Create TEMPORAL prototypes for 2 eras (adult, teen)
        eras = ["adult", "teen"]
        for i, era in enumerate(eras):
            face = FaceInstance(
                id=uuid.uuid4(),
                asset_id=mock_face_instance.asset_id,
                bbox_x=200 + i * 100,  # Start at 200 to avoid conflict
                bbox_y=200 + i * 100,  # Start at 200 to avoid conflict
                bbox_w=80,
                bbox_h=80,
                detection_confidence=0.95,
                quality_score=0.75,
                qdrant_point_id=uuid.uuid4(),
                person_id=mock_person.id,
            )
            db_session.add(face)
            await db_session.flush()

            prototype = PersonPrototype(
                person_id=mock_person.id,
                face_instance_id=face.id,
                qdrant_point_id=face.qdrant_point_id,
                role=PrototypeRole.TEMPORAL,
                age_era_bucket=era,
            )
            db_session.add(prototype)

        await db_session.commit()

        response = await test_client.get(
            f"/api/v1/faces/persons/{mock_person.id}/temporal-coverage"
        )

        assert response.status_code == 200
        data = response.json()
        assert "covered_eras" in data
        assert "missing_eras" in data
        assert "coverage_percentage" in data
        assert "total_prototypes" in data

        # Should have 2 eras covered
        assert len(data["covered_eras"]) == 2
        assert "adult" in data["covered_eras"]
        assert "teen" in data["covered_eras"]

        # Coverage percentage should be 2/6 = 33.33%
        assert data["coverage_percentage"] > 30
        assert data["coverage_percentage"] < 35


class TestRecomputePrototypes:
    """Tests for POST /persons/{id}/prototypes/recompute endpoint."""

    @pytest.mark.asyncio
    async def test_recompute_placeholder(self, test_client, db_session, mock_person):
        """POST /prototypes/recompute returns placeholder response."""
        request_body = {"preservePins": True}

        response = await test_client.post(
            f"/api/v1/faces/persons/{mock_person.id}/prototypes/recompute",
            json=request_body,
        )

        assert response.status_code == 200
        data = response.json()
        assert "prototypesCreated" in data
        assert "prototypesRemoved" in data
        assert "coverage" in data
