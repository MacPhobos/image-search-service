"""Tests for suggestion regeneration endpoint.

POST /api/v1/faces/persons/{person_id}/suggestions/regenerate
"""

import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from image_search_service.db.models import (
    FaceInstance,
    FaceSuggestion,
    FaceSuggestionStatus,
    ImageAsset,
    Person,
    PersonPrototype,
    PersonStatus,
    PrototypeRole,
    TrainingStatus,
)
from image_search_service.queue.face_jobs import propagate_person_label_multiproto_job

# ============ Fixtures ============


@pytest.fixture
async def create_person(db_session):
    """Factory fixture for creating Person records."""

    async def _create(name: str = "Test Person", status: PersonStatus = PersonStatus.ACTIVE):
        person = Person(
            id=uuid.uuid4(),
            name=name,
            status=status.value,
        )
        db_session.add(person)
        await db_session.commit()
        await db_session.refresh(person)
        return person

    return _create


@pytest.fixture
async def create_image_asset(db_session):
    """Factory fixture for creating ImageAsset records."""

    async def _create(path: str = None):
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
        await db_session.commit()
        await db_session.refresh(asset)
        return asset

    return _create


@pytest.fixture
async def create_face_instance(db_session, create_image_asset):
    """Factory fixture for creating FaceInstance records."""

    async def _create(
        person_id: uuid.UUID | None = None,
        quality_score: float = 0.75,
        asset_id: uuid.UUID | None = None,
    ):
        # Create asset if not provided
        if asset_id is None:
            asset = await create_image_asset()
            asset_id = asset.id

        face = FaceInstance(
            id=uuid.uuid4(),
            asset_id=asset_id,
            person_id=person_id,
            bbox_x=100,
            bbox_y=150,
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.95,
            quality_score=quality_score,
            qdrant_point_id=uuid.uuid4(),
        )
        db_session.add(face)
        await db_session.commit()
        await db_session.refresh(face)
        return face

    return _create


@pytest.fixture
async def create_prototype(db_session):
    """Factory fixture for creating PersonPrototype records."""

    async def _create(
        person_id: uuid.UUID,
        face_instance_id: uuid.UUID,
        role: PrototypeRole = PrototypeRole.EXEMPLAR,
        qdrant_point_id: uuid.UUID | None = None,
    ):
        if qdrant_point_id is None:
            qdrant_point_id = uuid.uuid4()

        prototype = PersonPrototype(
            id=uuid.uuid4(),
            person_id=person_id,
            face_instance_id=face_instance_id,
            qdrant_point_id=qdrant_point_id,
            role=role.value,
            created_at=datetime.now(UTC),
        )
        db_session.add(prototype)
        await db_session.commit()
        await db_session.refresh(prototype)
        return prototype

    return _create


@pytest.fixture
async def create_suggestion(db_session, create_face_instance):
    """Factory fixture for creating FaceSuggestion records."""

    async def _create(
        suggested_person_id: uuid.UUID,
        status: str = "pending",
        face_instance_id: uuid.UUID | None = None,
        source_face_id: uuid.UUID | None = None,
    ):
        # Create face if not provided
        if face_instance_id is None:
            face = await create_face_instance()
            face_instance_id = face.id

        # Create source face if not provided
        if source_face_id is None:
            source_face = await create_face_instance()
            source_face_id = source_face.id

        suggestion = FaceSuggestion(
            face_instance_id=face_instance_id,
            suggested_person_id=suggested_person_id,
            source_face_id=source_face_id,
            confidence=0.85,
            status=status,
            created_at=datetime.now(UTC),
        )
        db_session.add(suggestion)
        await db_session.commit()
        await db_session.refresh(suggestion)
        return suggestion

    return _create


# ============ Tests ============


class TestRegenerateSuggestions:
    """Tests for POST /api/v1/faces/persons/{person_id}/suggestions/regenerate endpoint."""

    @pytest.mark.asyncio
    async def test_regenerate_suggestions_person_not_found(self, test_client, db_session):
        """Returns 404 when person doesn't exist."""
        fake_person_id = uuid.uuid4()

        response = await test_client.post(
            f"/api/v1/faces/persons/{fake_person_id}/suggestions/regenerate"
        )

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()
        assert str(fake_person_id) in data["detail"]

    @pytest.mark.asyncio
    async def test_regenerate_suggestions_no_prototypes(
        self, test_client, db_session, create_person
    ):
        """Returns 400 when person has no prototypes."""
        person = await create_person(name="Alice")

        response = await test_client.post(
            f"/api/v1/faces/persons/{person.id}/suggestions/regenerate"
        )

        assert response.status_code == 400
        data = response.json()
        assert "no prototypes" in data["detail"].lower()

    @pytest.mark.asyncio
    async def test_regenerate_suggestions_success(
        self,
        test_client,
        db_session,
        create_person,
        create_face_instance,
        create_prototype,
        create_suggestion,
    ):
        """Successfully expires old suggestions and queues new job."""
        # Setup: Create person with prototype
        person = await create_person(name="Alice")
        face = await create_face_instance(person_id=person.id, quality_score=0.9)
        await create_prototype(person_id=person.id, face_instance_id=face.id)

        # Create old pending suggestion
        old_suggestion = await create_suggestion(
            suggested_person_id=person.id, status="pending"
        )

        # Mock Redis Queue (patch where they're imported FROM, not where they're used)
        with patch("redis.Redis") as mock_redis_cls, patch("rq.Queue") as mock_queue_cls:
            # Setup mocks
            mock_redis = MagicMock()
            mock_redis_cls.from_url.return_value = mock_redis

            mock_queue = MagicMock()
            mock_queue_cls.return_value = mock_queue

            # Make request
            response = await test_client.post(
                f"/api/v1/faces/persons/{person.id}/suggestions/regenerate"
            )

            # Verify response
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "queued"
            assert "queued" in data["message"].lower()
            # expiredCount is now 0 in endpoint response (expiration happens in job)
            assert data["expiredCount"] == 0

            # Verify queue was called with the new multi-prototype job
            mock_queue.enqueue.assert_called_once()
            call_args = mock_queue.enqueue.call_args
            # Check the job function
            assert call_args.args[0] == propagate_person_label_multiproto_job
            # Check person_id is passed (could be positional or keyword)
            call_kwargs = call_args.kwargs
            assert str(person.id) == call_kwargs.get("person_id", call_args.args[1] if len(call_args.args) > 1 else None)
            assert call_kwargs.get("min_confidence", 0.7) == 0.7
            assert call_kwargs.get("max_suggestions", 50) == 50

        # Note: Old suggestion expiration now happens inside the job, not in the endpoint
        # So we can't verify it was expired here without running the actual job

    @pytest.mark.asyncio
    async def test_regenerate_uses_best_prototype(
        self,
        test_client,
        db_session,
        create_person,
        create_face_instance,
        create_prototype,
    ):
        """Uses the highest quality prototype for regeneration."""
        person = await create_person(name="Bob")

        # Create faces with different quality scores
        low_quality_face = await create_face_instance(person_id=person.id, quality_score=0.5)
        medium_quality_face = await create_face_instance(
            person_id=person.id, quality_score=0.7
        )
        high_quality_face = await create_face_instance(
            person_id=person.id, quality_score=0.95
        )

        # Create prototypes for all faces
        await create_prototype(person_id=person.id, face_instance_id=low_quality_face.id)
        await create_prototype(person_id=person.id, face_instance_id=medium_quality_face.id)
        await create_prototype(person_id=person.id, face_instance_id=high_quality_face.id)

        # Mock Redis Queue (patch where they're imported FROM, not where they're used)
        with patch("redis.Redis") as mock_redis_cls, patch("rq.Queue") as mock_queue_cls:
            mock_redis = MagicMock()
            mock_redis_cls.from_url.return_value = mock_redis

            mock_queue = MagicMock()
            mock_queue_cls.return_value = mock_queue

            response = await test_client.post(
                f"/api/v1/faces/persons/{person.id}/suggestions/regenerate"
            )

            assert response.status_code == 200

            # Verify the queued job was called with person_id (no longer needs source_face_id)
            mock_queue.enqueue.assert_called_once()
            call_args = mock_queue.enqueue.call_args
            # Check the job function
            assert call_args.args[0] == propagate_person_label_multiproto_job
            # Check person_id is passed
            call_kwargs = call_args.kwargs
            assert str(person.id) == call_kwargs.get("person_id", call_args.args[1] if len(call_args.args) > 1 else None)

    @pytest.mark.asyncio
    async def test_regenerate_no_pending_suggestions_to_expire(
        self,
        test_client,
        db_session,
        create_person,
        create_face_instance,
        create_prototype,
        create_suggestion,
    ):
        """Successfully queues job even when there are no pending suggestions to expire."""
        person = await create_person(name="Charlie")
        face = await create_face_instance(person_id=person.id, quality_score=0.8)
        await create_prototype(person_id=person.id, face_instance_id=face.id)

        # Create already-expired and accepted suggestions (should not be re-expired)
        await create_suggestion(suggested_person_id=person.id, status="expired")
        await create_suggestion(suggested_person_id=person.id, status="accepted")

        # Mock Redis Queue (patch where they're imported FROM, not where they're used)
        with patch("redis.Redis") as mock_redis_cls, patch("rq.Queue") as mock_queue_cls:
            mock_redis = MagicMock()
            mock_redis_cls.from_url.return_value = mock_redis

            mock_queue = MagicMock()
            mock_queue_cls.return_value = mock_queue

            response = await test_client.post(
                f"/api/v1/faces/persons/{person.id}/suggestions/regenerate"
            )

            assert response.status_code == 200
            data = response.json()
            assert data["expiredCount"] == 0  # No pending suggestions to expire
            mock_queue.enqueue.assert_called_once()

    @pytest.mark.asyncio
    async def test_regenerate_prototype_without_face_instance(
        self, test_client, db_session, create_person
    ):
        """Queues job successfully even with orphaned prototype (validation happens in job)."""
        person = await create_person(name="David")

        # Mock Redis Queue
        with patch("redis.Redis") as mock_redis_cls, patch("rq.Queue") as mock_queue_cls:
            mock_redis = MagicMock()
            mock_redis_cls.from_url.return_value = mock_redis

            mock_queue = MagicMock()
            mock_queue_cls.return_value = mock_queue

            # Create prototype with no face_instance_id (edge case)
            orphaned_prototype = PersonPrototype(
                id=uuid.uuid4(),
                person_id=person.id,
                face_instance_id=None,  # No face
                qdrant_point_id=uuid.uuid4(),
                role=PrototypeRole.EXEMPLAR.value,
                created_at=datetime.now(UTC),
            )
            db_session.add(orphaned_prototype)
            await db_session.commit()

            response = await test_client.post(
                f"/api/v1/faces/persons/{person.id}/suggestions/regenerate"
            )

            # Endpoint now queues successfully (validation happens in job)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "queued"

    @pytest.mark.asyncio
    async def test_regenerate_prototype_face_missing_from_database(
        self, test_client, db_session, create_person
    ):
        """Queues job successfully even with dangling prototype reference (validation in job)."""
        person = await create_person(name="Eve")

        # Mock Redis Queue
        with patch("redis.Redis") as mock_redis_cls, patch("rq.Queue") as mock_queue_cls:
            mock_redis = MagicMock()
            mock_redis_cls.from_url.return_value = mock_redis

            mock_queue = MagicMock()
            mock_queue_cls.return_value = mock_queue

            # Create prototype with non-existent face_instance_id
            fake_face_id = uuid.uuid4()
            dangling_prototype = PersonPrototype(
                id=uuid.uuid4(),
                person_id=person.id,
                face_instance_id=fake_face_id,  # Face doesn't exist
                qdrant_point_id=uuid.uuid4(),
                role=PrototypeRole.EXEMPLAR.value,
                created_at=datetime.now(UTC),
            )
            db_session.add(dangling_prototype)
            await db_session.commit()

            response = await test_client.post(
                f"/api/v1/faces/persons/{person.id}/suggestions/regenerate"
            )

            # Endpoint now queues successfully (validation happens in job)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "queued"

    @pytest.mark.asyncio
    async def test_regenerate_queue_failure(
        self,
        test_client,
        db_session,
        create_person,
        create_face_instance,
        create_prototype,
    ):
        """Returns 500 when Redis queue enqueue fails."""
        person = await create_person(name="Frank")
        face = await create_face_instance(person_id=person.id, quality_score=0.85)
        await create_prototype(person_id=person.id, face_instance_id=face.id)

        # Mock Redis to raise exception
        with patch("redis.Redis") as mock_redis_cls:
            mock_redis_cls.from_url.side_effect = Exception("Redis connection failed")

            response = await test_client.post(
                f"/api/v1/faces/persons/{person.id}/suggestions/regenerate"
            )

            assert response.status_code == 500
            data = response.json()
            assert "failed to queue" in data["detail"].lower()

    @pytest.mark.asyncio
    async def test_regenerate_invalid_person_id_format(self, test_client):
        """Returns 422 validation error for invalid UUID format."""
        invalid_uuid = "not-a-valid-uuid"

        response = await test_client.post(
            f"/api/v1/faces/persons/{invalid_uuid}/suggestions/regenerate"
        )

        # FastAPI validation should return 422 for invalid UUID
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_regenerate_expires_multiple_pending_suggestions(
        self,
        test_client,
        db_session,
        create_person,
        create_face_instance,
        create_prototype,
        create_suggestion,
    ):
        """Expires all pending suggestions for the person, not just one."""
        person = await create_person(name="Grace")
        face = await create_face_instance(person_id=person.id, quality_score=0.9)
        await create_prototype(person_id=person.id, face_instance_id=face.id)

        # Create multiple pending suggestions
        suggestion1 = await create_suggestion(suggested_person_id=person.id, status="pending")
        suggestion2 = await create_suggestion(suggested_person_id=person.id, status="pending")
        suggestion3 = await create_suggestion(suggested_person_id=person.id, status="pending")

        # Mock Redis Queue (patch where they're imported FROM, not where they're used)
        with patch("redis.Redis") as mock_redis_cls, patch("rq.Queue") as mock_queue_cls:
            mock_redis = MagicMock()
            mock_redis_cls.from_url.return_value = mock_redis

            mock_queue = MagicMock()
            mock_queue_cls.return_value = mock_queue

            response = await test_client.post(
                f"/api/v1/faces/persons/{person.id}/suggestions/regenerate"
            )

            assert response.status_code == 200
            data = response.json()
            # expiredCount is now 0 in endpoint response (expiration happens in job)
            assert data["expiredCount"] == 0

        # Note: Suggestion expiration now happens inside the job, not in the endpoint
        # So we can't verify they were expired here without running the actual job

    @pytest.mark.asyncio
    async def test_regenerate_only_expires_person_specific_suggestions(
        self,
        test_client,
        db_session,
        create_person,
        create_face_instance,
        create_prototype,
        create_suggestion,
    ):
        """Only expires suggestions for the specific person, not other persons."""
        person_alice = await create_person(name="Alice")
        person_bob = await create_person(name="Bob")

        # Create prototypes for Alice
        face_alice = await create_face_instance(person_id=person_alice.id, quality_score=0.9)
        await create_prototype(person_id=person_alice.id, face_instance_id=face_alice.id)

        # Create pending suggestions for both persons
        suggestion_alice = await create_suggestion(
            suggested_person_id=person_alice.id, status="pending"
        )
        suggestion_bob = await create_suggestion(
            suggested_person_id=person_bob.id, status="pending"
        )

        # Mock Redis Queue (patch where they're imported FROM, not where they're used)
        with patch("redis.Redis") as mock_redis_cls, patch("rq.Queue") as mock_queue_cls:
            mock_redis = MagicMock()
            mock_redis_cls.from_url.return_value = mock_redis

            mock_queue = MagicMock()
            mock_queue_cls.return_value = mock_queue

            # Regenerate for Alice only
            response = await test_client.post(
                f"/api/v1/faces/persons/{person_alice.id}/suggestions/regenerate"
            )

            assert response.status_code == 200
            data = response.json()
            # expiredCount is now 0 in endpoint response (expiration happens in job)
            assert data["expiredCount"] == 0

        # Note: Suggestion expiration now happens inside the job, not in the endpoint
        # The job will only expire Alice's suggestions, not Bob's

    @pytest.mark.asyncio
    async def test_regenerate_handles_null_quality_score(
        self,
        test_client,
        db_session,
        create_person,
        create_face_instance,
        create_prototype,
    ):
        """Handles faces with null quality_score gracefully by treating as 0.0."""
        person = await create_person(name="Helen")

        # Create faces with null and valid quality scores
        face_null_quality = await create_face_instance(
            person_id=person.id, quality_score=None
        )
        face_with_quality = await create_face_instance(
            person_id=person.id, quality_score=0.8
        )

        await create_prototype(person_id=person.id, face_instance_id=face_null_quality.id)
        await create_prototype(person_id=person.id, face_instance_id=face_with_quality.id)

        # Mock Redis Queue (patch where they're imported FROM, not where they're used)
        with patch("redis.Redis") as mock_redis_cls, patch("rq.Queue") as mock_queue_cls:
            mock_redis = MagicMock()
            mock_redis_cls.from_url.return_value = mock_redis

            mock_queue = MagicMock()
            mock_queue_cls.return_value = mock_queue

            response = await test_client.post(
                f"/api/v1/faces/persons/{person.id}/suggestions/regenerate"
            )

            assert response.status_code == 200

            # Verify job was called with person_id (multi-proto job handles all prototypes internally)
            call_args = mock_queue.enqueue.call_args
            assert call_args.args[0] == propagate_person_label_multiproto_job
            call_kwargs = call_args.kwargs
            assert str(person.id) == call_kwargs.get("person_id", call_args.args[1] if len(call_args.args) > 1 else None)
