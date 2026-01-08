"""Integration tests for recompute prototypes with auto-rescan feature.

Tests the automatic suggestion regeneration when prototypes are recomputed.

POST /api/v1/faces/persons/{person_id}/prototypes/recompute
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


class TestRecomputeWithRescan:
    """Integration tests for recompute prototypes with auto-rescan feature."""

    @pytest.mark.asyncio
    async def test_recompute_no_rescan_by_default(
        self, test_client, db_session, create_person, create_face_instance, create_prototype
    ):
        """When config default is False and no parameter, rescan should NOT trigger."""
        person = await create_person(name="Alice")
        face = await create_face_instance(person_id=person.id, quality_score=0.9)
        await create_prototype(person_id=person.id, face_instance_id=face.id)

        # Mock Qdrant to prevent actual vector operations
        with patch(
            "image_search_service.vector.face_qdrant.get_face_qdrant_client"
        ) as mock_qdrant_factory:
            mock_qdrant = MagicMock()
            mock_qdrant_factory.return_value = mock_qdrant

            response = await test_client.post(
                f"/api/v1/faces/persons/{person.id}/prototypes/recompute",
                json={"preservePins": True},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["rescanTriggered"] is False
        assert data["rescanMessage"] is None

    @pytest.mark.asyncio
    async def test_recompute_with_explicit_rescan_true(
        self,
        test_client,
        db_session,
        create_person,
        create_face_instance,
        create_prototype,
    ):
        """When triggerRescan=true, should trigger rescan regardless of config."""
        person = await create_person(name="Bob")
        face = await create_face_instance(person_id=person.id, quality_score=0.85)
        await create_prototype(person_id=person.id, face_instance_id=face.id)

        # Mock Redis Queue and Qdrant
        with patch("redis.Redis") as mock_redis_cls, patch("rq.Queue") as mock_queue_cls, patch(
            "image_search_service.vector.face_qdrant.get_face_qdrant_client"
        ) as mock_qdrant_factory:
            # Setup Redis mocks
            mock_redis = MagicMock()
            mock_redis_cls.from_url.return_value = mock_redis

            mock_queue = MagicMock()
            mock_queue_cls.return_value = mock_queue

            # Setup Qdrant mock
            mock_qdrant = MagicMock()
            mock_qdrant_factory.return_value = mock_qdrant

            response = await test_client.post(
                f"/api/v1/faces/persons/{person.id}/prototypes/recompute",
                json={"preservePins": True, "triggerRescan": True},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["rescanTriggered"] is True
        assert data["rescanMessage"] is not None
        assert "queued" in data["rescanMessage"].lower()

        # Verify queue was called
        mock_queue.enqueue.assert_called_once()
        call_kwargs = mock_queue.enqueue.call_args.kwargs
        assert str(face.id) == call_kwargs["source_face_id"]
        assert str(person.id) == call_kwargs["person_id"]

    @pytest.mark.asyncio
    async def test_recompute_with_explicit_rescan_false(
        self,
        test_client,
        db_session,
        create_person,
        create_face_instance,
        create_prototype,
    ):
        """When triggerRescan=false, should NOT trigger rescan."""
        person = await create_person(name="Carol")
        face = await create_face_instance(person_id=person.id, quality_score=0.8)
        await create_prototype(person_id=person.id, face_instance_id=face.id)

        # Mock Qdrant
        with patch(
            "image_search_service.vector.face_qdrant.get_face_qdrant_client"
        ) as mock_qdrant_factory:
            mock_qdrant = MagicMock()
            mock_qdrant_factory.return_value = mock_qdrant

            response = await test_client.post(
                f"/api/v1/faces/persons/{person.id}/prototypes/recompute",
                json={"preservePins": True, "triggerRescan": False},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["rescanTriggered"] is False
        assert data["rescanMessage"] is None

    @pytest.mark.asyncio
    async def test_rescan_expires_pending_suggestions(
        self,
        test_client,
        db_session,
        create_person,
        create_face_instance,
        create_prototype,
        create_suggestion,
    ):
        """When rescan triggers, pending suggestions should be expired."""
        person = await create_person(name="Dave")
        face = await create_face_instance(person_id=person.id, quality_score=0.9)
        await create_prototype(person_id=person.id, face_instance_id=face.id)

        # Create 3 pending suggestions
        suggestion1 = await create_suggestion(suggested_person_id=person.id, status="pending")
        suggestion2 = await create_suggestion(suggested_person_id=person.id, status="pending")
        suggestion3 = await create_suggestion(suggested_person_id=person.id, status="pending")

        # Mock Redis Queue and Qdrant
        with patch("redis.Redis") as mock_redis_cls, patch("rq.Queue") as mock_queue_cls, patch(
            "image_search_service.vector.face_qdrant.get_face_qdrant_client"
        ) as mock_qdrant_factory:
            mock_redis = MagicMock()
            mock_redis_cls.from_url.return_value = mock_redis

            mock_queue = MagicMock()
            mock_queue_cls.return_value = mock_queue

            mock_qdrant = MagicMock()
            mock_qdrant_factory.return_value = mock_qdrant

            response = await test_client.post(
                f"/api/v1/faces/persons/{person.id}/prototypes/recompute",
                json={"triggerRescan": True},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["rescanTriggered"] is True
        assert "3" in data["rescanMessage"] or "expired" in data["rescanMessage"].lower()

        # Verify all suggestions were expired
        await db_session.refresh(suggestion1)
        await db_session.refresh(suggestion2)
        await db_session.refresh(suggestion3)

        assert suggestion1.status == FaceSuggestionStatus.EXPIRED.value
        assert suggestion2.status == FaceSuggestionStatus.EXPIRED.value
        assert suggestion3.status == FaceSuggestionStatus.EXPIRED.value

    @pytest.mark.asyncio
    async def test_rescan_uses_best_prototype(
        self,
        test_client,
        db_session,
        create_person,
        create_face_instance,
        create_prototype,
    ):
        """Rescan should use the highest quality prototype."""
        person = await create_person(name="Eve")

        # Create prototypes with different quality scores
        low_face = await create_face_instance(person_id=person.id, quality_score=0.5)
        high_face = await create_face_instance(person_id=person.id, quality_score=0.95)

        await create_prototype(person_id=person.id, face_instance_id=low_face.id)
        await create_prototype(person_id=person.id, face_instance_id=high_face.id)

        # Mock Redis Queue and Qdrant
        with patch("redis.Redis") as mock_redis_cls, patch("rq.Queue") as mock_queue_cls, patch(
            "image_search_service.vector.face_qdrant.get_face_qdrant_client"
        ) as mock_qdrant_factory:
            mock_redis = MagicMock()
            mock_redis_cls.from_url.return_value = mock_redis

            mock_queue = MagicMock()
            mock_queue_cls.return_value = mock_queue

            mock_qdrant = MagicMock()
            mock_qdrant_factory.return_value = mock_qdrant

            response = await test_client.post(
                f"/api/v1/faces/persons/{person.id}/prototypes/recompute",
                json={"triggerRescan": True},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["rescanTriggered"] is True

        # Verify the queued job uses high quality face
        call_kwargs = mock_queue.enqueue.call_args.kwargs
        assert str(high_face.id) == call_kwargs["source_face_id"]

    @pytest.mark.asyncio
    async def test_rescan_skipped_when_no_prototypes(
        self, test_client, db_session, create_person
    ):
        """If no prototypes exist after recompute, rescan should not trigger."""
        person = await create_person(name="Frank")
        # No faces, no prototypes

        # Mock Qdrant
        with patch(
            "image_search_service.vector.face_qdrant.get_face_qdrant_client"
        ) as mock_qdrant_factory:
            mock_qdrant = MagicMock()
            mock_qdrant_factory.return_value = mock_qdrant

            response = await test_client.post(
                f"/api/v1/faces/persons/{person.id}/prototypes/recompute",
                json={"triggerRescan": True},
            )

        assert response.status_code == 200
        data = response.json()
        # Rescan should not trigger without prototypes
        assert data["rescanTriggered"] is False

    @pytest.mark.asyncio
    async def test_config_default_triggers_rescan(
        self,
        test_client,
        db_session,
        create_person,
        create_face_instance,
        create_prototype,
        monkeypatch,
    ):
        """When config default is True and no parameter, rescan should trigger."""
        person = await create_person(name="Grace")
        face = await create_face_instance(person_id=person.id, quality_score=0.9)
        await create_prototype(person_id=person.id, face_instance_id=face.id)

        # Override settings to enable auto-rescan
        from image_search_service.core.config import get_settings

        # Clear cache and set environment variable
        get_settings.cache_clear()
        monkeypatch.setenv("FACE_SUGGESTIONS_AUTO_RESCAN_ON_RECOMPUTE", "true")
        get_settings.cache_clear()

        # Mock Redis Queue and Qdrant
        with patch("redis.Redis") as mock_redis_cls, patch("rq.Queue") as mock_queue_cls, patch(
            "image_search_service.vector.face_qdrant.get_face_qdrant_client"
        ) as mock_qdrant_factory:
            mock_redis = MagicMock()
            mock_redis_cls.from_url.return_value = mock_redis

            mock_queue = MagicMock()
            mock_queue_cls.return_value = mock_queue

            mock_qdrant = MagicMock()
            mock_qdrant_factory.return_value = mock_qdrant

            response = await test_client.post(
                f"/api/v1/faces/persons/{person.id}/prototypes/recompute",
                json={"preservePins": True},  # No triggerRescan parameter
            )

        assert response.status_code == 200
        data = response.json()
        assert data["rescanTriggered"] is True
        assert "queued" in data["rescanMessage"].lower()

        # Verify queue was called
        mock_queue.enqueue.assert_called_once()

        # Clean up
        get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_rescan_with_multiple_suggestions_only_expires_person_specific(
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
        suggestion_bob = await create_suggestion(suggested_person_id=person_bob.id, status="pending")

        # Mock Redis Queue and Qdrant
        with patch("redis.Redis") as mock_redis_cls, patch("rq.Queue") as mock_queue_cls, patch(
            "image_search_service.vector.face_qdrant.get_face_qdrant_client"
        ) as mock_qdrant_factory:
            mock_redis = MagicMock()
            mock_redis_cls.from_url.return_value = mock_redis

            mock_queue = MagicMock()
            mock_queue_cls.return_value = mock_queue

            mock_qdrant = MagicMock()
            mock_qdrant_factory.return_value = mock_qdrant

            # Regenerate for Alice only
            response = await test_client.post(
                f"/api/v1/faces/persons/{person_alice.id}/prototypes/recompute",
                json={"triggerRescan": True},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["rescanTriggered"] is True
        assert "1" in data["rescanMessage"]  # Only 1 suggestion expired

        # Verify Alice's suggestion was expired, Bob's was not
        await db_session.refresh(suggestion_alice)
        await db_session.refresh(suggestion_bob)

        assert suggestion_alice.status == FaceSuggestionStatus.EXPIRED.value
        assert suggestion_bob.status == FaceSuggestionStatus.PENDING.value  # Unchanged

    @pytest.mark.asyncio
    async def test_rescan_handles_null_quality_score(
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
        face_null_quality = await create_face_instance(person_id=person.id, quality_score=None)
        face_with_quality = await create_face_instance(person_id=person.id, quality_score=0.8)

        await create_prototype(person_id=person.id, face_instance_id=face_null_quality.id)
        await create_prototype(person_id=person.id, face_instance_id=face_with_quality.id)

        # Mock Redis Queue and Qdrant
        with patch("redis.Redis") as mock_redis_cls, patch("rq.Queue") as mock_queue_cls, patch(
            "image_search_service.vector.face_qdrant.get_face_qdrant_client"
        ) as mock_qdrant_factory:
            mock_redis = MagicMock()
            mock_redis_cls.from_url.return_value = mock_redis

            mock_queue = MagicMock()
            mock_queue_cls.return_value = mock_queue

            mock_qdrant = MagicMock()
            mock_qdrant_factory.return_value = mock_qdrant

            response = await test_client.post(
                f"/api/v1/faces/persons/{person.id}/prototypes/recompute",
                json={"triggerRescan": True},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["rescanTriggered"] is True

        # Should use face with quality=0.8 (not the null one)
        call_kwargs = mock_queue.enqueue.call_args.kwargs
        assert str(face_with_quality.id) == call_kwargs["source_face_id"]

    @pytest.mark.asyncio
    async def test_rescan_no_pending_suggestions_to_expire(
        self,
        test_client,
        db_session,
        create_person,
        create_face_instance,
        create_prototype,
        create_suggestion,
    ):
        """Successfully queues job even when there are no pending suggestions to expire."""
        person = await create_person(name="Isaac")
        face = await create_face_instance(person_id=person.id, quality_score=0.8)
        await create_prototype(person_id=person.id, face_instance_id=face.id)

        # Create already-expired and accepted suggestions (should not be re-expired)
        await create_suggestion(suggested_person_id=person.id, status="expired")
        await create_suggestion(suggested_person_id=person.id, status="accepted")

        # Mock Redis Queue and Qdrant
        with patch("redis.Redis") as mock_redis_cls, patch("rq.Queue") as mock_queue_cls, patch(
            "image_search_service.vector.face_qdrant.get_face_qdrant_client"
        ) as mock_qdrant_factory:
            mock_redis = MagicMock()
            mock_redis_cls.from_url.return_value = mock_redis

            mock_queue = MagicMock()
            mock_queue_cls.return_value = mock_queue

            mock_qdrant = MagicMock()
            mock_qdrant_factory.return_value = mock_qdrant

            response = await test_client.post(
                f"/api/v1/faces/persons/{person.id}/prototypes/recompute",
                json={"triggerRescan": True},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["rescanTriggered"] is True
        # Should mention 0 suggestions expired
        assert "0" in data["rescanMessage"]

        # Verify job was still queued
        mock_queue.enqueue.assert_called_once()

    @pytest.mark.asyncio
    async def test_rescan_queue_failure_handling(
        self,
        test_client,
        db_session,
        create_person,
        create_face_instance,
        create_prototype,
    ):
        """When Redis queue fails, should set rescanTriggered=False and include error message."""
        person = await create_person(name="Jack")
        face = await create_face_instance(person_id=person.id, quality_score=0.85)
        await create_prototype(person_id=person.id, face_instance_id=face.id)

        # Mock Redis to raise exception and Qdrant
        with patch("redis.Redis") as mock_redis_cls, patch(
            "image_search_service.vector.face_qdrant.get_face_qdrant_client"
        ) as mock_qdrant_factory:
            mock_redis_cls.from_url.side_effect = Exception("Redis connection failed")

            mock_qdrant = MagicMock()
            mock_qdrant_factory.return_value = mock_qdrant

            response = await test_client.post(
                f"/api/v1/faces/persons/{person.id}/prototypes/recompute",
                json={"triggerRescan": True},
            )

        assert response.status_code == 200
        data = response.json()
        # Rescan should not be marked as triggered
        assert data["rescanTriggered"] is False
        # Error message should be present
        assert data["rescanMessage"] is not None
        assert "failed" in data["rescanMessage"].lower()

    @pytest.mark.asyncio
    async def test_rescan_preserves_other_suggestion_statuses(
        self,
        test_client,
        db_session,
        create_person,
        create_face_instance,
        create_prototype,
        create_suggestion,
    ):
        """Rescan only expires pending suggestions, not accepted/rejected/expired ones."""
        person = await create_person(name="Karen")
        face = await create_face_instance(person_id=person.id, quality_score=0.9)
        await create_prototype(person_id=person.id, face_instance_id=face.id)

        # Create suggestions with different statuses
        pending1 = await create_suggestion(suggested_person_id=person.id, status="pending")
        pending2 = await create_suggestion(suggested_person_id=person.id, status="pending")
        accepted = await create_suggestion(suggested_person_id=person.id, status="accepted")
        rejected = await create_suggestion(suggested_person_id=person.id, status="rejected")
        already_expired = await create_suggestion(suggested_person_id=person.id, status="expired")

        # Mock Redis Queue and Qdrant
        with patch("redis.Redis") as mock_redis_cls, patch("rq.Queue") as mock_queue_cls, patch(
            "image_search_service.vector.face_qdrant.get_face_qdrant_client"
        ) as mock_qdrant_factory:
            mock_redis = MagicMock()
            mock_redis_cls.from_url.return_value = mock_redis

            mock_queue = MagicMock()
            mock_queue_cls.return_value = mock_queue

            mock_qdrant = MagicMock()
            mock_qdrant_factory.return_value = mock_qdrant

            response = await test_client.post(
                f"/api/v1/faces/persons/{person.id}/prototypes/recompute",
                json={"triggerRescan": True},
            )

        assert response.status_code == 200
        data = response.json()
        # Should only expire the 2 pending suggestions
        assert "2" in data["rescanMessage"]

        # Verify status changes
        await db_session.refresh(pending1)
        await db_session.refresh(pending2)
        await db_session.refresh(accepted)
        await db_session.refresh(rejected)
        await db_session.refresh(already_expired)

        # Pending ones should be expired
        assert pending1.status == FaceSuggestionStatus.EXPIRED.value
        assert pending2.status == FaceSuggestionStatus.EXPIRED.value

        # Others should be unchanged
        assert accepted.status == FaceSuggestionStatus.ACCEPTED.value
        assert rejected.status == FaceSuggestionStatus.REJECTED.value
        assert already_expired.status == FaceSuggestionStatus.EXPIRED.value
