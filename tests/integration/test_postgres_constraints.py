"""PostgreSQL constraint and foreign key integration tests.

These tests validate database-specific constraints that SQLite does not enforce:
- Functional indexes (e.g., func.lower(name))
- CASCADE/SET NULL foreign key behavior
- PostgreSQL enum type enforcement
- Unique constraints on composite keys

SQLite does not enforce functional indexes or enum types, so these
tests can only run on real PostgreSQL.
"""

import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from image_search_service.db.models import (
    FaceInstance,
    ImageAsset,
    Person,
    TrainingJob,
    TrainingSession,
)


@pytest.mark.postgres
async def test_person_name_case_insensitive_uniqueness(pg_session):  # type: ignore
    """Verify Person.name unique index enforces case-insensitive uniqueness.

    This test CANNOT pass on SQLite because SQLite does not enforce
    functional indexes. This is the primary motivation for PostgreSQL tests.
    """
    person1 = Person(name="John Smith")
    pg_session.add(person1)
    await pg_session.commit()

    person2 = Person(name="john smith")  # Different case, same name
    pg_session.add(person2)

    with pytest.raises(IntegrityError):
        await pg_session.commit()


@pytest.mark.postgres
async def test_face_instance_location_uniqueness(pg_session):  # type: ignore
    """Verify uq_face_instance_location prevents duplicate face detections."""
    asset = ImageAsset(path="/test/photo.jpg")
    pg_session.add(asset)
    await pg_session.flush()

    face1 = FaceInstance(
        asset_id=asset.id,
        bbox_x=100,
        bbox_y=200,
        bbox_w=50,
        bbox_h=50,
        detection_confidence=0.95,
        qdrant_point_id=uuid.uuid4(),
    )
    pg_session.add(face1)
    await pg_session.commit()

    face2 = FaceInstance(
        asset_id=asset.id,
        bbox_x=100,
        bbox_y=200,
        bbox_w=50,
        bbox_h=50,  # Same location
        detection_confidence=0.90,
        qdrant_point_id=uuid.uuid4(),
    )
    pg_session.add(face2)

    with pytest.raises(IntegrityError):
        await pg_session.commit()


@pytest.mark.postgres
async def test_cascade_delete_person_nullifies_face_instances(pg_session):  # type: ignore
    """Verify ON DELETE SET NULL for FaceInstance.person_id works on PostgreSQL."""
    person = Person(name="Test Person For Cascade")
    pg_session.add(person)
    await pg_session.flush()

    asset = ImageAsset(path="/test/photo.jpg")
    pg_session.add(asset)
    await pg_session.flush()

    face = FaceInstance(
        asset_id=asset.id,
        bbox_x=10,
        bbox_y=20,
        bbox_w=30,
        bbox_h=30,
        detection_confidence=0.9,
        qdrant_point_id=uuid.uuid4(),
        person_id=person.id,
    )
    pg_session.add(face)
    await pg_session.commit()

    face_id = face.id

    # Delete person -- face_instance.person_id should become NULL
    await pg_session.delete(person)
    await pg_session.commit()

    # Refresh face instance
    refreshed_face = await pg_session.get(FaceInstance, face_id)
    assert refreshed_face is not None
    assert refreshed_face.person_id is None  # SET NULL behavior


@pytest.mark.postgres
async def test_cascade_delete_session_removes_jobs(pg_session):  # type: ignore
    """Verify ON DELETE CASCADE for TrainingJob when session deleted."""
    session = TrainingSession(name="test-session", root_path="/test")
    pg_session.add(session)
    await pg_session.flush()

    asset = ImageAsset(path="/test/img.jpg")
    pg_session.add(asset)
    await pg_session.flush()

    job = TrainingJob(session_id=session.id, asset_id=asset.id)
    pg_session.add(job)
    await pg_session.commit()

    job_id = job.id

    # Delete session -- jobs should cascade delete
    await pg_session.delete(session)
    await pg_session.commit()

    # Verify job is gone
    result = await pg_session.get(TrainingJob, job_id)
    assert result is None


@pytest.mark.postgres
async def test_person_status_enum_enforced(pg_session):  # type: ignore
    """Verify PostgreSQL enum prevents invalid status values.

    SQLite stores enums as strings and accepts any value.
    PostgreSQL enforces the enum constraint.
    """
    person = Person(name="Enum Test Person")
    pg_session.add(person)
    await pg_session.commit()

    # Try to set invalid status via raw SQL
    with pytest.raises(Exception):  # DataError or ProgrammingError
        await pg_session.execute(
            text("UPDATE persons SET status = 'invalid_status' WHERE name = 'Enum Test Person'")
        )
        await pg_session.commit()
