"""Tests for post-training suggestions SQL query logic.

Tests the SQL queries used to select persons for post-training suggestions:
- 'all' mode: returns all persons with labeled faces
- 'top_n' mode: returns top N persons by face count
- Edge cases: no persons, persons without faces, ordering

These tests verify the SQL query logic without mocking Redis/Queue/Config.
"""

import json
import uuid

import pytest
from sqlalchemy import func
from sqlalchemy.orm import Session

from image_search_service.db.models import (
    FaceInstance,
    ImageAsset,
    Person,
)


@pytest.fixture
def image_assets(sync_db_session: Session) -> list[ImageAsset]:
    """Create test image assets."""
    assets = []
    for i in range(5):
        asset = ImageAsset(
            path=f"/test/images/test-image-{i}.jpg",
            width=800,
            height=600,
        )
        sync_db_session.add(asset)
        assets.append(asset)

    sync_db_session.commit()
    return assets


@pytest.fixture
def persons_with_faces(
    sync_db_session: Session
) -> list[tuple[Person, int]]:
    """Create persons with varying face counts for testing.

    Returns:
        List of (Person, face_count) tuples sorted by face count descending:
        - Person A: 50 faces
        - Person B: 30 faces
        - Person C: 20 faces
        - Person D: 10 faces
        - Person E: 5 faces
    """
    persons_data = [
        ("Person A", 50),  # Most faces
        ("Person B", 30),
        ("Person C", 20),
        ("Person D", 10),
        ("Person E", 5),  # Least faces
    ]

    persons = []
    for name, face_count in persons_data:
        person = Person(id=uuid.uuid4(), name=name)
        sync_db_session.add(person)
        sync_db_session.flush()

        # Create unique asset and face for each face count
        for i in range(face_count):
            # Create unique asset for each face
            asset = ImageAsset(
                path=f"/test/images/person_{name.replace(' ', '_')}_face_{i}.jpg",
                width=800,
                height=600,
            )
            sync_db_session.add(asset)
            sync_db_session.flush()

            face = FaceInstance(
                id=uuid.uuid4(),
                asset_id=asset.id,
                person_id=person.id,
                bbox_x=10,
                bbox_y=10,
                bbox_w=20,
                bbox_h=20,
                detection_confidence=0.95,
            )
            sync_db_session.add(face)

        persons.append((person, face_count))

    sync_db_session.commit()
    return persons


class TestPostTrainingSuggestionsAllMode:
    """Tests for 'all' mode: returns all persons with faces."""

    def test_all_mode_returns_all_persons_with_faces(
        self,
        sync_db_session: Session,
        persons_with_faces: list[tuple[Person, int]],
    ):
        """Test that query returns all persons with labeled faces."""
        # When: execute the "all mode" query (no LIMIT)
        persons_query = (
            sync_db_session.query(Person.id, func.count(FaceInstance.id).label("face_count"))
            .join(FaceInstance, FaceInstance.person_id == Person.id)
            .group_by(Person.id)
            .having(func.count(FaceInstance.id) > 0)
            .order_by(func.count(FaceInstance.id).desc())
        )

        persons = persons_query.all()

        # Then: all 5 persons returned
        assert len(persons) == 5

        # Verify results ordered by face count descending
        assert persons[0].face_count == 50  # Person A (most faces)
        assert persons[1].face_count == 30  # Person B
        assert persons[2].face_count == 20  # Person C
        assert persons[3].face_count == 10  # Person D
        assert persons[4].face_count == 5  # Person E (least faces)

    def test_all_mode_with_no_persons_returns_empty(
        self,
        sync_db_session: Session,
    ):
        """Test that query with no persons returns empty list gracefully."""
        # When: query for persons (no persons exist in database)
        persons_query = (
            sync_db_session.query(Person.id, func.count(FaceInstance.id).label("face_count"))
            .join(FaceInstance, FaceInstance.person_id == Person.id)
            .group_by(Person.id)
            .having(func.count(FaceInstance.id) > 0)
            .order_by(func.count(FaceInstance.id).desc())
        )

        persons = persons_query.all()

        # Then: no persons found (empty list)
        assert len(persons) == 0


class TestPostTrainingSuggestionsTopNMode:
    """Tests for 'top_n' mode: returns top N persons by face count."""

    def test_top_n_mode_limits_to_top_3_persons(
        self,
        sync_db_session: Session,
        persons_with_faces: list[tuple[Person, int]],
    ):
        """Test that top_n query with N=3 returns only top 3 persons."""
        # When: execute query with LIMIT 3
        top_n_count = 3
        persons_query = (
            sync_db_session.query(Person.id, func.count(FaceInstance.id).label("face_count"))
            .join(FaceInstance, FaceInstance.person_id == Person.id)
            .group_by(Person.id)
            .having(func.count(FaceInstance.id) > 0)
            .order_by(func.count(FaceInstance.id).desc())
            .limit(top_n_count)
        )

        persons = persons_query.all()

        # Then: only 3 persons returned (top 3 by face count)
        assert len(persons) == 3

        # Verify correct persons returned (highest face counts)
        assert persons[0].face_count == 50  # Person A
        assert persons[1].face_count == 30  # Person B
        assert persons[2].face_count == 20  # Person C

    def test_top_n_mode_with_n_equals_1_returns_only_top_person(
        self,
        sync_db_session: Session,
        persons_with_faces: list[tuple[Person, int]],
    ):
        """Test that top_n query with N=1 returns only the top person."""
        # When: execute query with LIMIT 1
        top_n_count = 1
        persons_query = (
            sync_db_session.query(Person.id, func.count(FaceInstance.id).label("face_count"))
            .join(FaceInstance, FaceInstance.person_id == Person.id)
            .group_by(Person.id)
            .having(func.count(FaceInstance.id) > 0)
            .order_by(func.count(FaceInstance.id).desc())
            .limit(top_n_count)
        )

        persons = persons_query.all()

        # Then: only 1 person returned (highest face count)
        assert len(persons) == 1
        assert persons[0].face_count == 50  # Person A with most faces

    def test_top_n_mode_with_n_greater_than_persons_returns_all(
        self,
        sync_db_session: Session,
        persons_with_faces: list[tuple[Person, int]],
    ):
        """Test that top_n query with N > total persons returns all persons."""
        # When: execute query with LIMIT 100 (more than 5 persons)
        top_n_count = 100
        persons_query = (
            sync_db_session.query(Person.id, func.count(FaceInstance.id).label("face_count"))
            .join(FaceInstance, FaceInstance.person_id == Person.id)
            .group_by(Person.id)
            .having(func.count(FaceInstance.id) > 0)
            .order_by(func.count(FaceInstance.id).desc())
            .limit(top_n_count)
        )

        persons = persons_query.all()

        # Then: all 5 persons returned (limit > available)
        assert len(persons) == 5


class TestPostTrainingSuggestionsEdgeCases:
    """Tests for edge cases and query behavior."""

    def test_persons_without_faces_excluded_from_results(
        self,
        sync_db_session: Session,
    ):
        """Test that persons with zero faces are excluded via HAVING clause."""
        # Given: create persons with and without faces
        person_with_faces = Person(id=uuid.uuid4(), name="Has Faces")
        person_without_faces = Person(id=uuid.uuid4(), name="No Faces")
        sync_db_session.add(person_with_faces)
        sync_db_session.add(person_without_faces)
        sync_db_session.flush()

        # Create asset and face for person_with_faces only
        asset = ImageAsset(
            path="/test/test.jpg",
            width=800,
            height=600,
        )
        sync_db_session.add(asset)
        sync_db_session.flush()

        face = FaceInstance(
            id=uuid.uuid4(),
            asset_id=asset.id,
            person_id=person_with_faces.id,
            bbox_x=10,
            bbox_y=10,
            bbox_w=20,
            bbox_h=20,
            detection_confidence=0.95,
        )
        sync_db_session.add(face)
        sync_db_session.commit()

        # When: query for persons (HAVING count > 0)
        persons_query = (
            sync_db_session.query(Person.id, func.count(FaceInstance.id).label("face_count"))
            .join(FaceInstance, FaceInstance.person_id == Person.id)
            .group_by(Person.id)
            .having(func.count(FaceInstance.id) > 0)
            .order_by(func.count(FaceInstance.id).desc())
        )

        persons = persons_query.all()

        # Then: only person with faces returned
        assert len(persons) == 1
        assert persons[0].id == person_with_faces.id

    def test_query_orders_by_face_count_descending(
        self,
        sync_db_session: Session,
        persons_with_faces: list[tuple[Person, int]],
    ):
        """Test that persons are ordered by face count (highest first)."""
        # When: query for persons
        persons_query = (
            sync_db_session.query(Person.id, func.count(FaceInstance.id).label("face_count"))
            .join(FaceInstance, FaceInstance.person_id == Person.id)
            .group_by(Person.id)
            .having(func.count(FaceInstance.id) > 0)
            .order_by(func.count(FaceInstance.id).desc())
        )

        persons = persons_query.all()

        # Then: persons ordered by face count descending
        face_counts = [p.face_count for p in persons]
        assert face_counts == [50, 30, 20, 10, 5]
        # Verify list is sorted descending
        assert face_counts == sorted(face_counts, reverse=True)


class TestPostTrainingSuggestionsQueryComponents:
    """Tests for individual query components and their behavior."""

    def test_having_clause_filters_zero_count(
        self,
        sync_db_session: Session,
    ):
        """Test that HAVING clause correctly filters persons with 0 faces."""
        # Given: person with no faces
        person = Person(id=uuid.uuid4(), name="Empty Person")
        sync_db_session.add(person)
        sync_db_session.commit()

        # When: query with HAVING count > 0
        persons_query = (
            sync_db_session.query(Person.id, func.count(FaceInstance.id).label("face_count"))
            .join(FaceInstance, FaceInstance.person_id == Person.id)
            .group_by(Person.id)
            .having(func.count(FaceInstance.id) > 0)
        )

        persons = persons_query.all()

        # Then: person with 0 faces excluded
        assert len(persons) == 0

    def test_join_requires_matching_person_id(
        self,
        sync_db_session: Session,
    ):
        """Test that JOIN correctly matches faces to persons."""
        # Given: person and face with matching person_id
        person = Person(id=uuid.uuid4(), name="Test Person")
        sync_db_session.add(person)
        sync_db_session.flush()

        asset = ImageAsset(
            path="/test/test.jpg",
            width=800,
            height=600,
        )
        sync_db_session.add(asset)
        sync_db_session.flush()

        face = FaceInstance(
            id=uuid.uuid4(),
            asset_id=asset.id,
            person_id=person.id,
            bbox_x=10,
            bbox_y=10,
            bbox_w=20,
            bbox_h=20,
            detection_confidence=0.95,
        )
        sync_db_session.add(face)
        sync_db_session.commit()

        # When: query with JOIN on person_id
        persons_query = (
            sync_db_session.query(Person.id, func.count(FaceInstance.id).label("face_count"))
            .join(FaceInstance, FaceInstance.person_id == Person.id)
            .group_by(Person.id)
            .having(func.count(FaceInstance.id) > 0)
        )

        persons = persons_query.all()

        # Then: person returned with correct face count
        assert len(persons) == 1
        assert persons[0].id == person.id
        assert persons[0].face_count == 1

    def test_limit_applies_after_ordering(
        self,
        sync_db_session: Session,
        persons_with_faces: list[tuple[Person, int]],
    ):
        """Test that LIMIT applies after ORDER BY (returns top N, not random N)."""
        # When: query with ORDER BY and LIMIT 2
        persons_query = (
            sync_db_session.query(Person.id, func.count(FaceInstance.id).label("face_count"))
            .join(FaceInstance, FaceInstance.person_id == Person.id)
            .group_by(Person.id)
            .having(func.count(FaceInstance.id) > 0)
            .order_by(func.count(FaceInstance.id).desc())
            .limit(2)
        )

        persons = persons_query.all()

        # Then: returns top 2 persons by face count
        assert len(persons) == 2
        assert persons[0].face_count == 50  # Highest
        assert persons[1].face_count == 30  # Second highest
        # NOT random 2 persons
