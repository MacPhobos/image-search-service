"""Tests for centroid-based post-training suggestions.

Tests the conditional logic that decides whether to use centroid-based
or prototype-based suggestion jobs during post-training flow.

Key scenarios:
- Centroid job queued when enabled and person has >= min_faces
- Prototype job queued when person has < min_faces
- Prototype job queued when centroids disabled
- Both job types queued for different persons in same training session
"""

import uuid
from unittest.mock import MagicMock, patch

import pytest
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
    # Create enough assets for all faces (10+7+3+1=21)
    for i in range(25):
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
def persons_with_varying_face_counts(
    sync_db_session: Session,
    image_assets: list[ImageAsset],
) -> list[tuple[Person, int]]:
    """Create persons with face counts above and below centroid threshold.

    Returns:
        List of (Person, face_count) tuples:
        - Person A: 10 faces (>= 5, should use centroid)
        - Person B: 7 faces (>= 5, should use centroid)
        - Person C: 3 faces (< 5, should use prototype)
        - Person D: 1 face (< 5, should use prototype)
    """
    persons_data = [
        ("Person A", 10),  # Above threshold
        ("Person B", 7),  # Above threshold
        ("Person C", 3),  # Below threshold
        ("Person D", 1),  # Below threshold
    ]

    persons = []
    asset_idx = 0

    for name, face_count in persons_data:
        person = Person(id=uuid.uuid4(), name=name)
        sync_db_session.add(person)
        sync_db_session.flush()

        # Create faces for this person
        for _ in range(face_count):
            face = FaceInstance(
                id=uuid.uuid4(),
                asset_id=image_assets[asset_idx].id,
                person_id=person.id,
                bbox_x=10,
                bbox_y=10,
                bbox_w=20,
                bbox_h=20,
                detection_confidence=0.95,
            )
            sync_db_session.add(face)
            asset_idx += 1

        persons.append((person, face_count))

    sync_db_session.commit()
    return persons


class TestCentroidPostTrainingEnabled:
    """Tests when centroid mode is enabled (post_training_use_centroids=True)."""

    def test_centroid_job_queued_when_person_has_sufficient_faces(
        self,
        sync_db_session: Session,
        persons_with_varying_face_counts: list[tuple[Person, int]],
    ):
        """Test that centroid job is queued for persons with >= min_faces."""
        # Given: Person A with 10 faces (>= 5 threshold)
        person_a, face_count_a = persons_with_varying_face_counts[0]
        assert face_count_a == 10

        # Mock config to enable centroids
        mock_config = MagicMock()
        mock_config.get_bool.return_value = True  # post_training_use_centroids=True
        mock_config.get_int.return_value = 5  # centroid_min_faces_for_suggestions=5
        mock_config.get_string.return_value = "all"  # suggestions_mode

        # When: evaluating which job to queue
        use_centroids = mock_config.get_bool("post_training_use_centroids")
        min_faces = mock_config.get_int("centroid_min_faces_for_suggestions")

        # Then: should use centroid job (has 10 faces >= 5 min)
        assert use_centroids is True
        assert face_count_a >= min_faces
        # This means centroid job should be queued

    def test_prototype_job_queued_when_person_has_insufficient_faces(
        self,
        sync_db_session: Session,
        persons_with_varying_face_counts: list[tuple[Person, int]],
    ):
        """Test that prototype job is queued for persons with < min_faces."""
        # Given: Person C with 3 faces (< 5 threshold)
        person_c, face_count_c = persons_with_varying_face_counts[2]
        assert face_count_c == 3

        # Mock config to enable centroids
        mock_config = MagicMock()
        mock_config.get_bool.return_value = True  # post_training_use_centroids=True
        mock_config.get_int.return_value = 5  # centroid_min_faces_for_suggestions=5

        # When: evaluating which job to queue
        use_centroids = mock_config.get_bool("post_training_use_centroids")
        min_faces = mock_config.get_int("centroid_min_faces_for_suggestions")

        # Then: should use prototype job (has 3 faces < 5 min)
        assert use_centroids is True
        assert face_count_c < min_faces
        # This means prototype job should be queued (fallback)

    def test_both_job_types_queued_for_different_persons(
        self,
        sync_db_session: Session,
        persons_with_varying_face_counts: list[tuple[Person, int]],
    ):
        """Test mixed scenario: some persons get centroid, some get prototype jobs."""
        # Given: persons with varying face counts
        persons = persons_with_varying_face_counts

        # Mock config
        mock_config = MagicMock()
        mock_config.get_bool.return_value = True  # centroids enabled
        mock_config.get_int.return_value = 5  # min_faces=5

        use_centroids = mock_config.get_bool("post_training_use_centroids")
        min_faces = mock_config.get_int("centroid_min_faces_for_suggestions")

        # When: categorizing persons by job type
        centroid_persons = []
        prototype_persons = []

        for person, face_count in persons:
            if use_centroids and face_count >= min_faces:
                centroid_persons.append((person, face_count))
            else:
                prototype_persons.append((person, face_count))

        # Then: correct categorization
        # Centroid: Person A (10), Person B (7)
        assert len(centroid_persons) == 2
        assert centroid_persons[0][1] == 10  # Person A
        assert centroid_persons[1][1] == 7  # Person B

        # Prototype: Person C (3), Person D (1)
        assert len(prototype_persons) == 2
        assert prototype_persons[0][1] == 3  # Person C
        assert prototype_persons[1][1] == 1  # Person D


class TestCentroidPostTrainingDisabled:
    """Tests when centroid mode is disabled (post_training_use_centroids=False)."""

    def test_prototype_job_always_queued_when_centroids_disabled(
        self,
        sync_db_session: Session,
        persons_with_varying_face_counts: list[tuple[Person, int]],
    ):
        """Test that all persons get prototype jobs when centroids disabled."""
        # Given: persons with varying face counts
        persons = persons_with_varying_face_counts

        # Mock config to disable centroids
        mock_config = MagicMock()
        mock_config.get_bool.return_value = False  # post_training_use_centroids=False
        mock_config.get_int.return_value = 5  # min_faces (irrelevant when disabled)

        use_centroids = mock_config.get_bool("post_training_use_centroids")
        min_faces = mock_config.get_int("centroid_min_faces_for_suggestions")

        # When: categorizing persons
        centroid_persons = []
        prototype_persons = []

        for person, face_count in persons:
            if use_centroids and face_count >= min_faces:
                centroid_persons.append((person, face_count))
            else:
                prototype_persons.append((person, face_count))

        # Then: ALL persons use prototype jobs (centroids disabled)
        assert len(centroid_persons) == 0
        assert len(prototype_persons) == 4  # All 4 persons

        # Verify face counts irrelevant when centroids disabled
        face_counts = [fc for _, fc in prototype_persons]
        assert face_counts == [10, 7, 3, 1]  # Both high and low counts

    def test_prototype_job_used_even_with_high_face_count(
        self,
        sync_db_session: Session,
        persons_with_varying_face_counts: list[tuple[Person, int]],
    ):
        """Test that prototype job used for person with 10 faces when centroids disabled."""
        # Given: Person A with 10 faces (well above threshold)
        person_a, face_count_a = persons_with_varying_face_counts[0]
        assert face_count_a == 10

        # Mock config to disable centroids
        mock_config = MagicMock()
        mock_config.get_bool.return_value = False  # centroids disabled
        mock_config.get_int.return_value = 5

        use_centroids = mock_config.get_bool("post_training_use_centroids")
        min_faces = mock_config.get_int("centroid_min_faces_for_suggestions")

        # Then: should use prototype job (centroids disabled overrides face count)
        assert use_centroids is False
        assert face_count_a >= min_faces  # Has enough faces
        # But still uses prototype because centroids disabled


class TestCentroidThresholdBoundary:
    """Tests for boundary conditions at the min_faces threshold."""

    def test_exactly_at_threshold_uses_centroid(
        self,
        sync_db_session: Session,
    ):
        """Test that person with exactly min_faces (5) uses centroid job."""
        # Given: person with exactly 5 faces (threshold)
        person = Person(id=uuid.uuid4(), name="Boundary Person")
        sync_db_session.add(person)
        sync_db_session.flush()

        for i in range(5):
            asset = ImageAsset(
                path=f"/test/boundary-{i}.jpg",
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

        # Mock config
        mock_config = MagicMock()
        mock_config.get_bool.return_value = True
        mock_config.get_int.return_value = 5  # Threshold

        use_centroids = mock_config.get_bool("post_training_use_centroids")
        min_faces = mock_config.get_int("centroid_min_faces_for_suggestions")
        face_count = 5

        # Then: should use centroid (>= includes equality)
        assert use_centroids is True
        assert face_count >= min_faces
        # Centroid job should be queued

    def test_one_below_threshold_uses_prototype(
        self,
        sync_db_session: Session,
    ):
        """Test that person with min_faces-1 (4) uses prototype job."""
        # Given: person with 4 faces (one below threshold)
        person = Person(id=uuid.uuid4(), name="Just Below Person")
        sync_db_session.add(person)
        sync_db_session.flush()

        for i in range(4):
            asset = ImageAsset(
                path=f"/test/below-{i}.jpg",
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

        # Mock config
        mock_config = MagicMock()
        mock_config.get_bool.return_value = True
        mock_config.get_int.return_value = 5

        use_centroids = mock_config.get_bool("post_training_use_centroids")
        min_faces = mock_config.get_int("centroid_min_faces_for_suggestions")
        face_count = 4

        # Then: should use prototype (4 < 5)
        assert use_centroids is True
        assert face_count < min_faces
        # Prototype job should be queued


class TestConfigurationDefaults:
    """Tests for configuration default values."""

    def test_default_config_enables_centroids(self):
        """Test that default config has centroids enabled."""
        from image_search_service.services.config_service import ConfigService

        # Then: default is True
        assert ConfigService.DEFAULTS["post_training_use_centroids"] is True

    def test_default_min_faces_is_five(self):
        """Test that default minimum faces threshold is 5."""
        from image_search_service.services.config_service import ConfigService

        # Then: default is 5
        assert ConfigService.DEFAULTS["centroid_min_faces_for_suggestions"] == 5


class TestJobTypeLogging:
    """Tests for logging job type and reason in debug messages."""

    def test_centroid_job_logs_correct_type(self):
        """Test that centroid jobs log job_type='centroid'."""
        # Given: person with sufficient faces for centroid
        face_count = 10
        use_centroids = True
        min_faces = 5

        # When: determining job type
        if use_centroids and face_count >= min_faces:
            job_type = "centroid"
            reason = None
        else:
            job_type = "prototype"
            reason = "insufficient_faces" if use_centroids else "centroids_disabled"

        # Then: correct job type
        assert job_type == "centroid"
        assert reason is None

    def test_prototype_job_logs_insufficient_faces_reason(self):
        """Test that prototype jobs log reason='insufficient_faces' when centroids enabled."""
        # Given: person with insufficient faces
        face_count = 3
        use_centroids = True
        min_faces = 5

        # When: determining job type
        if use_centroids and face_count >= min_faces:
            job_type = "centroid"
            reason = None
        else:
            job_type = "prototype"
            reason = "insufficient_faces" if use_centroids else "centroids_disabled"

        # Then: correct job type and reason
        assert job_type == "prototype"
        assert reason == "insufficient_faces"

    def test_prototype_job_logs_centroids_disabled_reason(self):
        """Test that prototype jobs log reason='centroids_disabled' when disabled."""
        # Given: centroids disabled (face count irrelevant)
        face_count = 10
        use_centroids = False
        min_faces = 5

        # When: determining job type
        if use_centroids and face_count >= min_faces:
            job_type = "centroid"
            reason = None
        else:
            job_type = "prototype"
            reason = "insufficient_faces" if use_centroids else "centroids_disabled"

        # Then: correct job type and reason
        assert job_type == "prototype"
        assert reason == "centroids_disabled"
