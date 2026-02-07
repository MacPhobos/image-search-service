"""PostgreSQL JSONB column integration tests.

These tests validate that JSONB columns behave correctly on real PostgreSQL,
catching issues that SQLite silently ignores.

PostgreSQL JSONB supports:
- Native binary JSON storage (more efficient than JSON text)
- Indexing, containment operators (@>, <@)
- Key-existence checks (?)
- Normalized storage (key deduplication, ordering)

SQLite stores these as plain text JSON, so JSONB-specific features
are not validated in the fast test tier.
"""

import uuid

import pytest

from image_search_service.db.models import (
    FaceAssignmentEvent,
    FaceInstance,
    FaceSuggestion,
    ImageAsset,
    Person,
    PersonCentroid,
    SystemConfig,
)


@pytest.mark.postgres
async def test_exif_metadata_jsonb_round_trip(pg_session):  # type: ignore
    """Verify JSONB stores and retrieves nested dicts correctly.

    JSONB normalizes key order and deduplicates keys, unlike JSON text.
    This test catches any code that depends on JSON text ordering.
    """
    asset = ImageAsset(
        path="/test/photo.jpg",
        exif_metadata={
            "DateTimeOriginal": "2024:01:15 10:30:00",
            "Make": "Canon",
            "GPSInfo": {"latitude": 37.7749, "longitude": -122.4194},
            "nested": {"deep": {"value": [1, 2, 3]}},
        },
    )
    pg_session.add(asset)
    await pg_session.commit()
    await pg_session.refresh(asset)

    # Verify nested structure preserved
    assert asset.exif_metadata is not None
    # Type narrowing for mypy: we know exif_metadata is a dict
    exif = asset.exif_metadata
    assert isinstance(exif, dict)
    assert exif["GPSInfo"]["latitude"] == 37.7749  # type: ignore
    assert exif["nested"]["deep"]["value"] == [1, 2, 3]  # type: ignore


@pytest.mark.postgres
async def test_face_landmarks_jsonb(pg_session):  # type: ignore
    """Verify face landmarks JSONB stores 5-point facial landmarks correctly."""
    # Create prerequisite ImageAsset
    asset = ImageAsset(path="/test/face_photo.jpg")
    pg_session.add(asset)
    await pg_session.flush()

    face = FaceInstance(
        asset_id=asset.id,
        bbox_x=100,
        bbox_y=200,
        bbox_w=50,
        bbox_h=50,
        detection_confidence=0.95,
        qdrant_point_id=uuid.uuid4(),
        landmarks={
            "left_eye": [120, 215],
            "right_eye": [140, 215],
            "nose": [130, 230],
            "left_mouth": [118, 240],
            "right_mouth": [142, 240],
        },
    )
    pg_session.add(face)
    await pg_session.commit()
    await pg_session.refresh(face)

    assert face.landmarks is not None
    assert face.landmarks["left_eye"] == [120, 215]
    assert len(face.landmarks) == 5


@pytest.mark.postgres
async def test_suggestion_prototype_scores_jsonb(pg_session):  # type: ignore
    """Verify prototype_scores JSONB preserves float precision."""
    # Set up prerequisite person
    person = Person(name="Test Person For Suggestion")
    pg_session.add(person)
    await pg_session.flush()

    # Set up prerequisite asset
    asset = ImageAsset(path="/test/suggestion_photo.jpg")
    pg_session.add(asset)
    await pg_session.flush()

    # Set up face_instance (target face)
    face_instance = FaceInstance(
        asset_id=asset.id,
        bbox_x=10,
        bbox_y=20,
        bbox_w=30,
        bbox_h=30,
        detection_confidence=0.9,
        qdrant_point_id=uuid.uuid4(),
    )
    pg_session.add(face_instance)
    await pg_session.flush()

    # Set up source_face
    source_face = FaceInstance(
        asset_id=asset.id,
        bbox_x=100,
        bbox_y=200,
        bbox_w=50,
        bbox_h=50,
        detection_confidence=0.95,
        qdrant_point_id=uuid.uuid4(),
    )
    pg_session.add(source_face)
    await pg_session.flush()

    suggestion = FaceSuggestion(
        face_instance_id=face_instance.id,
        suggested_person_id=person.id,
        confidence=0.85,
        source_face_id=source_face.id,
        matching_prototype_ids=["proto-1", "proto-2", "proto-3"],
        prototype_scores={
            "proto-1": 0.92341,
            "proto-2": 0.87654,
            "proto-3": 0.75123,
        },
        aggregate_confidence=0.85039,
        prototype_match_count=3,
    )
    pg_session.add(suggestion)
    await pg_session.commit()
    await pg_session.refresh(suggestion)

    # Verify float precision preserved
    assert suggestion.prototype_scores is not None
    assert abs(suggestion.prototype_scores["proto-1"] - 0.92341) < 1e-10
    assert suggestion.matching_prototype_ids == ["proto-1", "proto-2", "proto-3"]
    assert suggestion.prototype_match_count == 3


@pytest.mark.postgres
async def test_centroid_build_params_jsonb(pg_session):  # type: ignore
    """Verify build_params stores algorithm configuration correctly."""
    # Set up prerequisite Person
    person = Person(name="Test Person For Centroid")
    pg_session.add(person)
    await pg_session.flush()

    centroid = PersonCentroid(
        person_id=person.id,
        qdrant_point_id=uuid.uuid4(),
        model_version="arcface_r100_glint360k_v1",
        centroid_version=1,
        n_faces=25,
        build_params={
            "algorithm": "trimmed_mean",
            "trim_threshold": 0.05,
            "outlier_count": 2,
            "face_ids_hash": "abc123def456",
        },
    )
    pg_session.add(centroid)
    await pg_session.commit()
    await pg_session.refresh(centroid)

    assert centroid.build_params is not None
    assert centroid.build_params["algorithm"] == "trimmed_mean"
    assert centroid.build_params["trim_threshold"] == 0.05


@pytest.mark.postgres
async def test_assignment_event_jsonb_arrays(pg_session):  # type: ignore
    """Verify JSONB array columns store and retrieve correctly."""
    event = FaceAssignmentEvent(
        operation="MOVE_TO_PERSON",
        face_count=3,
        photo_count=2,
        affected_photo_ids=[101, 102],
        affected_face_instance_ids=["uuid-1", "uuid-2", "uuid-3"],
    )
    pg_session.add(event)
    await pg_session.commit()
    await pg_session.refresh(event)

    assert event.affected_photo_ids == [101, 102]
    assert event.affected_face_instance_ids is not None
    assert len(event.affected_face_instance_ids) == 3


@pytest.mark.postgres
async def test_system_config_allowed_values_jsonb(pg_session):  # type: ignore
    """Verify SystemConfig.allowed_values JSONB stores string arrays."""
    config = SystemConfig(
        key="face_detection_model",
        value="buffalo_l",
        data_type="string",
        description="Face detection model name",
        allowed_values=["buffalo_l", "buffalo_s", "antelopev2"],
        category="face_recognition",
    )
    pg_session.add(config)
    await pg_session.commit()
    await pg_session.refresh(config)

    assert config.allowed_values == ["buffalo_l", "buffalo_s", "antelopev2"]
    assert "buffalo_l" in config.allowed_values
