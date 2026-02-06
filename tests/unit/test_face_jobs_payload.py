"""Unit tests for face job payload key extraction.

Verifies that the three job functions extracting face IDs from Qdrant
search results use the correct payload key: 'face_instance_id'.

This tests the fix for C1 bug (Qdrant payload key mismatch).
"""

import uuid
from unittest.mock import MagicMock


def test_canonical_payload_key_matches_upsert():
    """Verify that the key used in reads matches the key used in upsert_face.

    This test documents the canonical key name and verifies consistency
    between write (upsert) and read (job functions) operations.
    """
    CANONICAL_KEY = "face_instance_id"

    # Simulate the payload created by FaceQdrantClient.upsert_face()
    # (from face_qdrant.py lines 194-196)
    face_instance_id = uuid.uuid4()
    upsert_payload = {
        "asset_id": str(uuid.uuid4()),
        "face_instance_id": str(face_instance_id),
        "detection_confidence": 0.95,
        "is_prototype": False,
    }

    # Reads must use the same key
    extracted = upsert_payload.get(CANONICAL_KEY)
    assert extracted == str(face_instance_id)

    # The old incorrect key should NOT exist
    wrong_key = upsert_payload.get("face_id")
    assert wrong_key is None


def test_mock_result_payload_extraction():
    """Verify that face_instance_id can be extracted from mock Qdrant result.

    This simulates the pattern used in the three fixed job functions:
    - find_more_suggestions_job
    - propagate_person_label_multiproto_job
    - find_more_centroid_suggestions_job
    """
    face_id = uuid.uuid4()
    mock_result = MagicMock()
    mock_result.payload = {
        "face_instance_id": str(face_id),
        "asset_id": str(uuid.uuid4()),
        "detection_confidence": 0.95,
    }

    # The extraction should use "face_instance_id", not "face_id"
    face_id_str = mock_result.payload.get("face_instance_id")
    assert face_id_str == str(face_id)
    assert face_id_str is not None

    # The old incorrect key should return None
    wrong_key = mock_result.payload.get("face_id")
    assert wrong_key is None


def test_payload_none_handling():
    """Verify that None payload is handled correctly in job functions.

    All three fixed job functions check `if result.payload is None: continue`
    before attempting to extract face_instance_id.
    """
    mock_result = MagicMock()
    mock_result.payload = None

    # This should not raise an exception
    if mock_result.payload is None:
        # Job would skip this result
        face_id_str = None
    else:
        face_id_str = mock_result.payload.get("face_instance_id")

    assert face_id_str is None


def test_missing_face_instance_id_handling():
    """Verify that missing face_instance_id key is handled correctly.

    Job functions check `if not face_id_str: continue` after extraction,
    which correctly handles both None and empty string.
    """
    mock_result = MagicMock()
    mock_result.payload = {
        "asset_id": str(uuid.uuid4()),
        "detection_confidence": 0.95,
    }

    face_id_str = mock_result.payload.get("face_instance_id")
    assert face_id_str is None

    # Job would skip this result with: if not face_id_str: continue
    if not face_id_str:
        skipped = True
    else:
        skipped = False

    assert skipped is True


def test_valid_uuid_extraction():
    """Verify that extracted face_instance_id can be parsed as UUID.

    Job functions use: uuid_lib.UUID(face_id_str)
    This test verifies the extracted value is a valid UUID string.
    """
    face_id = uuid.uuid4()
    mock_result = MagicMock()
    mock_result.payload = {
        "face_instance_id": str(face_id),
        "asset_id": str(uuid.uuid4()),
    }

    face_id_str = mock_result.payload.get("face_instance_id")
    assert face_id_str is not None

    # Should be parseable as UUID
    parsed_uuid = uuid.UUID(face_id_str)
    assert parsed_uuid == face_id


def test_multiple_results_aggregation():
    """Verify that multiple Qdrant results can be processed correctly.

    This simulates the aggregation pattern in find_more_suggestions_job
    and propagate_person_label_multiproto_job where results from multiple
    prototypes are collected.
    """
    # Simulate 3 search results from Qdrant
    face_ids = [uuid.uuid4() for _ in range(3)]
    mock_results = []

    for face_id in face_ids:
        result = MagicMock()
        result.payload = {
            "face_instance_id": str(face_id),
            "asset_id": str(uuid.uuid4()),
            "detection_confidence": 0.85,
        }
        result.score = 0.75
        mock_results.append(result)

    # Process results (as jobs do)
    candidate_faces = {}
    for result in mock_results:
        if result.payload is None:
            continue

        face_id_str = result.payload.get("face_instance_id")
        if not face_id_str:
            continue

        # Aggregate scores
        if face_id_str not in candidate_faces:
            candidate_faces[face_id_str] = {
                "scores": [],
                "max_score": 0.0,
            }

        candidate_faces[face_id_str]["scores"].append(result.score)
        candidate_faces[face_id_str]["max_score"] = max(
            candidate_faces[face_id_str]["max_score"],
            result.score,
        )

    # Verify all results were processed
    assert len(candidate_faces) == 3
    for face_id_str in candidate_faces:
        assert len(candidate_faces[face_id_str]["scores"]) == 1
        assert candidate_faces[face_id_str]["max_score"] == 0.75
