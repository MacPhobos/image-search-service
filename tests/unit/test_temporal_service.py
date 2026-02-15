"""Tests for temporal classification service."""

from datetime import datetime

import pytest

from image_search_service.db.models import AgeEraBucket
from image_search_service.services.temporal_service import (
    classify_age_era,
    compute_temporal_quality_score,
    enrich_face_with_temporal_data,
    estimate_person_birth_year,
    extract_decade_from_timestamp,
    extract_temporal_metadata,
    get_coverage_gaps,
    get_era_age_range,
)


class TestClassifyAgeEra:
    """Tests for age era classification."""

    @pytest.mark.parametrize(
        "age,expected",
        [
            pytest.param(0, AgeEraBucket.INFANT, id="zero-age"),
            pytest.param(2, AgeEraBucket.INFANT, id="infant"),
            pytest.param(3, AgeEraBucket.INFANT, id="infant-upper-bound"),
            pytest.param(4, AgeEraBucket.CHILD, id="child-lower-bound"),
            pytest.param(8, AgeEraBucket.CHILD, id="child"),
            pytest.param(12, AgeEraBucket.CHILD, id="child-upper-bound"),
            pytest.param(13, AgeEraBucket.TEEN, id="teen-lower-bound"),
            pytest.param(15, AgeEraBucket.TEEN, id="teen"),
            pytest.param(19, AgeEraBucket.TEEN, id="teen-upper-bound"),
            pytest.param(20, AgeEraBucket.YOUNG_ADULT, id="young-adult-lower-bound"),
            pytest.param(25, AgeEraBucket.YOUNG_ADULT, id="young-adult"),
            pytest.param(35, AgeEraBucket.YOUNG_ADULT, id="young-adult-upper-bound"),
            pytest.param(36, AgeEraBucket.ADULT, id="adult-lower-bound"),
            pytest.param(45, AgeEraBucket.ADULT, id="adult"),
            pytest.param(55, AgeEraBucket.ADULT, id="adult-upper-bound"),
            pytest.param(56, AgeEraBucket.SENIOR, id="senior-lower-bound"),
            pytest.param(65, AgeEraBucket.SENIOR, id="senior"),
            pytest.param(100, AgeEraBucket.SENIOR, id="very-old"),
            pytest.param(None, None, id="none-age"),
        ],
    )
    def test_classify_age_era(self, age, expected):
        assert classify_age_era(age) == expected


class TestGetEraAgeRange:
    """Tests for getting age range for era buckets."""

    @pytest.mark.parametrize(
        "era,expected_min,expected_max",
        [
            pytest.param(AgeEraBucket.INFANT, 0, 3, id="infant"),
            pytest.param(AgeEraBucket.CHILD, 4, 12, id="child"),
            pytest.param(AgeEraBucket.TEEN, 13, 19, id="teen"),
            pytest.param(AgeEraBucket.YOUNG_ADULT, 20, 35, id="young-adult"),
            pytest.param(AgeEraBucket.ADULT, 36, 55, id="adult"),
            pytest.param(AgeEraBucket.SENIOR, 56, 120, id="senior"),
        ],
    )
    def test_get_era_age_range(self, era, expected_min, expected_max):
        min_age, max_age = get_era_age_range(era)
        assert min_age == expected_min
        assert max_age == expected_max


class TestExtractDecade:
    """Tests for decade extraction from timestamps."""

    @pytest.mark.parametrize(
        "timestamp,expected",
        [
            pytest.param(datetime(1970, 3, 25), "1970s", id="1970s"),
            pytest.param(datetime(1980, 5, 10), "1980s", id="1980s"),
            pytest.param(datetime(1995, 6, 15), "1990s", id="1990s"),
            pytest.param(datetime(1999, 12, 31), "1990s", id="1990s-end"),
            pytest.param(datetime(2000, 1, 1), "2000s", id="2000s-start"),
            pytest.param(datetime(2005, 1, 1), "2000s", id="2000s"),
            pytest.param(datetime(2009, 12, 31), "2000s", id="2000s-end"),
            pytest.param(datetime(2010, 1, 1), "2010s", id="2010s-start"),
            pytest.param(datetime(2024, 12, 31), "2020s", id="2020s"),
            pytest.param(None, None, id="none"),
        ],
    )
    def test_extract_decade(self, timestamp, expected):
        assert extract_decade_from_timestamp(timestamp) == expected


class TestTemporalQualityScore:
    """Tests for temporal quality score computation."""

    @pytest.mark.parametrize(
        "kwargs,expected",
        [
            pytest.param(
                {"base_quality": 0.8},
                0.48,
                id="base-only",
            ),
            pytest.param(
                {"base_quality": 0.7, "pose": "frontal"},
                0.62,
                id="frontal-bonus",
            ),
            pytest.param(
                {"base_quality": 0.7, "bbox_area": 15000},
                0.52,
                id="large-bbox-bonus",
            ),
            pytest.param(
                {"base_quality": 0.7, "age_confidence": 0.9},
                0.51,
                id="age-confidence-bonus",
            ),
            pytest.param(
                {"base_quality": 0.8, "pose": "frontal", "bbox_area": 15000, "age_confidence": 1.0},
                0.88,
                id="all-bonuses",
            ),
            pytest.param(
                {"base_quality": 1.0, "pose": "frontal", "bbox_area": 20000, "age_confidence": 1.0},
                1.0,
                id="max-score-capped",
            ),
            pytest.param(
                {"base_quality": None, "pose": "frontal"},
                0.2,
                id="none-base-quality",
            ),
            pytest.param(
                {"base_quality": 0.7, "bbox_area": 5000},
                0.42,
                id="small-bbox-no-bonus",
            ),
            pytest.param(
                {"base_quality": 0.7, "pose": "profile"},
                0.42,
                id="non-frontal-no-bonus",
            ),
            pytest.param(
                {"base_quality": 0.0},
                0.0,
                id="zero-quality",
            ),
        ],
    )
    def test_temporal_quality_score(self, kwargs, expected):
        score = compute_temporal_quality_score(**kwargs)
        assert score == pytest.approx(expected)
        assert score <= 1.0


class TestCoverageGaps:
    """Tests for coverage gap detection."""

    @pytest.mark.parametrize(
        "existing,expected_gaps",
        [
            pytest.param(
                {e.value for e in AgeEraBucket},
                [],
                id="full-coverage",
            ),
            pytest.param(
                {"child", "teen", "young_adult", "adult", "senior"},
                [AgeEraBucket.INFANT],
                id="missing-infant",
            ),
            pytest.param(
                {"child", "adult"},
                [
                    AgeEraBucket.INFANT,
                    AgeEraBucket.TEEN,
                    AgeEraBucket.YOUNG_ADULT,
                    AgeEraBucket.SENIOR,
                ],
                id="missing-multiple",
            ),
            pytest.param(
                set(),
                [
                    AgeEraBucket.INFANT,
                    AgeEraBucket.CHILD,
                    AgeEraBucket.TEEN,
                    AgeEraBucket.YOUNG_ADULT,
                    AgeEraBucket.ADULT,
                    AgeEraBucket.SENIOR,
                ],
                id="empty-coverage",
            ),
        ],
    )
    def test_coverage_gaps(self, existing, expected_gaps):
        gaps = get_coverage_gaps(existing)
        assert set(gaps) == set(expected_gaps)
        assert len(gaps) == len(expected_gaps)


class TestEstimatePersonBirthYear:
    """Tests for birth year estimation."""

    def test_single_face(self):
        faces = [{"photo_timestamp": datetime(2020, 6, 15), "age_estimate": 30}]
        birth_year = estimate_person_birth_year(faces)
        assert birth_year == 1990

    def test_multiple_faces_consistent(self):
        faces = [
            {"photo_timestamp": datetime(2020, 1, 1), "age_estimate": 30},
            {"photo_timestamp": datetime(2015, 1, 1), "age_estimate": 25},
            {"photo_timestamp": datetime(2010, 1, 1), "age_estimate": 20},
        ]
        birth_year = estimate_person_birth_year(faces)
        assert birth_year == 1990

    def test_multiple_faces_outlier(self):
        faces = [
            {"photo_timestamp": datetime(2020, 1, 1), "age_estimate": 30},
            {"photo_timestamp": datetime(2015, 1, 1), "age_estimate": 25},
            {"photo_timestamp": datetime(2010, 1, 1), "age_estimate": 15},  # Outlier
        ]
        # Median should handle outlier
        birth_year = estimate_person_birth_year(faces)
        assert birth_year in [1990, 1995]  # Median of [1990, 1990, 1995]

    def test_no_faces(self):
        assert estimate_person_birth_year([]) is None

    def test_faces_missing_timestamp(self):
        faces = [{"photo_timestamp": None, "age_estimate": 30}]
        assert estimate_person_birth_year(faces) is None

    def test_faces_missing_age(self):
        faces = [{"photo_timestamp": datetime(2020, 1, 1), "age_estimate": None}]
        assert estimate_person_birth_year(faces) is None

    def test_mixed_valid_invalid(self):
        faces = [
            {"photo_timestamp": datetime(2020, 1, 1), "age_estimate": 30},
            {"photo_timestamp": None, "age_estimate": 25},
            {"photo_timestamp": datetime(2015, 1, 1), "age_estimate": None},
        ]
        # Should use only valid face
        birth_year = estimate_person_birth_year(faces)
        assert birth_year == 1990


class TestExtractTemporalMetadata:
    """Tests for temporal metadata extraction."""

    def test_none_landmarks(self):
        metadata = extract_temporal_metadata(None)
        assert metadata == {
            "age_estimate": None,
            "age_confidence": None,
            "pose": None,
            "bbox_area": None,
        }

    def test_full_metadata(self):
        landmarks = {
            "age_estimate": 25,
            "age_confidence": 0.9,
            "pose": "frontal",
            "bbox": {"width": 150, "height": 200},
        }
        metadata = extract_temporal_metadata(landmarks)
        assert metadata["age_estimate"] == 25
        assert metadata["age_confidence"] == 0.9
        assert metadata["pose"] == "frontal"
        assert metadata["bbox_area"] == 30000

    def test_missing_age(self):
        landmarks = {"pose": "frontal", "bbox": {"width": 150, "height": 200}}
        metadata = extract_temporal_metadata(landmarks)
        assert metadata["age_estimate"] is None
        assert metadata["age_confidence"] is None

    def test_missing_bbox(self):
        landmarks = {"age_estimate": 25, "pose": "frontal"}
        metadata = extract_temporal_metadata(landmarks)
        assert metadata["bbox_area"] is None

    def test_incomplete_bbox(self):
        landmarks = {
            "bbox": {"width": 150}  # Missing height
        }
        metadata = extract_temporal_metadata(landmarks)
        assert metadata["bbox_area"] is None


class TestEnrichFaceWithTemporalData:
    """Tests for face enrichment with temporal data."""

    def test_full_enrichment(self):
        landmarks = {
            "age_estimate": 25,
            "age_confidence": 0.9,
            "pose": "frontal",
            "bbox": {"width": 150, "height": 200},
            "quality_score": 0.8,
        }
        enriched = enrich_face_with_temporal_data(landmarks, datetime(2020, 6, 15))

        assert enriched["age_era_bucket"] == "young_adult"
        assert enriched["decade_bucket"] == "2020s"
        assert enriched["age_estimate"] == 25
        assert enriched["age_confidence"] == 0.9
        assert enriched["pose"] == "frontal"
        assert enriched["bbox_area"] == 30000
        assert enriched["temporal_quality_score"] > 0.0

    def test_no_landmarks(self):
        enriched = enrich_face_with_temporal_data(None, datetime(2020, 6, 15))
        assert enriched["age_era_bucket"] is None
        assert enriched["decade_bucket"] == "2020s"
        assert enriched["age_estimate"] is None

    def test_no_timestamp(self):
        landmarks = {"age_estimate": 25, "quality_score": 0.8}
        enriched = enrich_face_with_temporal_data(landmarks, None)
        assert enriched["age_era_bucket"] == "young_adult"
        assert enriched["decade_bucket"] is None

    def test_infant_classification(self):
        landmarks = {"age_estimate": 2, "quality_score": 0.7}
        enriched = enrich_face_with_temporal_data(landmarks, datetime(2020, 1, 1))
        assert enriched["age_era_bucket"] == "infant"

    def test_senior_classification(self):
        landmarks = {"age_estimate": 70, "quality_score": 0.7}
        enriched = enrich_face_with_temporal_data(landmarks, datetime(1990, 1, 1))
        assert enriched["age_era_bucket"] == "senior"
        assert enriched["decade_bucket"] == "1990s"
