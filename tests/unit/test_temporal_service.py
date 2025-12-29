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

    def test_infant_age(self):
        assert classify_age_era(2) == AgeEraBucket.INFANT

    def test_child_age(self):
        assert classify_age_era(8) == AgeEraBucket.CHILD

    def test_teen_age(self):
        assert classify_age_era(15) == AgeEraBucket.TEEN

    def test_young_adult_age(self):
        assert classify_age_era(25) == AgeEraBucket.YOUNG_ADULT

    def test_adult_age(self):
        assert classify_age_era(45) == AgeEraBucket.ADULT

    def test_senior_age(self):
        assert classify_age_era(65) == AgeEraBucket.SENIOR

    def test_none_age(self):
        assert classify_age_era(None) is None

    def test_boundary_infant_child(self):
        # Test exact boundary between infant (<=3) and child (4-12)
        assert classify_age_era(3) == AgeEraBucket.INFANT
        assert classify_age_era(4) == AgeEraBucket.CHILD

    def test_boundary_child_teen(self):
        # Test exact boundary between child (4-12) and teen (13-19)
        assert classify_age_era(12) == AgeEraBucket.CHILD
        assert classify_age_era(13) == AgeEraBucket.TEEN

    def test_boundary_teen_young_adult(self):
        # Test exact boundary between teen (13-19) and young_adult (20-35)
        assert classify_age_era(19) == AgeEraBucket.TEEN
        assert classify_age_era(20) == AgeEraBucket.YOUNG_ADULT

    def test_boundary_young_adult_adult(self):
        # Test exact boundary between young_adult (20-35) and adult (36-55)
        assert classify_age_era(35) == AgeEraBucket.YOUNG_ADULT
        assert classify_age_era(36) == AgeEraBucket.ADULT

    def test_boundary_adult_senior(self):
        # Test exact boundary between adult (36-55) and senior (56+)
        assert classify_age_era(55) == AgeEraBucket.ADULT
        assert classify_age_era(56) == AgeEraBucket.SENIOR

    def test_zero_age(self):
        assert classify_age_era(0) == AgeEraBucket.INFANT

    def test_very_old_age(self):
        assert classify_age_era(100) == AgeEraBucket.SENIOR


class TestGetEraAgeRange:
    """Tests for getting age range for era buckets."""

    def test_infant_range(self):
        min_age, max_age = get_era_age_range(AgeEraBucket.INFANT)
        assert min_age == 0
        assert max_age == 3

    def test_child_range(self):
        min_age, max_age = get_era_age_range(AgeEraBucket.CHILD)
        assert min_age == 4
        assert max_age == 12

    def test_teen_range(self):
        min_age, max_age = get_era_age_range(AgeEraBucket.TEEN)
        assert min_age == 13
        assert max_age == 19

    def test_young_adult_range(self):
        min_age, max_age = get_era_age_range(AgeEraBucket.YOUNG_ADULT)
        assert min_age == 20
        assert max_age == 35

    def test_adult_range(self):
        min_age, max_age = get_era_age_range(AgeEraBucket.ADULT)
        assert min_age == 36
        assert max_age == 55

    def test_senior_range(self):
        min_age, max_age = get_era_age_range(AgeEraBucket.SENIOR)
        assert min_age == 56
        assert max_age == 120


class TestExtractDecade:
    """Tests for decade extraction from timestamps."""

    def test_1990s(self):
        assert extract_decade_from_timestamp(datetime(1995, 6, 15)) == "1990s"

    def test_2000s(self):
        assert extract_decade_from_timestamp(datetime(2005, 1, 1)) == "2000s"

    def test_2020s(self):
        assert extract_decade_from_timestamp(datetime(2024, 12, 31)) == "2020s"

    def test_none_timestamp(self):
        assert extract_decade_from_timestamp(None) is None

    def test_decade_boundaries(self):
        # Test exact decade boundaries
        assert extract_decade_from_timestamp(datetime(1999, 12, 31)) == "1990s"
        assert extract_decade_from_timestamp(datetime(2000, 1, 1)) == "2000s"
        assert extract_decade_from_timestamp(datetime(2009, 12, 31)) == "2000s"
        assert extract_decade_from_timestamp(datetime(2010, 1, 1)) == "2010s"

    def test_historical_decades(self):
        assert extract_decade_from_timestamp(datetime(1980, 5, 10)) == "1980s"
        assert extract_decade_from_timestamp(datetime(1970, 3, 25)) == "1970s"


class TestTemporalQualityScore:
    """Tests for temporal quality score computation."""

    def test_base_quality_only(self):
        score = compute_temporal_quality_score(0.8)
        assert score == pytest.approx(0.48)  # 60% of 0.8

    def test_frontal_bonus(self):
        score = compute_temporal_quality_score(0.7, pose="frontal")
        assert score == pytest.approx(0.62)  # (0.7 * 0.6) + 0.2

    def test_large_bbox_bonus(self):
        score = compute_temporal_quality_score(0.7, bbox_area=15000)
        assert score == pytest.approx(0.52)  # (0.7 * 0.6) + 0.1

    def test_age_confidence_bonus(self):
        score = compute_temporal_quality_score(0.7, age_confidence=0.9)
        assert score == pytest.approx(0.51)  # (0.7 * 0.6) + (0.9 * 0.1)

    def test_all_bonuses(self):
        score = compute_temporal_quality_score(
            0.8, pose="frontal", bbox_area=15000, age_confidence=1.0
        )
        assert score == pytest.approx(0.88)  # (0.8 * 0.6) + 0.2 + 0.1 + 0.1

    def test_max_score_capped(self):
        score = compute_temporal_quality_score(
            1.0, pose="frontal", bbox_area=20000, age_confidence=1.0
        )
        assert score <= 1.0
        assert score == 1.0

    def test_none_base_quality(self):
        score = compute_temporal_quality_score(None, pose="frontal")
        assert score == pytest.approx(0.2)  # Only frontal bonus

    def test_small_bbox_no_bonus(self):
        score = compute_temporal_quality_score(0.7, bbox_area=5000)
        assert score == pytest.approx(0.42)  # No bbox bonus

    def test_non_frontal_pose(self):
        score = compute_temporal_quality_score(0.7, pose="profile")
        assert score == pytest.approx(0.42)  # No pose bonus

    def test_zero_quality(self):
        score = compute_temporal_quality_score(0.0)
        assert score == 0.0


class TestCoverageGaps:
    """Tests for coverage gap detection."""

    def test_full_coverage(self):
        all_eras = {e.value for e in AgeEraBucket}
        assert get_coverage_gaps(all_eras) == []

    def test_missing_infant(self):
        existing = {"child", "teen", "young_adult", "adult", "senior"}
        gaps = get_coverage_gaps(existing)
        assert AgeEraBucket.INFANT in gaps
        assert len(gaps) == 1

    def test_missing_multiple(self):
        existing = {"child", "adult"}
        gaps = get_coverage_gaps(existing)
        assert len(gaps) == 4
        assert AgeEraBucket.INFANT in gaps
        assert AgeEraBucket.TEEN in gaps
        assert AgeEraBucket.YOUNG_ADULT in gaps
        assert AgeEraBucket.SENIOR in gaps

    def test_empty_coverage(self):
        gaps = get_coverage_gaps(set())
        assert len(gaps) == 6
        # Verify all eras are returned
        assert AgeEraBucket.INFANT in gaps
        assert AgeEraBucket.CHILD in gaps
        assert AgeEraBucket.TEEN in gaps
        assert AgeEraBucket.YOUNG_ADULT in gaps
        assert AgeEraBucket.ADULT in gaps
        assert AgeEraBucket.SENIOR in gaps


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
