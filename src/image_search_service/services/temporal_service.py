"""Temporal classification and era assignment for face instances."""

from datetime import datetime
from typing import Any

from image_search_service.core.config import get_settings
from image_search_service.db.models import AgeEraBucket


def classify_age_era(age_estimate: int | None) -> AgeEraBucket | None:
    """Classify age into era bucket based on config thresholds.

    Args:
        age_estimate: Estimated age in years, or None if unknown

    Returns:
        AgeEraBucket enum value, or None if age is unknown
    """
    if age_estimate is None:
        return None

    settings = get_settings()

    if age_estimate <= settings.age_era_infant_max:
        return AgeEraBucket.INFANT
    elif age_estimate <= settings.age_era_child_max:
        return AgeEraBucket.CHILD
    elif age_estimate <= settings.age_era_teen_max:
        return AgeEraBucket.TEEN
    elif age_estimate <= settings.age_era_young_adult_max:
        return AgeEraBucket.YOUNG_ADULT
    elif age_estimate <= settings.age_era_adult_max:
        return AgeEraBucket.ADULT
    else:
        return AgeEraBucket.SENIOR


def get_era_age_range(era: AgeEraBucket) -> tuple[int, int]:
    """Get the min/max age range for an era bucket.

    Returns:
        Tuple of (min_age, max_age) inclusive
    """
    settings = get_settings()

    if era == AgeEraBucket.INFANT:
        return (0, settings.age_era_infant_max)
    elif era == AgeEraBucket.CHILD:
        return (settings.age_era_infant_max + 1, settings.age_era_child_max)
    elif era == AgeEraBucket.TEEN:
        return (settings.age_era_child_max + 1, settings.age_era_teen_max)
    elif era == AgeEraBucket.YOUNG_ADULT:
        return (settings.age_era_teen_max + 1, settings.age_era_young_adult_max)
    elif era == AgeEraBucket.ADULT:
        return (settings.age_era_young_adult_max + 1, settings.age_era_adult_max)
    else:  # SENIOR
        return (settings.age_era_adult_max + 1, 120)


def extract_decade_from_timestamp(timestamp: datetime | None) -> str | None:
    """Extract decade bucket (e.g., '1990s', '2000s') from photo timestamp.

    Args:
        timestamp: Photo capture datetime, or None if unknown

    Returns:
        Decade string like '2000s', or None if timestamp unknown
    """
    if timestamp is None:
        return None

    year = timestamp.year
    decade_start = (year // 10) * 10
    return f"{decade_start}s"


def compute_temporal_quality_score(
    base_quality: float | None,
    pose: str | None = None,
    bbox_area: int | None = None,
    age_confidence: float | None = None,
) -> float:
    """Compute enhanced quality score for temporal prototype selection.

    Scoring weights:
    - Base quality: 60%
    - Frontal pose bonus: +0.2
    - Large bbox (>10000 px²) clarity bonus: +0.1
    - Age confidence bonus: up to +0.1

    Returns:
        Score between 0.0 and 1.0
    """
    # Start with base quality weighted at 60%
    score = (base_quality or 0.0) * 0.6

    # Frontal pose bonus
    if pose == "frontal":
        score += 0.2

    # Large bbox clarity bonus
    if bbox_area is not None and bbox_area > 10000:
        score += 0.1

    # Age confidence bonus (scaled to max 0.1)
    if age_confidence is not None:
        score += age_confidence * 0.1

    # Cap at 1.0
    return min(score, 1.0)


def estimate_person_birth_year(
    faces_with_metadata: list[dict[str, Any]],
) -> int | None:
    """Estimate person's birth year from face instances with age estimates.

    Uses photo timestamps and age estimates to triangulate birth year.

    Args:
        faces_with_metadata: List of dicts with 'photo_timestamp' and 'age_estimate'

    Returns:
        Estimated birth year, or None if insufficient data
    """
    if not faces_with_metadata:
        return None

    # Filter to faces with both timestamp and age
    valid_faces = [
        f
        for f in faces_with_metadata
        if f.get("photo_timestamp") is not None and f.get("age_estimate") is not None
    ]

    if not valid_faces:
        return None

    # Calculate birth year for each face: photo_year - age_estimate
    birth_years: list[int] = []
    for face in valid_faces:
        photo_timestamp = face["photo_timestamp"]
        age_estimate = face["age_estimate"]

        if isinstance(photo_timestamp, datetime):
            photo_year = photo_timestamp.year
            birth_year = photo_year - age_estimate
            birth_years.append(birth_year)

    if not birth_years:
        return None

    # Return median birth year to reduce outlier impact
    birth_years.sort()
    median_index = len(birth_years) // 2
    return birth_years[median_index]


def get_coverage_gaps(
    existing_eras: set[str],
) -> list[AgeEraBucket]:
    """Identify which era buckets are missing coverage.

    Args:
        existing_eras: Set of era bucket values that have prototypes

    Returns:
        List of AgeEraBucket values that need coverage
    """
    all_eras = {e.value for e in AgeEraBucket}
    missing_eras = all_eras - existing_eras

    # Return as AgeEraBucket enums
    return [AgeEraBucket(era) for era in missing_eras]


def extract_temporal_metadata(landmarks: dict[str, Any] | None) -> dict[str, Any]:
    """Extract temporal-relevant metadata from face landmarks.

    Returns dict with:
    - age_estimate: int | None
    - age_confidence: float | None
    - pose: str | None ('frontal', 'profile', 'three_quarter')
    - bbox_area: int | None
    """
    if landmarks is None:
        return {
            "age_estimate": None,
            "age_confidence": None,
            "pose": None,
            "bbox_area": None,
        }

    # Extract age metadata if present
    age_estimate = landmarks.get("age_estimate")
    age_confidence = landmarks.get("age_confidence")

    # Extract pose classification if present
    pose = landmarks.get("pose")

    # Extract bounding box area if present
    bbox = landmarks.get("bbox")
    bbox_area = None
    if bbox and isinstance(bbox, dict):
        width = bbox.get("width")
        height = bbox.get("height")
        if width is not None and height is not None:
            bbox_area = width * height

    return {
        "age_estimate": age_estimate,
        "age_confidence": age_confidence,
        "pose": pose,
        "bbox_area": bbox_area,
    }


def enrich_face_with_temporal_data(
    face_landmarks: dict[str, Any] | None,
    photo_timestamp: datetime | None,
) -> dict[str, Any]:
    """Add temporal classification to face metadata.

    Returns enriched dict with added fields:
    - age_era_bucket: str | None
    - decade_bucket: str | None
    - temporal_quality_score: float
    """
    # Extract metadata
    metadata = extract_temporal_metadata(face_landmarks)

    # Classify age era
    age_era = classify_age_era(metadata["age_estimate"])
    age_era_bucket = age_era.value if age_era else None

    # Extract decade
    decade_bucket = extract_decade_from_timestamp(photo_timestamp)

    # Compute quality score (need base quality from landmarks)
    base_quality = face_landmarks.get("quality_score") if face_landmarks else None
    temporal_quality_score = compute_temporal_quality_score(
        base_quality=base_quality,
        pose=metadata["pose"],
        bbox_area=metadata["bbox_area"],
        age_confidence=metadata["age_confidence"],
    )

    return {
        "age_era_bucket": age_era_bucket,
        "decade_bucket": decade_bucket,
        "temporal_quality_score": temporal_quality_score,
        **metadata,  # Include extracted metadata
    }
