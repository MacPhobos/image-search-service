"""Unit tests for RRF fusion service."""

import pytest
from pydantic import BaseModel

from image_search_service.services.fusion import (
    FusedResult,
    RankedItem,
    reciprocal_rank_fusion,
    weighted_reciprocal_rank_fusion,
)


class MockAsset(BaseModel):
    """Mock asset for testing."""

    id: int
    name: str


def test_rrf_empty_lists() -> None:
    """Test RRF with empty input returns empty results."""
    result = reciprocal_rank_fusion([])
    assert result == []


def test_rrf_single_list() -> None:
    """Test RRF with single ranked list returns same order."""
    assets = [
        MockAsset(id=1, name="asset1"),
        MockAsset(id=2, name="asset2"),
        MockAsset(id=3, name="asset3"),
    ]

    ranked = [
        RankedItem(item=assets[0], rank=1, score=0.9, source="text"),
        RankedItem(item=assets[1], rank=2, score=0.8, source="text"),
        RankedItem(item=assets[2], rank=3, score=0.7, source="text"),
    ]

    fused = reciprocal_rank_fusion([ranked])

    assert len(fused) == 3
    assert fused[0].item.id == 1  # Highest rank
    assert fused[0].combined_rank == 1
    assert fused[1].item.id == 2
    assert fused[1].combined_rank == 2
    assert fused[2].item.id == 3
    assert fused[2].combined_rank == 3

    # Check RRF scores are computed correctly (1/(k+rank), k=60)
    assert abs(fused[0].rrf_score - 1.0 / 61) < 0.0001
    assert abs(fused[1].rrf_score - 1.0 / 62) < 0.0001
    assert abs(fused[2].rrf_score - 1.0 / 63) < 0.0001


def test_rrf_two_lists_same_order() -> None:
    """Test RRF with two lists having same ranking."""
    asset1 = MockAsset(id=1, name="asset1")
    asset2 = MockAsset(id=2, name="asset2")

    text_ranked = [
        RankedItem(item=asset1, rank=1, score=0.9, source="text"),
        RankedItem(item=asset2, rank=2, score=0.8, source="text"),
    ]

    image_ranked = [
        RankedItem(item=asset1, rank=1, score=0.95, source="image"),
        RankedItem(item=asset2, rank=2, score=0.85, source="image"),
    ]

    fused = reciprocal_rank_fusion([text_ranked, image_ranked], k=60)

    assert len(fused) == 2
    assert fused[0].item.id == 1  # Still highest after fusion
    assert fused[1].item.id == 2

    # Asset 1 appears at rank 1 in both sources
    assert fused[0].ranks == {"text": 1, "image": 1}
    assert fused[0].scores == {"text": 0.9, "image": 0.95}

    # RRF score should be sum of 1/(60+1) from both sources
    expected_score = 2 * (1.0 / 61)
    assert abs(fused[0].rrf_score - expected_score) < 0.0001


def test_rrf_two_lists_reversed_order() -> None:
    """Test RRF correctly fuses when rankings disagree."""
    asset1 = MockAsset(id=1, name="asset1")
    asset2 = MockAsset(id=2, name="asset2")

    # Text search ranks asset1 first
    text_ranked = [
        RankedItem(item=asset1, rank=1, score=0.9, source="text"),
        RankedItem(item=asset2, rank=2, score=0.8, source="text"),
    ]

    # Image search ranks asset2 first
    image_ranked = [
        RankedItem(item=asset2, rank=1, score=0.95, source="image"),
        RankedItem(item=asset1, rank=2, score=0.85, source="image"),
    ]

    fused = reciprocal_rank_fusion([text_ranked, image_ranked], k=60)

    assert len(fused) == 2

    # Both assets get same RRF score (sum of ranks 1 and 2)
    # Asset 1: 1/(60+1) + 1/(60+2) = 1/61 + 1/62
    # Asset 2: 1/(60+2) + 1/(60+1) = 1/62 + 1/61
    # Scores are equal, so order is arbitrary (but stable)
    assert abs(fused[0].rrf_score - fused[1].rrf_score) < 0.0001


def test_rrf_non_overlapping_results() -> None:
    """Test RRF handles items that appear in only one source."""
    asset1 = MockAsset(id=1, name="asset1")
    asset2 = MockAsset(id=2, name="asset2")
    asset3 = MockAsset(id=3, name="asset3")

    # Text search finds asset1 and asset2
    text_ranked = [
        RankedItem(item=asset1, rank=1, score=0.9, source="text"),
        RankedItem(item=asset2, rank=2, score=0.8, source="text"),
    ]

    # Image search finds asset2 and asset3
    image_ranked = [
        RankedItem(item=asset2, rank=1, score=0.95, source="image"),
        RankedItem(item=asset3, rank=2, score=0.85, source="image"),
    ]

    fused = reciprocal_rank_fusion([text_ranked, image_ranked], k=60)

    assert len(fused) == 3  # Union of all assets

    # Asset 2 appears in both, should have highest RRF score
    asset2_result = next(r for r in fused if r.item.id == 2)
    assert asset2_result.combined_rank == 1  # Should be first
    assert "text" in asset2_result.ranks
    assert "image" in asset2_result.ranks

    # Asset 1 appears only in text search
    asset1_result = next(r for r in fused if r.item.id == 1)
    assert "text" in asset1_result.ranks
    assert "image" not in asset1_result.ranks

    # Asset 3 appears only in image search
    asset3_result = next(r for r in fused if r.item.id == 3)
    assert "image" in asset3_result.ranks
    assert "text" not in asset3_result.ranks


def test_rrf_k_parameter_reduces_rank_differences() -> None:
    """Test that higher k values reduce impact of rank differences."""
    asset1 = MockAsset(id=1, name="asset1")
    asset2 = MockAsset(id=2, name="asset2")

    ranked = [
        RankedItem(item=asset1, rank=1, score=0.9, source="text"),
        RankedItem(item=asset2, rank=10, score=0.5, source="text"),
    ]

    # Small k (more sensitive to rank)
    fused_small_k = reciprocal_rank_fusion([ranked], k=1)
    score_ratio_small = fused_small_k[0].rrf_score / fused_small_k[1].rrf_score

    # Large k (less sensitive to rank)
    fused_large_k = reciprocal_rank_fusion([ranked], k=1000)
    score_ratio_large = fused_large_k[0].rrf_score / fused_large_k[1].rrf_score

    # Ratio should be smaller with large k (scores more similar)
    assert score_ratio_small > score_ratio_large


def test_weighted_rrf_equal_weights() -> None:
    """Test weighted RRF with equal weights matches standard RRF."""
    asset1 = MockAsset(id=1, name="asset1")

    text_ranked = [RankedItem(item=asset1, rank=1, score=0.9, source="text")]
    image_ranked = [RankedItem(item=asset1, rank=1, score=0.95, source="image")]

    # Standard RRF
    standard = reciprocal_rank_fusion([text_ranked, image_ranked], k=60)

    # Weighted RRF with equal weights
    weighted = weighted_reciprocal_rank_fusion(
        [text_ranked, image_ranked], source_weights={"text": 1.0, "image": 1.0}, k=60
    )

    # Scores should be equal
    assert abs(standard[0].rrf_score - weighted[0].rrf_score) < 0.0001


def test_weighted_rrf_text_heavy() -> None:
    """Test weighted RRF boosts text results."""
    asset1 = MockAsset(id=1, name="asset1")
    asset2 = MockAsset(id=2, name="asset2")

    # Text search ranks asset1 first
    text_ranked = [
        RankedItem(item=asset1, rank=1, score=0.9, source="text"),
        RankedItem(item=asset2, rank=2, score=0.8, source="text"),
    ]

    # Image search ranks asset2 first
    image_ranked = [
        RankedItem(item=asset2, rank=1, score=0.95, source="image"),
        RankedItem(item=asset1, rank=2, score=0.85, source="image"),
    ]

    # Heavy text weighting (70% text, 30% image)
    fused = weighted_reciprocal_rank_fusion(
        [text_ranked, image_ranked], source_weights={"text": 0.7, "image": 0.3}, k=60
    )

    # With text weighted higher, asset1 (ranked #1 in text) should win
    assert fused[0].item.id == 1


def test_weighted_rrf_image_heavy() -> None:
    """Test weighted RRF boosts image results."""
    asset1 = MockAsset(id=1, name="asset1")
    asset2 = MockAsset(id=2, name="asset2")

    # Text search ranks asset1 first
    text_ranked = [
        RankedItem(item=asset1, rank=1, score=0.9, source="text"),
        RankedItem(item=asset2, rank=2, score=0.8, source="text"),
    ]

    # Image search ranks asset2 first
    image_ranked = [
        RankedItem(item=asset2, rank=1, score=0.95, source="image"),
        RankedItem(item=asset1, rank=2, score=0.85, source="image"),
    ]

    # Heavy image weighting (30% text, 70% image)
    fused = weighted_reciprocal_rank_fusion(
        [text_ranked, image_ranked], source_weights={"text": 0.3, "image": 0.7}, k=60
    )

    # With image weighted higher, asset2 (ranked #1 in image) should win
    assert fused[0].item.id == 2


def test_weighted_rrf_missing_source_weight_defaults_to_one() -> None:
    """Test that missing source weights default to 1.0."""
    asset1 = MockAsset(id=1, name="asset1")

    text_ranked = [RankedItem(item=asset1, rank=1, score=0.9, source="text")]
    image_ranked = [RankedItem(item=asset1, rank=2, score=0.85, source="image")]

    # Only provide weight for text, image should default to 1.0
    fused = weighted_reciprocal_rank_fusion(
        [text_ranked, image_ranked], source_weights={"text": 2.0}, k=60  # Double text weight
    )

    # RRF score should be: 2.0 * 1/(60+1) + 1.0 * 1/(60+2)
    expected = 2.0 / 61 + 1.0 / 62
    assert abs(fused[0].rrf_score - expected) < 0.0001


def test_fused_result_metadata() -> None:
    """Test that FusedResult captures all metadata correctly."""
    asset1 = MockAsset(id=1, name="asset1")

    text_ranked = [RankedItem(item=asset1, rank=1, score=0.9, source="text")]
    image_ranked = [RankedItem(item=asset1, rank=3, score=0.75, source="image")]

    fused = reciprocal_rank_fusion([text_ranked, image_ranked], k=60)

    result = fused[0]
    assert result.item.id == 1
    assert result.ranks == {"text": 1, "image": 3}
    assert result.scores == {"text": 0.9, "image": 0.75}
    assert result.combined_rank == 1
    assert result.rrf_score > 0


def test_rrf_score_ordering() -> None:
    """Test that results are correctly ordered by RRF score."""
    assets = [MockAsset(id=i, name=f"asset{i}") for i in range(1, 6)]

    # Create ranking where asset 3 appears in both sources at good ranks
    text_ranked = [
        RankedItem(item=assets[2], rank=1, score=0.9, source="text"),  # asset 3
        RankedItem(item=assets[0], rank=2, score=0.8, source="text"),  # asset 1
        RankedItem(item=assets[1], rank=3, score=0.7, source="text"),  # asset 2
    ]

    image_ranked = [
        RankedItem(item=assets[2], rank=1, score=0.95, source="image"),  # asset 3
        RankedItem(item=assets[3], rank=2, score=0.85, source="image"),  # asset 4
        RankedItem(item=assets[4], rank=3, score=0.75, source="image"),  # asset 5
    ]

    fused = reciprocal_rank_fusion([text_ranked, image_ranked], k=60)

    # Asset 3 should be first (appears at rank 1 in both)
    assert fused[0].item.id == 3
    assert fused[0].combined_rank == 1

    # Verify RRF scores are monotonically decreasing
    for i in range(len(fused) - 1):
        assert fused[i].rrf_score >= fused[i + 1].rrf_score

    # Verify combined ranks are sequential
    for i, result in enumerate(fused, start=1):
        assert result.combined_rank == i
