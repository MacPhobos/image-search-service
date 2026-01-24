"""Reciprocal Rank Fusion (RRF) for combining multiple search result rankings.

RRF is a simple but effective method for combining ranked lists from different
retrieval systems. It addresses challenges like:
- Different score ranges across systems (CLIP vs SigLIP vs text search)
- Varying score distributions
- Non-overlapping result sets

Algorithm
=========
For each item that appears in any ranked list, compute:
    RRF_score = Σ 1 / (k + rank_i)

Where:
- k = constant (typically 60) to prevent division by zero and reduce outliers
- rank_i = position in ranking (1-indexed)
- Sum over all ranked lists where item appears

Benefits
========
- No need to normalize scores across different systems
- Handles missing items gracefully (items only in some lists)
- Proven effective in information retrieval (used in search engines, RAG systems)

Example
=======
Text search ranks: [A, B, C] → A=1, B=2, C=3
Image search ranks: [B, A, D] → B=1, A=2, D=3

RRF scores (k=60):
- A: 1/(60+1) + 1/(60+2) ≈ 0.0323
- B: 1/(60+2) + 1/(60+1) ≈ 0.0323
- C: 1/(60+3) ≈ 0.0159
- D: 1/(60+3) ≈ 0.0159

Final ranking: [A, B, C, D] or [B, A, C, D] (A and B tied)

References
==========
- Cormack, Gordon V., et al. "Reciprocal rank fusion outperforms condorcet
  and individual rank learning methods." SIGIR 2009.
"""

from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class RankedItem[T](BaseModel):
    """Single item with its rank in a result list.

    Attributes:
        item: The actual item (asset, document, etc.)
        rank: 1-indexed position in ranking (1 = first place)
        score: Optional original score from retrieval system
        source: Optional label for the ranking source (e.g., 'text', 'image')
    """

    item: T
    rank: int
    score: float | None = None
    source: str | None = None


class FusedResult[T](BaseModel):
    """Result after RRF fusion with aggregated scores.

    Attributes:
        item: The fused item
        rrf_score: Combined RRF score across all sources
        ranks: Dict mapping source name to rank in that source
        scores: Dict mapping source name to original score (if available)
        combined_rank: Final rank after fusion (1 = top result)
    """

    item: T
    rrf_score: float
    ranks: dict[str, int]  # source -> rank
    scores: dict[str, float]  # source -> score
    combined_rank: int


def reciprocal_rank_fusion[T](
    ranked_lists: list[list[RankedItem[T]]],
    k: int = 60,
) -> list[FusedResult[T]]:
    """Combine multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        ranked_lists: List of ranked lists to combine. Each inner list should be
            ordered by rank (best first). Items should have rank field set.
        k: Constant for RRF formula (default: 60). Higher k reduces impact of
            rank differences. Standard value is 60 from original RRF paper.

    Returns:
        Sorted list of FusedResult objects, ordered by RRF score (descending).
        Each result includes:
        - Combined RRF score
        - Ranks from each source
        - Original scores (if available)
        - Final combined rank

    Example:
        >>> from pydantic import BaseModel
        >>> class Asset(BaseModel):
        ...     id: int
        ...     path: str
        >>> text_results = [
        ...     RankedItem(item=Asset(id=1, path="a.jpg"), rank=1, score=0.9, source="text"),
        ...     RankedItem(item=Asset(id=2, path="b.jpg"), rank=2, score=0.8, source="text"),
        ... ]
        >>> image_results = [
        ...     RankedItem(item=Asset(id=2, path="b.jpg"), rank=1, score=0.95, source="image"),
        ...     RankedItem(item=Asset(id=1, path="a.jpg"), rank=2, score=0.85, source="image"),
        ... ]
        >>> fused = reciprocal_rank_fusion([text_results, image_results])
        >>> fused[0].item.id  # Asset 2 likely wins due to being #1 in image search
        2
    """
    if not ranked_lists:
        return []

    # Build mapping: item id -> (item, {source -> (rank, score)})
    # Use id() as key to handle unhashable Pydantic models
    item_rankings: dict[int, tuple[T, dict[str, tuple[int, float | None]]]] = {}

    for ranked_list in ranked_lists:
        for ranked_item in ranked_list:
            item = ranked_item.item
            item_id = id(item)  # Use object id as hash key

            if item_id not in item_rankings:
                item_rankings[item_id] = (item, {})

            source = ranked_item.source or "unknown"
            item_rankings[item_id][1][source] = (ranked_item.rank, ranked_item.score)

    # Calculate RRF scores
    fused_results: list[FusedResult[T]] = []

    for item_id, (item, source_ranks) in item_rankings.items():
        # Calculate RRF score: sum of 1/(k + rank) across all sources
        rrf_score = sum(1.0 / (k + rank) for rank, _ in source_ranks.values())

        # Extract ranks and scores by source
        ranks = {source: rank for source, (rank, _) in source_ranks.items()}
        scores = {source: score for source, (_, score) in source_ranks.items() if score is not None}

        fused_results.append(
            FusedResult(
                item=item,
                rrf_score=rrf_score,
                ranks=ranks,
                scores=scores,
                combined_rank=0,  # Will be set below
            )
        )

    # Sort by RRF score (descending) and assign combined ranks
    fused_results.sort(key=lambda x: x.rrf_score, reverse=True)

    for idx, result in enumerate(fused_results, start=1):
        result.combined_rank = idx

    return fused_results


def weighted_reciprocal_rank_fusion[T](
    ranked_lists: list[list[RankedItem[T]]],
    source_weights: dict[str, float] | None = None,
    k: int = 60,
) -> list[FusedResult[T]]:
    """Combine ranked lists with weighted RRF.

    Allows boosting certain sources over others. For example, in hybrid search,
    you might weight text search higher for keyword queries.

    Args:
        ranked_lists: List of ranked lists to combine
        source_weights: Dict mapping source name to weight multiplier.
            Default: all weights = 1.0. Higher weight = more influence.
            Example: {"text": 0.7, "image": 0.3} weights text 70%, image 30%
        k: Constant for RRF formula (default: 60)

    Returns:
        Sorted list of FusedResult objects by weighted RRF score

    Example:
        >>> # Boost text results over image results
        >>> fused = weighted_reciprocal_rank_fusion(
        ...     [text_results, image_results],
        ...     source_weights={"text": 0.7, "image": 0.3}
        ... )
    """
    if not ranked_lists:
        return []

    if source_weights is None:
        # Default: equal weights
        source_weights = {}

    # Build mapping: item id -> (item, {source -> (rank, score)})
    # Use id() as key to handle unhashable Pydantic models
    item_rankings: dict[int, tuple[T, dict[str, tuple[int, float | None]]]] = {}

    for ranked_list in ranked_lists:
        for ranked_item in ranked_list:
            item = ranked_item.item
            item_id = id(item)  # Use object id as hash key

            if item_id not in item_rankings:
                item_rankings[item_id] = (item, {})

            source = ranked_item.source or "unknown"
            item_rankings[item_id][1][source] = (ranked_item.rank, ranked_item.score)

    # Calculate weighted RRF scores
    fused_results: list[FusedResult[T]] = []

    for item_id, (item, source_ranks) in item_rankings.items():
        # Calculate weighted RRF score
        rrf_score = 0.0
        for source, (rank, _) in source_ranks.items():
            weight = source_weights.get(source, 1.0)
            rrf_score += weight * (1.0 / (k + rank))

        # Extract ranks and scores by source
        ranks = {source: rank for source, (rank, _) in source_ranks.items()}
        scores = {source: score for source, (_, score) in source_ranks.items() if score is not None}

        fused_results.append(
            FusedResult(
                item=item,
                rrf_score=rrf_score,
                ranks=ranks,
                scores=scores,
                combined_rank=0,  # Will be set below
            )
        )

    # Sort by weighted RRF score (descending) and assign combined ranks
    fused_results.sort(key=lambda x: x.rrf_score, reverse=True)

    for idx, result in enumerate(fused_results, start=1):
        result.combined_rank = idx

    return fused_results
