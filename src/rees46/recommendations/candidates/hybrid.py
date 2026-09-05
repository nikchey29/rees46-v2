"""Hybrid candidate generation using reciprocal-rank fusion."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence


def reciprocal_rank_fusion(
    sources: Mapping[str, Sequence[int]],
    k: int,
    source_weights: Mapping[str, float] | None = None,
    rank_constant: float = 60.0,
) -> list[int]:
    """Fuse ranked candidate lists without requiring calibrated model scores."""
    if k <= 0:
        raise ValueError("k must be positive")
    if rank_constant <= 0:
        raise ValueError("rank_constant must be positive")

    weights = source_weights or {}
    scores: dict[int, float] = defaultdict(float)

    for source, items in sources.items():
        weight = float(weights.get(source, 1.0))
        for rank, item in enumerate(items, start=1):
            scores[int(item)] += weight / (rank_constant + rank)

    ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    return [item for item, _ in ranked[:k]]


def candidate_features(
    sources: Mapping[str, Sequence[int]],
    candidates: Sequence[int],
) -> dict[int, dict[str, float]]:
    """Create source/rank features for ranking candidates."""
    result: dict[int, dict[str, float]] = {int(item): {} for item in candidates}
    for source, items in sources.items():
        rank_lookup = {int(item): rank for rank, item in enumerate(items, start=1)}
        for item in candidates:
            rank = rank_lookup.get(int(item))
            result[int(item)][f"{source}_present"] = float(rank is not None)
            result[int(item)][f"{source}_rr"] = 0.0 if rank is None else 1.0 / rank
    return result
