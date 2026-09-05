"""Ranking metrics used across REES46 recommendation experiments."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from math import log2


def precision_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Return Precision@K for one user."""
    if k <= 0:
        raise ValueError("k must be positive")
    top_k = recommended[:k]
    if not top_k:
        return 0.0
    hits = sum(item in relevant for item in top_k)
    return hits / k


def recall_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Return Recall@K for one user."""
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        return 0.0
    hits = sum(item in relevant for item in recommended[:k])
    return hits / len(relevant)


def hit_rate_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Return HitRate@K for one user."""
    return float(any(item in relevant for item in recommended[:k]))


def reciprocal_rank_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Return reciprocal rank of the first relevant item within K."""
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def average_precision_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Return Average Precision@K for one user."""
    if not relevant:
        return 0.0
    score = 0.0
    hits = 0
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / rank
    return score / min(len(relevant), k)


def ndcg_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Return binary NDCG@K for one user."""
    dcg = 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            dcg += 1.0 / log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg


def evaluate_recommendations(
    recommendations: Mapping[int, Sequence[int]],
    ground_truth: Mapping[int, set[int]],
    k_values: Iterable[int],
) -> dict[str, float]:
    """Evaluate recommendation lists over users with non-empty ground truth."""
    users = [user for user in ground_truth if ground_truth[user]]
    if not users:
        raise ValueError("ground_truth contains no evaluable users")

    metrics: dict[str, float] = {}
    for k in k_values:
        precision = recall = hit_rate = mrr = map_score = ndcg = 0.0
        for user in users:
            recs = recommendations.get(user, ())
            relevant = ground_truth[user]
            precision += precision_at_k(recs, relevant, k)
            recall += recall_at_k(recs, relevant, k)
            hit_rate += hit_rate_at_k(recs, relevant, k)
            mrr += reciprocal_rank_at_k(recs, relevant, k)
            map_score += average_precision_at_k(recs, relevant, k)
            ndcg += ndcg_at_k(recs, relevant, k)

        denom = float(len(users))
        metrics[f"precision@{k}"] = precision / denom
        metrics[f"recall@{k}"] = recall / denom
        metrics[f"hitrate@{k}"] = hit_rate / denom
        metrics[f"mrr@{k}"] = mrr / denom
        metrics[f"map@{k}"] = map_score / denom
        metrics[f"ndcg@{k}"] = ndcg / denom

    metrics["users_evaluated"] = float(len(users))
    return metrics
