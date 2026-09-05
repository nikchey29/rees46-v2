"""Session co-visitation recommender."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class CoVisitationModel:
    """Top item-to-item neighbours learned from session sequences."""

    neighbors: dict[int, tuple[tuple[int, float], ...]]

    @classmethod
    def fit(
        cls,
        sessions: Iterable[Sequence[int]],
        neighbors_per_item: int = 100,
        max_items_per_session: int = 40,
    ) -> CoVisitationModel:
        """Fit weighted pair counts from chronological session sequences."""

        if neighbors_per_item <= 0:
            raise ValueError("neighbors_per_item must be positive")

        if max_items_per_session <= 0:
            raise ValueError("max_items_per_session must be positive")

        counts: dict[int, dict[int, float]] = {}

        for sequence in sessions:
            unique = list(dict.fromkeys(int(item) for item in sequence))[-max_items_per_session:]

            for left_index, left in enumerate(unique):
                for right_index in range(
                    left_index + 1,
                    len(unique),
                ):
                    right = unique[right_index]
                    distance = right_index - left_index
                    weight = 1.0 / float(distance)

                    left_scores = counts.setdefault(left, {})
                    left_scores[right] = left_scores.get(right, 0.0) + weight

                    right_scores = counts.setdefault(right, {})
                    right_scores[left] = right_scores.get(left, 0.0) + weight

        neighbors: dict[
            int,
            tuple[tuple[int, float], ...],
        ] = {}

        for item, item_scores in counts.items():
            ranked = sorted(
                item_scores.items(),
                key=lambda pair: (
                    -pair[1],
                    pair[0],
                ),
            )

            neighbors[item] = tuple(ranked[:neighbors_per_item])

        return cls(neighbors=neighbors)

    def recommend(
        self,
        history: Sequence[int],
        k: int,
        exclude: Iterable[int] = (),
    ) -> list[int]:
        """Recommend items related to recent user history."""

        if k <= 0:
            return []

        excluded = set(exclude)

        scores: dict[int, float] = {}

        recent = list(dict.fromkeys(int(item) for item in history))[-10:]

        for recency_rank, item in enumerate(
            reversed(recent),
            start=1,
        ):
            recency_weight = 1.0 / float(recency_rank)

            for neighbor, pair_score in self.neighbors.get(item, ()):
                if neighbor in excluded:
                    continue

                scores[neighbor] = scores.get(neighbor, 0.0) + recency_weight * pair_score

        ranked = sorted(
            scores.items(),
            key=lambda pair: (
                -pair[1],
                pair[0],
            ),
        )

        return [item for item, _ in ranked[:k]]
