"""Popularity-based recommendation models."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd


def _as_int(value: object) -> int:
    """Convert a pandas scalar at the dataframe boundary to a Python int."""
    return int(cast(Any, value))


@dataclass(frozen=True)
class PopularityModel:
    """Global popularity recommender."""

    ranked_items: tuple[int, ...]

    @classmethod
    def fit(cls, interactions: pd.DataFrame) -> PopularityModel:
        """Fit from an interaction table containing weighted counts."""
        if interactions.empty:
            raise ValueError("interactions must not be empty")

        data = interactions.copy()
        if "interaction_strength" in data.columns:
            scores = data.groupby("product_id", sort=False)["interaction_strength"].sum()
        else:
            event_cols = [column for column in ("views", "carts", "purchases") if column in data]
            if not event_cols:
                raise ValueError("interactions contain no popularity signal")
            weights = {"views": 1.0, "carts": 3.0, "purchases": 5.0}
            score = sum(data[column] * weights[column] for column in event_cols)
            data = data.assign(_popularity_score=score)
            scores = data.groupby("product_id", sort=False)["_popularity_score"].sum()

        ranked = scores.sort_values(ascending=False).index.astype("int64").tolist()
        return cls(tuple(int(item) for item in ranked))

    def recommend(self, k: int, exclude: Iterable[int] = ()) -> list[int]:
        """Return the top unseen globally popular items."""
        excluded = set(exclude)
        result: list[int] = []
        for item in self.ranked_items:
            if item in excluded:
                continue
            result.append(item)
            if len(result) == k:
                break
        return result


@dataclass(frozen=True)
class CategoryPopularityModel:
    """Popularity ranking conditioned on category."""

    by_category: dict[int, tuple[int, ...]]
    fallback: PopularityModel

    @classmethod
    def fit(cls, interactions: pd.DataFrame) -> CategoryPopularityModel:
        """Fit category-specific popularity rankings."""
        required = {"product_id", "category_id"}
        if not required.issubset(interactions.columns):
            raise ValueError("category popularity requires product_id and category_id")

        fallback = PopularityModel.fit(interactions)
        data = interactions.copy()
        if "interaction_strength" in data.columns:
            score_column = "interaction_strength"
        else:
            data["_score"] = (
                data.get("views", 0) + 3 * data.get("carts", 0) + 5 * data.get("purchases", 0)
            )
            score_column = "_score"

        grouped = (
            data.groupby(["category_id", "product_id"], sort=False)[score_column]
            .sum()
            .reset_index()
        )
        by_category: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for row in grouped.itertuples(index=False):
            by_category[_as_int(row.category_id)].append(
                (_as_int(row.product_id), float(getattr(row, score_column)))
            )

        rankings = {
            category: tuple(
                item for item, _ in sorted(values, key=lambda pair: (-pair[1], pair[0]))
            )
            for category, values in by_category.items()
        }
        return cls(rankings, fallback)

    def recommend(
        self,
        category_id: int | None,
        k: int,
        exclude: Iterable[int] = (),
    ) -> list[int]:
        """Return category-aware recommendations with global fallback."""
        excluded = set(exclude)
        result: list[int] = []

        if category_id is not None:
            for item in self.by_category.get(category_id, ()):
                if item not in excluded:
                    result.append(item)
                if len(result) == k:
                    return result

        for item in self.fallback.ranked_items:
            if item not in excluded and item not in result:
                result.append(item)
            if len(result) == k:
                break

        return result
