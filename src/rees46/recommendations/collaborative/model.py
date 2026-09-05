"""Latent-factor collaborative filtering for implicit interactions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


@dataclass
class CollaborativeModel:
    """Compact user-item latent-factor recommender."""

    user_to_index: dict[int, int]
    item_to_index: dict[int, int]
    index_to_item: tuple[int, ...]
    user_factors: NDArray[np.float64]
    item_components: NDArray[np.float64]

    @classmethod
    def fit(
        cls,
        interactions: pd.DataFrame,
        components: int = 64,
        random_seed: int = 42,
    ) -> CollaborativeModel:
        """Fit truncated SVD over a weighted implicit user-item matrix."""
        required = {"user_id", "product_id", "interaction_strength"}
        if not required.issubset(interactions.columns):
            raise ValueError(
                "collaborative model requires user_id, product_id, interaction_strength"
            )
        if interactions.empty:
            raise ValueError("interactions must not be empty")

        users = sorted(int(value) for value in interactions["user_id"].unique())
        items = sorted(int(value) for value in interactions["product_id"].unique())
        user_to_index = {user: index for index, user in enumerate(users)}
        item_to_index = {item: index for index, item in enumerate(items)}

        rows = interactions["user_id"].map(user_to_index).to_numpy(dtype=np.int64)
        cols = interactions["product_id"].map(item_to_index).to_numpy(dtype=np.int64)
        values = interactions["interaction_strength"].clip(lower=0).to_numpy(dtype=np.float64)

        matrix = csr_matrix(
            (values, (rows, cols)), shape=(len(users), len(items)), dtype=np.float64
        )
        max_components = min(matrix.shape) - 1
        if max_components < 1:
            raise ValueError("collaborative model needs at least two users/items")
        n_components = min(components, max_components)

        svd = TruncatedSVD(n_components=n_components, random_state=random_seed)
        user_factors = np.asarray(svd.fit_transform(matrix), dtype=np.float64)
        item_components = np.asarray(svd.components_, dtype=np.float64)

        return cls(
            user_to_index=user_to_index,
            item_to_index=item_to_index,
            index_to_item=tuple(items),
            user_factors=user_factors,
            item_components=item_components,
        )

    def score_items(self, user_id: int) -> NDArray[np.float64]:
        """Return reconstructed preference scores over the model item universe."""
        user_index = self.user_to_index.get(user_id)
        if user_index is None:
            return np.empty(0, dtype=np.float64)
        return np.asarray(self.user_factors[user_index] @ self.item_components, dtype=np.float64)

    def recommend(
        self,
        user_id: int,
        k: int,
        exclude: Iterable[int] = (),
    ) -> list[int]:
        """Return top latent-factor items for a known user."""
        scores = self.score_items(user_id)
        if scores.size == 0:
            return []

        excluded = set(exclude)
        ranked_indices = np.argsort(-scores)
        result: list[int] = []
        for index in ranked_indices:
            item = self.index_to_item[int(index)]
            if item in excluded:
                continue
            result.append(item)
            if len(result) == k:
                break
        return result
