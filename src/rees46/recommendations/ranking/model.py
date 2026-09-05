"""Purchase-oriented candidate ranking model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingClassifier


@dataclass
class PurchaseRanker:
    """Gradient-boosted purchase-propensity ranker."""

    feature_names: tuple[str, ...]
    model: HistGradientBoostingClassifier

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        feature_names: list[str],
        label_column: str = "label",
        random_seed: int = 42,
    ) -> PurchaseRanker:
        """Fit a binary purchase model over candidate-level features."""
        if frame.empty:
            raise ValueError("ranking frame must not be empty")
        if label_column not in frame:
            raise ValueError(f"missing ranking label column: {label_column}")
        if frame[label_column].nunique() < 2:
            raise ValueError("ranking labels require both positive and negative examples")

        x = frame[feature_names].astype(float)
        y = frame[label_column].astype(int)
        model = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=150,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=random_seed,
        )
        model.fit(x, y)
        return cls(tuple(feature_names), model)

    def predict_scores(self, frame: pd.DataFrame) -> NDArray[np.float64]:
        """Return purchase probabilities for candidate rows."""
        probabilities = self.model.predict_proba(frame[list(self.feature_names)].astype(float))[
            :, 1
        ]
        return np.asarray(probabilities, dtype=np.float64)

    def rank(self, frame: pd.DataFrame, item_column: str = "product_id") -> list[int]:
        """Rank candidate items by predicted purchase probability."""
        scores = self.predict_scores(frame)
        order = np.argsort(-scores)
        items = frame[item_column].to_numpy(dtype=np.int64)
        return [int(items[index]) for index in order]
