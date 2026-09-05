"""TensorFlow sequential next-item recommendation experiment."""

from __future__ import annotations

import importlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


def build_sequence_examples(
    sessions: Sequence[Sequence[int]],
    vocab_size: int,
    max_len: int,
) -> tuple[NDArray[np.int32], NDArray[np.int32], dict[int, int], tuple[int, ...]]:
    """Convert item sessions into padded next-item examples over a bounded vocabulary."""
    counts: Counter[int] = Counter()
    for session in sessions:
        counts.update(int(item) for item in session)
    vocabulary = tuple(item for item, _ in counts.most_common(vocab_size))
    item_to_token = {item: index + 1 for index, item in enumerate(vocabulary)}

    features: list[list[int]] = []
    labels: list[int] = []
    for session in sessions:
        tokens = [item_to_token[item] for item in session if item in item_to_token]
        if len(tokens) < 2:
            continue
        for end in range(1, len(tokens)):
            prefix = tokens[max(0, end - max_len) : end]
            padded = [0] * (max_len - len(prefix)) + prefix
            features.append(padded)
            labels.append(tokens[end])

    if not features:
        raise ValueError("no sequence examples could be built")

    return (
        np.asarray(features, dtype=np.int32),
        np.asarray(labels, dtype=np.int32),
        item_to_token,
        vocabulary,
    )


@dataclass
class SequenceModel:
    """Thin wrapper around a Keras next-item model."""

    model: Any
    vocabulary: tuple[int, ...]
    item_to_token: dict[int, int]
    max_len: int

    def recommend(self, history: Sequence[int], k: int) -> list[int]:
        """Predict likely next items from the most recent sequence."""
        tokens = [self.item_to_token[item] for item in history if item in self.item_to_token]
        prefix = tokens[-self.max_len :]
        padded = [0] * (self.max_len - len(prefix)) + prefix
        scores = np.asarray(self.model.predict(np.asarray([padded]), verbose=0)[0])
        ranked = np.argsort(-scores)
        recommendations: list[int] = []
        seen = set(history)
        for token_index in ranked:
            token = int(token_index)
            if token == 0 or token > len(self.vocabulary):
                continue
            item = self.vocabulary[token - 1]
            if item in seen:
                continue
            recommendations.append(item)
            if len(recommendations) == k:
                break
        return recommendations


def fit_sequence_model(
    sessions: Sequence[Sequence[int]],
    vocab_size: int,
    max_len: int,
    epochs: int,
    batch_size: int,
    random_seed: int,
) -> SequenceModel:
    """Train a compact TensorFlow GRU next-item model."""
    try:
        tensorflow = importlib.import_module("tensorflow")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TensorFlow is required for the sequential experiment. Run `uv sync --all-extras`."
        ) from exc

    x, y, item_to_token, vocabulary = build_sequence_examples(sessions, vocab_size, max_len)
    tensorflow.keras.utils.set_random_seed(random_seed)

    model = tensorflow.keras.Sequential(
        [
            tensorflow.keras.layers.Input(shape=(max_len,)),
            tensorflow.keras.layers.Embedding(len(vocabulary) + 1, 64, mask_zero=True),
            tensorflow.keras.layers.GRU(96),
            tensorflow.keras.layers.Dense(len(vocabulary) + 1, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=2)
    return SequenceModel(
        model=model, vocabulary=vocabulary, item_to_token=item_to_token, max_len=max_len
    )
