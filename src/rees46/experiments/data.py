"""Leakage-safe experiment datasets derived from Gold/Silver partitions."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import duckdb
import pandas as pd

from rees46.experiments.config import SplitName, WorkloadSettings
from rees46.runtime.config import AppConfig

TRAIN_MONTHS: tuple[str, ...] = ("2019-10", "2019-11", "2019-12", "2020-01", "2020-02")
VALIDATION_MONTHS: tuple[str, ...] = ("2020-03",)
TEST_MONTHS: tuple[str, ...] = ("2020-04",)


def _as_int(value: object) -> int:
    """Convert a pandas scalar at the dataframe boundary to a Python int."""
    return int(cast(Any, value))


def _quoted_paths(paths: list[Path]) -> str:
    return "[" + ",".join("'" + str(path).replace("'", "''") + "'" for path in paths) + "]"


def _gold_paths(config: AppConfig, table: str, months: tuple[str, ...]) -> list[Path]:
    paths = [config.paths.gold / table / f"year_month={month}" / "data.parquet" for month in months]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Gold partitions:\n" + "\n".join(str(path) for path in missing)
        )
    return paths


def load_training_interactions(
    config: AppConfig,
    workload: WorkloadSettings,
    random_seed: int,
) -> pd.DataFrame:
    """Load a deterministic reservoir sample of pre-validation user-item facts."""
    paths = _gold_paths(config, "user_item_interactions", TRAIN_MONTHS)
    source = _quoted_paths(paths)
    connection = duckdb.connect()
    try:
        frame = connection.execute(
            f"""
            SELECT
                user_id,
                product_id,
                category_id,
                views,
                carts,
                purchases,
                interaction_strength,
                last_event_time
            FROM read_parquet({source})
            USING SAMPLE reservoir ({workload.max_train_interactions} ROWS)
            REPEATABLE ({random_seed})
            """
        ).fetchdf()
    finally:
        connection.close()

    if frame.empty:
        raise RuntimeError("training interaction sample is empty")

    aggregated = (
        frame.groupby(["user_id", "product_id"], as_index=False)
        .agg(
            category_id=("category_id", "last"),
            views=("views", "sum"),
            carts=("carts", "sum"),
            purchases=("purchases", "sum"),
            interaction_strength=("interaction_strength", "sum"),
            last_event_time=("last_event_time", "max"),
        )
        .sort_values(["user_id", "last_event_time"])
        .reset_index(drop=True)
    )
    return aggregated


def load_session_sequences(
    config: AppConfig,
    workload: WorkloadSettings,
    random_seed: int,
) -> list[list[int]]:
    """Load bounded train-only session sequences for co-visitation/sequence models."""
    paths = _gold_paths(config, "session_sequences", TRAIN_MONTHS)
    source = _quoted_paths(paths)
    connection = duckdb.connect()
    try:
        rows = connection.execute(
            f"""
            SELECT product_sequence
            FROM read_parquet({source})
            USING SAMPLE reservoir ({workload.max_sessions} ROWS)
            REPEATABLE ({random_seed})
            """
        ).fetchall()
    finally:
        connection.close()

    sessions = [[int(item) for item in row[0]] for row in rows if row and row[0] is not None]
    if not sessions:
        raise RuntimeError("train session sample is empty")
    return sessions


def load_ground_truth(
    config: AppConfig,
    split: SplitName,
    eligible_users: set[int],
    max_users: int,
) -> dict[int, set[int]]:
    """Load purchase ground truth from a future monthly Silver holdout."""
    months = VALIDATION_MONTHS if split == "validation" else TEST_MONTHS
    if split == "train":
        raise ValueError("ground truth must come from validation or test")

    paths = [config.paths.silver / f"year_month={month}" / "events.parquet" for month in months]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Silver holdout partitions:\n" + "\n".join(str(path) for path in missing)
        )

    connection = duckdb.connect()
    try:
        frame = connection.execute(
            f"""
            SELECT DISTINCT user_id, product_id
            FROM read_parquet({_quoted_paths(paths)})
            WHERE event_type = 'purchase'
            """
        ).fetchdf()
    finally:
        connection.close()

    frame = frame[frame["user_id"].isin(eligible_users)]
    if frame.empty:
        raise RuntimeError(f"no {split} purchases overlap sampled training users")

    selected_users = (
        frame[["user_id"]]
        .drop_duplicates()
        .assign(_hash=lambda data: data["user_id"].astype("int64") % 104729)
        .sort_values(["_hash", "user_id"])
        .head(max_users)["user_id"]
        .astype("int64")
        .tolist()
    )
    frame = frame[frame["user_id"].isin(selected_users)]

    ground_truth: dict[int, set[int]] = defaultdict(set)
    for row in frame.itertuples(index=False):
        ground_truth[_as_int(row.user_id)].add(_as_int(row.product_id))
    return dict(ground_truth)


def user_histories(interactions: pd.DataFrame) -> dict[int, list[int]]:
    """Create chronological product histories from sampled train interactions."""
    histories: dict[int, list[int]] = defaultdict(list)
    ordered = interactions.sort_values(["user_id", "last_event_time"])
    for row in ordered.itertuples(index=False):
        histories[_as_int(row.user_id)].append(_as_int(row.product_id))
    return dict(histories)


def user_primary_categories(interactions: pd.DataFrame) -> dict[int, int]:
    """Return each user's strongest category using train-only interaction strength."""
    grouped: pd.DataFrame = interactions.groupby(["user_id", "category_id"], as_index=False).agg(
        interaction_strength=("interaction_strength", "sum")
    )
    grouped = grouped.sort_values(by=["user_id", "interaction_strength"], ascending=[True, False])
    strongest = grouped.drop_duplicates("user_id")
    return {
        _as_int(row.user_id): _as_int(row.category_id) for row in strongest.itertuples(index=False)
    }


def load_holdout_sessions(
    config: AppConfig,
    split: SplitName,
    limit: int,
    random_seed: int,
) -> list[list[int]]:
    """Load session sequences from validation/test months for next-item evaluation."""
    if split == "train":
        months = TRAIN_MONTHS
    elif split == "validation":
        months = VALIDATION_MONTHS
    else:
        months = TEST_MONTHS
    paths = _gold_paths(config, "session_sequences", months)
    connection = duckdb.connect()
    try:
        rows = connection.execute(
            f"""
            SELECT product_sequence
            FROM read_parquet({_quoted_paths(paths)})
            USING SAMPLE reservoir ({limit} ROWS)
            REPEATABLE ({random_seed})
            WHERE event_count >= 2
            """
        ).fetchall()
    finally:
        connection.close()
    return [[int(item) for item in row[0]] for row in rows if row and row[0] is not None]
