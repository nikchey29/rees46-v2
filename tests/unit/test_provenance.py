import json
from pathlib import Path

from rees46.experiments.data import TEST_MONTHS, TRAIN_MONTHS, VALIDATION_MONTHS
from rees46.recommendations.candidates.hybrid import reciprocal_rank_fusion


def test_month_partitions_do_not_overlap() -> None:
    assert set(TRAIN_MONTHS).isdisjoint(VALIDATION_MONTHS)
    assert set(TRAIN_MONTHS).isdisjoint(TEST_MONTHS)
    assert set(VALIDATION_MONTHS).isdisjoint(TEST_MONTHS)


def test_measured_reconciliation_is_exact() -> None:
    payload = json.loads(Path("results/data_foundation.json").read_text(encoding="utf-8"))
    assert payload["bronze_rows"] - payload["exact_duplicates_removed"] == payload["silver_rows"]


def test_hybrid_candidates_are_unique() -> None:
    fused = reciprocal_rank_fusion({"a": [1, 2, 3], "b": [2, 3, 4]}, k=10)
    assert len(fused) == len(set(fused))
