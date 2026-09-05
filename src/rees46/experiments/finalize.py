"""Freeze measured local outputs into small Git-trackable result records."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from rees46.runtime.config import PROJECT_ROOT, AppConfig

TRACKABLE_METRICS = (
    "bronze_ingestion_counts.json",
    "silver_reconciliation.json",
    "gold_build_summary.json",
    "eda_summary.json",
    "model_benchmark.json",
)


def freeze_results(config: AppConfig) -> Path:
    """Copy small measured metrics into results/ without publishing large data/models."""
    destination = PROJECT_ROOT / "results" / "latest"
    destination.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for name in TRACKABLE_METRICS:
        source = config.paths.reports / "metrics" / name
        if source.exists():
            shutil.copy2(source, destination / name)
            copied.append(name)

    manifest = {
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "profile": config.profile.value,
        "copied_metrics": copied,
        "large_data_committed": False,
        "model_binaries_committed": False,
    }
    target = PROJECT_ROOT / "results" / "freeze_manifest.json"
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return target
