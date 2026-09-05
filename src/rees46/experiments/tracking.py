"""Optional MLflow experiment tracking."""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


@contextmanager
def mlflow_run(
    experiment_name: str,
    run_name: str,
    tracking_dir: Path,
    enabled: bool,
) -> Iterator[Any | None]:
    """Open an MLflow run when the optional tracking extra is installed."""
    if not enabled:
        yield None
        return
    try:
        mlflow = importlib.import_module("mlflow")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MLflow is required for tracked runs. Run `uv sync --all-extras`."
        ) from exc

    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        yield mlflow
