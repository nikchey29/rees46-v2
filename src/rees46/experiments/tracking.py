"""MLflow experiment tracking helpers."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_INVALID_METRIC_CHARS = re.compile(r"[^A-Za-z0-9_.\- /:]")


def _safe_metric_name(name: str) -> str:
    """Convert an internal metric name into an MLflow-safe metric name."""

    # Preserve the human meaning of recommender metrics such as Recall@20.
    name = name.replace("@", "_at_")

    # Protect against any future characters MLflow does not accept.
    name = _INVALID_METRIC_CHARS.sub("_", name)

    return name


class _MlflowAdapter:
    """Delegate to MLflow while enforcing safe metric names."""

    def __init__(self, module: Any) -> None:
        self._module = module

    def log_metric(
        self,
        key: str,
        value: float,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Log one metric using an MLflow-safe name."""

        return self._module.log_metric(
            _safe_metric_name(key),
            value,
            *args,
            **kwargs,
        )

    def log_metrics(
        self,
        metrics: Mapping[str, float],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Log multiple metrics using MLflow-safe names."""

        safe_metrics = {_safe_metric_name(key): value for key, value in metrics.items()}

        return self._module.log_metrics(
            safe_metrics,
            *args,
            **kwargs,
        )

    def __getattr__(self, name: str) -> Any:
        """Delegate every non-metric MLflow operation unchanged."""

        return getattr(self._module, name)


@contextmanager
def mlflow_run(
    experiment_name: str,
    run_name: str,
    tracking_dir: Path,
    enabled: bool,
) -> Iterator[Any | None]:
    """Create an optional local MLflow run backed by SQLite."""

    if not enabled:
        yield None
        return

    import mlflow

    tracking_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    database_path = (tracking_dir / "mlflow.db").resolve()

    tracking_uri = f"sqlite:///{database_path}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    adapter = _MlflowAdapter(mlflow)

    with mlflow.start_run(run_name=run_name):
        yield adapter
