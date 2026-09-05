"""Optional PostgreSQL publication layer for model metadata and item facts."""

from __future__ import annotations

import importlib
import json
import subprocess
from typing import Any
from uuid import uuid4

import duckdb

from rees46.runtime.config import PROJECT_ROOT, AppConfig


def _psycopg() -> Any:
    try:
        return importlib.import_module("psycopg")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PostgreSQL support requires the optional postgres extra. Run `uv sync --all-extras`."
        ) from exc


def initialize_schema(dsn: str) -> None:
    """Create the serving schema and tables."""
    psycopg = _psycopg()
    ddl = (PROJECT_ROOT / "sql" / "ddl" / "001_serving_schema.sql").read_text(encoding="utf-8")
    with psycopg.connect(dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(ddl)
        connection.commit()


def publish_item_stats(config: AppConfig, dsn: str, batch_size: int = 5000) -> int:
    """Upsert aggregated item facts from all monthly Gold partitions."""
    psycopg = _psycopg()
    item_glob = str(config.paths.gold / "item_stats" / "year_month=*" / "data.parquet")
    connection = duckdb.connect()
    try:
        cursor = connection.execute(
            """
            SELECT
                product_id,
                arg_max(category_id, last_seen) AS category_id,
                arg_max(category_code, last_seen) AS category_code,
                arg_max(brand, last_seen) AS brand,
                avg(mean_price) AS mean_price,
                sum(events) AS events,
                sum(views) AS views,
                sum(carts) AS carts,
                sum(purchases) AS purchases,
                max(unique_users) AS unique_users,
                max(unique_sessions) AS unique_sessions,
                min(first_seen) AS first_seen,
                max(last_seen) AS last_seen,
                sum(purchases)::DOUBLE / nullif(sum(views), 0) AS purchase_per_view
            FROM read_parquet(?)
            GROUP BY product_id
            """,
            [item_glob],
        )

        total = 0
        with psycopg.connect(dsn) as postgres:
            with postgres.cursor() as target:
                while True:
                    rows = cursor.fetchmany(batch_size)
                    if not rows:
                        break
                    target.executemany(
                        """
                        INSERT INTO rees46.item_stats (
                            product_id, category_id, category_code, brand, mean_price,
                            events, views, carts, purchases, unique_users, unique_sessions,
                            first_seen, last_seen, purchase_per_view
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (product_id) DO UPDATE SET
                            category_id = EXCLUDED.category_id,
                            category_code = EXCLUDED.category_code,
                            brand = EXCLUDED.brand,
                            mean_price = EXCLUDED.mean_price,
                            events = EXCLUDED.events,
                            views = EXCLUDED.views,
                            carts = EXCLUDED.carts,
                            purchases = EXCLUDED.purchases,
                            unique_users = EXCLUDED.unique_users,
                            unique_sessions = EXCLUDED.unique_sessions,
                            first_seen = EXCLUDED.first_seen,
                            last_seen = EXCLUDED.last_seen,
                            purchase_per_view = EXCLUDED.purchase_per_view
                        """,
                        rows,
                    )
                    total += len(rows)
            postgres.commit()
        return total
    finally:
        connection.close()


def publish_model_run(config: AppConfig, dsn: str) -> str:
    """Publish the current benchmark JSON as a model-run record."""
    psycopg = _psycopg()
    metrics_path = config.paths.reports / "metrics" / "model_benchmark.json"
    if not metrics_path.exists():
        raise FileNotFoundError("model_benchmark.json not found; run experiments first")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    run_id = str(uuid4())
    with psycopg.connect(dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO rees46.model_runs (run_id, profile, git_commit, metrics)
                VALUES (%s, %s, %s, %s::jsonb)
                """,
                (run_id, config.profile.value, commit, json.dumps(metrics)),
            )
        connection.commit()
    return run_id
