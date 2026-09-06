"""Build scalable month-partitioned recommendation Gold datasets."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import duckdb

from rees46.runtime.config import AppConfig
from rees46.runtime.logging import get_logger


def _write_parquet(
    connection: duckdb.DuckDBPyConnection,
    query: str,
    target: Path,
) -> None:
    """Execute a query and atomically write one Parquet dataset."""

    target.parent.mkdir(parents=True, exist_ok=True)

    # Gold outputs are immutable for a fixed Silver input. Reuse completed
    # atomic targets so an interrupted full run resumes at the failed table.
    if target.exists():
        return

    temp_target = target.with_suffix(".tmp.parquet")

    if temp_target.exists():
        temp_target.unlink()

    target_sql = str(temp_target).replace("'", "''")

    connection.execute(
        f"""
        COPY (
            {query}
        )
        TO '{target_sql}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD,
            ROW_GROUP_SIZE 250000
        )
        """
    )

    temp_target.replace(target)


def _count_rows(
    connection: duckdb.DuckDBPyConnection,
    path: Path,
) -> int:
    """Count rows in one Parquet dataset."""

    row = connection.execute(
        "SELECT count(*) FROM read_parquet(?)",
        [str(path)],
    ).fetchone()

    if row is None:
        raise RuntimeError(f"Could not count Parquet rows: {path}")

    return int(row[0])


def _build_session_sequences_bucketed(
    connection: duckdb.DuckDBPyConnection,
    source_sql: str,
    target: Path,
    buckets: int = 64,
) -> None:
    """Build ordered session lists in bounded hash buckets.

    DuckDB's list() aggregate cannot spill its intermediate state to disk.
    Hash-partitioning session events first keeps every list aggregation bounded
    while guaranteeing all rows for one session remain in the same bucket.
    """

    if buckets <= 0:
        raise ValueError("buckets must be positive")

    target.parent.mkdir(parents=True, exist_ok=True)
    bucket_source = target.parent / "_bucket_source"
    bucket_results = target.parent / "_bucket_results"

    shutil.rmtree(bucket_source, ignore_errors=True)
    shutil.rmtree(bucket_results, ignore_errors=True)
    bucket_results.mkdir(parents=True, exist_ok=True)

    bucket_source_sql = str(bucket_source).replace("'", "''")

    try:
        # One streaming scan of Silver creates bounded on-disk hash partitions.
        connection.execute(
            f"""
            COPY (
                SELECT
                    user_session,
                    user_id,
                    event_time,
                    product_id,
                    event_type,
                    hash(user_session) % {buckets} AS session_bucket
                FROM read_parquet('{source_sql}')
                WHERE user_session IS NOT NULL
            )
            TO '{bucket_source_sql}'
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD,
                PARTITION_BY (session_bucket),
                OVERWRITE_OR_IGNORE
            )
            """
        )

        partition_dirs = sorted(bucket_source.glob("session_bucket=*"))

        if not partition_dirs:
            raise RuntimeError("Session bucketing produced no partitions.")

        # list(... ORDER BY ...) is intentionally executed one bucket at a time.
        # All rows from a user_session share the same deterministic hash bucket.
        connection.execute("SET threads = 1")

        for partition_dir in partition_dirs:
            bucket_id = partition_dir.name.split("=", maxsplit=1)[1]
            bucket_glob = str(partition_dir / "*.parquet").replace("'", "''")
            part_target = bucket_results / f"part-{int(bucket_id):03d}.parquet"

            _write_parquet(
                connection,
                f"""
                SELECT
                    user_session,
                    any_value(user_id) AS user_id,
                    min(event_time) AS session_start,
                    max(event_time) AS session_end,
                    count(*) AS event_count,
                    list(product_id ORDER BY event_time) AS product_sequence,
                    list(event_type ORDER BY event_time) AS event_sequence
                FROM read_parquet('{bucket_glob}', hive_partitioning = false)
                GROUP BY user_session
                HAVING count(*) BETWEEN 2 AND 100
                """,
                part_target,
            )

        parts_glob = str(bucket_results / "part-*.parquet").replace("'", "''")

        # Combining finished bucket outputs is streaming and does not aggregate.
        _write_parquet(
            connection,
            f"""
            SELECT *
            FROM read_parquet('{parts_glob}', hive_partitioning = false)
            """,
            target,
        )
    finally:
        connection.execute("SET threads = 2")
        shutil.rmtree(bucket_source, ignore_errors=True)
        shutil.rmtree(bucket_results, ignore_errors=True)


def build_gold_month(
    connection: duckdb.DuckDBPyConnection,
    config: AppConfig,
    month: str,
) -> dict[str, int]:
    """Build all recommendation Gold tables for one month."""

    source = config.paths.silver / f"year_month={month}" / "events.parquet"

    if not source.exists():
        raise FileNotFoundError(f"Missing Silver partition: {source}")

    source_sql = str(source).replace("'", "''")

    counts: dict[str, int] = {}

    # =========================================================
    # USER × ITEM INTERACTIONS
    # =========================================================

    target = config.paths.gold / "user_item_interactions" / f"year_month={month}" / "data.parquet"

    _write_parquet(
        connection,
        f"""
        SELECT
            user_id,
            product_id,

            arg_max(category_id, event_time)
                AS category_id,

            arg_max(category_code, event_time)
                AS category_code,

            arg_max(brand, event_time)
                AS brand,

            avg(price)
                AS mean_price,

            count(*) FILTER (
                WHERE event_type = 'view'
            ) AS views,

            count(*) FILTER (
                WHERE event_type = 'cart'
            ) AS carts,

            count(*) FILTER (
                WHERE event_type = 'purchase'
            ) AS purchases,

            min(event_time)
                AS first_event_time,

            max(event_time)
                AS last_event_time,

            sum(
                CASE event_type
                    WHEN 'view' THEN 1.0
                    WHEN 'cart' THEN 3.0
                    WHEN 'purchase' THEN 5.0
                    WHEN 'remove_from_cart' THEN -1.0
                    ELSE 0.0
                END
            ) AS interaction_strength

        FROM read_parquet('{source_sql}')

        GROUP BY
            user_id,
            product_id
        """,
        target,
    )

    counts["user_item_interactions"] = _count_rows(
        connection,
        target,
    )

    # =========================================================
    # ITEM STATISTICS
    # =========================================================

    target = config.paths.gold / "item_stats" / f"year_month={month}" / "data.parquet"

    _write_parquet(
        connection,
        f"""
        SELECT
            product_id,

            arg_max(category_id, event_time)
                AS category_id,

            arg_max(category_code, event_time)
                AS category_code,

            arg_max(brand, event_time)
                AS brand,

            avg(price)
                AS mean_price,

            count(*)
                AS events,

            count(*) FILTER (
                WHERE event_type = 'view'
            ) AS views,

            count(*) FILTER (
                WHERE event_type = 'cart'
            ) AS carts,

            count(*) FILTER (
                WHERE event_type = 'purchase'
            ) AS purchases,

            count(DISTINCT user_id)
                AS unique_users,

            count(DISTINCT user_session)
                AS unique_sessions,

            min(event_time)
                AS first_seen,

            max(event_time)
                AS last_seen,

            (
                count(*) FILTER (
                    WHERE event_type = 'purchase'
                )
            )::DOUBLE
            /
            nullif(
                count(*) FILTER (
                    WHERE event_type = 'view'
                ),
                0
            ) AS purchase_per_view

        FROM read_parquet('{source_sql}')

        GROUP BY product_id
        """,
        target,
    )

    counts["item_stats"] = _count_rows(
        connection,
        target,
    )

    # =========================================================
    # USER STATISTICS
    # =========================================================

    target = config.paths.gold / "user_stats" / f"year_month={month}" / "data.parquet"

    _write_parquet(
        connection,
        f"""
        SELECT
            user_id,

            count(*)
                AS events,

            count(*) FILTER (
                WHERE event_type = 'view'
            ) AS views,

            count(*) FILTER (
                WHERE event_type = 'cart'
            ) AS carts,

            count(*) FILTER (
                WHERE event_type = 'purchase'
            ) AS purchases,

            count(DISTINCT product_id)
                AS unique_products,

            count(DISTINCT category_id)
                AS unique_categories,

            count(DISTINCT user_session)
                AS sessions,

            min(event_time)
                AS first_seen,

            max(event_time)
                AS last_seen

        FROM read_parquet('{source_sql}')

        GROUP BY user_id
        """,
        target,
    )

    counts["user_stats"] = _count_rows(
        connection,
        target,
    )

    # =========================================================
    # USER × CATEGORY AFFINITY
    # =========================================================

    target = config.paths.gold / "category_affinity" / f"year_month={month}" / "data.parquet"

    _write_parquet(
        connection,
        f"""
        SELECT
            user_id,
            category_id,

            arg_max(category_code, event_time)
                AS category_code,

            count(*) FILTER (
                WHERE event_type = 'view'
            ) AS views,

            count(*) FILTER (
                WHERE event_type = 'cart'
            ) AS carts,

            count(*) FILTER (
                WHERE event_type = 'purchase'
            ) AS purchases,

            sum(
                CASE event_type
                    WHEN 'view' THEN 1.0
                    WHEN 'cart' THEN 3.0
                    WHEN 'purchase' THEN 5.0
                    WHEN 'remove_from_cart' THEN -1.0
                    ELSE 0.0
                END
            ) AS affinity_score,

            max(event_time)
                AS last_event_time

        FROM read_parquet('{source_sql}')

        GROUP BY
            user_id,
            category_id
        """,
        target,
    )

    counts["category_affinity"] = _count_rows(
        connection,
        target,
    )

    # =========================================================
    # SESSION SEQUENCES
    # =========================================================

    target = config.paths.gold / "session_sequences" / f"year_month={month}" / "data.parquet"

    if not target.exists():
        _build_session_sequences_bucketed(
            connection=connection,
            source_sql=source_sql,
            target=target,
        )

    counts["session_sequences"] = _count_rows(
        connection,
        target,
    )

    return counts


def build_gold(
    config: AppConfig,
) -> dict[str, dict[str, int]]:
    """Build month-partitioned Gold datasets for the full corpus."""

    logger = get_logger()

    silver_files = sorted(config.paths.silver.glob("year_month=*/events.parquet"))

    if not silver_files:
        raise FileNotFoundError("No Silver partitions were found.")

    months = [path.parent.name.split("=", maxsplit=1)[1] for path in silver_files]

    report: dict[str, dict[str, int]] = {}

    connection = duckdb.connect()

    try:
        connection.execute("SET preserve_insertion_order = false")

        connection.execute("SET threads = 2")

        for month in months:
            logger.info(
                "gold_month_started",
                month=month,
            )

            counts = build_gold_month(
                connection=connection,
                config=config,
                month=month,
            )

            report[month] = counts

            logger.info(
                "gold_month_complete",
                month=month,
                **counts,
            )

    finally:
        connection.close()

    report_path = config.paths.reports / "metrics" / "gold_build_summary.json"

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info(
        "gold_build_complete",
        months=len(months),
    )

    return report
