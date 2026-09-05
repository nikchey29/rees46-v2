"""Build the canonical cleaned REES46 Silver event layer."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from rees46.runtime.config import AppConfig
from rees46.runtime.logging import get_logger


def _count_rows(
    connection: duckdb.DuckDBPyConnection,
    path: Path,
) -> int:
    row = connection.execute(
        "SELECT count(*) FROM read_parquet(?)",
        [str(path)],
    ).fetchone()

    if row is None:
        raise RuntimeError(f"Could not count rows in {path}")

    return int(row[0])


def build_silver_month(
    connection: duckdb.DuckDBPyConnection,
    source: Path,
    target: Path,
) -> dict[str, int]:
    """Deduplicate and normalize one Bronze partition."""

    target.parent.mkdir(parents=True, exist_ok=True)

    temp_target = target.with_suffix(".tmp.parquet")

    source_sql = str(source).replace("'", "''")
    target_sql = str(temp_target).replace("'", "''")

    bronze_rows = _count_rows(connection, source)

    query = f"""
        COPY (
            SELECT
                event_time,
                event_type,
                product_id,
                category_id,
                nullif(trim(category_code), '') AS category_code,
                nullif(trim(brand), '') AS brand,
                price,
                user_id,
                nullif(trim(user_session), '') AS user_session,
                source_month
            FROM (
                SELECT DISTINCT
                    event_time,
                    event_type,
                    product_id,
                    category_id,
                    category_code,
                    brand,
                    price,
                    user_id,
                    user_session,
                    source_month
                FROM read_parquet('{source_sql}')
                WHERE
                    event_time IS NOT NULL
                    AND event_type IS NOT NULL
                    AND product_id > 0
                    AND category_id > 0
                    AND price >= 0
                    AND user_id > 0
            )
        )
        TO '{target_sql}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD,
            ROW_GROUP_SIZE 500000
        )
    """

    connection.execute(query)

    silver_rows = _count_rows(connection, temp_target)

    temp_target.replace(target)

    return {
        "bronze_rows": bronze_rows,
        "silver_rows": silver_rows,
        "removed_exact_duplicates": bronze_rows - silver_rows,
    }


def build_silver(config: AppConfig) -> dict[str, dict[str, int]]:
    """Build all monthly Silver partitions."""

    logger = get_logger()

    reports: dict[str, dict[str, int]] = {}

    bronze_files = sorted(config.paths.bronze.glob("year_month=*/events.parquet"))

    if not bronze_files:
        raise FileNotFoundError("No Bronze partitions were found.")

    connection = duckdb.connect()

    try:
        connection.execute("SET preserve_insertion_order = false")

        for source in bronze_files:
            month = source.parent.name.split("=", maxsplit=1)[1]

            target = config.paths.silver / f"year_month={month}" / "events.parquet"

            logger.info(
                "silver_build_started",
                month=month,
            )

            report = build_silver_month(
                connection=connection,
                source=source,
                target=target,
            )

            reports[month] = report

            logger.info(
                "silver_build_complete",
                month=month,
                **report,
            )

    finally:
        connection.close()

    output = config.paths.reports / "metrics" / "silver_reconciliation.json"

    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        json.dumps(reports, indent=2),
        encoding="utf-8",
    )

    return reports
