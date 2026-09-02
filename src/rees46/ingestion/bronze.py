"""REES46 raw-to-Bronze ingestion."""

from __future__ import annotations

import re
from pathlib import Path

import duckdb

from rees46.ingestion.manifest import EXPECTED_FILES
from rees46.runtime.config import AppConfig
from rees46.runtime.logging import get_logger

_MONTH_PATTERN = re.compile(r"(?P<year>\d{4})-(?P<month>[A-Za-z]{3})\.csv\.gz")

_MONTH_NUMBERS = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


def _year_month(filename: str) -> str:
    match = _MONTH_PATTERN.fullmatch(filename)

    if match is None:
        raise ValueError(f"Unexpected REES46 filename: {filename}")

    year = int(match.group("year"))
    month = _MONTH_NUMBERS[match.group("month")]

    return f"{year:04d}-{month:02d}"


def ingest_file(
    connection: duckdb.DuckDBPyConnection,
    source: Path,
    target: Path,
    year_month: str,
) -> int:
    """Convert one raw compressed REES46 CSV to typed Parquet."""

    target.parent.mkdir(parents=True, exist_ok=True)

    temp_target = target.with_suffix(".tmp.parquet")

    source_sql = str(source).replace("'", "''")
    target_sql = str(temp_target).replace("'", "''")

    query = f"""
        COPY (
            SELECT
                CAST(
                    replace(event_time, ' UTC', '')
                    AS TIMESTAMP
                ) AS event_time,

                CAST(event_type AS VARCHAR) AS event_type,
                CAST(product_id AS BIGINT) AS product_id,
                CAST(category_id AS BIGINT) AS category_id,
                CAST(category_code AS VARCHAR) AS category_code,
                CAST(brand AS VARCHAR) AS brand,
                CAST(price AS DOUBLE) AS price,
                CAST(user_id AS BIGINT) AS user_id,
                CAST(user_session AS VARCHAR) AS user_session,
                '{year_month}'::VARCHAR AS source_month

            FROM read_csv(
                '{source_sql}',
                header = true,
                auto_detect = false,
                columns = {{
                    'event_time': 'VARCHAR',
                    'event_type': 'VARCHAR',
                    'product_id': 'BIGINT',
                    'category_id': 'BIGINT',
                    'category_code': 'VARCHAR',
                    'brand': 'VARCHAR',
                    'price': 'DOUBLE',
                    'user_id': 'BIGINT',
                    'user_session': 'VARCHAR'
                }}
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

    count = connection.execute(
        "SELECT count(*) FROM read_parquet(?)",
        [str(temp_target)],
    ).fetchone()

    if count is None:
        raise RuntimeError(f"Could not count Bronze rows: {temp_target}")

    row_count = int(count[0])

    temp_target.replace(target)

    return row_count


def ingest_all(config: AppConfig) -> dict[str, int]:
    """Ingest all seven REES46 monthly archives."""

    logger = get_logger()
    counts: dict[str, int] = {}

    connection = duckdb.connect()

    try:
        connection.execute("SET preserve_insertion_order = false")

        for filename in EXPECTED_FILES:
            year_month = _year_month(filename)

            source = config.paths.raw / filename
            target = config.paths.bronze / f"year_month={year_month}" / "events.parquet"

            logger.info(
                "bronze_ingestion_started",
                source=filename,
                month=year_month,
            )

            row_count = ingest_file(
                connection,
                source,
                target,
                year_month,
            )

            counts[year_month] = row_count

            logger.info(
                "bronze_ingestion_complete",
                month=year_month,
                rows=row_count,
            )
    finally:
        connection.close()

    return counts
