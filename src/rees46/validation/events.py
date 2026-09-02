"""Data-quality validation for REES46 event data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb

from rees46.ingestion.schema import VALID_EVENT_TYPES


class DataQualityError(RuntimeError):
    """Raised when a critical data-quality rule fails."""


def _scalar_int(
    connection: duckdb.DuckDBPyConnection,
    query: str,
    parameters: list[str],
) -> int:
    """Execute a scalar SQL query and safely return an integer."""

    row = connection.execute(
        query,
        parameters,
    ).fetchone()

    if row is None:
        raise RuntimeError("Expected a scalar DuckDB result but received no row.")

    return int(row[0])


def validate_bronze_file(
    connection: duckdb.DuckDBPyConnection,
    path: Path,
    expected_month: str,
) -> dict[str, Any]:
    """Validate one Bronze monthly partition."""

    source = str(path)

    total = _scalar_int(
        connection,
        """
        SELECT count(*)
        FROM read_parquet(?)
        """,
        [source],
    )

    critical_nulls = _scalar_int(
        connection,
        """
        SELECT count(*)
        FROM read_parquet(?)
        WHERE
            event_time IS NULL
            OR event_type IS NULL
            OR product_id IS NULL
            OR category_id IS NULL
            OR price IS NULL
            OR user_id IS NULL
        """,
        [source],
    )

    valid_events = sorted(VALID_EVENT_TYPES)

    invalid_events = _scalar_int(
        connection,
        """
        SELECT count(*)
        FROM read_parquet(?)
        WHERE event_type NOT IN (?, ?, ?, ?)
        """,
        [
            source,
            valid_events[0],
            valid_events[1],
            valid_events[2],
            valid_events[3],
        ],
    )

    invalid_ids = _scalar_int(
        connection,
        """
        SELECT count(*)
        FROM read_parquet(?)
        WHERE
            product_id <= 0
            OR category_id <= 0
            OR user_id <= 0
        """,
        [source],
    )

    negative_prices = _scalar_int(
        connection,
        """
        SELECT count(*)
        FROM read_parquet(?)
        WHERE price < 0
        """,
        [source],
    )

    wrong_month = _scalar_int(
        connection,
        """
        SELECT count(*)
        FROM read_parquet(?)
        WHERE strftime(event_time, '%Y-%m') <> ?
        """,
        [
            source,
            expected_month,
        ],
    )

    exact_duplicates = _scalar_int(
        connection,
        """
        SELECT coalesce(sum(n - 1), 0)
        FROM (
            SELECT count(*) AS n
            FROM read_parquet(?)
            GROUP BY
                event_time,
                event_type,
                product_id,
                category_id,
                category_code,
                brand,
                price,
                user_id,
                user_session
            HAVING count(*) > 1
        )
        """,
        [source],
    )

    event_rows = connection.execute(
        """
        SELECT
            event_type,
            count(*)
        FROM read_parquet(?)
        GROUP BY event_type
        ORDER BY event_type
        """,
        [source],
    ).fetchall()

    event_counts: dict[str, int] = {}

    for row in event_rows:
        if len(row) != 2:
            raise RuntimeError("Unexpected event-count query result.")

        event_counts[str(row[0])] = int(row[1])

    report: dict[str, Any] = {
        "month": expected_month,
        "rows": total,
        "critical_nulls": critical_nulls,
        "invalid_events": invalid_events,
        "invalid_ids": invalid_ids,
        "negative_prices": negative_prices,
        "wrong_month": wrong_month,
        "exact_duplicates": exact_duplicates,
        "event_counts": event_counts,
    }

    critical_failures = (
        critical_nulls + invalid_events + invalid_ids + negative_prices + wrong_month
    )

    if critical_failures:
        raise DataQualityError(f"Critical data-quality failure for {expected_month}: {report}")

    return report


def write_quality_report(
    report: dict[str, Any],
    target: Path,
) -> None:
    """Persist one data-quality report."""

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    target.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )
