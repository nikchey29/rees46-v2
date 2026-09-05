"""Behavioral exploratory analysis for REES46."""

from __future__ import annotations

import json
from typing import Any

import duckdb
import matplotlib.pyplot as plt

from rees46.runtime.config import AppConfig
from rees46.runtime.logging import get_logger


def run_eda(config: AppConfig) -> dict[str, Any]:
    """Generate dataset overview statistics and core figures."""

    logger = get_logger()

    silver_glob = str(config.paths.silver / "year_month=*" / "events.parquet")

    connection = duckdb.connect()

    try:
        overview = connection.execute(
            """
            SELECT
                count(*) AS events,
                count(DISTINCT user_id) AS users,
                count(DISTINCT product_id) AS products,
                count(DISTINCT category_id) AS categories,
                count(DISTINCT user_session) AS sessions,
                min(event_time) AS first_event,
                max(event_time) AS last_event
            FROM read_parquet(?)
            """,
            [silver_glob],
        ).fetchone()

        if overview is None:
            raise RuntimeError("EDA overview query returned no result.")

        event_rows = connection.execute(
            """
            SELECT
                event_type,
                count(*) AS events
            FROM read_parquet(?)
            GROUP BY event_type
            ORDER BY events DESC
            """,
            [silver_glob],
        ).fetchall()

        monthly_rows = connection.execute(
            """
            SELECT
                source_month,
                event_type,
                count(*) AS events
            FROM read_parquet(?)
            GROUP BY
                source_month,
                event_type
            ORDER BY
                source_month,
                event_type
            """,
            [silver_glob],
        ).fetchall()

    finally:
        connection.close()

    summary: dict[str, Any] = {
        "events": int(overview[0]),
        "users": int(overview[1]),
        "products": int(overview[2]),
        "categories": int(overview[3]),
        "sessions": int(overview[4]),
        "first_event": str(overview[5]),
        "last_event": str(overview[6]),
        "event_counts": {str(row[0]): int(row[1]) for row in event_rows},
    }

    metrics_path = config.paths.reports / "metrics" / "eda_summary.json"

    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    # Monthly event-volume figure.

    months = sorted({str(row[0]) for row in monthly_rows})

    event_types = sorted({str(row[1]) for row in monthly_rows})

    lookup = {
        (str(month), str(event_type)): int(events) for month, event_type, events in monthly_rows
    }

    figure_dir = config.paths.reports / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    for event_type in event_types:
        values = [lookup.get((month, event_type), 0) for month in months]

        plt.figure(figsize=(10, 5))
        plt.plot(months, values, marker="o")
        plt.title(f"REES46 Monthly {event_type.title()} Events")
        plt.xlabel("Month")
        plt.ylabel("Events")
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig(
            figure_dir / f"monthly_{event_type}_events.png",
            dpi=150,
        )

        plt.close()

    logger.info(
        "eda_complete",
        **summary,
    )

    return summary
