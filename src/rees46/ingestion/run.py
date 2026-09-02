"""Execute REES46 ingestion and Bronze validation."""

from __future__ import annotations

import json

import duckdb

from rees46.ingestion.bronze import ingest_all
from rees46.ingestion.manifest import write_manifest
from rees46.runtime.config import AppConfig
from rees46.runtime.logging import get_logger
from rees46.validation.events import (
    validate_bronze_file,
    write_quality_report,
)


def run_data_foundation(config: AppConfig) -> None:
    """Build the raw manifest, Bronze layer, and quality reports."""

    logger = get_logger()

    manifest_path = write_manifest(config)

    logger.info(
        "raw_manifest_created",
        path=str(manifest_path),
    )

    ingestion_counts = ingest_all(config)

    connection = duckdb.connect()

    try:
        for month, expected_rows in ingestion_counts.items():
            bronze_path = config.paths.bronze / f"year_month={month}" / "events.parquet"

            report = validate_bronze_file(
                connection=connection,
                path=bronze_path,
                expected_month=month,
            )

            actual_rows = int(report["rows"])

            if actual_rows != expected_rows:
                raise RuntimeError(
                    "Bronze row reconciliation failed: "
                    f"month={month}, "
                    f"expected={expected_rows}, "
                    f"actual={actual_rows}"
                )

            report_path = config.paths.reports / "metrics" / "data_quality" / f"{month}.json"

            write_quality_report(
                report=report,
                target=report_path,
            )

            logger.info(
                "bronze_validated",
                month=month,
                rows=actual_rows,
                exact_duplicates=report["exact_duplicates"],
            )

    finally:
        connection.close()

    summary_path = config.paths.reports / "metrics" / "bronze_ingestion_counts.json"

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path.write_text(
        json.dumps(
            ingestion_counts,
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info(
        "data_foundation_complete",
        months=len(ingestion_counts),
        rows=sum(ingestion_counts.values()),
    )
