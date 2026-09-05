"""Command-line interface for REES46 V2."""

from __future__ import annotations

from typing import Annotated

import typer

from rees46.analytics.eda import run_eda
from rees46.experiments.finalize import freeze_results
from rees46.experiments.runner import run_experiments
from rees46.features.gold import build_gold
from rees46.ingestion.run import run_data_foundation
from rees46.preprocessing.silver import build_silver
from rees46.runtime.config import Profile, load_config
from rees46.runtime.logging import configure_logging, get_logger
from rees46.runtime.pipeline import PIPELINE_ORDER
from rees46.warehouse.postgres import initialize_schema, publish_item_stats, publish_model_run

app = typer.Typer(
    name="rees46",
    help="REES46 V2 behavioral recommendation platform.",
    no_args_is_help=True,
)

ProfileOption = Annotated[Profile, typer.Option(help="Runtime profile.")]
SequenceOption = Annotated[
    bool,
    typer.Option("--sequence/--no-sequence", help="Run the TensorFlow sequential experiment."),
]
MlflowOption = Annotated[
    bool,
    typer.Option("--mlflow/--no-mlflow", help="Track the experiment with MLflow."),
]
DsnOption = Annotated[
    str,
    typer.Option("--dsn", envvar="REES46_POSTGRES_DSN", help="PostgreSQL connection string."),
]


@app.command()
def info(profile: ProfileOption = Profile.DEV) -> None:
    """Display validated runtime information."""
    config = load_config(profile)
    configure_logging(config)
    get_logger().info(
        "runtime_ready",
        profile=config.profile.value,
        project=config.project.name,
        raw_path=str(config.paths.raw),
    )


@app.command()
def pipeline(profile: ProfileOption = Profile.DEV) -> None:
    """Display the canonical pipeline execution contract."""
    config = load_config(profile)
    configure_logging(config)
    logger = get_logger()
    logger.info("pipeline_contract", profile=config.profile.value, stages=len(PIPELINE_ORDER))
    for position, stage in enumerate(PIPELINE_ORDER, start=1):
        logger.info("pipeline_stage", position=position, stage=stage.value)


@app.command("build-data")
def build_data(profile: ProfileOption = Profile.DEV) -> None:
    """Build and validate the REES46 Bronze data layer."""
    config = load_config(profile)
    configure_logging(config)
    run_data_foundation(config)


@app.command("build-silver")
def build_silver_command(profile: ProfileOption = Profile.DEV) -> None:
    """Build the canonical deduplicated Silver layer."""
    config = load_config(profile)
    configure_logging(config)
    build_silver(config)


@app.command("build-gold")
def build_gold_command(profile: ProfileOption = Profile.DEV) -> None:
    """Build scalable month-partitioned recommendation Gold datasets."""
    config = load_config(profile)
    configure_logging(config)
    build_gold(config)


@app.command("eda")
def eda_command(profile: ProfileOption = Profile.DEV) -> None:
    """Run behavioral exploratory analysis."""
    config = load_config(profile)
    configure_logging(config)
    run_eda(config)


@app.command("run-experiments")
def run_experiments_command(
    profile: ProfileOption = Profile.DEV,
    sequence: SequenceOption = True,
    mlflow: MlflowOption = True,
) -> None:
    """Run baselines, retrieval, ranking, sequence modeling, and offline evaluation."""
    config = load_config(profile)
    configure_logging(config)
    run_experiments(config, with_sequence=sequence, track_mlflow=mlflow)


@app.command("freeze-results")
def freeze_results_command(profile: ProfileOption = Profile.FULL) -> None:
    """Copy small measured outputs into the Git-trackable results directory."""
    config = load_config(profile)
    configure_logging(config)
    target = freeze_results(config)
    get_logger().info("results_frozen", manifest=str(target))


@app.command("publish-postgres")
def publish_postgres_command(
    dsn: DsnOption,
    profile: ProfileOption = Profile.FULL,
) -> None:
    """Publish item facts and benchmark metadata to PostgreSQL."""
    config = load_config(profile)
    configure_logging(config)
    initialize_schema(dsn)
    items = publish_item_stats(config, dsn)
    run_id = publish_model_run(config, dsn)
    get_logger().info("postgres_publish_complete", item_rows=items, run_id=run_id)


@app.command("finalize")
def finalize_command(
    profile: ProfileOption = Profile.FULL,
    sequence: SequenceOption = True,
    mlflow: MlflowOption = True,
) -> None:
    """Run remaining Gold/EDA/model work and freeze Git-trackable evidence."""
    config = load_config(profile)
    configure_logging(config)
    logger = get_logger()

    gold_summary = config.paths.reports / "metrics" / "gold_build_summary.json"
    if not gold_summary.exists():
        logger.info("finalize_stage", stage="build_gold")
        build_gold(config)

    logger.info("finalize_stage", stage="eda")
    run_eda(config)
    logger.info("finalize_stage", stage="experiments")
    run_experiments(config, with_sequence=sequence, track_mlflow=mlflow)
    manifest = freeze_results(config)
    logger.info("finalize_complete", manifest=str(manifest))


def main() -> None:
    """Run the REES46 CLI."""
    app()


if __name__ == "__main__":
    main()
