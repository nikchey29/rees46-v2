"""Command-line interface for REES46 V2."""

from typing import Annotated

import typer

from rees46.runtime.config import Profile, load_config
from rees46.runtime.logging import configure_logging, get_logger
from rees46.runtime.pipeline import PIPELINE_ORDER

app = typer.Typer(
    name="rees46",
    help="REES46 V2 behavioral recommendation platform.",
    no_args_is_help=True,
)

ProfileOption = Annotated[
    Profile,
    typer.Option(help="Runtime profile."),
]


@app.command()
def info(
    profile: ProfileOption = Profile.DEV,
) -> None:
    """Display validated runtime information."""
    config = load_config(profile)
    configure_logging(config)

    logger = get_logger()
    logger.info(
        "runtime_ready",
        profile=config.profile.value,
        project=config.project.name,
        raw_path=str(config.paths.raw),
    )


@app.command()
def pipeline(
    profile: ProfileOption = Profile.DEV,
) -> None:
    """Display the canonical pipeline execution contract."""
    config = load_config(profile)
    configure_logging(config)

    logger = get_logger()

    logger.info(
        "pipeline_contract",
        profile=config.profile.value,
        stages=len(PIPELINE_ORDER),
    )

    for position, stage in enumerate(PIPELINE_ORDER, start=1):
        logger.info(
            "pipeline_stage",
            position=position,
            stage=stage.value,
        )


def main() -> None:
    """Run the REES46 CLI."""
    app()


if __name__ == "__main__":
    main()
