"""Structured logging configuration."""

import logging

import structlog
from structlog.typing import Processor

from rees46.runtime.config import AppConfig


def configure_logging(config: AppConfig) -> None:
    """Configure application-wide structured logging."""

    log_level = getattr(logging, config.logging.level)

    logging.basicConfig(
        level=log_level,
        format="%(message)s",
    )

    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(
            fmt="iso",
            utc=True,
        ),
    ]

    if config.logging.json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger() -> structlog.stdlib.BoundLogger:
    """Return a correctly typed application logger."""
    return structlog.stdlib.get_logger()
