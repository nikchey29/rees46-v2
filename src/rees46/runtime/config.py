"""Typed runtime configuration for REES46 V2."""

from __future__ import annotations

from copy import deepcopy
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "configs"


class Profile(StrEnum):
    """Supported execution profiles."""

    DEV = "dev"
    FULL = "full"


class StrictModel(BaseModel):
    """Base model that rejects unknown configuration fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ProjectSettings(StrictModel):
    """Project-level settings."""

    name: str
    random_seed: int = Field(ge=0)


class PathSettings(StrictModel):
    """Canonical project paths."""

    raw: Path
    bronze: Path
    silver: Path
    gold: Path
    models: Path
    reports: Path


class LoggingSettings(StrictModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    json_output: bool


class RuntimeSettings(StrictModel):
    """General runtime behaviour."""

    fail_fast: bool


class AppConfig(StrictModel):
    """Complete validated REES46 application configuration."""

    profile: Profile
    project: ProjectSettings
    paths: PathSettings
    logging: LoggingSettings
    runtime: RuntimeSettings


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read one YAML configuration file."""

    with path.open("r", encoding="utf-8") as handle:
        loaded: Any = yaml.safe_load(handle)

    if loaded is None:
        return {}

    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")

    if not all(isinstance(key, str) for key in loaded):
        raise ValueError(f"Configuration keys must be strings: {path}")

    return cast(dict[str, Any], loaded)


def _deep_merge(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    """Recursively merge a profile configuration over the base."""

    result = deepcopy(base)

    for key, value in override.items():
        existing = result.get(key)

        if isinstance(existing, dict) and isinstance(value, dict):
            result[key] = _deep_merge(
                cast(dict[str, Any], existing),
                cast(dict[str, Any], value),
            )
        else:
            result[key] = deepcopy(value)

    return result


def _resolve_paths(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve configured project paths to absolute paths."""

    resolved = deepcopy(config)
    paths = resolved.get("paths")

    if not isinstance(paths, dict):
        return resolved

    resolved_paths: dict[str, Path] = {}

    for key, value in paths.items():
        path = Path(str(value)).expanduser()

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        resolved_paths[str(key)] = path.resolve()

    resolved["paths"] = resolved_paths

    return resolved


def load_config(profile: Profile | str) -> AppConfig:
    """Load and validate one REES46 runtime profile."""

    selected_profile = Profile(profile)

    base_config = _read_yaml(CONFIG_DIR / "base.yaml")
    profile_config = _read_yaml(CONFIG_DIR / f"{selected_profile.value}.yaml")

    merged = _deep_merge(base_config, profile_config)
    resolved = _resolve_paths(merged)

    return AppConfig.model_validate(resolved)
