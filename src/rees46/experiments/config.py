"""Typed configuration for recommendation experiments."""

from __future__ import annotations

from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from rees46.runtime.config import CONFIG_DIR, Profile


class StrictModel(BaseModel):
    """Reject unknown experiment settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class WorkloadSettings(StrictModel):
    """Resource limits for one experiment profile."""

    max_train_interactions: int = Field(gt=0)
    max_eval_users: int = Field(gt=0)
    max_sessions: int = Field(gt=0)
    covisitation_neighbors: int = Field(gt=0)
    collaborative_components: int = Field(gt=1)
    collaborative_max_users: int = Field(gt=1)
    collaborative_max_items: int = Field(gt=1)
    candidate_pool_size: int = Field(gt=0)
    sequential_vocab_size: int = Field(gt=100)
    sequential_max_len: int = Field(gt=1)
    sequential_epochs: int = Field(gt=0)
    sequential_batch_size: int = Field(gt=0)


class ExperimentConfig(StrictModel):
    """Complete experiment configuration."""

    experiment_name: str
    random_seed: int = Field(ge=0)
    k_values: tuple[int, ...]
    dev: WorkloadSettings
    full: WorkloadSettings

    def workload(self, profile: Profile | str) -> WorkloadSettings:
        """Return the resource profile for an execution mode."""
        selected = Profile(profile)
        return self.dev if selected is Profile.DEV else self.full


def load_experiment_config() -> ExperimentConfig:
    """Load and validate configs/modeling.yaml."""
    path = CONFIG_DIR / "modeling.yaml"
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return ExperimentConfig.model_validate(payload)


SplitName = Literal["train", "validation", "test"]
