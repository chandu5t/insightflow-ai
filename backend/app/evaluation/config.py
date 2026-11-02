"""Experiment configuration loading and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.evaluation.schemas import ExperimentConfig, SystemCondition


class ConfigurationError(ValueError):
    """Raised when an experiment configuration is invalid."""


def validate_experiment_config(raw: dict[str, Any] | ExperimentConfig) -> ExperimentConfig:
    """Validate that an experiment configuration complies with the contract."""
    if isinstance(raw, ExperimentConfig):
        config = raw
    else:
        try:
            config = ExperimentConfig.model_validate(raw)
        except ValidationError as exc:
            raise ConfigurationError(f"Malformed experiment configuration: {exc}") from exc

    if not config.experiment_id.strip():
        raise ConfigurationError("experiment_id cannot be blank.")

    # Validate condition
    if not isinstance(config.condition, SystemCondition):
        raise ConfigurationError(f"Unsupported condition: {config.condition}")

    # Validate baseline / ablation consistency
    if config.condition.value.startswith("baseline_") and not config.baseline_id:
        config.baseline_id = config.condition.value

    if config.condition.value.startswith("ablation_") and not config.ablation_id:
        config.ablation_id = config.condition.value

    if config.random_seed < 0:
        raise ConfigurationError("random_seed must be non-negative.")

    return config


def load_experiment_config(source: str | Path | dict[str, Any]) -> ExperimentConfig:
    """Load configuration from a file or dictionary."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
            return validate_experiment_config(data)
    elif isinstance(source, dict):
        return validate_experiment_config(source)
    else:
        raise TypeError(f"Unsupported configuration source type: {type(source)}")
