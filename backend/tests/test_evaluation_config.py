"""Unit tests for experiment configuration loading and validation."""

import pytest

from app.evaluation.config import (
    ConfigurationError,
    load_experiment_config,
    validate_experiment_config,
)
from app.evaluation.schemas import ExperimentConfig, SystemCondition


def test_valid_default_config():
    config = validate_experiment_config({
        "experiment_id": "test_001",
        "condition": "system_d_full_v2",
        "random_seed": 123,
    })
    assert config.experiment_id == "test_001"
    assert config.condition == SystemCondition.SYSTEM_D_FULL_V2
    assert config.random_seed == 123


def test_baseline_and_ablation_id_population():
    cfg_base = validate_experiment_config({
        "experiment_id": "b_test",
        "condition": "baseline_c_v1",
    })
    assert cfg_base.baseline_id == "baseline_c_v1"

    cfg_abl = validate_experiment_config({
        "experiment_id": "a_test",
        "condition": "ablation_a1_no_planner",
    })
    assert cfg_abl.ablation_id == "ablation_a1_no_planner"


def test_reject_invalid_config():
    with pytest.raises(ConfigurationError, match="cannot be blank"):
        validate_experiment_config({
            "experiment_id": "  ",
            "condition": "system_d_full_v2",
        })

    with pytest.raises(ConfigurationError, match="random_seed must be non-negative"):
        validate_experiment_config({
            "experiment_id": "bad_seed",
            "condition": "system_d_full_v2",
            "random_seed": -5,
        })
