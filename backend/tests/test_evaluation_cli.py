"""Unit tests for the evaluation CLI interface matching Contract v1.1 §23."""

import uuid
from pathlib import Path

from app.evaluation.cli import (
    cmd_evaluate,
    cmd_summarize,
    cmd_validate_benchmark,
    cmd_validate_config,
    main,
)
from app.evaluation import cli as evaluation_cli
from app.services.dataset_repository import JsonDatasetRepository
from tests.evaluation_helpers import seed_evaluation_dataset


def test_cli_validate_commands(tmp_path: Path):
    fixture_dir = Path(__file__).resolve().parents[1] / "data" / "evaluation"
    bench_file = str(fixture_dir / "benchmark_fixture_v27.json")
    conf_file = str(fixture_dir / "experiment_config_fixture_v27.json")

    # Validate benchmark
    assert cmd_validate_benchmark(bench_file) == 0
    assert cmd_validate_benchmark("nonexistent.json") == 1

    # Validate config
    assert cmd_validate_config(conf_file) == 0
    assert cmd_validate_config("nonexistent.json") == 1


def test_cli_evaluate_and_summarize(tmp_path: Path, monkeypatch):
    fixture_dir = Path(__file__).resolve().parents[1] / "data" / "evaluation"
    bench_file = str(fixture_dir / "benchmark_fixture_v27.json")
    conf_file = str(fixture_dir / "experiment_config_fixture_v27.json")
    out_dir = str(tmp_path / "cli_runs")

    # Seed sales_01.csv into default upload directory so evaluate can resolve it
    repo = JsonDatasetRepository(tmp_path / "uploads")
    dataset_id = uuid.UUID("00000000-0000-0000-0000-000000000000")
    seed_evaluation_dataset(repo, dataset_id)
    monkeypatch.setattr(evaluation_cli, "JsonDatasetRepository", lambda _directory: repo)

    # Run evaluate via CLI
    code = main(["evaluate", "--config", conf_file, "--benchmark", bench_file, "--output-dir", out_dir])
    assert code == 0

    run_path = str(tmp_path / "cli_runs" / "smoke_exp_001")
    assert Path(run_path).exists()

    # Run summarize via CLI
    sum_code = main(["summarize", "--run-dir", run_path])
    assert sum_code == 0
