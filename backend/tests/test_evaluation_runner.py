"""Unit and integration tests for EvaluationRunner and artifact generation."""

import uuid
import json
from pathlib import Path

import pytest

from app.evaluation.benchmark import load_benchmark
from app.evaluation.runner import EvaluationRunner
from app.evaluation.schemas import ExperimentConfig, SystemCondition
from app.evaluation.service import MockPlannerModel
from app.services.dataset_repository import JsonDatasetRepository
from tests.evaluation_helpers import seed_evaluation_dataset


def test_evaluation_runner_full_suite(tmp_path: Path):
    repo = JsonDatasetRepository(tmp_path / "uploads")
    dataset_id = uuid.uuid4()
    seed_evaluation_dataset(repo, dataset_id)

    fixture_path = Path(__file__).resolve().parents[1] / "data" / "evaluation" / "benchmark_fixture_v27.json"
    cases = load_benchmark(fixture_path)

    config = ExperimentConfig(
        experiment_id="test_exp_runner_01",
        benchmark_id="insightflow_bench_fixture",
        condition=SystemCondition.SYSTEM_D_FULL_V2,
        case_selection=["Q013"],
        evaluation_configuration={"development_fixture": True, "retriever_mode": "stub", "tolerance": 1e-6},
        output_dir=str(tmp_path / "runs"),
    )

    runner = EvaluationRunner(
        config=config,
        repository=repo,
        planner_model=MockPlannerModel(),
    )

    mapping = {"sales_01.csv": dataset_id}
    with pytest.raises(ValueError, match="explicit dataset ID mapping"):
        runner.run_suite(cases)
    manifest, aggregate, case_results, errors = runner.run_suite(cases, dataset_mapping=mapping)

    assert manifest.total_cases_evaluated == 1
    assert len(case_results) == 1
    assert aggregate.total_cases == 1
    assert aggregate.completed_cases == 1
    assert {record.metric_id for record in aggregate.metric_records} == {f"M{i}" for i in range(1, 11)}
    assert {record.metric_id for record in case_results[0].metric_records if record.metric_id.startswith("M")} == {
        f"M{i}" for i in range(1, 11)
    }

    # Check artifacts written to disk
    run_dir = tmp_path / "runs" / "test_exp_runner_01"
    assert (run_dir / "experiment_manifest.json").exists()
    assert (run_dir / "case_results.json").exists()
    assert (run_dir / "aggregate_results.json").exists()
    assert (run_dir / "error_results.json").exists()
    assert (run_dir / "ablation_results.json").exists()
    assert (run_dir / "reproducibility.json").exists()
    assert (run_dir / "report.md").exists()
    assert json.loads((run_dir / "ablation_results.json").read_text(encoding="utf-8"))["case_ids"] == ["Q013"]
    assert json.loads((run_dir / "experiment_manifest.json").read_text(encoding="utf-8"))["output_files"]["report"].endswith("report.md")
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "Official M3 Planning Accuracy" in report
    assert "No dependency graph is inferred" in report
    assert "M4 is unavailable for Baseline B" in report
    with pytest.raises(FileExistsError, match="Experiment output already exists"):
        runner.run_suite(cases, dataset_mapping=mapping)


def test_research_run_rejects_unmarked_mock_planner(tmp_path: Path):
    repo = JsonDatasetRepository(tmp_path / "uploads")
    config = ExperimentConfig(experiment_id="reject_mock", output_dir=str(tmp_path / "runs"))
    runner = EvaluationRunner(config, repo, planner_model=MockPlannerModel())
    fixture_path = Path(__file__).resolve().parents[1] / "data" / "evaluation" / "benchmark_fixture_v27.json"
    cases = load_benchmark(fixture_path)
    with pytest.raises(ValueError, match="Mock or mismatched planner models"):
        runner.run_suite(cases)
