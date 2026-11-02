"""Determinism and reproducibility tests matching Contract v1.1 §21, §26."""

import uuid
from pathlib import Path

from app.evaluation.benchmark import load_benchmark
from app.evaluation.runner import EvaluationRunner
from app.evaluation.schemas import ExperimentConfig, SystemCondition
from app.evaluation.service import MockPlannerModel
from app.services.dataset_repository import JsonDatasetRepository
from tests.evaluation_helpers import seed_evaluation_dataset


def test_experiment_determinism(tmp_path: Path):
    """Running identical configuration twice with same seed must yield equivalent evaluation outputs."""
    repo = JsonDatasetRepository(tmp_path / "uploads")
    dataset_id = uuid.uuid4()
    seed_evaluation_dataset(repo, dataset_id)

    fixture_path = Path(__file__).resolve().parents[1] / "data" / "evaluation" / "benchmark_fixture_v27.json"
    cases = load_benchmark(fixture_path)

    config_1 = ExperimentConfig(
        experiment_id="det_run_1",
        benchmark_id="insightflow_bench_fixture",
        condition=SystemCondition.SYSTEM_D_FULL_V2,
        case_selection=["Q013"],
        evaluation_configuration={"development_fixture": True, "retriever_mode": "stub", "tolerance": 1e-6},
        random_seed=42,
        output_dir=str(tmp_path / "runs"),
    )
    runner_1 = EvaluationRunner(config_1, repo, planner_model=MockPlannerModel())
    manifest_1, agg_1, cases_1, _ = runner_1.run_suite(cases, dataset_mapping={"sales_01.csv": dataset_id})

    config_2 = ExperimentConfig(
        experiment_id="det_run_2",
        benchmark_id="insightflow_bench_fixture",
        condition=SystemCondition.SYSTEM_D_FULL_V2,
        case_selection=["Q013"],
        evaluation_configuration={"development_fixture": True, "retriever_mode": "stub", "tolerance": 1e-6},
        random_seed=42,
        output_dir=str(tmp_path / "runs"),
    )
    runner_2 = EvaluationRunner(config_2, repo, planner_model=MockPlannerModel())
    manifest_2, agg_2, cases_2, _ = runner_2.run_suite(cases, dataset_mapping={"sales_01.csv": dataset_id})

    # Exact metrics equivalence
    assert agg_1.task_success_rate == agg_2.task_success_rate
    assert agg_1.numerical_accuracy == agg_2.numerical_accuracy
    assert cases_1[0].numerical_correct == cases_2[0].numerical_correct
    assert cases_1[0].raw_answer == cases_2[0].raw_answer
