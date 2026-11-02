"""Unit tests for A1–A5 ablations and controlled error injection."""

import uuid
from pathlib import Path

from app.evaluation.ablations import InjectedError, apply_error_injection
from app.evaluation.schemas import (
    BenchmarkCategory,
    DifficultyLevel,
    EvaluatorInput,
    GroundTruth,
    SystemCondition,
)
from app.evaluation.service import MockPlannerModel, evaluate_single_case
from app.multi_agent.schemas import AgentResult, MultiAgentResult
from app.planner.schemas import AnalysisPlan, PlanStep
from app.services.dataset_repository import JsonDatasetRepository
from tests.evaluation_helpers import seed_evaluation_dataset


def test_apply_error_injection():
    plan = AnalysisPlan(
        intent="total_revenue",
        reasoning_type="simple",
        steps=[PlanStep(step_id="s1", operation="derive_metric", description="Derive revenue", parameters={"metric": "revenue"})],
    )
    result = MultiAgentResult(
        workflow_status="completed",
        executed_steps=["s1"],
        step_statuses={"s1": "completed"},
        agent_results=[AgentResult(agent_name="analysis", step_id="s1", status="completed", result=100.0)],
        final_result=100.0,
    )

    injection = InjectedError(
        error_type="incorrect_intermediate_result",
        target_step_id="s1",
        injected_value=999.0,
        description="Double result to test verification failure",
    )

    mut_plan, mut_res, applied = apply_error_injection(plan, result, injection)
    assert applied is True
    assert mut_res.agent_results[0].result != 100.0
    assert mut_res.final_result != 100.0


def test_evaluate_single_case_ablation_and_injection(tmp_path: Path):
    repo = JsonDatasetRepository(tmp_path)
    dataset_id = uuid.uuid4()
    seed_evaluation_dataset(repo, dataset_id, filename="sales.csv")

    case_input = EvaluatorInput(
        case_id="Q1",
        dataset="sales.csv",
        question="How many rows are in the dataset?",
        category=BenchmarkCategory.A_AGGREGATION,
        difficulty=DifficultyLevel.EASY,
    )
    ground_truth = GroundTruth(
        value=3,
        expected_operations=["count"],
    )

    # 1. Full V2 run
    res_full = evaluate_single_case(
        case_input,
        ground_truth,
        SystemCondition.SYSTEM_D_FULL_V2,
        dataset_id,
        repo,
        planner_model=MockPlannerModel(),
    )
    assert res_full.success is True
    assert res_full.numerical_correct is True

    # 2. A2 Ablation (No verification)
    res_a2 = evaluate_single_case(
        case_input,
        ground_truth,
        SystemCondition.ABLATION_A2_NO_VERIFICATION,
        dataset_id,
        repo,
        planner_model=MockPlannerModel(),
    )
    assert res_a2.success is True
    assert res_a2.verification_status is None
    assert any(s.stage_name == "verification" and s.status == "skipped" for s in res_a2.stage_results)

    # 3. Controlled Error Injection (H3: Verifier detects error)
    injection = InjectedError(
        error_type="incorrect_intermediate_result",
        target_step_id="s1",
        injected_value=999.0,
    )
    res_inj = evaluate_single_case(
        case_input,
        ground_truth,
        SystemCondition.SYSTEM_D_FULL_V2,
        dataset_id,
        repo,
        planner_model=MockPlannerModel(),
        error_injection=injection,
    )
    assert res_inj.metrics.get("injected_errors_count") == 1
    assert res_inj.metrics.get("detected_injected_errors_count") == 1
    assert res_inj.verification_status in {"failed", "unsupported"}
