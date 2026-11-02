from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path
from uuid import UUID

import pytest

from app.evaluation.ablations import InjectedError, apply_error_injection
from app.evaluation.benchmark import extract_ground_truth
from app.evaluation.metrics import aggregate_metrics, evaluate_numerical_accuracy, is_numerical_target
from app.evaluation import official_benchmark
from app.evaluation.official_benchmark import (
    BenchmarkValidationErrorV28,
    load_official_benchmark,
    verify_official_benchmark_hashes,
)
from app.evaluation.schemas import CaseEvaluationResult, EvaluatorInput, GroundTruth, SystemCondition
from app.evaluation.service import evaluate_single_case
from app.multi_agent.schemas import AgentResult, MultiAgentResult
from app.planner.schemas import AnalysisPlan, PlanStep


ROOT = Path(__file__).resolve().parents[1]
STEP7_DIR = ROOT / "data" / "evaluation" / "step7_v1.0"


def test_frozen_benchmark_m1_denominator_is_reproducible():
    cases = load_official_benchmark()
    expected_ids = {case.id for case in cases if is_numerical_target(extract_ground_truth(case).value)}
    assert len(expected_ids) == 57

    measured = [
        CaseEvaluationResult(
            case_id=case.id,
            condition="system_d_full_v2",
            run_id="applicability-only",
            success=False,
            numerical_correct=evaluate_numerical_accuracy(
                extract_ground_truth(case).value, extract_ground_truth(case).value,
                extract_ground_truth(case).tolerance,
            ),
        )
        for case in cases
    ]
    aggregate = aggregate_metrics(measured)
    assert aggregate.numerical_evaluated_count == len(expected_ids) == 57
    assert aggregate.numerical_excluded_count == len(cases) - len(expected_ids) == 43
    assert aggregate.numerical_correct_count == 57


def test_step7_freeze_hashes_and_m1_counts_match_approved_snapshot():
    freeze = verify_official_benchmark_hashes()
    bench_dir = ROOT / "data" / "evaluation" / "insightflow_bench_v1.0"
    for relative_path, expected_hash in freeze["benchmark"]["sha256"].items():
        artifact = (bench_dir / relative_path).read_bytes()
        assert b"\r" not in artifact
        digest = hashlib.sha256(artifact).hexdigest()
        assert digest == expected_hash
    assert freeze["benchmark"]["hash_representation"] == "canonical_lf_bytes"
    assert freeze["benchmark"]["hash_manifest_version"] == "insightflow-bench-canonical-lf-v1.0"
    assert freeze["benchmark"]["question_count"] == len(load_official_benchmark()) == 100
    assert freeze["M1_numerical_accuracy"]["frozen_benchmark_applicable_cases"] == 57
    assert freeze["M1_numerical_accuracy"]["frozen_benchmark_non_applicable_cases"] == 43


def test_canonical_lf_hash_manifest_has_exact_artifact_map():
    manifest_path = STEP7_DIR / "artifact_hashes_v2.8.2_v1.0.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    assert manifest["manifest_version"] == "insightflow-bench-canonical-lf-v1.0"
    assert manifest["hash_algorithm"] == "SHA-256"
    assert manifest["hash_representation"] == "canonical_lf_bytes"
    assert manifest["hash_input"] == "Exact artifact bytes; LF line endings only; any CR byte is invalid."
    assert set(manifest["artifacts"]) == {
        "questions.json",
        "ground_truth.json",
        "datasets/module3_sales.csv",
        "datasets/module3_sales_direct.csv",
    }
    assert all(entry["path"] == key for key, entry in manifest["artifacts"].items())


def test_runtime_integrity_rejects_mismatching_artifact_bytes(tmp_path, monkeypatch):
    benchmark_dir = tmp_path / "insightflow_bench_v1.0"
    (benchmark_dir / "datasets").mkdir(parents=True)
    source_dir = ROOT / "data" / "evaluation" / "insightflow_bench_v1.0"
    for relative_path in (
        "questions.json", "ground_truth.json",
        "datasets/module3_sales.csv", "datasets/module3_sales_direct.csv",
    ):
        shutil.copyfile(source_dir / relative_path, benchmark_dir / relative_path)
    freeze_path = tmp_path / "step7_freeze_v1.0.json"
    shutil.copyfile(STEP7_DIR / "step7_freeze_v1.0.json", freeze_path)
    manifest_path = tmp_path / "artifact_hashes_v2.8.2_v1.0.json"
    shutil.copyfile(STEP7_DIR / "artifact_hashes_v2.8.2_v1.0.json", manifest_path)
    monkeypatch.setattr(official_benchmark, "BENCHMARK_DIRECTORY", benchmark_dir)
    monkeypatch.setattr(official_benchmark, "STEP7_FREEZE_PATH", freeze_path)
    monkeypatch.setattr(official_benchmark, "CANONICAL_HASH_MANIFEST_PATH", manifest_path)
    target = benchmark_dir / "questions.json"
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(BenchmarkValidationErrorV28, match="does not match the Step 7 hash"):
        verify_official_benchmark_hashes()


def test_runtime_integrity_rejects_crlf_artifact_bytes(tmp_path, monkeypatch):
    benchmark_dir = tmp_path / "insightflow_bench_v1.0"
    (benchmark_dir / "datasets").mkdir(parents=True)
    source_dir = ROOT / "data" / "evaluation" / "insightflow_bench_v1.0"
    for relative_path in (
        "questions.json", "ground_truth.json",
        "datasets/module3_sales.csv", "datasets/module3_sales_direct.csv",
    ):
        shutil.copyfile(source_dir / relative_path, benchmark_dir / relative_path)
    target = benchmark_dir / "questions.json"
    target.write_bytes(target.read_bytes().replace(b"\n", b"\r\n"))
    freeze_path = tmp_path / "step7_freeze_v1.0.json"
    shutil.copyfile(STEP7_DIR / "step7_freeze_v1.0.json", freeze_path)
    manifest_path = tmp_path / "artifact_hashes_v2.8.2_v1.0.json"
    shutil.copyfile(STEP7_DIR / "artifact_hashes_v2.8.2_v1.0.json", manifest_path)
    monkeypatch.setattr(official_benchmark, "BENCHMARK_DIRECTORY", benchmark_dir)
    monkeypatch.setattr(official_benchmark, "STEP7_FREEZE_PATH", freeze_path)
    monkeypatch.setattr(official_benchmark, "CANONICAL_HASH_MANIFEST_PATH", manifest_path)
    with pytest.raises(BenchmarkValidationErrorV28, match="not canonical LF"):
        verify_official_benchmark_hashes()


def test_preregistered_injection_schedule_matches_frozen_cases():
    schedule = json.loads((STEP7_DIR / "error_injection_schedule_v1.0.json").read_text(encoding="utf-8"))
    cases = {case.id: case for case in load_official_benchmark()}
    rows = schedule["scheduled_cases"]
    assert len(rows) == 4
    assert len({row["case_id"] for row in rows}) == len(rows)
    assert all(row["repetition_count"] == 1 for row in rows)
    for row in rows:
        case = cases[row["case_id"]]
        assert row["category"] == case.category.value
        assert row["difficulty"] == case.difficulty.value
        assert schedule["injection_stage"] == "after_v23_execution_before_v24_verification"
        assert schedule["error_type"] == "incorrect_intermediate_result"
        assert "last successful Analysis Agent result" in schedule["target_selection_rule"]
        assert schedule["expected_detector_target"].startswith("V2.4 numerical_verification")
        assert schedule["self_correction_expected_applicable"] is False
        assert set(case.operations) & {"count", "distinct_count"}


def test_intermediate_injection_resolves_target_deterministically():
    plan = AnalysisPlan(
        intent="count_orders",
        reasoning_type="simple",
        steps=[PlanStep(step_id="planned", operation="count", description="Count rows")],
    )
    result = MultiAgentResult(
        workflow_status="completed",
        executed_steps=["model_step_1"],
        step_statuses={"model_step_1": "completed"},
        agent_results=[AgentResult(
            agent_name="analysis", step_id="model_step_1", status="completed", result=13,
            metadata={"operation": "count"},
        )],
        final_result=13,
    )
    injection = InjectedError(
        error_type="incorrect_intermediate_result",
        target_step_id=None,
        injected_value=-999999,
    )
    _, mutated, applied = apply_error_injection(plan, result, injection)
    assert applied is True
    assert injection.target_step_id == "model_step_1"
    assert mutated.agent_results[0].result == -999999
    assert mutated.final_result == -999999


def test_a1_fails_closed_instead_of_substituting_a_hard_coded_plan():
    with pytest.raises(NotImplementedError, match="A1 is not executable"):
        evaluate_single_case(
            case_input=EvaluatorInput(
                case_id="IFB-001", dataset="module3_sales.csv", question="What is revenue?",
                category="aggregation", difficulty="easy",
            ),
            ground_truth=GroundTruth(value=256000),
            condition=SystemCondition.ABLATION_A1_NO_PLANNER,
            dataset_id=UUID("00000000-0000-4000-8000-000000000001"),
            repository=None,  # A1 must stop before any dataset or runtime dependency is accessed.
        )
