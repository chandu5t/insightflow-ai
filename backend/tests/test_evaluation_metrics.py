"""Unit tests for the 10 official V2.1 evaluation metrics matching Contract v1.1 §8."""

from app.evaluation.metrics import (
    aggregate_metrics,
    build_case_metric_records,
    evaluate_numerical_accuracy,
    is_numerical_target,
    evaluate_planning_accuracy,
    evaluate_tool_selection_accuracy,
)
from app.evaluation.schemas import CaseEvaluationResult


def test_m1_numerical_accuracy_scalar_and_grouped():
    # Exact integers
    assert evaluate_numerical_accuracy(100, 100) is True
    assert evaluate_numerical_accuracy(100, 101) is False

    # Float comparisons require a frozen, explicitly supplied experiment tolerance.
    assert evaluate_numerical_accuracy(100.0000001, 100.0, 1e-6) is True
    assert evaluate_numerical_accuracy(100.01, 100.0, 1e-6) is False
    assert evaluate_numerical_accuracy(100.25, 100.0) is None
    assert evaluate_numerical_accuracy(1000000.01, 1000000.0, 1e-6) is False

    # Grouped dictionary
    obs_dict = {"North": 100.0, "South": 200.0}
    exp_dict = {"North": 100.0000001, "South": 200.0}
    assert evaluate_numerical_accuracy(obs_dict, exp_dict, 1e-6) is True

    bad_dict = {"North": 100.0, "East": 200.0}
    assert evaluate_numerical_accuracy(obs_dict, bad_dict) is False


def test_m1_unavailable_float_comparisons_propagate_through_nested_values():
    assert evaluate_numerical_accuracy(
        {"North": 100.25}, {"North": 100.0}
    ) is None
    assert evaluate_numerical_accuracy(
        [{"value": 100.25}, {"value": 200}],
        [{"value": 100.0}, {"value": 200}],
    ) is None
    # A definite mismatch remains incorrect even if another element is unavailable.
    assert evaluate_numerical_accuracy(
        [100.25, 3], [100.0, 4]
    ) is False


def test_m1_excludes_non_numerical_and_mixed_ground_truth_targets():
    assert is_numerical_target(12) is True
    assert evaluate_numerical_accuracy(12, 12) is True
    assert is_numerical_target({"North": 12, "South": 8}) is True
    assert evaluate_numerical_accuracy("AOV means ...", "AOV means ...") is None
    assert evaluate_numerical_accuracy(
        {"status": "clarification_required", "detail": "Specify a measure."},
        {"status": "clarification_required", "detail": "Specify a measure."},
    ) is None
    assert evaluate_numerical_accuracy(
        {"status": "unsupported", "detail": "No data."},
        {"status": "unsupported", "detail": "No data."},
    ) is None
    assert evaluate_numerical_accuracy(False, False) is None
    assert evaluate_numerical_accuracy({"label": "North", "value": 12}, {"label": "North", "value": 12}) is None


def test_m3_planning_accuracy():
    # Correct sequence and operations
    assert evaluate_planning_accuracy(["group_by", "aggregate", "rank"], ["group_by", "aggregate", "rank"]) is True
    # Additional unrequested operations are incorrect under the frozen expected sequence.
    assert evaluate_planning_accuracy(["select_columns", "group_by", "aggregate", "sort", "rank"], ["group_by", "aggregate"]) is False
    # Missing operation
    assert evaluate_planning_accuracy(["group_by"], ["group_by", "aggregate"]) is False
    # Inverted order
    assert evaluate_planning_accuracy(["aggregate", "group_by"], ["group_by", "aggregate"]) is False
    assert evaluate_planning_accuracy(["group_by", "aggregate"], ["group_by", "aggregate"]) is True
    assert evaluate_planning_accuracy(["count"], []) is None


def test_structured_metric_records_separate_m3_and_diagnostics():
    case = CaseEvaluationResult(
        case_id="Q_MULTI",
        condition="system_d_full_v2",
        run_id="run-1",
        success=True,
        numerical_correct=None,
        plan_correct=True,
        llm_calls=1,
        metrics={
            "planning_operation_sequence_correct": True,
            "planning_dependency_evaluability": "unavailable_no_dependency_ground_truth",
            "claims_evaluated_count": 3,
            "unsupported_claims_count": 1,
            "explanations_evaluated_count": 2,
            "grounded_explanations_count": 1,
            "metric_unavailable_reasons": {},
        },
    )
    records = build_case_metric_records(case, "metric-frozen-v1")
    official = {record.metric_id: record for record in records if record.metric_id.startswith("M")}
    diagnostics = {record.metric_id: record for record in records if record.diagnostic}

    assert set(official) == {f"M{i}" for i in range(1, 11)}
    assert official["M3"].value is True
    assert official["M3"].diagnostic is False
    assert official["M3"].metric_version == "metric-frozen-v1"
    assert official["M3"].definition_reference.endswith("M3 — Planning Accuracy")
    assert diagnostics["D1_planning_operation_sequence_match"].diagnostic is True
    assert diagnostics["D2_planning_dependency_evaluability"].diagnostic is True
    assert all(record.stage and record.scope == "case" for record in records)
    assert official["M1"].status == "unavailable"
    assert official["M7"].denominator == 3
    assert official["M7"].error_count == 1
    assert official["M8"].denominator == 2
    assert official["M8"].error_count == 1


def test_baseline_b_m4_is_explicitly_unreported_without_a_frozen_mapping():
    case = CaseEvaluationResult(
        case_id="Q_B",
        condition="baseline_b_tool_augmented",
        run_id="run-1",
        success=True,
        metrics={"metric_unavailable_reasons": {
            "M4": "V1 tool labels and V2.2 operation labels have no frozen canonical mapping."
        }},
    )
    m4 = next(record for record in build_case_metric_records(case, "v1") if record.metric_id == "M4")
    assert m4.status == "unavailable"
    assert m4.value is None
    assert "no frozen canonical mapping" in m4.unavailable_reason


def test_m4_tool_selection_accuracy():
    assert evaluate_tool_selection_accuracy("derive_metric", ["derive_metric"]) is True
    assert evaluate_tool_selection_accuracy(["group_by", "aggregate"], ["aggregate", "group_by"]) is True
    assert evaluate_tool_selection_accuracy(["not_aggregate"], ["aggregate"]) is False
    assert evaluate_tool_selection_accuracy("group_by", ["aggregate"]) is False
    assert evaluate_tool_selection_accuracy(None, ["aggregate"]) is False


def test_aggregate_metrics_preserves_denominators():
    c1 = CaseEvaluationResult(
        case_id="Q1",
        condition="system_d_full_v2",
        run_id="run-1",
        success=True,
        numerical_correct=True,
        plan_correct=True,
        latency_ms=100.0,
        llm_calls=2,
        input_tokens=100,
        output_tokens=50,
        metrics={
            "tool_selection_correct": True,
            "claims_evaluated_count": 1,
            "unsupported_claims_count": 0,
            "explanations_evaluated_count": 1,
            "grounded_explanations_count": 1,
            "planning_latency_ms": 20.0,
            "analysis_latency_ms": 50.0,
            "verification_latency_ms": 30.0,
        },
    )

    c2 = CaseEvaluationResult(
        case_id="Q2",
        condition="system_d_full_v2",
        run_id="run-1",
        success=False,
        numerical_correct=False,
        plan_correct=False,
        latency_ms=200.0,
        llm_calls=2,
        input_tokens=120,
        output_tokens=60,
        metrics={
            "tool_selection_correct": False,
            "claims_evaluated_count": 1,
            "unsupported_claims_count": 1,
            "explanations_evaluated_count": 1,
            "grounded_explanations_count": 0,
            "planning_latency_ms": 30.0,
            "analysis_latency_ms": 100.0,
            "verification_latency_ms": 70.0,
        },
    )

    agg = aggregate_metrics([c1, c2], metric_version="metric-v2.1-test")

    assert agg.total_cases == 2
    assert agg.completed_cases == 1
    assert agg.task_success_rate == 0.5
    assert agg.numerical_accuracy == 0.5
    assert agg.planning_accuracy == 0.5
    assert agg.tool_selection_accuracy == 0.5
    assert agg.unsupported_claim_rate == 0.5
    assert agg.explanation_groundedness == 0.5
    assert agg.mean_total_latency_ms == 150.0
    assert agg.total_llm_calls == 4

    # Crucial: when there are NO injected errors or detected errors,
    # the rates must remain None, not converted to 0.0!
    assert agg.injected_errors_count == 0
    assert agg.verification_detection_rate is None
    assert agg.detected_errors_count == 0
    assert agg.correction_success_rate is None
    aggregate_records = {record.metric_id: record for record in agg.metric_records}
    assert set(aggregate_records) == {f"M{i}" for i in range(1, 11)}
    assert aggregate_records["M1"].scope == "aggregate"
    assert aggregate_records["M1"].metric_version == "metric-v2.1-test"
    assert aggregate_records["M1"].denominator == 2
    assert aggregate_records["M10"].status == "partial"
    assert aggregate_records["M10"].value["total_llm_calls"] == 4
