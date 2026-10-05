"""Calculation of the 10 official V2.1 evaluation metrics matching Contract v1.1 §8."""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Any, Iterable, Sequence

from app.evaluation.schemas import AggregateMetrics, CaseEvaluationResult, MetricRecord


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float, Decimal)) and not isinstance(value, bool) and math.isfinite(float(value))


def is_numerical_target(value: Any) -> bool:
    """Whether a ground-truth value is entirely a numeric M1 target.

    Numeric scalars and non-empty numeric-only lists/mappings are eligible.
    Booleans, strings, mixed categorical/numeric structures, nulls, and empty
    containers are not M1 targets.
    """
    if _is_finite_number(value):
        return True
    if isinstance(value, dict):
        return bool(value) and all(is_numerical_target(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return bool(value) and all(is_numerical_target(item) for item in value)
    return False


def evaluate_numerical_accuracy(
    observed: Any,
    expected: Any,
    tolerance: float | None = None,
) -> bool | None:
    """Evaluate M1: Numerical Accuracy under exact integer or float tolerance rules."""
    if not is_numerical_target(expected):
        return None
    if observed is None:
        return False

    # Scalar numeric comparison
    if _is_finite_number(observed) and _is_finite_number(expected):
        obs_f = float(observed)
        exp_f = float(expected)
        # If both are integers/whole counts
        if float(obs_f).is_integer() and float(exp_f).is_integer():
            return int(obs_f) == int(exp_f)
        if (
            tolerance is None
            or isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not math.isfinite(float(tolerance))
            or tolerance < 0
        ):
            return None
        return abs(obs_f - exp_f) <= tolerance

    # Dictionary / Grouped map comparison
    if isinstance(observed, dict) and isinstance(expected, dict):
        if set(observed.keys()) != set(expected.keys()):
            return False
        return _combine_comparisons(
            evaluate_numerical_accuracy(observed[k], expected[k], tolerance)
            for k in expected
        )

    # List / tabular rows comparison
    if isinstance(observed, list) and isinstance(expected, list):
        if len(observed) != len(expected):
            return False
        return _combine_comparisons(
            evaluate_numerical_accuracy(obs_item, exp_item, tolerance)
            for obs_item, exp_item in zip(observed, expected)
        )

    # String / boolean exact comparison
    return str(observed).strip().lower() == str(expected).strip().lower()


def _combine_comparisons(values: Iterable[bool | None]) -> bool | None:
    """A known mismatch dominates; otherwise preserve any unavailable child."""
    outcomes = list(values)
    if any(value is False for value in outcomes):
        return False
    if any(value is None for value in outcomes):
        return None
    return True


def evaluate_planning_accuracy(
    observed_steps: list[str],
    expected_steps: list[str],
) -> bool | None:
    """Evaluate the frozen operation-sequence component of M3; no dependency graph is inferred."""
    if not expected_steps:
        return None
    if not observed_steps:
        return False

    # Normalize operation names
    obs_norm = [s.strip().lower() for s in observed_steps]
    exp_norm = [s.strip().lower() for s in expected_steps]

    # Compare only the explicit expected operation sequence. This is not a
    # substitute for dependency correctness or a claim that dependency labels exist.
    return obs_norm == exp_norm


_METRIC_DEFINITIONS: dict[str, tuple[str, str]] = {
    "M1": ("analysis", "V2.7 Contract v1.1 §8 M1 — Numerical Accuracy"),
    "M2": ("workflow", "V2.7 Contract v1.1 §8 M2 — Task Success Rate"),
    "M3": ("planner", "V2.7 Contract v1.1 §8 M3 — Planning Accuracy"),
    "M4": ("planner", "V2.7 Contract v1.1 §8 M4 — Tool Selection Accuracy"),
    "M5": ("verification", "V2.7 Contract v1.1 §8 M5 — Verification Detection Rate"),
    "M6": ("correction", "V2.7 Contract v1.1 §8 M6 — Correction Success Rate"),
    "M7": ("answer", "V2.7 Contract v1.1 §8 M7 — Hallucination / Unsupported-Claim Rate"),
    "M8": ("explanation", "V2.7 Contract v1.1 §8 M8 — Explanation Groundedness"),
    "M9": ("all", "V2.7 Contract v1.1 §8 M9 — Latency"),
    "M10": ("all", "V2.7 Contract v1.1 §8 M10 — LLM/API Usage"),
}


def _record(
    metric_id: str,
    metric_version: str,
    scope: str,
    value: Any,
    status: str,
    *,
    unavailable_reason: str | None = None,
    denominator: int | float | None = None,
    error_count: int | None = None,
    error_rate: float | None = None,
    diagnostic: bool = False,
) -> MetricRecord:
    stage, definition = _METRIC_DEFINITIONS.get(
        metric_id,
        ("planner", "V2.7 implementation diagnostic; not an official research metric"),
    )
    return MetricRecord(
        metric_id=metric_id,
        metric_version=metric_version,
        stage=stage,
        definition_reference=definition,
        scope=scope,  # type: ignore[arg-type]
        value=value,
        status=status,  # type: ignore[arg-type]
        unavailable_reason=unavailable_reason,
        denominator=denominator,
        error_count=error_count,
        error_rate=error_rate,
        diagnostic=diagnostic,
    )


def build_case_metric_records(
    result: CaseEvaluationResult,
    metric_version: str,
) -> list[MetricRecord]:
    """Build case-level official metric records plus isolated diagnostics."""
    records: list[MetricRecord] = []
    metric_values: dict[str, tuple[Any, str, str | None]] = {}
    metric_details: dict[str, tuple[int | None, int | None, float | None]] = {}

    metric_values["M1"] = (
        result.numerical_correct,
        "available" if result.numerical_correct is not None else "unavailable",
        None if result.numerical_correct is not None else (
            "Ground truth is non-numeric/mixed or a required numerical tolerance is unavailable."
        ),
    )
    metric_values["M2"] = (result.success, "available", None)
    metric_values["M3"] = (
        result.plan_correct,
        "available" if result.plan_correct is not None else "unavailable",
        None if result.plan_correct is not None else "No frozen expected operation sequence is available for this case.",
    )

    if "tool_selection_correct" in result.metrics:
        metric_values["M4"] = (result.metrics["tool_selection_correct"], "available", None)
    else:
        reason = result.metrics.get("metric_unavailable_reasons", {}).get(
            "M4", "This condition has no comparable frozen V2.2 operation-label selection."
        )
        metric_values["M4"] = (None, "unavailable", reason)

    injected = result.metrics.get("injected_errors_count", 0)
    detected_injected = result.metrics.get("detected_injected_errors_count")
    metric_values["M5"] = (
        bool(detected_injected) if injected and detected_injected is not None else None,
        "available" if injected and detected_injected is not None else "unavailable",
        None if injected and detected_injected is not None else "No successfully applied controlled error injection was evaluated.",
    )
    if injected and detected_injected is not None:
        metric_details["M5"] = (int(injected), int(injected - int(bool(detected_injected))),
                                 (injected - int(bool(detected_injected))) / injected)
    detected = result.metrics.get("detected_errors_count", 0)
    corrected = result.metrics.get("corrected_errors_count", 0) if detected else None
    metric_values["M6"] = (
        bool(corrected) if detected else None,
        "available" if detected and corrected is not None else "unavailable",
        None if detected and corrected is not None else "No detected verification failure was eligible for correction evaluation.",
    )
    if detected:
        metric_details["M6"] = (int(detected), int(detected - int(bool(corrected))),
                                 (detected - int(bool(corrected))) / detected)

    for metric_id, reason in (
        ("M7", "No independent claim-annotation protocol was supplied."),
        ("M8", "No independent explanation-groundedness annotation protocol was supplied."),
    ):
        denom_key = "claims_evaluated_count" if metric_id == "M7" else "explanations_evaluated_count"
        numerator_key = "unsupported_claims_count" if metric_id == "M7" else "grounded_explanations_count"
        denom = result.metrics.get(denom_key, 0)
        numerator = result.metrics.get(numerator_key, 0)
        metric_values[metric_id] = (
            numerator / denom if denom else None,
            "available" if denom else "unavailable",
            None if denom else reason,
        )
        if denom:
            error_count = numerator if metric_id == "M7" else denom - numerator
            metric_details[metric_id] = (int(denom), int(error_count), error_count / denom)

    metric_values["M9"] = ({"total_latency_ms": result.latency_ms, **{
        key.removesuffix("_latency_ms"): value
        for key, value in result.metrics.items()
        if key.endswith("_latency_ms") and value is not None
    }}, "available", None)
    llm_usage = {
        "llm_calls": result.llm_calls,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "estimated_api_cost_usd": None,
    }
    usage_status = "available" if all(value is not None for value in llm_usage.values()) else (
        "partial" if any(value is not None for value in llm_usage.values()) else "unavailable"
    )
    metric_values["M10"] = (
        llm_usage,
        usage_status,
        "Token usage and/or frozen provider pricing are unavailable; no estimate was fabricated."
        if usage_status != "available" else None,
    )

    for metric_id, (value, status, reason) in metric_values.items():
        records.append(_record(
            metric_id, metric_version, "case", value, status,
            unavailable_reason=reason,
            denominator=metric_details.get(metric_id, (1 if status in {"available", "partial"} else None, None, None))[0],
            error_count=metric_details.get(metric_id, (None, int(value is False) if metric_id in {"M1", "M2", "M3", "M4", "M5", "M6"} and status == "available" else None, None))[1],
            error_rate=metric_details.get(metric_id, (None, None, float(value is False) if metric_id in {"M1", "M2", "M3", "M4", "M5", "M6"} and status == "available" else None))[2],
        ))

    if "planning_operation_sequence_correct" in result.metrics:
        records.append(_record(
            "D1_planning_operation_sequence_match", metric_version, "case",
            result.metrics["planning_operation_sequence_correct"], "available", diagnostic=True,
        ))
    if "planning_dependency_evaluability" in result.metrics:
        records.append(_record(
            "D2_planning_dependency_evaluability", metric_version, "case", None, "unavailable",
            unavailable_reason="No frozen expected-dependency annotation is available.", diagnostic=True,
        ))
    return records


def build_aggregate_metric_records(
    aggregate: AggregateMetrics,
    metric_version: str,
) -> list[MetricRecord]:
    """Create the ten official aggregate records from auditable aggregate fields."""
    rows: list[tuple[str, Any, int | float | None, int | None, str | None]] = [
        ("M1", aggregate.numerical_accuracy, aggregate.numerical_evaluated_count,
         aggregate.numerical_evaluated_count - aggregate.numerical_correct_count, "No numerical case had sufficient ground truth/tolerance."),
        ("M2", aggregate.task_success_rate, aggregate.total_cases, aggregate.failed_cases, None),
        ("M3", aggregate.planning_accuracy, aggregate.planning_evaluated_count,
         aggregate.planning_evaluated_count - aggregate.planning_correct_count, "No case had a frozen expected operation sequence."),
        ("M4", aggregate.tool_selection_accuracy, aggregate.tool_selection_evaluated_count,
         aggregate.tool_selection_evaluated_count - aggregate.tool_selection_correct_count, "No comparable operation selections were available; Baseline B has no canonical cross-version mapping."),
        ("M5", aggregate.verification_detection_rate, aggregate.injected_errors_count,
         aggregate.injected_errors_count - aggregate.detected_injected_errors_count, "No successfully applied controlled errors were evaluated."),
        ("M6", aggregate.correction_success_rate, aggregate.detected_errors_count,
         aggregate.detected_errors_count - aggregate.corrected_errors_count, "No detected errors were available for correction evaluation."),
        ("M7", aggregate.unsupported_claim_rate, aggregate.claims_evaluated_count,
         aggregate.unsupported_claims_count, "No independent claim annotations were supplied."),
        ("M8", aggregate.explanation_groundedness, aggregate.explanations_evaluated_count,
         aggregate.explanations_evaluated_count - aggregate.grounded_explanations_count, "No independent explanation annotations were supplied."),
        ("M9", {
            "mean_total_latency_ms": aggregate.mean_total_latency_ms,
            "mean_planning_latency_ms": aggregate.mean_planning_latency_ms,
            "mean_analysis_latency_ms": aggregate.mean_analysis_latency_ms,
            "mean_verification_latency_ms": aggregate.mean_verification_latency_ms,
            "mean_correction_latency_ms": aggregate.mean_correction_latency_ms,
        }, aggregate.total_cases, None, None),
        ("M10", {
            "total_llm_calls": aggregate.total_llm_calls,
            "mean_llm_calls_per_case": aggregate.mean_llm_calls_per_case,
            "total_input_tokens": aggregate.total_input_tokens,
            "total_output_tokens": aggregate.total_output_tokens,
            "estimated_api_cost_usd": aggregate.estimated_api_cost_usd,
        }, aggregate.total_cases, None,
         "Some usage fields are unavailable; token values/pricing were not fabricated."),
    ]
    records: list[MetricRecord] = []
    for metric_id, value, denominator, errors, reason in rows:
        if metric_id == "M10" and isinstance(value, dict):
            present = [v for v in value.values() if v is not None]
            status = "available" if len(present) == len(value) else "partial" if present else "unavailable"
        else:
            status = "available" if value is not None and denominator is not None and denominator > 0 else "unavailable"
        if denominator == 0:
            errors = None
        error_rate = (errors / denominator) if errors is not None and denominator else None
        records.append(_record(
            metric_id, metric_version, "aggregate", value, status,
            unavailable_reason=None if status == "available" else reason,
            denominator=denominator,
            error_count=errors,
            error_rate=error_rate,
        ))
    return records


def evaluate_tool_selection_accuracy(
    observed_tool: str | list[str] | None,
    expected_operations: list[str],
) -> bool:
    """Evaluate M4: Tool Selection Accuracy."""
    if not expected_operations:
        return True
    if not observed_tool:
        return False

    observed = [observed_tool] if isinstance(observed_tool, str) else observed_tool
    observed_norm = sorted(op.strip().lower() for op in observed)
    expected_norm = sorted(op.strip().lower() for op in expected_operations)
    return observed_norm == expected_norm


def aggregate_metrics(cases: Sequence[CaseEvaluationResult], metric_version: str = "v2.1") -> AggregateMetrics:
    """Aggregate individual case results into the official M1–M10 research metrics."""
    total = len(cases)
    if total == 0:
        empty = AggregateMetrics(
            total_cases=0,
            completed_cases=0,
            failed_cases=0,
            task_success_rate=None,
        )
        empty.metric_records = build_aggregate_metric_records(empty, metric_version)
        return empty

    completed = sum(1 for c in cases if c.success)
    failed = total - completed
    task_success_rate = completed / total

    # M1: Numerical Accuracy
    num_eval = sum(1 for c in cases if c.numerical_correct is not None)
    num_corr = sum(1 for c in cases if c.numerical_correct is True)
    num_excluded = total - num_eval
    num_acc = (num_corr / num_eval) if num_eval > 0 else None

    # M3: Planning Accuracy
    plan_eval = sum(1 for c in cases if c.plan_correct is not None)
    plan_corr = sum(1 for c in cases if c.plan_correct is True)
    plan_acc = (plan_corr / plan_eval) if plan_eval > 0 else None

    # M4: Tool Selection Accuracy
    tool_eval = sum(1 for c in cases if "tool_selection_correct" in c.metrics)
    tool_corr = sum(1 for c in cases if c.metrics.get("tool_selection_correct") is True)
    tool_acc = (tool_corr / tool_eval) if tool_eval > 0 else None

    # M5: Verification Detection Rate (injected errors)
    inj_count = sum(c.metrics.get("injected_errors_count", 0) for c in cases)
    det_inj_count = sum(c.metrics.get("detected_injected_errors_count", 0) for c in cases)
    verif_rate = (det_inj_count / inj_count) if inj_count > 0 else None

    # M6: Correction Success Rate
    det_err = sum(c.metrics.get("detected_errors_count", 0) for c in cases)
    corr_err = sum(c.metrics.get("corrected_errors_count", 0) for c in cases)
    corr_rate = (corr_err / det_err) if det_err > 0 else None

    # M7: Hallucination / Unsupported-Claim Rate
    claims_eval = sum(c.metrics.get("claims_evaluated_count", 0) for c in cases)
    unsupp_claims = sum(c.metrics.get("unsupported_claims_count", 0) for c in cases)
    unsupp_rate = (unsupp_claims / claims_eval) if claims_eval > 0 else None

    # M8: Explanation Groundedness
    exp_eval = sum(c.metrics.get("explanations_evaluated_count", 0) for c in cases)
    grounded = sum(c.metrics.get("grounded_explanations_count", 0) for c in cases)
    grounded_rate = (grounded / exp_eval) if exp_eval > 0 else None

    # M9: Latencies
    total_lat = sum(c.latency_ms for c in cases)
    mean_total_lat = total_lat / total if total > 0 else 0.0

    plan_latencies = [c.metrics["planning_latency_ms"] for c in cases if c.metrics.get("planning_latency_ms") is not None]
    mean_plan_lat = sum(plan_latencies) / len(plan_latencies) if plan_latencies else None

    analysis_latencies = [c.metrics["analysis_latency_ms"] for c in cases if c.metrics.get("analysis_latency_ms") is not None]
    mean_analysis_lat = sum(analysis_latencies) / len(analysis_latencies) if analysis_latencies else None

    verif_latencies = [c.metrics["verification_latency_ms"] for c in cases if c.metrics.get("verification_latency_ms") is not None]
    mean_verif_lat = sum(verif_latencies) / len(verif_latencies) if verif_latencies else None

    corr_latencies = [c.metrics["correction_latency_ms"] for c in cases if c.metrics.get("correction_latency_ms") is not None]
    mean_corr_lat = sum(corr_latencies) / len(corr_latencies) if corr_latencies else None

    # M10: LLM/API Usage
    usage_known = all(c.llm_calls is not None for c in cases)
    token_usage_known = all(c.input_tokens is not None and c.output_tokens is not None for c in cases)
    tot_llm = sum(c.llm_calls or 0 for c in cases) if usage_known else None
    mean_llm = tot_llm / total if usage_known and total else None
    tot_in_tok = sum(c.input_tokens or 0 for c in cases) if token_usage_known else None
    tot_out_tok = sum(c.output_tokens or 0 for c in cases) if token_usage_known else None
    # Cost is left undefined unless token usage and a frozen pricing configuration
    # are supplied; this evaluator does not invent a provider price.
    est_cost = None

    aggregate = AggregateMetrics(
        total_cases=total,
        completed_cases=completed,
        failed_cases=failed,
        numerical_evaluated_count=num_eval,
        numerical_correct_count=num_corr,
        numerical_excluded_count=num_excluded,
        numerical_accuracy=round(num_acc, 4) if num_acc is not None else None,
        task_success_rate=round(task_success_rate, 4),
        planning_evaluated_count=plan_eval,
        planning_correct_count=plan_corr,
        planning_accuracy=round(plan_acc, 4) if plan_acc is not None else None,
        tool_selection_evaluated_count=tool_eval,
        tool_selection_correct_count=tool_corr,
        tool_selection_accuracy=round(tool_acc, 4) if tool_acc is not None else None,
        injected_errors_count=inj_count,
        detected_injected_errors_count=det_inj_count,
        verification_detection_rate=round(verif_rate, 4) if verif_rate is not None else None,
        detected_errors_count=det_err,
        corrected_errors_count=corr_err,
        correction_success_rate=round(corr_rate, 4) if corr_rate is not None else None,
        claims_evaluated_count=claims_eval,
        unsupported_claims_count=unsupp_claims,
        unsupported_claim_rate=round(unsupp_rate, 4) if unsupp_rate is not None else None,
        explanations_evaluated_count=exp_eval,
        grounded_explanations_count=grounded,
        explanation_groundedness=round(grounded_rate, 4) if grounded_rate is not None else None,
        mean_total_latency_ms=round(mean_total_lat, 2),
        mean_planning_latency_ms=round(mean_plan_lat, 2) if mean_plan_lat is not None else None,
        mean_analysis_latency_ms=round(mean_analysis_lat, 2) if mean_analysis_lat is not None else None,
        mean_verification_latency_ms=round(mean_verif_lat, 2) if mean_verif_lat is not None else None,
        mean_correction_latency_ms=round(mean_corr_lat, 2) if mean_corr_lat is not None else None,
        total_llm_calls=tot_llm,
        mean_llm_calls_per_case=round(mean_llm, 2) if mean_llm is not None else None,
        total_input_tokens=tot_in_tok,
        total_output_tokens=tot_out_tok,
        estimated_api_cost_usd=round(est_cost, 6) if est_cost is not None else None,
    )
    aggregate.metric_records = build_aggregate_metric_records(aggregate, metric_version)
    return aggregate
