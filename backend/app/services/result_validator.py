"""Checks that a tool result is consistent before it is shown to the user."""

import math
from collections.abc import Iterable

import pandas as pd

from app.core.errors import ToolError
from app.schemas.profile_schema import MissingValueReport
from app.schemas.query_schema import QueryPlan, ValidationCheck, ValidationInfo
from app.schemas.tool_schema import AggregationResult, GroupingResult, RankingResult
from app.tools.aggregation_tool import calculate_sum 

from enum import StrEnum

from app.services.query_dispatcher import DEFAULT_RANK_LIMIT
from app.tools.revenue import resolve_numeric_values
from app.utils.column_mapper import map_columns


def _check(name: str, passed: bool, detail: str) -> ValidationCheck:
    return ValidationCheck(name=name, passed=bool(passed), detail=None if passed else detail)


def _all_finite(values: Iterable[float | None]) -> bool:
    return all(value is None or math.isfinite(value) for value in values)


def _group_total_check(frame: pd.DataFrame, plan: QueryPlan, result: GroupingResult) -> ValidationCheck | None:
    """When no row was left out, the group totals must add up to the overall total."""
    if result.metric != "sum" or result.rows_missing_group or result.rows_missing_value:
        return None
    if plan.metric not in ("revenue", "quantity", "unit_price"):
        return None
    name, detail = "group_totals_match_overall", "The group totals do not add up to the overall total."
    try:
        overall = calculate_sum(frame, plan.metric).value
    except ToolError:
        return _check(name, False, detail)
    total = sum(group.value or 0 for group in result.groups)
    tolerance = 1e-6 * max(1, len(result.groups))
    return _check(name, math.isclose(total, overall, rel_tol=1e-9, abs_tol=tolerance), detail)


def _ranking_checks(result: RankingResult) -> list[ValidationCheck]:
    values = [item.value for item in result.items]
    pairs = list(zip(values, values[1:]))
    in_order = all(a >= b for a, b in pairs) if result.order == "desc" else all(a <= b for a, b in pairs)
    ranks = [item.rank for item in result.items]
    return [
        _check("finite_numbers", _all_finite(values), "A ranked value is not a finite number."),
        _check("ranking_order", in_order, "The ranked values are not in the requested order."),
        _check("rank_numbers", not ranks or (ranks[0] == 1 and ranks == sorted(ranks)),
               "The rank numbers are not consistent."),
        _check("item_count", len(result.items) == result.n_returned <= result.n_requested,
               "The number of ranked items does not match the request."),
    ]


def validate_result(frame: pd.DataFrame, plan: QueryPlan, result: object) -> ValidationInfo:
    """Run deterministic checks on a tool result. Status is "failed" if any check fails."""
    rows = len(frame)
    checks: list[ValidationCheck] = []

    if isinstance(result, AggregationResult):
        checks.append(_check("finite_numbers", _all_finite([result.value, result.total_revenue]),
                             "The result is not a finite number."))
        checks.append(_check("row_accounting", result.rows_used + result.rows_excluded_missing == rows,
                             f"Rows used plus rows left out must equal the {rows} rows of the dataset."))
    elif isinstance(result, GroupingResult):
        checks.append(_check("finite_numbers", _all_finite(group.value for group in result.groups),
                             "A group value is not a finite number."))
        checks.append(_check(
            "row_accounting",
            result.rows_used + result.rows_missing_value + result.rows_missing_group == rows,
            f"Rows used and left out must equal the {rows} rows of the dataset.",
        ))
        total_check = _group_total_check(frame, plan, result)
        if total_check is not None:
            checks.append(total_check)
    elif isinstance(result, RankingResult):
        checks.extend(_ranking_checks(result))
    elif isinstance(result, MissingValueReport):
        checks.append(_check("missing_total", result.total_missing_cells == sum(c.missing_count for c in result.columns),
                             "The missing-cell total does not match the columns."))
        checks.append(_check("cell_total", result.total_cells == frame.shape[0] * frame.shape[1],
                             "The cell total does not match the dataset size."))
    else:
        checks.append(_check("known_result_type", False, "The tool returned an unexpected result type."))

    status = "passed" if all(check.passed for check in checks) else "failed"
    return ValidationInfo(status=status, checks=checks)




# ======================================================================================
# Module 5: reconciliation checks (independent recomputation, with machine-readable codes)
# ======================================================================================


class ValidationCode(StrEnum):
    WRONG_RESULT_TYPE = "WRONG_RESULT_TYPE"
    OPERATION_MISMATCH = "OPERATION_MISMATCH"
    FRAME_ROW_MISMATCH = "FRAME_ROW_MISMATCH"
    COUNT_MISMATCH = "COUNT_MISMATCH"
    UNEXPLAINED_MISSING = "UNEXPLAINED_MISSING"
    RECONCILIATION_MISMATCH = "RECONCILIATION_MISMATCH"
    UNGROUNDED_NUMBER = "UNGROUNDED_NUMBER"


_TOOL_FOR_PLAN = {
    "revenue": "aggregation_tool",
    "aggregation": "aggregation_tool",
    "grouping": "grouping_tool",
    "ranking": "ranking_tool",
    "missing_values": "missing_value_tool",
}
_METRIC_FOR_AGGREGATION = {
    "sum": "sum", "average": "average", "min": "minimum", "max": "maximum", "count": "count",
}
_SAMPLE_GROUPS = 5


def _coded(name: str, passed: bool, code: ValidationCode, detail: str) -> ValidationCheck:
    return ValidationCheck(
        name=name,
        passed=bool(passed),
        code=None if passed else code.value,
        detail=None if passed else detail,
    )


def _close(a: float, b: float, count: int = 1) -> bool:
    """Tools round to 6 decimals, so allow a tiny absolute difference (more for many rounded values)."""
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-6 * max(1, count))


def _finish(checks: list[ValidationCheck]) -> ValidationInfo:
    status = "passed" if all(check.passed for check in checks) else "failed"
    return ValidationInfo(status=status, checks=checks)


def merge_validations(*infos: ValidationInfo) -> ValidationInfo:
    """Combine validation results. The status is "failed" if any check failed."""
    checks = [check for info in infos for check in info.checks]
    failed = any(info.status == "failed" for info in infos) or any(not check.passed for check in checks)
    return ValidationInfo(status="failed" if failed else "passed", checks=checks)


def _expected_aggregation_metric(plan: QueryPlan) -> str | None:
    if plan.intent == "total_revenue":
        return "total_revenue"
    if plan.metric == "average_order_value":
        return "average_order_value"
    if plan.metric == "records":
        return "count"
    return _METRIC_FOR_AGGREGATION.get(plan.aggregation or "")


def _operation_check(plan: QueryPlan, result: object) -> ValidationCheck | None:
    name = "operation_matches_plan"
    if isinstance(result, AggregationResult):
        expected = _expected_aggregation_metric(plan)
        detail = f"Expected the metric '{expected}', but the tool returned '{result.metric}'."
        return _coded(name, expected is not None and result.metric == expected, ValidationCode.OPERATION_MISMATCH, detail)
    if isinstance(result, GroupingResult):
        detail = f"Expected the aggregation '{plan.aggregation}', but the tool returned '{result.metric}'."
        return _coded(name, result.metric == plan.aggregation, ValidationCode.OPERATION_MISMATCH, detail)
    if isinstance(result, RankingResult):
        ok = (
            result.metric == plan.aggregation
            and result.order == (plan.sort_order or "desc")
            and result.n_requested == (plan.limit or DEFAULT_RANK_LIMIT)
        )
        return _coded(name, ok, ValidationCode.OPERATION_MISMATCH,
                      "The ranking metric, order or size does not match the plan.")
    return None


def _metric_values(frame: pd.DataFrame, metric: str | None) -> pd.Series | None:
    if metric not in ("revenue", "quantity", "unit_price"):
        return None
    try:
        mapping = map_columns([str(name) for name in frame.columns])
        return resolve_numeric_values(frame, metric, mapping).values
    except ToolError:
        return None


def _aggregation_recomputed(frame: pd.DataFrame, plan: QueryPlan, result: AggregationResult) -> ValidationCheck | None:
    if result.metric == "average_order_value":
        ok = bool(result.order_count) and result.total_revenue is not None and _close(
            result.value, result.total_revenue / result.order_count
        )
        return _coded("average_order_value_consistent", ok, ValidationCode.RECONCILIATION_MISMATCH,
                      "The average order value is not total revenue divided by the order count.")
    if result.metric not in ("sum", "total_revenue", "average", "minimum", "maximum"):
        return None
    values = _metric_values(frame, "revenue" if result.metric == "total_revenue" else plan.metric)
    if values is None or values.dropna().empty:
        return None
    valid = values.dropna()
    recomputed = {
        "sum": valid.sum(), "total_revenue": valid.sum(), "average": valid.mean(),
        "minimum": valid.min(), "maximum": valid.max(),
    }[result.metric]
    return _coded("aggregation_recomputed", _close(result.value, float(recomputed)),
                  ValidationCode.RECONCILIATION_MISMATCH,
                  f"The tool returned {result.value}, but recomputing gives {float(recomputed)}.")


def _group_value(values: pd.Series | None, group_series: pd.Series, group: str, aggregation: str | None) -> float | None:
    mask = group_series == group
    if values is None:  # metric "records": count the rows of the group
        return float(mask.sum())
    selected = values[mask].dropna()
    if aggregation == "count":
        return float(len(selected))
    if selected.empty:
        return None
    return float(selected.sum() if aggregation == "sum" else selected.mean())


def _groups_recomputed(
    frame: pd.DataFrame, plan: QueryPlan, group_by: str, pairs: list[tuple[str | None, float | int | None]]
) -> ValidationCheck:
    name, detail = "group_values_recomputed", "A group value does not match a recomputation with plain Pandas."
    if group_by not in frame.columns:
        return _coded(name, False, ValidationCode.RECONCILIATION_MISMATCH, f"The group column '{group_by}' is missing.")
    values = _metric_values(frame, plan.metric)
    if plan.metric != "records" and values is None:
        return _coded(name, True, ValidationCode.RECONCILIATION_MISMATCH, detail)  # cannot recompute: skip
    for group, value in pairs:
        if group is None:
            continue
        expected = _group_value(values, frame[group_by], group, plan.aggregation)
        same = expected is None if value is None else expected is not None and _close(float(value), expected)
        if not same:
            return _coded(name, False, ValidationCode.RECONCILIATION_MISMATCH,
                          f"The value of group '{group}' does not match the recomputed value.")
    return _coded(name, True, ValidationCode.RECONCILIATION_MISMATCH, detail)


def reconcile_result(
    frame: pd.DataFrame,
    plan: QueryPlan,
    result: object,
    *,
    expected_row_count: int | None = None,
) -> ValidationInfo:
    """Independent checks on a tool result. Never raises: a problem becomes a failed, coded check."""
    checks: list[ValidationCheck] = []
    expected_tool = _TOOL_FOR_PLAN.get(plan.tool_name or "")
    actual_tool = getattr(result, "tool", None)
    checks.append(_coded(
        "result_type_matches_tool", expected_tool is not None and actual_tool == expected_tool,
        ValidationCode.WRONG_RESULT_TYPE, f"Expected a '{expected_tool}' result, but got '{actual_tool}'.",
    ))
    if expected_tool is None or actual_tool != expected_tool:
        return _finish(checks)  # the other checks make no sense for the wrong type

    operation = _operation_check(plan, result)
    if operation is not None:
        checks.append(operation)

    if expected_row_count is not None:
        checks.append(_coded(
            "frame_matches_metadata", len(frame) == expected_row_count, ValidationCode.FRAME_ROW_MISMATCH,
            f"The dataset has {len(frame)} rows, but the upload recorded {expected_row_count}.",
        ))
        if isinstance(result, AggregationResult) and result.calculation_method == "row_count":
            checks.append(_coded(
                "record_count_matches_metadata", result.value == expected_row_count, ValidationCode.COUNT_MISMATCH,
                f"The record count is {result.value}, but the upload recorded {expected_row_count} rows.",
            ))

    if isinstance(result, AggregationResult):
        recomputed = _aggregation_recomputed(frame, plan, result)
        if recomputed is not None:
            checks.append(recomputed)
    elif isinstance(result, GroupingResult):
        checks.append(_coded(
            "no_unexplained_missing_values",
            all(group.value is not None for group in result.groups) or result.rows_missing_value > 0,
            ValidationCode.UNEXPLAINED_MISSING,
            "A group has no value, but no row had an empty value.",
        ))
        sample = [(group.group, group.value) for group in result.groups[:_SAMPLE_GROUPS]]
        checks.append(_groups_recomputed(frame, plan, result.group_by, sample))
    elif isinstance(result, RankingResult):
        pairs = [(item.group, item.value) for item in result.items]
        checks.append(_groups_recomputed(frame, plan, result.group_by, pairs))

    return _finish(checks)    