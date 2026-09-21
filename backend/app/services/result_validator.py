"""Checks that a tool result is consistent before it is shown to the user."""

import math
from collections.abc import Iterable

import pandas as pd

from app.core.errors import ToolError
from app.schemas.profile_schema import MissingValueReport
from app.schemas.query_schema import QueryPlan, ValidationCheck, ValidationInfo
from app.schemas.tool_schema import AggregationResult, GroupingResult, RankingResult
from app.tools.aggregation_tool import calculate_sum


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