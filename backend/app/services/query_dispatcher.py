"""Safe tool dispatcher.

A validated QueryPlan can only reach the five approved handlers in TOOL_HANDLERS. The tool name is
looked up as a plain dictionary key. Nothing is run from text, and nothing is imported at run time.
"""

from collections.abc import Callable

import pandas as pd
from pydantic import BaseModel

from app.core.errors import AppError, ErrorCode
from app.schemas.query_schema import QueryPlan
from app.tools.aggregation_tool import (
    calculate_average,
    calculate_average_order_value,
    calculate_count,
    calculate_maximum,
    calculate_minimum,
    calculate_sum,
    calculate_total_revenue,
)
from app.tools.grouping_tool import group_and_aggregate
from app.tools.missing_value_tool import analyze_missing_values
from app.tools.ranking_tool import rank_by_group

DEFAULT_RANK_LIMIT = 5
_VALUE_FOR_METRIC: dict[str, str | None] = {
    "revenue": "revenue",
    "quantity": "quantity",
    "unit_price": "unit_price",
    "records": None,  # counts rows
}


def _invalid(message: str) -> AppError:
    return AppError(ErrorCode.INVALID_QUERY_PLAN, message, status_code=422)


def _need(value: object, name: str) -> object:
    if value is None:
        raise _invalid(f"The query plan is missing '{name}'.")
    return value


def _value_for(metric: object) -> str | None:
    if metric not in _VALUE_FOR_METRIC:
        raise _invalid(f"The measure '{metric}' cannot be used here.")
    return _VALUE_FOR_METRIC[str(metric)]


def _run_total_revenue(frame: pd.DataFrame, plan: QueryPlan) -> BaseModel:
    return calculate_total_revenue(frame)


def _run_aggregation(frame: pd.DataFrame, plan: QueryPlan) -> BaseModel:
    metric = str(_need(plan.metric, "metric"))
    aggregation = str(_need(plan.aggregation, "aggregation"))
    if metric == "records":
        return calculate_count(frame)
    if metric == "average_order_value":
        return calculate_average_order_value(frame)
    if metric not in _VALUE_FOR_METRIC:
        raise _invalid(f"The measure '{metric}' cannot be used here.")
    if aggregation == "count":
        return calculate_count(frame, metric)
    functions = {
        "sum": calculate_sum,
        "average": calculate_average,
        "min": calculate_minimum,
        "max": calculate_maximum,
    }
    function = functions.get(aggregation)
    if function is None:
        raise _invalid(f"The aggregation '{aggregation}' is not supported.")
    return function(frame, metric)


def _grouped_metric(plan: QueryPlan) -> str:
    aggregation = str(_need(plan.aggregation, "aggregation"))
    if aggregation not in ("sum", "average", "count"):
        raise _invalid(f"The aggregation '{aggregation}' cannot be used for groups.")
    return aggregation


def _run_grouping(frame: pd.DataFrame, plan: QueryPlan) -> BaseModel:
    return group_and_aggregate(
        frame,
        group_by=str(_need(plan.group_by, "group_by")),
        metric=_grouped_metric(plan),  # type: ignore[arg-type]
        value=_value_for(_need(plan.metric, "metric")),
    )


def _run_ranking(frame: pd.DataFrame, plan: QueryPlan) -> BaseModel:
    return rank_by_group(
        frame,
        group_by=str(_need(plan.group_by, "group_by")),
        metric=_grouped_metric(plan),  # type: ignore[arg-type]
        value=_value_for(_need(plan.metric, "metric")),
        n=plan.limit or DEFAULT_RANK_LIMIT,
        order=plan.sort_order or "desc",
    )


def _run_missing_values(frame: pd.DataFrame, plan: QueryPlan) -> BaseModel:
    return analyze_missing_values(frame)


TOOL_HANDLERS: dict[str, Callable[[pd.DataFrame, QueryPlan], BaseModel]] = {
    "revenue": _run_total_revenue,
    "aggregation": _run_aggregation,
    "grouping": _run_grouping,
    "ranking": _run_ranking,
    "missing_values": _run_missing_values,
}


def dispatch(frame: pd.DataFrame, plan: QueryPlan) -> BaseModel:
    """Run the approved tool for this plan. Raises AppError for an unknown tool or bad arguments.

    ToolError (missing column, bad numbers, not enough data) is NOT caught here. The caller
    turns it into an "insufficient_data" answer.
    """
    handler = TOOL_HANDLERS.get(plan.tool_name or "")
    if handler is None:
        raise AppError(
            ErrorCode.UNSUPPORTED_TOOL,
            "The requested tool is not available.",
            status_code=422,
            details={"tool_name": str(plan.tool_name)[:50]},
        )
    return handler(frame, plan)