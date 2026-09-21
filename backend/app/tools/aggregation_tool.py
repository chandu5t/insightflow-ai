"""Aggregation: sum, average, count, total revenue and average order value."""

from collections.abc import Mapping

import pandas as pd

from app.core.errors import ErrorCode, ToolError
from app.schemas.tool_schema import AggregationResult
from app.tools.revenue import compute_revenue, resolve_numeric_values
from app.utils.column_mapper import map_columns, resolve_column
from app.utils.dataframe_utils import safe_float


def _mapping(frame: pd.DataFrame, column_overrides: Mapping[str, str] | None):
    return map_columns([str(name) for name in frame.columns], overrides=column_overrides)


def _require_values(valid: pd.Series, what: str) -> None:
    if valid.empty:
        raise ToolError(
            ErrorCode.INSUFFICIENT_DATA,
            f"There are no valid values to calculate {what}.",
            details={"rows_available": 0},
        )


def _numeric_result(
    metric: str, requested: str, frame: pd.DataFrame, overrides: Mapping[str, str] | None
) -> tuple[pd.Series, AggregationResult]:
    values = resolve_numeric_values(frame, requested, _mapping(frame, overrides))
    valid = values.values.dropna()
    _require_values(valid, metric.replace("_", " "))
    result = AggregationResult(
        metric=metric,
        value=0,  # replaced by the caller
        columns_used=values.columns_used,
        calculation_method=values.method,
        rows_used=len(valid),
        rows_excluded_missing=len(frame) - len(valid),
    )
    return valid, result


def calculate_sum(
    frame: pd.DataFrame, column: str, *, column_overrides: Mapping[str, str] | None = None
) -> AggregationResult:
    """Sum of a numeric column (or of "revenue"). Empty cells are left out and counted."""
    valid, result = _numeric_result("sum", column, frame, column_overrides)
    result.value = safe_float(valid.sum()) or 0.0
    return result


def calculate_average(
    frame: pd.DataFrame, column: str, *, column_overrides: Mapping[str, str] | None = None
) -> AggregationResult:
    """Average of the valid values of a numeric column. Empty cells are not counted as zero."""
    valid, result = _numeric_result("average", column, frame, column_overrides)
    result.value = safe_float(valid.mean()) or 0.0
    return result

def calculate_minimum(
    frame: pd.DataFrame, column: str, *, column_overrides: Mapping[str, str] | None = None
) -> AggregationResult:
    """Smallest valid value of a numeric column (or of "revenue" per row)."""
    valid, result = _numeric_result("minimum", column, frame, column_overrides)
    result.value = safe_float(valid.min()) or 0.0
    return result


def calculate_maximum(
    frame: pd.DataFrame, column: str, *, column_overrides: Mapping[str, str] | None = None
) -> AggregationResult:
    """Largest valid value of a numeric column (or of "revenue" per row)."""
    valid, result = _numeric_result("maximum", column, frame, column_overrides)
    result.value = safe_float(valid.max()) or 0.0
    return result

def calculate_count(frame: pd.DataFrame, column: str | None = None) -> AggregationResult:
    """Without a column: number of rows. With a column: number of non-empty values in it."""
    if column is None:
        return AggregationResult(
            metric="count",
            value=len(frame),
            columns_used=[],
            calculation_method="row_count",
            rows_used=len(frame),
            rows_excluded_missing=0,
        )
    name = resolve_column([str(c) for c in frame.columns], column, _mapping(frame, None))
    non_missing = int(frame[name].notna().sum())
    return AggregationResult(
        metric="count",
        value=non_missing,
        columns_used=[name],
        calculation_method="non_missing_count",
        rows_used=non_missing,
        rows_excluded_missing=len(frame) - non_missing,
    )


def calculate_total_revenue(
    frame: pd.DataFrame, *, column_overrides: Mapping[str, str] | None = None
) -> AggregationResult:
    """Total revenue using the revenue rules (direct column, or quantity x unit price)."""
    valid, result = _numeric_result("total_revenue", "revenue", frame, column_overrides)
    result.value = safe_float(valid.sum()) or 0.0
    return result


def calculate_average_order_value(
    frame: pd.DataFrame,
    *,
    column_overrides: Mapping[str, str] | None = None,
    allow_row_count_fallback: bool = False,
) -> AggregationResult:
    """Average order value = total revenue / number of orders.

    - With an order id column: the number of DISTINCT order ids (not the number of rows).
    - Without one: we stop with INSUFFICIENT_DATA, unless allow_row_count_fallback=True.
      Then every row counts as one order and the method says "row_count_fallback".
    - Only rows with both a valid revenue and (if used) an order id are counted, so the
      revenue and the orders always come from the same rows.
    """
    mapping = _mapping(frame, column_overrides)
    revenue = compute_revenue(frame, mapping)
    mapping.require_if_ambiguous("order_id")
    order_column = mapping.get("order_id")

    if order_column is None and not allow_row_count_fallback:
        raise ToolError(
            ErrorCode.INSUFFICIENT_DATA,
            "Average order value needs an order id column, so orders can be counted correctly.",
            details={"missing_roles": ["order_id"], "accepted_names": list(mapping.accepted_names["order_id"])},
        )

    usable = revenue.values.notna()
    if order_column is not None:
        usable &= frame[order_column].notna()
        order_count = int(frame.loc[usable, order_column].nunique())
        order_method = "distinct_order_id_count"
        columns_used = [*revenue.columns_used, order_column]
    else:
        order_count = int(usable.sum())
        order_method = "row_count_fallback"
        columns_used = list(revenue.columns_used)

    if order_count == 0:
        raise ToolError(
            ErrorCode.INSUFFICIENT_DATA,
            "There are no orders to average, so the average order value cannot be calculated.",
            details={"order_count": 0},
        )

    total_revenue = float(revenue.values[usable].sum())
    rows_used = int(usable.sum())
    notes: list[str] = []
    if order_method == "row_count_fallback":
        notes.append("There is no order id column, so each row was counted as one order.")
    if rows_used < len(frame):
        notes.append(f"{len(frame) - rows_used} row(s) were left out because revenue or order id was empty.")

    return AggregationResult(
        metric="average_order_value",
        value=safe_float(total_revenue / order_count) or 0.0,
        columns_used=columns_used,
        calculation_method=revenue.method,
        rows_used=rows_used,
        rows_excluded_missing=len(frame) - rows_used,
        total_revenue=safe_float(total_revenue),
        order_count=order_count,
        order_count_method=order_method,
        notes=notes,
    )