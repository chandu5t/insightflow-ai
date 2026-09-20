"""Grouping: sum, average or count of a value for each group of one selected column."""

from collections.abc import Mapping

import pandas as pd

from app.core.errors import ErrorCode, ToolError
from app.schemas.tool_schema import GroupMetric, GroupRow, GroupingResult
from app.tools.revenue import resolve_numeric_values
from app.utils.column_mapper import map_columns, resolve_column
from app.utils.dataframe_utils import safe_float

# Used only inside this file to keep "missing group" rows together. Real data cannot contain it.
_MISSING_KEY = "\x00missing"


def _group_sort_key(group: str | None) -> tuple[bool, str, str]:
    # Alphabetical (ignoring letter case). The missing group is always last.
    return (group is None, (group or "").casefold(), group or "")


def group_and_aggregate(
    frame: pd.DataFrame,
    *,
    group_by: str,
    metric: GroupMetric = "sum",
    value: str | None = "revenue",
    include_missing_group: bool = False,
    column_overrides: Mapping[str, str] | None = None,
) -> GroupingResult:
    """Group by ONE column and calculate a metric for each group.

    group_by   a role (product, region, ...) or an exact column name
    metric     sum, average or count
    value      what to calculate: "revenue" (revenue rules), a role, or a numeric column name.
               For count, value=None counts rows.
    Rows whose group is empty are left out and counted in rows_missing_group,
    unless include_missing_group=True (then they appear as group None).
    Groups are returned in alphabetical order. Use the ranking tool to rank them.
    """
    if metric not in ("sum", "average", "count"):
        raise ToolError(
            ErrorCode.INVALID_PARAMETER,
            "metric must be 'sum', 'average' or 'count'.",
            details={"metric": str(metric)},
        )

    mapping = map_columns([str(name) for name in frame.columns], overrides=column_overrides)
    group_column = resolve_column([str(name) for name in frame.columns], group_by, mapping)

    if value is None:
        if metric != "count":
            raise ToolError(ErrorCode.INVALID_PARAMETER, f"metric '{metric}' needs a value column.")
        values = pd.Series(1.0, index=frame.index)
        method, columns_used = "row_count", []
    else:
        numbers = resolve_numeric_values(frame, value, mapping)
        values, method, columns_used = numbers.values, numbers.method, numbers.columns_used

    groups = frame[group_column]
    rows_missing_group = int(groups.isna().sum())
    work = pd.DataFrame({"group": groups.fillna(_MISSING_KEY) if include_missing_group else groups, "value": values})
    work = work[work["group"].notna()]
    valid = work[work["value"].notna()]

    row_counts = work.groupby("group").size()
    if metric == "sum":
        aggregated = valid.groupby("group")["value"].sum()
    elif metric == "average":
        aggregated = valid.groupby("group")["value"].mean()
    else:
        aggregated = valid.groupby("group")["value"].count()

    rows: list[GroupRow] = []
    for key in row_counts.index:
        group = None if key == _MISSING_KEY else str(key)
        if key in aggregated.index:
            number = aggregated[key]
            group_value: float | int | None = int(number) if metric == "count" else safe_float(number)
        else:
            group_value = 0 if metric == "count" else None  # no valid value in this group
        rows.append(GroupRow(group=group, value=group_value, row_count=int(row_counts[key])))
    rows.sort(key=lambda row: _group_sort_key(row.group))

    notes: list[str] = []
    if rows_missing_group and not include_missing_group:
        notes.append(f"{rows_missing_group} row(s) were left out because '{group_column}' was empty.")
    if any(row.value is None for row in rows):
        notes.append("Some groups have no valid values, so their result is empty (null).")

    return GroupingResult(
        group_by=group_column,
        metric=metric,
        value_columns_used=columns_used,
        calculation_method=method,
        groups=rows,
        group_count=len(rows),
        rows_used=len(valid),
        rows_missing_group=rows_missing_group,
        rows_missing_value=int(len(work) - len(valid)),
        notes=notes,
    )


def revenue_by_group(frame: pd.DataFrame, group_by: str, **options: object) -> GroupingResult:
    """Total revenue for each group, for example revenue_by_group(frame, "product")."""
    return group_and_aggregate(frame, group_by=group_by, metric="sum", value="revenue", **options)  # type: ignore[arg-type]


def quantity_by_group(frame: pd.DataFrame, group_by: str, **options: object) -> GroupingResult:
    """Total quantity for each group, for example quantity_by_group(frame, "product")."""
    return group_and_aggregate(frame, group_by=group_by, metric="sum", value="quantity", **options)  # type: ignore[arg-type]