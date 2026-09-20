"""Ranking: top N or bottom N groups. The order is stable, so results never change between runs."""

from collections.abc import Mapping

import pandas as pd

from app.core.errors import ErrorCode, ToolError
from app.schemas.tool_schema import GroupMetric, GroupingResult, RankedItem, RankingResult, SortOrder
from app.tools.grouping_tool import group_and_aggregate

MAX_TOP_N = 100


def _validate_n(n: object) -> int:
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= MAX_TOP_N:
        raise ToolError(
            ErrorCode.INVALID_PARAMETER,
            f"n must be a whole number from 1 to {MAX_TOP_N}.",
            details={"n": str(n), "min": 1, "max": MAX_TOP_N},
        )
    return n


def rank_groups(grouping: GroupingResult, *, n: int = 5, order: SortOrder = "desc") -> RankingResult:
    """Rank the groups of a grouping result by their value.

    Sorting: by value (high to low for "desc", low to high for "asc"), then by group name
    A to Z (ignoring letter case). Equal values share the same rank (1, 2, 2, 4).
    truncated_tie is True when the cut-off at N splits a tie.
    """
    n = _validate_n(n)
    if order not in ("desc", "asc"):
        raise ToolError(
            ErrorCode.INVALID_PARAMETER,
            "order must be 'desc' or 'asc'.",
            details={"order": str(order)},
        )

    candidates = [row for row in grouping.groups if row.value is not None]
    sign = -1 if order == "desc" else 1
    candidates.sort(
        key=lambda row: (
            sign * float(row.value),  # type: ignore[arg-type]
            row.group is None,
            (row.group or "").casefold(),
            row.group or "",
        )
    )

    items: list[RankedItem] = []
    previous_value: float | int | None = None
    previous_rank = 0
    for position, row in enumerate(candidates, start=1):
        rank = previous_rank if row.value == previous_value else position
        previous_value, previous_rank = row.value, rank
        items.append(RankedItem(rank=rank, group=row.group, value=row.value, row_count=row.row_count))  # type: ignore[arg-type]

    truncated_tie = len(items) > n and items[n].value == items[n - 1].value
    notes = list(grouping.notes)
    if truncated_tie:
        notes.append("The cut-off splits a tie: more groups have the same value as the last one shown.")

    return RankingResult(
        group_by=grouping.group_by,
        metric=grouping.metric,
        order=order,
        value_columns_used=grouping.value_columns_used,
        calculation_method=grouping.calculation_method,
        n_requested=n,
        n_returned=min(n, len(items)),
        total_groups_ranked=len(items),
        truncated_tie=truncated_tie,
        items=items[:n],
        notes=notes,
    )


def rank_by_group(
    frame: pd.DataFrame,
    *,
    group_by: str,
    metric: GroupMetric = "sum",
    value: str | None = "revenue",
    n: int = 5,
    order: SortOrder = "desc",
    include_missing_group: bool = False,
    column_overrides: Mapping[str, str] | None = None,
) -> RankingResult:
    """Group, then rank. Example: top 5 products by revenue."""
    _validate_n(n)  # check before doing any work
    grouping = group_and_aggregate(
        frame,
        group_by=group_by,
        metric=metric,
        value=value,
        include_missing_group=include_missing_group,
        column_overrides=column_overrides,
    )
    return rank_groups(grouping, n=n, order=order)