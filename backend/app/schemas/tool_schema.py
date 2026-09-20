"""Pydantic models for the results of the aggregation, grouping and ranking tools."""

from typing import Literal

from pydantic import BaseModel, Field

GroupMetric = Literal["sum", "average", "count"]
SortOrder = Literal["desc", "asc"]


class AggregationResult(BaseModel):
    tool: Literal["aggregation_tool"] = "aggregation_tool"
    metric: str
    value: float | int
    columns_used: list[str]
    calculation_method: str
    rows_used: int
    rows_excluded_missing: int
    # Only for average_order_value:
    total_revenue: float | None = None
    order_count: int | None = None
    order_count_method: str | None = None
    notes: list[str] = Field(default_factory=list)


class GroupRow(BaseModel):
    group: str | None  # None means "the group value was missing"
    value: float | int | None
    row_count: int


class GroupingResult(BaseModel):
    tool: Literal["grouping_tool"] = "grouping_tool"
    group_by: str
    metric: GroupMetric
    value_columns_used: list[str]
    calculation_method: str
    groups: list[GroupRow]
    group_count: int
    rows_used: int
    rows_missing_group: int
    rows_missing_value: int
    notes: list[str] = Field(default_factory=list)


class RankedItem(BaseModel):
    rank: int
    group: str | None
    value: float | int
    row_count: int


class RankingResult(BaseModel):
    tool: Literal["ranking_tool"] = "ranking_tool"
    group_by: str
    metric: GroupMetric
    order: SortOrder
    value_columns_used: list[str]
    calculation_method: str
    n_requested: int
    n_returned: int
    total_groups_ranked: int
    truncated_tie: bool = Field(
        description="True when the group just after the cut-off has the same value as the last returned group"
    )
    items: list[RankedItem]
    notes: list[str] = Field(default_factory=list)