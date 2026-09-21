"""Tests for the QueryPlan schema."""

import pytest
from pydantic import ValidationError

from app.schemas.query_schema import QueryPlan

RANK = dict(
    intent="rank", tool_name="ranking", metric="revenue", aggregation="sum",
    group_by="region", sort_order="desc", limit=3, confidence=0.9,
)


def make(**overrides) -> QueryPlan:
    return QueryPlan(**{**RANK, **overrides})


def test_valid_plans() -> None:
    assert make().limit == 3
    QueryPlan(intent="total_revenue", tool_name="revenue", confidence=0.9)
    QueryPlan(intent="aggregate", tool_name="aggregation", metric="quantity", aggregation="max")
    QueryPlan(intent="group", tool_name="grouping", metric="records", aggregation="count", group_by="product")
    QueryPlan(intent="missing_values", tool_name="missing_values")
    QueryPlan(intent="definition")
    QueryPlan(intent="unsupported", reasoning="Not a data question.")


def test_optional_fields_default_to_none() -> None:
    plan = QueryPlan(intent="total_revenue", tool_name="revenue")

    assert plan.metric is None and plan.aggregation is None and plan.group_by is None
    assert plan.sort_order is None and plan.limit is None and plan.reasoning == ""


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("intent", "delete_everything"), ("tool_name", "run_python"), ("metric", "profit"),
        ("aggregation", "median"), ("sort_order", "up"), ("limit", 0), ("limit", 101), ("limit", -5),
        ("confidence", 1.5), ("confidence", -0.1),
    ],
)
def test_invalid_field_values_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        make(**{field: value})


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ValidationError):
        make(code="import os")


def test_intent_and_tool_must_match() -> None:
    with pytest.raises(ValidationError, match="needs tool_name"):
        QueryPlan(intent="total_revenue", tool_name="grouping")
    with pytest.raises(ValidationError, match="needs tool_name"):
        QueryPlan(intent="definition", tool_name="revenue")
    with pytest.raises(ValidationError, match="needs tool_name"):
        make(tool_name=None)


def test_required_fields_per_intent() -> None:
    with pytest.raises(ValidationError, match="needs group_by"):
        make(group_by=None)
    with pytest.raises(ValidationError, match="needs sort_order"):
        make(sort_order=None)
    with pytest.raises(ValidationError, match="needs metric and aggregation"):
        QueryPlan(intent="aggregate", tool_name="aggregation", metric="quantity")
    with pytest.raises(ValidationError):
        make(group_by="   ")


@pytest.mark.parametrize(
    ("metric", "aggregation"),
    [("records", "sum"), ("average_order_value", "sum"), ("revenue", "count")],
)
def test_disallowed_metric_aggregation_pairs_for_aggregate(metric: str, aggregation: str) -> None:
    with pytest.raises(ValidationError, match="not allowed"):
        QueryPlan(intent="aggregate", tool_name="aggregation", metric=metric, aggregation=aggregation)


def test_grouping_cannot_use_min_max() -> None:
    with pytest.raises(ValidationError, match="not allowed"):
        make(intent="group", tool_name="grouping", aggregation="max", sort_order=None, limit=None)


def test_long_reasoning_is_shortened_not_rejected() -> None:
    assert len(make(reasoning="x" * 1000).reasoning) == 300