"""Tests for the template explanations."""

import pytest

from app.schemas.query_schema import QueryPlan
from app.services.explainer import explain, uses_money
from app.tools.aggregation_tool import (
    calculate_average,
    calculate_average_order_value,
    calculate_count,
    calculate_maximum,
    calculate_total_revenue,
)
from app.tools.grouping_tool import revenue_by_group
from app.tools.missing_value_tool import analyze_missing_values
from app.tools.ranking_tool import rank_by_group
from app.utils.dataframe_utils import load_dataframe
from tests.frames import DIRECT_PATH, SALES_PATH, frame_from_csv


@pytest.fixture(scope="module")
def sales():
    return load_dataframe(SALES_PATH)


def plan(**fields) -> QueryPlan:
    return QueryPlan(confidence=0.9, **fields)


AGG = dict(intent="aggregate", tool_name="aggregation")
RANK = dict(intent="rank", tool_name="ranking", metric="revenue", aggregation="sum", group_by="region", sort_order="desc")


def test_total_revenue_derived(sales) -> None:
    text = explain(plan(intent="total_revenue", tool_name="revenue"), calculate_total_revenue(sales), "₹")

    assert "₹256,000.00" in text and "quantity multiplied by unit price" in text
    assert "'quantity'" in text and "'unit_price'" in text


def test_total_revenue_direct_and_currency_symbol() -> None:
    result = calculate_total_revenue(load_dataframe(DIRECT_PATH))

    text = explain(plan(intent="total_revenue", tool_name="revenue"), result, "$")

    assert "$121,800.00" in text and "taken from the 'Total Revenue' column" in text


def test_average_and_maximum(sales) -> None:
    avg = explain(plan(**AGG, metric="unit_price", aggregation="average"), calculate_average(sales, "unit_price"), "₹")
    high = explain(plan(**AGG, metric="quantity", aggregation="max"), calculate_maximum(sales, "quantity"), "₹")

    assert "average unit price is ₹13,807.69" in avg
    assert "maximum quantity is 5." in high


def test_record_count_and_average_order_value(sales) -> None:
    count = explain(plan(**AGG, metric="records", aggregation="count"), calculate_count(sales), "₹")
    aov = explain(plan(**AGG, metric="average_order_value", aggregation="average"), calculate_average_order_value(sales), "₹")

    assert "13 records" in count
    assert "₹25,600.00" in aov and "10 distinct orders" in aov


def test_highest_region(sales) -> None:
    text = explain(plan(**RANK, limit=1), rank_by_group(sales, group_by="region", n=1), "₹")

    assert "region with the highest total revenue is North" in text and "₹106,500.00" in text


def test_top_three_list(sales) -> None:
    result = rank_by_group(sales, group_by="product", n=3)

    text = explain(plan(**{**RANK, "group_by": "product"}, limit=3), result, "₹")

    assert "1. Laptop" in text and "2. Monitor" in text and "3. Headset" in text


def test_a_tie_at_the_cut_off_is_mentioned(sales) -> None:
    result = rank_by_group(sales, group_by="product", n=1, order="asc")

    text = explain(plan(**{**RANK, "group_by": "product", "sort_order": "asc"}, limit=1), result, "₹")

    assert "Keyboard" in text and "same value" in text


def test_grouping_summary(sales) -> None:
    text = explain(plan(intent="group", tool_name="grouping", metric="revenue", aggregation="sum", group_by="region"),
                   revenue_by_group(sales, "region"), "₹")

    assert "for each region" in text and "3 groups" in text and "North" in text


def test_missing_values(sales) -> None:
    text = explain(plan(intent="missing_values", tool_name="missing_values"), analyze_missing_values(sales), "₹")
    none = explain(plan(intent="missing_values", tool_name="missing_values"),
                   analyze_missing_values(frame_from_csv("a,b\n1,2")), "₹")

    assert "1 missing" in text and "region (1)" in text
    assert "No missing values" in none


def test_unknown_result_types_are_refused() -> None:
    with pytest.raises(ValueError):
        explain(plan(intent="total_revenue", tool_name="revenue"), object(), "₹")


def test_uses_money() -> None:
    assert uses_money(plan(intent="total_revenue", tool_name="revenue"))
    assert uses_money(plan(**AGG, metric="unit_price", aggregation="average"))
    assert not uses_money(plan(**AGG, metric="quantity", aggregation="max"))