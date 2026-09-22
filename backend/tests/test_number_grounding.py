"""Independent tests for the number-grounding guard."""

from decimal import Decimal

import pytest

from app.schemas.query_schema import QueryPlan
from app.services.explainer import explain
from app.services.number_grounding import build_grounding_context, check_grounding, extract_numbers
from app.tools.aggregation_tool import (
    calculate_average,
    calculate_average_order_value,
    calculate_count,
    calculate_maximum,
    calculate_sum,
    calculate_total_revenue,
)
from app.tools.grouping_tool import group_and_aggregate, revenue_by_group
from app.tools.missing_value_tool import analyze_missing_values
from app.tools.ranking_tool import rank_by_group
from app.utils.dataframe_utils import load_dataframe
from tests.frames import SALES_PATH, frame_from_csv

TOTAL_PLAN = QueryPlan(intent="total_revenue", tool_name="revenue")


def check(text: str, plan: QueryPlan, result):
    return check_grounding(text, build_grounding_context(plan, result))


@pytest.fixture(scope="module")
def sales():
    return load_dataframe(SALES_PATH)


# ---- extraction ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("₹256,000.00", [Decimal("256000")]),
        ("13 records", [Decimal(13)]),
        ("-1,234.5 and +7", [Decimal("-1234.5"), Decimal(7)]),
        ("1.5e6", [Decimal(1500000)]),
        ("2,56,000", [Decimal(256000)]),
        ("about 42% or 3 percent", [Decimal(42), Decimal(3)]),
        ("1. Laptop, 2. Monitor", [Decimal(1), Decimal(2)]),
        ("\u22125", [Decimal(-5)]),
        ("no numbers here", []),
    ],
)
def test_extract_numbers(text: str, expected: list[Decimal]) -> None:
    assert [number.value for number in extract_numbers(text)] == expected


def test_percent_detection() -> None:
    assert extract_numbers("42%")[0].is_percent and extract_numbers("3 percent")[0].is_percent
    assert not extract_numbers("3 percentage")[0].is_percent and not extract_numbers("42")[0].is_percent


# ---- supported and unsupported numbers -----------------------------------------------------------
def test_supported_numbers_pass(sales) -> None:
    result = calculate_total_revenue(sales)

    assert check("The total revenue is ₹256,000.", TOTAL_PLAN, result).passed
    assert check("The total revenue is ₹256,000.00 (256000)", TOTAL_PLAN, result).passed
    assert check("There are no numbers here.", TOTAL_PLAN, result).checked == 0


def test_a_hallucinated_percentage_is_rejected(sales) -> None:
    result = calculate_total_revenue(sales)

    outcome = check("The total revenue is ₹256,000, which represents a 42% increase.", TOTAL_PLAN, result)

    assert not outcome.passed and outcome.unsupported == ("42%",)


def test_a_hallucinated_plain_number_is_rejected(sales) -> None:
    outcome = check("The total revenue is ₹999,000.", TOTAL_PLAN, calculate_total_revenue(sales))

    assert not outcome.passed and outcome.unsupported == ("999,000",)


def test_decimal_rounding(sales) -> None:
    plan = QueryPlan(intent="aggregate", tool_name="aggregation", metric="unit_price", aggregation="average")
    result = calculate_average(sales, "unit_price")  # 13807.692308

    assert check("₹13,807.69", plan, result).passed
    assert check("₹13,807.7", plan, result).passed
    assert check("₹13,808", plan, result).passed
    assert not check("₹13,900", plan, result).passed
    assert not check("₹13,807.79", plan, result).passed


def test_negative_numbers() -> None:
    plan = QueryPlan(intent="aggregate", tool_name="aggregation", metric="quantity", aggregation="sum")
    result = calculate_sum(frame_from_csv("n\n-5\n3"), "n")  # -2

    assert check("The total is -2.", plan, result).passed
    assert check("The total is \u22122.", plan, result).passed
    assert not check("The total is 2.", plan, result).passed


def test_scientific_notation() -> None:
    plan = QueryPlan(intent="aggregate", tool_name="aggregation", metric="quantity", aggregation="sum")
    result = calculate_sum(frame_from_csv("n,x\n1500000,a"), "n")

    assert check("About 1.5e6 units.", plan, result).passed
    assert not check("About 2e6 units.", plan, result).passed


def test_a_huge_exponent_is_never_supported(sales) -> None:
    assert not check("The revenue is 1e999999999999.", TOTAL_PLAN, calculate_total_revenue(sales)).passed


def test_percentages_must_come_from_the_result(sales) -> None:
    plan = QueryPlan(intent="missing_values", tool_name="missing_values")
    report = analyze_missing_values(sales)  # missing_percentage is 1.1

    assert check("1 missing cell (1.1%).", plan, report).passed
    assert check("That is 1.1 percent.", plan, report).passed
    assert not check("That is 5%.", plan, report).passed


def test_the_plan_limit_is_an_allowed_number(sales) -> None:
    plan = QueryPlan(intent="rank", tool_name="ranking", metric="revenue", aggregation="sum",
                     group_by="product", sort_order="desc", limit=4)
    result = rank_by_group(sales, group_by="product", n=4)

    assert check("The top 4 products are listed.", plan, result).passed


# ---- names that contain digits ------------------------------------------------------------------
def test_digits_inside_group_names_are_not_checked() -> None:
    frame = frame_from_csv("store,qty,price\nStore 12,2,10\nStore 7,1,5")
    plan = QueryPlan(intent="group", tool_name="grouping", metric="revenue", aggregation="sum", group_by="store")
    result = revenue_by_group(frame, "store")

    assert check("Store 12 has revenue of ₹20.00 and Store 7 has ₹5.00.", plan, result).passed
    assert not check("Store 12 has revenue of ₹99.00.", plan, result).passed


def test_numeric_group_labels_are_allowed_numbers() -> None:
    frame = frame_from_csv("g,v\n1,10\n2,20")
    plan = QueryPlan(intent="group", tool_name="grouping", metric="quantity", aggregation="sum", group_by="g")
    result = group_and_aggregate(frame, group_by="g", metric="sum", value="v")

    assert check("Group 2 has 20.", plan, result).passed
    assert not check("Group 3 has 20.", plan, result).passed


# ---- known limitation (documented in DECISIONS D-053) -----------------------------------------------
def test_numbers_written_as_words_are_not_detected(sales) -> None:
    assert check("The total revenue is two hundred thousand.", TOTAL_PLAN, calculate_total_revenue(sales)).passed


# ---- the real templates must always be grounded ----------------------------------------------------
def test_every_template_explanation_passes_the_guard(sales) -> None:
    aggregate = lambda metric, agg: QueryPlan(intent="aggregate", tool_name="aggregation", metric=metric, aggregation=agg)  # noqa: E731
    rank = lambda group, order, limit: QueryPlan(intent="rank", tool_name="ranking", metric="revenue", aggregation="sum",  # noqa: E731
                                                 group_by=group, sort_order=order, limit=limit)
    cases = [
        (TOTAL_PLAN, calculate_total_revenue(sales)),
        (aggregate("unit_price", "average"), calculate_average(sales, "unit_price")),
        (aggregate("quantity", "max"), calculate_maximum(sales, "quantity")),
        (aggregate("records", "count"), calculate_count(sales)),
        (aggregate("quantity", "count"), calculate_count(sales, "quantity")),
        (aggregate("average_order_value", "average"), calculate_average_order_value(sales)),
        (QueryPlan(intent="group", tool_name="grouping", metric="revenue", aggregation="sum", group_by="region"),
         revenue_by_group(sales, "region")),
        (rank("region", "desc", 1), rank_by_group(sales, group_by="region", n=1)),
        (rank("product", "desc", 3), rank_by_group(sales, group_by="product", n=3)),
        (rank("product", "asc", 1), rank_by_group(sales, group_by="product", n=1, order="asc")),
        (QueryPlan(intent="missing_values", tool_name="missing_values"), analyze_missing_values(sales)),
        (QueryPlan(intent="missing_values", tool_name="missing_values"),
         analyze_missing_values(frame_from_csv("a,b\n1,2"))),
    ]
    for plan, result in cases:
        text = explain(plan, result, "₹")
        outcome = check(text, plan, result)
        assert outcome.passed, f"{text!r} -> {outcome.unsupported}"