"""Tests for the result validator."""

import pytest

from app.schemas.query_schema import QueryPlan
from app.services.result_validator import validate_result
from app.tools.aggregation_tool import calculate_total_revenue
from app.tools.grouping_tool import revenue_by_group
from app.tools.missing_value_tool import analyze_missing_values
from app.tools.ranking_tool import rank_by_group
from app.utils.dataframe_utils import load_dataframe
from tests.frames import SALES_PATH


@pytest.fixture(scope="module")
def sales():
    return load_dataframe(SALES_PATH)


def group_plan(group_by: str) -> QueryPlan:
    return QueryPlan(intent="group", tool_name="grouping", metric="revenue", aggregation="sum", group_by=group_by)


TOTAL_PLAN = QueryPlan(intent="total_revenue", tool_name="revenue")


def failed(info) -> set[str]:
    return {check.name for check in info.checks if not check.passed}


def test_a_good_aggregation_passes(sales) -> None:
    info = validate_result(sales, TOTAL_PLAN, calculate_total_revenue(sales))

    assert info.status == "passed" and {c.name for c in info.checks} == {"finite_numbers", "row_accounting"}


def test_wrong_row_accounting_fails(sales) -> None:
    bad = calculate_total_revenue(sales).model_copy(update={"rows_used": 5})

    info = validate_result(sales, TOTAL_PLAN, bad)

    assert info.status == "failed" and failed(info) == {"row_accounting"}


def test_a_value_that_is_not_finite_fails(sales) -> None:
    bad = calculate_total_revenue(sales).model_copy(update={"value": float("nan")})

    assert "finite_numbers" in failed(validate_result(sales, TOTAL_PLAN, bad))


def test_grouping_passes_and_group_totals_are_compared_with_the_overall_total(sales) -> None:
    info = validate_result(sales, group_plan("product"), revenue_by_group(sales, "product"))

    assert info.status == "passed" and "group_totals_match_overall" in {c.name for c in info.checks}


def test_group_total_check_is_skipped_when_rows_were_left_out(sales) -> None:
    info = validate_result(sales, group_plan("region"), revenue_by_group(sales, "region"))  # one empty region

    assert info.status == "passed" and "group_totals_match_overall" not in {c.name for c in info.checks}


def test_wrong_group_values_fail(sales) -> None:
    result = revenue_by_group(sales, "product")
    tampered = [g.model_copy(update={"value": g.value + 1}) for g in result.groups]

    info = validate_result(sales, group_plan("product"), result.model_copy(update={"groups": tampered}))

    assert "group_totals_match_overall" in failed(info)


def test_ranking_passes_and_wrong_order_fails(sales) -> None:
    plan = QueryPlan(intent="rank", tool_name="ranking", metric="revenue", aggregation="sum",
                     group_by="product", sort_order="desc", limit=5)
    ranking = rank_by_group(sales, group_by="product", n=5)

    assert validate_result(sales, plan, ranking).status == "passed"
    reversed_result = ranking.model_copy(update={"items": list(reversed(ranking.items))})
    assert "ranking_order" in failed(validate_result(sales, plan, reversed_result))


def test_missing_value_report_is_checked(sales) -> None:
    plan = QueryPlan(intent="missing_values", tool_name="missing_values")
    report = analyze_missing_values(sales)

    assert validate_result(sales, plan, report).status == "passed"
    bad = report.model_copy(update={"total_missing_cells": 99})
    assert validate_result(sales, plan, bad).status == "failed"


def test_unknown_result_type_fails(sales) -> None:
    assert validate_result(sales, TOTAL_PLAN, object()).status == "failed"