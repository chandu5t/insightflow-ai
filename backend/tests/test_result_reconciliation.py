"""Tests for the reconciliation checks and the merged validation."""

import pytest

from app.schemas.query_schema import QueryPlan, ValidationCheck, ValidationInfo
from app.services.result_validator import (
    ValidationCode,
    merge_validations,
    reconcile_result,
    validate_result,
)
from app.tools.aggregation_tool import (
    calculate_average,
    calculate_average_order_value,
    calculate_count,
    calculate_maximum,
    calculate_sum,
    calculate_total_revenue,
)
from app.tools.grouping_tool import revenue_by_group
from app.tools.missing_value_tool import analyze_missing_values
from app.tools.ranking_tool import rank_by_group
from app.utils.dataframe_utils import load_dataframe
from tests.frames import SALES_PATH, frame_from_csv

TOTAL = QueryPlan(intent="total_revenue", tool_name="revenue")
AGG = dict(intent="aggregate", tool_name="aggregation")


@pytest.fixture(scope="module")
def sales():
    return load_dataframe(SALES_PATH)


def failed(info) -> dict[str, str | None]:
    return {check.name: check.code for check in info.checks if not check.passed}


def names(info) -> set[str]:
    return {check.name for check in info.checks}


# ---- valid results pass ------------------------------------------------------------------------------
def test_valid_aggregations_pass(sales) -> None:
    info = reconcile_result(sales, TOTAL, calculate_total_revenue(sales), expected_row_count=13)

    assert info.status == "passed"
    assert {"result_type_matches_tool", "operation_matches_plan", "frame_matches_metadata",
            "aggregation_recomputed"} <= names(info)
    for plan, result in [
        (QueryPlan(**AGG, metric="unit_price", aggregation="average"), calculate_average(sales, "unit_price")),
        (QueryPlan(**AGG, metric="quantity", aggregation="max"), calculate_maximum(sales, "quantity")),
        (QueryPlan(**AGG, metric="records", aggregation="count"), calculate_count(sales)),
        (QueryPlan(**AGG, metric="average_order_value", aggregation="average"), calculate_average_order_value(sales)),
    ]:
        assert reconcile_result(sales, plan, result, expected_row_count=13).status == "passed"


def test_valid_grouping_ranking_and_missing_report_pass(sales) -> None:
    group_plan = QueryPlan(intent="group", tool_name="grouping", metric="revenue", aggregation="sum", group_by="product")
    rank_plan = QueryPlan(intent="rank", tool_name="ranking", metric="revenue", aggregation="sum",
                          group_by="product", sort_order="desc", limit=3)
    missing_plan = QueryPlan(intent="missing_values", tool_name="missing_values")

    assert reconcile_result(sales, group_plan, revenue_by_group(sales, "product")).status == "passed"
    assert reconcile_result(sales, rank_plan, rank_by_group(sales, group_by="product", n=3)).status == "passed"
    assert reconcile_result(sales, missing_plan, analyze_missing_values(sales), expected_row_count=13).status == "passed"


# ---- failures, each with a machine-readable code ----------------------------------------------------------
def test_wrong_result_type_fails(sales) -> None:
    plan = QueryPlan(**AGG, metric="quantity", aggregation="sum")

    info = reconcile_result(sales, plan, revenue_by_group(sales, "product"))

    assert info.status == "failed" and failed(info) == {"result_type_matches_tool": ValidationCode.WRONG_RESULT_TYPE.value}


def test_unknown_result_object_fails_without_raising(sales) -> None:
    assert failed(reconcile_result(sales, TOTAL, object())) == {"result_type_matches_tool": "WRONG_RESULT_TYPE"}


def test_operation_mismatch_fails(sales) -> None:
    plan = QueryPlan(**AGG, metric="quantity", aggregation="max")

    info = reconcile_result(sales, plan, calculate_average(sales, "quantity"))  # the tool did "average"

    assert failed(info)["operation_matches_plan"] == "OPERATION_MISMATCH"


def test_a_plan_with_missing_required_fields_fails_cleanly(sales) -> None:
    broken = QueryPlan.model_construct(intent="aggregate", tool_name="aggregation", metric="quantity", aggregation=None)

    info = reconcile_result(sales, broken, calculate_sum(sales, "quantity"))

    assert failed(info)["operation_matches_plan"] == "OPERATION_MISMATCH"


def test_ranking_order_and_size_must_match_the_plan(sales) -> None:
    plan = QueryPlan(intent="rank", tool_name="ranking", metric="revenue", aggregation="sum",
                     group_by="product", sort_order="desc", limit=3)

    ascending = rank_by_group(sales, group_by="product", n=3, order="asc")

    assert failed(reconcile_result(sales, plan, ascending))["operation_matches_plan"] == "OPERATION_MISMATCH"


def test_dataset_and_metadata_row_counts_must_agree(sales) -> None:
    info = reconcile_result(sales, TOTAL, calculate_total_revenue(sales), expected_row_count=12)

    assert failed(info)["frame_matches_metadata"] == "FRAME_ROW_MISMATCH"


def test_record_count_must_match_the_metadata(sales) -> None:
    plan = QueryPlan(**AGG, metric="records", aggregation="count")
    tampered = calculate_count(sales).model_copy(update={"value": 12})

    info = reconcile_result(sales, plan, tampered, expected_row_count=13)

    assert failed(info) == {"record_count_matches_metadata": "COUNT_MISMATCH"}


def test_a_recomputation_mismatch_fails(sales) -> None:
    tampered = calculate_total_revenue(sales).model_copy(update={"value": 256001.0})

    assert failed(reconcile_result(sales, TOTAL, tampered)) == {"aggregation_recomputed": "RECONCILIATION_MISMATCH"}


def test_average_order_value_must_equal_total_over_orders(sales) -> None:
    plan = QueryPlan(**AGG, metric="average_order_value", aggregation="average")
    tampered = calculate_average_order_value(sales).model_copy(update={"value": 30000.0})

    assert failed(reconcile_result(sales, plan, tampered)) == {"average_order_value_consistent": "RECONCILIATION_MISMATCH"}


def test_a_wrong_group_value_fails(sales) -> None:
    plan = QueryPlan(intent="group", tool_name="grouping", metric="revenue", aggregation="sum", group_by="product")
    result = revenue_by_group(sales, "product")
    groups = [result.groups[0].model_copy(update={"value": result.groups[0].value + 1}), *result.groups[1:]]

    info = reconcile_result(sales, plan, result.model_copy(update={"groups": groups}))

    assert failed(info) == {"group_values_recomputed": "RECONCILIATION_MISMATCH"}


def test_a_wrong_ranked_value_fails(sales) -> None:
    plan = QueryPlan(intent="rank", tool_name="ranking", metric="revenue", aggregation="sum",
                     group_by="product", sort_order="desc", limit=3)
    result = rank_by_group(sales, group_by="product", n=3)
    items = [result.items[0].model_copy(update={"value": 1.0}), *result.items[1:]]

    info = reconcile_result(sales, plan, result.model_copy(update={"items": items}))

    assert failed(info) == {"group_values_recomputed": "RECONCILIATION_MISMATCH"}


def test_an_empty_group_value_needs_an_explanation(sales) -> None:
    plan = QueryPlan(intent="group", tool_name="grouping", metric="revenue", aggregation="sum", group_by="product")
    result = revenue_by_group(sales, "product")
    groups = [result.groups[0].model_copy(update={"value": None}), *result.groups[1:]]

    info = reconcile_result(sales, plan, result.model_copy(update={"groups": groups}))

    assert failed(info)["no_unexplained_missing_values"] == "UNEXPLAINED_MISSING"


# ---- floating-point tolerance --------------------------------------------------------------------------------
def test_tiny_rounding_differences_are_accepted(sales) -> None:
    nearly = calculate_total_revenue(sales).model_copy(update={"value": 256000.0 + 1e-9})

    assert reconcile_result(sales, TOTAL, nearly).status == "passed"


def test_classic_floating_point_sums_are_accepted() -> None:
    plan = QueryPlan(intent="total_revenue", tool_name="revenue")
    frame = frame_from_csv("revenue,x\n0.1,a\n0.2,b")  # 0.1 + 0.2 = 0.30000000000000004 in binary floating point

    assert reconcile_result(frame, plan, calculate_total_revenue(frame)).status == "passed"

# ---- merging and regression --------------------------------------------------------------------------------------
def test_merge_validations_keeps_all_checks_and_fails_if_any_failed() -> None:
    good = ValidationInfo(status="passed", checks=[ValidationCheck(name="a", passed=True)])
    bad = ValidationInfo(status="failed", checks=[ValidationCheck(name="b", passed=False, code="X", detail="d")])

    merged = merge_validations(good, bad)

    assert merged.status == "failed" and [c.name for c in merged.checks] == ["a", "b"]
    assert merge_validations(good, good).status == "passed"


def test_the_existing_validator_is_unchanged(sales) -> None:
    info = validate_result(sales, TOTAL, calculate_total_revenue(sales))

    assert info.status == "passed" and {c.name for c in info.checks} == {"finite_numbers", "row_accounting"}
    assert all(check.code is None for check in info.checks)
