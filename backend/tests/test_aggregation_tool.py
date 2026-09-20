"""Tests for the aggregation tool (sum, average, count, total revenue, average order value)."""

import pytest

from app.core.errors import ErrorCode, ToolError
from app.tools.aggregation_tool import (
    calculate_average,
    calculate_average_order_value,
    calculate_count,
    calculate_sum,
    calculate_total_revenue,
)
from tests.frames import frame_from_csv


def code_of(callable_, *args, **kwargs) -> ErrorCode:
    with pytest.raises(ToolError) as info:
        callable_(*args, **kwargs)
    return info.value.code


# ---- sum ----------------------------------------------------------------------------------------
def test_sum_of_a_numeric_column() -> None:
    result = calculate_sum(frame_from_csv("qty\n1\n2\n3.5"), "qty")

    assert result.metric == "sum" and result.value == 6.5
    assert result.columns_used == ["qty"] and result.calculation_method == "numeric_column"
    assert (result.rows_used, result.rows_excluded_missing) == (3, 0)


def test_sum_skips_empty_cells_and_counts_them() -> None:
    result = calculate_sum(frame_from_csv("qty,x\n1,a\n,b\n3,c"), "qty")

    assert result.value == 4.0 and result.rows_used == 2 and result.rows_excluded_missing == 1


def test_sum_by_role_name_uses_the_mapper() -> None:
    assert calculate_sum(frame_from_csv("Units Sold\n2\n3"), "quantity").value == 5.0


def test_sum_of_revenue_uses_the_revenue_rules() -> None:
    result = calculate_sum(frame_from_csv("qty,price\n2,10\n1,5"), "revenue")

    assert result.value == 25.0 and result.calculation_method == "derived_quantity_times_unit_price"


def test_sum_of_negative_values() -> None:
    assert calculate_sum(frame_from_csv("n\n5\n-8"), "n").value == -3.0


def test_sum_of_zero_is_zero() -> None:
    assert calculate_sum(frame_from_csv("n\n0\n0"), "n").value == 0.0


def test_sum_with_no_valid_values_is_insufficient_data() -> None:
    assert code_of(calculate_sum, frame_from_csv("n,m\n,1\n,2"), "n") == ErrorCode.INSUFFICIENT_DATA


# ---- average --------------------------------------------------------------------------------------
def test_average_ignores_empty_cells_instead_of_counting_zero() -> None:
    result = calculate_average(frame_from_csv("n,x\n10,a\n,b\n20,c"), "n")

    assert result.value == 15.0 and result.rows_used == 2 and result.rows_excluded_missing == 1


def test_average_is_rounded_to_six_decimals() -> None:
    assert calculate_average(frame_from_csv("n\n1\n1\n2"), "n").value == 1.333333


def test_average_with_no_valid_values_is_insufficient_data() -> None:
    assert code_of(calculate_average, frame_from_csv("n,m\n,1"), "n") == ErrorCode.INSUFFICIENT_DATA


# ---- count ------------------------------------------------------------------------------------------
def test_count_rows() -> None:
    result = calculate_count(frame_from_csv("a,b\n1,\n2,\n3,"))

    assert result.value == 3 and result.calculation_method == "row_count" and result.columns_used == []


def test_count_non_empty_values_of_a_column() -> None:
    result = calculate_count(frame_from_csv("a,b\n1,x\n2,\n3,"), "b")

    assert result.value == 1 and result.rows_excluded_missing == 2 and result.calculation_method == "non_missing_count"


def test_count_works_on_text_columns_and_role_names() -> None:
    frame = frame_from_csv("Product Name\nx\ny")

    assert calculate_count(frame, "product").value == 2 and calculate_count(frame, "Product Name").value == 2


def test_count_of_an_empty_dataset_is_zero() -> None:
    assert calculate_count(frame_from_csv("a\n1").iloc[0:0]).value == 0


# ---- total revenue -----------------------------------------------------------------------------------
def test_total_revenue_from_a_direct_column() -> None:
    result = calculate_total_revenue(frame_from_csv("Total Revenue,qty,price\n100,1,1\n50,1,1"))

    assert result.metric == "total_revenue" and result.value == 150.0
    assert result.calculation_method == "direct_revenue_column" and result.columns_used == ["Total Revenue"]


def test_total_revenue_derived() -> None:
    result = calculate_total_revenue(frame_from_csv("qty,price\n2,10\n3,5"))

    assert result.value == 35.0 and result.calculation_method == "derived_quantity_times_unit_price"
    assert result.columns_used == ["qty", "price"]


def test_total_revenue_leaves_out_rows_with_empty_inputs() -> None:
    result = calculate_total_revenue(frame_from_csv("qty,price\n2,10\n,5\n1,"))

    assert result.value == 20.0 and result.rows_used == 1 and result.rows_excluded_missing == 2


def test_total_revenue_needs_revenue_or_quantity_and_price() -> None:
    assert code_of(calculate_total_revenue, frame_from_csv("product\na")) == ErrorCode.INSUFFICIENT_DATA


def test_total_revenue_rejects_ambiguous_columns_and_accepts_an_override() -> None:
    frame = frame_from_csv("revenue,sales\n1,10\n2,20")

    assert code_of(calculate_total_revenue, frame) == ErrorCode.AMBIGUOUS_COLUMN
    assert calculate_total_revenue(frame, column_overrides={"revenue": "sales"}).value == 30.0


def test_total_revenue_of_an_empty_dataset_is_insufficient_data() -> None:
    assert code_of(calculate_total_revenue, frame_from_csv("revenue\n1").iloc[0:0]) == ErrorCode.INSUFFICIENT_DATA


# ---- invalid values and missing columns ---------------------------------------------------------------
@pytest.mark.parametrize("function", [calculate_sum, calculate_average])
def test_invalid_numeric_values_are_rejected(function) -> None:
    assert code_of(function, frame_from_csv("n\n1\nabc"), "n") == ErrorCode.INVALID_NUMERIC_VALUES


@pytest.mark.parametrize("function", [calculate_sum, calculate_average, calculate_count])
def test_missing_column_is_a_clear_error(function) -> None:
    assert code_of(function, frame_from_csv("a\n1"), "nope") == ErrorCode.MISSING_COLUMN


def test_missing_role_column_is_a_clear_error() -> None:
    assert code_of(calculate_sum, frame_from_csv("a\n1"), "quantity") == ErrorCode.MISSING_COLUMN


# ---- average order value ---------------------------------------------------------------------------------
ORDERS = """
    order_id,qty,price
    A,1,100
    A,1,100
    B,2,50
    C,1,200
"""  # 4 rows, 3 distinct orders, total revenue 500


def test_aov_uses_distinct_order_ids_and_not_the_row_count() -> None:
    result = calculate_average_order_value(frame_from_csv(ORDERS))

    assert result.order_count == 3 and result.order_count_method == "distinct_order_id_count"
    assert result.total_revenue == 500.0
    assert result.value == pytest.approx(166.666667, abs=1e-6)  # NOT 500 / 4 = 125
    assert result.value != 125.0


def test_aov_with_repeated_order_ids_counts_each_order_once() -> None:
    result = calculate_average_order_value(frame_from_csv("order_id,revenue\nX,10\nX,20\nX,30"))

    assert result.order_count == 1 and result.value == 60.0


def test_aov_direct_revenue() -> None:
    result = calculate_average_order_value(frame_from_csv("Order No,Sales\n1,100\n2,300"))

    assert result.value == 200.0 and result.calculation_method == "direct_revenue_column"
    assert result.columns_used == ["Sales", "Order No"]


def test_aov_derived_revenue() -> None:
    result = calculate_average_order_value(frame_from_csv(ORDERS))

    assert result.calculation_method == "derived_quantity_times_unit_price"
    assert result.columns_used == ["qty", "price", "order_id"]


def test_aov_without_order_id_does_not_estimate() -> None:
    error_code = code_of(calculate_average_order_value, frame_from_csv("revenue\n10\n20"))

    assert error_code == ErrorCode.INSUFFICIENT_DATA


def test_aov_without_order_id_can_use_row_count_only_when_asked() -> None:
    result = calculate_average_order_value(frame_from_csv("revenue\n10\n20"), allow_row_count_fallback=True)

    assert result.value == 15.0 and result.order_count == 2 and result.order_count_method == "row_count_fallback"
    assert any("each row was counted as one order" in note for note in result.notes)


def test_aov_without_revenue_fields_does_not_estimate() -> None:
    assert code_of(calculate_average_order_value, frame_from_csv("order_id,product\n1,a")) == ErrorCode.INSUFFICIENT_DATA


def test_aov_with_zero_orders_does_not_divide_by_zero() -> None:
    all_blank_ids = frame_from_csv("order_id,revenue\n,10\n,20")

    assert code_of(calculate_average_order_value, all_blank_ids) == ErrorCode.INSUFFICIENT_DATA
    assert code_of(calculate_average_order_value, frame_from_csv("order_id,revenue\n1,5").iloc[0:0]) == ErrorCode.INSUFFICIENT_DATA


def test_aov_leaves_out_rows_without_order_id_or_revenue_and_says_so() -> None:
    result = calculate_average_order_value(frame_from_csv("order_id,revenue\n1,100\n,50\n2,\n3,300"))

    assert result.order_count == 2 and result.total_revenue == 400.0 and result.value == 200.0
    assert result.rows_used == 2 and result.rows_excluded_missing == 2
    assert any("2 row(s) were left out" in note for note in result.notes)


def test_aov_rejects_ambiguous_order_id_columns() -> None:
    assert code_of(calculate_average_order_value, frame_from_csv("order_id,order_no,revenue\n1,1,5")) == ErrorCode.AMBIGUOUS_COLUMN


def test_aov_rejects_invalid_revenue_values() -> None:
    assert code_of(calculate_average_order_value, frame_from_csv("order_id,revenue\n1,abc")) == ErrorCode.INVALID_NUMERIC_VALUES