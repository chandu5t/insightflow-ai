"""Tests for the minimum and maximum calculations."""

import pytest

from app.core.errors import ErrorCode, ToolError
from app.tools.aggregation_tool import calculate_maximum, calculate_minimum
from tests.frames import frame_from_csv


def code_of(function, *args) -> ErrorCode:
    with pytest.raises(ToolError) as info:
        function(*args)
    return info.value.code


def test_minimum_and_maximum_of_a_column() -> None:
    frame = frame_from_csv("qty,x\n3,a\n1,b\n7,c")

    low, high = calculate_minimum(frame, "qty"), calculate_maximum(frame, "qty")

    assert (low.value, high.value) == (1.0, 7.0)
    assert (low.metric, high.metric) == ("minimum", "maximum")
    assert high.columns_used == ["qty"]


def test_empty_cells_are_left_out_and_counted() -> None:
    result = calculate_maximum(frame_from_csv("qty,x\n3,a\n,b\n9,c"), "qty")

    assert result.value == 9.0 and result.rows_used == 2 and result.rows_excluded_missing == 1


def test_role_names_and_negative_values() -> None:
    assert calculate_maximum(frame_from_csv("Units Sold\n2\n5"), "quantity").value == 5.0
    assert calculate_minimum(frame_from_csv("n\n-5\n3"), "n").value == -5.0


def test_maximum_of_revenue_uses_the_revenue_rules() -> None:
    result = calculate_maximum(frame_from_csv("qty,price\n2,10\n1,50"), "revenue")

    assert result.value == 50.0 and result.calculation_method == "derived_quantity_times_unit_price"


def test_errors_are_clear() -> None:
    assert code_of(calculate_maximum, frame_from_csv("n,m\n,1\n,2"), "n") == ErrorCode.INSUFFICIENT_DATA
    assert code_of(calculate_minimum, frame_from_csv("n\n1\nabc"), "n") == ErrorCode.INVALID_NUMERIC_VALUES
    assert code_of(calculate_maximum, frame_from_csv("a\n1"), "nope") == ErrorCode.MISSING_COLUMN