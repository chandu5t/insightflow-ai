"""Tests for the grouping tool."""

import pytest

from app.core.errors import ErrorCode, ToolError
from app.tools.grouping_tool import group_and_aggregate, quantity_by_group, revenue_by_group
from tests.frames import frame_from_csv

SALES = """
    product,region,qty,price
    Pen,North,2,10
    Pen,South,3,10
    Book,North,1,100
    Book,North,1,100
    Cup,,4,5
"""


def as_dict(result) -> dict[str | None, float | int | None]:
    return {row.group: row.value for row in result.groups}


def code_of(callable_, *args, **kwargs) -> ErrorCode:
    with pytest.raises(ToolError) as info:
        callable_(*args, **kwargs)
    return info.value.code


def test_revenue_by_product() -> None:
    result = revenue_by_group(frame_from_csv(SALES), "product")

    assert as_dict(result) == {"Book": 200.0, "Cup": 20.0, "Pen": 50.0}
    assert result.group_by == "product" and result.metric == "sum"
    assert result.calculation_method == "derived_quantity_times_unit_price"
    assert result.value_columns_used == ["qty", "price"]


def test_revenue_by_region_leaves_out_empty_groups_and_reports_them() -> None:
    result = revenue_by_group(frame_from_csv(SALES), "region")

    assert as_dict(result) == {"North": 220.0, "South": 30.0}
    assert result.rows_missing_group == 1 and result.rows_used == 4
    assert any("1 row(s) were left out because 'region' was empty" in note for note in result.notes)


def test_missing_group_can_be_included_as_none() -> None:
    result = revenue_by_group(frame_from_csv(SALES), "region", include_missing_group=True)

    assert as_dict(result) == {"North": 220.0, "South": 30.0, None: 20.0}
    assert result.groups[-1].group is None  # always last


def test_quantity_by_product() -> None:
    assert as_dict(quantity_by_group(frame_from_csv(SALES), "product")) == {"Book": 2.0, "Cup": 4.0, "Pen": 5.0}


def test_sum_average_and_count() -> None:
    frame = frame_from_csv("g,v\na,10\na,20\nb,5")

    assert as_dict(group_and_aggregate(frame, group_by="g", metric="sum", value="v")) == {"a": 30.0, "b": 5.0}
    assert as_dict(group_and_aggregate(frame, group_by="g", metric="average", value="v")) == {"a": 15.0, "b": 5.0}
    assert as_dict(group_and_aggregate(frame, group_by="g", metric="count", value="v")) == {"a": 2, "b": 1}


def test_count_without_a_value_counts_rows() -> None:
    result = group_and_aggregate(frame_from_csv(SALES), group_by="product", metric="count", value=None)

    assert as_dict(result) == {"Book": 2, "Cup": 1, "Pen": 2} and result.calculation_method == "row_count"
    assert all(isinstance(row.value, int) for row in result.groups)


def test_count_with_a_value_counts_only_non_empty_values() -> None:
    frame = frame_from_csv("g,v\na,1\na,\nb,")

    assert as_dict(group_and_aggregate(frame, group_by="g", metric="count", value="v")) == {"a": 1, "b": 0}


def test_group_with_no_valid_value_gets_null_not_zero_for_sum_and_average() -> None:
    frame = frame_from_csv("g,v\na,1\nb,")

    for metric in ("sum", "average"):
        result = group_and_aggregate(frame, group_by="g", metric=metric, value="v")
        assert as_dict(result) == {"a": 1.0, "b": None}
        assert result.rows_missing_value == 1 and any("no valid values" in note for note in result.notes)


def test_row_counts_include_rows_with_an_empty_value() -> None:
    frame = frame_from_csv("g,v\na,1\na,")

    assert group_and_aggregate(frame, group_by="g", value="v").groups[0].row_count == 2


def test_group_by_an_exact_column_name_and_a_role_name() -> None:
    frame = frame_from_csv("Product Name,category,qty\nx,c1,1\ny,c1,2")

    assert as_dict(group_and_aggregate(frame, group_by="category", value="qty")) == {"c1": 3.0}
    assert as_dict(group_and_aggregate(frame, group_by="product", value="qty")) == {"x": 1.0, "y": 2.0}


def test_groups_are_case_sensitive_and_never_merged_silently() -> None:
    result = group_and_aggregate(frame_from_csv("g,v\nlaptop,1\nLaptop,2"), group_by="g", value="v")

    assert as_dict(result) == {"Laptop": 2.0, "laptop": 1.0}


def test_groups_are_sorted_alphabetically_ignoring_case() -> None:
    result = group_and_aggregate(frame_from_csv("g,v\nb,1\nA,1\nC,1"), group_by="g", value="v")

    assert [row.group for row in result.groups] == ["A", "b", "C"]


def test_results_do_not_depend_on_row_order() -> None:
    frame = frame_from_csv(SALES)

    assert revenue_by_group(frame, "product") == revenue_by_group(frame.iloc[::-1], "product")


def test_missing_group_column_is_a_clear_error() -> None:
    assert code_of(group_and_aggregate, frame_from_csv("a,v\n1,2"), group_by="zzz", value="v") == ErrorCode.MISSING_COLUMN
    assert code_of(revenue_by_group, frame_from_csv("qty,price\n1,2"), "product") == ErrorCode.MISSING_COLUMN


def test_missing_value_column_is_a_clear_error() -> None:
    assert code_of(group_and_aggregate, frame_from_csv("g\na"), group_by="g", value="zzz") == ErrorCode.MISSING_COLUMN


def test_revenue_needs_revenue_columns() -> None:
    assert code_of(revenue_by_group, frame_from_csv("product\na"), "product") == ErrorCode.INSUFFICIENT_DATA


def test_invalid_numbers_are_rejected() -> None:
    assert code_of(group_and_aggregate, frame_from_csv("g,v\na,x"), group_by="g", value="v") == ErrorCode.INVALID_NUMERIC_VALUES


def test_invalid_metric_and_missing_value_for_sum() -> None:
    frame = frame_from_csv("g,v\na,1")

    assert code_of(group_and_aggregate, frame, group_by="g", metric="median", value="v") == ErrorCode.INVALID_PARAMETER
    assert code_of(group_and_aggregate, frame, group_by="g", metric="sum", value=None) == ErrorCode.INVALID_PARAMETER


def test_ambiguous_group_role_is_rejected_and_an_override_solves_it() -> None:
    frame = frame_from_csv("region,area,v\nN,X,1")

    assert code_of(group_and_aggregate, frame, group_by="region", value="v") == ErrorCode.AMBIGUOUS_COLUMN
    result = group_and_aggregate(frame, group_by="region", value="v", column_overrides={"region": "area"})
    assert result.group_by == "area" and as_dict(result) == {"X": 1.0}


def test_grouping_an_empty_dataset_gives_no_groups() -> None:
    result = group_and_aggregate(frame_from_csv("g,v\na,1").iloc[0:0], group_by="g", value="v")

    assert result.groups == [] and result.group_count == 0