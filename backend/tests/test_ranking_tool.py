"""Tests for the ranking tool."""

import random

import pytest

from app.core.errors import ErrorCode, ToolError
from app.schemas.tool_schema import GroupingResult, GroupRow
from app.tools.grouping_tool import group_and_aggregate
from app.tools.ranking_tool import rank_by_group, rank_groups
from tests.frames import frame_from_csv

SALES = """
    product,qty,price
    Pen,1,10
    Pen,1,10
    Book,1,100
    Cup,2,5
    Bag,3,10
    Ink,1,10
"""  # revenue: Book 100, Bag 30, Pen 20, Cup 10, Ink 10 (tie)


def names(result) -> list[str | None]:
    return [item.group for item in result.items]


def code_of(callable_, *args, **kwargs) -> ErrorCode:
    with pytest.raises(ToolError) as info:
        callable_(*args, **kwargs)
    return info.value.code


def make_grouping(values: dict[str | None, float | None]) -> GroupingResult:
    return GroupingResult(
        group_by="g", metric="sum", value_columns_used=["v"], calculation_method="numeric_column",
        groups=[GroupRow(group=name, value=value, row_count=1) for name, value in values.items()],
        group_count=len(values), rows_used=len(values), rows_missing_group=0, rows_missing_value=0,
    )


def test_top_n_descending() -> None:
    result = rank_by_group(frame_from_csv(SALES), group_by="product", n=3)

    assert names(result) == ["Book", "Bag", "Pen"]
    assert [item.value for item in result.items] == [100.0, 30.0, 20.0]
    assert [item.rank for item in result.items] == [1, 2, 3]
    assert result.order == "desc" and result.n_requested == 3 and result.n_returned == 3


def test_bottom_n_ascending() -> None:
    result = rank_by_group(frame_from_csv(SALES), group_by="product", n=2, order="asc")

    assert names(result) == ["Cup", "Ink"] and result.order == "asc"


def test_ascending_and_descending_are_exact_opposites_when_there_are_no_ties() -> None:
    grouping = make_grouping({"a": 3, "b": 1, "c": 2})

    assert names(rank_groups(grouping, n=3, order="desc")) == ["a", "c", "b"]
    assert names(rank_groups(grouping, n=3, order="asc")) == ["b", "c", "a"]


def test_n_larger_than_the_number_of_groups_returns_all_groups() -> None:
    result = rank_by_group(frame_from_csv(SALES), group_by="product", n=50)

    assert result.n_requested == 50 and result.n_returned == 5 and result.total_groups_ranked == 5


# ---- ties -------------------------------------------------------------------------------------
def test_ties_share_a_rank_and_are_ordered_by_name() -> None:
    result = rank_by_group(frame_from_csv(SALES), group_by="product", n=5)

    assert names(result) == ["Book", "Bag", "Pen", "Cup", "Ink"]
    assert [item.rank for item in result.items] == [1, 2, 3, 4, 4]  # competition ranking


def test_tie_order_is_ignoring_letter_case() -> None:
    result = rank_groups(make_grouping({"b": 1, "C": 1, "a": 1}), n=3)

    # a plain (case-sensitive) sort would give C, a, b
    assert names(result) == ["a", "b", "C"] and [item.rank for item in result.items] == [1, 1, 1]


def test_result_is_the_same_for_any_input_order() -> None:
    values = {"a": 5, "b": 5, "c": 3, "d": 3, "e": 9}
    expected = names(rank_groups(make_grouping(values), n=5))

    for seed in range(20):
        keys = list(values)
        random.Random(seed).shuffle(keys)
        assert names(rank_groups(make_grouping({k: values[k] for k in keys}), n=5)) == expected


def test_cut_off_inside_a_tie_is_reported() -> None:
    result = rank_by_group(frame_from_csv(SALES), group_by="product", n=4)

    assert names(result)[-1] == "Cup" and result.truncated_tie is True
    assert any("splits a tie" in note for note in result.notes)


def test_cut_off_between_different_values_is_not_a_tie() -> None:
    assert rank_by_group(frame_from_csv(SALES), group_by="product", n=3).truncated_tie is False
    assert rank_by_group(frame_from_csv(SALES), group_by="product", n=5).truncated_tie is False


# ---- groups with no value, missing group -------------------------------------------------------------
def test_groups_without_a_value_are_not_ranked() -> None:
    result = rank_groups(make_grouping({"a": 1, "b": None, "c": 2}), n=5)

    assert names(result) == ["c", "a"] and result.total_groups_ranked == 2


def test_missing_group_is_ranked_last_among_equal_values() -> None:
    frame = frame_from_csv("g,v\na,5\n,5")

    result = rank_by_group(frame, group_by="g", value="v", n=2, include_missing_group=True)

    assert names(result) == ["a", None]


# ---- other metrics ---------------------------------------------------------------------------------
def test_rank_by_count_and_average() -> None:
    frame = frame_from_csv("g,v\na,1\na,1\nb,10")

    assert names(rank_by_group(frame, group_by="g", metric="count", value=None, n=2)) == ["a", "b"]
    assert names(rank_by_group(frame, group_by="g", metric="average", value="v", n=2)) == ["b", "a"]


def test_ranking_grouped_results_directly() -> None:
    grouping = group_and_aggregate(frame_from_csv(SALES), group_by="product")

    result = rank_groups(grouping, n=1)

    assert names(result) == ["Book"] and result.calculation_method == "derived_quantity_times_unit_price"
    assert result.group_by == "product" and result.value_columns_used == ["qty", "price"]


# ---- validation ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("bad_n", [0, -1, 101, 2.5, "3", None, True])
def test_invalid_n_is_rejected(bad_n: object) -> None:
    grouping = make_grouping({"a": 1})

    assert code_of(rank_groups, grouping, n=bad_n) == ErrorCode.INVALID_PARAMETER
    assert code_of(rank_by_group, frame_from_csv("g,v\na,1"), group_by="g", value="v", n=bad_n) == ErrorCode.INVALID_PARAMETER


def test_invalid_order_is_rejected() -> None:
    assert code_of(rank_groups, make_grouping({"a": 1}), n=1, order="up") == ErrorCode.INVALID_PARAMETER


def test_missing_columns_are_clear_errors() -> None:
    assert code_of(rank_by_group, frame_from_csv("a,v\n1,2"), group_by="zzz", value="v") == ErrorCode.MISSING_COLUMN
    assert code_of(rank_by_group, frame_from_csv("product\na"), group_by="product") == ErrorCode.INSUFFICIENT_DATA