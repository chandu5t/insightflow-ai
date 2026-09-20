"""Tests for the missing-value tool."""

import json

import pandas as pd
import pytest

from app.tools.missing_value_tool import analyze_missing_values, count_duplicate_rows, percentage
from tests.frames import frame_from_csv

CSV = """
    a,b,c
    1,x,
    2,,
    3,z,
    4,,
"""


def test_missing_counts_and_percentages_per_column() -> None:
    report = analyze_missing_values(frame_from_csv(CSV))
    by_name = {column.name: column for column in report.columns}

    assert (by_name["a"].missing_count, by_name["a"].missing_percentage) == (0, 0.0)
    assert (by_name["b"].missing_count, by_name["b"].missing_percentage) == (2, 50.0)
    assert (by_name["c"].missing_count, by_name["c"].missing_percentage) == (4, 100.0)


def test_totals_and_lists() -> None:
    report = analyze_missing_values(frame_from_csv(CSV))

    assert report.total_rows == 4 and report.total_columns == 3 and report.total_cells == 12
    assert report.total_missing_cells == 6
    assert report.missing_percentage == 50.0
    assert report.columns_with_missing == ["b", "c"]
    assert report.fully_empty_columns == ["c"]


def test_rows_with_missing_values_are_counted_and_sampled_by_number_only() -> None:
    report = analyze_missing_values(frame_from_csv(CSV))

    assert report.rows_with_missing_count == 4
    assert report.rows_with_missing_sample == [1, 2, 3, 4]


def test_row_sample_is_limited() -> None:
    frame = pd.DataFrame({"a": [None] * 30})

    report = analyze_missing_values(frame, max_row_numbers=3)

    assert report.rows_with_missing_count == 30
    assert report.rows_with_missing_sample == [1, 2, 3]


def test_dataset_without_missing_values() -> None:
    report = analyze_missing_values(frame_from_csv("a,b\n1,2\n3,4"))

    assert report.total_missing_cells == 0
    assert report.columns_with_missing == [] and report.fully_empty_columns == []
    assert report.rows_with_missing_count == 0 and report.rows_with_missing_sample == []
    assert report.missing_percentage == 0.0


def test_duplicate_rows_are_counted_after_the_first_copy() -> None:
    frame = frame_from_csv("a,b\n1,x\n1,x\n1,x\n2,y")

    assert count_duplicate_rows(frame) == 2
    assert analyze_missing_values(frame).duplicate_row_count == 2


def test_rows_that_differ_only_by_a_missing_value_are_not_duplicates() -> None:
    assert count_duplicate_rows(frame_from_csv("a,b\n1,x\n1,")) == 0


def test_empty_frame_gives_zero_percentages_not_errors() -> None:
    report = analyze_missing_values(pd.DataFrame({"a": [], "b": []}))

    assert report.total_cells == 0 and report.missing_percentage == 0.0
    assert report.fully_empty_columns == []
    assert report.duplicate_row_count == 0


def test_report_contains_no_cell_values_and_is_json_safe() -> None:
    text = json.dumps(analyze_missing_values(frame_from_csv(CSV)).model_dump(), allow_nan=False)

    assert "\"z\"" not in text and "\"x\"" not in text


@pytest.mark.parametrize(("part", "whole", "expected"), [(1, 3, 33.33), (0, 5, 0.0), (5, 5, 100.0), (1, 0, 0.0)])
def test_percentage_helper(part: int, whole: int, expected: float) -> None:
    assert percentage(part, whole) == expected