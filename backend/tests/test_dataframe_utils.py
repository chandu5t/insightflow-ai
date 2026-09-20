"""Tests for loading data and reading numbers safely."""

import math
from pathlib import Path

import pandas as pd
import pytest

from app.core.errors import AppError, ErrorCode, ToolError
from app.utils.dataframe_utils import (
    clean_text_frame,
    load_dataframe,
    parse_numeric,
    safe_float,
    to_numeric_strict,
)
from tests.frames import SALES_PATH, frame_from_csv


def test_clean_text_frame_trims_spaces_and_marks_empty_cells_missing() -> None:
    frame = clean_text_frame(pd.DataFrame({"a": ["  x ", "", "   "], "b": ["1", "2", "3"]}, dtype=str))

    assert frame["a"].tolist()[0] == "x"
    assert frame["a"].isna().tolist() == [False, True, True]


def test_words_like_na_and_null_are_not_treated_as_missing() -> None:
    frame = frame_from_csv("a\nN/A\nnull\nNA\n")

    assert frame["a"].isna().sum() == 0


def test_load_dataframe_reads_the_stored_csv_as_text() -> None:
    frame = load_dataframe(SALES_PATH)

    assert frame.shape == (13, 7)
    assert frame["quantity"].iloc[0] == "2"  # still text
    assert frame["region"].isna().sum() == 1


def test_load_dataframe_keeps_leading_zeros(tmp_path: Path) -> None:
    path = tmp_path / "d.csv"
    path.write_text("code\n007\n", encoding="utf-8")

    assert load_dataframe(path)["code"].iloc[0] == "007"


def test_load_dataframe_ignores_a_utf8_bom(tmp_path: Path) -> None:
    path = tmp_path / "d.csv"
    path.write_bytes(b"\xef\xbb\xbforder_id,qty\n1,2\n")

    assert list(load_dataframe(path).columns) == ["order_id", "qty"]


@pytest.mark.parametrize("content", [None, ""])
def test_unreadable_dataset_raises_dataset_unreadable(tmp_path: Path, content: str | None) -> None:
    path = tmp_path / "d.csv"
    if content is not None:
        path.write_text(content, encoding="utf-8")

    with pytest.raises(AppError) as info:
        load_dataframe(path)

    assert info.value.code == ErrorCode.DATASET_UNREADABLE
    assert info.value.status_code == 500
    assert str(tmp_path) not in info.value.message  # no path in the message


def test_parse_numeric_separates_numbers_blanks_and_invalid_values() -> None:
    series = pd.Series(["1", "2.5", None, "abc", "inf", "nan", "1,200", "-3", "1e3", "$5"], dtype=str)

    numbers, invalid = parse_numeric(series)

    assert numbers.dropna().tolist() == [1.0, 2.5, -3.0, 1000.0]
    assert invalid.tolist() == [False, False, False, True, True, True, True, False, False, True]


def test_to_numeric_strict_accepts_valid_numbers_and_blanks() -> None:
    result = to_numeric_strict(pd.Series(["1", None, "3"], dtype=str), "qty")

    assert result.tolist()[0] == 1.0 and math.isnan(result.tolist()[1])


def test_to_numeric_strict_rejects_invalid_values_with_examples() -> None:
    values = pd.Series(["1"] + [f"bad{i}" for i in range(8)] + ["x" * 100], dtype=str)

    with pytest.raises(ToolError) as info:
        to_numeric_strict(values, "quantity")

    assert info.value.code == ErrorCode.INVALID_NUMERIC_VALUES
    assert info.value.details["column"] == "quantity"
    assert info.value.details["invalid_count"] == 9
    assert len(info.value.details["examples"]) == 5
    assert all(len(example) <= 30 for example in info.value.details["examples"])


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None), (float("nan"), None), (float("inf"), None), (-math.inf, None), ("abc", None),
     (1.23456789, 1.234568), (5, 5.0), ("2.5", 2.5)],
)
def test_safe_float(value: object, expected: float | None) -> None:
    assert safe_float(value) == expected