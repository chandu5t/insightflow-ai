"""Tests for the profiling tool."""

import json

import pandas as pd
import pytest

from app.tools.profiling_tool import infer_data_type, profile_dataframe
from app.utils.dataframe_utils import load_dataframe
from tests.frames import SALES_PATH, frame_from_csv


def text(values: list[str]) -> pd.Series:
    return pd.Series(values, dtype="str")


def column(profile, name: str):
    return next(item for item in profile.columns if item.name == name)


# ---- type inference ------------------------------------------------------------------
@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["1", "2", "-5", "+3"], "integer"),
        (["1", "2.5"], "float"),
        (["12.0"], "float"),
        (["1e3", "2"], "float"),
        (["true", "False", "TRUE"], "boolean"),
        (["2025-01-05", "2025-02-01"], "datetime"),
        (["2025-01-05 10:30:00", "2025-01-06T08:00"], "datetime"),
        (["a", "b"], "text"),
        (["007", "008"], "text"),  # leading zeros: an id, not a number
        (["1,200", "2,500"], "text"),  # thousand separators are not guessed
        (["$50"], "text"),
        (["05/01/2025"], "text"),  # day/month order is not guessed
        (["2025-13-45"], "text"),  # looks like a date, is not one
        (["1", "abc"], "text"),
        (["inf"], "text"),
        (["nan"], "text"),
        (["yes", "no"], "text"),
        ([], "empty"),
    ],
)
def test_infer_data_type(values: list[str], expected: str) -> None:
    assert infer_data_type(text(values)) == expected


# ---- shape, names, missing values, duplicates ----------------------------------------------
def test_counts_and_names() -> None:
    profile = profile_dataframe(load_dataframe(SALES_PATH))

    assert profile.row_count == 13 and profile.column_count == 7
    assert profile.column_names == ["order_id", "order_date", "product", "category", "quantity", "unit_price", "region"]
    assert [c.name for c in profile.columns] == profile.column_names


def test_types_of_the_evaluation_dataset() -> None:
    profile = profile_dataframe(load_dataframe(SALES_PATH))

    types = {c.name: c.data_type for c in profile.columns}
    assert types == {
        "order_id": "integer", "order_date": "datetime", "product": "text", "category": "text",
        "quantity": "integer", "unit_price": "integer", "region": "text",
    }


def test_missing_values_and_duplicates() -> None:
    profile = profile_dataframe(load_dataframe(SALES_PATH))

    assert column(profile, "region").missing_count == 1
    assert column(profile, "region").missing_percentage == 7.69
    assert column(profile, "quantity").missing_count == 0
    assert profile.total_missing_cells == 1
    assert profile.duplicate_row_count == 1
    assert any("duplicate" in warning for warning in profile.warnings)


# ---- numeric statistics ---------------------------------------------------------------------------
def test_numeric_statistics() -> None:
    stats = column(profile_dataframe(load_dataframe(SALES_PATH)), "quantity").numeric_stats

    assert stats is not None
    assert stats.count == 13
    assert stats.mean == pytest.approx(2.307692, abs=1e-6)
    assert stats.std == pytest.approx(1.250641, abs=1e-6)
    assert (stats.min, stats.p25, stats.median, stats.p75, stats.max) == (1.0, 1.0, 2.0, 3.0, 5.0)


def test_numeric_statistics_ignore_missing_values() -> None:
    profile = profile_dataframe(frame_from_csv("n,x\n1,a\n,b\n3,c"))

    assert column(profile, "n").missing_count == 1
    assert column(profile, "n").numeric_stats.count == 2 and column(profile, "n").numeric_stats.mean == 2.0


def test_single_value_column_has_no_standard_deviation() -> None:
    stats = column(profile_dataframe(frame_from_csv("n\n5")), "n").numeric_stats

    assert stats.count == 1 and stats.std is None and stats.median == 5.0


def test_non_numeric_columns_have_no_numeric_statistics() -> None:
    profile = profile_dataframe(load_dataframe(SALES_PATH))

    assert column(profile, "product").numeric_stats is None
    assert column(profile, "order_date").numeric_stats is None


# ---- categorical, boolean and datetime information ------------------------------------------------
def test_categorical_information_is_ordered_by_count_then_name() -> None:
    info = column(profile_dataframe(load_dataframe(SALES_PATH)), "product").categorical

    assert info.unique_count == 5 and info.values_truncated is False
    assert [(v.value, v.count) for v in info.top_values] == [
        ("Headset", 3), ("Laptop", 3), ("Mouse", 3), ("Keyboard", 2), ("Monitor", 2)
    ]


def test_many_unique_values_are_limited_and_marked() -> None:
    frame = pd.DataFrame({"code": [f"C{i:03d}" for i in range(100)]})

    info = column(profile_dataframe(frame, top_values=10), "code").categorical

    assert info.unique_count == 100
    assert len(info.top_values) == 10
    assert info.values_truncated is True


def test_long_values_are_shortened() -> None:
    info = column(profile_dataframe(frame_from_csv("note\n" + "x" * 200), value_max_length=20), "note").categorical

    assert len(info.top_values[0].value) == 20 and info.top_values[0].value.endswith("…")


def test_boolean_column_information() -> None:
    profile = profile_dataframe(frame_from_csv("paid\nTrue\nFalse\nTrue"))
    paid = column(profile, "paid")

    assert paid.data_type == "boolean"
    assert [(v.value, v.count) for v in paid.categorical.top_values] == [("True", 2), ("False", 1)]


def test_datetime_column_information() -> None:
    profile = profile_dataframe(load_dataframe(SALES_PATH))
    dates = column(profile, "order_date")

    assert dates.datetime_stats.min == "2025-01-05" and dates.datetime_stats.max == "2025-01-14"
    assert dates.categorical is None


def test_datetime_with_time_keeps_the_time() -> None:
    dates = column(profile_dataframe(frame_from_csv("t\n2025-01-05 10:30:00\n2025-01-06 08:00:00")), "t")

    assert dates.datetime_stats.min == "2025-01-05 10:30:00"


# ---- empty columns, empty datasets, warnings -------------------------------------------------------
def test_column_with_only_missing_values() -> None:
    profile = profile_dataframe(frame_from_csv("a,b\n1,\n2,"))
    empty = column(profile, "b")

    assert empty.data_type == "empty" and empty.missing_percentage == 100.0 and empty.unique_count == 0
    assert empty.numeric_stats is None and empty.categorical is None
    assert "Column 'b' is completely empty." in profile.warnings


def test_dataset_with_no_rows() -> None:
    profile = profile_dataframe(pd.DataFrame({"a": pd.Series([], dtype="str"), "b": pd.Series([], dtype="str")}))

    assert profile.row_count == 0 and profile.column_count == 2
    assert all(c.data_type == "empty" and c.missing_percentage == 0.0 for c in profile.columns)
    assert profile.warnings == []


def test_mostly_numeric_text_column_gets_a_warning() -> None:
    profile = profile_dataframe(frame_from_csv("qty\n1\n2\n3\nabc"))

    assert column(profile, "qty").data_type == "text"
    assert any("Column 'qty' has 1 value(s) that are not valid numbers" in w for w in profile.warnings)


def test_ambiguous_column_mapping_is_reported() -> None:
    profile = profile_dataframe(frame_from_csv("revenue,sales,product\n1,2,a"))

    assert profile.column_mapping.ambiguous == {"revenue": ["revenue", "sales"]}
    assert profile.column_mapping.resolved == {"product": "product"}
    assert any("More than one column could be 'revenue'" in w for w in profile.warnings)


def test_column_mapping_of_the_evaluation_dataset() -> None:
    mapping = profile_dataframe(load_dataframe(SALES_PATH)).column_mapping

    assert mapping.resolved["quantity"] == "quantity" and mapping.missing == ["revenue"]


# ---- JSON safety and limits -----------------------------------------------------------------------------
def test_profile_is_json_safe_with_nan_and_infinity_text() -> None:
    frame = frame_from_csv("a,b,c\ninf,nan,1e308\n-inf,1,1e308")

    profile = profile_dataframe(frame)

    json.dumps(profile.model_dump(), allow_nan=False)  # must not raise


def test_extreme_numbers_stay_json_safe() -> None:
    stats = column(profile_dataframe(frame_from_csv("n\n1e308\n1e308")), "n").numeric_stats

    json.dumps(stats.model_dump(), allow_nan=False)
    assert stats.min == 1e308


def test_profile_contains_no_full_rows() -> None:
    dumped = profile_dataframe(load_dataframe(SALES_PATH)).model_dump()

    assert "rows" not in dumped and "data" not in dumped