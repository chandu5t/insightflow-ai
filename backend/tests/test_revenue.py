"""Tests for the revenue rules."""

import math

import pytest

from app.core.errors import ErrorCode, ToolError
from app.tools.revenue import compute_revenue, resolve_numeric_values
from app.utils.column_mapper import map_columns
from tests.frames import frame_from_csv


def revenue_of(csv_text: str, overrides: dict[str, str] | None = None):
    frame = frame_from_csv(csv_text)
    return compute_revenue(frame, map_columns(list(frame.columns), overrides=overrides))


def error_of(csv_text: str, overrides: dict[str, str] | None = None) -> ToolError:
    with pytest.raises(ToolError) as info:
        revenue_of(csv_text, overrides)
    return info.value


# ---- Rule 1: direct revenue -----------------------------------------------------------------
def test_direct_revenue_column_is_used() -> None:
    result = revenue_of("revenue,product\n100,a\n250.5,b")

    assert result.method == "direct_revenue_column"
    assert result.columns_used == ["revenue"]
    assert result.values.tolist() == [100.0, 250.5]


@pytest.mark.parametrize("name", ["total_revenue", "Sales", "TOTAL SALES"])
def test_direct_revenue_column_with_other_accepted_names(name: str) -> None:
    result = revenue_of(f"{name}\n10\n20")

    assert result.method == "direct_revenue_column" and result.columns_used == [name]


def test_direct_revenue_wins_even_when_quantity_and_price_exist() -> None:
    result = revenue_of("revenue,qty,price\n999,2,5")

    assert result.method == "direct_revenue_column" and result.values.tolist() == [999.0]


def test_negative_revenue_is_kept_as_it_is() -> None:
    assert revenue_of("revenue\n100\n-40").values.sum() == 60.0


# ---- Rule 2: derived revenue ----------------------------------------------------------------
def test_revenue_is_derived_from_quantity_times_unit_price() -> None:
    result = revenue_of("quantity,unit_price\n2,50\n3,10.5")

    assert result.method == "derived_quantity_times_unit_price"
    assert result.columns_used == ["quantity", "unit_price"]
    assert result.values.tolist() == [100.0, 31.5]


def test_derived_revenue_with_alternative_names() -> None:
    result = revenue_of("Units Sold,Selling Price\n4,25")

    assert result.columns_used == ["Units Sold", "Selling Price"] and result.values.tolist() == [100.0]


def test_derived_revenue_is_empty_for_rows_with_an_empty_input() -> None:
    values = revenue_of("qty,price\n2,50\n,50\n2,").values.tolist()

    assert values[0] == 100.0 and math.isnan(values[1]) and math.isnan(values[2])


# ---- Rule 3: not enough columns --------------------------------------------------------------
def test_no_revenue_and_no_quantity_is_insufficient_data() -> None:
    error = error_of("product,region\na,b")

    assert error.code == ErrorCode.INSUFFICIENT_DATA and error.status_code == 422
    assert error.details["missing_roles"] == ["revenue", "quantity", "unit_price"]


@pytest.mark.parametrize(("csv_text", "missing"), [
    ("quantity,product\n1,a", ["revenue", "unit_price"]),
    ("unit_price,product\n1,a", ["revenue", "quantity"]),
])
def test_only_one_of_quantity_and_unit_price_is_insufficient_data(csv_text: str, missing: list[str]) -> None:
    error = error_of(csv_text)

    assert error.code == ErrorCode.INSUFFICIENT_DATA and error.details["missing_roles"] == missing


# ---- Rule 5: invalid values and ambiguity ---------------------------------------------------------
@pytest.mark.parametrize("bad", ["abc", "1,200", "$50", "inf", "nan"])
def test_invalid_direct_revenue_values_are_rejected(bad: str) -> None:
    error = error_of(f'revenue\n10\n"{bad}"')

    assert error.code == ErrorCode.INVALID_NUMERIC_VALUES
    assert error.details["column"] == "revenue" and error.details["examples"] == [bad]


def test_invalid_quantity_is_rejected_when_deriving() -> None:
    error = error_of("qty,price\n2,5\ntwo,5")

    assert error.code == ErrorCode.INVALID_NUMERIC_VALUES and error.details["column"] == "qty"


def test_ambiguous_revenue_columns_are_rejected_even_if_quantity_and_price_exist() -> None:
    error = error_of("revenue,sales,qty,price\n1,2,3,4")

    assert error.code == ErrorCode.AMBIGUOUS_COLUMN
    assert error.details["candidates"] == ["revenue", "sales"]


def test_ambiguous_quantity_is_rejected_when_deriving() -> None:
    assert error_of("qty,units,price\n1,2,3").code == ErrorCode.AMBIGUOUS_COLUMN


def test_override_solves_ambiguous_revenue() -> None:
    result = revenue_of("revenue,sales\n1,2", overrides={"revenue": "sales"})

    assert result.columns_used == ["sales"] and result.values.tolist() == [2.0]


# ---- empty datasets --------------------------------------------------------------------------------------
def test_empty_dataset_with_revenue_column_gives_no_values() -> None:
    frame = frame_from_csv("revenue\n1").iloc[0:0]

    values = compute_revenue(frame, map_columns(list(frame.columns))).values

    assert len(values) == 0


def test_empty_dataset_without_revenue_columns_is_insufficient_data() -> None:
    frame = frame_from_csv("product\na").iloc[0:0]

    with pytest.raises(ToolError) as info:
        compute_revenue(frame, map_columns(list(frame.columns)))

    assert info.value.code == ErrorCode.INSUFFICIENT_DATA


# ---- resolve_numeric_values ------------------------------------------------------------------------------
def test_revenue_request_uses_the_revenue_rules() -> None:
    frame = frame_from_csv("qty,price\n2,5")

    result = resolve_numeric_values(frame, "revenue", map_columns(list(frame.columns)))

    assert result.method == "derived_quantity_times_unit_price"


def test_other_requests_use_one_numeric_column() -> None:
    frame = frame_from_csv("qty,discount\n2,0.5")
    mapping = map_columns(list(frame.columns))

    assert resolve_numeric_values(frame, "discount", mapping).columns_used == ["discount"]
    assert resolve_numeric_values(frame, "quantity", mapping).columns_used == ["qty"]


def test_numeric_request_for_a_missing_column() -> None:
    frame = frame_from_csv("qty\n2")

    with pytest.raises(ToolError) as info:
        resolve_numeric_values(frame, "profit", map_columns(list(frame.columns)))

    assert info.value.code == ErrorCode.MISSING_COLUMN