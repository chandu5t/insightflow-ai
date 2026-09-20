"""Tests for the explicit column mapper."""

import pytest

from app.core.errors import ErrorCode, ToolError
from app.utils.column_mapper import (
    CANONICAL_ALIASES,
    map_columns,
    normalize_column_name,
    resolve_column,
    validate_aliases,
)


def error_of(callable_, *args, **kwargs) -> ToolError:
    with pytest.raises(ToolError) as info:
        callable_(*args, **kwargs)
    return info.value


# ---- the configured names (regression: these are the agreed names) -----------------
def test_agreed_alias_lists() -> None:
    assert CANONICAL_ALIASES["revenue"] == ("revenue", "total_revenue", "sales", "total_sales")
    assert CANONICAL_ALIASES["quantity"] == ("quantity", "qty", "units", "units_sold")
    assert CANONICAL_ALIASES["unit_price"] == ("unit_price", "price", "selling_price")
    assert CANONICAL_ALIASES["product"] == ("product", "product_name", "item", "item_name")
    assert CANONICAL_ALIASES["order_id"] == ("order_id", "orderid", "order_number", "order_no")
    assert CANONICAL_ALIASES["region"] == ("region", "area", "territory")


def test_default_alias_configuration_is_valid() -> None:
    validate_aliases(CANONICAL_ALIASES)  # must not raise


# ---- standard and equivalent names ---------------------------------------------------
def test_standard_names_are_mapped_to_themselves() -> None:
    columns = ["order_id", "product", "region", "quantity", "unit_price", "revenue"]

    mapping = map_columns(columns)

    assert mapping.resolved == {role: role for role in columns}
    assert mapping.missing == [] and mapping.ambiguous == {}


@pytest.mark.parametrize(
    ("column", "role"),
    [
        ("total_revenue", "revenue"), ("Sales", "revenue"), ("TOTAL_SALES", "revenue"),
        ("qty", "quantity"), ("units", "quantity"), ("Units Sold", "quantity"),
        ("price", "unit_price"), ("Selling Price", "unit_price"),
        ("item", "product"), ("Product Name", "product"), ("Item-Name", "product"),
        ("orderid", "order_id"), ("OrderID", "order_id"), ("Order No", "order_id"), ("order_number", "order_id"),
        ("Area", "region"), ("TERRITORY", "region"),
    ],
)
def test_equivalent_names_map_to_the_role(column: str, role: str) -> None:
    assert map_columns([column]).resolved == {role: column}


# ---- case, whitespace and punctuation; original name is kept ----------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [("Unit Price", "unit_price"), ("  UNIT_PRICE  ", "unit_price"), ("unit-price", "unit_price"),
     ("Unit  __ Price", "unit_price"), ("Order ID", "order_id"), ("OrderID", "orderid"), ("", "")],
)
def test_normalize_column_name(raw: str, expected: str) -> None:
    assert normalize_column_name(raw) == expected


def test_original_column_name_is_preserved() -> None:
    mapping = map_columns(["  Unit Price ", "Total Revenue"])

    assert mapping.get("unit_price") == "  Unit Price "
    assert mapping.get("revenue") == "Total Revenue"


# ---- names that are NOT configured are never guessed ------------------------------------
def test_unknown_columns_are_not_mapped() -> None:
    mapping = map_columns(["profit", "discount", "prices", "revenue_total"])

    assert mapping.resolved == {}
    assert sorted(mapping.missing) == sorted(CANONICAL_ALIASES)


def test_absent_roles_are_listed_as_missing() -> None:
    mapping = map_columns(["product", "qty"])

    assert set(mapping.missing) == {"revenue", "unit_price", "order_id", "region"}


# ---- ambiguity ----------------------------------------------------------------------------------
def test_two_columns_for_one_role_are_ambiguous_not_guessed() -> None:
    mapping = map_columns(["revenue", "sales", "qty"])

    assert mapping.ambiguous == {"revenue": ["revenue", "sales"]}
    assert mapping.get("revenue") is None
    assert "revenue" not in mapping.missing
    assert mapping.get("quantity") == "qty"


def test_names_that_look_the_same_after_normalising_are_ambiguous() -> None:
    assert map_columns(["Unit Price", "unit_price"]).ambiguous == {"unit_price": ["Unit Price", "unit_price"]}


def test_ambiguous_candidates_keep_the_dataset_column_order() -> None:
    assert map_columns(["total_sales", "x", "revenue"]).ambiguous["revenue"] == ["total_sales", "revenue"]


# ---- require() ----------------------------------------------------------------------------------------
def test_require_returns_columns_for_the_roles() -> None:
    mapping = map_columns(["Qty", "Price"])

    assert mapping.require("quantity", "unit_price") == {"quantity": "Qty", "unit_price": "Price"}


def test_require_reports_all_missing_roles_with_accepted_names() -> None:
    error = error_of(map_columns(["product"]).require, "quantity", "unit_price", "product")

    assert error.code == ErrorCode.MISSING_COLUMN
    assert error.status_code == 422
    assert error.details["missing_roles"] == ["quantity", "unit_price"]
    assert error.details["accepted_names"]["quantity"] == ["quantity", "qty", "units", "units_sold"]


def test_require_raises_for_ambiguous_role() -> None:
    error = error_of(map_columns(["sales", "revenue"]).require, "revenue")

    assert error.code == ErrorCode.AMBIGUOUS_COLUMN
    assert error.details == {"role": "revenue", "candidates": ["sales", "revenue"]}


def test_require_unknown_role_is_invalid_parameter() -> None:
    assert error_of(map_columns(["a"]).require, "profit").code == ErrorCode.INVALID_PARAMETER


# ---- overrides ----------------------------------------------------------------------------------------
def test_override_solves_an_ambiguous_role() -> None:
    mapping = map_columns(["revenue", "sales"], overrides={"revenue": "sales"})

    assert mapping.get("revenue") == "sales"
    assert mapping.ambiguous == {}


def test_override_of_a_missing_column_is_an_error() -> None:
    error = error_of(map_columns, ["a", "b"], overrides={"revenue": "nope"})

    assert error.code == ErrorCode.MISSING_COLUMN
    assert error.details["available_columns"] == ["a", "b"]


def test_override_of_an_unknown_role_is_an_error() -> None:
    assert error_of(map_columns, ["a"], overrides={"profit": "a"}).code == ErrorCode.INVALID_PARAMETER


# ---- extending and validating the configuration ------------------------------------------------
def test_new_roles_can_be_added_through_configuration() -> None:
    mapping = map_columns(["Cost"], aliases={"cost_price": ("cost", "buying_price")})

    assert mapping.resolved == {"cost_price": "Cost"}


def test_alias_used_for_two_roles_is_rejected() -> None:
    with pytest.raises(ValueError, match="used for both"):
        validate_aliases({"revenue": ("sales",), "quantity": ("sales",)})


def test_empty_alias_is_rejected() -> None:
    with pytest.raises(ValueError, match="Empty alias"):
        validate_aliases({"revenue": ("  ",)})


# ---- resolve_column -------------------------------------------------------------------------------------
def test_resolve_column_uses_the_mapper_for_role_names() -> None:
    columns = ["Product Name", "Qty"]

    assert resolve_column(columns, "product", map_columns(columns)) == "Product Name"


def test_resolve_column_accepts_an_exact_column_name() -> None:
    columns = ["category", "qty"]

    assert resolve_column(columns, "category", map_columns(columns)) == "category"


def test_resolve_column_unknown_name_is_missing_column() -> None:
    columns = ["a"]

    assert error_of(resolve_column, columns, "zzz", map_columns(columns)).code == ErrorCode.MISSING_COLUMN


def test_resolve_column_role_that_is_ambiguous_is_an_error() -> None:
    columns = ["region", "area"]

    assert error_of(resolve_column, columns, "region", map_columns(columns)).code == ErrorCode.AMBIGUOUS_COLUMN


def test_resolve_column_role_that_is_absent_is_missing_column() -> None:
    columns = ["a"]

    assert error_of(resolve_column, columns, "region", map_columns(columns)).code == ErrorCode.MISSING_COLUMN