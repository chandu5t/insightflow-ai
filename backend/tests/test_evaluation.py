"""Evaluation tests: the tools must give the EXPECTED numbers for the fixed datasets.

All expected values were calculated by hand. See backend/data/evaluation/README.md.
"""

import pytest

from app.core.errors import ErrorCode, ToolError
from app.tools.aggregation_tool import (
    calculate_average,
    calculate_average_order_value,
    calculate_count,
    calculate_sum,
    calculate_total_revenue,
)
from app.tools.grouping_tool import group_and_aggregate, quantity_by_group, revenue_by_group
from app.tools.missing_value_tool import analyze_missing_values
from app.tools.ranking_tool import rank_by_group
from app.utils.column_mapper import map_columns
from app.utils.dataframe_utils import load_dataframe
from tests.frames import DIRECT_PATH, SALES_PATH


@pytest.fixture(scope="module")
def sales():
    """13 rows, revenue must be derived (quantity x unit_price)."""
    return load_dataframe(SALES_PATH)


@pytest.fixture(scope="module")
def direct():
    """6 rows, has a 'Total Revenue' column."""
    return load_dataframe(DIRECT_PATH)


def as_dict(result) -> dict:
    return {row.group: row.value for row in result.groups}


def test_the_evaluation_files_have_the_expected_shape(sales, direct) -> None:
    assert sales.shape == (13, 7) and direct.shape == (6, 5)
    assert list(direct.columns) == ["Order ID", "Product Name", "Region", "Quantity", "Total Revenue"]


# ---- derived revenue dataset ---------------------------------------------------------------------
def test_derived_total_revenue(sales) -> None:
    result = calculate_total_revenue(sales)

    assert result.value == 256000.0
    assert result.calculation_method == "derived_quantity_times_unit_price"
    assert result.columns_used == ["quantity", "unit_price"]


def test_distinct_order_count_versus_row_count(sales) -> None:
    result = calculate_average_order_value(sales)

    assert (result.order_count, calculate_count(sales).value) == (10, 13)
    assert result.value == 25600.0  # 256000 / 10 distinct orders
    assert result.value != pytest.approx(256000 / 13)  # NOT the row-count answer (19692.31)
    assert result.order_count_method == "distinct_order_id_count"


def test_basic_aggregations(sales) -> None:
    assert calculate_sum(sales, "quantity").value == 30.0
    assert calculate_average(sales, "quantity").value == pytest.approx(2.307692, abs=1e-6)
    assert calculate_average(sales, "unit_price").value == pytest.approx(13807.692308, abs=1e-6)
    assert calculate_count(sales, "region").value == 12


def test_revenue_by_product(sales) -> None:
    result = revenue_by_group(sales, "product")

    assert as_dict(result) == {"Headset": 16000.0, "Keyboard": 5000.0, "Laptop": 200000.0, "Monitor": 30000.0, "Mouse": 5000.0}
    assert sum(row.value for row in result.groups) == 256000.0


def test_revenue_by_region_with_the_missing_region_reported(sales) -> None:
    result = revenue_by_group(sales, "region")

    assert as_dict(result) == {"East": 72000.0, "North": 106500.0, "South": 75000.0}
    assert result.rows_missing_group == 1
    assert as_dict(revenue_by_group(sales, "region", include_missing_group=True))[None] == 2500.0


def test_quantity_by_product(sales) -> None:
    assert as_dict(quantity_by_group(sales, "product")) == {
        "Headset": 8.0, "Keyboard": 5.0, "Laptop": 4.0, "Monitor": 3.0, "Mouse": 10.0
    }


def test_row_counts_per_product(sales) -> None:
    result = group_and_aggregate(sales, group_by="product", metric="count", value=None)

    assert as_dict(result) == {"Headset": 3, "Keyboard": 2, "Laptop": 3, "Monitor": 2, "Mouse": 3}


def test_top_products_by_revenue_with_a_tie(sales) -> None:
    result = rank_by_group(sales, group_by="product", n=5)

    assert [(i.rank, i.group, i.value) for i in result.items] == [
        (1, "Laptop", 200000.0), (2, "Monitor", 30000.0), (3, "Headset", 16000.0),
        (4, "Keyboard", 5000.0), (4, "Mouse", 5000.0),
    ]
    assert rank_by_group(sales, group_by="product", n=4).truncated_tie is True


def test_bottom_two_products_by_revenue(sales) -> None:
    result = rank_by_group(sales, group_by="product", n=2, order="asc")

    assert [(i.rank, i.group) for i in result.items] == [(1, "Keyboard"), (1, "Mouse")]
    assert result.truncated_tie is False


def test_top_region_by_revenue(sales) -> None:
    assert rank_by_group(sales, group_by="region", n=1).items[0].group == "North"


def test_top_product_by_units_sold(sales) -> None:
    result = rank_by_group(sales, group_by="product", value="quantity", n=2)

    assert [(i.group, i.value) for i in result.items] == [("Mouse", 10.0), ("Headset", 8.0)]


def test_missing_values_and_duplicates(sales) -> None:
    report = analyze_missing_values(sales)

    assert (report.total_cells, report.total_missing_cells) == (91, 1)
    assert report.columns_with_missing == ["region"] and report.rows_with_missing_sample == [9]
    assert report.duplicate_row_count == 1


def test_column_mapping_of_the_derived_dataset(sales) -> None:
    mapping = map_columns(list(sales.columns))

    assert mapping.resolved == {
        "quantity": "quantity", "unit_price": "unit_price", "product": "product",
        "order_id": "order_id", "region": "region",
    }
    assert mapping.missing == ["revenue"] and mapping.ambiguous == {}


# ---- direct revenue dataset -----------------------------------------------------------------------------
def test_direct_total_revenue(direct) -> None:
    result = calculate_total_revenue(direct)

    assert result.value == 121800.0
    assert result.calculation_method == "direct_revenue_column" and result.columns_used == ["Total Revenue"]


def test_direct_average_order_value(direct) -> None:
    result = calculate_average_order_value(direct)

    assert (result.order_count, calculate_count(direct).value) == (5, 6)
    assert result.value == 24360.0 and result.value != pytest.approx(121800 / 6)


def test_direct_grouping_uses_the_mapped_names(direct) -> None:
    assert as_dict(revenue_by_group(direct, "product")) == {"Cable": 800.0, "Laptop": 118000.0, "Mouse": 3000.0}
    assert as_dict(revenue_by_group(direct, "region")) == {"East": 58000.0, "North": 62200.0, "South": 1600.0}
    assert as_dict(quantity_by_group(direct, "product")) == {"Cable": 4.0, "Laptop": 2.0, "Mouse": 6.0}


def test_direct_column_mapping_with_spaces_and_capitals(direct) -> None:
    assert map_columns(list(direct.columns)).resolved == {
        "order_id": "Order ID", "product": "Product Name", "region": "Region",
        "quantity": "Quantity", "revenue": "Total Revenue",
    }


def test_derived_and_direct_methods_are_reported_differently(sales, direct) -> None:
    assert calculate_total_revenue(sales).calculation_method != calculate_total_revenue(direct).calculation_method


# ---- missing required columns and ambiguous mappings --------------------------------------------------------
@pytest.mark.parametrize("dropped", [["unit_price"], ["quantity"], ["quantity", "unit_price"]])
def test_missing_required_columns_stop_the_calculation(sales, dropped) -> None:
    with pytest.raises(ToolError) as info:
        calculate_total_revenue(sales.drop(columns=dropped))

    assert info.value.code == ErrorCode.INSUFFICIENT_DATA


def test_average_order_value_without_order_id_stops(sales) -> None:
    with pytest.raises(ToolError) as info:
        calculate_average_order_value(sales.drop(columns=["order_id"]))

    assert info.value.code == ErrorCode.INSUFFICIENT_DATA


def test_grouping_by_a_missing_column_stops(direct) -> None:
    with pytest.raises(ToolError) as info:
        revenue_by_group(direct.drop(columns=["Region"]), "region")

    assert info.value.code == ErrorCode.MISSING_COLUMN


def test_an_extra_revenue_like_column_makes_the_mapping_ambiguous(direct) -> None:
    ambiguous = direct.assign(Sales=direct["Total Revenue"])

    assert map_columns(list(ambiguous.columns)).ambiguous == {"revenue": ["Total Revenue", "Sales"]}
    with pytest.raises(ToolError) as info:
        calculate_total_revenue(ambiguous)
    assert info.value.code == ErrorCode.AMBIGUOUS_COLUMN
    assert calculate_total_revenue(ambiguous, column_overrides={"revenue": "Sales"}).value == 121800.0


def test_incorrect_mapping_override_is_rejected(direct) -> None:
    with pytest.raises(ToolError) as info:
        calculate_total_revenue(direct, column_overrides={"revenue": "Nope"})

    assert info.value.code == ErrorCode.MISSING_COLUMN


def test_a_bad_number_in_the_evaluation_data_is_not_ignored(sales) -> None:
    broken = sales.copy()
    broken.loc[0, "unit_price"] = "50,000"  # a thousand separator is not a number

    with pytest.raises(ToolError) as info:
        calculate_total_revenue(broken)

    assert info.value.code == ErrorCode.INVALID_NUMERIC_VALUES
    assert info.value.details == {"column": "unit_price", "invalid_count": 1, "examples": ["50,000"]}