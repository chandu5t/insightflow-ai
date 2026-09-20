"""Revenue rules. All tools use this one module, so revenue is calculated in ONE place.

Rule 1  A direct revenue column (revenue, total_revenue, sales, total_sales) is used as it is,
        after checking that every value is a valid number.
Rule 2  Without it, revenue = quantity x unit_price, row by row.
Rule 3  Without both, we stop with INSUFFICIENT_DATA. We never estimate.
Rule 4  The method is always returned, so the answer can explain how it was calculated.
Rule 5  Empty cells: the row has no revenue and is left out of totals (and counted).
        Invalid text such as "abc" or "1,200": we stop with INVALID_NUMERIC_VALUES.
        Negative numbers are used as they are (for example refunds).
        More than one possible column for a role: we stop with AMBIGUOUS_COLUMN.
"""

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from app.core.errors import ErrorCode, ToolError
from app.utils.column_mapper import ColumnMapping, resolve_column
from app.utils.dataframe_utils import to_numeric_strict

RevenueMethod = Literal["direct_revenue_column", "derived_quantity_times_unit_price"]


@dataclass(frozen=True)
class NumericValues:
    """A column of numbers plus how it was obtained."""

    values: pd.Series
    method: str
    columns_used: list[str]


def compute_revenue(frame: pd.DataFrame, mapping: ColumnMapping) -> NumericValues:
    """Revenue for every row (NaN where it cannot be calculated)."""
    mapping.require_if_ambiguous("revenue")
    revenue_column = mapping.get("revenue")
    if revenue_column is not None:
        return NumericValues(
            to_numeric_strict(frame[revenue_column], revenue_column),
            "direct_revenue_column",
            [revenue_column],
        )

    mapping.require_if_ambiguous("quantity")
    mapping.require_if_ambiguous("unit_price")
    quantity_column, price_column = mapping.get("quantity"), mapping.get("unit_price")
    if quantity_column is None or price_column is None:
        raise ToolError(
            ErrorCode.INSUFFICIENT_DATA,
            "Revenue cannot be calculated. The dataset needs a revenue column, "
            "or both a quantity column and a unit price column.",
            details={
                "needs": "revenue OR (quantity AND unit_price)",
                "missing_roles": [role for role in ("revenue", "quantity", "unit_price") if mapping.get(role) is None],
            },
        )
    quantity = to_numeric_strict(frame[quantity_column], quantity_column)
    price = to_numeric_strict(frame[price_column], price_column)
    return NumericValues(
        quantity * price,
        "derived_quantity_times_unit_price",
        [quantity_column, price_column],
    )


def resolve_numeric_values(frame: pd.DataFrame, requested: str, mapping: ColumnMapping) -> NumericValues:
    """Numbers for a request. "revenue" uses the revenue rules. Anything else is one numeric column."""
    if requested == "revenue":
        return compute_revenue(frame, mapping)
    column = resolve_column(list(frame.columns), requested, mapping)
    return NumericValues(to_numeric_strict(frame[column], column), "numeric_column", [column])