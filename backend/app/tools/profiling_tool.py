"""Dataset profiling: types, missing values, statistics and safe, limited value lists."""

import re

import numpy as np
import pandas as pd

from app.schemas.profile_schema import (
    CategoricalInfo,
    ColumnMappingInfo,
    ColumnProfile,
    DataframeProfile,
    DataType,
    DatetimeStats,
    NumericStats,
    ValueCount,
)
from app.tools.missing_value_tool import analyze_missing_values
from app.utils.column_mapper import map_columns
from app.utils.dataframe_utils import parse_numeric, safe_float

_INTEGER = re.compile(r"^[+-]?\d+$")
_LEADING_ZERO = re.compile(r"^[+-]?0\d")  # 007, 00123: looks like an id, not a number
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?$")


def infer_data_type(values: pd.Series) -> DataType:
    """Decide the type of a column from its non-missing text values.

    integer  every value is a whole number (no leading zeros)
    float    every value is a number, at least one has decimals
    boolean  every value is true/false (any letter case)
    datetime every value is an ISO date or date-time (YYYY-MM-DD[ HH:MM[:SS]])
    text     anything else (also numbers with leading zeros, "1,200", "$50", "N/A")
    empty    the column has no values at all
    Day/month formats such as 05/01/2025 are NOT guessed. They stay text.
    """
    if values.empty:
        return "empty"
    if values.str.lower().isin(["true", "false"]).all():
        return "boolean"
    if not values.str.match(_LEADING_ZERO).any():
        _, invalid = parse_numeric(values)
        if not invalid.any():
            return "integer" if values.str.match(_INTEGER).all() else "float"
    if values.str.match(_ISO_DATE).all():
        parsed = pd.to_datetime(values, format="ISO8601", errors="coerce")
        if parsed.notna().all():
            return "datetime"
    return "text"


def _numeric_stats(numbers: pd.Series) -> NumericStats:
    # Extremely large numbers can overflow to infinity. safe_float() then returns None (null).
    with np.errstate(over="ignore", invalid="ignore"):
        quartiles = numbers.quantile([0.25, 0.5, 0.75])
        return NumericStats(
            count=int(numbers.count()),
            mean=safe_float(numbers.mean()),
            std=safe_float(numbers.std()),  # None when there is only one value
            min=safe_float(numbers.min()),
            p25=safe_float(quartiles.loc[0.25]),
            median=safe_float(quartiles.loc[0.5]),
            p75=safe_float(quartiles.loc[0.75]),
            max=safe_float(numbers.max()),
        )


def _format_timestamp(value: pd.Timestamp, date_only: bool) -> str:
    return value.strftime("%Y-%m-%d" if date_only else "%Y-%m-%d %H:%M:%S")


def _datetime_stats(values: pd.Series) -> DatetimeStats:
    parsed = pd.to_datetime(values, format="ISO8601", errors="coerce")
    date_only = bool((parsed == parsed.dt.normalize()).all())
    return DatetimeStats(
        min=_format_timestamp(parsed.min(), date_only),
        max=_format_timestamp(parsed.max(), date_only),
    )


def _shorten(text: str, max_length: int) -> str:
    return text if len(text) <= max_length else text[: max_length - 1] + "…"


def _categorical_info(values: pd.Series, top_values: int, value_max_length: int) -> CategoricalInfo:
    counts = values.value_counts()
    # Most common first. Equal counts are ordered by the value itself, so the result is stable.
    ordered = sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
    return CategoricalInfo(
        unique_count=len(ordered),
        top_values=[
            ValueCount(value=_shorten(str(value), value_max_length), count=int(count))
            for value, count in ordered[:top_values]
        ],
        values_truncated=len(ordered) > top_values,
    )


def _profile_column(
    name: str,
    values: pd.Series,
    missing_count: int,
    missing_percentage: float,
    top_values: int,
    value_max_length: int,
) -> ColumnProfile:
    data_type = infer_data_type(values)
    profile = ColumnProfile(
        name=name,
        data_type=data_type,
        missing_count=missing_count,
        missing_percentage=missing_percentage,
        unique_count=int(values.nunique()),
    )
    if data_type in ("integer", "float"):
        numbers, _ = parse_numeric(values)
        profile.unique_count = int(numbers.nunique())
        profile.numeric_stats = _numeric_stats(numbers)
    elif data_type == "datetime":
        profile.datetime_stats = _datetime_stats(values)
    elif data_type in ("text", "boolean"):
        profile.categorical = _categorical_info(values, top_values, value_max_length)
    return profile


def _mostly_numeric_warning(name: str, values: pd.Series) -> str | None:
    """Warn when a text column is mostly numbers with a few bad values."""
    _, invalid = parse_numeric(values)
    bad = int(invalid.sum())
    good = len(values) - bad
    if bad > 0 and good > bad:
        return f"Column '{name}' has {bad} value(s) that are not valid numbers, so it is treated as text."
    return None


def profile_dataframe(
    frame: pd.DataFrame,
    *,
    top_values: int = 10,
    value_max_length: int = 50,
) -> DataframeProfile:
    """Build a safe, limited profile. At most `top_values` values are listed per column."""
    missing = analyze_missing_values(frame)
    missing_by_name = {column.name: column for column in missing.columns}

    columns: list[ColumnProfile] = []
    warnings: list[str] = []
    for name in frame.columns:
        values = frame[name].dropna()
        column_missing = missing_by_name[str(name)]
        column = _profile_column(
            str(name),
            values,
            column_missing.missing_count,
            column_missing.missing_percentage,
            top_values,
            value_max_length,
        )
        columns.append(column)
        if column.data_type == "empty" and len(frame) > 0:
            warnings.append(f"Column '{name}' is completely empty.")
        if column.data_type == "text":
            warning = _mostly_numeric_warning(str(name), values)
            if warning:
                warnings.append(warning)

    if missing.duplicate_row_count > 0:
        warnings.append(f"{missing.duplicate_row_count} duplicate row(s) found (exact copies of an earlier row).")

    mapping = map_columns([str(name) for name in frame.columns])
    for role, candidates in mapping.ambiguous.items():
        warnings.append(
            f"More than one column could be '{role}': {', '.join(candidates)}. "
            "Analysis needs one clear choice."
        )

    return DataframeProfile(
        row_count=missing.total_rows,
        column_count=missing.total_columns,
        column_names=[str(name) for name in frame.columns],
        duplicate_row_count=missing.duplicate_row_count,
        total_missing_cells=missing.total_missing_cells,
        columns=columns,
        column_mapping=ColumnMappingInfo(
            resolved=mapping.resolved,
            ambiguous=mapping.ambiguous,
            missing=mapping.missing,
        ),
        warnings=warnings,
    )