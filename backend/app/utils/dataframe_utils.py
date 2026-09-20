"""Helpers to load a stored dataset into Pandas and to read numbers safely.

Rules used everywhere in the analysis tools:
- Every value is read as text first, and spaces at both ends are removed.
- Only an EMPTY cell (or a cell with only spaces) is "missing". Words like "N/A" or "null"
  are NOT treated as missing, because we never guess. They are normal text.
- A number must be a plain number. "1,200", "$50", "abc", "inf" and "nan" are NOT numbers.
"""

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.errors import AppError, ErrorCode, ToolError

logger = logging.getLogger(__name__)

MAX_INVALID_EXAMPLES = 5
MAX_EXAMPLE_LENGTH = 30


def clean_text_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Trim spaces from every value and turn empty cells into missing values (NaN)."""
    trimmed = frame.apply(lambda column: column.str.strip())
    return trimmed.mask(trimmed.eq(""))


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    """Read a stored dataset (the clean CSV made in Module 2) as text."""
    try:
        frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, OSError) as exc:
        logger.error("Stored dataset could not be read (%s)", type(exc).__name__)
        raise AppError(
            ErrorCode.DATASET_UNREADABLE,
            "The stored dataset could not be read.",
            status_code=500,
        ) from exc
    return clean_text_frame(frame)


def safe_float(value: object, digits: int = 6) -> float | None:
    """Return a JSON-safe float (rounded), or None for NaN, infinity or non-numbers."""
    if value is None:
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return round(number, digits)


def parse_numeric(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Convert text to numbers. Returns (numbers, invalid_mask).

    Empty cells become NaN and are NOT invalid. Text, infinity and 'nan' are invalid.
    """
    converted = pd.to_numeric(series, errors="coerce").astype("float64")
    invalid = series.notna() & ~np.isfinite(converted)
    return converted.mask(invalid), invalid


def to_numeric_strict(series: pd.Series, column: str) -> pd.Series:
    """Like parse_numeric, but refuse to continue when any value is not a valid number."""
    numbers, invalid = parse_numeric(series)
    if invalid.any():
        examples = [
            str(value)[:MAX_EXAMPLE_LENGTH]
            for value in series[invalid].drop_duplicates().head(MAX_INVALID_EXAMPLES)
        ]
        raise ToolError(
            ErrorCode.INVALID_NUMERIC_VALUES,
            f"Column '{column}' has {int(invalid.sum())} value(s) that are not valid numbers.",
            details={
                "column": column,
                "invalid_count": int(invalid.sum()),
                "examples": examples,
            },
        )
    return numbers