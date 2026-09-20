"""Missing-value analysis."""

import pandas as pd

from app.schemas.profile_schema import ColumnMissing, MissingValueReport

MAX_ROW_NUMBERS = 10


def percentage(part: int, whole: int) -> float:
    """Calculate percentage."""
    return round(part / whole * 100, 2) if whole else 0.0


def count_duplicate_rows(frame: pd.DataFrame) -> int:
    """Count duplicate rows excluding the first occurrence."""
    if frame.empty:
        return 0
    return int(frame.duplicated().sum())


def analyze_missing_values(
    frame: pd.DataFrame,
    *,
    max_row_numbers: int = MAX_ROW_NUMBERS,
) -> MissingValueReport:
    """Analyze missing values in a DataFrame."""
    total_rows, total_columns = frame.shape
    missing_counts = frame.isna().sum()

    columns = [
        ColumnMissing(
            name=str(name),
            missing_count=int(missing_counts[name]),
            missing_percentage=percentage(
                int(missing_counts[name]), total_rows
            ),
        )
        for name in frame.columns
    ]

    total_cells = total_rows * total_columns
    total_missing = int(missing_counts.sum())

    row_has_missing = frame.isna().any(axis=1)

    sample = [
        int(position) + 1
        for position in row_has_missing.to_numpy().nonzero()[0][
            :max_row_numbers
        ]
    ]

    return MissingValueReport(
        total_rows=total_rows,
        total_columns=total_columns,
        total_cells=total_cells,
        total_missing_cells=total_missing,
        missing_percentage=percentage(total_missing, total_cells),
        columns=columns,
        columns_with_missing=[
            c.name for c in columns if c.missing_count > 0
        ],
        fully_empty_columns=[
            c.name
            for c in columns
            if total_rows > 0 and c.missing_count == total_rows
        ],
        rows_with_missing_count=int(row_has_missing.sum()),
        rows_with_missing_sample=sample,
        duplicate_row_count=count_duplicate_rows(frame),
    )
