"""Checks on the column headers of a table (used for both CSV and XLSX)."""

import re

from app.core.errors import AppError, ErrorCode

_OTHER_SEPARATORS = re.compile(r"[;\t|]")


def validate_headers(
    raw_headers: list[str],
    *,
    max_columns: int,
    check_separator: bool = False,
) -> list[str]:
    """Clean the header row and reject bad ones. Returns the cleaned column names."""
    names = [header.strip() for header in raw_headers]

    if len(names) > max_columns:
        raise AppError(
            ErrorCode.TOO_MANY_COLUMNS,
            f"The file has {len(names)} columns. The limit is {max_columns}.",
            details={"columns": len(names), "max_columns": max_columns},
        )

    if check_separator and len(names) == 1 and _OTHER_SEPARATORS.search(names[0]):
        raise AppError(
            ErrorCode.MALFORMED_ROWS,
            "Only one column was found, but the header contains ';', '|' or a tab. "
            "The file may use a different separator. Please save it as a comma-separated CSV.",
        )

    blank_positions = [index for index, name in enumerate(names, start=1) if not name]
    if blank_positions:
        raise AppError(
            ErrorCode.MISSING_HEADERS,
            "Every column must have a name in the first row.",
            details={"blank_columns": blank_positions},
        )

    seen: dict[str, str] = {}
    duplicates: list[str] = []
    for name in names:
        key = name.casefold()  # "Sales" and "sales" are treated as the same name
        if key in seen:
            duplicates.append(name)
        else:
            seen[key] = name
    if duplicates:
        raise AppError(
            ErrorCode.DUPLICATE_COLUMNS,
            "Column names must be unique (capital and small letters are treated as the same).",
            details={"duplicate_columns": duplicates},
        )

    return names