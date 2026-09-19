"""Unit tests for CSV/XLSX parsing, header rules and limits."""

import csv
import io
import zipfile
from datetime import date, datetime

import pytest

from app.core.errors import AppError, ErrorCode
from app.services.file_parser import (
    ParseResult,
    TableLimits,
    parse_csv,
    parse_xlsx,
    read_preview,
)
from tests.helpers import make_csv, make_xlsx

LIMITS = TableLimits(max_rows=100, max_columns=10)
BIG = 10 * 1024 * 1024


def run_csv(data: bytes, limits: TableLimits = LIMITS) -> tuple[ParseResult, list[list[str]]]:
    out = io.StringIO()
    result = parse_csv(data, out, limits)
    return result, list(csv.reader(io.StringIO(out.getvalue())))


def run_xlsx(data: bytes, limits: TableLimits = LIMITS, max_uncompressed: int = BIG):
    out = io.StringIO()
    result = parse_xlsx(data, out, limits, max_uncompressed)
    return result, list(csv.reader(io.StringIO(out.getvalue())))


def code_of(error: pytest.ExceptionInfo[AppError]) -> ErrorCode:
    return error.value.code


# ================================ CSV ==========================================
def test_valid_csv_is_parsed() -> None:
    result, rows = run_csv(b"name,qty\nPen,5\nBook,7\n")

    assert result.columns == ["name", "qty"]
    assert result.row_count == 2
    assert result.warnings == []
    assert rows == [["name", "qty"], ["Pen", "5"], ["Book", "7"]]


def test_utf8_bom_is_removed() -> None:
    result, _ = run_csv(b"\xef\xbb\xbfname,qty\nPen,5\n")

    assert result.columns == ["name", "qty"]


def test_header_names_are_trimmed() -> None:
    result, _ = run_csv(b"  name , qty \nPen,5\n")

    assert result.columns == ["name", "qty"]


def test_windows_line_endings_and_blank_lines_are_handled() -> None:
    result, rows = run_csv(b"name,qty\r\n\r\nPen,5\r\n\r\nBook,7\r\n")

    assert result.row_count == 2
    assert rows[1:] == [["Pen", "5"], ["Book", "7"]]


def test_quoted_values_with_commas_and_new_lines_are_kept() -> None:
    _, rows = run_csv(b'name,note\n"Pen, blue","line1\nline2"\n')

    assert rows[1] == ["Pen, blue", "line1\nline2"]


def test_leading_zeros_are_not_changed() -> None:
    _, rows = run_csv(b"code,qty\n007,5\n")

    assert rows[1][0] == "007"


def test_short_rows_are_padded_with_empty_values() -> None:
    _, rows = run_csv(b"a,b,c\n1,2,3\n4\n")

    assert rows[2] == ["4", "", ""]


def test_trailing_empty_header_columns_are_ignored() -> None:
    result, _ = run_csv(b"a,b,,\n1,2,,\n")

    assert result.columns == ["a", "b"]


def test_row_with_extra_values_is_rejected() -> None:
    with pytest.raises(AppError) as error:
        run_csv(b"a,b\n1,2\n3,4,5\n")

    assert code_of(error) == ErrorCode.MALFORMED_ROWS
    assert error.value.details["row"] == 3


def test_windows_1252_file_is_read_with_a_warning() -> None:
    result, rows = run_csv("city,note\nMumbai,caf\xe9\n".encode("cp1252"))

    assert rows[1][1] == "café"
    assert "Windows-1252" in result.warnings[0]


def test_unreadable_encoding_is_rejected() -> None:
    with pytest.raises(AppError) as error:
        run_csv(b"a,b\n\x81,2\n")  # 0x81 is invalid in both UTF-8 and Windows-1252

    assert code_of(error) == ErrorCode.INVALID_ENCODING


def test_utf16_file_is_rejected_with_clear_code() -> None:
    with pytest.raises(AppError) as error:
        run_csv("a,b\n1,2\n".encode("utf-16"))

    assert code_of(error) == ErrorCode.INVALID_ENCODING


def test_binary_content_is_rejected_as_corrupted() -> None:
    with pytest.raises(AppError) as error:
        run_csv(b"a,b\n1,\x00\x01\x02\n")

    assert code_of(error) == ErrorCode.CORRUPTED_FILE


def test_whitespace_only_csv_is_empty() -> None:
    with pytest.raises(AppError) as error:
        run_csv(b"\n  \n\n")

    assert code_of(error) == ErrorCode.EMPTY_FILE


def test_blank_header_name_is_rejected() -> None:
    with pytest.raises(AppError) as error:
        run_csv(b"a,,c\n1,2,3\n")

    assert code_of(error) == ErrorCode.MISSING_HEADERS
    assert error.value.details["blank_columns"] == [2]


def test_duplicate_column_names_are_rejected_ignoring_case() -> None:
    with pytest.raises(AppError) as error:
        run_csv(b"Sales,sales,qty\n1,2,3\n")

    assert code_of(error) == ErrorCode.DUPLICATE_COLUMNS


def test_header_only_file_has_no_data_rows() -> None:
    with pytest.raises(AppError) as error:
        run_csv(b"a,b,c\n")

    assert code_of(error) == ErrorCode.NO_DATA_ROWS


def test_header_followed_by_blank_rows_has_no_data_rows() -> None:
    with pytest.raises(AppError) as error:
        run_csv(b"a,b,c\n,,\n\n")

    assert code_of(error) == ErrorCode.NO_DATA_ROWS


def test_row_limit_is_enforced() -> None:
    limits = TableLimits(max_rows=3, max_columns=10)

    result, _ = run_csv(make_csv([["a"], [1], [2], [3]]), limits)
    assert result.row_count == 3

    with pytest.raises(AppError) as error:
        run_csv(make_csv([["a"], [1], [2], [3], [4]]), limits)
    assert code_of(error) == ErrorCode.TOO_MANY_ROWS


def test_column_limit_is_enforced() -> None:
    limits = TableLimits(max_rows=10, max_columns=3)

    with pytest.raises(AppError) as error:
        run_csv(make_csv([["a", "b", "c", "d"], [1, 2, 3, 4]]), limits)

    assert code_of(error) == ErrorCode.TOO_MANY_COLUMNS


def test_semicolon_separated_file_gets_a_helpful_error() -> None:
    with pytest.raises(AppError) as error:
        run_csv(b"name;qty\nPen;5\n")

    assert code_of(error) == ErrorCode.MALFORMED_ROWS
    assert "separator" in error.value.message


def test_bad_quote_is_rejected_not_crashed() -> None:
    with pytest.raises(AppError) as error:
        run_csv(b'a,b\n"unclosed,2\n')

    assert code_of(error) == ErrorCode.MALFORMED_ROWS


# ================================ XLSX =========================================
def test_valid_xlsx_is_converted_to_csv() -> None:
    data = make_xlsx([["name", "qty", "price"], ["Pen", 5, 9.5], ["Book", 7, 120]])

    result, rows = run_xlsx(data)

    assert result.columns == ["name", "qty", "price"]
    assert result.row_count == 2
    assert rows == [["name", "qty", "price"], ["Pen", "5", "9.5"], ["Book", "7", "120"]]


def test_xlsx_dates_and_empty_cells_are_converted() -> None:
    data = make_xlsx(
        [
            ["day", "when", "note"],
            [date(2025, 1, 5), datetime(2025, 1, 5, 14, 30), None],
        ]
    )

    _, rows = run_xlsx(data)

    assert rows[1] == ["2025-01-05", "2025-01-05 14:30:00", ""]


def test_only_first_sheet_is_used_and_a_warning_is_returned() -> None:
    data = make_xlsx([["a"], [1]], extra_sheet_rows=[["x"], [99]])

    result, rows = run_xlsx(data)

    assert rows == [["a"], ["1"]]
    assert "Only the first sheet" in result.warnings[0]


def test_xlsx_row_and_column_limits_are_enforced() -> None:
    limits = TableLimits(max_rows=2, max_columns=3)

    with pytest.raises(AppError) as rows_error:
        run_xlsx(make_xlsx([["a"], [1], [2], [3]]), limits)
    assert code_of(rows_error) == ErrorCode.TOO_MANY_ROWS

    with pytest.raises(AppError) as columns_error:
        run_xlsx(make_xlsx([["a", "b", "c", "d"], [1, 2, 3, 4]]), limits)
    assert code_of(columns_error) == ErrorCode.TOO_MANY_COLUMNS


def test_xlsx_with_missing_header_is_rejected() -> None:
    with pytest.raises(AppError) as error:
        run_xlsx(make_xlsx([["a", None, "c"], [1, 2, 3]]))

    assert code_of(error) == ErrorCode.MISSING_HEADERS


def test_xlsx_with_only_header_has_no_data_rows() -> None:
    with pytest.raises(AppError) as error:
        run_xlsx(make_xlsx([["a", "b"]]))

    assert code_of(error) == ErrorCode.NO_DATA_ROWS


def test_empty_xlsx_sheet_is_rejected() -> None:
    with pytest.raises(AppError) as error:
        run_xlsx(make_xlsx([]))

    assert code_of(error) in {ErrorCode.EMPTY_FILE, ErrorCode.NO_DATA_ROWS}


def test_xlsx_title_row_above_the_table_is_rejected() -> None:
    data = make_xlsx([["Sales report 2025"], ["a", "b"], [1, 2]])

    with pytest.raises(AppError) as error:
        run_xlsx(data)

    assert code_of(error) == ErrorCode.MALFORMED_ROWS


def test_random_bytes_are_rejected_as_corrupted_xlsx() -> None:
    with pytest.raises(AppError) as error:
        run_xlsx(b"this is not an excel file")

    assert code_of(error) == ErrorCode.CORRUPTED_FILE


def test_truncated_xlsx_is_rejected_as_corrupted() -> None:
    good = make_xlsx([["a"], [1]])

    with pytest.raises(AppError) as error:
        run_xlsx(good[: len(good) // 2])

    assert code_of(error) == ErrorCode.CORRUPTED_FILE


def test_zip_that_is_not_a_workbook_is_rejected_as_corrupted() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("hello.txt", "just a text file")

    with pytest.raises(AppError) as error:
        run_xlsx(buffer.getvalue())

    assert code_of(error) == ErrorCode.CORRUPTED_FILE


def test_xlsx_that_expands_too_much_is_rejected() -> None:
    data = make_xlsx([["a"], [1]])

    with pytest.raises(AppError) as error:
        run_xlsx(data, max_uncompressed=100)  # only 100 bytes allowed after unzipping

    assert code_of(error) == ErrorCode.FILE_TOO_LARGE
    assert error.value.status_code == 413


# ================================ Preview ======================================
def test_read_preview_limits_rows_and_marks_empty_cells(tmp_path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,\n2,x\n3,y\n", encoding="utf-8")

    rows, truncated = read_preview(path, rows=2, max_cell_length=50)

    assert rows == [["1", None], ["2", "x"]]
    assert truncated is False


def test_read_preview_truncates_long_cells_to_the_limit(tmp_path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("a\n" + "z" * 300 + "\nshort\n", encoding="utf-8")

    rows, truncated = read_preview(path, rows=5, max_cell_length=20)

    assert len(rows[0][0]) == 20
    assert rows[0][0].endswith("…")
    assert rows[1] == ["short"]
    assert truncated is True