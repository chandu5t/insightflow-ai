"""Unit tests for filename, extension, MIME, size and dataset-id checks."""

import io

import pytest

from app.core.errors import AppError, ErrorCode
from app.utils.upload_validator import (
    detect_source_format,
    parse_dataset_id,
    read_upload_bytes,
    sanitize_filename,
)


# ---- sanitize_filename: path traversal protection -------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("sales.csv", "sales.csv"),
        ("../../evil.csv", "evil.csv"),
        ("..\\..\\windows\\evil.csv", "evil.csv"),
        ("C:\\Users\\me\\data.xlsx", "data.xlsx"),
        ("/etc/passwd", "passwd"),
        ("bad<>:name|?.csv", "bad___name__.csv"),
        ("report\x00.csv", "report.csv"),
        ("....csv", "csv"),
        ("  .hidden.csv  ", "hidden.csv"),
        ("", "dataset"),
        (None, "dataset"),
        ("..", "dataset"),
    ],
)
def test_sanitize_filename(raw: str | None, expected: str) -> None:
    assert sanitize_filename(raw) == expected


def test_sanitize_filename_keeps_non_english_letters() -> None:
    assert sanitize_filename("बिक्री_डेटा.csv") == "बिक्री_डेटा.csv"


def test_sanitize_filename_limits_length_and_keeps_extension() -> None:
    result = sanitize_filename("a" * 500 + ".csv")

    assert len(result) <= 100
    assert result.endswith(".csv")


# ---- detect_source_format: extension and MIME ---------------------------------
def test_csv_and_xlsx_are_accepted() -> None:
    assert detect_source_format("sales.csv", "text/csv") == "csv"
    assert detect_source_format("SALES.CSV", None) == "csv"
    assert detect_source_format("book.xlsx", "application/octet-stream") == "xlsx"


def test_windows_csv_mime_type_is_accepted() -> None:
    assert detect_source_format("sales.csv", "application/vnd.ms-excel") == "csv"


def test_mime_type_with_parameters_is_accepted() -> None:
    assert detect_source_format("sales.csv", "text/csv; charset=utf-8") == "csv"


@pytest.mark.parametrize("filename", ["old.xls", "macro.xlsm", "OLD.XLS"])
def test_legacy_excel_formats_are_rejected(filename: str) -> None:
    with pytest.raises(AppError) as error:
        detect_source_format(filename, None)

    assert error.value.code == ErrorCode.UNSUPPORTED_FILE_TYPE
    assert error.value.status_code == 400


@pytest.mark.parametrize("filename", ["notes.txt", "run.exe", "data.json", "noextension", "sales.csv.exe", ".csv"])
def test_unsupported_extensions_are_rejected(filename: str) -> None:
    with pytest.raises(AppError) as error:
        detect_source_format(filename, None)

    assert error.value.code == ErrorCode.UNSUPPORTED_FILE_TYPE


def test_wrong_mime_type_is_rejected() -> None:
    with pytest.raises(AppError) as error:
        detect_source_format("sales.csv", "image/png")

    assert error.value.code == ErrorCode.UNSUPPORTED_FILE_TYPE


# ---- read_upload_bytes: size limit and empty file ------------------------------
def test_read_upload_bytes_returns_content() -> None:
    assert read_upload_bytes(io.BytesIO(b"abc"), max_bytes=10) == b"abc"


def test_file_exactly_at_the_limit_is_accepted() -> None:
    assert len(read_upload_bytes(io.BytesIO(b"x" * 100), max_bytes=100)) == 100


def test_file_one_byte_over_the_limit_is_rejected() -> None:
    with pytest.raises(AppError) as error:
        read_upload_bytes(io.BytesIO(b"x" * 101), max_bytes=100)

    assert error.value.code == ErrorCode.FILE_TOO_LARGE
    assert error.value.status_code == 413


def test_empty_upload_is_rejected() -> None:
    with pytest.raises(AppError) as error:
        read_upload_bytes(io.BytesIO(b""), max_bytes=100)

    assert error.value.code == ErrorCode.EMPTY_FILE


# ---- parse_dataset_id -------------------------------------------------------------
def test_valid_uuid_is_accepted() -> None:
    raw = "123e4567-e89b-12d3-a456-426614174000"

    assert str(parse_dataset_id(raw)) == raw
    assert str(parse_dataset_id(raw.upper())) == raw


@pytest.mark.parametrize(
    "raw",
    [
        "not-a-uuid",
        "",
        "../../etc/passwd",
        "..\\..\\secret",
        "123e4567e89b12d3a456426614174000",  # no hyphens
        "{123e4567-e89b-12d3-a456-426614174000}",
        "urn:uuid:123e4567-e89b-12d3-a456-426614174000",
        "123e4567-e89b-12d3-a456-426614174000/../x",
        "dataset_001",
    ],
)
def test_invalid_dataset_ids_are_rejected(raw: str) -> None:
    with pytest.raises(AppError) as error:
        parse_dataset_id(raw)

    assert error.value.code == ErrorCode.INVALID_DATASET_ID
    assert error.value.status_code == 422