"""Checks on the upload itself: filename, extension, MIME type, size and dataset id."""

import unicodedata
import uuid
from pathlib import PurePath
from typing import BinaryIO

from app.core.config import BYTES_PER_MB
from app.core.errors import AppError, ErrorCode
from app.schemas.dataset_schema import SourceFormat

ALLOWED_EXTENSIONS: dict[str, SourceFormat] = {".csv": "csv", ".xlsx": "xlsx"}

# The MIME type is sent by the browser, so it can be wrong or fake. We only use it
# as a quick sanity check. The real check is that the file content can be parsed.
# "application/vnd.ms-excel" is included for CSV because Windows often sends it.
ALLOWED_MIME_TYPES: dict[SourceFormat, set[str]] = {
    "csv": {
        "text/csv",
        "text/plain",
        "text/x-csv",
        "text/comma-separated-values",
        "application/csv",
        "application/x-csv",
        "application/vnd.ms-excel",
        "application/octet-stream",
    },
    "xlsx": {
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/zip",
        "application/x-zip-compressed",
        "application/octet-stream",
    },
}

MAX_FILENAME_LENGTH = 100
READ_CHUNK_SIZE = 64 * 1024


def sanitize_filename(raw_name: str | None) -> str:
    """Return a safe display name. The result is never used to build a file path."""
    name = (raw_name or "").replace("\x00", "")
    # Keep only the last part, so "../../x.csv" and "C:\\temp\\x.csv" become "x.csv".
    name = name.replace("\\", "/").split("/")[-1]
    # Keep letters (any language), digits, combining marks, and a few safe symbols.
    safe_chars = [
        ch
        if ch.isalnum() or unicodedata.category(ch).startswith("M") or ch in "._- ()"
        else "_"
        for ch in name
    ]
    name = "".join(safe_chars).strip(" .")
    while ".." in name:
        name = name.replace("..", ".")
    if len(name) > MAX_FILENAME_LENGTH:
        stem, suffix = PurePath(name).stem, PurePath(name).suffix
        name = stem[: MAX_FILENAME_LENGTH - len(suffix)] + suffix
    return name or "dataset"


def detect_source_format(filename: str, content_type: str | None) -> SourceFormat:
    """Check the extension and MIME type. Return "csv" or "xlsx", or raise AppError."""
    extension = PurePath(filename).suffix.lower()

    if extension == ".xls":
        raise AppError(
            ErrorCode.UNSUPPORTED_FILE_TYPE,
            "Old Excel files (.xls) are not supported. Please save the file as .xlsx or .csv and upload again.",
            details={"extension": extension, "allowed": sorted(ALLOWED_EXTENSIONS)},
        )
    if extension == ".xlsm":
        raise AppError(
            ErrorCode.UNSUPPORTED_FILE_TYPE,
            "Macro-enabled Excel files (.xlsm) are not supported. Please save the file as .xlsx or .csv and upload again.",
            details={"extension": extension, "allowed": sorted(ALLOWED_EXTENSIONS)},
        )
    if extension not in ALLOWED_EXTENSIONS:
        raise AppError(
            ErrorCode.UNSUPPORTED_FILE_TYPE,
            "Only .csv and .xlsx files are supported.",
            details={"extension": extension or "(none)", "allowed": sorted(ALLOWED_EXTENSIONS)},
        )

    source_format = ALLOWED_EXTENSIONS[extension]

    if content_type:
        mime_type = content_type.split(";")[0].strip().lower()
        if mime_type and mime_type not in ALLOWED_MIME_TYPES[source_format]:
            raise AppError(
                ErrorCode.UNSUPPORTED_FILE_TYPE,
                f"The file type '{mime_type}' does not match a .{source_format} file.",
                details={"content_type": mime_type},
            )

    return source_format


def read_upload_bytes(stream: BinaryIO, max_bytes: int) -> bytes:
    """Read the upload in small chunks and stop as soon as it becomes too large."""
    chunks: list[bytes] = []
    total = 0
    while True:
        # Ask for at most one byte more than the limit, so we can detect "too large".
        chunk = stream.read(min(READ_CHUNK_SIZE, max_bytes + 1 - total))
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise AppError(
                ErrorCode.FILE_TOO_LARGE,
                f"The file is larger than the allowed limit of {max_bytes / BYTES_PER_MB:g} MB.",
                status_code=413,
                details={"max_bytes": max_bytes},
            )
        chunks.append(chunk)

    data = b"".join(chunks)
    if not data:
        raise AppError(ErrorCode.EMPTY_FILE, "The uploaded file is empty.")
    return data


def parse_dataset_id(raw_id: str) -> uuid.UUID:
    """Accept only a normal UUID like 123e4567-e89b-12d3-a456-426614174000."""
    try:
        parsed = uuid.UUID(raw_id)
    except (ValueError, AttributeError, TypeError):
        parsed = None

    # Compare with the standard text form, so odd formats ({...}, urn:uuid:...) are rejected.
    if parsed is None or str(parsed) != raw_id.lower():
        raise AppError(
            ErrorCode.INVALID_DATASET_ID,
            "The dataset id is not valid.",
            status_code=422,
        )
    return parsed