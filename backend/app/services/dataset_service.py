"""Business logic for datasets: ingest an upload and build a preview.

This file connects the small parts together (validator -> parser -> repository).
It knows nothing about FastAPI, so it is easy to test.
"""

import logging
import uuid
from datetime import UTC, datetime
from typing import BinaryIO

from app.core.config import Settings
from app.core.errors import AppError, ErrorCode
from app.schemas.dataset_schema import DatasetMetadata, DatasetPreview, DatasetSummary
from app.services.dataset_repository import DatasetRepository
from app.services.file_parser import (
    ParseResult,
    TableLimits,
    parse_csv,
    parse_xlsx,
    read_preview,
)
from app.utils.upload_validator import (
    detect_source_format,
    parse_dataset_id,
    read_upload_bytes,
    sanitize_filename,
)

logger = logging.getLogger(__name__)


def ingest_upload(
    *,
    filename: str | None,
    content_type: str | None,
    stream: BinaryIO,
    settings: Settings,
    repository: DatasetRepository,
) -> DatasetSummary:
    """Validate an uploaded file, convert it to a clean CSV and store it."""
    safe_filename = sanitize_filename(filename)
    source_format = detect_source_format(safe_filename, content_type)
    data = read_upload_bytes(stream, settings.max_upload_size_bytes)

    limits = TableLimits(max_rows=settings.max_rows, max_columns=settings.max_columns)
    dataset_id = uuid.uuid4()
    temp_path = repository.new_temp_path(dataset_id)

    try:
        with temp_path.open("w", encoding="utf-8", newline="") as destination:
            result: ParseResult
            if source_format == "csv":
                result = parse_csv(data, destination, limits)
            else:
                result = parse_xlsx(data, destination, limits, settings.max_xlsx_uncompressed_bytes)

        metadata = DatasetMetadata(
            dataset_id=dataset_id,
            filename=safe_filename,
            source_format=source_format,
            row_count=result.row_count,
            column_count=len(result.columns),
            column_names=result.columns,
            original_size_bytes=len(data),
            uploaded_at=datetime.now(UTC),
            warnings=result.warnings,
        )
        repository.save(metadata, temp_path)
    finally:
        # After a successful save the temp file is already moved, so this does nothing.
        # After any error it removes the half-written file.
        temp_path.unlink(missing_ok=True)

    # Log only technical facts, never the file contents.
    logger.info(
        "Dataset stored: id=%s format=%s rows=%d columns=%d size_bytes=%d",
        dataset_id,
        source_format,
        metadata.row_count,
        metadata.column_count,
        metadata.original_size_bytes,
    )
    return DatasetSummary(**metadata.model_dump())


def get_dataset_preview(
    *,
    raw_dataset_id: str,
    rows: int | None,
    settings: Settings,
    repository: DatasetRepository,
) -> DatasetPreview:
    """Return the first few rows of a stored dataset."""
    dataset_id = parse_dataset_id(raw_dataset_id)

    requested_rows = settings.preview_default_rows if rows is None else rows
    if not 1 <= requested_rows <= settings.preview_max_rows:
        raise AppError(
            ErrorCode.INVALID_PREVIEW_ROWS,
            f"rows must be between 1 and {settings.preview_max_rows}.",
            status_code=422,
            details={"max_rows": settings.preview_max_rows},
        )

    metadata = repository.get(dataset_id)
    csv_path = repository.get_csv_path(dataset_id)
    preview_rows, cells_truncated = read_preview(
        csv_path, requested_rows, settings.preview_max_cell_length
    )

    return DatasetPreview(
        dataset_id=metadata.dataset_id,
        filename=metadata.filename,
        source_format=metadata.source_format,
        row_count=metadata.row_count,
        column_count=metadata.column_count,
        column_names=metadata.column_names,
        requested_rows=requested_rows,
        preview_rows=preview_rows,
        cells_truncated=cells_truncated,
        max_cell_length=settings.preview_max_cell_length,
    )