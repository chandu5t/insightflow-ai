"""Pydantic models for dataset upload, storage and preview."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

SourceFormat = Literal["csv", "xlsx"]


class DatasetMetadata(BaseModel):
    """What we store on disk (as JSON) for every dataset.

    The path of the CSV file is NOT stored here. It is always calculated from
    dataset_id, so nobody can change the metadata to point to another file.
    """

    dataset_id: UUID
    filename: str
    source_format: SourceFormat
    row_count: int
    column_count: int
    column_names: list[str]
    original_size_bytes: int
    uploaded_at: datetime
    status: Literal["uploaded"] = "uploaded"
    warnings: list[str] = Field(default_factory=list)


class DatasetSummary(BaseModel):
    """Response of POST /datasets/upload."""

    dataset_id: UUID
    filename: str
    source_format: SourceFormat
    row_count: int
    column_count: int
    column_names: list[str]
    uploaded_at: datetime
    status: Literal["uploaded"]
    warnings: list[str]


class DatasetPreview(BaseModel):
    """Response of GET /datasets/{dataset_id}/preview."""

    dataset_id: UUID
    filename: str
    source_format: SourceFormat
    row_count: int
    column_count: int
    column_names: list[str]
    requested_rows: int
    preview_rows: list[list[str | None]]
    cells_truncated: bool
    max_cell_length: int