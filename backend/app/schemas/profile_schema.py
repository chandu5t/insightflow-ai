"""Pydantic models for the dataset profile and the missing-value report."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.dataset_schema import SourceFormat

DataType = Literal["integer", "float", "boolean", "datetime", "text", "empty"]


class ColumnMissing(BaseModel):
    name: str
    missing_count: int
    missing_percentage: float


class MissingValueReport(BaseModel):
    """Output of the missing-value tool. It never contains cell values."""

    tool: Literal["missing_value_tool"] = "missing_value_tool"
    total_rows: int
    total_columns: int
    total_cells: int
    total_missing_cells: int
    missing_percentage: float
    columns: list[ColumnMissing]
    columns_with_missing: list[str]
    fully_empty_columns: list[str]
    rows_with_missing_count: int
    rows_with_missing_sample: list[int] = Field(
        description="First data row numbers (1 = first data row) that have a missing value"
    )
    duplicate_row_count: int


class NumericStats(BaseModel):
    count: int
    mean: float | None
    std: float | None
    min: float | None
    p25: float | None
    median: float | None
    p75: float | None
    max: float | None


class DatetimeStats(BaseModel):
    min: str
    max: str


class ValueCount(BaseModel):
    value: str
    count: int


class CategoricalInfo(BaseModel):
    unique_count: int
    top_values: list[ValueCount]
    values_truncated: bool = Field(description="True when there are more unique values than listed")


class ColumnProfile(BaseModel):
    name: str
    data_type: DataType
    missing_count: int
    missing_percentage: float
    unique_count: int
    numeric_stats: NumericStats | None = None
    datetime_stats: DatetimeStats | None = None
    categorical: CategoricalInfo | None = None


class ColumnMappingInfo(BaseModel):
    """Which column plays which business role (see utils/column_mapper.py)."""

    resolved: dict[str, str]
    ambiguous: dict[str, list[str]]
    missing: list[str]


class DataframeProfile(BaseModel):
    row_count: int
    column_count: int
    column_names: list[str]
    duplicate_row_count: int
    total_missing_cells: int
    columns: list[ColumnProfile]
    column_mapping: ColumnMappingInfo
    warnings: list[str]


class DatasetProfile(DataframeProfile):
    """Response of GET /datasets/{dataset_id}/profile."""

    dataset_id: UUID
    filename: str
    source_format: SourceFormat