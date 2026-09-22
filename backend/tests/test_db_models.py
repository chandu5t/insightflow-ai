"""Tests for the SQLAlchemy model definitions themselves -- no live database needed."""

import uuid
from datetime import UTC, datetime

from app.db.models import AnalysisHistoryRow, DatasetRow


def test_dataset_row_table_name_and_columns() -> None:
    assert DatasetRow.__tablename__ == "datasets"
    columns = {column.name for column in DatasetRow.__table__.columns}
    assert columns == {
        "dataset_id", "filename", "source_format", "row_count", "column_count",
        "column_names", "original_size_bytes", "uploaded_at", "status", "warnings", "created_at",
    }
    assert list(DatasetRow.__table__.primary_key.columns.keys()) == ["dataset_id"]


def test_analysis_history_row_table_name_and_columns() -> None:
    assert AnalysisHistoryRow.__tablename__ == "analysis_history"
    columns = {column.name for column in AnalysisHistoryRow.__table__.columns}
    assert columns == {"id", "dataset_id", "question", "intent", "status", "response", "created_at"}


def test_analysis_history_has_a_foreign_key_to_datasets() -> None:
    foreign_keys = list(AnalysisHistoryRow.__table__.columns["dataset_id"].foreign_keys)
    assert len(foreign_keys) == 1
    assert foreign_keys[0].column.table.name == "datasets"


def test_indexed_columns_match_the_documented_query_patterns() -> None:
    indexed = {column.name for column in AnalysisHistoryRow.__table__.columns if column.index}
    assert indexed == {"dataset_id", "created_at"}
    assert not any(column.index for column in DatasetRow.__table__.columns if column.name != "dataset_id")


def test_models_can_be_instantiated_without_a_database() -> None:
    row = DatasetRow(
        dataset_id=uuid.uuid4(), filename="sales.csv", source_format="csv", row_count=10,
        column_count=3, column_names=["a", "b", "c"], original_size_bytes=100,
        uploaded_at=datetime.now(UTC), status="uploaded", warnings=[],
    )
    assert row.filename == "sales.csv"