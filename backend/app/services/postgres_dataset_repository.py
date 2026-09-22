"""PostgreSQL-backed dataset repository (Module 6).

CSV files still live on local disk, in exactly the same {dataset_id}.csv / .csv.tmp
layout as JsonDatasetRepository (see D-057) -- only the METADATA moves to PostgreSQL.
Implements the same DatasetRepository Protocol, so dataset_service.py and
profiling_service.py need no changes at all.
"""

import os
from collections.abc import Callable
from pathlib import Path
from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.errors import AppError, ErrorCode
from app.db.models import DatasetRow
from app.schemas.dataset_schema import DatasetMetadata


class PostgresDatasetRepository:
    def __init__(self, session_factory: Callable[[], Session], base_dir: Path) -> None:
        self._session_factory = session_factory
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ---- these two never touch the database: the CSV always lives on disk -----------
    def _csv_path(self, dataset_id: UUID) -> Path:
        return self.base_dir / f"{dataset_id}.csv"

    def new_temp_path(self, dataset_id: UUID) -> Path:
        return self.base_dir / f"{dataset_id}.csv.tmp"

    def save(self, metadata: DatasetMetadata, temp_csv_path: Path) -> None:
        """Move the finished CSV into place, then insert the metadata row.

        If the database insert fails, the just-moved CSV is removed, so a dataset never
        exists as a file with no matching database row. (Note: dataset_id is always a
        freshly generated UUID4 per upload, so a genuine collision never happens in
        practice -- test_a_database_failure_during_save_raises_database_error exercises
        this path by deliberately reusing an id, which is why that one test's cleanup
        also removes the file from the FIRST, otherwise-successful save. This is an
        accepted, documented trade-off, not something Module 6 needs to solve for.)
        """
        final_csv_path = self._csv_path(metadata.dataset_id)
        os.replace(temp_csv_path, final_csv_path)

        try:
            with self._session_factory() as session, session.begin():
                session.add(
                    DatasetRow(
                        dataset_id=metadata.dataset_id,
                        filename=metadata.filename,
                        source_format=metadata.source_format,
                        row_count=metadata.row_count,
                        column_count=metadata.column_count,
                        column_names=metadata.column_names,
                        original_size_bytes=metadata.original_size_bytes,
                        uploaded_at=metadata.uploaded_at,
                        status=metadata.status,
                        warnings=metadata.warnings,
                    )
                )
        except SQLAlchemyError as exc:
            final_csv_path.unlink(missing_ok=True)
            raise AppError(
                ErrorCode.DATABASE_ERROR,
                "The dataset could not be saved because the database is unavailable.",
                status_code=503,
            ) from exc

    def get(self, dataset_id: UUID) -> DatasetMetadata:
        try:
            with self._session_factory() as session:
                row = session.get(DatasetRow, dataset_id)
        except SQLAlchemyError as exc:
            raise AppError(
                ErrorCode.DATABASE_ERROR,
                "The dataset could not be retrieved because the database is unavailable.",
                status_code=503,
            ) from exc
        if row is None:
            raise self._not_found()
        return DatasetMetadata(
            dataset_id=row.dataset_id,
            filename=row.filename,
            source_format=row.source_format,
            row_count=row.row_count,
            column_count=row.column_count,
            column_names=row.column_names,
            original_size_bytes=row.original_size_bytes,
            uploaded_at=row.uploaded_at,
            status=row.status,
            warnings=row.warnings,
        )

    def get_csv_path(self, dataset_id: UUID) -> Path:
        csv_path = self._csv_path(dataset_id)
        if not csv_path.is_file():
            raise self._not_found()
        return csv_path

    @staticmethod
    def _not_found() -> AppError:
        return AppError(ErrorCode.DATASET_NOT_FOUND, "Dataset not found.", status_code=404)