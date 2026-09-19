"""Storage of datasets on the local disk.

Files created for one dataset (inside the upload folder):
    {dataset_id}.csv         the clean CSV made from the upload
    {dataset_id}.meta.json   the metadata (see DatasetMetadata)

Module 6 will add a PostgreSQL version of this class. The rest of the code only
depends on the DatasetRepository interface, so nothing else needs to change.
"""

import os
from pathlib import Path
from typing import Protocol
from uuid import UUID

from app.core.errors import AppError, ErrorCode
from app.schemas.dataset_schema import DatasetMetadata


class DatasetRepository(Protocol):
    """The methods every storage implementation must provide."""

    def new_temp_path(self, dataset_id: UUID) -> Path: ...

    def save(self, metadata: DatasetMetadata, temp_csv_path: Path) -> None: ...

    def get(self, dataset_id: UUID) -> DatasetMetadata: ...

    def get_csv_path(self, dataset_id: UUID) -> Path: ...


class JsonDatasetRepository:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # File names are built only from a UUID object, never from user text.
    def _csv_path(self, dataset_id: UUID) -> Path:
        return self.base_dir / f"{dataset_id}.csv"

    def _metadata_path(self, dataset_id: UUID) -> Path:
        return self.base_dir / f"{dataset_id}.meta.json"

    def new_temp_path(self, dataset_id: UUID) -> Path:
        """A temporary file in the same folder, so the final move is a quick rename."""
        return self.base_dir / f"{dataset_id}.csv.tmp"

    def save(self, metadata: DatasetMetadata, temp_csv_path: Path) -> None:
        """Move the finished CSV into place, then write the metadata."""
        os.replace(temp_csv_path, self._csv_path(metadata.dataset_id))

        metadata_path = self._metadata_path(metadata.dataset_id)
        temp_metadata_path = metadata_path.with_name(metadata_path.name + ".tmp")
        temp_metadata_path.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
        os.replace(temp_metadata_path, metadata_path)

    def get(self, dataset_id: UUID) -> DatasetMetadata:
        metadata_path = self._metadata_path(dataset_id)
        if not metadata_path.is_file():
            raise self._not_found()
        return DatasetMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))

    def get_csv_path(self, dataset_id: UUID) -> Path:
        csv_path = self._csv_path(dataset_id)
        if not csv_path.is_file():
            raise self._not_found()
        return csv_path

    @staticmethod
    def _not_found() -> AppError:
        return AppError(ErrorCode.DATASET_NOT_FOUND, "Dataset not found.", status_code=404)