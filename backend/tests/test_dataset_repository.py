"""Tests for the JSON file repository."""

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from app.core.errors import AppError, ErrorCode
from app.schemas.dataset_schema import DatasetMetadata
from app.services.dataset_repository import JsonDatasetRepository


def make_metadata() -> DatasetMetadata:
    return DatasetMetadata(
        dataset_id=uuid4(),
        filename="sales.csv",
        source_format="csv",
        row_count=2,
        column_count=2,
        column_names=["a", "b"],
        original_size_bytes=20,
        uploaded_at=datetime.now(UTC),
    )


def test_repository_creates_its_folder(tmp_path: Path) -> None:
    folder = tmp_path / "new" / "uploads"

    JsonDatasetRepository(folder)

    assert folder.is_dir()


def test_save_and_get_round_trip(tmp_path: Path) -> None:
    repository = JsonDatasetRepository(tmp_path)
    metadata = make_metadata()
    temp_path = repository.new_temp_path(metadata.dataset_id)
    temp_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    repository.save(metadata, temp_path)

    assert repository.get(metadata.dataset_id) == metadata
    assert repository.get_csv_path(metadata.dataset_id).read_text(encoding="utf-8") == "a,b\n1,2\n3,4\n"
    assert not temp_path.exists()  # the temporary file was moved


def test_files_are_named_from_the_dataset_id_only(tmp_path: Path) -> None:
    repository = JsonDatasetRepository(tmp_path)
    metadata = make_metadata()
    temp_path = repository.new_temp_path(metadata.dataset_id)
    temp_path.write_text("a\n1\n", encoding="utf-8")

    repository.save(metadata, temp_path)

    names = sorted(path.name for path in tmp_path.iterdir())
    assert names == [f"{metadata.dataset_id}.csv", f"{metadata.dataset_id}.meta.json"]


def test_metadata_file_does_not_contain_any_file_path(tmp_path: Path) -> None:
    repository = JsonDatasetRepository(tmp_path)
    metadata = make_metadata()
    temp_path = repository.new_temp_path(metadata.dataset_id)
    temp_path.write_text("a\n1\n", encoding="utf-8")
    repository.save(metadata, temp_path)

    stored_text = (tmp_path / f"{metadata.dataset_id}.meta.json").read_text(encoding="utf-8")

    assert "file_path" not in stored_text
    assert str(tmp_path) not in stored_text


def test_unknown_dataset_raises_not_found(tmp_path: Path) -> None:
    repository = JsonDatasetRepository(tmp_path)

    with pytest.raises(AppError) as error:
        repository.get(uuid4())
    assert error.value.code == ErrorCode.DATASET_NOT_FOUND
    assert error.value.status_code == 404

    with pytest.raises(AppError):
        repository.get_csv_path(uuid4())