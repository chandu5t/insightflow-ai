"""Integration tests for PostgresDatasetRepository against a REAL PostgreSQL database.

Run `docker compose up -d db` first. These are automatically SKIPPED if Postgres is
not reachable (see tests/db_fixtures.py) -- `pytest -q` for the full suite never
requires Docker to pass.
"""

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from app.core.config import Settings
from app.core.errors import AppError, ErrorCode
from app.db.engine import get_session_factory
from app.schemas.dataset_schema import DatasetMetadata
from app.services.postgres_dataset_repository import PostgresDatasetRepository
from tests.db_fixtures import clean_postgres_tables, postgres_engine, postgres_session_factory, postgres_settings  # noqa: F401


def make_metadata(**overrides) -> DatasetMetadata:
    defaults = dict(
        dataset_id=uuid4(), filename="sales.csv", source_format="csv", row_count=13,
        column_count=7, column_names=["order_id", "product"], original_size_bytes=512,
        uploaded_at=datetime.now(UTC), status="uploaded", warnings=[],
    )
    defaults.update(overrides)
    return DatasetMetadata(**defaults)


@pytest.fixture
def repository(postgres_session_factory, tmp_path: Path, clean_postgres_tables):
    return PostgresDatasetRepository(postgres_session_factory, tmp_path)


def test_save_then_get_round_trips_the_metadata(repository: PostgresDatasetRepository) -> None:
    metadata = make_metadata()
    temp_path = repository.new_temp_path(metadata.dataset_id)
    temp_path.write_text("a,b\n1,2\n", encoding="utf-8")

    repository.save(metadata, temp_path)
    fetched = repository.get(metadata.dataset_id)

    assert fetched == metadata


def test_csv_file_is_still_on_disk_at_the_expected_path(
    repository: PostgresDatasetRepository, tmp_path: Path
) -> None:
    metadata = make_metadata()
    temp_path = repository.new_temp_path(metadata.dataset_id)
    temp_path.write_text("a,b\n1,2\n", encoding="utf-8")
    repository.save(metadata, temp_path)

    csv_path = repository.get_csv_path(metadata.dataset_id)

    assert csv_path == tmp_path / f"{metadata.dataset_id}.csv"
    assert csv_path.read_text(encoding="utf-8") == "a,b\n1,2\n"
    assert not temp_path.exists()  # the temp file was moved, not copied


def test_metadata_survives_a_fresh_repository_instance(
    postgres_session_factory, tmp_path: Path, clean_postgres_tables
) -> None:
    """Simulates an application restart: a brand-new repository object still finds the data."""
    metadata = make_metadata()
    first = PostgresDatasetRepository(postgres_session_factory, tmp_path)
    temp_path = first.new_temp_path(metadata.dataset_id)
    temp_path.write_text("a\n1\n", encoding="utf-8")
    first.save(metadata, temp_path)

    second = PostgresDatasetRepository(postgres_session_factory, tmp_path)

    assert second.get(metadata.dataset_id) == metadata


def test_unknown_dataset_raises_dataset_not_found(repository: PostgresDatasetRepository) -> None:
    with pytest.raises(AppError) as info:
        repository.get(uuid4())

    assert info.value.code == ErrorCode.DATASET_NOT_FOUND
    assert info.value.status_code == 404


def test_missing_csv_file_raises_dataset_not_found_even_if_the_row_exists(
    repository: PostgresDatasetRepository,
) -> None:
    metadata = make_metadata()
    temp_path = repository.new_temp_path(metadata.dataset_id)
    temp_path.write_text("a\n1\n", encoding="utf-8")
    repository.save(metadata, temp_path)
    repository.get_csv_path(metadata.dataset_id).unlink()  # simulate the file being deleted

    with pytest.raises(AppError) as info:
        repository.get_csv_path(metadata.dataset_id)

    assert info.value.code == ErrorCode.DATASET_NOT_FOUND


def test_saving_two_different_datasets_does_not_collide(repository: PostgresDatasetRepository) -> None:
    first, second = make_metadata(filename="a.csv"), make_metadata(filename="b.csv")
    for metadata in (first, second):
        temp_path = repository.new_temp_path(metadata.dataset_id)
        temp_path.write_text("a\n1\n", encoding="utf-8")
        repository.save(metadata, temp_path)

    assert repository.get(first.dataset_id).filename == "a.csv"
    assert repository.get(second.dataset_id).filename == "b.csv"


def test_a_database_failure_during_save_raises_database_error(
    postgres_session_factory, tmp_path: Path, clean_postgres_tables
) -> None:
    """A duplicate dataset_id (a primary-key violation) exercises the DB-failure path.

    In real use dataset_id is always a fresh UUID4 per upload, so this never happens in
    practice -- this test proves the ERROR HANDLING works, using the most reliable way to
    make PostgreSQL itself reject a write in a genuine integration test.
    """
    repository = PostgresDatasetRepository(postgres_session_factory, tmp_path)
    dataset_id = uuid4()
    metadata = make_metadata(dataset_id=dataset_id)
    first_temp = repository.new_temp_path(dataset_id)
    first_temp.write_text("a\n1\n", encoding="utf-8")
    repository.save(metadata, first_temp)  # succeeds

    second_temp = repository.new_temp_path(dataset_id)
    second_temp.write_text("a\n2\n", encoding="utf-8")

    with pytest.raises(AppError) as info:
        repository.save(metadata, second_temp)  # same dataset_id -> primary key violation

    assert info.value.code == ErrorCode.DATABASE_ERROR
    assert info.value.status_code == 503


def test_unreachable_database_raises_database_error_not_a_raw_exception(tmp_path: Path) -> None:
    """Does NOT need db_fixtures / a running Postgres -- it deliberately points elsewhere."""
    unreachable_settings = Settings(
        gemini_api_key=None, storage_backend="postgres",
        postgres_host="127.0.0.1", postgres_port=59999,  # nothing listens here
    )
    repository = PostgresDatasetRepository(get_session_factory(unreachable_settings), tmp_path)

    with pytest.raises(AppError) as info:
        repository.get(uuid4())

    assert info.value.code == ErrorCode.DATABASE_ERROR
    assert info.value.status_code == 503