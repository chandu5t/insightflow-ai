"""Tests for get_db_session's commit/rollback/close behaviour, against a real database."""

from datetime import UTC, datetime
from uuid import uuid4

from app.db.engine import get_session_factory
from app.db.models import DatasetRow
from app.db.session import get_db_session
from tests.db_fixtures import clean_postgres_tables, postgres_engine, postgres_settings  # noqa: F401


def _sample_row(dataset_id):
    return DatasetRow(
        dataset_id=dataset_id, filename="x.csv", source_format="csv", row_count=1,
        column_count=1, column_names=["a"], original_size_bytes=1,
        uploaded_at=datetime.now(UTC), status="uploaded", warnings=[],
    )


def test_get_db_session_commits_and_the_row_is_visible_afterwards(
    postgres_engine, postgres_settings, clean_postgres_tables
) -> None:
    """Drives the FastAPI generator dependency directly (the way FastAPI itself would,
    minus the framework), proving commit-on-success and close-on-exit both happen.
    """
    dataset_id = uuid4()
    generator = get_db_session(postgres_settings)
    session = next(generator)
    session.add(_sample_row(dataset_id))
    next(generator, None)  # advances past `yield`, triggering commit() + close()

    with get_session_factory(postgres_settings)() as check_session:
        assert check_session.get(DatasetRow, dataset_id) is not None


def test_get_db_session_rolls_back_on_error(
    postgres_engine, postgres_settings, clean_postgres_tables
) -> None:
    dataset_id = uuid4()
    generator = get_db_session(postgres_settings)
    session = next(generator)
    session.add(_sample_row(dataset_id))
    try:
        generator.throw(RuntimeError("simulated failure"))
    except RuntimeError:
        pass

    with get_session_factory(postgres_settings)() as check_session:
        assert check_session.get(DatasetRow, dataset_id) is None  # rolled back, never committed