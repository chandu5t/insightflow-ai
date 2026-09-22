"""Fixtures for PostgreSQL INTEGRATION tests (a real database, not SQLite).

These need `docker compose up -d db` running first. If PostgreSQL is not reachable,
the fixture SKIPS the test with a clear message rather than failing -- so `pytest -q`
for the full suite passes without Docker running; only the Postgres-specific tests skip.
"""

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from app.core.config import get_settings
from app.db import models  # noqa: F401  (registers the tables on Base.metadata)
from app.db.base import Base
from app.db.engine import get_engine, get_session_factory


@pytest.fixture(scope="session")
def postgres_settings():
    return get_settings()


@pytest.fixture(scope="session")
def postgres_engine(postgres_settings):
    engine = get_engine(postgres_settings.database_url)
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except OperationalError:
        pytest.skip(
            "PostgreSQL is not reachable. Start it first: "
            "docker compose up -d db (from the project root)."
        )
    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture
def postgres_session_factory(postgres_engine, postgres_settings):
    return get_session_factory(postgres_settings)


@pytest.fixture
def clean_postgres_tables(postgres_engine):
    """Empties both tables before AND after the test, so tests never see leftover rows."""

    def _clean() -> None:
        with postgres_engine.begin() as connection:
            connection.execute(text("DELETE FROM analysis_history"))
            connection.execute(text("DELETE FROM datasets"))

    _clean()
    yield
    _clean()