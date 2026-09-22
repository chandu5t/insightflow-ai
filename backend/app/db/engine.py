"""Engine and session-factory creation.

create_engine() and sessionmaker() are both LAZY: neither opens a real network
connection by itself. The first actual query is what connects. This is why
constructing a PostgresDatasetRepository (or calling get_session_factory) is always
safe even when PostgreSQL is not running -- see tests/test_storage_backend_selection.py.
"""

from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings


@lru_cache
def get_engine(database_url: str) -> Engine:
    """One Engine per URL, reused for the process lifetime.

    pool_pre_ping=True checks each pooled connection with a lightweight query before
    handing it out, transparently reconnecting if PostgreSQL was restarted -- this
    matters for the Module 6 persistence test (stop/start the containers without
    restarting the backend process, then query again).
    """
    return create_engine(database_url, pool_pre_ping=True, future=True)


def get_session_factory(settings: Settings) -> sessionmaker[Session]:
    """A sessionmaker bound to the engine for these settings. Calling it returns a new Session."""
    engine = get_engine(settings.database_url)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)