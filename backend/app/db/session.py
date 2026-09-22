"""FastAPI dependency for a request-scoped database session, and startup schema creation."""

import logging
from collections.abc import Generator

from fastapi import Depends
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.base import Base
from app.db.engine import get_engine, get_session_factory

logger = logging.getLogger(__name__)


def get_db_session(settings: Settings = Depends(get_settings)) -> Generator[Session, None, None]:
    """A new Session per request. Commits on success, rolls back on any exception, always closes.

    Not currently used by the two Module 6 repositories (they manage short-lived sessions
    internally, matching JsonDatasetRepository's simple synchronous method style). Provided
    ready-to-use for a future endpoint that needs direct database access, such as a
    GET /datasets/{id}/history endpoint in a later module. Tested directly in test_db_session.py.
    """
    session = get_session_factory(settings)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(settings: Settings) -> None:
    """Create the datasets/analysis_history tables if they do not already exist.

    Uses Base.metadata.create_all() -- idempotent (safe every startup) but, unlike Alembic,
    cannot ALTER an existing table. With only two tables and no production data yet, this is
    the simpler choice for this stage (see D-057). If a column is ever added or renamed later,
    a manual migration -- or introducing Alembic then -- becomes necessary.
    """
    from app.db import models  # noqa: F401  (import so Base.metadata knows about the tables)

    engine = get_engine(settings.database_url)
    Base.metadata.create_all(bind=engine)
    logger.info("PostgreSQL tables ensured (datasets, analysis_history).")