"""Persists one row per analyzed question. See docs/DECISIONS.md D-057 for what is/isn't recorded."""

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.models import AnalysisHistoryRow
from app.schemas.query_schema import QueryResponse

logger = logging.getLogger(__name__)


class AnalysisHistoryRepository(Protocol):
    def record(self, dataset_id: UUID, question: str, response: QueryResponse) -> None: ...


class PostgresAnalysisHistoryRepository:
    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory

    def record(self, dataset_id: UUID, question: str, response: QueryResponse) -> None:
        """Never raises: a history-write failure must not turn a good answer into an error.

        Database errors (including a missing dataset foreign key, which cannot happen in
        normal use since the dataset was just looked up successfully by answer_question())
        are logged and swallowed here.
        """
        intent = response.query_plan.intent if response.query_plan else None
        try:
            with self._session_factory() as session, session.begin():
                session.add(
                    AnalysisHistoryRow(
                        dataset_id=dataset_id,
                        question=question,
                        intent=intent,
                        status=response.status,
                        response=response.model_dump(mode="json"),
                        created_at=datetime.now(UTC),
                    )
                )
        except SQLAlchemyError:
            logger.exception("Failed to save analysis history for dataset=%s", dataset_id)


class NullAnalysisHistoryRepository:
    """Used when STORAGE_BACKEND=json. Recording is a no-op, so JSON mode needs no database at all."""

    def record(self, dataset_id: UUID, question: str, response: QueryResponse) -> None:
        return None