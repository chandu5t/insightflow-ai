"""Integration tests for the analysis-history repositories against a real database."""

from datetime import UTC, datetime
from uuid import uuid4

from app.db.models import AnalysisHistoryRow, DatasetRow
from app.schemas.query_schema import ClassifierInfo, QueryPlan, QueryResponse, ValidationInfo
from app.services.analysis_history_repository import (
    NullAnalysisHistoryRepository,
    PostgresAnalysisHistoryRepository,
)
from tests.db_fixtures import clean_postgres_tables, postgres_engine, postgres_session_factory, postgres_settings  # noqa: F401


def make_response(**overrides) -> QueryResponse:
    defaults = dict(
        status="success", question="What is the total revenue?", dataset_id=uuid4(),
        classifier=ClassifierInfo(used="rule_based", fallback_reason="GEMINI_NOT_CONFIGURED"),
        validation=ValidationInfo(status="passed"),
    )
    defaults.update(overrides)
    return QueryResponse(**defaults)


def _insert_dataset(session_factory, dataset_id) -> None:
    with session_factory() as session, session.begin():
        session.add(DatasetRow(
            dataset_id=dataset_id, filename="a.csv", source_format="csv", row_count=1,
            column_count=1, column_names=["a"], original_size_bytes=1,
            uploaded_at=datetime.now(UTC), status="uploaded", warnings=[],
        ))


def test_record_inserts_a_readable_row(postgres_session_factory, clean_postgres_tables) -> None:
    dataset_id = uuid4()
    _insert_dataset(postgres_session_factory, dataset_id)
    response = make_response(dataset_id=dataset_id)
    repository = PostgresAnalysisHistoryRepository(postgres_session_factory)

    repository.record(dataset_id=dataset_id, question=response.question, response=response)

    with postgres_session_factory() as session:
        rows = session.query(AnalysisHistoryRow).filter_by(dataset_id=dataset_id).all()
    assert len(rows) == 1
    assert rows[0].question == "What is the total revenue?"
    assert rows[0].status == "success"
    assert rows[0].response["status"] == "success"


def test_record_stores_the_intent_from_the_query_plan(postgres_session_factory, clean_postgres_tables) -> None:
    dataset_id = uuid4()
    _insert_dataset(postgres_session_factory, dataset_id)
    response = make_response(
        dataset_id=dataset_id, query_plan=QueryPlan(intent="total_revenue", tool_name="revenue")
    )
    repository = PostgresAnalysisHistoryRepository(postgres_session_factory)

    repository.record(dataset_id=dataset_id, question=response.question, response=response)

    with postgres_session_factory() as session:
        row = session.query(AnalysisHistoryRow).filter_by(dataset_id=dataset_id).one()
    assert row.intent == "total_revenue"


def test_record_never_raises_when_the_dataset_foreign_key_is_missing(
    postgres_session_factory, clean_postgres_tables
) -> None:
    """No matching dataset row exists, so the foreign key rejects the insert -- record()
    must swallow this (logging it) rather than breaking the caller's actual analysis response.
    """
    response = make_response()  # dataset_id was never inserted into `datasets`
    repository = PostgresAnalysisHistoryRepository(postgres_session_factory)

    repository.record(dataset_id=response.dataset_id, question=response.question, response=response)  # must not raise

    with postgres_session_factory() as session:
        assert session.query(AnalysisHistoryRow).filter_by(dataset_id=response.dataset_id).count() == 0


def test_null_repository_does_nothing_and_never_touches_the_database() -> None:
    NullAnalysisHistoryRepository().record(uuid4(), "any question", make_response())  # no database needed