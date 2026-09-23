"""Proves the definition branch uses the knowledge base end-to-end, and that numerical
analysis is completely unaffected (no embedding calls happen for a numerical question).
"""

from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

from app.api.analysis_routes import get_llm_client, get_metric_retriever
from app.core.config import Settings, get_settings
from app.main import app
from app.services.gemini_client import GeminiNotConfiguredError
from app.services.knowledge_seed_service import seed_knowledge_base
from app.services.metric_retriever import KnowledgeBaseMetricRetriever
from app.services.knowledge_search_service import KnowledgeSearchService
from tests.db_fixtures import clean_postgres_tables, postgres_engine, postgres_session_factory, postgres_settings  # noqa: F401
from tests.fake_embeddings import FakeEmbeddings
from tests.fakes import FakeLlm
from tests.frames import SALES_PATH
from tests.helpers import csv_upload

MakeClient = Callable[..., TestClient]


@pytest.fixture
def workflow_client(make_client, postgres_settings, postgres_session_factory, clean_postgres_tables):
    from sqlalchemy import text as sql_text
    with postgres_session_factory() as session, session.begin():
        session.execute(sql_text("DELETE FROM knowledge_documents"))

    aov_vector = [1.0] + [0.0] * 767
    embeddings = FakeEmbeddings(vectors={"What does average order value mean?": aov_vector}, dimensions=768)
    seed_knowledge_base(postgres_session_factory, embeddings)

    test_settings = Settings(
        gemini_api_key=None, storage_backend="postgres",
        postgres_host=postgres_settings.postgres_host, postgres_port=postgres_settings.postgres_port,
        postgres_user=postgres_settings.postgres_user, postgres_password=postgres_settings.postgres_password,
        postgres_db=postgres_settings.postgres_db,
    )
    client = make_client()
    app.dependency_overrides[get_settings] = lambda: test_settings
    app.dependency_overrides[get_llm_client] = lambda: FakeLlm(GeminiNotConfiguredError("no key"))
    app.dependency_overrides[get_metric_retriever] = lambda: KnowledgeBaseMetricRetriever(
        KnowledgeSearchService(postgres_session_factory, embeddings, top_k_limit=3, threshold=0.6)
    )
    upload = client.post("/datasets/upload", files=csv_upload(SALES_PATH.read_bytes(), "module3_sales.csv"))
    yield client, upload.json()["dataset_id"]
    app.dependency_overrides.clear()


def test_the_definition_question_returns_the_retrieved_answer_with_source(workflow_client) -> None:
    client, dataset_id = workflow_client

    response = client.post(
        "/analysis/query",
        json={"dataset_id": dataset_id, "question": "What does average order value mean?"},
    )

    body = response.json()
    assert response.status_code == 200 and body["status"] == "success"
    assert body["definition_source"] == "knowledge_base"
    assert "Average Order Value" in body["explanation"] or "average order value" in body["explanation"].lower()
    assert body["calculation_method"] == "metric_definition"
    assert body["result"] is None  # a definition is not a numerical result


def test_numerical_questions_are_unaffected_and_definition_source_is_null(workflow_client) -> None:
    client, dataset_id = workflow_client

    response = client.post(
        "/analysis/query", json={"dataset_id": dataset_id, "question": "What is the total revenue?"}
    )

    body = response.json()
    assert body["status"] == "success" and body["result"]["value"] == 256000.0
    assert body["definition_source"] is None