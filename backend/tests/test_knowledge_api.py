"""API tests for POST /knowledge/search."""

from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

from app.api.knowledge_routes import get_knowledge_search_service
from app.core.config import Settings, get_settings
from app.main import app
from app.services.knowledge_seed_service import seed_knowledge_base
from tests.db_fixtures import clean_postgres_tables, postgres_engine, postgres_session_factory, postgres_settings  # noqa: F401
from tests.fake_embeddings import FakeEmbeddings

MakeClient = Callable[..., TestClient]


@pytest.fixture
def knowledge_client(make_client: MakeClient, postgres_settings, postgres_session_factory, clean_postgres_tables):
    from sqlalchemy import text as sql_text
    with postgres_settings and postgres_session_factory() as session, session.begin():
        session.execute(sql_text("DELETE FROM knowledge_documents"))

    embeddings = FakeEmbeddings(
        vectors={
            "average order value": [1.0] + [0.0] * 767,
        },
        dimensions=768,
    )
    seed_knowledge_base(postgres_session_factory, embeddings)

    test_settings = Settings(
        gemini_api_key=None, storage_backend="postgres",
        postgres_host=postgres_settings.postgres_host, postgres_port=postgres_settings.postgres_port,
        postgres_user=postgres_settings.postgres_user, postgres_password=postgres_settings.postgres_password,
        postgres_db=postgres_settings.postgres_db,
    )
    client = make_client(**{})
    app.dependency_overrides[get_settings] = lambda: test_settings
    app.dependency_overrides[get_knowledge_search_service] = lambda: __import__(
        "app.services.knowledge_search_service", fromlist=["KnowledgeSearchService"]
    ).KnowledgeSearchService(
        postgres_session_factory, embeddings, top_k_limit=3, threshold=0.6
    )
    yield client
    app.dependency_overrides.clear()


def test_search_returns_a_200_with_results(knowledge_client: TestClient) -> None:
    response = knowledge_client.post("/knowledge/search", json={"query": "average order value"})

    body = response.json()
    assert response.status_code == 200
    assert body["query"] == "average order value"
    assert body["fallback"] is False
    assert body["results"][0]["slug"] == "average-order-value"
    assert "similarity" in body["results"][0]


def test_empty_query_is_a_422(knowledge_client: TestClient) -> None:
    response = knowledge_client.post("/knowledge/search", json={"query": ""})
    assert response.status_code == 422


def test_top_k_above_three_is_a_422(knowledge_client: TestClient) -> None:
    response = knowledge_client.post("/knowledge/search", json={"query": "revenue", "top_k": 5})
    assert response.status_code == 422


def test_json_storage_mode_returns_503(make_client: MakeClient) -> None:
    client = make_client()  # default storage_backend="json"

    response = client.post("/knowledge/search", json={"query": "average order value"})

    assert response.status_code == 503
    assert response.json()["code"] == "KNOWLEDGE_BASE_UNAVAILABLE"