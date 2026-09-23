"""API tests that prove POST /analysis/query is unchanged after the workflow refactor."""

from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

from app.api.analysis_routes import get_explainer, get_llm_client, get_metric_retriever
from app.main import app
from app.services.gemini_client import GeminiNotConfiguredError
from app.services.metric_retriever import MetricDefinitionResult
from tests.fakes import FakeLlm
from tests.frames import SALES_PATH
from tests.helpers import csv_upload
from tests.workflow_helpers import FakeRetriever, SpyExplainer

MakeClient = Callable[..., TestClient]
BODY_KEYS = {
    "status", "question", "dataset_id", "classifier", "query_plan", "tool_used", "result",
    "explanation", "calculation_method", "assumptions", "validation", "message", "error", "definition_source",
}


@pytest.fixture
def workflow_client(make_client: MakeClient):
    """A client with a fake Gemini. Optional fake retriever and explainer."""

    def _build(retriever=None, explainer=None) -> tuple[TestClient, str]:
        client = make_client()
        fake_llm = FakeLlm(GeminiNotConfiguredError("no key"))
        app.dependency_overrides[get_llm_client] = lambda: fake_llm
        if retriever is not None:
            app.dependency_overrides[get_metric_retriever] = lambda: retriever
        if explainer is not None:
            app.dependency_overrides[get_explainer] = lambda: explainer
        response = client.post("/datasets/upload", files=csv_upload(SALES_PATH.read_bytes(), "module3_sales.csv"))
        assert response.status_code == 201, response.text
        return client, response.json()["dataset_id"]

    return _build


def ask(client: TestClient, dataset_id: str, question: str):
    return client.post("/analysis/query", json={"dataset_id": dataset_id, "question": question})


def test_the_response_contract_is_unchanged(workflow_client) -> None:
    client, dataset_id = workflow_client()

    response = ask(client, dataset_id, "What is the total revenue?")

    body = response.json()
    assert response.status_code == 200 and set(body) == BODY_KEYS
    assert body["status"] == "success" and body["result"]["value"] == 256000.0
    assert body["validation"]["status"] == "passed" and body["error"] is None
    assert body["calculation_method"] == "derived_quantity_times_unit_price"


def test_unsupported_and_definition_responses_are_unchanged(workflow_client) -> None:
    client, dataset_id = workflow_client()

    unsupported = ask(client, dataset_id, "Predict next month's revenue").json()
    definition = ask(client, dataset_id, "What is revenue?").json()

    assert unsupported["status"] == "unsupported" and unsupported["error"]["details"] == {"reason": "unsupported_question"}
    assert definition["status"] == "unsupported" and "Module 7" in definition["message"]
    assert definition["error"]["details"] == {"reason": "definition_not_available"}


def test_a_replaced_retriever_is_used_and_the_result_is_a_definition(workflow_client) -> None:
    found = MetricDefinitionResult(found=True, source="knowledge_base", is_stub=False, query="q",
                                   definition="Revenue is the money earned from sales.", message="Found.")
    retriever = FakeRetriever(found)
    client, dataset_id = workflow_client(retriever=retriever)

    body = ask(client, dataset_id, "What is revenue?").json()

    assert body["status"] == "success" and body["explanation"] == "Revenue is the money earned from sales."
    assert body["result"] is None and body["calculation_method"] == "metric_definition"
    assert retriever.calls == ["What is revenue?"]


def test_the_explainer_is_only_called_for_validated_results(workflow_client) -> None:
    explainer = SpyExplainer()
    client, dataset_id = workflow_client(explainer=explainer)

    ask(client, dataset_id, "Predict next month's revenue")
    assert explainer.calls == 0
    ask(client, dataset_id, "What is the total revenue?")
    assert explainer.calls == 1