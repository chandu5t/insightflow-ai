"""API tests: a hallucinated number in an explanation never reaches the response."""

from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

from app.api.analysis_routes import get_explainer, get_llm_client
from app.main import app
from app.services.gemini_client import GeminiNotConfiguredError
from tests.fakes import FakeLlm
from tests.frames import SALES_PATH
from tests.helpers import csv_upload

MakeClient = Callable[..., TestClient]
HALLUCINATION = "The total revenue is ₹256,000.00, which represents a 42% increase."


@pytest.fixture
def grounded_client(make_client: MakeClient):
    def _build(explainer) -> tuple[TestClient, str]:
        client = make_client()
        fake_llm = FakeLlm(GeminiNotConfiguredError("no key"))
        app.dependency_overrides[get_llm_client] = lambda: fake_llm
        app.dependency_overrides[get_explainer] = lambda: explainer
        response = client.post("/datasets/upload", files=csv_upload(SALES_PATH.read_bytes(), "module3_sales.csv"))
        return client, response.json()["dataset_id"]

    return _build


def ask(client: TestClient, dataset_id: str, question: str):
    return client.post("/analysis/query", json={"dataset_id": dataset_id, "question": question})


def test_a_hallucinated_number_returns_a_safe_error(grounded_client) -> None:
    client, dataset_id = grounded_client(lambda plan, result, symbol: HALLUCINATION)

    response = ask(client, dataset_id, "What is the total revenue?")

    body = response.json()
    assert response.status_code == 200 and body["status"] == "error"
    assert body["error"]["code"] == "EXPLANATION_NOT_GROUNDED"
    assert body["error"]["details"] == {"unsupported_numbers": ["42%"]}
    assert body["explanation"] is None and body["result"] is None and body["tool_used"] is None
    assert body["validation"]["status"] == "failed"
    grounding = [c for c in body["validation"]["checks"] if c["name"] == "number_grounding"][0]
    assert grounding["passed"] is False and grounding["code"] == "UNGROUNDED_NUMBER"
    assert "increase" not in response.text  # the bad explanation is never returned


def test_a_grounded_explanation_passes_and_reports_the_check(grounded_client) -> None:
    client, dataset_id = grounded_client(lambda plan, result, symbol: "The total revenue is ₹256,000.")

    body = ask(client, dataset_id, "What is the total revenue?").json()

    assert body["status"] == "success" and body["explanation"] == "The total revenue is ₹256,000."
    assert body["validation"]["status"] == "passed"
    checks = {c["name"]: c for c in body["validation"]["checks"]}
    assert checks["number_grounding"]["passed"] is True
    assert {"finite_numbers", "row_accounting", "operation_matches_plan", "aggregation_recomputed"} <= set(checks)


def test_the_real_template_is_never_rejected(grounded_client) -> None:
    from app.services.explainer import explain

    client, dataset_id = grounded_client(explain)

    for question in ["What is the total revenue?", "Which region generated the highest revenue?",
                     "Show revenue by region.", "Which products have the highest revenue?",
                     "What is the average order value?", "Are there missing values?",
                     "How many records are present?", "What is the maximum quantity?"]:
        assert ask(client, dataset_id, question).json()["status"] == "success", question