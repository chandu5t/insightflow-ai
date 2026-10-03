"""HTTP integration tests for the additive V2.2 planner endpoint."""

import json

from app.api.planner_routes import get_planner_model
from app.core.config import Settings
from app.main import app


class ApiPlanner:
    model_name = "api-fake"

    def __init__(self, reply):
        self.reply = reply
        self.calls = 0
        self.user_prompts = []

    def generate_json(self, *, system_prompt, user_prompt):
        self.calls += 1
        self.user_prompts.append(user_prompt)
        return self.reply


def test_plan_endpoint_returns_valid_plan_and_does_not_require_dataset(make_client):
    reply = json.dumps({
        "intent": "metric_definition", "reasoning_type": "definition", "steps": [], "unsupported_reason": None,
    })
    model = ApiPlanner(reply)
    client = make_client()
    app.dependency_overrides[get_planner_model] = lambda: model
    response = client.post("/analysis/plan", json={"question": "What does AOV mean?"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["valid"] is True
    assert body["plan"]["intent"] == "metric_definition"
    assert body["plan"]["steps"] == []
    assert body["metadata"]["planner_version"] == "v2.2"
    assert model.calls == 1


def test_plan_endpoint_returns_invalid_generated_plan_without_http_failure(make_client):
    reply = json.dumps({
        "intent": "total_revenue", "reasoning_type": "simple", "unsupported_reason": None,
        "steps": [{"step_id": "s1", "operation": "magic", "description": "bad", "inputs": [], "parameters": {}, "depends_on": []}],
    })
    model = ApiPlanner(reply)
    client = make_client()
    app.dependency_overrides[get_planner_model] = lambda: model
    response = client.post("/analysis/plan", json={"question": "Total revenue?"})
    assert response.status_code == 200
    assert response.json()["valid"] is False
    assert response.json()["validation_errors"][0]["code"] == "UNKNOWN_OPERATION"
    assert model.calls == 1


def test_plan_endpoint_supplies_caller_context_and_rejects_bad_requests(make_client):
    model = ApiPlanner(json.dumps({
        "intent": "unsupported_analysis", "reasoning_type": "unsupported", "steps": [],
        "unsupported_reason": "The requested feature is unsupported.",
    }))
    client = make_client()
    app.dependency_overrides[get_planner_model] = lambda: model
    response = client.post("/analysis/plan", json={
        "question": "Forecast next quarter", "dataset_context": {"columns": ["revenue"]},
        "metric_definitions": [{"name": "revenue", "definition": "Provided by caller"}],
    })
    assert response.status_code == 200 and response.json()["valid"] is True
    assert '"name":"revenue"' in model.user_prompts[0]
    assert client.post("/analysis/plan", json={}).status_code == 422


def test_plan_endpoint_reports_provider_failure_as_5xx(make_client):
    class Failed:
        model_name = "api-fake"
        def generate_json(self, *, system_prompt, user_prompt):
            raise RuntimeError("do not expose this detail")

    client = make_client()
    app.dependency_overrides[get_planner_model] = lambda: Failed()
    response = client.post("/analysis/plan", json={"question": "q"})
    assert response.status_code == 502
    assert "do not expose" not in response.text
    assert response.json()["code"] == "LLM_PROVIDER_ERROR"


def test_plan_endpoint_question_length_uses_existing_limit(make_client):
    client = make_client(max_question_length=4)
    response = client.post("/analysis/plan", json={"question": "12345"})
    assert response.status_code == 422
    assert response.json()["code"] == "QUESTION_TOO_LONG"
