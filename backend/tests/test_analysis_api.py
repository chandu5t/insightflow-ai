"""API tests for POST /analysis/query. Gemini is always replaced by a fake. No network is used."""

from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

from app.api.analysis_routes import get_llm_client
from app.core.config import Settings
from app.core.errors import ErrorCode, ToolError
from app.main import app
from app.schemas.query_schema import ValidationCheck, ValidationInfo
from app.services import query_service
from app.services.gemini_client import GeminiClient, GeminiNotConfiguredError, GeminiRequestError
from app.services.query_dispatcher import TOOL_HANDLERS
from tests.fakes import FakeLlm, plan_json
from tests.frames import SALES_PATH
from tests.helpers import csv_upload

UNKNOWN_ID = "123e4567-e89b-42d3-a456-426614174000"
MakeClient = Callable[..., TestClient]
BODY_KEYS = {
    "status", "question", "dataset_id", "classifier", "query_plan", "tool_used", "result",
    "explanation", "calculation_method", "assumptions", "validation", "message", "error",
}


@pytest.fixture
def analysis(make_client: MakeClient):
    """Build a client whose Gemini is replaced by `llm` (default: Gemini is not configured)."""

    def _build(llm=None, **settings) -> TestClient:
        client = make_client(**settings)
        fake = llm if llm is not None else FakeLlm(GeminiNotConfiguredError("no key"))
        app.dependency_overrides[get_llm_client] = lambda: fake
        return client

    return _build


def upload(client: TestClient, content: bytes | None = None, name: str = "module3_sales.csv") -> str:
    data = SALES_PATH.read_bytes() if content is None else content
    response = client.post("/datasets/upload", files=csv_upload(data, name))
    assert response.status_code == 201, response.text
    return response.json()["dataset_id"]


def ask(client: TestClient, dataset_id: str, question: str):
    return client.post("/analysis/query", json={"dataset_id": dataset_id, "question": question})


# ---- success: Gemini path and rule-based path ----------------------------------------------
def test_success_with_a_gemini_plan(analysis) -> None:
    llm = FakeLlm(plan_json(intent="rank", tool_name="ranking", metric="revenue", aggregation="sum",
                            group_by="region", sort_order="desc", limit=1))
    client = analysis(llm)

    response = ask(client, upload(client), "Which region generated the highest revenue?")

    body = response.json()
    assert response.status_code == 200 and set(body) == BODY_KEYS
    assert body["status"] == "success" and body["error"] is None and body["message"] is None
    assert body["classifier"] == {"used": "gemini", "model": "fake-gemini", "fallback_reason": None}
    assert body["tool_used"] == "ranking_tool"
    assert body["result"]["items"][0] == {"rank": 1, "group": "North", "value": 106500.0, "row_count": 4}
    assert "North" in body["explanation"] and "₹106,500.00" in body["explanation"]
    assert body["validation"]["status"] == "passed"
    assert body["calculation_method"] == "derived_quantity_times_unit_price"
    assert len(llm.calls) == 1


EXAMPLES = [
    ("What is the total revenue?", lambda r: r["value"] == 256000.0),
    ("Which region generated the highest revenue?",
     lambda r: r["items"][0]["group"] == "North" and r["items"][0]["value"] == 106500.0),
    ("What is the average unit price?", lambda r: r["value"] == pytest.approx(13807.692308, abs=1e-6)),
    ("What is the maximum quantity?", lambda r: r["value"] == 5.0),
    ("Show revenue by region.",
     lambda r: {g["group"]: g["value"] for g in r["groups"]} == {"East": 72000.0, "North": 106500.0, "South": 75000.0}),
    ("Which products have the highest revenue?",
     lambda r: [i["group"] for i in r["items"]] == ["Laptop", "Monitor", "Headset", "Keyboard", "Mouse"]),
    ("How many records are present?", lambda r: r["value"] == 13),
    ("Are there missing values?",
     lambda r: r["total_missing_cells"] == 1 and r["columns_with_missing"] == ["region"]),
    ("What is the average order value?", lambda r: r["value"] == 25600.0),
]


@pytest.mark.parametrize(("question", "check"), EXAMPLES)
def test_supported_questions_are_answered_by_the_fallback(analysis, question: str, check) -> None:
    client = analysis()

    response = ask(client, upload(client), question)

    body = response.json()
    assert response.status_code == 200 and body["status"] == "success", body
    assert body["classifier"]["used"] == "rule_based"
    assert body["classifier"]["fallback_reason"] == "GEMINI_NOT_CONFIGURED"
    assert check(body["result"]), body["result"]
    assert body["explanation"] and body["validation"]["status"] == "passed"


def test_a_real_client_without_a_key_falls_back_without_network(analysis) -> None:
    client = analysis(GeminiClient(api_key="", model="m", timeout_seconds=1))

    body = ask(client, upload(client), "What is the total revenue?").json()

    assert body["status"] == "success" and body["classifier"]["fallback_reason"] == "GEMINI_NOT_CONFIGURED"


def test_gemini_failure_and_bad_output_fall_back(analysis) -> None:
    client = analysis(FakeLlm(GeminiRequestError("timeout")))
    body = ask(client, upload(client), "What is the total revenue?").json()
    assert body["status"] == "success"
    assert body["classifier"] == {"used": "rule_based", "model": None, "fallback_reason": "GEMINI_REQUEST_FAILED"}

    client = analysis(FakeLlm("this is not json"))
    body = ask(client, upload(client), "What is the total revenue?").json()
    assert body["status"] == "success" and body["classifier"]["fallback_reason"] == "INVALID_QUERY_PLAN"


def test_the_assumptions_and_currency_note(analysis) -> None:
    client = analysis()

    body = ask(client, upload(client), "What is the total revenue?").json()

    joined = " ".join(body["assumptions"])
    assert "No filters were applied" in joined and "₹" in joined


def test_the_currency_symbol_comes_from_settings(analysis) -> None:
    client = analysis(currency_symbol="$")

    body = ask(client, upload(client), "What is the total revenue?").json()

    assert "$256,000.00" in body["explanation"]


def test_several_questions_in_a_row(analysis) -> None:
    client = analysis()
    dataset_id = upload(client)

    values = [ask(client, dataset_id, q).json()["result"]["value"]
              for q in ("What is the total revenue?", "What is the maximum quantity?", "How many records are present?")]

    assert values == [256000.0, 5.0, 13]


# ---- unsupported and definition -----------------------------------------------------------------
def test_definition_question_returns_the_module_7_fallback(analysis) -> None:
    client = analysis()

    response = ask(client, upload(client), "What is revenue?")

    body = response.json()
    assert response.status_code == 200 and body["status"] == "unsupported"
    assert body["result"] is None and body["explanation"] is None
    assert "Module 7" in body["message"]
    assert body["error"]["code"] == "UNSUPPORTED_QUESTION"
    assert body["error"]["details"] == {"reason": "definition_not_available"}
    assert body["query_plan"]["intent"] == "definition"


def test_unsupported_question(analysis) -> None:
    client = analysis()

    response = ask(client, upload(client), "Predict next month's revenue")

    body = response.json()
    assert response.status_code == 200 and body["status"] == "unsupported"
    assert body["error"]["details"] == {"reason": "unsupported_question"}
    assert "Forecasts" in body["message"] and "You can ask for" in body["message"]
    assert body["result"] is None


# ---- insufficient data --------------------------------------------------------------------------
def test_missing_revenue_columns(analysis) -> None:
    client = analysis()
    dataset_id = upload(client, b"product,region\nPen,North\n", "no_revenue.csv")

    response = ask(client, dataset_id, "What is the total revenue?")

    body = response.json()
    assert response.status_code == 200 and body["status"] == "insufficient_data"
    assert body["result"] is None and body["explanation"] is None and body["message"]
    assert body["error"]["code"] == "INSUFFICIENT_DATA"
    assert body["error"]["details"]["missing_roles"] == ["revenue", "quantity", "unit_price"]


def test_missing_group_column(analysis) -> None:
    client = analysis()
    dataset_id = upload(client, b"qty,price\n2,10\n", "no_region.csv")

    body = ask(client, dataset_id, "Which region generated the highest revenue?").json()

    assert body["status"] == "insufficient_data" and body["error"]["code"] == "MISSING_COLUMN"
    assert body["error"]["details"]["missing_roles"] == ["region"]


def test_ambiguous_and_invalid_columns(analysis) -> None:
    client = analysis()

    ambiguous = upload(client, b"revenue,sales\n1,2\n", "ambiguous.csv")
    invalid = upload(client, b"revenue\nabc\n", "invalid.csv")

    body = ask(client, ambiguous, "What is the total revenue?").json()
    assert body["status"] == "insufficient_data" and body["error"]["code"] == "AMBIGUOUS_COLUMN"
    body = ask(client, invalid, "What is the total revenue?").json()
    assert body["status"] == "insufficient_data" and body["error"]["code"] == "INVALID_NUMERIC_VALUES"


def test_a_gemini_plan_for_an_unknown_column_is_insufficient_data(analysis) -> None:
    llm = FakeLlm(plan_json(intent="group", tool_name="grouping", metric="revenue", aggregation="sum", group_by="zzz"))
    client = analysis(llm)

    body = ask(client, upload(client), "Revenue by zzz").json()

    assert body["status"] == "insufficient_data" and body["error"]["code"] == "MISSING_COLUMN"
    assert body["classifier"]["used"] == "gemini"


# ---- tool failure, crash, failed validation -------------------------------------------------------
def test_tool_failure_is_insufficient_data(analysis, monkeypatch) -> None:
    def failing(frame, plan):
        raise ToolError(ErrorCode.INSUFFICIENT_DATA, "Not enough data.", {"why": "test"})

    monkeypatch.setitem(TOOL_HANDLERS, "revenue", failing)
    client = analysis()

    body = ask(client, upload(client), "What is the total revenue?").json()

    assert body["status"] == "insufficient_data"
    assert body["error"] == {"code": "INSUFFICIENT_DATA", "message": "Not enough data.", "details": {"why": "test"}}


def test_unexpected_crash_returns_the_standard_500(analysis, monkeypatch) -> None:
    def crash(frame, plan):
        raise RuntimeError("secret internal detail")

    monkeypatch.setitem(TOOL_HANDLERS, "revenue", crash)
    client = analysis()
    dataset_id = upload(client)
    crash_client = TestClient(app, raise_server_exceptions=False)

    response = ask(crash_client, dataset_id, "What is the total revenue?")

    assert response.status_code == 500 and response.json()["code"] == "INTERNAL_ERROR"
    assert "secret" not in response.text


def test_failed_validation_returns_status_error_and_hides_the_result(analysis, monkeypatch) -> None:
    failed = ValidationInfo(status="failed", checks=[ValidationCheck(name="row_accounting", passed=False, detail="x")])
    monkeypatch.setattr(query_service, "validate_result", lambda frame, plan, result: failed)
    client = analysis()

    response = ask(client, upload(client), "What is the total revenue?")

    body = response.json()
    assert response.status_code == 200 and body["status"] == "error"
    assert body["result"] is None and body["explanation"] is None
    assert body["error"]["code"] == "RESULT_VALIDATION_FAILED"
    assert body["error"]["details"] == {"failed_checks": ["row_accounting"]}
    assert body["validation"]["status"] == "failed"


# ---- request errors --------------------------------------------------------------------------------
def test_unknown_dataset_is_404(analysis) -> None:
    response = ask(analysis(), UNKNOWN_ID, "What is the total revenue?")

    assert response.status_code == 404
    assert response.json() == {"error": "Dataset not found.", "code": "DATASET_NOT_FOUND", "details": {}}


def test_invalid_dataset_id_is_422(analysis) -> None:
    response = ask(analysis(), "not-a-uuid", "What is the total revenue?")

    assert response.status_code == 422 and response.json()["code"] == "INVALID_DATASET_ID"


def test_empty_question_is_422(analysis) -> None:
    response = ask(analysis(), UNKNOWN_ID, "   ")

    assert response.status_code == 422 and response.json()["code"] == "EMPTY_QUESTION"


def test_too_long_question_is_422(analysis) -> None:
    response = ask(analysis(max_question_length=20), UNKNOWN_ID, "x" * 21)

    assert response.status_code == 422 and response.json()["code"] == "QUESTION_TOO_LONG"
    assert response.json()["details"] == {"max_length": 20}


def test_bad_bodies_and_methods(analysis) -> None:
    client = analysis()

    assert client.post("/analysis/query", json={}).json()["code"] == "VALIDATION_ERROR"
    extra = client.post("/analysis/query", json={"dataset_id": UNKNOWN_ID, "question": "q", "sql": "drop"})
    assert extra.status_code == 422 and extra.json()["code"] == "VALIDATION_ERROR"
    wrong = client.get("/analysis/query")
    assert wrong.status_code == 405 and wrong.json()["code"] == "METHOD_NOT_ALLOWED"


# ---- security and privacy ----------------------------------------------------------------------------
def test_the_api_key_never_appears_in_responses_or_settings_text(analysis) -> None:
    client = analysis(FakeLlm(GeminiRequestError("HTTPError")), gemini_api_key="sk-SECRET-123")

    response = ask(client, upload(client), "What is the total revenue?")

    assert "sk-SECRET-123" not in response.text
    settings = Settings(gemini_api_key="sk-SECRET-123")
    assert "sk-SECRET-123" not in repr(settings) and "sk-SECRET-123" not in str(settings)


def test_gemini_receives_column_names_but_no_data_values(analysis) -> None:
    llm = FakeLlm(plan_json())
    client = analysis(llm)

    ask(client, upload(client), "What is the total revenue?")

    prompt = llm.calls[0]["user_prompt"]
    assert "unit_price" in prompt and "region" in prompt
    for value in ("Laptop", "Headset", "Keyboard", "50000"):
        assert value not in prompt