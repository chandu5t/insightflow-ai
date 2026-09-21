"""Tests for the routing decisions. The `trace` shows which nodes really ran."""

import pytest

from app.schemas.query_schema import ValidationCheck, ValidationInfo
from app.services.metric_retriever import MetricDefinitionResult
from app.services.query_dispatcher import TOOL_HANDLERS
from app.utils.dataframe_utils import load_dataframe
from tests.fakes import FakeLlm, plan_json
from tests.frames import SALES_PATH, frame_from_csv
from tests.workflow_helpers import FakeRetriever, SpyExplainer, make_deps, run_workflow

FULL_PATH = ["classify", "route", "execute", "validate", "explain", "respond"]


@pytest.fixture(scope="module")
def sales():
    return load_dataframe(SALES_PATH)


def test_supported_question_takes_the_full_path(sales) -> None:
    final = run_workflow("What is the total revenue?", sales)

    assert final["trace"] == FULL_PATH
    assert final["route_decision"] == "tool" and final["tool_name"] == "revenue"
    response = final["response"]
    assert response.status == "success" and response.result.value == 256000.0
    assert response.validation.status == "passed" and final["explanation"]


def test_a_gemini_plan_is_routed_and_its_arguments_are_recorded(sales) -> None:
    llm = FakeLlm(plan_json(intent="rank", tool_name="ranking", metric="revenue", aggregation="sum",
                            group_by="region", sort_order="desc", limit=1))

    final = run_workflow("Which region generated the highest revenue?", sales, deps=make_deps(llm=llm))

    assert final["trace"] == FULL_PATH and final["classifier"].used == "gemini"
    assert final["tool_arguments"] == {"metric": "revenue", "aggregation": "sum", "group_by": "region",
                                       "sort_order": "desc", "limit": 1}
    assert final["response"].result.items[0].group == "North"


def test_the_rule_fallback_is_preserved(sales) -> None:
    final = run_workflow("What is the total revenue?", sales)

    classifier = final["response"].classifier
    assert classifier.used == "rule_based" and classifier.fallback_reason == "GEMINI_NOT_CONFIGURED"


def test_definition_question_uses_the_metric_definition_branch(sales) -> None:
    retriever, explainer = FakeRetriever(), SpyExplainer()

    final = run_workflow("What is revenue?", sales, deps=make_deps(retriever=retriever, explainer=explainer))

    assert final["trace"] == ["classify", "route", "metric_definition", "respond"]
    assert final["route_decision"] == "metric_definition" and retriever.calls == ["What is revenue?"]
    assert explainer.calls == 0
    response = final["response"]
    assert response.status == "unsupported" and response.result is None and response.explanation is None
    assert response.error.details == {"reason": "definition_not_available"} and "Module 7" in response.message


def test_a_found_definition_is_a_success_without_any_calculation(sales) -> None:
    found = MetricDefinitionResult(found=True, source="knowledge_base", is_stub=False, query="q",
                                   definition="Revenue is the money earned from sales.", message="Found.")

    final = run_workflow("What is revenue?", sales, deps=make_deps(retriever=FakeRetriever(found)))

    response = final["response"]
    assert response.status == "success" and response.explanation == "Revenue is the money earned from sales."
    assert response.result is None and response.tool_used is None
    assert response.calculation_method == "metric_definition" and response.validation.status == "not_run"


def test_unsupported_question_goes_straight_to_respond(sales) -> None:
    explainer = SpyExplainer()

    final = run_workflow("Predict next month's revenue", sales, deps=make_deps(explainer=explainer))

    assert final["trace"] == ["classify", "route", "respond"] and final["route_decision"] == "unsupported"
    response = final["response"]
    assert response.status == "unsupported" and explainer.calls == 0
    assert response.error.details == {"reason": "unsupported_question"}


def test_insufficient_data_skips_validation_and_explanation() -> None:
    explainer = SpyExplainer()
    frame = frame_from_csv("product,region\nPen,North")

    final = run_workflow("What is the total revenue?", frame, deps=make_deps(explainer=explainer))

    assert final["trace"] == ["classify", "route", "execute", "respond"] and explainer.calls == 0
    response = final["response"]
    assert response.status == "insufficient_data" and response.error.code == "INSUFFICIENT_DATA"
    assert response.result is None and response.explanation is None


def test_an_unknown_tool_is_routed_to_an_error(sales, monkeypatch) -> None:
    monkeypatch.delitem(TOOL_HANDLERS, "revenue")

    final = run_workflow("What is the total revenue?", sales)

    assert final["trace"] == ["classify", "route", "respond"] and final["route_decision"] == "invalid"
    assert final["response"].status == "error" and final["response"].error.code == "UNSUPPORTED_TOOL"


def test_failed_validation_skips_the_explanation(sales) -> None:
    failed = ValidationInfo(status="failed", checks=[ValidationCheck(name="row_accounting", passed=False, detail="x")])
    explainer = SpyExplainer()
    deps = make_deps(validate=lambda frame, plan, result, rows: failed, explainer=explainer)

    final = run_workflow("What is the total revenue?", sales, deps=deps)

    assert final["trace"] == ["classify", "route", "execute", "validate", "respond"] and explainer.calls == 0
    response = final["response"]
    assert response.status == "error" and response.error.code == "RESULT_VALIDATION_FAILED"
    assert response.error.details == {"failed_checks": ["row_accounting"]}
    assert response.result is None and response.explanation is None and response.validation.status == "failed"


def test_unexpected_errors_are_not_swallowed(sales, monkeypatch) -> None:
    def crash(frame, plan):
        raise RuntimeError("boom")

    monkeypatch.setitem(TOOL_HANDLERS, "revenue", crash)

    with pytest.raises(RuntimeError, match="boom"):
        run_workflow("What is the total revenue?", sales)