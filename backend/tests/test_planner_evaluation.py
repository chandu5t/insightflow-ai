"""Tests for V2.2 component fixture scoring; not research-result claims."""

import json
from pathlib import Path

from app.planner.evaluation import evaluate_planner_fixture
from app.planner.schemas import AnalysisPlan, PlannerMetadata, PlannerResponse

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "evaluation" / "v2.2_planner_fixture.json"


def test_fixture_is_versioned_and_covers_required_categories():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert fixture["fixture_version"] == "v2.2-development-1"
    assert len(fixture["cases"]) >= 9
    assert {case["category"] for case in fixture["cases"]} >= {
        "simple_aggregation", "grouping", "ranking", "filtering", "comparison",
        "multi_step_analysis", "metric_definition", "unsupported", "ambiguous",
    }


def test_metric_calculation_is_deterministic_for_supplied_plan():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    plan = AnalysisPlan.model_validate({
        "intent": "total_revenue", "reasoning_type": "simple", "unsupported_reason": None,
        "steps": [
            {"step_id": "a", "operation": "derive_metric", "description": "derive", "inputs": [], "parameters": {}, "depends_on": []},
            {"step_id": "b", "operation": "aggregate", "description": "sum", "inputs": [], "parameters": {"function": "sum"}, "depends_on": ["a"]},
        ],
    })
    response = PlannerResponse(
        plan=plan, valid=True, validation_errors=[],
        metadata=PlannerMetadata(request_id="test", planner_model="fake", retry_used=False, latency_ms=2.0),
    )
    scores = evaluate_planner_fixture(fixture["cases"][:1], {"P001": response})
    assert scores == {
        "evaluated": 1, "intent_accuracy": 1.0, "plan_validity_rate": 1.0,
        "operation_selection_precision": 1.0, "step_coverage": 1.0,
        "dependency_accuracy": 1.0, "unsupported_operation_rate": 0.0,
        "mean_latency_ms": 2.0,
    }


def test_empty_evaluation_is_explicit():
    result = evaluate_planner_fixture([], {})
    assert result["evaluated"] == 0 and result["plan_validity_rate"] == 0.0
