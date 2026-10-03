"""Unit tests for V2.2 planner schemas, normalization, and structural validation."""

import pytest
from pydantic import ValidationError

from app.planner.normalizer import normalize_plan_payload
from app.planner.operations import OPERATION_REGISTRY
from app.planner.schemas import AnalysisPlan
from app.planner.validator import validate_plan


def plan(steps, **overrides):
    payload = {
        "intent": "regional_revenue_ranking",
        "reasoning_type": "multi_step",
        "steps": steps,
        "unsupported_reason": None,
    }
    payload.update(overrides)
    return AnalysisPlan.model_validate(payload)


def step(step_id, operation, *, inputs=None, parameters=None, depends_on=None):
    return {
        "step_id": step_id,
        "operation": operation,
        "description": f"Use {operation}",
        "inputs": inputs or [],
        "parameters": parameters or {},
        "depends_on": depends_on or [],
    }


def test_registry_contains_exactly_the_frozen_fourteen_operations():
    assert len(OPERATION_REGISTRY) == 14
    assert set(OPERATION_REGISTRY) == {
        "select_columns", "filter_rows", "derive_metric", "group_by", "aggregate", "sort", "rank",
        "top_n", "bottom_n", "count", "distinct_count", "calculate_difference",
        "calculate_percentage_difference", "compare_groups",
    }


@pytest.mark.parametrize(("alias", "canonical"), [("GROUP_BY", "group_by"), ("GroupBy", "group_by")])
def test_normalizes_only_registered_explicit_aliases(alias, canonical):
    result = normalize_plan_payload({"steps": [{"operation": alias}]})
    assert result["steps"][0]["operation"] == canonical
    unknown = normalize_plan_payload({"steps": [{"operation": "MagicBusinessPrediction"}]})
    assert unknown["steps"][0]["operation"] == "MagicBusinessPrediction"


def test_schema_rejects_wrong_types_and_extra_fields():
    with pytest.raises(ValidationError):
        AnalysisPlan.model_validate({"intent": "bogus", "reasoning_type": "simple", "steps": []})
    with pytest.raises(ValidationError):
        AnalysisPlan.model_validate({"intent": "total_revenue", "reasoning_type": "simple", "steps": [], "python": "print(1)"})


def test_valid_dependencies_and_empty_serialized_fields():
    output = plan([
        step("s1", "derive_metric", parameters={"formula": "quantity * unit_price", "output_name": "revenue"}),
        step("s2", "aggregate", parameters={"function": "sum"}, depends_on=["s1"]),
    ])
    assert validate_plan(output) == []
    assert output.model_dump(mode="json")["steps"][0]["inputs"] == []


@pytest.mark.parametrize(
    ("steps", "expected_code"),
    [
        ([step("s1", "magic_analysis")], "UNKNOWN_OPERATION"),
        ([step("s1", "top_n")], "MISSING_REQUIRED_PARAMETER"),
        ([step("s1", "aggregate", parameters={"function": "sum", "sql": "select 1"})], "UNSUPPORTED_PARAMETER"),
        ([step("s1", "count", depends_on=["missing"])], "UNKNOWN_DEPENDENCY"),
        ([step("s1", "count", depends_on=["s1"])], "SELF_DEPENDENCY"),
        ([step("s1", "count", depends_on=["s2"]), step("s2", "count")], "INVALID_DEPENDENCY_ORDER"),
        ([step("s1", "count"), step("s1", "count")], "DUPLICATE_STEP_ID"),
        ([step("s1", "count", depends_on=["s2"]), step("s2", "count", depends_on=["s1"])], "CIRCULAR_DEPENDENCY"),
    ],
)
def test_rejects_invalid_plans(steps, expected_code):
    errors = validate_plan(plan(steps))
    assert expected_code in {error.code for error in errors}


def test_unsupported_analysis_is_a_valid_outcome_when_well_formed():
    output = plan([], intent="unsupported_analysis", reasoning_type="unsupported", unsupported_reason="Forecasting is unsupported.")
    assert validate_plan(output) == []


def test_unsupported_outcome_requires_reason_and_no_steps():
    output = plan([step("s1", "count")], intent="unsupported_analysis", reasoning_type="unsupported")
    assert {error.code for error in validate_plan(output)} >= {"INVALID_UNSUPPORTED_OUTCOME", "MISSING_UNSUPPORTED_REASON"}
