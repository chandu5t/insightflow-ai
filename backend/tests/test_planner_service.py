"""Service tests use local fakes and never contact the Gemini API."""

import json

import pytest
from app.core.errors import AppError, ErrorCode
from app.planner.schemas import PlannerRequest
from app.planner.service import SYSTEM_PROMPT, create_plan
from app.services.gemini_client import GeminiNotConfiguredError, GeminiRequestError


def generated(**overrides):
    value = {
        "intent": "total_revenue",
        "reasoning_type": "simple",
        "steps": [
            {"step_id": "step_1", "operation": "derive_metric", "description": "Derive revenue", "inputs": ["quantity", "unit_price"], "parameters": {"formula": {"op": "multiply", "left": {"column": "quantity"}, "right": {"column": "unit_price"}}, "output_name": "revenue"}, "depends_on": []},
            {"step_id": "step_2", "operation": "aggregate", "description": "Sum revenue", "inputs": ["step_1"], "parameters": {"function": "sum"}, "depends_on": ["step_1"]},
        ],
        "unsupported_reason": None,
    }
    value.update(overrides)
    return json.dumps(value)


class SequenceModel:
    model_name = "fake-planner"

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def generate_json(self, *, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def test_success_returns_structured_plan_and_logs_request(caplog):
    model = SequenceModel(generated())
    result = create_plan(PlannerRequest(question=" What is   total revenue? "), llm_client=model)
    assert result.valid and result.plan.intent == "total_revenue"
    assert result.metadata.planner_model == "fake-planner" and not result.metadata.retry_used
    assert len(model.calls) == 1
    assert "never calculate" in model.calls[0][0].lower()
    assert '"question":"What is total revenue?"' in model.calls[0][1]
    assert "planner_completed" in caplog.text


def test_retries_provider_failure_once_then_succeeds():
    model = SequenceModel(GeminiRequestError("timeout"), generated())
    result = create_plan(PlannerRequest(question="What is revenue?"), llm_client=model)
    assert result.valid and result.metadata.retry_used and len(model.calls) == 2


def test_retries_unusable_output_once_then_returns_plan():
    model = SequenceModel("not json", generated())
    result = create_plan(PlannerRequest(question="What is revenue?"), llm_client=model)
    assert result.valid and result.metadata.retry_used and len(model.calls) == 2


def test_still_unusable_output_returns_safe_explicit_error():
    model = SequenceModel("private raw output one", "private raw output two")
    with pytest.raises(AppError) as raised:
        create_plan(PlannerRequest(question="q"), llm_client=model)
    assert raised.value.code == ErrorCode.PLAN_SCHEMA_ERROR
    assert "private raw" not in raised.value.message
    assert len(model.calls) == 2


def test_structurally_generated_but_invalid_plan_is_returned_without_retry():
    model = SequenceModel(generated(steps=[{
        "step_id": "s1", "operation": "magic_analysis", "description": "Unknown operation",
        "inputs": [], "parameters": {}, "depends_on": [],
    }]))
    result = create_plan(PlannerRequest(question="q"), llm_client=model)
    assert not result.valid and result.validation_errors[0].code == "UNKNOWN_OPERATION"
    assert len(model.calls) == 1 and not result.metadata.retry_used


def test_missing_key_is_explicit_and_does_not_fallback():
    model = SequenceModel(GeminiNotConfiguredError("no key"))
    with pytest.raises(AppError) as raised:
        create_plan(PlannerRequest(question="q"), llm_client=model)
    assert raised.value.status_code == 503
    assert raised.value.code == ErrorCode.PLANNER_GENERATION_ERROR
    assert len(model.calls) == 1
