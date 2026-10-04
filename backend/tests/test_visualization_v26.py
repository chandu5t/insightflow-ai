"""Focused tests for V2.6's additive, schema-driven visualization boundary."""

from copy import deepcopy
from uuid import UUID

import pytest

from app.multi_agent.schemas import MultiAgentResult
from app.planner.schemas import AnalysisPlan
from app.self_correction.schemas import CorrectionResponse
from app.verification.schemas import VerificationResult
from app.visualization.schemas import VisualizationRequest, VisualizationSpec
from app.visualization.service import visualize

DATASET_ID = UUID("7b8c5d3e-67a9-4df5-ae7e-e837a6f7bd12")


def _plan(*, operation="select_columns", parameters=None, inputs=None, depends_on=None, step_id="s1"):
    params = parameters if parameters is not None else {"columns": ["region", "value"]}
    return AnalysisPlan.model_validate({
        "intent": "regional_revenue_ranking" if operation == "rank" else "total_revenue",
        "reasoning_type": "simple",
        "steps": [{
            "step_id": step_id,
            "operation": operation,
            "description": "Values by region",
            "inputs": inputs if inputs is not None else ["region", "value"],
            "parameters": params,
            "depends_on": depends_on or [],
        }],
    })


def _result(rows=None, *, fields=None, operation="select_columns", step_id="s1", agent_metadata=None):
    rows = rows if rows is not None else [{"region": "North", "value": 10}, {"region": "South", "value": 20}]
    fields = fields if fields is not None else list(rows[0])
    table = {"columns": fields, "rows": rows}
    return MultiAgentResult.model_validate({
        "workflow_status": "completed",
        "executed_steps": [step_id],
        "step_statuses": {step_id: "completed"},
        "agent_results": [
            {"agent_name": "data_understanding", "step_id": None, "status": "completed",
             "result": {"columns": fields}, "metadata": {"dataset_id": str(DATASET_ID)}, "error": None},
            {"agent_name": "analysis", "step_id": step_id, "status": "completed", "result": table,
             "metadata": {"operation": operation, **(agent_metadata or {})}, "error": None},
        ],
        "final_result": table,
        "errors": [],
    })


def _verification(step_id="s1", *, status="passed"):
    return VerificationResult.model_validate({
        "status": status,
        "checks": [],
        "failed_checks": [],
        "unsupported_checks": [],
        "verified_steps": [step_id] if status == "passed" else [],
        "evidence": [],
        "errors": [],
        "metadata": {"request_id": "test-v26", "tolerance_absolute": 1e-9},
    })


def _request(*, plan=None, execution=None, verification=None, correction=None, question="Show values"):
    return VisualizationRequest(
        question=question,
        dataset_id=DATASET_ID,
        analysis_plan=plan or _plan(),
        execution_result=execution or _result(),
        verification_result=verification or _verification(),
        correction_result=correction,
    )


def _correction(plan, execution, verification, *, final_plan=None, final_execution=None,
                final_verification=None, terminal_status="not_corrected", question="Show values"):
    return CorrectionResponse(
        terminal_status=terminal_status,
        question=question,
        dataset_id=DATASET_ID,
        original_plan=plan,
        final_plan=final_plan or plan,
        original_execution_result=execution,
        final_execution_result=final_execution or execution,
        original_verification_result=verification,
        final_verification_result=final_verification or verification,
        attempts_used=0,
    )


def test_bar_result_preserves_rows_values_and_traceability():
    request = _request()
    response = visualize(request)
    assert response.status == "generated"
    assert response.visualization is not None
    assert response.visualization.chart_type == "bar"
    assert response.visualization.data == request.execution_result.final_result["rows"]
    assert response.visualization.traceability.source_step_id == "s1"
    assert response.visualization.traceability.source_operation == "select_columns"
    assert response.visualization.traceability.dataset_id == DATASET_ID


def test_bar_preserves_fractional_values_and_every_row():
    rows = [
        {"region": f"Region {index}", "value": 0.123456789 + index}
        for index in range(15)
    ]
    response = visualize(_request(execution=_result(rows)))
    assert response.status == "generated"
    assert response.visualization.data == rows
    assert response.visualization.metadata["output_row_count"] == len(rows)


def test_line_uses_explicit_ordered_time_values():
    rows = [{"month": "2025-01-01", "value": 10}, {"month": "2025-02-01", "value": 20}]
    plan = _plan(inputs=["month", "value"])
    execution = _result(rows, fields=["month", "value"])
    response = visualize(_request(plan=plan, execution=execution))
    assert response.status == "generated"
    assert response.visualization.chart_type == "line"
    assert response.visualization.metadata["transformation_metadata"] == "explicit_time_values"


def test_scatter_uses_existing_paired_numeric_observations():
    rows = [{"x": 1, "y": 3}, {"x": 2, "y": 5}]
    plan = _plan(inputs=["x", "y"])
    execution = _result(rows, fields=["x", "y"])
    response = visualize(_request(plan=plan, execution=execution))
    assert response.status == "generated"
    assert response.visualization.chart_type == "scatter"
    assert response.visualization.data == rows


def test_pie_requires_explicit_whole_and_component_relationship():
    rows = [{"region": "North", "value": 10}, {"region": "South", "value": 20}]
    execution = _result(rows, agent_metadata={"part_to_whole": {
        "whole_id": "sales-total", "whole_value": 30,
        "relationship": "components_of_whole", "category_field": "region", "value_field": "value",
    }})
    response = visualize(_request(execution=execution))
    assert response.status == "generated"
    assert response.visualization.chart_type == "pie"
    assert response.visualization.metadata["part_to_whole_id"] == "sales-total"


def test_numeric_category_pairs_without_explicit_evidence_never_select_pie():
    response = visualize(_request())
    assert response.status == "generated"
    assert response.visualization.chart_type == "bar"


def test_pie_with_contradictory_explicit_whole_fails_closed():
    execution = _result(agent_metadata={"part_to_whole": {
        "whole_id": "sales-total", "whole_value": 31,
        "relationship": "components_of_whole", "category_field": "region", "value_field": "value",
    }})
    response = visualize(_request(execution=execution))
    assert response.status == "failed"
    assert response.errors[0].category == "visualization_data_mismatch"


def test_empty_result_is_insufficient_data():
    execution = _result([], fields=["region", "value"])
    response = visualize(_request(execution=execution))
    assert response.status == "unsupported"
    assert response.errors[0].category == "insufficient_data"


def test_single_numeric_field_has_unsupported_chart_semantics():
    plan = _plan(inputs=["value"])
    execution = _result([{"value": 10}, {"value": 20}], fields=["value"])
    response = visualize(_request(plan=plan, execution=execution))
    assert response.status == "unsupported"
    assert response.errors[0].category == "unsupported_visualization"


def test_verification_failed_or_unsupported_never_generates():
    for status in ("failed", "unsupported"):
        response = visualize(_request(verification=_verification(status=status)))
        assert response.status == "unsupported"
        assert response.visualization is None


def test_correction_final_artifacts_are_authoritative():
    original_plan = _plan(step_id="original")
    original_execution = _result(step_id="original")
    original_verification = _verification("original")
    final_plan = _plan(step_id="final")
    final_execution = _result(step_id="final")
    final_verification = _verification("final")
    correction = _correction(
        original_plan, original_execution, original_verification,
        final_plan=final_plan, final_execution=final_execution,
        final_verification=final_verification, terminal_status="corrected",
    )
    response = visualize(_request(
        plan=original_plan, execution=original_execution, verification=original_verification,
        correction=correction,
    ))
    assert response.status == "generated"
    assert response.visualization.source_step_id == "final"
    assert response.visualization.traceability.correction_status == "corrected"


def test_mismatched_original_artifacts_fail_closed():
    plan = _plan()
    execution = _result()
    verification = _verification()
    correction = _correction(plan, execution, verification)
    changed = _result([{"region": "Other", "value": 99}])
    response = visualize(_request(execution=changed, correction=correction))
    assert response.status == "failed"
    assert response.errors[0].category == "visualization_data_mismatch"


def test_unsuitable_valid_correction_terminal_is_unsupported():
    plan, execution, verification = _plan(), _result(), _verification()
    correction = _correction(plan, execution, verification, terminal_status="unsupported")
    response = visualize(_request(plan=plan, execution=execution, verification=verification, correction=correction))
    assert response.status == "unsupported"
    assert response.visualization is None


def test_malformed_plan_payload_returns_structured_failure():
    request = _request()
    request.analysis_plan = {"intent": "bad"}
    response = visualize(request)
    assert response.status == "failed"
    assert response.errors[0].category == "invalid_visualization_input"


def test_missing_or_unsafe_values_do_not_get_coerced():
    rows = [{"region": "North", "value": None}, {"region": "South", "value": 20}]
    response = visualize(_request(execution=_result(rows)))
    assert response.status == "unsupported"
    assert response.errors[0].code == "MISSING_VALUES"


def test_visualization_spec_rejects_unknown_chart_type():
    valid = visualize(_request()).visualization.model_dump()
    valid["chart_type"] = "area"
    try:
        VisualizationSpec.model_validate(valid)
    except Exception as exc:
        assert exc.__class__.__name__ == "ValidationError"
    else:
        raise AssertionError("Unknown chart type should fail schema validation")


@pytest.mark.parametrize("mutation", [
    lambda payload: payload.pop("title"),
    lambda payload: payload.update(x_axis={"field": "region"}),
    lambda payload: payload.update(data=["not-a-record"]),
])
def test_visualization_spec_rejects_missing_or_malformed_fields(mutation):
    payload = visualize(_request()).visualization.model_dump()
    mutation(payload)
    with pytest.raises(Exception) as error:
        VisualizationSpec.model_validate(payload)
    assert error.type.__name__ == "ValidationError"


def test_failed_execution_is_rejected_before_visualization():
    execution = _result()
    execution.workflow_status = "failed"
    response = visualize(_request(execution=execution))
    assert response.status == "failed"
    assert response.errors[0].category == "visualization_data_mismatch"


def test_internal_generation_failure_is_structured(monkeypatch):
    from app.visualization import service

    def explode(*args, **kwargs):
        raise RuntimeError("private internal details")

    monkeypatch.setattr(service, "select_chart", explode)
    response = service.visualize(_request())
    assert response.status == "failed"
    assert response.errors[0].category == "visualization_generation_failure"
    assert "private internal details" not in response.errors[0].message


def test_api_generated_and_malformed_payloads_are_structured(make_client):
    client = make_client()
    request = _request()
    body = request.model_dump(mode="json")
    response = client.post("/analysis/visualize", json=body)
    assert response.status_code == 200
    assert response.json()["status"] == "generated"
    malformed = deepcopy(body)
    malformed["analysis_plan"] = {"intent": "nonsense"}
    response = client.post("/analysis/visualize", json=malformed)
    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert response.json()["errors"][0]["category"] == "invalid_visualization_input"


def test_api_is_additive_and_registered(make_client):
    client = make_client()
    response = client.post("/analysis/visualize", json={})
    assert response.status_code == 422
    assert client.get("/health").status_code == 200
