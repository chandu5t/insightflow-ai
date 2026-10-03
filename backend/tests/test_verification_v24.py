"""Focused tests for additive V2.4 verification."""

from datetime import datetime, timezone
from uuid import UUID

import pandas as pd
import pytest
from pydantic import ValidationError

from app.api.dataset_routes import get_dataset_repository
from app.api.multi_agent_routes import router as multi_agent_router
from app.api.verification_routes import router as verification_router
from app.core.errors import AppError, ErrorCode
from app.main import app
from app.planner.schemas import AnalysisPlan, PlanStep
from app.schemas.dataset_schema import DatasetMetadata
from app.verification.schemas import VerificationResult
from app.verification.service import ABSOLUTE_TOLERANCE, verify_execution

DATASET_ID = UUID("12345678-1234-5678-1234-567812345678")


class MemoryRepository:
    def __init__(self, tmp_path, frame=None):
        self.frame = frame if frame is not None else pd.DataFrame({"value": ["1", "2"], "group": ["a", "b"]})
        self.path = tmp_path / f"{DATASET_ID}.csv"
        self.frame.to_csv(self.path, index=False)

    def get(self, dataset_id):
        assert dataset_id == DATASET_ID
        return DatasetMetadata(
            dataset_id=DATASET_ID, filename="data.csv", source_format="csv",
            row_count=len(self.frame), column_count=len(self.frame.columns),
            column_names=list(self.frame.columns), original_size_bytes=1,
            uploaded_at=datetime.now(timezone.utc),
        )

    def get_csv_path(self, dataset_id):
        assert dataset_id == DATASET_ID
        return self.path


def make_plan(*steps):
    return AnalysisPlan(intent="count_orders", reasoning_type="multi_step", steps=list(steps))


def step(step_id, operation, *, depends_on=None, inputs=None, parameters=None):
    return PlanStep(
        step_id=step_id, operation=operation, description=operation,
        depends_on=depends_on or [], inputs=inputs or [], parameters=parameters or {},
    )


def result_for(plan, values, *, workflow_status="completed", executed=None, statuses=None, final_result=...):
    executed = [item.step_id for item in plan.steps] if executed is None else executed
    statuses = {item.step_id: "completed" for item in plan.steps} if statuses is None else statuses
    agents = [
        {"agent_name": "analysis", "step_id": item.step_id,
         "status": values[item.step_id]["status"], "result": values[item.step_id]["result"],
         "metadata": {"operation": item.operation}, "error": values[item.step_id].get("error")}
        for item in plan.steps if item.step_id in values
    ]
    if final_result is ...:
        final_result = agents[-1]["result"] if agents else None
    return {
        "workflow_status": workflow_status,
        "executed_steps": executed,
        "step_statuses": statuses,
        "agent_results": agents,
        "final_result": final_result,
        "errors": [],
    }


def completed(value):
    return {"status": "completed", "result": value, "error": None}


def categories(outcome):
    return {check.category for check in outcome.checks if check.category}


def test_verification_result_schema_uses_frozen_statuses():
    response = VerificationResult(
        status="passed", checks=[], metadata={"request_id": "test", "tolerance_absolute": ABSOLUTE_TOLERANCE}
    )
    assert response.status == "passed"
    with pytest.raises(ValidationError):
        VerificationResult(
            status="okay", checks=[], metadata={"request_id": "test", "tolerance_absolute": ABSOLUTE_TOLERANCE}
        )


def test_valid_count_result_passes_all_applicable_checks(tmp_path):
    plan = make_plan(step("count", "count"))
    outcome = verify_execution(plan, result_for(plan, {"count": completed(2)}),
                               repository=MemoryRepository(tmp_path), dataset_id=DATASET_ID)
    assert outcome.status == "passed"
    assert outcome.verified_steps == ["count"]
    assert not outcome.failed_checks and not outcome.unsupported_checks


@pytest.mark.parametrize("bad_result", [
    {"executed_steps": [], "step_statuses": {}, "agent_results": [], "final_result": None, "errors": []},
    {"workflow_status": "running", "executed_steps": [], "step_statuses": {},
     "agent_results": [], "final_result": None, "errors": []},
    {"workflow_status": "completed", "executed_steps": [], "step_statuses": {},
     "agent_results": [{"agent_name": "mystery", "status": "completed"}], "final_result": None, "errors": []},
    {"workflow_status": "completed", "executed_steps": [], "step_statuses": {},
     "agent_results": [{"agent_name": "analysis", "step_id": "s", "status": "completed",
                         "metadata": ["malformed"]}], "final_result": None, "errors": []},
])
def test_invalid_result_schema_stops_downstream_checks(bad_result):
    outcome = verify_execution(make_plan(), bad_result)
    assert outcome.status == "failed"
    assert categories(outcome) == {"invalid_result_schema"}
    assert len(outcome.checks) == 1
    assert [item.category for item in outcome.failed_checks] == ["invalid_result_schema"]


def test_failed_status_takes_precedence_over_unsupported():
    plan = make_plan(step("agg", "aggregate"))
    raw = result_for(plan, {"agg": completed(8)}, executed=["agg", "not-planned"])
    outcome = verify_execution(plan, raw)
    assert outcome.status == "failed"
    assert outcome.unsupported_checks
    assert all(check.status == "failed" for check in outcome.failed_checks)
    assert all(check.status == "unsupported" for check in outcome.unsupported_checks)


def test_unsupported_without_failure_produces_unsupported_overall():
    plan = make_plan(step("agg", "aggregate", parameters={"function": "sum"}))
    outcome = verify_execution(plan, result_for(plan, {"agg": completed(8)}))
    assert outcome.status == "unsupported"
    assert not outcome.failed_checks
    assert any(check.category == "verification_semantics_undefined" for check in outcome.unsupported_checks)


def test_plan_execution_match_and_final_grounding_passes(tmp_path):
    plan = make_plan(step("rows", "count"))
    outcome = verify_execution(plan, result_for(plan, {"rows": completed(2)}),
                               repository=MemoryRepository(tmp_path), dataset_id=DATASET_ID)
    assert outcome.status == "passed"
    assert any(check.name == "grounding" and check.status == "passed" for check in outcome.checks)
    assert any(reference.startswith("agent_results[0]") for reference in outcome.evidence)


def test_missing_planned_step_fails_for_completed_workflow():
    plan = make_plan(step("one", "count"), step("two", "count", depends_on=["one"]))
    raw = result_for(plan, {"one": completed(1)}, executed=["one"], statuses={"one": "completed", "two": "pending"})
    outcome = verify_execution(plan, raw)
    assert outcome.status == "failed"
    assert "plan_execution_mismatch" in categories(outcome)


def test_missing_planned_step_is_unsupported_after_failed_workflow():
    plan = make_plan(step("one", "count"), step("two", "count", depends_on=["one"]))
    raw = result_for(
        plan, {"one": completed(1)}, workflow_status="failed", executed=["one"],
        statuses={"one": "completed", "two": "pending"},
    )
    outcome = verify_execution(plan, raw)
    coverage = next(check for check in outcome.checks if check.name == "planned_step_coverage")
    assert outcome.status == "unsupported"
    assert coverage.status == "unsupported"
    assert not outcome.failed_checks


def test_completed_workflow_missing_required_final_result_fails_consistency():
    plan = make_plan(step("count", "count"))
    raw = result_for(plan, {"count": completed(2)}, final_result=None)
    outcome = verify_execution(plan, raw)
    consistency = next(check for check in outcome.checks if check.name == "final_result_consistency")
    assert outcome.status == "failed"
    assert consistency.status == "failed"
    assert consistency.category == "consistency_violation"


def test_unsupported_step_is_not_reported_as_fully_verified():
    plan = make_plan(step("aggregate", "aggregate", parameters={"function": "sum"}))
    outcome = verify_execution(plan, result_for(plan, {"aggregate": completed(8)}))
    assert outcome.status == "unsupported"
    assert outcome.verified_steps == []


def test_unexpected_executed_step_fails():
    plan = make_plan(step("one", "count"))
    outcome = verify_execution(plan, result_for(plan, {"one": completed(2)}, executed=["one", "extra"]))
    assert outcome.status == "failed"
    assert any(check.name == "unexpected_executed_steps" for check in outcome.failed_checks)


def test_executed_operation_must_match_the_plan():
    plan = make_plan(step("count", "count"))
    raw = result_for(plan, {"count": completed(2)})
    raw["agent_results"][0]["metadata"]["operation"] = "distinct_count"
    outcome = verify_execution(plan, raw)
    assert outcome.status == "failed"
    assert any(check.name == "step_operation_identity" and check.status == "failed" for check in outcome.checks)


def test_declared_dependencies_are_checked_against_completed_execution():
    plan = make_plan(step("left", "count"), step("right", "count"),
                     step("difference", "calculate_difference", depends_on=["left", "right"]))
    raw = result_for(plan, {"left": completed(9), "right": completed(4), "difference": completed(5)})
    checks = [check for check in verify_execution(plan, raw).checks if check.name == "dependency_validation"]
    assert len(checks) == 2 and all(check.status == "passed" for check in checks)


def test_missing_dependency_id_fails_plan_validation():
    plan = make_plan(step("one", "count", depends_on=["missing"]))
    outcome = verify_execution(plan, result_for(plan, {"one": completed(2)}))
    assert outcome.status == "failed" and "plan_execution_mismatch" in categories(outcome)


def test_inputs_are_not_interpreted_as_dependencies():
    plan = make_plan(step("select", "select_columns", inputs=["group"], parameters={"columns": ["group"]}))
    outcome = verify_execution(plan, result_for(plan, {"select": completed({"columns": ["group"], "rows": []})}))
    assert outcome.status == "passed"
    assert not [check for check in outcome.checks if check.name == "dependency_validation"]


def test_named_count_inputs_without_dependencies_are_not_independently_verified(tmp_path):
    plan = make_plan(step("count", "count", inputs=["rows"]))
    outcome = verify_execution(plan, result_for(plan, {"count": completed(2)}),
                               repository=MemoryRepository(tmp_path), dataset_id=DATASET_ID)
    assert outcome.status == "unsupported"
    assert "verification_semantics_undefined" in categories(outcome)


def test_dependency_out_of_execution_order_fails():
    plan = make_plan(step("left", "count"), step("right", "count"),
                     step("difference", "calculate_difference", depends_on=["left", "right"]))
    raw = result_for(plan, {"left": completed(9), "right": completed(4), "difference": completed(5)},
                     executed=["difference", "left", "right"])
    outcome = verify_execution(plan, raw)
    assert outcome.status == "failed"
    assert any(check.name == "dependency_validation" for check in outcome.failed_checks)


def test_count_numerical_mismatch_fails(tmp_path):
    plan = make_plan(step("count", "count"))
    outcome = verify_execution(plan, result_for(plan, {"count": completed(3)}),
                               repository=MemoryRepository(tmp_path), dataset_id=DATASET_ID)
    numerical = next(check for check in outcome.checks if check.name == "numerical_verification")
    assert outcome.status == "failed" and numerical.category == "numerical_mismatch"
    assert numerical.expected == 2 and numerical.observed == 3


def test_distinct_count_is_independently_verified(tmp_path):
    plan = make_plan(step("distinct", "distinct_count", inputs=["group"], parameters={"column": "group"}))
    outcome = verify_execution(plan, result_for(plan, {"distinct": completed(2)}),
                               repository=MemoryRepository(tmp_path), dataset_id=DATASET_ID)
    numerical = next(check for check in outcome.checks if check.name == "numerical_verification")
    assert numerical.status == "passed" and numerical.expected == 2


def test_distinct_count_with_missing_data_is_unsupported(tmp_path):
    frame = pd.DataFrame({"group": ["a", ""]})
    plan = make_plan(step("distinct", "distinct_count", parameters={"column": "group"}))
    outcome = verify_execution(plan, result_for(plan, {"distinct": completed(1)}),
                               repository=MemoryRepository(tmp_path, frame), dataset_id=DATASET_ID)
    assert outcome.status == "unsupported"
    assert "verification_semantics_undefined" in categories(outcome)


def test_calculate_difference_independently_verifies_two_dependencies():
    plan = make_plan(step("left", "count"), step("right", "count"),
                     step("diff", "calculate_difference", depends_on=["left", "right"]))
    raw = result_for(plan, {"left": completed(9), "right": completed(4), "diff": completed(5)})
    numerical = next(check for check in verify_execution(plan, raw).checks
                     if check.name == "numerical_verification" and check.step_ids == ["diff"])
    assert numerical.status == "passed" and numerical.expected == 5


def test_calculate_difference_mismatch_fails():
    plan = make_plan(step("left", "count"), step("right", "count"),
                     step("diff", "calculate_difference", depends_on=["left", "right"]))
    raw = result_for(plan, {"left": completed(9), "right": completed(4), "diff": completed(6)})
    numerical = next(check for check in verify_execution(plan, raw).checks
                     if check.name == "numerical_verification" and check.step_ids == ["diff"])
    assert numerical.status == "failed" and numerical.category == "numerical_mismatch"


@pytest.mark.parametrize(("observed", "expected_status"), [
    (5.0 + 0.5e-9, "passed"),
    (5.0 + 2e-9, "failed"),
])
def test_floating_point_absolute_tolerance(observed, expected_status):
    plan = make_plan(step("left", "count"), step("right", "count"),
                     step("diff", "calculate_difference", depends_on=["left", "right"]))
    raw = result_for(plan, {"left": completed(9), "right": completed(4), "diff": completed(observed)})
    numerical = next(check for check in verify_execution(plan, raw).checks
                     if check.name == "numerical_verification" and check.step_ids == ["diff"])
    assert numerical.status == expected_status
    assert numerical.metadata["absolute_tolerance"] == 1e-9


def test_unsupported_numeric_operation_is_explicit():
    plan = make_plan(step("rank", "rank", parameters={"by": "value", "order": "desc"}))
    outcome = verify_execution(plan, result_for(plan, {"rank": completed([{"rank": 1}])}))
    check = next(check for check in outcome.unsupported_checks if check.name == "numerical_verification")
    assert outcome.status == "unsupported" and check.category == "verification_semantics_undefined"


def test_final_result_without_matching_successful_evidence_fails_grounding():
    plan = make_plan(step("count", "count"))
    outcome = verify_execution(plan, result_for(plan, {"count": completed(2)}, final_result=99))
    grounding = next(check for check in outcome.checks if check.name == "grounding")
    assert grounding.status == "failed" and grounding.category == "unsupported_claim"


def test_failed_and_null_agent_results_are_excluded_from_grounding():
    raw = {"workflow_status": "failed", "executed_steps": [], "step_statuses": {},
           "agent_results": [
               {"agent_name": "data_understanding", "step_id": None, "status": "failed",
                "result": None, "metadata": {}, "error": {
                   "category": "agent_failure", "code": "FAILED", "message": "safe"}},
               {"agent_name": "knowledge", "step_id": None, "status": "completed",
                "result": None, "metadata": {}, "error": None},
           ], "final_result": None, "errors": []}
    outcome = verify_execution(make_plan(), raw)
    assert not [check for check in outcome.checks if check.name == "agent_result_grounding"]


def test_completed_workflow_with_failed_step_is_inconsistent():
    plan = make_plan(step("count", "count"))
    raw = result_for(plan, {"count": completed(2)}, statuses={"count": "failed"})
    outcome = verify_execution(plan, raw)
    assert outcome.status == "failed" and "consistency_violation" in categories(outcome)


def test_missing_dataset_context_is_unsupported_not_passed():
    plan = make_plan(step("count", "count"))
    outcome = verify_execution(plan, result_for(plan, {"count": completed(2)}))
    assert outcome.status == "unsupported"
    assert "verification_semantics_undefined" in categories(outcome)


def test_expected_dataset_unavailability_is_unsupported():
    class MissingRepository:
        def get(self, dataset_id):
            raise AppError(ErrorCode.DATASET_NOT_FOUND, "Dataset not found.", status_code=404)

    plan = make_plan(step("count", "count"))
    outcome = verify_execution(plan, result_for(plan, {"count": completed(2)}),
                               repository=MissingRepository(), dataset_id=DATASET_ID)
    assert outcome.status == "unsupported"
    assert "verification_semantics_undefined" in categories(outcome)


def test_unexpected_internal_exception_is_structured(monkeypatch):
    import app.verification.service as service

    def explode(*args, **kwargs):
        raise RuntimeError("private detail")

    monkeypatch.setattr(service, "_plan_execution_checks", explode)
    outcome = service.verify_execution(make_plan(), {
        "workflow_status": "completed", "executed_steps": [], "step_statuses": {},
        "agent_results": [], "final_result": None, "errors": [],
    })
    assert outcome.status == "failed" and "verification_failure" in categories(outcome)
    assert "private detail" not in str(outcome.model_dump())


def test_verify_api_handles_valid_and_invalid_workflow_results(client, tmp_path):
    repository = MemoryRepository(tmp_path)
    app.dependency_overrides[get_dataset_repository] = lambda: repository
    plan = make_plan(step("count", "count"))
    body = {"analysis_plan": plan.model_dump(mode="json"),
            "multi_agent_result": result_for(plan, {"count": completed(2)}),
            "dataset_id": str(DATASET_ID)}
    response = client.post("/analysis/verify", json=body)
    assert response.status_code == 200 and response.json()["status"] == "passed"
    response = client.post("/analysis/verify", json={**body, "multi_agent_result": {"workflow_status": "bad"}})
    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert response.json()["errors"][0]["category"] == "invalid_result_schema"


def test_verify_api_is_additive_and_v23_contract_remains_registered():
    assert "/analysis/verify" in {route.path for route in verification_router.routes}
    assert "/analysis/multi-agent" in {route.path for route in multi_agent_router.routes}
    route = next(route for route in multi_agent_router.routes if route.path == "/analysis/multi-agent")
    assert route.response_model.__name__ == "MultiAgentResult"
