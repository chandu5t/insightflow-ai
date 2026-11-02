"""Focused tests for bounded, additive V2.5 self-correction."""

from datetime import datetime, timezone
from uuid import UUID

import pandas as pd
import pytest
from pydantic import ValidationError

from app.api.analysis_routes import get_metric_retriever
from app.api.dataset_routes import get_dataset_repository
from app.main import app
from app.multi_agent.schemas import MultiAgentRequest, MultiAgentResult
from app.multi_agent.workflow import run_multi_agent_workflow
from app.planner.schemas import AnalysisPlan, PlanStep, PlanValidationError
from app.planner.validator import validate_plan
from app.schemas.dataset_schema import DatasetMetadata
from app.self_correction.schemas import CorrectionRequest, CorrectionResponse
from app.self_correction.service import correct_execution
from app.self_correction.strategies import STRATEGY_ID, assess_single_step_id_mismatch
from app.services.metric_retriever import StubMetricRetriever
from app.verification.schemas import VerificationCheck, VerificationResult
from app.verification.service import verify_execution

DATASET_ID = UUID("12345678-1234-5678-1234-567812345678")


class MemoryRepository:
    def __init__(self, tmp_path):
        self.frame = pd.DataFrame({"value": [1, 2, 3]})
        self.path = tmp_path / f"{DATASET_ID}.csv"
        self.frame.to_csv(self.path, index=False)

    def get(self, dataset_id):
        assert dataset_id == DATASET_ID
        return DatasetMetadata(
            dataset_id=DATASET_ID,
            filename="rows.csv",
            source_format="csv",
            row_count=len(self.frame),
            column_count=len(self.frame.columns),
            column_names=list(self.frame.columns),
            original_size_bytes=1,
            uploaded_at=datetime.now(timezone.utc),
        )

    def get_csv_path(self, dataset_id):
        assert dataset_id == DATASET_ID
        return self.path


def plan(step_id="planned", *, operation="count", depends_on=None):
    return AnalysisPlan(
        intent="count_orders",
        reasoning_type="simple",
        steps=[PlanStep(
            step_id=step_id,
            operation=operation,
            description="Count dataset rows",
            inputs=[],
            parameters={},
            depends_on=depends_on or [],
        )],
    )


def pipeline_inputs(repository, *, planned_id="planned", executed_id="executed", operation="count"):
    planned = plan(planned_id, operation=operation)
    execution_plan = plan(executed_id, operation=operation)
    request = MultiAgentRequest(
        question="Count rows in the dataset",
        dataset_id=DATASET_ID,
        analysis_plan=execution_plan,
    )
    execution = run_multi_agent_workflow(request, repository, StubMetricRetriever())
    verification = verify_execution(
        planned, execution, repository=repository, dataset_id=DATASET_ID
    )
    return planned, execution, verification


def correction_request(planned, execution, verification, *, budget=1):
    return CorrectionRequest(
        question="Count rows in the dataset",
        dataset_id=DATASET_ID,
        analysis_plan=planned,
        execution_result=execution,
        verification_result=verification,
        correction_budget=budget,
    )


def assert_qualifying_initial_mismatch(planned, execution, verification):
    assert verification.status == "failed"
    assert [(item.name, item.step_ids) for item in verification.failed_checks] == [
        ("unexpected_executed_steps", ["executed"]),
        ("planned_step_coverage", ["planned"]),
        ("workflow_step_consistency", ["planned"]),
    ]
    assert len(verification.unsupported_checks) == 1
    check = verification.unsupported_checks[0]
    assert (check.name, check.category, check.step_ids) == (
        "planned_step_status", "verification_semantics_undefined", ["planned"]
    )
    decision = assess_single_step_id_mismatch(
        planned, execution, verification, allow_initial_unsupported_exception=True
    )
    assert decision.eligible, decision.reason


def test_valid_request_and_exact_reachable_mismatch(tmp_path):
    repository = MemoryRepository(tmp_path)
    planned, execution, verification = pipeline_inputs(repository)
    assert_qualifying_initial_mismatch(planned, execution, verification)
    request = correction_request(planned, execution, verification)
    assert request.correction_budget == 1
    assert request.analysis_plan.steps[0].step_id == "planned"


@pytest.mark.parametrize("budget", [-1, 3])
def test_invalid_budget_is_rejected(budget):
    with pytest.raises(ValidationError):
        CorrectionRequest(
            question="Count rows", dataset_id=DATASET_ID, analysis_plan=plan(),
            execution_result=MultiAgentResult(workflow_status="completed"),
            verification_result=VerificationResult(
                status="passed", metadata={"request_id": "test", "tolerance_absolute": 1e-9}
            ),
            correction_budget=budget,
        )


def test_invalid_request_schema_is_rejected():
    with pytest.raises(ValidationError):
        CorrectionRequest.model_validate({"question": "", "dataset_id": "bad"})


def test_initial_passed_returns_not_corrected_without_consuming_budget(tmp_path):
    repository = MemoryRepository(tmp_path)
    planned = plan("same")
    execution = run_multi_agent_workflow(
        MultiAgentRequest(question="Count rows", dataset_id=DATASET_ID, analysis_plan=planned),
        repository, StubMetricRetriever(),
    )
    verification = verify_execution(planned, execution, repository=repository, dataset_id=DATASET_ID)
    response = correct_execution(correction_request(planned, execution, verification, budget=0), repository,
                                 StubMetricRetriever())
    assert response.terminal_status == "not_corrected"
    assert response.terminal_reason == "already_verified"
    assert response.attempts_used == 0


def test_initial_unsupported_returns_terminal_unsupported(tmp_path):
    repository = MemoryRepository(tmp_path)
    planned = plan("agg", operation="aggregate")
    execution = MultiAgentResult(workflow_status="completed", executed_steps=["agg"],
                                 step_statuses={"agg": "completed"})
    unsupported = VerificationResult(
        status="unsupported",
        unsupported_checks=[VerificationCheck(
            name="numerical_verification", status="unsupported", message="not defined",
            category="verification_semantics_undefined", step_ids=["agg"],
        )],
        checks=[VerificationCheck(
            name="numerical_verification", status="unsupported", message="not defined",
            category="verification_semantics_undefined", step_ids=["agg"],
        )],
        metadata={"request_id": "test", "tolerance_absolute": 1e-9},
    )
    response = correct_execution(correction_request(planned, execution, unsupported), repository,
                                 StubMetricRetriever())
    assert response.terminal_status == "unsupported"
    assert response.terminal_reason == "verification_unsupported"
    assert response.attempts_used == 0


def test_budget_zero_is_checked_after_eligibility(tmp_path):
    repository = MemoryRepository(tmp_path)
    planned, execution, verification = pipeline_inputs(repository)
    response = correct_execution(correction_request(planned, execution, verification, budget=0), repository,
                                 StubMetricRetriever())
    assert response.terminal_status == "budget_exhausted"
    assert response.attempts_used == 0
    assert not response.correction_attempts


def test_budget_two_allows_one_qualified_reconciliation(tmp_path):
    repository = MemoryRepository(tmp_path)
    planned, execution, verification = pipeline_inputs(repository)
    response = correct_execution(correction_request(planned, execution, verification, budget=2), repository,
                                 StubMetricRetriever())
    assert response.terminal_status == "corrected"
    assert response.attempts_used == 1
    assert response.correction_attempts[0].strategy_id == STRATEGY_ID
    assert response.final_verification_result.status == "passed"
    assert response.question == "Count rows in the dataset"
    assert response.dataset_id == DATASET_ID


def test_end_to_end_correction_revalidates_reexecutes_and_reverifies(tmp_path):
    repository = MemoryRepository(tmp_path)
    planned, execution, verification = pipeline_inputs(repository)
    original_dump = planned.model_dump(mode="json")
    response = correct_execution(correction_request(planned, execution, verification), repository,
                                 StubMetricRetriever())
    assert response.terminal_status == "corrected"
    assert response.final_plan.steps[0].step_id == "executed"
    updated = response.final_plan.model_dump(mode="json")
    updated["steps"][0]["step_id"] = original_dump["steps"][0]["step_id"]
    assert updated == original_dump
    assert planned.model_dump(mode="json") == original_dump
    assert validate_plan(response.final_plan) == []
    assert response.final_execution_result.workflow_status == "completed"
    assert response.final_verification_result.status == "passed"
    assert response.metadata["execution_count"] == 2
    assert response.metadata["verification_count"] == 2


def test_strategy_rejects_missing_eligibility_evidence(tmp_path):
    repository = MemoryRepository(tmp_path)
    planned, execution, verification = pipeline_inputs(repository)
    execution.executed_steps = []
    result = assess_single_step_id_mismatch(planned, execution, verification,
                                            allow_initial_unsupported_exception=True)
    assert not result.eligible


def test_strategy_rejects_multiple_planned_steps(tmp_path):
    repository = MemoryRepository(tmp_path)
    planned, execution, verification = pipeline_inputs(repository)
    two_step_plan = planned.model_copy(update={"steps": [planned.steps[0], plan("other").steps[0]]})
    result = assess_single_step_id_mismatch(two_step_plan, execution, verification,
                                            allow_initial_unsupported_exception=True)
    assert not result.eligible


def test_strategy_rejects_dependencies(tmp_path):
    repository = MemoryRepository(tmp_path)
    planned, execution, verification = pipeline_inputs(repository)
    dependent = plan("planned", depends_on=["earlier"])
    result = assess_single_step_id_mismatch(dependent, execution, verification,
                                            allow_initial_unsupported_exception=True)
    assert not result.eligible


def test_strategy_rejects_operation_metadata_mismatch(tmp_path):
    repository = MemoryRepository(tmp_path)
    planned, execution, verification = pipeline_inputs(repository)
    execution.agent_results[-1].metadata["operation"] = "distinct_count"
    result = assess_single_step_id_mismatch(planned, execution, verification,
                                            allow_initial_unsupported_exception=True)
    assert not result.eligible


def _with_check(verification, check, *, add_failed=False, add_unsupported=False):
    checks = [*verification.checks, check]
    failed = [*verification.failed_checks, check] if add_failed else verification.failed_checks
    unsupported = [*verification.unsupported_checks, check] if add_unsupported else verification.unsupported_checks
    return verification.model_copy(update={"checks": checks, "failed_checks": failed,
                                           "unsupported_checks": unsupported}, deep=True)


@pytest.mark.parametrize("extra", [
    VerificationCheck(name="other_failure", status="failed", message="other", step_ids=["planned"],
                      category="plan_execution_mismatch"),
])
def test_unrelated_failed_check_is_rejected(tmp_path, extra):
    repository = MemoryRepository(tmp_path)
    planned, execution, verification = pipeline_inputs(repository)
    outcome = _with_check(verification, extra, add_failed=True)
    decision = assess_single_step_id_mismatch(planned, execution, outcome,
                                              allow_initial_unsupported_exception=True)
    assert not decision.eligible


@pytest.mark.parametrize("extra", [
    VerificationCheck(name="other_unsupported", status="unsupported", message="other", step_ids=["planned"],
                      category="verification_semantics_undefined"),
    VerificationCheck(name="planned_step_status", status="unsupported", message="wrong category", step_ids=["planned"],
                      category="plan_execution_mismatch"),
    VerificationCheck(name="planned_step_status", status="unsupported", message="wrong ID", step_ids=["executed"],
                      category="verification_semantics_undefined"),
    VerificationCheck(name="wrong_name", status="unsupported", message="wrong name", step_ids=["planned"],
                      category="verification_semantics_undefined"),
])
def test_unrelated_or_malformed_unsupported_check_is_rejected(tmp_path, extra):
    repository = MemoryRepository(tmp_path)
    planned, execution, verification = pipeline_inputs(repository)
    outcome = _with_check(verification, extra, add_unsupported=True)
    decision = assess_single_step_id_mismatch(planned, execution, outcome,
                                              allow_initial_unsupported_exception=True)
    assert not decision.eligible


def test_changed_unsupported_status_cannot_be_reinterpreted(tmp_path):
    repository = MemoryRepository(tmp_path)
    planned, execution, verification = pipeline_inputs(repository)
    outcome = verification.model_copy(update={"unsupported_checks": []}, deep=True)
    decision = assess_single_step_id_mismatch(planned, execution, outcome,
                                              allow_initial_unsupported_exception=False)
    assert not decision.eligible


def test_reverification_unsupported_is_terminal_and_exception_is_not_reused(tmp_path, monkeypatch):
    repository = MemoryRepository(tmp_path)
    planned, execution, verification = pipeline_inputs(repository)
    import app.self_correction.service as service

    unsupported = VerificationResult(
        status="unsupported",
        unsupported_checks=[VerificationCheck(
            name="numerical_verification", status="unsupported", message="undefined",
            category="verification_semantics_undefined", step_ids=["executed"],
        )],
        checks=[VerificationCheck(
            name="numerical_verification", status="unsupported", message="undefined",
            category="verification_semantics_undefined", step_ids=["executed"],
        )], metadata={"request_id": "test", "tolerance_absolute": 1e-9},
    )
    monkeypatch.setattr(service, "verify_execution", lambda *args, **kwargs: unsupported)
    response = service.correct_execution(correction_request(planned, execution, verification), repository,
                                         StubMetricRetriever())
    assert response.terminal_status == "unsupported"
    assert response.attempts_used == 1
    assert response.correction_attempts[0].correction_status == "failed_verification"


def test_reverification_failure_is_terminal_and_attempts_stay_bounded(tmp_path, monkeypatch):
    repository = MemoryRepository(tmp_path)
    planned, execution, verification = pipeline_inputs(repository)
    import app.self_correction.service as service

    failed_check = VerificationCheck(name="numerical_verification", status="failed", message="mismatch",
                                     category="numerical_mismatch", step_ids=["executed"])
    failed = VerificationResult(
        status="failed", checks=[failed_check], failed_checks=[failed_check],
        metadata={"request_id": "test", "tolerance_absolute": 1e-9},
    )
    verify_calls = []
    monkeypatch.setattr(service, "verify_execution", lambda *args, **kwargs: verify_calls.append(1) or failed)
    response = service.correct_execution(correction_request(planned, execution, verification, budget=2), repository,
                                         StubMetricRetriever())
    assert response.terminal_status == "uncorrectable"
    assert response.attempts_used == 1
    assert len(response.correction_attempts) == 1
    assert len(verify_calls) == 1


def test_correction_application_failure_is_structured(tmp_path, monkeypatch):
    repository = MemoryRepository(tmp_path)
    planned, execution, verification = pipeline_inputs(repository)
    import app.self_correction.service as service

    monkeypatch.setattr(service, "apply_single_step_id_reconciliation",
                        lambda *args: (_ for _ in ()).throw(ValueError("private")))
    response = service.correct_execution(correction_request(planned, execution, verification), repository,
                                         StubMetricRetriever())
    assert response.terminal_status == "correction_failure"
    assert response.errors[0].category == "correction_application_failed"
    assert "private" not in str(response.model_dump())


def test_v22_validation_failure_prevents_v23_execution(tmp_path, monkeypatch):
    repository = MemoryRepository(tmp_path)
    planned, execution, verification = pipeline_inputs(repository)
    import app.self_correction.service as service

    monkeypatch.setattr(service, "validate_plan", lambda _: [PlanValidationError(
        code="TEST_INVALID", message="invalid", step_id="executed"
    )])
    monkeypatch.setattr(service, "run_multi_agent_workflow",
                        lambda *args: pytest.fail("V2.3 must not run an invalid corrected plan"))
    response = service.correct_execution(correction_request(planned, execution, verification), repository,
                                         StubMetricRetriever())
    assert response.terminal_status == "correction_validation_failed"
    assert response.attempts_used == 0


def test_v23_reexecution_failure_is_structured(tmp_path, monkeypatch):
    repository = MemoryRepository(tmp_path)
    planned, execution, verification = pipeline_inputs(repository)
    import app.self_correction.service as service

    monkeypatch.setattr(service, "run_multi_agent_workflow",
                        lambda *args: MultiAgentResult(workflow_status="failed"))
    response = service.correct_execution(correction_request(planned, execution, verification), repository,
                                         StubMetricRetriever())
    assert response.terminal_status == "re_execution_failure"
    assert response.attempts_used == 1
    assert response.correction_attempts[0].error.category == "re_execution_failed"


def test_additive_api_and_existing_dependency_injection(tmp_path, client):
    repository = MemoryRepository(tmp_path)
    planned, execution, verification = pipeline_inputs(repository)
    app.dependency_overrides[get_dataset_repository] = lambda: repository
    app.dependency_overrides[get_metric_retriever] = StubMetricRetriever
    try:
        response = client.post("/analysis/correct", json=correction_request(
            planned, execution, verification
        ).model_dump(mode="json"))
        assert response.status_code == 200
        assert response.json()["terminal_status"] == "corrected"
        assert response.json()["attempts_used"] == 1
    finally:
        app.dependency_overrides.pop(get_dataset_repository, None)
        app.dependency_overrides.pop(get_metric_retriever, None)


def test_corrected_plan_keeps_all_non_id_fields_identical(tmp_path):
    repository = MemoryRepository(tmp_path)
    planned, execution, verification = pipeline_inputs(repository)
    response = correct_execution(correction_request(planned, execution, verification), repository,
                                 StubMetricRetriever())
    original = response.original_plan.model_dump(mode="json")
    corrected = response.final_plan.model_dump(mode="json")
    assert corrected["steps"][0]["step_id"] == "executed"
    corrected["steps"][0]["step_id"] = original["steps"][0]["step_id"]
    assert corrected == original
    assert response.original_execution_result.executed_steps == ["executed"]
    assert response.original_verification_result.status == "failed"
    assert response.question == "Count rows in the dataset"
    assert response.dataset_id == DATASET_ID


def test_invalid_corrected_plan_cannot_pass_registry_validation():
    bad_plan = plan("planned", operation="unknown_operation")
    assert validate_plan(bad_plan)
