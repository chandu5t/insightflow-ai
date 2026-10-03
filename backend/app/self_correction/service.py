"""Bounded deterministic V2.5 correction orchestration."""

from app.multi_agent.schemas import MultiAgentRequest, MultiAgentResult
from app.multi_agent.workflow import run_multi_agent_workflow
from app.planner.schemas import AnalysisPlan
from app.planner.validator import validate_plan
from app.self_correction.schemas import (
    CorrectionAttempt,
    CorrectionError,
    CorrectionRequest,
    CorrectionResponse,
)
from app.self_correction.strategies import (
    STRATEGY_ID,
    apply_single_step_id_reconciliation,
    assess_single_step_id_mismatch,
)
from app.services.dataset_repository import DatasetRepository
from app.services.metric_retriever import MetricRetriever
from app.verification.service import verify_execution


def _error(category: str, code: str, message: str) -> CorrectionError:
    return CorrectionError(category=category, code=code, message=message)


def _response(request: CorrectionRequest, *, status: str, reason: str,
              plan: AnalysisPlan | None = None, execution: MultiAgentResult | None = None,
              verification=None, attempts: list[CorrectionAttempt] | None = None,
              errors: list[CorrectionError] | None = None, attempts_used: int = 0) -> CorrectionResponse:
    return CorrectionResponse(
        terminal_status=status,
        terminal_reason=reason,
        question=request.question,
        dataset_id=request.dataset_id,
        original_plan=request.analysis_plan,
        final_plan=plan or request.analysis_plan,
        original_execution_result=request.execution_result,
        final_execution_result=execution or request.execution_result,
        original_verification_result=request.verification_result,
        final_verification_result=verification or request.verification_result,
        attempts_used=attempts_used,
        correction_attempts=attempts or [],
        errors=errors or [],
        metadata={
            "correction_budget": request.correction_budget,
            "strategy_id": STRATEGY_ID,
            "execution_count": 1 + sum(item.re_execution_result is not None for item in (attempts or [])),
            "verification_count": 1 + sum(item.verification_result is not None for item in (attempts or [])),
        },
    )


def correct_execution(
    request: CorrectionRequest,
    repository: DatasetRepository,
    retriever: MetricRetriever,
) -> CorrectionResponse:
    """Apply only the contract-frozen one-step ID reconciliation, if eligible."""
    initial = request.verification_result
    if initial.status == "passed":
        return _response(request, status="not_corrected", reason="already_verified")
    if initial.status == "unsupported":
        return _response(request, status="unsupported", reason="verification_unsupported")

    eligibility = assess_single_step_id_mismatch(
        request.analysis_plan,
        request.execution_result,
        initial,
        allow_initial_unsupported_exception=True,
    )
    if not eligibility.eligible:
        if initial.unsupported_checks or any(check.status == "unsupported" for check in initial.checks):
            return _response(request, status="unsupported", reason="verification_unsupported")
        return _response(request, status="uncorrectable", reason=eligibility.reason or "no_eligible_strategy")
    if request.correction_budget == 0:
        return _response(request, status="budget_exhausted", reason="correction_budget_exhausted")

    attempts: list[CorrectionAttempt] = []
    errors: list[CorrectionError] = []
    current_plan = request.analysis_plan
    current_execution = request.execution_result
    current_verification = initial

    # The frozen registry has one strategy and the narrow unsupported exception
    # is initial-only. Any unsupported check on re-verification is terminal,
    # which means there can be no second eligible attempt for this strategy.
    attempt_number = 1
    try:
        corrected = apply_single_step_id_reconciliation(
            current_plan, eligibility.evidence["executed_step_id"]
        )
    except Exception:
        error = _error("correction_application_failed", "CORRECTION_APPLICATION_FAILED",
                       "The registered step-ID correction could not be applied safely.")
        errors.append(error)
        attempts.append(CorrectionAttempt(
            attempt_number=attempt_number, trigger_verification=current_verification,
            strategy_id=STRATEGY_ID, eligibility_evidence=eligibility.evidence,
            input_plan=current_plan, correction_status="failed", error=error,
        ))
        return _response(request, status="correction_failure", reason="correction_application_failed",
                         attempts=attempts, errors=errors)

    validation_errors = validate_plan(corrected)
    if validation_errors:
        error = _error("correction_validation_failed", "CORRECTION_VALIDATION_FAILED",
                       "The corrected plan did not pass V2.2 validation.")
        errors.append(error)
        attempts.append(CorrectionAttempt(
            attempt_number=attempt_number, trigger_verification=current_verification,
            strategy_id=STRATEGY_ID, eligibility_evidence=eligibility.evidence,
            input_plan=current_plan, corrected_plan=corrected,
            correction_status="validation_failed", error=error,
            metadata={"validation_codes": [item.code for item in validation_errors]},
        ))
        return _response(request, status="correction_validation_failed", reason="corrected_plan_invalid",
                         plan=corrected, attempts=attempts, errors=errors)

    attempt = CorrectionAttempt(
        attempt_number=attempt_number, trigger_verification=current_verification,
        strategy_id=STRATEGY_ID, eligibility_evidence=eligibility.evidence,
        input_plan=current_plan, corrected_plan=corrected, correction_status="applied",
    )
    attempts.append(attempt)
    try:
        multi_request = MultiAgentRequest(
            question=request.question,
            dataset_id=request.dataset_id,
            analysis_plan=corrected,
        )
        current_execution = run_multi_agent_workflow(multi_request, repository, retriever)
    except Exception:
        error = _error("re_execution_failed", "RE_EXECUTION_FAILED",
                       "The existing V2.3 workflow could not re-execute the corrected plan.")
        errors.append(error)
        attempt.correction_status = "re_execution_failed"
        attempt.error = error
        return _response(request, status="re_execution_failure", reason="v23_re_execution_failed",
                         plan=corrected, attempts=attempts, errors=errors, attempts_used=1)

    attempt.re_execution_result = current_execution
    if current_execution.workflow_status != "completed":
        error = _error("re_execution_failed", "RE_EXECUTION_FAILED",
                       "The existing V2.3 workflow returned a failed execution result.")
        errors.append(error)
        attempt.correction_status = "re_execution_failed"
        attempt.error = error
        return _response(request, status="re_execution_failure", reason="v23_re_execution_failed",
                         plan=corrected, execution=current_execution, attempts=attempts,
                         errors=errors, attempts_used=1)

    try:
        current_verification = verify_execution(
            corrected, current_execution, repository=repository, dataset_id=request.dataset_id
        )
    except Exception:
        error = _error("verification_failed", "VERIFICATION_FAILED",
                       "The existing V2.4 verification service could not verify the re-execution.")
        errors.append(error)
        attempt.error = error
        attempt.metadata["verification_invocation_failed"] = True
        return _response(request, status="self_correction_failure", reason="v24_re_verification_failed",
                         plan=corrected, execution=current_execution, attempts=attempts,
                         errors=errors, attempts_used=1)

    attempt.verification_result = current_verification
    if current_verification.status == "passed":
        attempt.correction_status = "verified"
        return _response(request, status="corrected", reason="verification_passed_after_correction",
                         plan=corrected, execution=current_execution, verification=current_verification,
                         attempts=attempts, attempts_used=1)
    if current_verification.status == "unsupported" or current_verification.unsupported_checks:
        attempt.correction_status = "failed_verification"
        return _response(request, status="unsupported", reason="verification_unsupported_after_re_execution",
                         plan=corrected, execution=current_execution, verification=current_verification,
                         attempts=attempts, attempts_used=1)

    attempt.correction_status = "failed_verification"
    return _response(request, status="uncorrectable", reason="re_verification_failed_without_supported_strategy",
                     plan=corrected, execution=current_execution, verification=current_verification,
                     attempts=attempts, attempts_used=1)
