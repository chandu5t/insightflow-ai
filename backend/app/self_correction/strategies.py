"""Deterministic correction eligibility and application for V2.5."""

from dataclasses import dataclass
from typing import Any

from app.multi_agent.schemas import MultiAgentResult
from app.planner.schemas import AnalysisPlan, PlanStep
from app.planner.validator import validate_plan
from app.verification.schemas import VerificationCheck, VerificationResult

STRATEGY_ID = "reconcile_single_step_id_mismatch"


@dataclass(frozen=True)
class StrategyEligibility:
    eligible: bool
    evidence: dict[str, Any]
    reason: str | None = None


def _has_exact_check(check: VerificationCheck, *, name: str, status: str,
                     step_id: str, category: str | None = None) -> bool:
    return (
        check.name == name
        and check.status == status
        and check.step_ids == [step_id]
        and check.category == category
    )


def assess_single_step_id_mismatch(
    plan: AnalysisPlan,
    execution: MultiAgentResult,
    verification: VerificationResult,
    *,
    allow_initial_unsupported_exception: bool,
) -> StrategyEligibility:
    """Prove the sole supported structural mismatch from structured evidence."""
    evidence: dict[str, Any] = {"strategy_id": STRATEGY_ID}
    if verification.status != "failed":
        return StrategyEligibility(False, evidence, "verification_status_not_failed")
    if validate_plan(plan):
        return StrategyEligibility(False, evidence, "analysis_plan_is_not_v22_valid")
    if len(plan.steps) != 1:
        return StrategyEligibility(False, evidence, "requires_exactly_one_planned_step")
    planned = plan.steps[0]
    if planned.depends_on:
        return StrategyEligibility(False, evidence, "dependencies_are_not_supported")

    if execution.workflow_status != "completed" or len(execution.executed_steps) != 1:
        return StrategyEligibility(False, evidence, "requires_one_completed_executed_step")
    if execution.errors or any(result.status != "completed" for result in execution.agent_results):
        return StrategyEligibility(False, evidence, "execution_contains_unrelated_agent_or_workflow_failures")
    executed_id = execution.executed_steps[0]
    if execution.step_statuses.get(executed_id) != "completed":
        return StrategyEligibility(False, evidence, "executed_step_is_not_completed")
    if planned.step_id == executed_id:
        return StrategyEligibility(False, evidence, "step_ids_do_not_differ")

    successful_analysis = [
        result for result in execution.agent_results
        if result.agent_name == "analysis" and result.status == "completed"
    ]
    if len(successful_analysis) != 1 or successful_analysis[0].step_id != executed_id:
        return StrategyEligibility(False, evidence, "requires_one_successful_analysis_result_for_executed_id")
    agent = successful_analysis[0]
    operation = agent.metadata.get("operation")
    if operation != planned.operation:
        return StrategyEligibility(False, evidence, "operation_metadata_mismatch")

    expected_failed = {
        ("unexpected_executed_steps", executed_id),
        ("planned_step_coverage", planned.step_id),
        ("workflow_step_consistency", planned.step_id),
    }
    failed = verification.failed_checks
    observed_failed = {(check.name, step_id) for check in failed for step_id in check.step_ids}
    failed_checks_consistent = (
        len(failed) == 3
        and observed_failed == expected_failed
        and all(check.status == "failed" for check in failed)
        and all(check in verification.checks for check in failed)
        and sum(check.status == "failed" for check in verification.checks) == 3
    )
    if not failed_checks_consistent:
        return StrategyEligibility(False, evidence, "failed_checks_do_not_match_exact_mismatch")
    by_name = {check.name: check for check in failed}
    if not (
        _has_exact_check(by_name["unexpected_executed_steps"], name="unexpected_executed_steps",
                         status="failed", step_id=executed_id, category="plan_execution_mismatch")
        and _has_exact_check(by_name["planned_step_coverage"], name="planned_step_coverage",
                             status="failed", step_id=planned.step_id, category="plan_execution_mismatch")
        and _has_exact_check(by_name["workflow_step_consistency"], name="workflow_step_consistency",
                             status="failed", step_id=planned.step_id, category="consistency_violation")
    ):
        return StrategyEligibility(False, evidence, "failed_check_details_do_not_match_contract")
    consistency = by_name["workflow_step_consistency"]
    if consistency.metadata.get("error_count", 0) != 0:
        return StrategyEligibility(False, evidence, "workflow_consistency_includes_unrelated_errors")

    unsupported = verification.unsupported_checks
    if allow_initial_unsupported_exception:
        if (
            len(unsupported) != 1
            or not _has_exact_check(
                unsupported[0], name="planned_step_status", status="unsupported",
                step_id=planned.step_id, category="verification_semantics_undefined",
            )
            or unsupported[0] not in verification.checks
            or sum(check.status == "unsupported" for check in verification.checks) != 1
        ):
            return StrategyEligibility(False, evidence, "unsupported_checks_do_not_match_narrow_initial_exception")
    elif unsupported or any(check.status == "unsupported" for check in verification.checks):
        return StrategyEligibility(False, evidence, "unsupported_verification_is_terminal")

    # This is the structural causal link: the only planned ID is P, the only
    # completed executed ID is E, and V2.4 reports each mismatch check against
    # those exact IDs. The unsupported check is caused by P being absent from
    # the status map while E is the sole completed execution.
    if allow_initial_unsupported_exception and planned.step_id in execution.step_statuses:
        return StrategyEligibility(False, evidence, "planned_step_status_is_not_unrepresented")

    evidence.update({
        "planned_step_id": planned.step_id,
        "executed_step_id": executed_id,
        "successful_analysis_result_step_id": agent.step_id,
        "planned_operation": planned.operation,
        "executed_operation": operation,
        "planned_step_id_absent_from_execution_statuses": planned.step_id not in execution.step_statuses,
        "failed_checks": sorted(f"{name}:{step_id}" for name, step_id in observed_failed),
        "unsupported_checks": [check.model_dump(mode="json") for check in unsupported],
        "dependencies": list(planned.depends_on),
        "correction_target": "analysis_plan.steps[0].step_id",
        "corrected_step_id": executed_id,
    })
    return StrategyEligibility(True, evidence)


def apply_single_step_id_reconciliation(plan: AnalysisPlan, executed_id: str) -> AnalysisPlan:
    """Return a copy with only the first step's ID replaced by executed_id."""
    if len(plan.steps) != 1 or plan.steps[0].depends_on:
        raise ValueError("The single-step reconciliation preconditions are not met.")
    updated_step = PlanStep.model_validate({
        **plan.steps[0].model_dump(mode="python"),
        "step_id": executed_id,
    })
    return plan.model_copy(update={"steps": [updated_step]}, deep=True)
