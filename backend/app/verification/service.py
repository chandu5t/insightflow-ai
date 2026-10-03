"""Deterministic V2.4 verification; this module never executes plan steps."""

import math
import uuid
from collections.abc import Mapping
from collections import Counter
from typing import Any

import pandas as pd
from pydantic import ValidationError

from app.core.errors import AppError
from app.multi_agent.schemas import AgentResult, MultiAgentResult
from app.planner.schemas import AnalysisPlan
from app.planner.validator import validate_plan
from app.services.dataset_repository import DatasetRepository
from app.utils.dataframe_utils import load_dataframe
from app.verification.schemas import (
    VerificationCheck,
    VerificationError,
    VerificationResult,
    VerificationStatus,
)

ABSOLUTE_TOLERANCE = 1e-9
_NUMERIC_UNSUPPORTED = {
    "aggregate",
    "calculate_percentage_difference",
    "compare_groups",
    "derive_metric",
    "rank",
}
_NON_NUMERIC_OPERATIONS = {"select_columns", "group_by", "sort", "top_n", "bottom_n"}


def _check(
    name: str,
    status: VerificationStatus,
    message: str,
    *,
    category: str | None = None,
    step_ids: list[str] | None = None,
    expected: Any = None,
    observed: Any = None,
    evidence_reference: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> VerificationCheck:
    return VerificationCheck(
        name=name,
        status=status,
        message=message,
        category=category,
        step_ids=step_ids or [],
        expected=expected,
        observed=observed,
        evidence_reference=evidence_reference,
        metadata=metadata or {},
    )


def _result(checks: list[VerificationCheck], *, verified_steps: list[str] | None = None,
            evidence: list[str] | None = None, errors: list[VerificationError] | None = None) -> VerificationResult:
    failed = [check for check in checks if check.status == "failed"]
    unsupported = [check for check in checks if check.status == "unsupported"]
    status: VerificationStatus = "failed" if failed else "unsupported" if unsupported else "passed"
    return VerificationResult(
        status=status,
        checks=checks,
        failed_checks=failed,
        unsupported_checks=unsupported,
        verified_steps=verified_steps or [],
        evidence=evidence or [],
        errors=errors or [],
        metadata={"request_id": str(uuid.uuid4()), "tolerance_absolute": ABSOLUTE_TOLERANCE},
    )


def _schema_failure(exc: Exception) -> VerificationResult:
    check = _check(
        "schema_validation", "failed", "The V2.3 workflow result does not match its structured schema.",
        category="invalid_result_schema", metadata={"validation_error_type": type(exc).__name__},
    )
    error = VerificationError(
        category="invalid_result_schema", code="INVALID_RESULT_SCHEMA",
        message="The V2.3 workflow result could not be validated.",
    )
    return _result([check], errors=[error])


def _unexpected_failure() -> VerificationResult:
    check = _check(
        "verification_service", "failed", "An unexpected error prevented verification.",
        category="verification_failure",
    )
    error = VerificationError(
        category="verification_failure", code="VERIFICATION_FAILURE",
        message="Verification could not be completed safely.",
    )
    return _result([check], errors=[error])


def _custom_result_schema_errors(result: MultiAgentResult) -> list[str]:
    """Check cross-field invariants not expressed by the V2.3 Pydantic models."""
    errors: list[str] = []
    for index, agent in enumerate(result.agent_results):
        if agent.agent_name == "analysis" and not agent.step_id:
            errors.append(f"agent_results[{index}] analysis result is missing step_id")
        if agent.status == "failed" and agent.error is None:
            errors.append(f"agent_results[{index}] failed result is missing its structured error")
        if agent.status == "completed" and agent.error is not None:
            errors.append(f"agent_results[{index}] completed result unexpectedly contains an error")
    return errors


def _required_result_fields(raw: Any) -> list[str]:
    """Reject omitted serialized V2.3 fields instead of accepting Pydantic defaults."""
    if isinstance(raw, MultiAgentResult):
        raw = raw.model_dump(mode="python")
    if not isinstance(raw, Mapping):
        return ["workflow result must be an object"]
    errors = [f"workflow result is missing field '{name}'" for name in (
        "workflow_status", "executed_steps", "step_statuses", "agent_results", "final_result", "errors"
    ) if name not in raw]
    agents = raw.get("agent_results")
    if isinstance(agents, list):
        for index, agent in enumerate(agents):
            if isinstance(agent, Mapping):
                errors.extend(
                    f"agent_results[{index}] is missing field '{name}'"
                    for name in ("agent_name", "step_id", "status", "result", "metadata", "error")
                    if name not in agent
                )
    return errors


def _agent_step_index(result: MultiAgentResult) -> dict[str, AgentResult]:
    return {
        agent.step_id: agent
        for agent in result.agent_results
        if agent.agent_name == "analysis" and agent.step_id is not None
    }


def _plan_execution_checks(plan: AnalysisPlan, result: MultiAgentResult) -> list[VerificationCheck]:
    checks: list[VerificationCheck] = []
    plan_errors = validate_plan(plan)
    if plan_errors:
        checks.append(_check(
            "plan_validation", "failed", "The supplied plan is not a valid V2.2 plan.",
            category="plan_execution_mismatch",
            metadata={"validation_codes": [error.code for error in plan_errors]},
        ))

    plan_ids = [step.step_id for step in plan.steps]
    plan_by_id = {step.step_id: step for step in plan.steps}
    result_by_step = _agent_step_index(result)
    agent_step_ids = [agent.step_id for agent in result.agent_results if agent.step_id is not None]
    duplicate_agent_steps = [step_id for step_id, count in Counter(agent_step_ids).items() if count > 1]
    if duplicate_agent_steps:
        checks.append(_check(
            "agent_step_uniqueness", "failed", "Multiple Analysis Agent results claim the same step ID.",
            category="plan_execution_mismatch", step_ids=sorted(duplicate_agent_steps),
        ))
    duplicates = [step_id for step_id, count in Counter(result.executed_steps).items() if count > 1]
    if duplicates:
        checks.append(_check(
            "executed_step_uniqueness", "failed", "Execution contains duplicate step IDs.",
            category="plan_execution_mismatch", step_ids=sorted(duplicates),
        ))

    unexpected = [step_id for step_id in result.executed_steps if step_id not in plan_by_id]
    unexpected_statuses = [step_id for step_id in result.step_statuses if step_id not in plan_by_id]
    unexpected.extend(step_id for step_id in unexpected_statuses if step_id not in unexpected)
    if unexpected:
        checks.append(_check(
            "unexpected_executed_steps", "failed", "Execution contains steps absent from the plan.",
            category="plan_execution_mismatch", step_ids=unexpected,
        ))

    plan_order = {step_id: index for index, step_id in enumerate(plan_ids)}
    planned_execution_order = [step_id for step_id in result.executed_steps if step_id in plan_order]
    if planned_execution_order != sorted(planned_execution_order, key=plan_order.__getitem__):
        checks.append(_check(
            "execution_order", "failed", "Executed planned steps do not preserve plan order.",
            category="plan_execution_mismatch", step_ids=planned_execution_order,
            expected=[step_id for step_id in plan_ids if step_id in set(planned_execution_order)],
            observed=planned_execution_order,
        ))

    missing = [step_id for step_id in plan_ids if step_id not in result.executed_steps]
    # A failed workflow may stop before later steps; those must be explicitly
    # represented as pending/failed, never as silently omitted completed work.
    silently_missing = [
        step_id for step_id in missing
        if result.workflow_status == "completed" or result.step_statuses.get(step_id) == "completed"
    ]
    if silently_missing:
        checks.append(_check(
            "planned_step_coverage", "failed", "A required planned step is missing from execution.",
            category="plan_execution_mismatch", step_ids=silently_missing,
        ))
    elif missing:
        checks.append(_check(
            "planned_step_coverage", "unsupported",
            "The workflow stopped before all planned steps produced execution results.",
            category="verification_semantics_undefined", step_ids=missing,
        ))
    else:
        checks.append(_check(
            "planned_step_coverage", "passed", "Planned steps are represented consistently with workflow status.",
            step_ids=plan_ids,
        ))
    unrepresented = [
        step_id for step_id in missing
        if result.step_statuses.get(step_id) not in {"pending", "running", "failed", "completed"}
    ]
    if unrepresented:
        checks.append(_check(
            "planned_step_status", "unsupported",
            "The result does not establish whether an unexecuted planned step was pending, running, or failed.",
            category="verification_semantics_undefined", step_ids=unrepresented,
        ))

    for step_id in result.executed_steps:
        planned = plan_by_id.get(step_id)
        agent = result_by_step.get(step_id)
        if planned is None:
            continue
        if agent is None:
            other_agent = next((item for item in result.agent_results if item.step_id == step_id), None)
            if other_agent is not None:
                checks.append(_check(
                    "step_agent_identity", "failed",
                    "A planned analytical step was returned by an agent other than the Analysis Agent.",
                    category="plan_execution_mismatch", step_ids=[step_id],
                    expected="analysis", observed=other_agent.agent_name,
                ))
                continue
            checks.append(_check(
                "step_operation_identity", "unsupported",
                "No structured Analysis Agent result is available for the executed step.",
                category="verification_semantics_undefined", step_ids=[step_id],
            ))
            continue
        operation = agent.metadata.get("operation")
        if operation != planned.operation:
            checks.append(_check(
                "step_operation_identity", "failed",
                "The executed operation does not match the planned operation.",
                category="plan_execution_mismatch", step_ids=[step_id],
                expected=planned.operation, observed=operation,
            ))
        else:
            checks.append(_check(
                "step_operation_identity", "passed", "Executed operation matches the plan.",
                step_ids=[step_id], expected=planned.operation, observed=operation,
                evidence_reference=f"agent_results:{step_id}",
            ))
        if agent.status != "completed" or result.step_statuses.get(step_id) != "completed":
            checks.append(_check(
                "executed_step_status", "failed", "An executed step is not recorded as completed.",
                category="plan_execution_mismatch", step_ids=[step_id],
                expected="completed", observed=result.step_statuses.get(step_id),
            ))

    order = {step_id: index for index, step_id in enumerate(result.executed_steps)}
    for step in plan.steps:
        for dependency in step.depends_on:
            if dependency not in plan_by_id:
                checks.append(_check(
                    "dependency_validation", "failed", "A declared dependency step does not exist.",
                    category="plan_execution_mismatch", step_ids=[step.step_id, dependency],
                ))
                continue
            if step.step_id in order and dependency in order:
                dependency_status = result.step_statuses.get(dependency)
                if order[dependency] >= order[step.step_id] or dependency_status != "completed":
                    checks.append(_check(
                        "dependency_validation", "failed",
                        "A dependency did not complete successfully before its dependent step.",
                        category="plan_execution_mismatch", step_ids=[dependency, step.step_id],
                        expected="dependency completed first", observed=dependency_status,
                    ))
                else:
                    checks.append(_check(
                        "dependency_validation", "passed", "Declared dependency completed before the step.",
                        step_ids=[dependency, step.step_id],
                        evidence_reference=f"executed_steps:{dependency}->{step.step_id}",
                    ))
            elif step.step_id in order and result.workflow_status == "completed":
                checks.append(_check(
                    "dependency_validation", "failed",
                    "A dependent step executed without its declared dependency.",
                    category="plan_execution_mismatch", step_ids=[dependency, step.step_id],
                ))
            elif step.step_id in order:
                dependency_status = result.step_statuses.get(dependency)
                if dependency_status in {"pending", "running", "failed"}:
                    checks.append(_check(
                        "dependency_validation", "failed",
                        "A dependent step executed before its dependency completed successfully.",
                        category="plan_execution_mismatch", step_ids=[dependency, step.step_id],
                        expected="completed", observed=dependency_status,
                    ))
                else:
                    checks.append(_check(
                        "dependency_validation", "unsupported",
                        "Dependency completion cannot be established from the workflow result.",
                        category="verification_semantics_undefined", step_ids=[dependency, step.step_id],
                    ))
    return checks


def _as_finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _load_frame(repository: DatasetRepository | None, dataset_id: Any) -> pd.DataFrame | None:
    if repository is None or dataset_id is None:
        return None
    repository.get(dataset_id)
    return load_dataframe(repository.get_csv_path(dataset_id))


def _numeric_checks(
    plan: AnalysisPlan,
    result: MultiAgentResult,
    repository: DatasetRepository | None,
    dataset_id: Any,
) -> list[VerificationCheck]:
    by_step = _agent_step_index(result)
    plan_by_id = {step.step_id: step for step in plan.steps}
    executed = set(result.executed_steps)
    frame: pd.DataFrame | None = None
    frame_error: str | None = None
    checks: list[VerificationCheck] = []

    for step_id in result.executed_steps:
        step = plan_by_id.get(step_id)
        agent = by_step.get(step_id)
        if step is None or agent is None or agent.status != "completed":
            continue
        operation = step.operation
        if operation in _NON_NUMERIC_OPERATIONS:
            continue
        if operation in _NUMERIC_UNSUPPORTED:
            checks.append(_check(
                "numerical_verification", "unsupported",
                f"V2.3 does not freeze numerical verification semantics for '{operation}'.",
                category="verification_semantics_undefined", step_ids=[step_id],
                observed=agent.result, evidence_reference=f"agent_results:{step_id}",
                metadata={"operation": operation},
            ))
            continue
        if operation not in {"count", "distinct_count", "calculate_difference"}:
            checks.append(_check(
                "numerical_verification", "unsupported",
                f"No frozen numerical verification rule exists for '{operation}'.",
                category="verification_semantics_undefined", step_ids=[step_id],
                observed=agent.result, evidence_reference=f"agent_results:{step_id}",
                metadata={"operation": operation},
            ))
            continue

        expected: Any = None
        observed = agent.result
        evidence: str | None = f"agent_results:{step_id}"
        try:
            if operation in {"count", "distinct_count"}:
                if step.depends_on:
                    checks.append(_check(
                        "numerical_verification", "unsupported",
                        "Independent dataset-level verification is not defined for this dependent operation result.",
                        category="verification_semantics_undefined", step_ids=[step_id],
                        observed=observed, evidence_reference=evidence,
                        metadata={"operation": operation},
                    ))
                    continue
                if operation == "count" and step.inputs:
                    checks.append(_check(
                        "numerical_verification", "unsupported",
                        "V2.3 does not execute named count inputs without declared dependency results.",
                        category="verification_semantics_undefined", step_ids=[step_id],
                        observed=observed, evidence_reference=evidence,
                        metadata={"operation": operation},
                    ))
                    continue
                if frame is None and frame_error is None:
                    try:
                        frame = _load_frame(repository, dataset_id)
                    except AppError as exc:
                        frame_error = exc.code.value
                if frame is None:
                    checks.append(_check(
                        "numerical_verification", "unsupported",
                        "Dataset context required for independent verification is unavailable.",
                        category="verification_semantics_undefined", step_ids=[step_id],
                        observed=observed, evidence_reference=evidence,
                        metadata={"operation": operation, "dataset_error": frame_error},
                    ))
                    continue
                if operation == "count":
                    if "column" in step.parameters:
                        checks.append(_check(
                            "numerical_verification", "unsupported",
                            "V2.3 does not define count(column) verification semantics.",
                            category="verification_semantics_undefined", step_ids=[step_id],
                            observed=observed, evidence_reference=evidence,
                            metadata={"operation": operation},
                        ))
                        continue
                    expected = len(frame)
                else:
                    column = step.parameters.get("column")
                    if not isinstance(column, str) or column not in frame.columns:
                        checks.append(_check(
                            "numerical_verification", "unsupported",
                            "The distinct-count source column is unavailable.",
                            category="verification_semantics_undefined", step_ids=[step_id],
                            observed=observed, evidence_reference=evidence,
                            metadata={"operation": operation},
                        ))
                        continue
                    if step.inputs and column not in step.inputs:
                        checks.append(_check(
                            "numerical_verification", "unsupported",
                            "The distinct-count column is not included in the operation inputs.",
                            category="verification_semantics_undefined", step_ids=[step_id],
                            observed=observed, evidence_reference=evidence,
                            metadata={"operation": operation, "column": column},
                        ))
                        continue
                    if frame[column].isna().any():
                        checks.append(_check(
                            "numerical_verification", "unsupported",
                            "V2.3 rejects distinct_count when missing values affect the result.",
                            category="verification_semantics_undefined", step_ids=[step_id],
                            observed=observed, evidence_reference=evidence,
                            metadata={"operation": operation, "column": column},
                        ))
                        continue
                    expected = int(frame[column].nunique(dropna=True))
                matches = isinstance(observed, int) and not isinstance(observed, bool) and observed == expected
            else:
                if len(step.depends_on) != 2:
                    checks.append(_check(
                        "numerical_verification", "unsupported",
                        "calculate_difference requires two declared dependency results.",
                        category="verification_semantics_undefined", step_ids=[step_id],
                        observed=observed, evidence_reference=evidence,
                    ))
                    continue
                dependency_agents = [by_step.get(dependency) for dependency in step.depends_on]
                if any(agent is None or agent.status != "completed" for agent in dependency_agents):
                    checks.append(_check(
                        "numerical_verification", "unsupported",
                        "The structured successful dependency values required for recalculation are unavailable.",
                        category="verification_semantics_undefined", step_ids=[*step.depends_on, step_id],
                        observed=observed, evidence_reference=evidence,
                    ))
                    continue
                left = _as_finite_number(dependency_agents[0].result)
                right = _as_finite_number(dependency_agents[1].result)
                actual = _as_finite_number(observed)
                if left is None or right is None or actual is None:
                    checks.append(_check(
                        "numerical_verification", "unsupported",
                        "The dependency or observed value is not a finite numeric scalar.",
                        category="verification_semantics_undefined", step_ids=[*step.depends_on, step_id],
                        observed=observed, evidence_reference=evidence,
                    ))
                    continue
                expected = left - right
                matches = math.isclose(actual, expected, rel_tol=0.0, abs_tol=ABSOLUTE_TOLERANCE)
                evidence = f"agent_results:{step.depends_on[0]},{step.depends_on[1]},{step_id}"

            checks.append(_check(
                "numerical_verification", "passed" if matches else "failed",
                "The numerical result matches the independent calculation." if matches
                else "The numerical result differs from the independent calculation.",
                category=None if matches else "numerical_mismatch", step_ids=[step_id],
                expected=expected, observed=observed, evidence_reference=evidence,
                metadata={"operation": operation, "absolute_tolerance": ABSOLUTE_TOLERANCE},
            ))
        except AppError as exc:
            checks.append(_check(
                "numerical_verification", "unsupported",
                "Dataset context required for independent verification is unavailable.",
                category="verification_semantics_undefined", step_ids=[step_id],
                observed=observed, evidence_reference=evidence,
                metadata={"operation": operation, "dataset_error": exc.code.value},
            ))
    return checks


def _grounding_checks(result: MultiAgentResult) -> tuple[list[VerificationCheck], list[str]]:
    successful = [
        (index, agent) for index, agent in enumerate(result.agent_results)
        if agent.status == "completed" and agent.result is not None
    ]
    evidence = [
        f"agent_results[{index}]" + (f":{agent.step_id}" if agent.step_id else f":{agent.agent_name}")
        for index, agent in successful
    ]
    checks: list[VerificationCheck] = []
    if result.final_result is None:
        checks.append(_check(
            "grounding", "passed", "No structured final value was reported.",
            metadata={"target": "final_result"},
        ))
    else:
        match = next(((index, agent) for index, agent in successful if agent.result == result.final_result), None)
        if match is None:
            checks.append(_check(
                "grounding", "failed", "The structured final result is not present in a successful agent result.",
                category="unsupported_claim", observed=result.final_result,
                metadata={"target": "final_result"},
            ))
        else:
            index, agent = match
            ref = f"agent_results[{index}]" + (f":{agent.step_id}" if agent.step_id else f":{agent.agent_name}")
            checks.append(_check(
                "grounding", "passed", "The structured final result traces to a successful agent result.",
                evidence_reference=ref, metadata={"target": "final_result"},
            ))
    for index, agent in successful:
        checks.append(_check(
            "agent_result_grounding", "passed",
            "The surfaced structured agent result is its own traceable execution evidence.",
            step_ids=[agent.step_id] if agent.step_id else [],
            evidence_reference=f"agent_results[{index}]" + (f":{agent.step_id}" if agent.step_id else f":{agent.agent_name}"),
            metadata={"agent_name": agent.agent_name},
        ))
    return checks, evidence


def _consistency_checks(plan: AnalysisPlan, result: MultiAgentResult) -> list[VerificationCheck]:
    checks: list[VerificationCheck] = []
    final_required = bool(plan.steps) or plan.intent == "metric_definition"
    if final_required and result.final_result is None:
        if result.workflow_status == "completed":
            checks.append(_check(
                "final_result_consistency", "failed",
                "A completed workflow is missing its required final result.",
                category="consistency_violation",
            ))
        else:
            checks.append(_check(
                "final_result_consistency", "unsupported",
                "The workflow did not produce the required final result.",
                category="verification_semantics_undefined",
            ))
    else:
        checks.append(_check(
            "final_result_consistency", "passed",
            "Final result presence is consistent with the plan and workflow.",
        ))

    if result.workflow_status == "completed":
        incomplete = [step.step_id for step in plan.steps if result.step_statuses.get(step.step_id) != "completed"]
        failed_agents = [agent.step_id for agent in result.agent_results if agent.status == "failed"]
        if incomplete or failed_agents or result.errors:
            checks.append(_check(
                "workflow_step_consistency", "failed",
                "Workflow is completed despite incomplete or failed execution state.",
                category="consistency_violation", step_ids=[step for step in incomplete + failed_agents if step],
                metadata={"error_count": len(result.errors)},
            ))
        else:
            checks.append(_check(
                "workflow_step_consistency", "passed",
                "Completed workflow status agrees with completed planned steps.",
                step_ids=[step.step_id for step in plan.steps],
            ))
    else:
        checks.append(_check(
            "workflow_step_consistency", "passed",
            "Failed workflow status is represented explicitly; incomplete steps are not treated as completed.",
            step_ids=[step_id for step_id, status in result.step_statuses.items() if status == "failed"],
        ))

    duplicates = [step_id for step_id, count in Counter(result.executed_steps).items() if count > 1]
    if duplicates:
        checks.append(_check(
            "step_status_consistency", "failed", "A step appears more than once in executed_steps.",
            category="consistency_violation", step_ids=sorted(duplicates),
        ))
    else:
        bad_status = [step_id for step_id in result.executed_steps if result.step_statuses.get(step_id) != "completed"]
        if bad_status:
            checks.append(_check(
                "step_status_consistency", "failed", "Executed steps are not all recorded as completed.",
                category="consistency_violation", step_ids=bad_status,
            ))
        else:
            checks.append(_check(
                "step_status_consistency", "passed", "Executed-step list agrees with per-step statuses."
            ))
    return checks


def _verified_steps(
    plan: AnalysisPlan, result: MultiAgentResult, checks: list[VerificationCheck]
) -> list[str]:
    if any(check.name == "plan_validation" and check.status == "failed" for check in checks):
        return []
    by_step = _agent_step_index(result)
    plan_by_id = {step.step_id: step for step in plan.steps}
    verified: list[str] = []
    for step_id in result.executed_steps:
        step = plan_by_id.get(step_id)
        agent = by_step.get(step_id)
        if (step is None or agent is None or agent.status != "completed"
                or result.step_statuses.get(step_id) != "completed"
                or agent.metadata.get("operation") != step.operation):
            continue
        relevant = [check for check in checks if step_id in check.step_ids]
        if all(check.status == "passed" for check in relevant):
            verified.append(step_id)
    return verified


def verify_execution(
    plan: AnalysisPlan,
    workflow_result: Any,
    *,
    repository: DatasetRepository | None = None,
    dataset_id: Any = None,
) -> VerificationResult:
    """Verify a serialized V2.3 result without changing or re-executing it."""
    try:
        missing_fields = _required_result_fields(workflow_result)
        if missing_fields:
            raise ValueError("; ".join(missing_fields))
        result = MultiAgentResult.model_validate(workflow_result)
        custom_errors = _custom_result_schema_errors(result)
        if custom_errors:
            raise ValueError("; ".join(custom_errors))
        checks: list[VerificationCheck] = [
            _check("schema_validation", "passed", "V2.3 workflow result matches its structured schema.")
        ]
        checks.extend(_plan_execution_checks(plan, result))
        checks.extend(_numeric_checks(plan, result, repository, dataset_id))
        grounding, evidence = _grounding_checks(result)
        checks.extend(grounding)
        checks.extend(_consistency_checks(plan, result))
        verified_steps = _verified_steps(plan, result, checks)
        return _result(checks, verified_steps=verified_steps, evidence=evidence)
    except (ValidationError, TypeError, ValueError) as exc:
        return _schema_failure(exc)
    except Exception:
        return _unexpected_failure()
