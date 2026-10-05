"""Deterministic V2.4 verification; this module never executes plan steps."""

import math
import uuid
from collections.abc import Mapping
from collections import Counter
from typing import Any

import pandas as pd
import numpy as np
from pydantic import ValidationError

from app.core.errors import AppError
from app.multi_agent.schemas import AgentResult, MultiAgentResult
from app.planner.schemas import AnalysisPlan
from app.planner.validator import validate_plan
from app.services.dataset_repository import DatasetRepository
from app.utils.dataframe_utils import load_dataframe
from app.planner.semantic_manifest import load_semantic_manifest
from app.verification.schemas import (
    VerificationCheck,
    VerificationError,
    VerificationResult,
    VerificationStatus,
)

ABSOLUTE_TOLERANCE = 1e-9
_NUMERIC_UNSUPPORTED = {
    "rank",
}
_NON_NUMERIC_OPERATIONS = {"select_columns", "group_by", "sort", "top_n", "bottom_n"}
_V281_OPERATIONS = {"filter_rows", "derive_metric", "group_by", "aggregate",
                    "calculate_percentage_difference", "compare_groups"}


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
    if isinstance(value, dict) and value.get("kind") == "scalar":
        value = value.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _load_frame(repository: DatasetRepository | None, dataset_id: Any) -> pd.DataFrame | None:
    if repository is None or dataset_id is None:
        return None
    repository.get(dataset_id)
    return load_dataframe(repository.get_csv_path(dataset_id))


def _frame_json(frame: pd.DataFrame) -> dict[str, Any]:
    return {"columns": list(frame.columns),
            "rows": frame.where(pd.notna(frame), None).to_dict(orient="records")}


def _same_value(actual: Any, expected: Any) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual is expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isfinite(float(actual)) and math.isfinite(float(expected)) and math.isclose(
            float(actual), float(expected), rel_tol=0.0, abs_tol=ABSOLUTE_TOLERANCE)
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(_same_value(actual[k], expected[k]) for k in actual)
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(_same_value(a, e) for a, e in zip(actual, expected))
    return actual == expected


def _reference_numeric(value: Any, metadata: dict[str, Any] | None = None) -> float:
    if metadata is not None and metadata.get("semantic_role") not in {"additive_measure", "non_additive_measure", "unknown", "boolean"}:
        raise ValueError("reference role is not numeric")
    if isinstance(value, dict) and value.get("kind") == "scalar":
        value = value.get("value")
    number = _as_finite_number(value)
    if number is None:
        raise ValueError("reference is not a finite numeric scalar")
    return number


def _select_comparison_value(agent: AgentResult, selector: Any) -> float:
    value = agent.result
    if selector == "scalar":
        return _reference_numeric(value, agent.metadata)
    if selector.get("kind") == "group_key":
        key = selector["key"]
        groups = value.get("groups") if isinstance(value, dict) else None
        if isinstance(groups, list):
            match = [item.get("value") for item in groups if item.get("key") == key]
            if len(match) != 1:
                raise ValueError("group selector is missing or ambiguous")
            return _reference_numeric(match[0])
        raise ValueError("group selector has no aggregate evidence")
    if selector.get("kind") == "list_item":
        if agent.metadata.get("operation") not in {"rank", "top_n", "bottom_n"}:
            raise ValueError("list selector is not from an ordered result")
        return _reference_numeric(value[selector["index"]][selector["field"]])
    if selector.get("kind") == "leader":
        if agent.metadata.get("operation") not in {"rank", "top_n"}:
            raise ValueError("leader selector is not from rank/top_n")
        return _reference_numeric(value[0]["value"])
    raise ValueError("unsupported selector")


def _reference_expression(expr: Any, frame: pd.DataFrame, roles: dict[str, str],
                          by_step: dict[str, AgentResult]) -> tuple[Any, str]:
    if not isinstance(expr, dict):
        raise ValueError("expression is not structured")
    if set(expr) == {"literal"}:
        val = expr["literal"]
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
            raise ValueError("literal is not finite numeric")
        return float(val), "unknown"
    if set(expr) == {"column"}:
        col = expr["column"]
        if col not in frame.columns:
            raise ValueError("expression column is absent")
        values = pd.to_numeric(frame[col], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError("expression source is missing or nonfinite")
        return values.astype(float), roles.get(col, "unknown")
    if set(expr) == {"step_id"}:
        agent = by_step.get(expr["step_id"])
        if agent is None or agent.status != "completed":
            raise ValueError("expression dependency is unavailable")
        value = agent.result
        if isinstance(value, list):
            number = pd.to_numeric(pd.Series(value), errors="coerce")
            if number.isna().any() or not np.isfinite(number.to_numpy(dtype=float)).all():
                raise ValueError("expression dependency is nonfinite")
            return number, agent.metadata.get("semantic_role", "unknown")
        return _reference_numeric(value, agent.metadata), agent.metadata.get("semantic_role", "unknown")
    if expr.get("op") == "negate" and set(expr) == {"op", "value"}:
        value, role = _reference_expression(expr["value"], frame, roles, by_step)
        return -value, role
    if expr.get("op") not in {"add", "subtract", "multiply", "divide"} or set(expr) != {"op", "left", "right"}:
        raise ValueError("expression operation is unsupported")
    left, lrole = _reference_expression(expr["left"], frame, roles, by_step)
    right, rrole = _reference_expression(expr["right"], frame, roles, by_step)
    op = expr["op"]
    if op == "divide" and ((isinstance(right, (int, float)) and right == 0)
                            or (isinstance(right, pd.Series) and (right == 0).any())):
        raise ValueError("division by zero")
    value = {"add": lambda: left + right, "subtract": lambda: left - right,
             "multiply": lambda: left * right, "divide": lambda: left / right}[op]()
    role = "additive_measure" if op in {"add", "subtract"} and lrole == rrole == "additive_measure" else "unknown"
    return value, role


def _approved_additive_product(expr: Any, manifest) -> bool:
    if not isinstance(expr, dict) or expr.get("op") != "multiply":
        return False
    left, right = expr.get("left"), expr.get("right")
    if not (isinstance(left, dict) and set(left) == {"column"}
            and isinstance(right, dict) and set(right) == {"column"}):
        return False
    pairs = set(manifest.additive_products) if manifest is not None else set()
    return (left["column"], right["column"]) in pairs or (right["column"], left["column"]) in pairs


def _expression_lineage(expr: Any) -> list[dict[str, Any]]:
    if not isinstance(expr, dict):
        return []
    if set(expr) == {"column"}:
        return [{"kind": "column", "name": expr["column"]}]
    if set(expr) == {"step_id"}:
        return [{"kind": "step", "step_id": expr["step_id"]}]
    if expr.get("op") == "negate":
        return _expression_lineage(expr.get("value"))
    return _expression_lineage(expr.get("left")) + _expression_lineage(expr.get("right"))


def _expected_lineage(step, agent: AgentResult, result: MultiAgentResult) -> list[dict[str, Any]]:
    if step.operation in {"filter_rows", "group_by"}:
        if step.depends_on:
            return [{"kind": "step", "step_id": item} for item in step.depends_on]
        dataset_agent = next((item for item in result.agent_results
                              if item.agent_name == "data_understanding" and item.status == "completed"), None)
        if dataset_agent is None or not dataset_agent.metadata.get("dataset_id"):
            return []
        return [{"kind": "dataset", "dataset_id": dataset_agent.metadata["dataset_id"]}]
    if step.operation == "derive_metric":
        return _expression_lineage(step.parameters.get("formula"))
    if step.operation == "aggregate":
        if step.depends_on:
            upstream = next((item for item in result.agent_results if item.agent_name == "analysis"
                             and item.step_id == step.depends_on[0] and item.status == "completed"), None)
            if upstream is None:
                return []
            lineage = list(upstream.metadata.get("lineage", []))
            lineage.append({"kind": "step", "step_id": step.depends_on[0]})
            if upstream.metadata.get("operation") != "derive_metric":
                if len(step.inputs) != 1:
                    return []
                lineage.append({"kind": "column", "name": step.inputs[0]})
            return lineage
        return [{"kind": "column", "name": step.inputs[0]}] if len(step.inputs) == 1 else []
    if step.operation == "calculate_percentage_difference":
        return [{"kind": "step", "step_id": item} for item in step.depends_on]
    if step.operation == "compare_groups":
        return [{"kind": "step", "step_id": step.parameters[side]["step_id"]} for side in ("left", "right")]
    return []


def _v281_expected(step, result: MultiAgentResult, frame: pd.DataFrame,
                   manifest, by_step: dict[str, AgentResult]) -> tuple[Any, Any, list[str]]:
    """Independent reference calculations for the V2.8.1 operation subset."""
    p, op = step.parameters, step.operation
    dependencies = [by_step.get(step_id) for step_id in step.depends_on]
    if any(agent is None or agent.status != "completed" for agent in dependencies):
        raise ValueError("successful dependency evidence is unavailable")
    source_frame = frame
    if dependencies and step.operation in {"filter_rows", "derive_metric", "group_by"}:
        table_agent = next((agent for agent in dependencies if isinstance(agent.result, dict)
                            and isinstance(agent.result.get("columns"), list)
                            and isinstance(agent.result.get("rows"), list)), None)
        if table_agent is not None:
            table = table_agent.result
            source_frame = pd.DataFrame(table["rows"], columns=table["columns"])
        elif step.operation != "derive_metric":
            raise ValueError("table dependency does not expose structured columns and rows")
    if op == "filter_rows":
        selected = source_frame.copy()
        if p.get("mode", "predicate") == "exact_duplicate_rows":
            expected = _frame_json(selected.drop_duplicates(keep="first"))
        else:
            mask = pd.Series(True, index=selected.index)
            for cond in p["conditions"]:
                col, operator = cond["column"], cond["operator"]
                if col not in selected.columns:
                    raise ValueError("filter column is absent")
                series = selected[col]
                missing = series.isna()
                if operator == "is_missing":
                    current = missing
                elif operator == "is_not_missing":
                    current = ~missing
                else:
                    valid = ~missing
                    role = manifest.role_for(col) if manifest else "unknown"
                    literal = cond["value"]
                    if role in {"additive_measure", "non_additive_measure"}:
                        left = pd.to_numeric(series, errors="coerce")
                        if left[valid].isna().any() or not np.isfinite(left[valid].to_numpy(dtype=float)).all():
                            raise ValueError("filter numeric source is invalid")
                        if isinstance(literal, list) and not all(
                            isinstance(x, (int, float)) and not isinstance(x, bool) for x in literal
                        ):
                            raise ValueError("filter numeric membership values have invalid types")
                        if not isinstance(literal, list) and (
                            not isinstance(literal, (int, float)) or isinstance(literal, bool)
                        ):
                            raise ValueError("filter numeric literal has invalid type")
                        right = [float(x) for x in literal] if isinstance(literal, list) else float(literal)
                    else:
                        left, right = series, literal
                        if isinstance(literal, list) and not all(isinstance(x, str) for x in literal):
                            raise ValueError("filter string membership values have invalid types")
                        if not isinstance(literal, (str, list)):
                            raise ValueError("filter string literal has invalid type")
                        if role == "date" and operator in {"lt", "lte", "gt", "gte"}:
                            raise ValueError("date ordering is not frozen")
                        if operator in {"lt", "lte", "gt", "gte"}:
                            raise ValueError("ordered string predicates are unsupported")
                    if operator in {"eq", "in"}:
                        current = left.isin(right if isinstance(right, list) else [right])
                    elif operator in {"ne", "not_in"}:
                        current = ~left.isin(right if isinstance(right, list) else [right])
                    else:
                        current = {"lt": left < right, "lte": left <= right,
                                   "gt": left > right, "gte": left >= right}[operator]
                    current &= valid
                mask &= current.fillna(False)
            expected = _frame_json(selected.loc[mask])
        return expected, "dataset_snapshot+plan+semantic_manifest", []
    if op == "group_by":
        col = p["column"]
        if col not in source_frame.columns:
            raise ValueError("grouping column is absent")
        keyed = source_frame.copy()
        missing = keyed[col].isna()
        if missing.any() and keyed.loc[~missing, col].astype(str).eq("(missing)").any():
            raise ValueError("missing group label collides with literal value")
        keyed[col] = keyed[col].where(~missing, "(missing)")
        expected = {str(key): _frame_json(group) for key, group in keyed.groupby(col, sort=True, dropna=False)}
        return expected, "dataset_snapshot+plan", []
    if op == "derive_metric":
        expected, role = _reference_expression(p["formula"], source_frame, manifest.columns if manifest else {}, by_step)
        if _approved_additive_product(p["formula"], manifest):
            role = "additive_measure"
        if isinstance(expected, pd.Series):
            expected = expected.tolist()
        if agent := by_step.get(step.step_id):
            if agent.metadata.get("semantic_role") != role:
                raise ValueError("reported derived semantic role disagrees with independent typing")
            if agent.metadata.get("output_name") != p.get("output_name"):
                raise ValueError("reported derived output name disagrees with the plan")
        return expected, "dataset_snapshot+plan+dependency_outputs+lineage", []
    if op == "aggregate":
        if p.get("function") != "sum" or len(step.inputs) != 1:
            raise ValueError("aggregate request is invalid")
        col = step.inputs[0]
        raw: Any = frame
        if dependencies:
            raw = dependencies[0].result
        role = manifest.role_for(col) if manifest else "unknown"
        if isinstance(raw, dict) and "columns" in raw and "rows" in raw:
            raw = pd.DataFrame(raw["rows"], columns=raw["columns"])
        if isinstance(raw, dict) and raw and all(isinstance(v, dict) and "columns" in v for v in raw.values()):
            groups = []
            for key in sorted(raw):
                sub = pd.DataFrame(raw[key]["rows"], columns=raw[key]["columns"])
                if role != "additive_measure" or col not in sub.columns:
                    raise ValueError("grouped source is not a trusted additive measure")
                nums = pd.to_numeric(sub[col], errors="coerce")
                if nums.empty or nums.isna().any() or not np.isfinite(nums.to_numpy(dtype=float)).all():
                    raise ValueError("grouped values are empty, missing, or nonfinite")
                groups.append({"key": str(key), "value": float(nums.sum())})
            return {"kind": "grouped", "groups": groups}, "dataset_snapshot+plan+dependencies+lineage", []
        if isinstance(raw, list):
            if dependencies[0].metadata.get("semantic_role") != "additive_measure":
                raise ValueError("derived dependency is not typed additive")
            nums = pd.to_numeric(pd.Series(raw), errors="coerce")
        elif isinstance(raw, pd.DataFrame) and col in raw.columns:
            if role != "additive_measure":
                raise ValueError("source column is not a trusted additive measure")
            nums = pd.to_numeric(raw[col], errors="coerce")
        else:
            raise ValueError("aggregate input evidence is unavailable")
        if nums.empty or nums.isna().any() or not np.isfinite(nums.to_numpy(dtype=float)).all():
            raise ValueError("aggregate input is empty, missing, invalid, or nonfinite")
        return {"kind": "scalar", "value": float(nums.sum())}, "dataset_snapshot+semantic_manifest+lineage", []
    if op == "calculate_percentage_difference":
        if len(dependencies) != 2:
            raise ValueError("percentage difference requires two dependencies")
        value = _reference_numeric(dependencies[0].result, dependencies[0].metadata)
        reference = _reference_numeric(dependencies[1].result, dependencies[1].metadata)
        if reference == 0:
            raise ValueError("zero reference value")
        return ((value - reference) / reference) * 100, "successful_dependency_outputs", step.depends_on
    if op == "compare_groups":
        operands = []
        for side in ("left", "right"):
            spec = p[side]
            upstream = by_step.get(spec["step_id"])
            if upstream is None or upstream.status != "completed":
                raise ValueError("comparison dependency is unavailable")
            operands.append(_select_comparison_value(upstream, spec["selector"]) * float(spec["multiplier"]))
        comparator = p["comparator"]
        truth = {"lt": operands[0] < operands[1], "lte": operands[0] <= operands[1],
                 "eq": operands[0] == operands[1], "gte": operands[0] >= operands[1],
                 "gt": operands[0] > operands[1]}[comparator]
        return {"left_value": operands[0], "comparator": comparator,
                "right_value": operands[1], "result": truth}, "successful_dependency_outputs", step.depends_on
    raise ValueError("no V2.8.1 reference rule for operation")


def _semantic_operation_checks(plan: AnalysisPlan, result: MultiAgentResult,
                               repository: DatasetRepository | None, dataset_id: Any) -> list[VerificationCheck]:
    by_step = _agent_step_index(result)
    frame: pd.DataFrame | None = None
    manifest = None
    checks: list[VerificationCheck] = []
    for step in plan.steps:
        agent = by_step.get(step.step_id)
        if step.operation not in _V281_OPERATIONS or agent is None or agent.status != "completed":
            continue
        try:
            if not isinstance(agent.metadata.get("lineage"), list) or not agent.metadata["lineage"]:
                raise ValueError("structured lineage evidence is unavailable")
            if step.operation in {"filter_rows", "derive_metric", "group_by", "aggregate"}:
                if frame is None:
                    if repository is None or dataset_id is None:
                        raise ValueError("dataset evidence is unavailable")
                    path = repository.get_csv_path(dataset_id)
                    frame = load_dataframe(path)
                    manifest = load_semantic_manifest(path)
                execution_context = next((item for item in result.agent_results
                                          if item.agent_name == "data_understanding" and item.status == "completed"), None)
                if execution_context is None:
                    raise ValueError("successful dataset-understanding evidence is unavailable")
                version = getattr(manifest, "manifest_version", None)
                digest = getattr(manifest, "dataset_sha256", None)
                if execution_context.metadata.get("semantic_manifest_version") != version:
                    raise ValueError("execution and verification semantic manifest versions differ")
                if execution_context.metadata.get("dataset_sha256") != digest:
                    raise ValueError("execution and verification dataset snapshot hashes differ")
            expected, source, dependency_ids = _v281_expected(
                step, result, frame if frame is not None else pd.DataFrame(), manifest, by_step
            )
            if step.operation == "aggregate" and agent.metadata.get("semantic_role") != "additive_measure":
                raise ValueError("aggregate output is not marked as an additive measure")
            if step.operation == "calculate_percentage_difference" and agent.metadata.get("semantic_role") != "non_additive_measure":
                raise ValueError("percentage output semantic role is invalid")
            if step.operation == "compare_groups" and agent.metadata.get("semantic_role") != "boolean":
                raise ValueError("comparison output semantic role is invalid")
            expected_lineage = _expected_lineage(step, agent, result)
            if not expected_lineage:
                raise ValueError("lineage cannot be reconstructed from the validated plan and dependencies")
            if agent.metadata.get("lineage") != expected_lineage:
                checks.append(_check(
                    "semantic_operation_verification", "failed",
                    "The surfaced operation lineage differs from the validated plan and successful dependencies.",
                    category="consistency_violation", step_ids=[step.step_id],
                    expected=expected_lineage, observed=agent.metadata.get("lineage"),
                    evidence_reference=f"agent_results:{step.step_id}",
                    metadata={"operation": step.operation},
                ))
                continue
            observed = agent.result
            passed = _same_value(observed, expected)
            checks.append(_check(
                "semantic_operation_verification", "passed" if passed else "failed",
                "The operation result matches independent recomputation." if passed
                else "The operation result differs from independent recomputation.",
                category=None if passed else "numerical_mismatch",
                step_ids=[step.step_id], expected=expected, observed=observed,
                evidence_reference=(f"dataset:{dataset_id};" if step.operation in {"filter_rows", "derive_metric", "group_by", "aggregate"} else "")
                    + f"agent_results:{','.join(dependency_ids + [step.step_id])}",
                metadata={"operation": step.operation, "semantic_manifest_version": getattr(manifest, "manifest_version", None),
                          "evidence_source": source, "absolute_tolerance": ABSOLUTE_TOLERANCE},
            ))
        except (ValueError, AppError) as exc:
            checks.append(_check(
                "semantic_operation_verification", "unsupported",
                "Independent verification evidence or frozen semantics are unavailable.",
                category="verification_semantics_undefined", step_ids=[step.step_id],
                observed=agent.result, evidence_reference=f"agent_results:{step.step_id}",
                metadata={"operation": step.operation, "reason": type(exc).__name__},
            ))
        except Exception as exc:
            checks.append(_check(
                "semantic_operation_verification", "failed",
                "An unexpected verifier exception prevented independent verification.",
                category="verification_failure", step_ids=[step.step_id],
                observed=agent.result, evidence_reference=f"agent_results:{step.step_id}",
                metadata={"operation": step.operation, "exception_type": type(exc).__name__},
            ))
    return checks


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
        if operation in _V281_OPERATIONS:
            continue
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
        checks.extend(_semantic_operation_checks(plan, result, repository, dataset_id))
        grounding, evidence = _grounding_checks(result)
        checks.extend(grounding)
        checks.extend(_consistency_checks(plan, result))
        verified_steps = _verified_steps(plan, result, checks)
        return _result(checks, verified_steps=verified_steps, evidence=evidence)
    except (ValidationError, TypeError, ValueError) as exc:
        return _schema_failure(exc)
    except Exception:
        return _unexpected_failure()
