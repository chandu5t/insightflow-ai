"""Structural validation for V2.2 plans. This is not an execution validator."""

from collections import defaultdict

from app.planner.operations import OPERATION_REGISTRY
from app.planner.schemas import AnalysisPlan, PlanValidationError


def validate_plan(plan: AnalysisPlan) -> list[PlanValidationError]:
    errors: list[PlanValidationError] = []
    first_by_id: dict[str, int] = {}
    for index, step in enumerate(plan.steps):
        if step.step_id in first_by_id:
            errors.append(PlanValidationError(
                code="DUPLICATE_STEP_ID", message=f"Duplicate step ID: {step.step_id}", step_id=step.step_id
            ))
        else:
            first_by_id[step.step_id] = index

    dependencies: dict[str, list[str]] = defaultdict(list)
    for index, step in enumerate(plan.steps):
        operation = OPERATION_REGISTRY.get(step.operation)
        if operation is None:
            errors.append(PlanValidationError(
                code="UNKNOWN_OPERATION", message=f"Unsupported operation: {step.operation}", step_id=step.step_id
            ))
        else:
            missing = [name for name in operation.required_parameters if name not in step.parameters]
            if missing:
                errors.append(PlanValidationError(
                    code="MISSING_REQUIRED_PARAMETER",
                    message=f"Operation '{step.operation}' requires parameter(s): {', '.join(missing)}",
                    step_id=step.step_id,
                ))
            unexpected = [name for name in step.parameters if name not in operation.allowed_parameters]
            if unexpected:
                errors.append(PlanValidationError(
                    code="UNSUPPORTED_PARAMETER",
                    message=f"Operation '{step.operation}' does not allow parameter(s): {', '.join(sorted(unexpected))}",
                    step_id=step.step_id,
                ))

        for dependency in step.depends_on:
            if dependency == step.step_id:
                errors.append(PlanValidationError(
                    code="SELF_DEPENDENCY", message="A step cannot depend on itself.", step_id=step.step_id
                ))
            if dependency not in first_by_id and not any(item.step_id == dependency for item in plan.steps):
                errors.append(PlanValidationError(
                    code="UNKNOWN_DEPENDENCY", message=f"Dependency references unknown step: {dependency}", step_id=step.step_id
                ))
                continue
            dependencies[step.step_id].append(dependency)
            dependency_index = next((i for i, item in enumerate(plan.steps) if item.step_id == dependency), None)
            if dependency_index is not None and dependency_index >= index:
                errors.append(PlanValidationError(
                    code="INVALID_DEPENDENCY_ORDER",
                    message=f"Dependency '{dependency}' must appear before step '{step.step_id}'.",
                    step_id=step.step_id,
                ))

    # DFS over known IDs; a forward acyclic reference is reported as ordering-invalid,
    # while a cycle receives its own explicit error as well.
    visiting: set[str] = set()
    visited: set[str] = set()
    cycle_steps: set[str] = set()

    def visit(step_id: str, trail: list[str]) -> None:
        if step_id in visiting:
            cycle_steps.update(trail[trail.index(step_id):] if step_id in trail else [step_id])
            return
        if step_id in visited:
            return
        visiting.add(step_id)
        for dependency in dependencies.get(step_id, []):
            if dependency in first_by_id:
                visit(dependency, [*trail, dependency])
        visiting.remove(step_id)
        visited.add(step_id)

    for step in plan.steps:
        visit(step.step_id, [step.step_id])
    for step_id in sorted(cycle_steps, key=lambda value: first_by_id.get(value, 0)):
        errors.append(PlanValidationError(
            code="CIRCULAR_DEPENDENCY", message="The plan contains a circular dependency.", step_id=step_id
        ))

    if plan.intent == "unsupported_analysis":
        if plan.reasoning_type != "unsupported":
            errors.append(PlanValidationError(code="INVALID_UNSUPPORTED_OUTCOME", message="Unsupported analysis must use reasoning_type 'unsupported'."))
        if plan.steps:
            errors.append(PlanValidationError(code="INVALID_UNSUPPORTED_OUTCOME", message="Unsupported analysis must not contain executable steps."))
        if not plan.unsupported_reason:
            errors.append(PlanValidationError(code="MISSING_UNSUPPORTED_REASON", message="Unsupported analysis requires unsupported_reason."))
    elif plan.reasoning_type == "unsupported":
        errors.append(PlanValidationError(code="INVALID_UNSUPPORTED_OUTCOME", message="reasoning_type 'unsupported' requires intent 'unsupported_analysis'."))

    if plan.intent == "metric_definition" and plan.reasoning_type != "definition":
        errors.append(PlanValidationError(code="INVALID_REASONING_TYPE", message="Metric-definition intent requires reasoning_type 'definition'."))
    if plan.reasoning_type == "definition" and plan.intent != "metric_definition":
        errors.append(PlanValidationError(code="INVALID_REASONING_TYPE", message="reasoning_type 'definition' requires intent 'metric_definition'."))
    return errors
