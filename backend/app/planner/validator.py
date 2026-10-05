"""Structural validation for V2.2 plans. This is not an execution validator."""

from collections import defaultdict
import math

from app.planner.operations import OPERATION_REGISTRY
from app.planner.schemas import AnalysisPlan, PlanValidationError

_FILTER_OPERATORS = {"eq", "ne", "lt", "lte", "gt", "gte", "in", "not_in", "is_missing", "is_not_missing"}
_BINARY_EXPR = {"add", "subtract", "multiply", "divide"}
_COMPARE_OPERATORS = {"lt", "lte", "eq", "gte", "gt"}


def _expression_errors(expr: object) -> list[str]:
    if not isinstance(expr, dict):
        return ["formula must be a structured arithmetic expression tree"]
    if set(expr) == {"literal"}:
        value = expr["literal"]
        try:
            finite = math.isfinite(float(value))
        except (OverflowError, TypeError, ValueError):
            finite = False
        return [] if isinstance(value, (int, float)) and not isinstance(value, bool) and finite else ["numeric literals must be finite"]
    if set(expr) == {"column"}:
        return [] if isinstance(expr["column"], str) and expr["column"] else ["column reference must be non-empty"]
    if set(expr) == {"step_id"}:
        return [] if isinstance(expr["step_id"], str) and expr["step_id"] else ["step reference must be non-empty"]
    if expr.get("op") == "negate" and set(expr) == {"op", "value"}:
        return _expression_errors(expr["value"])
    if isinstance(expr.get("op"), str) and expr.get("op") in _BINARY_EXPR and set(expr) == {"op", "left", "right"}:
        return _expression_errors(expr["left"]) + _expression_errors(expr["right"])
    return ["formula contains an unsupported expression node or field"]


def _expression_step_refs(expr: object) -> set[str]:
    if not isinstance(expr, dict):
        return set()
    if set(expr) == {"step_id"} and isinstance(expr["step_id"], str):
        return {expr["step_id"]}
    refs: set[str] = set()
    for key in ("left", "right", "value"):
        if key in expr:
            refs.update(_expression_step_refs(expr[key]))
    return refs


def _expression_columns(expr: object) -> set[str]:
    if not isinstance(expr, dict):
        return set()
    if set(expr) == {"column"} and isinstance(expr["column"], str):
        return {expr["column"]}
    cols: set[str] = set()
    for key in ("left", "right", "value"):
        if key in expr:
            cols.update(_expression_columns(expr[key]))
    return cols


def _operation_parameter_errors(step) -> list[tuple[str, str]]:
    p, op = step.parameters, step.operation
    errors: list[tuple[str, str]] = []
    if op == "filter_rows":
        mode = p.get("mode", "predicate")
        if mode == "exact_duplicate_rows":
            if p.get("keep") != "first" or "conditions" in p:
                errors.append(("INVALID_FILTER_MODE", "exact duplicate mode requires keep='first' and no conditions."))
        elif mode == "predicate":
            if "keep" in p:
                errors.append(("INVALID_FILTER_MODE", "keep is valid only for exact_duplicate_rows mode."))
            conditions = p.get("conditions")
            if not isinstance(conditions, list) or not conditions:
                errors.append(("INVALID_FILTER_PREDICATE", "predicate mode requires non-empty conditions."))
            else:
                for condition in conditions:
                    if not isinstance(condition, dict) or set(condition) - {"column", "operator", "value"}:
                        errors.append(("INVALID_FILTER_PREDICATE", "predicate fields are invalid.")); break
                    operator = condition.get("operator")
                    if not isinstance(condition.get("column"), str) or not condition["column"] or not isinstance(operator, str) or operator not in _FILTER_OPERATORS:
                        errors.append(("INVALID_FILTER_PREDICATE", "predicate column or operator is invalid.")); break
                    missing_op = operator in {"is_missing", "is_not_missing"}
                    if missing_op == ("value" in condition):
                        errors.append(("INVALID_FILTER_PREDICATE", "missing predicates take no value; other predicates require one.")); break
                    if operator in {"in", "not_in"} and (not isinstance(condition.get("value"), list) or not condition["value"]):
                        errors.append(("INVALID_FILTER_PREDICATE", "in/not_in require a non-empty list.")); break
                    if operator not in {"in", "not_in", "is_missing", "is_not_missing"} and isinstance(condition.get("value"), list):
                        errors.append(("INVALID_FILTER_PREDICATE", "Scalar comparison operators require one scalar value.")); break
        else:
            errors.append(("INVALID_FILTER_MODE", "filter_rows mode is unsupported."))
    elif op == "derive_metric":
        expression_errors = _expression_errors(p.get("formula"))
        if expression_errors:
            errors.append(("INVALID_METRIC_EXPRESSION", expression_errors[0]))
        for ref in _expression_step_refs(p.get("formula")):
            if ref not in step.depends_on:
                errors.append(("INVALID_METRIC_DEPENDENCY", "Each step reference in the formula must be a declared dependency."))
        formula_columns = _expression_columns(p.get("formula"))
        if formula_columns and set(step.inputs) != formula_columns:
            errors.append(("INVALID_METRIC_INPUTS", "derive_metric inputs must exactly match formula column references."))
        if not isinstance(p.get("output_name"), str) or not p["output_name"]:
            errors.append(("INVALID_METRIC_OUTPUT", "derive_metric requires a non-empty output_name."))
    elif op == "aggregate":
        if p.get("function") != "sum":
            errors.append(("UNSUPPORTED_AGGREGATE", "Only sum is supported."))
        if len(step.inputs) != 1:
            errors.append(("INVALID_AGGREGATE_INPUT", "aggregate requires exactly one input value."))
    elif op == "calculate_percentage_difference" and len(step.depends_on) != 2:
        errors.append(("INVALID_PERCENTAGE_DIFFERENCE_INPUT", "percentage difference requires exactly two prior steps."))
    elif op == "compare_groups":
        if not isinstance(p.get("comparator"), str) or p.get("comparator") not in _COMPARE_OPERATORS:
            errors.append(("INVALID_COMPARATOR", "compare_groups comparator is unsupported."))
        for side in ("left", "right"):
            operand = p.get(side)
            if not isinstance(operand, dict) or set(operand) != {"step_id", "selector", "multiplier"}:
                errors.append(("INVALID_COMPARISON_OPERAND", f"{side} requires step_id, selector, and multiplier.")); continue
            ref = operand["step_id"]
            if not isinstance(ref, str) or not ref or ref not in step.depends_on:
                errors.append(("INVALID_COMPARISON_DEPENDENCY", f"{side} reference must be a declared dependency."))
            multiplier = operand["multiplier"]
            try:
                finite_multiplier = math.isfinite(float(multiplier))
            except (OverflowError, TypeError, ValueError):
                finite_multiplier = False
            if not isinstance(multiplier, (int, float)) or isinstance(multiplier, bool) or not finite_multiplier:
                errors.append(("INVALID_COMPARISON_OPERAND", f"{side} multiplier must be finite."))
            selector = operand["selector"]
            valid_selector = selector == "scalar"
            valid_selector |= isinstance(selector, dict) and set(selector) == {"kind", "key"} and selector.get("kind") == "group_key" and isinstance(selector.get("key"), str)
            valid_selector |= isinstance(selector, dict) and set(selector) == {"kind", "index", "field"} and selector.get("kind") == "list_item" and isinstance(selector.get("index"), int) and not isinstance(selector.get("index"), bool) and selector.get("index", -1) >= 0 and isinstance(selector.get("field"), str)
            valid_selector |= isinstance(selector, dict) and selector == {"kind": "leader"}
            if not valid_selector:
                errors.append(("INVALID_COMPARISON_SELECTOR", f"{side} selector is unsupported."))
    return errors


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

            for code, message in _operation_parameter_errors(step):
                errors.append(PlanValidationError(code=code, message=message, step_id=step.step_id))

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
