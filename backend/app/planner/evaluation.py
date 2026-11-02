"""Deterministic metrics for the small V2.2 planner development fixture."""

from collections.abc import Mapping, Sequence

from app.planner.operations import OPERATION_REGISTRY
from app.planner.schemas import PlannerResponse


def evaluate_planner_fixture(
    cases: Sequence[Mapping[str, object]],
    responses: Mapping[str, PlannerResponse],
) -> dict[str, float | int]:
    """Compute component metrics from supplied outputs; this function runs no LLM."""
    evaluated = [(case, responses.get(str(case["id"]))) for case in cases]
    if not evaluated:
        return {
            "evaluated": 0, "intent_accuracy": 0.0, "plan_validity_rate": 0.0,
            "operation_selection_precision": 0.0, "step_coverage": 0.0,
            "dependency_accuracy": 0.0, "unsupported_operation_rate": 0.0,
            "mean_latency_ms": 0.0,
        }

    intents_correct = sum(response is not None and response.plan.intent == case["expected_intent"] for case, response in evaluated)
    valid_count = sum(response is not None and response.valid for _, response in evaluated)
    selected = required = covered = 0
    expected_edges = predicted_edges = correct_edges = 0
    unknown_operations = total_operations = 0
    for case, response in evaluated:
        expected = set(case.get("expected_operations", []))
        actual = [step.operation for step in response.plan.steps] if response is not None else []
        actual_set = set(actual)
        required += len(expected)
        covered += len(expected & actual_set)
        selected += len(actual_set)
        total_operations += len(actual)
        unknown_operations += sum(operation not in OPERATION_REGISTRY for operation in actual)
        expected_edge_set = {tuple(edge) for edge in case.get("expected_dependency_edges", [])}
        operation_by_id = {step.step_id: step.operation for step in response.plan.steps} if response is not None else {}
        actual_edge_set = {
            (step.operation, operation_by_id[dependency])
            for step in (response.plan.steps if response is not None else [])
            for dependency in step.depends_on
            if dependency in operation_by_id
        }
        expected_edges += len(expected_edge_set)
        predicted_edges += len(actual_edge_set)
        correct_edges += len(expected_edge_set & actual_edge_set)

    operation_accuracy = covered / selected if selected else (1.0 if required == 0 else 0.0)
    coverage = covered / required if required else 1.0
    edge_denominator = expected_edges + predicted_edges
    dependency_accuracy = (2 * correct_edges / edge_denominator) if edge_denominator else 1.0
    return {
        "evaluated": len(evaluated),
        "intent_accuracy": intents_correct / len(evaluated),
        "plan_validity_rate": valid_count / len(evaluated),
        "operation_selection_precision": operation_accuracy,
        "step_coverage": coverage,
        "dependency_accuracy": dependency_accuracy,
        "unsupported_operation_rate": unknown_operations / total_operations if total_operations else 0.0,
        "mean_latency_ms": (
            sum(response.metadata.latency_ms for _, response in evaluated if response is not None)
            / sum(response is not None for _, response in evaluated)
            if any(response is not None for _, response in evaluated) else 0.0
        ),
    }
