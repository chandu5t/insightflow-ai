"""Deterministic operation canonicalization; never repairs plan meaning or structure."""

from typing import Any

from app.planner.operations import OPERATION_ALIASES


def normalize_plan_payload(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    normalized = dict(payload)
    steps = normalized.get("steps")
    if isinstance(steps, list):
        normalized_steps = []
        for step in steps:
            if isinstance(step, dict):
                step = dict(step)
                operation = step.get("operation")
                if isinstance(operation, str):
                    step["operation"] = OPERATION_ALIASES.get(operation, operation)
            normalized_steps.append(step)
        normalized["steps"] = normalized_steps
    return normalized
