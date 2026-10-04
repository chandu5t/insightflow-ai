"""InsightFlow-Bench loading, strict validation, and ground-truth isolation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from pydantic import ValidationError

from app.evaluation.schemas import (
    BenchmarkCase,
    BenchmarkCategory,
    DifficultyLevel,
    EvaluatorInput,
    GroundTruth,
)


class BenchmarkValidationError(ValueError):
    """Raised when a benchmark suite or case is malformed."""


def validate_benchmark_case(raw: dict[str, Any] | BenchmarkCase) -> BenchmarkCase:
    """Validate a single benchmark case under strict V2.1 rules."""
    if isinstance(raw, BenchmarkCase):
        case = raw
    else:
        try:
            case = BenchmarkCase.model_validate(raw)
        except ValidationError as exc:
            raise BenchmarkValidationError(f"Invalid benchmark case structure: {exc}") from exc

    if not case.id.strip():
        raise BenchmarkValidationError("Case id cannot be empty.")
    if not case.dataset.strip():
        raise BenchmarkValidationError(f"Case '{case.id}' must specify a valid dataset.")
    if not case.question.strip():
        raise BenchmarkValidationError(f"Case '{case.id}' question cannot be blank.")

    # Check category validity
    if not isinstance(case.category, BenchmarkCategory):
        raise BenchmarkValidationError(f"Case '{case.id}' has invalid category: {case.category}")

    # Check difficulty validity
    if not isinstance(case.difficulty, DifficultyLevel):
        raise BenchmarkValidationError(f"Case '{case.id}' has invalid difficulty: {case.difficulty}")

    # Ground truth must be present
    if case.ground_truth is None and case.category != BenchmarkCategory.K_UNSUPPORTED_QUESTIONS:
        raise BenchmarkValidationError(f"Case '{case.id}' is missing ground_truth.")

    # Multi-step questions should indicate expected operations or multi-step structures
    if case.category == BenchmarkCategory.I_MULTI_STEP_REASONING:
        if len(case.operations) < 2 and not case.expected_intermediate_results:
            raise BenchmarkValidationError(
                f"Multi-step case '{case.id}' must specify at least two expected operations or intermediate results."
            )

    return case


def validate_benchmark_suite(cases: Sequence[dict[str, Any] | BenchmarkCase]) -> list[BenchmarkCase]:
    """Validate a complete benchmark suite, ensuring unique IDs and valid records."""
    if not cases:
        raise BenchmarkValidationError("Benchmark suite cannot be empty.")

    validated: list[BenchmarkCase] = []
    seen_ids: set[str] = set()

    for item in cases:
        case = validate_benchmark_case(item)
        if case.id in seen_ids:
            raise BenchmarkValidationError(f"Duplicate benchmark case id detected: '{case.id}'.")
        seen_ids.add(case.id)
        validated.append(case)

    return validated


def load_benchmark(source: str | Path | list[dict[str, Any]]) -> list[BenchmarkCase]:
    """Load benchmark cases from a JSON file, directory, or raw list."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {path}")
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
            if not isinstance(data, list):
                raise BenchmarkValidationError(f"Benchmark file '{path}' must contain a JSON array of cases.")
            return validate_benchmark_suite(data)
    elif isinstance(source, list):
        return validate_benchmark_suite(source)
    else:
        raise TypeError(f"Unsupported benchmark source type: {type(source)}")


def sanitize_case_for_execution(case: BenchmarkCase) -> EvaluatorInput:
    """Enforce data-leakage boundaries by stripping ground truth and expected answers."""
    return EvaluatorInput(
        case_id=case.id,
        dataset=case.dataset,
        question=case.question,
        category=case.category,
        difficulty=case.difficulty,
    )


def extract_ground_truth(case: BenchmarkCase) -> GroundTruth:
    """Extract authoritative ground truth representation for comparison."""
    if isinstance(case.ground_truth, dict) and "value" in case.ground_truth:
        gt = GroundTruth.model_validate(case.ground_truth)
    else:
        gt = GroundTruth(
            value=case.ground_truth,
            expected_operations=case.operations,
            expected_formula=case.expected_formula,
            expected_intermediate_results=case.expected_intermediate_results,
            final_result=case.final_result if case.final_result is not None else case.ground_truth,
        )
    return gt
