"""Unit tests for benchmark loading, strict validation, and ground-truth isolation."""

import pytest
from pydantic import ValidationError

from app.evaluation.benchmark import (
    BenchmarkValidationError,
    extract_ground_truth,
    load_benchmark,
    sanitize_case_for_execution,
    validate_benchmark_case,
    validate_benchmark_suite,
)
from app.evaluation.schemas import BenchmarkCase, BenchmarkCategory, DifficultyLevel


def test_valid_benchmark_case():
    case = validate_benchmark_case({
        "id": "Q001",
        "dataset": "sales.csv",
        "question": "What is the revenue?",
        "category": "aggregation",
        "difficulty": "easy",
        "expected_operation": "derive_metric",
        "ground_truth": 1000.0,
    })
    assert case.id == "Q001"
    assert case.category == BenchmarkCategory.A_AGGREGATION
    assert case.difficulty == DifficultyLevel.EASY
    assert case.operations == ["derive_metric"]


def test_all_12_categories_and_3_difficulties():
    categories = [
        "aggregation", "filtering", "grouping", "ranking", "comparison",
        "time_analysis", "missing_value_reasoning", "metric_definition",
        "multi_step_reasoning", "ambiguous_questions", "unsupported_questions",
        "adversarial_error_oriented",
    ]
    difficulties = ["easy", "medium", "hard"]

    for i, (cat, diff) in enumerate(zip(categories, difficulties * 4)):
        raw = {
            "id": f"C_{i}",
            "dataset": "sales.csv",
            "question": f"Question {i}",
            "category": cat,
            "difficulty": diff,
            "expected_operation": "derive_metric" if cat != "multi_step_reasoning" else None,
            "expected_operations": ["derive_metric", "rank"] if cat == "multi_step_reasoning" else [],
            "ground_truth": 100.0 if cat != "unsupported_questions" else "unsupported",
        }
        case = validate_benchmark_case(raw)
        assert case.category.value == cat
        assert case.difficulty.value == diff


def test_reject_blank_or_invalid_fields():
    with pytest.raises(BenchmarkValidationError, match="must not be blank"):
        validate_benchmark_case({
            "id": "Q_BLANK",
            "dataset": "sales.csv",
            "question": "   ",
            "category": "aggregation",
            "difficulty": "easy",
            "ground_truth": 10.0,
        })

    with pytest.raises(BenchmarkValidationError, match="Invalid benchmark case structure"):
        validate_benchmark_case({
            "id": "Q_BAD_CAT",
            "dataset": "sales.csv",
            "question": "Test",
            "category": "not_a_real_category",
            "difficulty": "easy",
            "ground_truth": 10.0,
        })


def test_reject_duplicate_ids():
    raw_suite = [
        {
            "id": "DUP_1",
            "dataset": "sales.csv",
            "question": "Q1",
            "category": "aggregation",
            "difficulty": "easy",
            "ground_truth": 10.0,
        },
        {
            "id": "DUP_1",
            "dataset": "sales.csv",
            "question": "Q2",
            "category": "aggregation",
            "difficulty": "easy",
            "ground_truth": 20.0,
        },
    ]
    with pytest.raises(BenchmarkValidationError, match="Duplicate benchmark case id detected"):
        validate_benchmark_suite(raw_suite)


def test_data_leakage_isolation():
    case = validate_benchmark_case({
        "id": "Q_SECRET",
        "dataset": "sales.csv",
        "question": "What is total revenue?",
        "category": "aggregation",
        "difficulty": "easy",
        "expected_operation": "derive_metric",
        "expected_operations": ["derive_metric"],
        "expected_formula": "qty * price",
        "ground_truth": 999999.0,
        "final_result": 999999.0,
    })

    sanitized = sanitize_case_for_execution(case)
    # Sanitized input must NEVER contain ground truth or expected formulas
    assert not hasattr(sanitized, "ground_truth")
    assert not hasattr(sanitized, "expected_formula")
    assert not hasattr(sanitized, "expected_operations")
    assert not hasattr(sanitized, "final_result")

    gt = extract_ground_truth(case)
    assert gt.value == 999999.0
    assert gt.expected_operations == ["derive_metric"]
    assert gt.expected_formula == "qty * price"
