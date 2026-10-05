from __future__ import annotations

import copy
import csv
from collections import defaultdict, Counter
from pathlib import Path
from uuid import UUID

import pytest

from app.evaluation.benchmark import extract_ground_truth, sanitize_case_for_execution
from app.evaluation.official_benchmark import (
    BENCHMARK_DIRECTORY,
    BenchmarkValidationErrorV28,
    OfficialBenchmarkDatasetRepository,
    load_official_benchmark,
    validate_official_benchmark_records,
)
from app.planner.operations import OPERATION_REGISTRY


def _records():
    import json

    questions = json.loads((BENCHMARK_DIRECTORY / "questions.json").read_text(encoding="utf-8"))
    truths = json.loads((BENCHMARK_DIRECTORY / "ground_truth.json").read_text(encoding="utf-8"))
    return questions, truths


def test_official_snapshot_loads_through_v27_interface_and_is_complete():
    cases = load_official_benchmark()
    assert len(cases) == 100
    assert len({case.id for case in cases}) == 100
    assert set(Counter(case.category.value for case in cases)) == {
        "aggregation", "filtering", "grouping", "ranking", "comparison", "time_analysis",
        "missing_value_reasoning", "metric_definition", "multi_step_reasoning",
        "ambiguous_questions", "unsupported_questions", "adversarial_error_oriented",
    }
    assert set(Counter(case.difficulty.value for case in cases)) == {"easy", "medium", "hard"}
    assert all(case.ground_truth is not None for case in cases)
    assert all(set(case.operations) <= set(OPERATION_REGISTRY) for case in cases)


def test_official_dataset_repository_uses_only_hash_frozen_snapshot_files():
    repository = OfficialBenchmarkDatasetRepository()
    mapping = repository.dataset_mapping
    assert set(mapping) == {"module3_sales.csv", "module3_sales_direct.csv"}
    expected_rows = {"module3_sales.csv": 13, "module3_sales_direct.csv": 6}
    for filename, dataset_id in mapping.items():
        path = BENCHMARK_DIRECTORY / "datasets" / filename
        assert repository.get_csv_path(UUID(dataset_id)) == path
        metadata = repository.get(UUID(dataset_id))
        assert metadata.filename == filename
        assert metadata.row_count == expected_rows[filename]


def test_ground_truth_is_matched_but_never_passed_to_system_input():
    cases = load_official_benchmark()
    case = cases[0]
    expected = extract_ground_truth(case)
    system_input = sanitize_case_for_execution(case).model_dump()
    assert expected.value == 256000
    assert case.id == system_input["case_id"]
    assert "ground_truth" not in system_input
    assert "expected_operations" not in system_input
    assert "expected_formula" not in system_input


def test_separate_artifacts_have_exactly_matching_unique_ids_and_required_provenance():
    questions, truths = _records()
    assert len(questions) == len(truths) == 100
    assert len({row["id"] for row in questions}) == len(questions)
    assert len({row["id"] for row in truths}) == len(truths)
    assert {row["id"] for row in questions} == {row["id"] for row in truths}
    for row in truths:
        gt = row["ground_truth"]
        assert gt["value"] is not None
        assert isinstance(gt["expected_operations"], list)
        assert isinstance(gt["expected_intermediate_results"], list)
        assert gt["expected_formula"] is not None
        assert "reference_method" in gt["metadata"]


@pytest.mark.parametrize("invalidator", ["count", "duplicate_question", "missing_truth", "duplicate_truth", "id_mismatch", "category", "difficulty", "operation"])
def test_invalid_official_benchmark_is_rejected(invalidator: str):
    questions, truths = _records()
    questions = copy.deepcopy(questions)
    truths = copy.deepcopy(truths)
    if invalidator == "count":
        questions.pop()
    elif invalidator == "duplicate_question":
        questions[-1]["id"] = questions[0]["id"]
    elif invalidator == "missing_truth":
        truths.pop()
    elif invalidator == "duplicate_truth":
        truths[-1]["id"] = truths[0]["id"]
    elif invalidator == "id_mismatch":
        truths[-1]["id"] = "IFB-999"
    elif invalidator == "category":
        questions[0]["category"] = "not_a_category"
    elif invalidator == "difficulty":
        questions[0]["difficulty"] = "extreme"
    elif invalidator == "operation":
        truths[0]["ground_truth"]["expected_operations"] = ["run_generated_code"]
    with pytest.raises(BenchmarkValidationErrorV28):
        validate_official_benchmark_records(questions, truths)


def test_ground_truth_numeric_values_match_independent_csv_reference_calculations():
    questions, truths = _records()
    questions_by_id = {row["id"]: row for row in questions}
    truth_by_id = {row["id"]: row["ground_truth"] for row in truths}
    data_dir = BENCHMARK_DIRECTORY / "datasets"

    def load_rows(filename: str):
        with (data_dir / filename).open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    derived = load_rows("module3_sales.csv")
    direct = load_rows("module3_sales_direct.csv")
    revenue = lambda row: int(row["quantity"]) * int(row["unit_price"])
    region = lambda row: row["region"] or "(missing)"
    by_region: dict[str, int] = defaultdict(int)
    by_product: dict[str, int] = defaultdict(int)
    by_category: dict[str, int] = defaultdict(int)
    by_day: dict[str, int] = defaultdict(int)
    for row in derived:
        by_region[region(row)] += revenue(row)
        by_product[row["product"]] += revenue(row)
        by_category[row["category"]] += revenue(row)
        by_day[row["order_date"]] += revenue(row)
    direct_by_region: dict[str, int] = defaultdict(int)
    direct_by_product: dict[str, int] = defaultdict(int)
    for row in direct:
        direct_by_region[row["Region"]] += int(row["Total Revenue"])
        direct_by_product[row["Product Name"]] += int(row["Total Revenue"])

    total_revenue = sum(revenue(row) for row in derived)
    row_count = len(derived)
    distinct_orders = len({row["order_id"] for row in derived})
    references = {
        "derived_revenue_total": total_revenue,
        "derived_rows": row_count,
        "derived_distinct_orders": distinct_orders,
        "north_revenue": by_region["North"],
        "south_revenue": by_region["South"],
        "east_revenue": by_region["East"],
        "missing_region_revenue": by_region["(missing)"],
        "derived_revenue_by_region": dict(sorted(by_region.items())),
        "derived_revenue_by_product": dict(sorted(by_product.items())),
        "derived_revenue_by_category": dict(sorted(by_category.items())),
        "daily_revenue": dict(sorted(by_day.items())),
        "direct_revenue_total": sum(int(row["Total Revenue"]) for row in direct),
        "direct_north_revenue": direct_by_region["North"],
        "direct_south_revenue": direct_by_region["South"],
        "direct_laptop_revenue": direct_by_product["Laptop"],
        "derived_north_minus_south": by_region["North"] - by_region["South"],
        "north_pct_over_south": (by_region["North"] - by_region["South"]) * 100 // by_region["South"],
        "missing_region_rows": sum(not row["region"] for row in derived),
        "missing_cells_derived": sum(value == "" for row in derived for value in row.values()),
        "missing_cells_direct": sum(value == "" for row in direct for value in row.values()),
        "duplicate_adjusted_revenue": total_revenue - revenue(derived[-1]),
        "first_week_revenue": sum(value for day, value in by_day.items() if day <= "2025-01-07"),
    }
    for case_id, gt in truth_by_id.items():
        key = gt["metadata"]["reference_key"]
        if key in references:
            assert gt["value"] == references[key], f"ground truth mismatch for {case_id} ({key})"

