"""Load the versioned InsightFlow-Bench snapshot and its separate ground truth."""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.evaluation.benchmark import BenchmarkValidationError, load_benchmark
from app.evaluation.schemas import BenchmarkCategory, DifficultyLevel, GroundTruth
from app.planner.operations import OPERATION_REGISTRY
from app.schemas.dataset_schema import DatasetMetadata
from app.utils.dataframe_utils import load_dataframe


BENCHMARK_VERSION = "1.0"
EXPECTED_CASE_COUNT = 100
BENCHMARK_DIRECTORY = Path(__file__).resolve().parents[2] / "data" / "evaluation" / "insightflow_bench_v1.0"
STEP7_FREEZE_PATH = BENCHMARK_DIRECTORY.parent / "step7_v1.0" / "step7_freeze_v1.0.json"
OFFICIAL_DATASETS = ("module3_sales.csv", "module3_sales_direct.csv")


class BenchmarkValidationErrorV28(ValueError):
    """Raised when the official separated benchmark artifacts are inconsistent."""


class OfficialBenchmarkDatasetRepository:
    """Read-only DatasetRepository view over the hash-frozen benchmark snapshots."""

    def __init__(self) -> None:
        verify_official_benchmark_hashes()
        self._paths = {name: BENCHMARK_DIRECTORY / "datasets" / name for name in OFFICIAL_DATASETS}
        self._ids = {
            name: uuid5(NAMESPACE_URL, f"InsightFlow-Bench-v{BENCHMARK_VERSION}/{name}")
            for name in OFFICIAL_DATASETS
        }
        self._names_by_id = {value: name for name, value in self._ids.items()}

    @property
    def dataset_mapping(self) -> dict[str, str]:
        """Stable benchmark dataset references for ExperimentConfig and run metadata."""
        return {name: str(dataset_id) for name, dataset_id in self._ids.items()}

    def get_csv_path(self, dataset_id: UUID) -> Path:
        name = self._names_by_id.get(UUID(str(dataset_id)))
        if name is None:
            raise ValueError("Dataset ID is not part of frozen InsightFlow-Bench v1.0.")
        return self._paths[name]

    def get(self, dataset_id: UUID) -> DatasetMetadata:
        path = self.get_csv_path(dataset_id)
        frame = load_dataframe(path)
        return DatasetMetadata(
            dataset_id=UUID(str(dataset_id)),
            filename=path.name,
            source_format="csv",
            row_count=len(frame),
            column_count=len(frame.columns),
            column_names=[str(column) for column in frame.columns],
            original_size_bytes=path.stat().st_size,
            uploaded_at=datetime(2000, 1, 1, tzinfo=timezone.utc),
            warnings=[],
        )


def verify_official_benchmark_hashes() -> dict[str, Any]:
    """Verify benchmark, ground-truth, and dataset files against the Step 7 freeze."""
    try:
        freeze = json.loads(STEP7_FREEZE_PATH.read_text(encoding="utf-8"))
        expected_hashes = freeze["benchmark"]["sha256"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise BenchmarkValidationErrorV28("The Step 7 freeze artifact is unavailable or malformed.") from exc
    for relative_path, expected in expected_hashes.items():
        path = BENCHMARK_DIRECTORY / relative_path
        try:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise BenchmarkValidationErrorV28(
                f"Frozen benchmark artifact '{relative_path}' is unavailable."
            ) from exc
        if actual != expected:
            raise BenchmarkValidationErrorV28(
                f"Frozen benchmark artifact '{relative_path}' does not match the Step 7 hash."
            )
    return freeze


class QuestionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    question: str = Field(min_length=1)
    category: BenchmarkCategory
    difficulty: DifficultyLevel


class GroundTruthRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    ground_truth: GroundTruth


def _read_array(source: str | Path) -> list[Any]:
    path = Path(source)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkValidationErrorV28(f"Could not read benchmark artifact '{path.name}'.") from exc
    if not isinstance(value, list):
        raise BenchmarkValidationErrorV28(f"Benchmark artifact '{path.name}' must contain a JSON array.")
    return value


def validate_official_benchmark_records(
    question_rows: list[dict[str, Any]],
    ground_truth_rows: list[dict[str, Any]],
    *,
    expected_case_count: int = EXPECTED_CASE_COUNT,
) -> list[Any]:
    """Validate separate files and adapt them to the unchanged V2.7 BenchmarkCase interface."""
    try:
        questions = [QuestionRecord.model_validate(row) for row in question_rows]
        truths = [GroundTruthRecord.model_validate(row) for row in ground_truth_rows]
    except ValidationError as exc:
        raise BenchmarkValidationErrorV28(f"Malformed question or ground-truth record: {exc}") from exc

    q_ids = [record.id for record in questions]
    gt_ids = [record.id for record in truths]
    if len(set(q_ids)) != len(q_ids):
        raise BenchmarkValidationErrorV28("Question IDs must be unique.")
    if len(set(gt_ids)) != len(gt_ids):
        raise BenchmarkValidationErrorV28("Ground-truth IDs must be unique.")
    if len(questions) != expected_case_count:
        raise BenchmarkValidationErrorV28(
            f"Expected exactly {expected_case_count} questions; received {len(questions)}."
        )
    if set(q_ids) != set(gt_ids):
        missing = sorted(set(q_ids) - set(gt_ids))
        extra = sorted(set(gt_ids) - set(q_ids))
        raise BenchmarkValidationErrorV28(f"Question/ground-truth IDs do not match (missing={missing}, extra={extra}).")

    unknown_datasets = sorted({record.dataset for record in questions} - set(OFFICIAL_DATASETS))
    if unknown_datasets:
        raise BenchmarkValidationErrorV28(f"Benchmark questions reference unknown frozen datasets: {unknown_datasets}.")

    categories = {record.category for record in questions}
    difficulties = {record.difficulty for record in questions}
    if categories != set(BenchmarkCategory):
        absent = sorted(category.value for category in set(BenchmarkCategory) - categories)
        raise BenchmarkValidationErrorV28(f"All 12 categories must be covered; missing {absent}.")
    if difficulties != set(DifficultyLevel):
        absent = sorted(difficulty.value for difficulty in set(DifficultyLevel) - difficulties)
        raise BenchmarkValidationErrorV28(f"Easy/Medium/Hard coverage is required; missing {absent}.")

    truth_by_id = {record.id: record.ground_truth for record in truths}
    joined: list[dict[str, Any]] = []
    for question in questions:
        truth = truth_by_id[question.id]
        if truth.value is None:
            raise BenchmarkValidationErrorV28(f"Ground truth value is required for '{question.id}'.")
        unknown_operations = sorted(set(truth.expected_operations) - set(OPERATION_REGISTRY))
        if unknown_operations:
            raise BenchmarkValidationErrorV28(
                f"Ground truth for '{question.id}' names unregistered operations: {unknown_operations}."
            )
        if question.category == BenchmarkCategory.I_MULTI_STEP_REASONING and len(truth.expected_operations) < 2:
            raise BenchmarkValidationErrorV28(
                f"Multi-step case '{question.id}' must preserve at least two expected operations."
            )
        joined.append({
            "id": question.id,
            "dataset": question.dataset,
            "question": question.question,
            "category": question.category.value,
            "difficulty": question.difficulty.value,
            "expected_operation": truth.expected_operations[0] if len(truth.expected_operations) == 1 else None,
            "expected_operations": list(truth.expected_operations),
            "expected_formula": truth.expected_formula,
            "ground_truth": truth.model_dump(mode="json"),
            "expected_intermediate_results": list(truth.expected_intermediate_results),
            "final_result": truth.final_result,
            "metadata": dict(truth.metadata),
        })

    try:
        # This is the production V2.7 validation/loading interface. The join is
        # in-memory; neither artifact is rewritten or exposed to system input.
        return load_benchmark(joined)
    except BenchmarkValidationError as exc:
        raise BenchmarkValidationErrorV28(f"V2.7 benchmark validation failed: {exc}") from exc


def load_official_benchmark(
    questions_path: str | Path = BENCHMARK_DIRECTORY / "questions.json",
    ground_truth_path: str | Path = BENCHMARK_DIRECTORY / "ground_truth.json",
) -> list[Any]:
    """Load the frozen question and independent ground-truth snapshots."""
    if Path(questions_path).resolve() == (BENCHMARK_DIRECTORY / "questions.json").resolve() and Path(ground_truth_path).resolve() == (BENCHMARK_DIRECTORY / "ground_truth.json").resolve():
        verify_official_benchmark_hashes()
    return validate_official_benchmark_records(
        _read_array(questions_path), _read_array(ground_truth_path)
    )
