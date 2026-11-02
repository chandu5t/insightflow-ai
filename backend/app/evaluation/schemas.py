"""Pydantic schemas and type contracts for the V2.7 Experimental Evaluation framework."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import math
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BenchmarkCategory(str, Enum):
    """The 12 official V2.1 InsightFlow-Bench research categories."""

    A_AGGREGATION = "aggregation"
    B_FILTERING = "filtering"
    C_GROUPING = "grouping"
    D_RANKING = "ranking"
    E_COMPARISON = "comparison"
    F_TIME_ANALYSIS = "time_analysis"
    G_MISSING_VALUE_REASONING = "missing_value_reasoning"
    H_METRIC_DEFINITION = "metric_definition"
    I_MULTI_STEP_REASONING = "multi_step_reasoning"
    J_AMBIGUOUS_QUESTIONS = "ambiguous_questions"
    K_UNSUPPORTED_QUESTIONS = "unsupported_questions"
    L_ADVERSARIAL = "adversarial_error_oriented"


class DifficultyLevel(str, Enum):
    """The 3 official V2.1 question difficulty levels."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class GroundTruth(BaseModel):
    """Immutable, independent ground truth representation."""

    model_config = ConfigDict(extra="forbid")

    value: Any = None
    expected_operations: list[str] = Field(default_factory=list)
    expected_formula: str | None = None
    expected_intermediate_results: list[dict[str, Any]] = Field(default_factory=list)
    final_result: Any = None
    canonical_definition: dict[str, Any] | None = None
    tolerance: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkCase(BaseModel):
    """Canonical InsightFlow-Bench record specification matching V2.1 §14."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    question: str = Field(min_length=1)
    category: BenchmarkCategory
    difficulty: DifficultyLevel
    expected_operation: str | None = None
    expected_operations: list[str] = Field(default_factory=list)
    expected_formula: str | None = None
    ground_truth: Any
    expected_intermediate_results: list[dict[str, Any]] = Field(default_factory=list)
    final_result: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("question")
    @classmethod
    def clean_question(cls, value: str) -> str:
        trimmed = " ".join(value.split())
        if not trimmed:
            raise ValueError("question must not be blank")
        return trimmed

    @property
    def operations(self) -> list[str]:
        """Return the unified list of expected operations."""
        if self.expected_operations:
            return self.expected_operations
        if self.expected_operation:
            return [self.expected_operation]
        return []


class EvaluatorInput(BaseModel):
    """Sanitized case input passed to the system under evaluation without ground truth."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    dataset: str
    question: str
    category: BenchmarkCategory
    difficulty: DifficultyLevel


class SystemCondition(str, Enum):
    """The official V2.1 evaluation conditions (Baselines, System D, and Ablations)."""

    BASELINE_A_LLM_ONLY = "baseline_a_llm_only"
    BASELINE_B_TOOL_AUGMENTED = "baseline_b_tool_augmented"
    BASELINE_C_V1 = "baseline_c_v1"
    SYSTEM_D_FULL_V2 = "system_d_full_v2"
    ABLATION_A1_NO_PLANNER = "ablation_a1_no_planner"
    ABLATION_A2_NO_VERIFICATION = "ablation_a2_no_verification"
    ABLATION_A3_NO_CORRECTION = "ablation_a3_no_correction"
    ABLATION_A4_NO_RAG = "ablation_a4_no_rag"


class ExperimentConfig(BaseModel):
    """Configuration contract matching V2.1 §20, §27 and Contract v1.1 §14."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str = Field(default_factory=lambda: f"exp_{uuid4().hex[:12]}")
    benchmark_id: str = "insightflow_bench"
    benchmark_version: str = "1.0"
    dataset_version: str = "1.0"
    system_version: str = "2.7.0"
    configuration_version: str = "1.1"
    condition: SystemCondition = SystemCondition.SYSTEM_D_FULL_V2
    baseline_id: str | None = None
    ablation_id: str | None = None
    model_name: str = "gemini-3.5-flash-lite"
    model_version: str = "2026-stable"
    model_configuration: dict[str, Any] = Field(default_factory=dict)
    prompt_configuration_version: str = "v2.7"
    random_seed: int = 42
    case_selection: list[str] | None = None
    metric_version: str = "v2.1"
    evaluation_configuration: dict[str, Any] = Field(default_factory=dict)
    statistical_procedure: str | None = None
    output_dir: str = "data/evaluation/runs"

    @field_validator("evaluation_configuration")
    @classmethod
    def validate_evaluation_configuration(cls, value: dict[str, Any]) -> dict[str, Any]:
        tolerance = value.get("tolerance")
        if tolerance is not None and (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not math.isfinite(float(tolerance))
            or tolerance < 0
        ):
            raise ValueError("evaluation_configuration.tolerance must be a finite non-negative number.")
        mapping = value.get("dataset_mapping")
        if mapping is not None and (not isinstance(mapping, dict) or any(not isinstance(k, str) for k in mapping)):
            raise ValueError("evaluation_configuration.dataset_mapping must map dataset names to dataset IDs.")
        return value


class ReproducibilityMetadata(BaseModel):
    """Reproducibility contract matching V2.1 §27 and Contract v1.1 §17."""

    model_config = ConfigDict(extra="forbid")

    benchmark_version: str
    dataset_version: str
    model_name: str
    model_version: str
    model_configuration: dict[str, Any]
    prompt_configuration_version: str
    software_version: str
    git_commit: str = "uncommitted_workspace"
    experiment_id: str
    run_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    random_seed: int
    evaluation_configuration: dict[str, Any]
    runtime_environment: dict[str, Any] = Field(default_factory=dict)


class CaseStageResult(BaseModel):
    """Stage-level observation for individual pipeline steps."""

    model_config = ConfigDict(extra="allow")

    stage_name: str
    status: Literal["completed", "failed", "skipped", "unsupported"]
    latency_ms: float = 0.0
    output_summary: Any = None
    errors: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MetricRecord(BaseModel):
    """Auditable metric observation or explicitly labeled diagnostic."""

    model_config = ConfigDict(extra="forbid")

    metric_id: str
    metric_version: str
    stage: str
    definition_reference: str
    scope: Literal["case", "aggregate"]
    value: Any = None
    status: Literal["available", "partial", "unavailable", "not_applicable"]
    unavailable_reason: str | None = None
    denominator: int | float | None = None
    error_count: int | None = None
    error_rate: float | None = None
    diagnostic: bool = False


class ErrorRecord(BaseModel):
    """Error taxonomy record matching V2.1 §24 and Contract v1.1 §10."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    stage: str
    error_category: str
    error_subcategory: str | None = None
    expected_behavior: str
    observed_behavior: str
    recoverable: bool = False
    corrected: bool = False
    evidence: str = ""


class CaseEvaluationResult(BaseModel):
    """Per-case evaluation record matching V2.1 §26 and Contract v1.1 §16."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    condition: str
    run_id: str
    success: bool
    numerical_correct: bool | None = None
    plan_correct: bool | None = None
    verification_status: str | None = None
    correction_attempts: int = 0
    latency_ms: float = 0.0
    llm_calls: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    metric_records: list[MetricRecord] = Field(default_factory=list)
    errors: list[ErrorRecord] = Field(default_factory=list)
    stage_results: list[CaseStageResult] = Field(default_factory=list)
    raw_answer: Any = None


class AggregateMetrics(BaseModel):
    """Aggregate metrics preserving denominators and auditable rates."""

    model_config = ConfigDict(extra="forbid")

    total_cases: int
    completed_cases: int
    failed_cases: int

    # M1: Numerical Accuracy
    numerical_evaluated_count: int = 0
    numerical_correct_count: int = 0
    numerical_excluded_count: int = 0
    numerical_accuracy: float | None = None

    # M2: Task Success Rate
    task_success_rate: float | None

    # M3: Planning Accuracy
    planning_evaluated_count: int = 0
    planning_correct_count: int = 0
    planning_accuracy: float | None = None

    # M4: Tool Selection Accuracy
    tool_selection_evaluated_count: int = 0
    tool_selection_correct_count: int = 0
    tool_selection_accuracy: float | None = None

    # M5: Verification Detection Rate
    injected_errors_count: int = 0
    detected_injected_errors_count: int = 0
    verification_detection_rate: float | None = None

    # M6: Correction Success Rate
    detected_errors_count: int = 0
    corrected_errors_count: int = 0
    correction_success_rate: float | None = None

    # M7: Hallucination / Unsupported-Claim Rate
    claims_evaluated_count: int = 0
    unsupported_claims_count: int = 0
    unsupported_claim_rate: float | None = None

    # M8: Explanation Groundedness
    explanations_evaluated_count: int = 0
    grounded_explanations_count: int = 0
    explanation_groundedness: float | None = None

    # M9: Latencies (ms)
    mean_total_latency_ms: float | None = None
    mean_planning_latency_ms: float | None = None
    mean_analysis_latency_ms: float | None = None
    mean_verification_latency_ms: float | None = None
    mean_correction_latency_ms: float | None = None

    # M10: LLM/API Usage
    total_llm_calls: int | None = None
    mean_llm_calls_per_case: float | None = None
    total_input_tokens: int | None = None
    total_output_tokens: int | None = None
    estimated_api_cost_usd: float | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)
    metric_records: list[MetricRecord] = Field(default_factory=list)


class ExperimentManifest(BaseModel):
    """Manifest describing the executed evaluation run."""

    model_config = ConfigDict(extra="forbid")

    manifest_version: str = "2.7.0"
    config: ExperimentConfig
    reproducibility: ReproducibilityMetadata
    total_cases_evaluated: int
    start_time: str
    end_time: str
    duration_seconds: float
    output_files: dict[str, str] = Field(default_factory=dict)
