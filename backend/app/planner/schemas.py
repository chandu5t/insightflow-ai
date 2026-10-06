"""Pydantic contracts for the V2.2 Analytical Planner."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

AnalysisIntent = Literal[
    "total_revenue",
    "average_order_value",
    "regional_revenue_ranking",
    "filtered_revenue",
    "time_based_revenue",
    "count_orders",
    "distinct_count",
    "compare_groups",
    "metric_definition",
    "unsupported_analysis",
]
ReasoningType = Literal["simple", "multi_step", "definition", "unsupported"]


class MetricDefinitionContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: str
    metric_key: str
    definition_version: str
    dataset_sha256: str
    manifest_version: str
    exact_column_references: list[str]
    allowed_expression: dict[str, Any]
    output_semantic_role: Literal[
        "identifier", "categorical", "additive_measure", "non_additive_measure",
        "date", "boolean", "unknown",
    ]
    source_reference: str


class DatasetContext(BaseModel):
    """Caller-supplied schema context; the planner never loads a dataset."""

    model_config = ConfigDict(extra="forbid")

    columns: list[str] = Field(default_factory=list)
    semantic_roles: dict[str, Literal[
        "identifier", "categorical", "additive_measure", "non_additive_measure",
        "date", "boolean", "unknown",
    ]] = Field(default_factory=dict)
    dataset_sha256: str | None = None
    manifest_version: str | None = None

    @field_validator("columns")
    @classmethod
    def exact_unique_columns(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)) or any(not item for item in value):
            raise ValueError("dataset columns must be non-empty and unique exact physical headers")
        return value

    @field_validator("semantic_roles")
    @classmethod
    def role_keys_are_physical_columns(cls, value, info):
        columns = info.data.get("columns", [])
        if any(column not in columns for column in value):
            raise ValueError("semantic role keys must exactly match supplied physical columns")
        return value


class PlannerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    dataset_context: DatasetContext | None = None
    metric_definitions: list[MetricDefinitionContext] | None = None

    @field_validator("question")
    @classmethod
    def trim_question(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("question must not be blank")
        return cleaned


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    description: str = Field(min_length=1)
    inputs: list[str] = Field(default_factory=list)
    parameters: dict[str, object] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)


class AnalysisPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: AnalysisIntent
    reasoning_type: ReasoningType
    steps: list[PlanStep]
    unsupported_reason: str | None = None


class PlanValidationError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    step_id: str | None = None


class PlannerMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    planner_version: str = "v2.2"
    validation_version: str = "v2.2"
    request_id: str
    planner_model: str
    retry_used: bool
    latency_ms: float = Field(ge=0)


class PlannerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan: AnalysisPlan
    valid: bool
    validation_errors: list[PlanValidationError] = Field(default_factory=list)
    metadata: PlannerMetadata
