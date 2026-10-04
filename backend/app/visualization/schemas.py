"""Pydantic contracts for the additive V2.6 visualization boundary."""

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

ChartType = Literal["bar", "line", "pie", "scatter"]
VisualizationStatus = Literal["generated", "unsupported", "failed"]
VisualizationErrorCategory = Literal[
    "invalid_visualization_input",
    "unsupported_visualization",
    "insufficient_data",
    "invalid_visualization_schema",
    "visualization_data_mismatch",
    "visualization_generation_failure",
]


class VisualizationRequest(BaseModel):
    """Keep artifacts raw at the API boundary so failures use V2.6's response schema."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    dataset_id: UUID
    analysis_plan: Any
    execution_result: Any
    verification_result: Any
    correction_result: Any | None = None


class VisualizationError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: VisualizationErrorCategory
    code: str
    message: str


class VisualizationTraceability(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    dataset_id: UUID
    source_step_id: str
    source_operation: str | None = None
    verification_status: Literal["passed", "failed", "unsupported"]
    correction_status: str | None = None


class VisualizationAxis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    label: str
    data_type: Literal["categorical", "numeric", "ordered", "temporal"]


class VisualizationSeries(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    field: str


class VisualizationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    visualization_id: str
    chart_type: ChartType
    title: str
    description: str
    source_step_id: str
    x_axis: VisualizationAxis
    y_axis: VisualizationAxis | None = None
    series: list[VisualizationSeries] = Field(default_factory=list)
    data: list[dict[str, Any]]
    metadata: dict[str, Any] = Field(default_factory=dict)
    traceability: VisualizationTraceability


class VisualizationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: VisualizationStatus
    visualization: VisualizationSpec | None = None
    errors: list[VisualizationError] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
