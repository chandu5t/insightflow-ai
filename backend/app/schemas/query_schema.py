"""Pydantic models for questions: the query plan, the request and the response."""

from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.profile_schema import MissingValueReport
from app.schemas.tool_schema import AggregationResult, GroupingResult, RankingResult

Intent = Literal[
    "total_revenue", "aggregate", "group", "rank", "missing_values", "definition", "unsupported"
]
ToolName = Literal["revenue", "aggregation", "grouping", "ranking", "missing_values"]
Metric = Literal["revenue", "quantity", "unit_price", "records", "average_order_value"]
Aggregation = Literal["sum", "average", "count", "min", "max"]
SortDirection = Literal["asc", "desc"]

# Each intent may use exactly one tool (or none).
INTENT_TOOL: dict[str, str | None] = {
    "total_revenue": "revenue",
    "aggregate": "aggregation",
    "group": "grouping",
    "rank": "ranking",
    "missing_values": "missing_values",
    "definition": None,
    "unsupported": None,
}
_ALL = {"sum", "average", "count", "min", "max"}
AGGREGATE_ALLOWED: dict[str, set[str]] = {
    "revenue": {"sum", "average", "min", "max"},
    "quantity": _ALL,
    "unit_price": _ALL,
    "records": {"count"},
    "average_order_value": {"average"},
}
GROUPED_ALLOWED: dict[str, set[str]] = {
    "revenue": {"sum", "average", "count"},
    "quantity": {"sum", "average", "count"},
    "unit_price": {"sum", "average", "count"},
    "records": {"count"},
}


class QueryPlan(BaseModel):
    """What the user wants, in a form the dispatcher can check. It never contains code."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    intent: Intent
    tool_name: ToolName | None = None
    metric: Metric | None = None
    aggregation: Aggregation | None = None
    group_by: str | None = Field(default=None, min_length=1, max_length=100)
    sort_order: SortDirection | None = None
    limit: int | None = Field(default=None, ge=1, le=100)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: str = Field(default="", max_length=300)

    @field_validator("reasoning", mode="before")
    @classmethod
    def shorten_reasoning(cls, value: object) -> object:
        return value[:300] if isinstance(value, str) else value

    @model_validator(mode="after")
    def check_combination(self) -> "QueryPlan":
        expected_tool = INTENT_TOOL[self.intent]
        if self.tool_name != expected_tool:
            raise ValueError(
                f"intent '{self.intent}' needs tool_name {expected_tool!r}, got {self.tool_name!r}"
            )
        if self.intent == "aggregate":
            self._check_metric(AGGREGATE_ALLOWED)
        elif self.intent in ("group", "rank"):
            if not self.group_by:
                raise ValueError(f"intent '{self.intent}' needs group_by")
            self._check_metric(GROUPED_ALLOWED)
            if self.intent == "rank" and self.sort_order is None:
                raise ValueError("intent 'rank' needs sort_order")
        return self

    def _check_metric(self, allowed: dict[str, set[str]]) -> None:
        if self.metric is None or self.aggregation is None:
            raise ValueError(f"intent '{self.intent}' needs metric and aggregation")
        if self.aggregation not in allowed[self.metric]:
            raise ValueError(
                f"aggregation '{self.aggregation}' is not allowed for metric '{self.metric}'"
            )


class QueryRequest(BaseModel):
    """Body of POST /analysis/query. The dataset id is checked later, to give a clear error code."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    question: str


class ClassifierInfo(BaseModel):
    used: Literal["gemini", "rule_based"]
    model: str | None = None
    fallback_reason: str | None = None  # for example GEMINI_NOT_CONFIGURED


class ValidationCheck(BaseModel):
    name: str
    passed: bool
    detail: str | None = None


class ValidationInfo(BaseModel):
    status: Literal["passed", "failed", "not_run"]
    checks: list[ValidationCheck] = Field(default_factory=list)


class QueryErrorInfo(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


ToolResult = Annotated[
    AggregationResult | GroupingResult | RankingResult | MissingValueReport,
    Field(discriminator="tool"),
]


class QueryResponse(BaseModel):
    """Response of POST /analysis/query."""

    status: Literal["success", "unsupported", "insufficient_data", "error"]
    question: str
    dataset_id: UUID
    classifier: ClassifierInfo
    query_plan: QueryPlan | None = None
    tool_used: str | None = None
    result: ToolResult | None = None
    explanation: str | None = None
    calculation_method: str | None = None
    assumptions: list[str] = Field(default_factory=list)
    validation: ValidationInfo = Field(default_factory=lambda: ValidationInfo(status="not_run"))
    message: str | None = None
    error: QueryErrorInfo | None = None