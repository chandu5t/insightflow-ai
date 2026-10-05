"""Request, state, and result contracts for the V2.3 workflow."""

from typing import Any, Literal, NotRequired, TypedDict
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.planner.schemas import AnalysisPlan

AgentName = Literal["data_understanding", "analysis", "knowledge"]
StepStatus = Literal["pending", "running", "completed", "failed"]
WorkflowStatus = Literal["pending", "running", "completed", "failed"]
FailureCategory = Literal[
    "invalid_plan", "invalid_route", "agent_failure", "workflow_failure", "invalid_state"
]


class MultiAgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    dataset_id: UUID
    analysis_plan: AnalysisPlan

    @field_validator("question")
    @classmethod
    def trim_question(cls, value: str) -> str:
        result = " ".join(value.split())
        if not result:
            raise ValueError("question must not be blank")
        return result


class WorkflowError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: FailureCategory
    code: str
    message: str
    step_id: str | None = None


class AgentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_name: AgentName
    step_id: str | None = None
    status: Literal["completed", "failed"]
    result: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: WorkflowError | None = None


class MultiAgentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workflow_status: Literal["completed", "failed"]
    executed_steps: list[str] = Field(default_factory=list)
    step_statuses: dict[str, StepStatus] = Field(default_factory=dict)
    agent_results: list[AgentResult] = Field(default_factory=list)
    final_result: Any = None
    errors: list[WorkflowError] = Field(default_factory=list)


class WorkflowState(TypedDict):
    question: str
    dataset_id: UUID
    analysis_plan: AnalysisPlan
    current_step: str | None
    completed_steps: list[str]
    step_statuses: dict[str, StepStatus]
    step_results: dict[str, Any]
    agent_outputs: list[AgentResult]
    workflow_status: WorkflowStatus
    errors: list[WorkflowError]
    routing_agent: NotRequired[AgentName | None]
    dataset_context: NotRequired[dict[str, Any]]
    semantic_manifest: NotRequired[Any]
    frame: NotRequired[Any]
    final_result: NotRequired[Any]
