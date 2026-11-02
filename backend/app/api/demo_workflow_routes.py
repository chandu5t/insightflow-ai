"""Additive dashboard endpoint that composes the existing V2 services."""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from app.api.analysis_routes import get_metric_retriever
from app.api.dataset_routes import get_dataset_repository
from app.api.planner_routes import get_planner_model
from app.core.config import Settings, get_settings
from app.multi_agent.demo_workflow import run_demo_workflow
from app.planner.service import PlannerModel
from app.services.dataset_repository import DatasetRepository
from app.services.metric_retriever import MetricRetriever

router = APIRouter(prefix="/analysis", tags=["Demo workflow"])


class DemoWorkflowRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    dataset_id: UUID


class DemoWorkflowResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    question: str
    dataset_id: UUID
    stages: dict[str, str]
    planner_response: Any | None = None
    analysis_plan: Any | None = None
    execution_result: Any | None = None
    verification_result: Any | None = None
    correction_result: Any | None = None
    final_plan: Any | None = None
    final_execution_result: Any | None = None
    final_verification_result: Any | None = None
    visualization_result: Any | None = None
    errors: list[dict[str, Any]] = Field(default_factory=list)


@router.post("/workflow", response_model=DemoWorkflowResponse)
def run_dashboard_workflow(
    request: DemoWorkflowRequest,
    repository: DatasetRepository = Depends(get_dataset_repository),
    retriever: MetricRetriever = Depends(get_metric_retriever),
    planner_model: PlannerModel = Depends(get_planner_model),
    settings: Settings = Depends(get_settings),
) -> DemoWorkflowResponse:
    if len(request.question) > settings.max_question_length:
        from app.core.errors import AppError, ErrorCode

        raise AppError(
            ErrorCode.QUESTION_TOO_LONG,
            f"The question exceeds the configured maximum of {settings.max_question_length} characters.",
            status_code=422,
            details={"max_length": settings.max_question_length},
        )
    if not request.question.strip():
        from app.core.errors import AppError, ErrorCode

        raise AppError(ErrorCode.VALIDATION_ERROR, "A non-empty question is required.", status_code=422)
    return DemoWorkflowResponse(**run_demo_workflow(
        question=request.question,
        dataset_id=request.dataset_id,
        repository=repository,
        retriever=retriever,
        planner_model=planner_model,
    ))
