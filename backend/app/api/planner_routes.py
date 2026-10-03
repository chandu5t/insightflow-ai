"""Additive V2.2 API for planning only; it never loads data or executes a plan."""

from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.core.errors import AppError, ErrorCode
from app.planner.schemas import PlannerRequest, PlannerResponse
from app.planner.service import PlannerModel, create_plan
from app.services.gemini_client import GeminiClient

router = APIRouter(prefix="/analysis", tags=["Analytical Planner v2.2"])


def get_planner_model(settings: Settings = Depends(get_settings)) -> PlannerModel:
    return GeminiClient(
        api_key=settings.gemini_api_key_value,
        model=settings.gemini_model,
        timeout_seconds=settings.gemini_timeout_seconds,
    )


@router.post("/plan", response_model=PlannerResponse)
def plan_question(
    request: PlannerRequest,
    settings: Settings = Depends(get_settings),
    planner_model: PlannerModel = Depends(get_planner_model),
) -> PlannerResponse:
    if len(request.question) > settings.max_question_length:
        raise AppError(
            ErrorCode.QUESTION_TOO_LONG,
            f"The question is longer than {settings.max_question_length} characters.",
            status_code=422,
            details={"max_length": settings.max_question_length},
        )
    return create_plan(request, llm_client=planner_model)
