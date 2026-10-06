"""Additive V2.2 API for planning only; it never loads data or executes a plan."""

from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.core.errors import AppError, ErrorCode
from app.planner.metric_registry import validate_authoritative_planner_context
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
    authorized_definitions = None
    context = request.dataset_context
    context_has_trusted_semantics = bool(
        context
        and (
            context.manifest_version
            or any(role != "unknown" for role in context.semantic_roles.values())
        )
    )
    if request.metric_definitions or context_has_trusted_semantics:
        if context is None or not validate_authoritative_planner_context(
            dataset_sha256=context.dataset_sha256,
            manifest_version=context.manifest_version,
            columns=context.columns,
            semantic_roles=context.semantic_roles,
            definitions=request.metric_definitions,
        ):
            raise AppError(
                ErrorCode.VALIDATION_ERROR,
                "Metric definitions must match a trusted dataset-bound registry entry.",
                status_code=422,
            )
        authorized_definitions = request.metric_definitions or []
    return create_plan(
        request,
        llm_client=planner_model,
        authorized_metric_definitions=authorized_definitions,
    )
