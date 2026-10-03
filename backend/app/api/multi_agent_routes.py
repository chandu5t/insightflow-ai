"""Additive V2.3 multi-agent workflow API."""

from fastapi import APIRouter, Depends

from app.api.analysis_routes import get_metric_retriever
from app.api.dataset_routes import get_dataset_repository
from app.core.config import Settings, get_settings
from app.multi_agent.schemas import MultiAgentRequest, MultiAgentResult
from app.multi_agent.workflow import run_multi_agent_workflow
from app.services.dataset_repository import DatasetRepository
from app.services.metric_retriever import MetricRetriever

router = APIRouter(prefix="/analysis", tags=["Multi-Agent Workflow v2.3"])


@router.post("/multi-agent", response_model=MultiAgentResult)
def run_multi_agent(
    request: MultiAgentRequest,
    repository: DatasetRepository = Depends(get_dataset_repository),
    retriever: MetricRetriever = Depends(get_metric_retriever),
    settings: Settings = Depends(get_settings),
) -> MultiAgentResult:
    if len(request.question) > settings.max_question_length:
        from app.core.errors import AppError, ErrorCode
        raise AppError(ErrorCode.QUESTION_TOO_LONG,
                       f"The question is longer than {settings.max_question_length} characters.",
                       status_code=422, details={"max_length": settings.max_question_length})
    return run_multi_agent_workflow(request, repository, retriever)
