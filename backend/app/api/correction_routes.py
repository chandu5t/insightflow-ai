"""Additive V2.5 correction endpoint."""

from fastapi import APIRouter, Depends

from app.api.analysis_routes import get_metric_retriever
from app.api.dataset_routes import get_dataset_repository
from app.core.config import Settings, get_settings
from app.core.errors import AppError, ErrorCode
from app.self_correction.schemas import CorrectionRequest, CorrectionResponse
from app.self_correction.service import correct_execution
from app.services.dataset_repository import DatasetRepository
from app.services.metric_retriever import MetricRetriever

router = APIRouter(prefix="/analysis", tags=["Self-Correction v2.5"])


@router.post("/correct", response_model=CorrectionResponse)
def correct_analysis(
    request: CorrectionRequest,
    repository: DatasetRepository = Depends(get_dataset_repository),
    retriever: MetricRetriever = Depends(get_metric_retriever),
    settings: Settings = Depends(get_settings),
) -> CorrectionResponse:
    if len(request.question) > settings.max_question_length:
        raise AppError(
            ErrorCode.QUESTION_TOO_LONG,
            f"The question is longer than {settings.max_question_length} characters.",
            status_code=422,
            details={"max_length": settings.max_question_length},
        )
    return correct_execution(request, repository, retriever)
