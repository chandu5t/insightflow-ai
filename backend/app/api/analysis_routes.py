"""Route for asking questions about a dataset. The logic is in query_service.py."""

from fastapi import APIRouter, Depends

from app.api.dataset_routes import get_dataset_repository
from app.core.config import Settings, get_settings
from app.schemas.error_schema import ErrorResponse
from app.schemas.query_schema import QueryRequest, QueryResponse
from app.services import query_service
from app.services.dataset_repository import DatasetRepository
from app.services.gemini_client import GeminiClient, LlmClient

router = APIRouter(prefix="/analysis", tags=["Analysis"])


def get_llm_client(settings: Settings = Depends(get_settings)) -> LlmClient:
    """The Gemini client. Tests replace this dependency with a fake, so they never call Gemini."""
    return GeminiClient(
        api_key=settings.gemini_api_key_value,
        model=settings.gemini_model,
        timeout_seconds=settings.gemini_timeout_seconds,
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
def query_dataset(
    request: QueryRequest,
    settings: Settings = Depends(get_settings),
    repository: DatasetRepository = Depends(get_dataset_repository),
    llm_client: LlmClient = Depends(get_llm_client),
) -> QueryResponse:
    """Ask a question in plain English about an uploaded dataset."""
    return query_service.answer_question(
        raw_dataset_id=request.dataset_id,
        question=request.question,
        settings=settings,
        repository=repository,
        llm_client=llm_client,
    )