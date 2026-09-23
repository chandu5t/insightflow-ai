"""Route for asking questions about a dataset. The logic is in query_service.py and app/workflow."""

import logging
from collections.abc import Callable

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.dataset_routes import get_dataset_repository
from app.core.config import Settings, get_settings
from app.db.engine import get_session_factory
from app.schemas.error_schema import ErrorResponse
from app.schemas.query_schema import QueryPlan, QueryRequest, QueryResponse
from app.services import query_service
from app.services.analysis_history_repository import (
    AnalysisHistoryRepository,
    NullAnalysisHistoryRepository,
    PostgresAnalysisHistoryRepository,
)
from app.services.dataset_repository import DatasetRepository
from app.services.explainer import explain
from app.services.gemini_client import GeminiClient, LlmClient
from app.services.metric_retriever import MetricRetriever, StubMetricRetriever, KnowledgeBaseMetricRetriever
from app.db.engine import get_session_factory
from app.services.embedding_service import GeminiEmbeddingService
from app.services.knowledge_search_service import KnowledgeSearchService

router = APIRouter(prefix="/analysis", tags=["Analysis"])
logger = logging.getLogger(__name__)


def get_llm_client(settings: Settings = Depends(get_settings)) -> LlmClient:
    """The Gemini client. Tests replace this dependency with a fake, so they never call Gemini."""
    return GeminiClient(
        api_key=settings.gemini_api_key_value,
        model=settings.gemini_model,
        timeout_seconds=settings.gemini_timeout_seconds,
    )


def get_metric_retriever(settings: Settings = Depends(get_settings)) -> MetricRetriever:
    """Real knowledge-base lookup when Postgres + Gemini are both configured; the
    Module 5 stub otherwise (JSON mode, or no Gemini key -- e.g. most test runs).
    """
    if settings.storage_backend == "postgres" and settings.gemini_api_key_value:
        embedding_service = GeminiEmbeddingService(
            api_key=settings.gemini_api_key_value,
            model=settings.gemini_embedding_model,
            dimensions=settings.knowledge_embedding_dimensions,
            timeout_seconds=settings.gemini_timeout_seconds,
        )
        search_service = KnowledgeSearchService(
            get_session_factory(settings), embedding_service,
            top_k_limit=settings.knowledge_top_k, threshold=settings.knowledge_similarity_threshold,
        )
        return KnowledgeBaseMetricRetriever(search_service)
    return StubMetricRetriever()


def get_explainer() -> Callable[[QueryPlan, BaseModel, str], str]:
    """Builds the explanation text. Tests replace it to prove that a bad explanation is rejected."""
    return explain


def get_analysis_history_repository(settings: Settings = Depends(get_settings)) -> AnalysisHistoryRepository:
    """Where answered questions are recorded (Module 6). A no-op when STORAGE_BACKEND=json,
    so JSON-mode requests never need PostgreSQL running.
    """
    if settings.storage_backend == "postgres":
        return PostgresAnalysisHistoryRepository(get_session_factory(settings))
    return NullAnalysisHistoryRepository()


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
    retriever: MetricRetriever = Depends(get_metric_retriever),
    explainer: Callable[[QueryPlan, BaseModel, str], str] = Depends(get_explainer),
    history: AnalysisHistoryRepository = Depends(get_analysis_history_repository),
) -> QueryResponse:
    """Ask a question in plain English about an uploaded dataset.

    A bad dataset id, an empty question, or an unknown dataset raise AppError INSIDE
    answer_question() below and are returned exactly as before Module 6 -- they never
    reach a QueryResponse, so nothing is recorded for them (see D-057). Everything that
    DOES produce a QueryResponse (success, unsupported, insufficient_data, or an internal
    "error" status such as a failed validation or grounding check) is recorded exactly
    once, right here, after answer_question() returns -- so there is exactly one call
    site and no possibility of a duplicate record for one request.
    """
    response = query_service.answer_question(
        raw_dataset_id=request.dataset_id,
        question=request.question,
        settings=settings,
        repository=repository,
        llm_client=llm_client,
        retriever=retriever,
        explainer=explainer,
    )
    try:
        history.record(dataset_id=response.dataset_id, question=response.question, response=response)
    except Exception:
        # A history-write failure must never turn a successfully answered question into
        # a failed HTTP response -- the user's actual answer is unaffected either way.
        logger.exception("Unexpected error while recording analysis history")
    return response