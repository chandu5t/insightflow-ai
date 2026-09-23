"""Route for direct knowledge-base search. Logic is in knowledge_search_service.py."""

from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.core.errors import AppError, ErrorCode
from app.db.engine import get_session_factory
from app.schemas.error_schema import ErrorResponse
from app.schemas.knowledge_schema import (
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
    KnowledgeSearchResultItem,
)
from app.services.embedding_service import GeminiEmbeddingService
from app.services.gemini_client import (
    GeminiNotConfiguredError,
    GeminiRequestError,
)
from app.services.knowledge_search_service import KnowledgeSearchService


router = APIRouter(prefix="/knowledge", tags=["Knowledge"])


def get_knowledge_search_service(
    settings: Settings = Depends(get_settings),
) -> KnowledgeSearchService:
    """Build the knowledge search service using application settings."""

    embedding_service = GeminiEmbeddingService(
        api_key=settings.gemini_api_key_value,
        model=settings.gemini_embedding_model,
        dimensions=settings.knowledge_embedding_dimensions,
        timeout_seconds=settings.gemini_timeout_seconds,
    )

    return KnowledgeSearchService(
        get_session_factory(settings),
        embedding_service,
        top_k_limit=settings.knowledge_top_k,
        threshold=settings.knowledge_similarity_threshold,
    )


@router.post(
    "/search",
    response_model=KnowledgeSearchResponse,
    responses={
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def search_knowledge(
    request: KnowledgeSearchRequest,
    settings: Settings = Depends(get_settings),
    search_service: KnowledgeSearchService = Depends(
        get_knowledge_search_service
    ),
) -> KnowledgeSearchResponse:
    """Search the built-in business-metric knowledge base."""

    if settings.storage_backend != "postgres":
        raise AppError(
            ErrorCode.KNOWLEDGE_BASE_UNAVAILABLE,
            (
                "The knowledge base requires PostgreSQL storage "
                "(STORAGE_BACKEND=postgres)."
            ),
            status_code=503,
        )

    try:
        results = search_service.search(
            request.query,
            top_k=request.top_k,
        )

    except GeminiNotConfiguredError as exc:
        raise AppError(
            ErrorCode.KNOWLEDGE_BASE_UNAVAILABLE,
            "Knowledge search requires a configured Gemini API key.",
            status_code=503,
        ) from exc

    except GeminiRequestError as exc:
        raise AppError(
            ErrorCode.KNOWLEDGE_BASE_UNAVAILABLE,
            "Knowledge search is temporarily unavailable.",
            status_code=503,
        ) from exc

    return KnowledgeSearchResponse(
        query=request.query,
        results=[
            KnowledgeSearchResultItem(**vars(item))
            for item in results
        ],
        fallback=len(results) == 0,
    )