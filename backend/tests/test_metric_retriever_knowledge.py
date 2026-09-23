"""Tests for KnowledgeBaseMetricRetriever's fallback behavior."""

from app.core.errors import AppError, ErrorCode
from app.services.gemini_client import GeminiNotConfiguredError
from app.services.knowledge_search_service import KnowledgeSearchResult
from app.services.metric_retriever import KnowledgeBaseMetricRetriever


class FakeSearchService:
    def __init__(self, results=None, error=None) -> None:
        self._results = results or []
        self._error = error

    def search(self, query, *, top_k=1):
        if self._error:
            raise self._error
        return self._results


def test_a_match_above_threshold_is_returned_with_the_knowledge_base_source() -> None:
    result = KnowledgeSearchResult(
        id="1", slug="average-order-value", name="AOV", definition="AOV is ...",
        category="sales", source="InsightFlow AI built-in knowledge base", similarity=0.87,
    )
    retriever = KnowledgeBaseMetricRetriever(FakeSearchService(results=[result]))

    outcome = retriever.lookup("What does average order value mean?")

    assert outcome.found and outcome.source == "knowledge_base" and outcome.is_stub is False
    assert outcome.definition == "AOV is ..." and "0.87" in outcome.message


def test_no_results_falls_back_to_the_stub_response() -> None:
    outcome = KnowledgeBaseMetricRetriever(FakeSearchService(results=[])).lookup("What is zorblatt?")

    assert outcome.found is False and outcome.is_stub is True and outcome.source == "stub"
    assert "Module 7" in outcome.message  # exact stub wording from Module 5


def test_embedding_failure_falls_back_to_the_stub_response() -> None:
    outcome = KnowledgeBaseMetricRetriever(
        FakeSearchService(error=GeminiNotConfiguredError("no key"))
    ).lookup("What is revenue?")

    assert outcome.found is False and outcome.is_stub is True


def test_database_failure_falls_back_to_the_stub_response() -> None:
    outcome = KnowledgeBaseMetricRetriever(
        FakeSearchService(error=AppError(ErrorCode.DATABASE_ERROR, "down", status_code=503))
    ).lookup("What is revenue?")

    assert outcome.found is False and outcome.is_stub is True