"""Metric definition lookup and knowledge-base retrieval."""

import logging
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import BaseModel, model_validator

from app.core.errors import AppError
from app.services.gemini_client import (
    GeminiNotConfiguredError,
    GeminiRequestError,
)

if TYPE_CHECKING:
    from app.services.knowledge_search_service import (
        KnowledgeSearchService,
    )


logger = logging.getLogger(__name__)


DEFINITION_MESSAGE = (
    "Business metric definitions are not available yet. "
    "They will be added in Module 7. "
    "For now, ask a calculation question such as "
    "'What is the total revenue?'."
)


class MetricDefinitionResult(BaseModel):
    found: bool
    source: Literal["stub", "knowledge_base"]
    is_stub: bool
    query: str
    definition: str | None = None
    message: str

    @model_validator(mode="after")
    def check_honesty(self) -> "MetricDefinitionResult":
        if self.is_stub != (self.source == "stub"):
            raise ValueError(
                "is_stub must be true exactly when the source is 'stub'."
            )

        if self.is_stub and self.found:
            raise ValueError(
                "A stub retriever cannot report a definition as found."
            )

        if self.found and not (
            self.definition and self.definition.strip()
        ):
            raise ValueError(
                "A found result needs a definition text."
            )

        if not self.found and self.definition is not None:
            raise ValueError(
                "A result that was not found cannot contain a definition."
            )

        return self


class MetricRetriever(Protocol):
    """Interface required by the analysis workflow."""

    def lookup(self, question: str) -> MetricDefinitionResult:
        ...


class StubMetricRetriever:
    """Always returns a transparent 'not available' response."""

    def lookup(self, question: str) -> MetricDefinitionResult:
        return MetricDefinitionResult(
            found=False,
            source="stub",
            is_stub=True,
            query=question,
            definition=None,
            message=DEFINITION_MESSAGE,
        )


class KnowledgeBaseMetricRetriever:
    """Retrieve metric definitions using the knowledge base.

    Falls back to StubMetricRetriever when retrieval fails
    or no matching definition is found.
    """

    def __init__(
        self,
        search_service: "KnowledgeSearchService",
    ) -> None:
        self._search_service = search_service

    def lookup(self, question: str) -> MetricDefinitionResult:
        try:
            results = self._search_service.search(
                question,
                top_k=1,
            )

        except (
            GeminiNotConfiguredError,
            GeminiRequestError,
            AppError,
        ):
            logger.exception(
                "Knowledge retrieval failed; "
                "falling back to 'not available'."
            )
            return StubMetricRetriever().lookup(question)

        if not results:
            return StubMetricRetriever().lookup(question)

        top = results[0]

        return MetricDefinitionResult(
            found=True,
            source="knowledge_base",
            is_stub=False,
            query=question,
            definition=top.definition,
            message=(
                "Definition retrieved from the InsightFlow AI "
                "built-in knowledge base "
                f"(similarity {top.similarity:.2f})."
            ),
        )