"""Metric definition lookup. Module 5 has only a STUB. Module 7 will add a real retriever.

The stub NEVER claims to have found a definition. The result model enforces that rule.
"""

from typing import Literal, Protocol

from pydantic import BaseModel, model_validator

DEFINITION_MESSAGE = (
    "Business metric definitions are not available yet. They will be added in Module 7. "
    "For now, ask a calculation question such as 'What is the total revenue?'."
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
            raise ValueError("is_stub must be true exactly when the source is 'stub'.")
        if self.is_stub and self.found:
            raise ValueError("A stub retriever cannot report a definition as found.")
        if self.found and not (self.definition and self.definition.strip()):
            raise ValueError("A found result needs a definition text.")
        if not self.found and self.definition is not None:
            raise ValueError("A result that was not found cannot contain a definition.")
        return self


class MetricRetriever(Protocol):
    """What the workflow needs. A future RAG retriever only has to provide this one method."""

    def lookup(self, question: str) -> MetricDefinitionResult: ...


class StubMetricRetriever:
    """Always answers 'not available'. Nothing is retrieved, nothing is invented."""

    def lookup(self, question: str) -> MetricDefinitionResult:
        return MetricDefinitionResult(
            found=False,
            source="stub",
            is_stub=True,
            query=question,
            definition=None,
            message=DEFINITION_MESSAGE,
        )