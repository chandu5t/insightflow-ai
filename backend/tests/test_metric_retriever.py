"""Tests for the metric retriever stub."""

import pytest
from pydantic import ValidationError

from app.services.metric_retriever import DEFINITION_MESSAGE, MetricDefinitionResult, StubMetricRetriever


def test_stub_result_has_the_expected_structure() -> None:
    result = StubMetricRetriever().lookup("What is average order value?")

    assert result.found is False and result.is_stub is True and result.source == "stub"
    assert result.definition is None and result.query == "What is average order value?"
    assert result.message == DEFINITION_MESSAGE and "Module 7" in result.message


@pytest.mark.parametrize("question", ["What is revenue?", "define conversion rate", "", "anything"])
def test_the_stub_never_claims_to_have_retrieved_anything(question: str) -> None:
    result = StubMetricRetriever().lookup(question)

    assert not result.found and result.definition is None and result.is_stub


def test_a_stub_result_cannot_claim_success() -> None:
    with pytest.raises(ValidationError, match="stub"):
        MetricDefinitionResult(found=True, source="stub", is_stub=True, query="q", definition="x", message="m")


def test_stub_flag_must_match_the_source() -> None:
    with pytest.raises(ValidationError):
        MetricDefinitionResult(found=False, source="stub", is_stub=False, query="q", message="m")


@pytest.mark.parametrize("definition", [None, "", "   "])
def test_a_found_result_needs_real_text(definition) -> None:
    with pytest.raises(ValidationError):
        MetricDefinitionResult(found=True, source="knowledge_base", is_stub=False, query="q",
                               definition=definition, message="m")


def test_a_missing_result_cannot_carry_a_definition() -> None:
    with pytest.raises(ValidationError):
        MetricDefinitionResult(found=False, source="knowledge_base", is_stub=False, query="q",
                               definition="invented", message="m")