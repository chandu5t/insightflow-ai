"""Helpers for workflow tests. No test calls the real Gemini API."""

from uuid import uuid4

import pandas as pd

from app.core.config import Settings
from app.services.explainer import explain
from app.services.gemini_client import GeminiNotConfiguredError
from app.services.metric_retriever import MetricDefinitionResult, StubMetricRetriever
from app.services.query_service import validate_for_workflow
from app.workflow.graph import run_query_workflow
from app.workflow.nodes import WorkflowDependencies
from tests.fakes import FakeLlm


class FakeRetriever:
    """Records the questions it receives. Returns the given result, or the stub's answer."""

    def __init__(self, result: MetricDefinitionResult | None = None) -> None:
        self.result = result
        self.calls: list[str] = []

    def lookup(self, question: str) -> MetricDefinitionResult:
        self.calls.append(question)
        return self.result if self.result is not None else StubMetricRetriever().lookup(question)


class SpyExplainer:
    """Wraps the real explainer, or returns fixed text, and counts the calls."""

    def __init__(self, text: str | None = None) -> None:
        self.text = text
        self.calls = 0

    def __call__(self, plan, result, symbol):
        self.calls += 1
        return self.text if self.text is not None else explain(plan, result, symbol)


def make_deps(*, llm=None, retriever=None, validate=None, explainer=None, **settings) -> WorkflowDependencies:
    return WorkflowDependencies(
        llm_client=llm if llm is not None else FakeLlm(GeminiNotConfiguredError("no key")),
        settings=Settings(gemini_api_key=None, **settings),
        retriever=retriever if retriever is not None else StubMetricRetriever(),
        validate=validate if validate is not None else validate_for_workflow,
        explain=explainer if explainer is not None else explain,
    )


def run_workflow(question: str, frame: pd.DataFrame, *, deps=None, expected_row_count=None):
    state = {
        "dataset_id": uuid4(),
        "question": question,
        "frame": frame,
        "expected_row_count": len(frame) if expected_row_count is None else expected_row_count,
        "trace": [],
    }
    return run_query_workflow(deps or make_deps(), state)