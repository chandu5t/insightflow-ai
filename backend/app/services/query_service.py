"""Answers a question by running the LangGraph workflow (see app/workflow)."""

from collections.abc import Callable

import pandas as pd
from pydantic import BaseModel

from app.core.config import Settings
from app.core.errors import AppError, ErrorCode
from app.schemas.query_schema import QueryPlan, QueryResponse, ValidationInfo
from app.services.dataset_repository import DatasetRepository
from app.services.explainer import explain
from app.services.gemini_client import LlmClient
from app.services.metric_retriever import DEFINITION_MESSAGE, MetricRetriever, StubMetricRetriever
from app.services.response_builder import NO_FILTERS_NOTE, SUPPORTED_HELP
from app.services.result_validator import merge_validations, reconcile_result, validate_result
from app.utils.dataframe_utils import load_dataframe
from app.utils.upload_validator import parse_dataset_id
from app.workflow.graph import run_query_workflow
from app.workflow.nodes import WorkflowDependencies

__all__ = ["answer_question", "validate_for_workflow", "DEFINITION_MESSAGE", "SUPPORTED_HELP", "NO_FILTERS_NOTE"]


def _clean_question(question: str, max_length: int) -> str:
    """Collapse spaces and remove control characters. Reject empty or too long questions."""
    cleaned = "".join(ch for ch in " ".join(question.split()) if ch.isprintable())
    if not cleaned:
        raise AppError(ErrorCode.EMPTY_QUESTION, "Please type a question.", status_code=422)
    if len(cleaned) > max_length:
        raise AppError(
            ErrorCode.QUESTION_TOO_LONG,
            f"The question is longer than {max_length} characters.",
            status_code=422,
            details={"max_length": max_length},
        )
    return cleaned


def validate_for_workflow(
    frame: pd.DataFrame, plan: QueryPlan, result: BaseModel, expected_row_count: int | None
) -> ValidationInfo:
    """The existing checks plus the independent reconciliation checks.

    `validate_result` is looked up here at call time, so patching `query_service.validate_result` still works.
    """
    return merge_validations(
        validate_result(frame, plan, result),
        reconcile_result(frame, plan, result, expected_row_count=expected_row_count),
    )

def answer_question(
    *,
    raw_dataset_id: str,
    question: str,
    settings: Settings,
    repository: DatasetRepository,
    llm_client: LlmClient,
    retriever: MetricRetriever | None = None,
    explainer: Callable[[QueryPlan, BaseModel, str], str] | None = None,
) -> QueryResponse:
    """Answer one question about a stored dataset. Raises AppError for a bad id, question or dataset."""
    dataset_id = parse_dataset_id(raw_dataset_id)
    clean = _clean_question(question, settings.max_question_length)
    metadata = repository.get(dataset_id)  # raises DATASET_NOT_FOUND (404) for an unknown dataset
    frame = load_dataframe(repository.get_csv_path(dataset_id))

    deps = WorkflowDependencies(
        llm_client=llm_client,
        settings=settings,
        retriever=retriever or StubMetricRetriever(),
        validate=validate_for_workflow,
        explain=explainer or explain,
    )
    final_state = run_query_workflow(
        deps,
        {
            "dataset_id": dataset_id,
            "question": clean,
            "frame": frame,
            "expected_row_count": metadata.row_count,
            "trace": [],
        },
    )
    return final_state["response"]