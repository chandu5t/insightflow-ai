"""The state that flows through the LangGraph workflow.

A TypedDict is LangGraph's native style. The values are the project's own Pydantic objects.
IMPORTANT: a state key must never have the same name as a graph node (LangGraph refuses that).
Nodes: classify, route, execute, validate, explain, metric_definition, respond.
"""

import operator
from typing import Annotated, Any, Literal, TypedDict
from uuid import UUID

import pandas as pd
from pydantic import BaseModel

from app.schemas.query_schema import ClassifierInfo, QueryErrorInfo, QueryPlan, QueryResponse, ValidationInfo
from app.services.gemini_classifier import Classification
from app.services.metric_retriever import MetricDefinitionResult

Route = Literal["tool", "metric_definition", "unsupported", "invalid"]
Outcome = Literal["success", "unsupported", "insufficient_data", "error"]


class WorkflowState(TypedDict, total=False):
    # input (set before the graph starts)
    dataset_id: UUID
    question: str
    frame: pd.DataFrame  # fine without a checkpointer. Move it out of the state if one is added.
    expected_row_count: int
    # classify
    classification: Classification
    query_plan: QueryPlan
    classifier: ClassifierInfo
    assumptions: list[str]
    # route
    route_decision: Route
    tool_name: str | None
    tool_arguments: dict[str, Any]
    # execute
    raw_result: BaseModel
    # validate
    validation: ValidationInfo
    validated_result: BaseModel
    # explain / metric definition
    explanation: str
    definition_result: MetricDefinitionResult
    # outcome (used by respond)
    status: Outcome
    message: str | None
    error: QueryErrorInfo | None
    response: QueryResponse
    # the names of the nodes that ran, in order (each node adds its own name)
    trace: Annotated[list[str], operator.add]