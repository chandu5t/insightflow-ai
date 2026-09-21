"""The nodes of the LangGraph workflow. Each node: state in, state updates out.

Nodes do NOT calculate. Calculations stay in the Module 3 tools, reached only through the dispatcher.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd
from pydantic import BaseModel

from app.core.config import Settings
from app.core.errors import AppError, ErrorCode, ToolError
from app.schemas.query_schema import QueryPlan, ValidationInfo
from app.services.gemini_classifier import classify_question
from app.services.gemini_client import LlmClient
from app.services.metric_retriever import MetricRetriever
from app.services.query_dispatcher import TOOL_HANDLERS, dispatch
from app.services.response_builder import build_response, error_info, unsupported_message
from app.workflow.state import Route, WorkflowState

logger = logging.getLogger(__name__)

ValidateFn = Callable[[pd.DataFrame, QueryPlan, BaseModel, int | None], ValidationInfo]
ExplainFn = Callable[[QueryPlan, BaseModel, str], str]


@dataclass(frozen=True)
class WorkflowDependencies:
    """Everything the nodes need from outside. Tests pass fakes here, so nothing calls Gemini."""

    llm_client: LlmClient
    settings: Settings
    retriever: MetricRetriever
    validate: ValidateFn
    explain: ExplainFn


# ---- routing functions (pure: they only read the state) -------------------------------------
def route_after_route(state: WorkflowState) -> Route:
    return state["route_decision"]


def route_after_execute(state: WorkflowState) -> str:
    return "validate" if state.get("raw_result") is not None else "respond"


def route_after_validate(state: WorkflowState) -> str:
    return "explain" if state.get("validated_result") is not None else "respond"


def _tool_arguments(plan: QueryPlan) -> dict[str, Any]:
    """The plan fields the dispatcher will use. Recorded in the state for logging and tests."""
    fields = {
        "metric": plan.metric,
        "aggregation": plan.aggregation,
        "group_by": plan.group_by,
        "sort_order": plan.sort_order,
        "limit": plan.limit,
    }
    return {name: value for name, value in fields.items() if value is not None}


def make_nodes(deps: WorkflowDependencies) -> dict[str, Callable[[WorkflowState], dict[str, Any]]]:
    """Create the nodes. They share `deps` through a closure."""

    def classify(state: WorkflowState) -> dict[str, Any]:
        columns = [str(name) for name in state["frame"].columns]
        classification = classify_question(state["question"], columns, deps.llm_client, deps.settings)
        return {
            "classification": classification,
            "query_plan": classification.plan,
            "classifier": classification.info,
            "assumptions": list(classification.assumptions),
            "trace": ["classify"],
        }

    def route(state: WorkflowState) -> dict[str, Any]:
        plan = state["query_plan"]
        if plan.intent == "definition":
            return {"route_decision": "metric_definition", "trace": ["route"]}
        if plan.intent == "unsupported":
            message = unsupported_message(plan.reasoning)
            return {
                "route_decision": "unsupported",
                "status": "unsupported",
                "message": message,
                "error": error_info(ErrorCode.UNSUPPORTED_QUESTION, message, {"reason": "unsupported_question"}),
                "trace": ["route"],
            }
        if plan.tool_name in TOOL_HANDLERS:
            return {
                "route_decision": "tool",
                "tool_name": plan.tool_name,
                "tool_arguments": _tool_arguments(plan),
                "trace": ["route"],
            }
        message = "The requested tool is not available."
        return {
            "route_decision": "invalid",
            "status": "error",
            "message": message,
            "error": error_info(ErrorCode.UNSUPPORTED_TOOL, message, {"tool_name": str(plan.tool_name)[:50]}),
            "trace": ["route"],
        }

    def execute(state: WorkflowState) -> dict[str, Any]:
        try:
            result = dispatch(state["frame"], state["query_plan"])
        except ToolError as exc:
            status = "error" if exc.code == ErrorCode.INVALID_PARAMETER else "insufficient_data"
            return {
                "status": status,
                "message": exc.message,
                "error": error_info(exc.code, exc.message, exc.details),
                "trace": ["execute"],
            }
        except AppError as exc:  # invalid plan arguments
            return {
                "status": "error",
                "message": exc.message,
                "error": error_info(exc.code, exc.message, exc.details),
                "trace": ["execute"],
            }
        return {"raw_result": result, "trace": ["execute"]}

    def validate(state: WorkflowState) -> dict[str, Any]:
        result = state["raw_result"]
        info = deps.validate(state["frame"], state["query_plan"], result, state.get("expected_row_count"))
        if info.status == "failed":
            message = "The result could not be validated, so it is not shown."
            failed = [check.name for check in info.checks if not check.passed]
            return {
                "validation": info,
                "status": "error",
                "message": message,
                "error": error_info(ErrorCode.RESULT_VALIDATION_FAILED, message, {"failed_checks": failed}),
                "trace": ["validate"],
            }
        return {"validation": info, "validated_result": result, "trace": ["validate"]}

    def explain(state: WorkflowState) -> dict[str, Any]:
        text = deps.explain(state["query_plan"], state["validated_result"], deps.settings.currency_symbol)
        return {"explanation": text, "status": "success", "trace": ["explain"]}

    def metric_definition(state: WorkflowState) -> dict[str, Any]:
        found = deps.retriever.lookup(state["question"])
        if found.found:
            return {
                "definition_result": found,
                "status": "success",
                "explanation": found.definition,
                "trace": ["metric_definition"],
            }
        return {
            "definition_result": found,
            "status": "unsupported",
            "message": found.message,
            "error": error_info(ErrorCode.UNSUPPORTED_QUESTION, found.message, {"reason": "definition_not_available"}),
            "trace": ["metric_definition"],
        }

    def respond(state: WorkflowState) -> dict[str, Any]:
        response = build_response(state, currency_symbol=deps.settings.currency_symbol)
        path = " > ".join([*state.get("trace", []), "respond"])
        # Only technical facts are logged. The question text is not.
        logger.info(
            "Workflow finished: dataset=%s classifier=%s route=%s status=%s path=[%s] question_length=%d",
            state["dataset_id"], state["classifier"].used, state.get("route_decision"),
            state["status"], path, len(state["question"]),
        )
        return {"response": response, "trace": ["respond"]}

    return {
        "classify": classify,
        "route": route,
        "execute": execute,
        "validate": validate,
        "explain": explain,
        "metric_definition": metric_definition,
        "respond": respond,
    }