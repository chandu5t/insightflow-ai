"""Ablation configurations (A1–A5) and controlled error injection matching V2.1 §21–§22 and Contract v1.1 §9, §11."""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.multi_agent.schemas import AgentResult, MultiAgentResult
from app.planner.schemas import AnalysisPlan, PlanStep


class InjectedError(BaseModel):
    """Specification of an intentionally introduced analytical error for H3/H4 evaluation."""

    model_config = ConfigDict(extra="forbid")

    error_type: Literal[
        "wrong_aggregation", "wrong_column", "incorrect_filter", "incorrect_formula",
        "incorrect_ranking", "incorrect_intermediate_result",
    ]
    target_step_id: str | None = None
    original_value: Any = None
    injected_value: Any = None
    description: str = ""


class AblationConfig(BaseModel):
    """Configuration toggles for A1–A5 ablations."""

    model_config = ConfigDict(extra="forbid")

    disable_planner: bool = False
    disable_verification: bool = False
    disable_self_correction: bool = False
    disable_rag: bool = False


def apply_error_injection(
    plan: AnalysisPlan,
    multi_agent_result: MultiAgentResult,
    injection: InjectedError,
) -> tuple[AnalysisPlan, MultiAgentResult, bool]:
    """Inject a controlled analytical error into plan or execution results for testing verifier detection."""
    mutated_plan = plan.model_copy(deep=True)
    mutated_result = multi_agent_result.model_copy(deep=True)
    applied = False

    if injection.error_type in {"wrong_column", "incorrect_filter", "incorrect_formula", "incorrect_ranking"}:
        for step in mutated_plan.steps:
            if step.step_id == injection.target_step_id and injection.injected_value is not None:
                field = {
                    "wrong_column": "column",
                    "incorrect_filter": "conditions",
                    "incorrect_formula": "formula",
                    "incorrect_ranking": "order",
                }[injection.error_type]
                if field not in step.parameters:
                    break
                injection.original_value = step.parameters[field]
                step.parameters[field] = injection.injected_value
                applied = True
                break

    elif injection.error_type == "wrong_aggregation":
        for step in mutated_plan.steps:
            if step.step_id == injection.target_step_id and injection.injected_value is not None:
                if "function" not in step.parameters:
                    break
                injection.original_value = step.parameters["function"]
                step.parameters["function"] = injection.injected_value
                applied = True
                break

    elif injection.error_type == "incorrect_intermediate_result":
        # Mutate result of an agent step
        for agent in mutated_result.agent_results:
            if (agent.agent_name == "analysis" and agent.step_id == injection.target_step_id
                    and injection.injected_value is not None):
                injection.original_value = agent.result
                agent.result = injection.injected_value
                applied = True
                break
        if applied and injection.target_step_id == mutated_result.executed_steps[-1]:
            mutated_result.final_result = injection.injected_value

    return mutated_plan, mutated_result, applied
