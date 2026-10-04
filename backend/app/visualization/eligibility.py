"""Deterministic eligibility mapping from structured result rows to chart types."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

from app.multi_agent.schemas import AgentResult, MultiAgentResult
from app.planner.schemas import AnalysisPlan, PlanStep


class UnsupportedVisualization(ValueError):
    def __init__(self, code: str, message: str, *, category: str = "unsupported_visualization") -> None:
        super().__init__(message)
        self.code = code
        self.category = category


class UnsafeVisualizationData(ValueError):
    """A source value cannot be represented without silent coercion."""


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _json_safe(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_json_safe(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _json_safe(item) for key, item in value.items())
    return False


def _tabular(result: Any) -> tuple[list[str], list[dict[str, Any]]] | None:
    if isinstance(result, dict) and isinstance(result.get("columns"), list) and isinstance(result.get("rows"), list):
        columns, rows = result["columns"], result["rows"]
        if not all(isinstance(column, str) for column in columns):
            return None
        if not all(isinstance(row, dict) and set(row) >= set(columns) for row in rows):
            return None
        return columns, rows
    if isinstance(result, list) and result and all(isinstance(row, dict) for row in result):
        columns = list(result[0])
        if not all(set(row) == set(columns) for row in result):
            return None
        return columns, result
    if isinstance(result, dict) and result and all(_finite_number(value) for value in result.values()):
        return ["key", "value"], [{"key": key, "value": value} for key, value in result.items()]
    return None


def _columns(columns: list[str], rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    numeric: list[str] = []
    categorical: list[str] = []
    for column in columns:
        values = [row.get(column) for row in rows]
        if values and all(_finite_number(value) for value in values):
            numeric.append(column)
        elif values and all(isinstance(value, (str, bool)) and value is not None for value in values):
            categorical.append(column)
    return categorical, numeric


def _explicit_time(values: list[Any]) -> bool:
    if len(values) < 2:
        return False
    parsed: list[datetime] = []
    for value in values:
        if not isinstance(value, str):
            return False
        try:
            parsed.append(datetime.fromisoformat(value.replace("Z", "+00:00")))
        except ValueError:
            return False
    awareness = [value.tzinfo is not None for value in parsed]
    if any(awareness) and not all(awareness):
        return False
    if all(awareness):
        parsed = [value.astimezone(timezone.utc) for value in parsed]
    return all(left <= right for left, right in zip(parsed, parsed[1:]))


def _ordered_source(step: PlanStep, plan: AnalysisPlan) -> bool:
    if step.operation in {"sort", "rank"}:
        return True
    if step.operation not in {"top_n", "bottom_n"} or len(step.depends_on) != 1:
        return False
    by_id = {item.step_id: item for item in plan.steps}
    parent = by_id.get(step.depends_on[0])
    return parent is not None and _ordered_source(parent, plan)


def _pie_evidence(
    agent: AgentResult, result: dict[str, Any], category: str, measure: str
) -> tuple[bool, str | None, float | int | None]:
    """Require explicit evidence embedded in supplied artifacts; never infer it."""
    evidence = agent.metadata.get("part_to_whole")
    if not isinstance(evidence, dict):
        evidence = result.get("part_to_whole")
    if not isinstance(evidence, dict):
        return False, None, None
    whole = evidence.get("whole_value")
    whole_id = evidence.get("whole_id")
    relation = evidence.get("relationship")
    if not (
        isinstance(whole_id, str) and whole_id
        and relation == "components_of_whole"
        and _finite_number(whole) and float(whole) > 0
        and evidence.get("category_field") == category
        and evidence.get("value_field") == measure
    ):
        return False, None, None
    return True, whole_id, whole


def select_chart(
    plan: AnalysisPlan,
    execution: MultiAgentResult,
    verification_steps: list[str],
) -> dict[str, Any]:
    """Return an eligible chart description or a structured unsupported error.

    Deterministic precedence when more than one type is eligible: pie with
    explicit whole evidence, ordered line, scatter, then categorical bar.
    """
    plan_by_id = {step.step_id: step for step in plan.steps}
    agents = {
        agent.step_id: agent for agent in execution.agent_results
        if agent.agent_name == "analysis" and agent.step_id is not None and agent.status == "completed"
    }
    source_step = next(
        (plan_by_id[step_id] for step_id in reversed(execution.executed_steps)
         if step_id in plan_by_id and step_id in agents and step_id in verification_steps
         and agents[step_id].result == execution.final_result),
        None,
    )
    if source_step is None:
        raise UnsupportedVisualization("NO_VERIFIED_ANALYSIS_OUTPUT", "No verified analysis result is available to visualize.")
    agent = agents[source_step.step_id]
    tabular = _tabular(agent.result)
    if tabular is None:
        raise UnsupportedVisualization("UNSUPPORTED_RESULT_SHAPE", "The verified result is not a supported structured table.")
    fields, rows = tabular
    if len(rows) < 1:
        raise UnsupportedVisualization("INSUFFICIENT_DATA", "The verified result contains no observations.", category="insufficient_data")
    if not _json_safe(rows):
        raise UnsafeVisualizationData("The verified result contains values that cannot be represented safely.")
    if any(any(row.get(field) is None for field in fields) for row in rows):
        raise UnsupportedVisualization("MISSING_VALUES", "Missing values prevent safe visualization.")
    categories, numerics = _columns(fields, rows)
    if not categories and len(numerics) >= 2 and len(rows) >= 2:
        return {"chart_type": "scatter", "step": source_step, "agent": agent, "fields": fields,
                "rows": rows, "x_field": numerics[0], "y_field": numerics[1], "ordered": False}

    if categories and numerics:
        category, measure = categories[0], numerics[0]
        pie_ok, whole_id, whole_value = _pie_evidence(agent, agent.result, category, measure)
        if pie_ok and all(float(row[measure]) >= 0 for row in rows):
            try:
                components_total = sum((Decimal(str(row[measure])) for row in rows), Decimal(0))
                declared_whole = Decimal(str(whole_value))
            except InvalidOperation as exc:
                raise UnsafeVisualizationData("The explicit pie whole cannot be compared safely.") from exc
            if components_total != declared_whole:
                raise UnsafeVisualizationData("The explicit pie whole does not equal its declared components.")
            if len(rows) > 12:
                raise UnsupportedVisualization(
                    "PIE_SLICE_LIMIT", "The result has more than the documented 12-slice pie limit."
                )
            return {"chart_type": "pie", "step": source_step, "agent": agent, "fields": fields,
                    "rows": rows, "x_field": category, "y_field": measure, "ordered": False,
                    "whole_id": whole_id, "whole_value": whole_value}
        if len(rows) >= 2 and _explicit_time([row[category] for row in rows]):
            return {"chart_type": "line", "step": source_step, "agent": agent, "fields": fields,
                    "rows": rows, "x_field": category, "y_field": measure, "ordered": True,
                    "ordering": "explicit_time_values"}
        if len(rows) >= 2 and _ordered_source(source_step, plan):
            return {"chart_type": "line", "step": source_step, "agent": agent, "fields": fields,
                    "rows": rows, "x_field": category, "y_field": measure, "ordered": True,
                    "ordering": f"{source_step.operation}_result_order"}
        return {"chart_type": "bar", "step": source_step, "agent": agent, "fields": fields,
                "rows": rows, "x_field": category, "y_field": measure, "ordered": False}

    if len(numerics) >= 2 and len(rows) >= 2:
        return {"chart_type": "scatter", "step": source_step, "agent": agent, "fields": fields,
                "rows": rows, "x_field": numerics[0], "y_field": numerics[1], "ordered": False}
    raise UnsupportedVisualization("UNSUPPORTED_RESULT_SEMANTICS", "The result does not contain fields for a supported chart.")
