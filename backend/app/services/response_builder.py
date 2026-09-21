"""Builds the API response from the final workflow state. The response format is unchanged."""

from typing import Any

from app.core.errors import ErrorCode
from app.schemas.query_schema import QueryErrorInfo, QueryResponse
from app.schemas.tool_schema import AggregationResult, GroupingResult, RankingResult
from app.services.explainer import uses_money
from app.workflow.state import WorkflowState

SUPPORTED_HELP = (
    "You can ask for: the total revenue, the average order value, the sum, average, minimum, maximum or "
    "count of quantity or unit price, the number of records, revenue or quantity by a column such as "
    "region or product, the top or bottom groups, and missing values."
)
NO_FILTERS_NOTE = "No filters were applied. The answer covers every row of the dataset."
_TOOL_NOTES = (AggregationResult, GroupingResult, RankingResult)


def error_info(code: ErrorCode, message: str, details: dict | None = None) -> QueryErrorInfo:
    return QueryErrorInfo(code=code.value, message=message, details=details or {})


def calculation_method_of(result: object) -> str:
    if isinstance(result, AggregationResult):
        if result.order_count_method:
            return f"{result.calculation_method}; {result.order_count_method}"
        return result.calculation_method
    if isinstance(result, (GroupingResult, RankingResult)):
        return result.calculation_method
    return "empty_cell_count"


def unsupported_message(reasoning: str) -> str:
    return f"I can't answer that yet. {reasoning} {SUPPORTED_HELP}".replace("  ", " ")


def build_response(state: WorkflowState, *, currency_symbol: str) -> QueryResponse:
    """Turn the final state into the same QueryResponse that Module 4 returned."""
    plan = state["query_plan"]
    common: dict[str, Any] = {
        "status": state["status"],
        "question": state["question"],
        "dataset_id": state["dataset_id"],
        "classifier": state["classifier"],
        "query_plan": plan,
    }

    definition = state.get("definition_result")
    if state["status"] == "success" and definition is not None:
        return QueryResponse(
            **common,
            explanation=definition.definition,
            calculation_method="metric_definition",
            assumptions=[f"This definition comes from the {definition.source} source. No calculation was performed."],
        )

    if state["status"] == "success":
        result = state["validated_result"]
        notes = result.notes if isinstance(result, _TOOL_NOTES) else []
        assumptions = [NO_FILTERS_NOTE, *state.get("assumptions", []), *notes]
        if uses_money(plan):
            assumptions.append(
                f"Amounts are shown with the symbol '{currency_symbol}'. The data file does not contain a currency."
            )
        return QueryResponse(
            **common,
            tool_used=result.tool,
            result=result,
            explanation=state["explanation"],
            calculation_method=calculation_method_of(result),
            assumptions=assumptions,
            validation=state["validation"],
        )

    failure: dict[str, Any] = {"message": state.get("message"), "error": state.get("error")}
    if state.get("validation") is not None:
        failure["validation"] = state["validation"]
    return QueryResponse(**common, **failure)