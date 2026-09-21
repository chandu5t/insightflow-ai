"""Template explanations. Every sentence is built from validated tool results, never from an LLM."""

from app.schemas.profile_schema import MissingValueReport
from app.schemas.query_schema import QueryPlan
from app.schemas.tool_schema import AggregationResult, GroupingResult, RankingResult

_MONEY_METRICS = {"revenue", "unit_price", "average_order_value"}
_LABELS = {
    "revenue": "revenue", "quantity": "quantity", "unit_price": "unit price",
    "records": "records", "average_order_value": "average order value",
}
_WORDS = {"sum": "total", "average": "average", "min": "minimum", "max": "maximum", "count": "number of"}
_RESULT_WORDS = {"sum": "total", "average": "average", "minimum": "minimum", "maximum": "maximum", "count": "number of"}


def uses_money(plan: QueryPlan) -> bool:
    """True when the answer is an amount of money (so the currency note applies)."""
    return plan.intent == "total_revenue" or plan.metric in _MONEY_METRICS


def _money(value: float, symbol: str) -> str:
    return f"{symbol}{value:,.2f}"


def _number(value: float) -> str:
    return f"{int(value):,}" if float(value).is_integer() else f"{value:,.2f}"


def _format(plan: QueryPlan, value: float, symbol: str) -> str:
    return _money(value, symbol) if plan.metric in _MONEY_METRICS else _number(value)


def _label(plan: QueryPlan) -> str:
    return _LABELS.get(plan.metric or "", "value")


def _method_text(result: AggregationResult) -> str:
    columns = result.columns_used
    if result.calculation_method == "direct_revenue_column":
        return f"taken from the '{columns[0]}' column"
    if result.calculation_method == "derived_quantity_times_unit_price":
        return f"calculated as quantity multiplied by unit price (columns '{columns[0]}' and '{columns[1]}')"
    return ""


def _explain_aggregation(plan: QueryPlan, result: AggregationResult, symbol: str) -> str:
    if result.metric == "total_revenue":
        return f"The total revenue is {_money(result.value, symbol)}, {_method_text(result)}."
    if result.metric == "average_order_value":
        how = ("distinct orders" if result.order_count_method == "distinct_order_id_count"
               else "orders (each row was counted as one order)")
        return (f"The average order value is {_money(result.value, symbol)}: total revenue of "
                f"{_money(result.total_revenue or 0.0, symbol)} divided by {result.order_count} {how}.")
    if result.metric == "count":
        if result.calculation_method == "row_count":
            return f"The dataset has {_number(result.value)} records."
        return f"There are {_number(result.value)} non-empty {_label(plan)} values."
    word = _RESULT_WORDS[result.metric]
    label = _label(plan) + (" per row" if plan.metric == "revenue" else "")
    text = f"The {word} {label} is {_format(plan, result.value, symbol)}"
    method = _method_text(result) if plan.metric == "revenue" else ""
    return f"{text}, {method}." if method else f"{text}."


def _explain_grouping(plan: QueryPlan, result: GroupingResult, symbol: str) -> str:
    word = _WORDS.get(result.metric, result.metric)
    text = f"Calculated the {word} {_label(plan)} for each {result.group_by} ({result.group_count} groups)."
    named = [g for g in result.groups if g.group is not None and g.value is not None]
    if not named:
        return text + " No group had a valid value."
    top = max(named, key=lambda g: g.value)  # type: ignore[arg-type,return-value]
    low = min(named, key=lambda g: g.value)  # type: ignore[arg-type,return-value]
    return (f"{text} The highest is {top.group} ({_format(plan, top.value, symbol)}); "  # type: ignore[arg-type]
            f"the lowest is {low.group} ({_format(plan, low.value, symbol)}).")  # type: ignore[arg-type]


def _explain_ranking(plan: QueryPlan, result: RankingResult, symbol: str) -> str:
    if not result.items:
        return "No groups could be ranked because none of them had a valid value."
    direction = "highest" if result.order == "desc" else "lowest"
    measure = f"{_WORDS.get(result.metric, result.metric)} {_label(plan)}"
    if result.n_returned == 1:
        top = result.items[0]
        text = (f"The {result.group_by} with the {direction} {measure} is {top.group}, "
                f"with {measure} of {_format(plan, top.value, symbol)}.")
    else:
        lines = ", ".join(f"{i.rank}. {i.group} ({_format(plan, i.value, symbol)})" for i in result.items)
        edge = "top" if result.order == "desc" else "bottom"
        text = f"The {edge} {result.n_returned} {result.group_by} values by {measure} are: {lines}."
    if result.truncated_tie:
        text += f" Note: at least one other {result.group_by} has the same value as the last one shown."
    return text


def _explain_missing(report: MissingValueReport) -> str:
    if report.total_missing_cells == 0:
        return "No missing values were found in the dataset."
    columns = ", ".join(f"{c.name} ({c.missing_count})" for c in report.columns if c.missing_count > 0)
    return (f"The dataset has {report.total_missing_cells} missing cells out of {report.total_cells} "
            f"({report.missing_percentage}%). Columns with missing values: {columns}.")


def explain(plan: QueryPlan, result: object, symbol: str) -> str:
    """Return a plain-English explanation of a validated result. Raises ValueError for unknown results."""
    if isinstance(result, AggregationResult):
        return _explain_aggregation(plan, result, symbol)
    if isinstance(result, GroupingResult):
        return _explain_grouping(plan, result, symbol)
    if isinstance(result, RankingResult):
        return _explain_ranking(plan, result, symbol)
    if isinstance(result, MissingValueReport):
        return _explain_missing(result)
    raise ValueError("There is no explanation template for this result type.")