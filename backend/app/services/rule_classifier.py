"""Rule-based question classifier. It is the fallback when Gemini cannot be used.

It only understands clear, simple English patterns. It never guesses: when a question is unclear,
has two possible meanings, or needs a feature we do not have (filters, forecasts), the result
is intent "unsupported" with a reason.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic import ValidationError

from app.schemas.query_schema import QueryPlan
from app.utils.column_mapper import map_columns, normalize_column_name


@dataclass(frozen=True)
class RuleOutcome:
    plan: QueryPlan
    assumptions: list[str] = field(default_factory=list)


_DEFINITION_TERMS = {
    "revenue", "sales", "quantity", "unit price", "price", "average order value", "aov", "order value",
}
_DEFINITION_FULL = [re.compile(r"what does .+ mean"), re.compile(r"(?:what is the )?meaning of .+")]
_DEFINITION_TERM = re.compile(r"(?:what is|define|explain)(?: an?)? (?P<term>[a-z ]+)")

_UNSUPPORTED_WORDS = {
    "predict", "prediction", "forecast", "next", "why", "trend", "trends", "correlation", "correlate",
    "compare", "comparison", "growth", "percent", "percentage", "median", "mode", "variance",
    "deviation", "delete", "remove", "update", "modify", "insert", "drop", "chart", "plot", "graph",
    "email",
}
_FILTER_WORDS = {
    "where", "only", "during", "between", "after", "before", "since", "until", "excluding", "except",
    "without", "last", "this", "yesterday", "today",
}
_FILTER_PHRASE = re.compile(
    r"\b(?:in|for|from|of|at)\s+(?:the\s+)?(?!(?:the|a|an|each|every|by|all|any)\b)[a-z0-9]+\s+"
    r"(?:regions?|areas?|territor(?:y|ies)|products?|items?)\b"
)

_MISSING = re.compile(r"\b(?:missing|null|nulls|blank|blanks|nan)\b|\bempty (?:values|cells|fields)\b")
_AOV = re.compile(r"\b(?:average order value|aov|average order size)\b")

_METRIC_PATTERNS = {
    "unit_price": re.compile(r"\b(?:unit prices?|selling prices?|prices?)\b"),
    "revenue": re.compile(r"\b(?:revenues?|sales|income)\b"),
    "quantity": re.compile(r"\b(?:quantity|quantities|units|qty)\b"),
    "records": re.compile(r"\b(?:records?|rows?|entries)\b"),
}
_DEFAULT_AGGREGATION = {"revenue": "sum", "quantity": "sum", "unit_price": "average", "records": "count"}

_GROUP_WORDS = {
    "region": {"region", "regions", "area", "areas", "territory", "territories"},
    "product": {"product", "products", "item", "items"},
}
_SINGULAR = {"region", "area", "territory", "product", "item"}

_DESC = {"highest", "largest", "biggest", "top", "best", "most", "greatest", "maximum", "max"}
_ASC = {"lowest", "smallest", "bottom", "worst", "least", "fewest", "minimum", "min"}
_AVG = {"average", "avg", "mean"}
_SUM = {"total", "sum", "overall", "combined"}

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10,
}
_LIMIT = re.compile(r"\b(?:top|bottom|best|worst)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b")


def _normalise(question: str) -> str:
    text = question.lower().replace("'", "").replace("’", "")
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    text = re.sub(r"\bwhats\b", "what is", text)
    text = re.sub(r"\b(?:of each|of every|for each|for every|per|each|every)\b", "by", text)
    return re.sub(r"\s+", " ", text)


def _unsupported(reason: str) -> RuleOutcome:
    return RuleOutcome(QueryPlan(intent="unsupported", confidence=1.0, reasoning=reason))


def _is_definition(text: str) -> bool:
    if any(pattern.fullmatch(text) for pattern in _DEFINITION_FULL):
        return True
    match = _DEFINITION_TERM.fullmatch(text)
    return bool(match and match.group("term") in _DEFINITION_TERMS)


def _aggregation_cue(text: str, words: set[str]) -> str | None:
    if words & _AVG:
        return "average"
    if words & _SUM:
        return "sum"
    if "how many" in text or words & {"count", "number"}:
        return "count"
    return None


def _explicit_limit(text: str) -> int | None:
    match = _LIMIT.search(text)
    if not match:
        return None
    token = match.group(1)
    return int(token) if token.isdigit() else _NUMBER_WORDS[token]


def _group_candidates(text: str, words: set[str], columns: Sequence[str]) -> dict[str, bool]:
    """Return {group_by: is_singular}. Roles come from fixed words, other groups from real column names."""
    found: dict[str, bool] = {}
    for role, forms in _GROUP_WORDS.items():
        matched = words & forms
        if matched:
            found[role] = any(word in _SINGULAR for word in matched)
    mapping = map_columns(list(columns))
    reserved = {
        mapping.get(role)
        for role in ("revenue", "quantity", "unit_price", "order_id", "region", "product")
    } - {None}
    for column in columns:
        if column in reserved:
            continue
        name = normalize_column_name(column).replace("_", " ")
        if not name:
            continue
        if re.search(rf"\b{re.escape(name)}\b", text):
            found[column] = True
        elif re.search(rf"\b{re.escape(name)}s\b", text):
            found[column] = False
    return found


def _classify(question: str, columns: Sequence[str]) -> RuleOutcome:
    text = _normalise(question)
    words = set(text.split())
    if not words:
        return _unsupported("The question has no readable English words.")
    if _is_definition(text):
        return RuleOutcome(
            QueryPlan(intent="definition", confidence=1.0, reasoning="Asks what a business term means.")
        )
    if words & _UNSUPPORTED_WORDS:
        return _unsupported("Forecasts, comparisons, extra statistics and changing data are not supported.")
    if words & _FILTER_WORDS or _FILTER_PHRASE.search(text):
        return _unsupported("Filtering (for example by a date or one region) is not supported yet.")

    desc, asc = words & _DESC, words & _ASC
    if desc and asc:
        return _unsupported("The question asks for both the highest and the lowest.")
    direction = "desc" if desc else "asc" if asc else None
    grouping_cue = bool({"by", "which"} & words) or direction is not None

    if _MISSING.search(text):
        return RuleOutcome(
            QueryPlan(intent="missing_values", tool_name="missing_values", confidence=1.0,
                      reasoning="Asks about missing values.")
        )
    if _AOV.search(text):
        if grouping_cue:
            return _unsupported("The average order value can only be calculated for the whole dataset.")
        return RuleOutcome(
            QueryPlan(intent="aggregate", tool_name="aggregation", metric="average_order_value",
                      aggregation="average", confidence=0.9, reasoning="Asks for the average order value.")
        )

    metrics = {name for name, pattern in _METRIC_PATTERNS.items() if pattern.search(text)}
    if len(metrics) > 1:
        return _unsupported("The question mentions more than one measure. Please ask about one at a time.")
    if not metrics:
        return _unsupported("No supported measure (revenue, quantity, unit price or records) was found.")
    metric = metrics.pop()

    groups = _group_candidates(text, words, columns) if grouping_cue else {}
    if len(groups) > 1:
        return _unsupported("Grouping by more than one column is not supported.")

    aggregation = _aggregation_cue(text, words)
    assumptions: list[str] = []

    if groups:
        [(group_by, singular)] = groups.items()
        if metric == "records":
            aggregation = "count"
        elif aggregation is None:
            aggregation = _DEFAULT_AGGREGATION[metric]
            if direction is None:
                assumptions.append(f"No aggregation was stated, so the {aggregation} was used.")
        if direction is None:
            plan = QueryPlan(intent="group", tool_name="grouping", metric=metric, aggregation=aggregation,
                             group_by=group_by, confidence=0.8, reasoning=f"Groups {metric} by {group_by}.")
        else:
            explicit = _explicit_limit(text)
            limit = explicit if explicit is not None else (1 if singular else 5)
            assumptions.append(f"Groups are ranked by the {aggregation} of {metric} for each {group_by}.")
            plan = QueryPlan(intent="rank", tool_name="ranking", metric=metric, aggregation=aggregation,
                             group_by=group_by, sort_order=direction, limit=limit, confidence=0.8,
                             reasoning=f"Ranks {group_by} by {metric}.")
        return RuleOutcome(plan, assumptions)

    if direction is not None:
        if metric == "records":
            return _unsupported("Highest or lowest needs a measure such as revenue, quantity or unit price.")
        return RuleOutcome(
            QueryPlan(intent="aggregate", tool_name="aggregation", metric=metric,
                      aggregation="max" if direction == "desc" else "min", confidence=0.8,
                      reasoning=f"Asks for the {'maximum' if direction == 'desc' else 'minimum'} {metric}.")
        )
    if metric == "records":
        if aggregation != "count":
            return _unsupported("To count records, ask for example: How many records are there?")
        return RuleOutcome(
            QueryPlan(intent="aggregate", tool_name="aggregation", metric="records", aggregation="count",
                      confidence=0.9, reasoning="Asks how many records there are.")
        )
    if aggregation is None:
        aggregation = _DEFAULT_AGGREGATION[metric]
        assumptions.append(f"No aggregation was stated, so the {aggregation} was used.")
    if metric == "revenue" and aggregation == "sum":
        plan = QueryPlan(intent="total_revenue", tool_name="revenue", confidence=0.9,
                         reasoning="Asks for the total revenue.")
    else:
        plan = QueryPlan(intent="aggregate", tool_name="aggregation", metric=metric, aggregation=aggregation,
                         confidence=0.85, reasoning=f"Asks for the {aggregation} of {metric}.")
    return RuleOutcome(plan, assumptions)


def classify_with_rules(question: str, columns: Sequence[str]) -> RuleOutcome:
    """Classify a question with fixed rules. Never raises: unclear input becomes "unsupported"."""
    try:
        return _classify(question, columns)
    except ValidationError:
        return _unsupported("The question could not be turned into a valid query plan.")