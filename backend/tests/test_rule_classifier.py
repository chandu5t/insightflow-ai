"""Tests for the rule-based classifier."""

import pytest

from app.services.rule_classifier import classify_with_rules

COLUMNS = ["order_id", "order_date", "product", "category", "quantity", "unit_price", "region"]


def plan_of(question: str, columns=COLUMNS):
    return classify_with_rules(question, columns).plan


def fields(plan) -> tuple:
    return (plan.intent, plan.tool_name, plan.metric, plan.aggregation, plan.group_by, plan.sort_order, plan.limit)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is the total revenue?", ("total_revenue", "revenue", None, None, None, None, None)),
        ("Which region generated the highest revenue?", ("rank", "ranking", "revenue", "sum", "region", "desc", 1)),
        ("Which region generated the lowest revenue?", ("rank", "ranking", "revenue", "sum", "region", "asc", 1)),
        ("Which products have the highest revenue?", ("rank", "ranking", "revenue", "sum", "product", "desc", 5)),
        ("Top 3 products by revenue", ("rank", "ranking", "revenue", "sum", "product", "desc", 3)),
        ("Which product sold the most units?", ("rank", "ranking", "quantity", "sum", "product", "desc", 1)),
        ("Show revenue by region.", ("group", "grouping", "revenue", "sum", "region", None, None)),
        ("Quantity per product", ("group", "grouping", "quantity", "sum", "product", None, None)),
        ("Revenue by category", ("group", "grouping", "revenue", "sum", "category", None, None)),
        ("What is the average unit price?", ("aggregate", "aggregation", "unit_price", "average", None, None, None)),
        ("What is the maximum quantity?", ("aggregate", "aggregation", "quantity", "max", None, None, None)),
        ("What is the minimum unit price?", ("aggregate", "aggregation", "unit_price", "min", None, None, None)),
        ("How many records are present?", ("aggregate", "aggregation", "records", "count", None, None, None)),
        ("What is the average order value?", ("aggregate", "aggregation", "average_order_value", "average", None, None, None)),
        ("Are there missing values?", ("missing_values", "missing_values", None, None, None, None, None)),
        ("WHAT'S THE TOTAL REVENUE???", ("total_revenue", "revenue", None, None, None, None, None)),
    ],
)
def test_supported_questions(question: str, expected: tuple) -> None:
    assert fields(plan_of(question)) == expected


@pytest.mark.parametrize(
    "question",
    ["What is revenue?", "What does average order value mean?", "Define revenue", "what is unit price"],
)
def test_definition_questions(question: str) -> None:
    plan = plan_of(question)

    assert plan.intent == "definition" and plan.tool_name is None


def test_a_calculation_question_is_not_a_definition() -> None:
    assert plan_of("What is the total revenue?").intent == "total_revenue"
    assert plan_of("What is the average order value?").intent == "aggregate"


@pytest.mark.parametrize(
    "question",
    [
        "Predict next month's revenue",              # forecast
        "Why did sales drop?",
        "What is the median unit price?",
        "Revenue in the north region",               # filter
        "Total revenue after january",               # filter
        "Revenue and quantity by region",            # two measures
        "Revenue by region and product",             # two group columns
        "Average order value by region",             # cannot be grouped
        "Which region is the best?",                 # no measure
        "How many products?",                        # distinct counts are not supported
        "records",                                   # no count word
        "Highest and lowest revenue region",         # contradictory
        "hello",
        "",
        "   ",
        "🙂",
        "'; DROP TABLE sales; --",
        "x" * 1000,
    ],
)
def test_unsupported_or_unclear_questions_never_guess(question: str) -> None:
    plan = plan_of(question)

    assert plan.intent == "unsupported" and plan.tool_name is None and plan.reasoning


def test_missing_columns_are_not_the_classifiers_job() -> None:
    plan = plan_of("Which region generated the highest revenue?", ["qty", "price"])

    assert plan.intent == "rank" and plan.group_by == "region"


def test_assumptions_are_reported_when_the_aggregation_is_not_stated() -> None:
    outcome = classify_with_rules("Show revenue by region", COLUMNS)

    assert any("No aggregation was stated" in note for note in outcome.assumptions)
    assert classify_with_rules("What is the total revenue?", COLUMNS).assumptions == []


def test_ranking_assumption_explains_how_groups_are_ranked() -> None:
    outcome = classify_with_rules("Which region generated the highest revenue?", COLUMNS)

    assert any("ranked by the sum of revenue" in note for note in outcome.assumptions)


def test_odd_input_never_raises() -> None:
    for text in ["", "\n\t", "a" * 5000, "SELECT * FROM x", "1 2 3", "%%%", "top 0 products by revenue"]:
        assert classify_with_rules(text, COLUMNS).plan.intent in {"unsupported", "group", "rank", "aggregate", "total_revenue", "missing_values", "definition"}