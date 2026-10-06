"""Strict, deterministic date parsing shared by planning validation and execution."""

from __future__ import annotations

from datetime import date
import re
from typing import Any


_DATE_PATTERN = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}\Z")
_ORDERED_OPERATORS = {"lt", "lte", "gt", "gte"}


class InvalidDateValue(ValueError):
    """A date literal or non-missing source value violates strict YYYY-MM-DD."""


def parse_strict_date(value: Any) -> date:
    """Parse a canonical calendar date without trimming, coercion, or inference."""
    if not isinstance(value, str) or _DATE_PATTERN.fullmatch(value) is None:
        raise InvalidDateValue("Date values must use exact YYYY-MM-DD format.")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise InvalidDateValue("Date value is not a valid calendar date.") from exc
    if parsed.isoformat() != value:
        raise InvalidDateValue("Date values must use exact YYYY-MM-DD format.")
    return parsed


def date_predicate_errors(
    conditions: list[dict[str, Any]], semantic_roles: dict[str, str] | None,
) -> list[tuple[str, str]]:
    """Return deterministic date-role, literal, and reversed-range errors."""
    errors: list[tuple[str, str]] = []
    if semantic_roles is None:
        return errors

    by_date_column: dict[str, list[tuple[str, date]]] = {}
    for condition in conditions:
        if not isinstance(condition, dict):
            continue
        column = condition.get("column")
        operator = condition.get("operator")
        if not isinstance(column, str):
            continue
        role = semantic_roles.get(column)
        if role != "date":
            value = condition.get("value")
            date_text = isinstance(value, str) or (
                isinstance(value, list) and any(isinstance(item, str) for item in value)
            )
            if operator in _ORDERED_OPERATORS and date_text and role not in {
                "additive_measure", "non_additive_measure"
            }:
                errors.append((
                    "UNTRUSTED_DATE_COLUMN",
                    "Ordered predicates require a column with trusted semantic role 'date'.",
                ))
            continue

        if operator in {"is_missing", "is_not_missing"}:
            continue
        value = condition.get("value")
        values = value if operator in {"in", "not_in"} and isinstance(value, list) else [value]
        parsed_values: list[date] = []
        try:
            parsed_values = [parse_strict_date(item) for item in values]
        except InvalidDateValue:
            errors.append((
                "INVALID_DATE_LITERAL",
                "Date literals must be valid calendar dates in exact YYYY-MM-DD format.",
            ))
            continue
        if operator in _ORDERED_OPERATORS and len(parsed_values) == 1:
            by_date_column.setdefault(column, []).append((operator, parsed_values[0]))

    for bounds in by_date_column.values():
        lower = [value for operator, value in bounds if operator in {"gt", "gte"}]
        upper = [value for operator, value in bounds if operator in {"lt", "lte"}]
        if lower and upper and max(lower) > min(upper):
            errors.append((
                "INVALID_DATE_RANGE",
                "The lower date bound must not be later than the upper date bound.",
            ))
    return errors
