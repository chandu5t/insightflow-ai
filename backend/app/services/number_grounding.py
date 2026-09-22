"""Number-grounding guard: every number in an explanation must be supported by the validated result.

See docs/DECISIONS.md (D-053) for the rules and the limitations.
"""

import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from pydantic import BaseModel

from app.schemas.query_schema import QueryPlan

_NUMBER = re.compile(
    r"(?<!\w)"
    r"(?P<sign>[-+])?"
    r"(?P<int>\d{1,3}(?:,\d{2,3})+|\d+)"
    r"(?:\.(?P<frac>\d+))?"
    r"(?:[eE](?P<exp>[-+]?\d+))?"
    r"(?P<percent>\s?(?:%|percent\b))?"
)


@dataclass(frozen=True)
class ExtractedNumber:
    text: str
    value: Decimal | None  # None when the number cannot be represented (it is then never supported)
    tolerance: Decimal
    is_percent: bool


@dataclass(frozen=True)
class GroundingContext:
    allowed: tuple[Decimal, ...]
    allowed_percentages: tuple[Decimal, ...]
    literals: tuple[str, ...]  # data names that contain digits, removed before checking


@dataclass(frozen=True)
class GroundingResult:
    passed: bool
    unsupported: tuple[str, ...]
    checked: int


def extract_numbers(text: str) -> list[ExtractedNumber]:
    """Find every number in the text. Currency symbols and units are ignored."""
    cleaned = text.replace("\u2212", "-")
    found: list[ExtractedNumber] = []
    for match in _NUMBER.finditer(cleaned):
        frac = match.group("frac") or ""
        exponent = int(match.group("exp") or 0)
        digits = match.group("int").replace(",", "")
        try:
            value: Decimal | None = Decimal(f"{digits}.{frac}" if frac else digits).scaleb(exponent)
            if match.group("sign") == "-":
                value = -value
            tolerance = Decimal("0.5").scaleb(exponent - len(frac))
        except (InvalidOperation, OverflowError, ValueError):
            value, tolerance = None, Decimal(0)
        found.append(
            ExtractedNumber(
                text=match.group(0).strip(),
                value=value,
                tolerance=tolerance,
                is_percent=match.group("percent") is not None,
            )
        )
    return found


def _as_decimal(text: str) -> Decimal | None:
    try:
        return Decimal(text.replace(",", "").strip())
    except InvalidOperation:
        return None


def build_grounding_context(plan: QueryPlan, result: BaseModel) -> GroundingContext:
    """Collect the numbers (and the digit-containing names) of a validated result."""
    numbers: list[Decimal] = []
    percentages: list[Decimal] = []
    literals: set[str] = set()

    def walk(node: object, key: str = "") -> None:
        if isinstance(node, dict):
            for child_key, child in node.items():
                walk(child, str(child_key))
        elif isinstance(node, (list, tuple)):
            for child in node:
                walk(child, key)
        elif isinstance(node, bool) or node is None:
            return
        elif isinstance(node, (int, float)):
            if key in {"rows_used", "rows_excluded_missing"}:
                return

            if isinstance(node, float) and not math.isfinite(node):
                return

            number = Decimal(str(node))
            numbers.append(number)

            if "percentage" in key:
                percentages.append(number)       
        elif isinstance(node, str):
            as_number = _as_decimal(node)
            if as_number is not None:
                numbers.append(as_number)  # a numeric label such as a group named "5"
            elif any(ch.isdigit() for ch in node):
                literals.add(node)

    walk(result.model_dump())
    if plan.limit is not None:
        numbers.append(Decimal(plan.limit))
    ordered = tuple(sorted(literals, key=len, reverse=True))  # longest names first
    return GroundingContext(tuple(numbers), tuple(percentages), ordered)


def _supported(number: ExtractedNumber, pool: tuple[Decimal, ...]) -> bool:
    if number.value is None:
        return False
    for allowed in pool:
        try:
            if abs(allowed - number.value) < number.tolerance:
                return True
        except InvalidOperation:
            continue
    return False


def check_grounding(explanation: str, context: GroundingContext) -> GroundingResult:
    """Return which numbers of the explanation are NOT supported by the result."""
    text = explanation
    for literal in context.literals:
        text = text.replace(literal, " ")
    unsupported: list[str] = []
    checked = 0
    for number in extract_numbers(text):
        checked += 1
        pool = context.allowed_percentages if number.is_percent else context.allowed
        if not _supported(number, pool):
            unsupported.append(number.text)
    return GroundingResult(passed=not unsupported, unsupported=tuple(unsupported), checked=checked)