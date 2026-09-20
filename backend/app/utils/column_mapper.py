"""Explicit column mapping: which real column plays which business role.

Only names written in CANONICAL_ALIASES are recognised. Nothing is guessed.
To support a new name, add it to the tuple of the right role. That is all.

Names are compared after normalising: capital/small letters, spaces, '-' and '_' are ignored
("Unit Price", "unit-price" and "UNIT_PRICE" all become "unit_price").
If two real columns match the same role, the result is AMBIGUOUS. We never pick one silently.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from app.core.errors import ErrorCode, ToolError

CANONICAL_ALIASES: dict[str, tuple[str, ...]] = {
    "revenue": ("revenue", "total_revenue", "sales", "total_sales"),
    "quantity": ("quantity", "qty", "units", "units_sold"),
    "unit_price": ("unit_price", "price", "selling_price"),
    "product": ("product", "product_name", "item", "item_name"),
    "order_id": ("order_id", "orderid", "order_number", "order_no"),
    "region": ("region", "area", "territory"),
}

_SEPARATORS = re.compile(r"[\W_]+")


def normalize_column_name(name: str) -> str:
    """'  Unit Price ' -> 'unit_price'. The original name is never changed, only compared."""
    return _SEPARATORS.sub("_", name.strip().casefold()).strip("_")


def validate_aliases(aliases: Mapping[str, Sequence[str]]) -> None:
    """Developer check: a name must not belong to two roles. Raises ValueError if it does."""
    owner: dict[str, str] = {}
    for role, names in aliases.items():
        for alias in (role, *names):
            key = normalize_column_name(alias)
            if not key:
                raise ValueError(f"Empty alias for role '{role}'")
            if key in owner and owner[key] != role:
                raise ValueError(f"'{alias}' is used for both '{owner[key]}' and '{role}'")
            owner[key] = role


@dataclass(frozen=True)
class ColumnMapping:
    """Result of mapping. `resolved` maps role -> original column name."""

    resolved: dict[str, str] = field(default_factory=dict)
    ambiguous: dict[str, list[str]] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    accepted_names: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(self.accepted_names)

    def get(self, role: str) -> str | None:
        """The original column name for a role, or None (missing or ambiguous)."""
        return self.resolved.get(role)

    def require_if_ambiguous(self, role: str) -> None:
        """Raise AMBIGUOUS_COLUMN when the role matches more than one column. Otherwise do nothing."""
        if role in self.ambiguous:
            self.require(role)

    def require(self, *roles: str) -> dict[str, str]:
        """Return role -> column for all roles, or raise a clear ToolError."""
        unknown = [role for role in roles if role not in self.accepted_names]
        if unknown:
            raise ToolError(
                ErrorCode.INVALID_PARAMETER,
                f"Unknown column role: {', '.join(unknown)}.",
                details={"known_roles": sorted(self.accepted_names)},
            )
        for role in roles:
            if role in self.ambiguous:
                raise ToolError(
                    ErrorCode.AMBIGUOUS_COLUMN,
                    f"More than one column could be '{role}': {', '.join(self.ambiguous[role])}. "
                    "Please choose one.",
                    details={"role": role, "candidates": self.ambiguous[role]},
                )
        missing = [role for role in roles if role not in self.resolved]
        if missing:
            raise ToolError(
                ErrorCode.MISSING_COLUMN,
                f"The dataset has no column for: {', '.join(missing)}.",
                details={
                    "missing_roles": missing,
                    "accepted_names": {role: list(self.accepted_names[role]) for role in missing},
                },
            )
        return {role: self.resolved[role] for role in roles}


def map_columns(
    columns: Sequence[str],
    aliases: Mapping[str, Sequence[str]] = CANONICAL_ALIASES,
    overrides: Mapping[str, str] | None = None,
) -> ColumnMapping:
    """Find which real column plays each role.

    `overrides` lets the caller say exactly which column to use for a role
    (for example {"revenue": "sales"}). It is the way to solve an ambiguous result.
    """
    validate_aliases(aliases)
    overrides = overrides or {}

    for role, column in overrides.items():
        if role not in aliases:
            raise ToolError(
                ErrorCode.INVALID_PARAMETER,
                f"Unknown column role '{role}'.",
                details={"known_roles": sorted(aliases)},
            )
        if column not in columns:
            raise ToolError(
                ErrorCode.MISSING_COLUMN,
                f"Column '{column}' does not exist in the dataset.",
                details={"column": column, "available_columns": list(columns)},
            )

    alias_to_role = {
        normalize_column_name(alias): role for role, names in aliases.items() for alias in (role, *names)
    }
    candidates: dict[str, list[str]] = {role: [] for role in aliases}
    for column in columns:
        role = alias_to_role.get(normalize_column_name(column))
        if role is not None:
            candidates[role].append(column)

    resolved: dict[str, str] = {}
    ambiguous: dict[str, list[str]] = {}
    missing: list[str] = []
    for role in aliases:
        if role in overrides:
            resolved[role] = overrides[role]
        elif len(candidates[role]) == 1:
            resolved[role] = candidates[role][0]
        elif len(candidates[role]) > 1:
            ambiguous[role] = candidates[role]
        else:
            missing.append(role)

    return ColumnMapping(
        resolved=resolved,
        ambiguous=ambiguous,
        missing=missing,
        accepted_names={role: tuple(names) for role, names in aliases.items()},
    )


def resolve_column(columns: Sequence[str], requested: str, mapping: ColumnMapping) -> str:
    """Turn a request into a real column name.

    - A role name (revenue, quantity, unit_price, product, order_id, region) always goes
      through the mapper, so ambiguity is always detected.
    - Any other text must be the exact name of a real column.
    """
    if requested in mapping.roles:
        return mapping.require(requested)[requested]
    if requested in columns:
        return requested
    raise ToolError(
        ErrorCode.MISSING_COLUMN,
        f"Column '{requested}' does not exist in the dataset.",
        details={"column": requested, "available_columns": list(columns)},
    )