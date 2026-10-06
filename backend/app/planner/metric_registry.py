"""Application-owned metric definitions bound to trusted dataset snapshots."""

from __future__ import annotations

import math
import re

from app.planner.schemas import MetricDefinitionContext
from app.planner.semantic_manifest import (
    ALLOWED_ROLES,
    DatasetSemanticManifest,
    semantic_manifest_for_snapshot,
)

METRIC_REGISTRY_VERSION = "insightflow-metric-registry-v1"


class MetricRegistryError(ValueError):
    """Raised when a registered definition does not match its trusted snapshot."""


# Registry content is application-owned. It contains no benchmark identifiers,
# questions, answers, or observed evaluation results.
_REGISTRY: dict[tuple[str, str], MetricDefinitionContext] = {
    (
        "eae24ee4e675e8b99d6f0893a87c4b84717a0e2eb876a7f19654f88e46837d54",
        "revenue",
    ): MetricDefinitionContext(
        display_name="Revenue",
        metric_key="revenue",
        definition_version="revenue-v1",
        dataset_sha256="eae24ee4e675e8b99d6f0893a87c4b84717a0e2eb876a7f19654f88e46837d54",
        manifest_version="insightflow-bench-v1.0-semantic-v1",
        exact_column_references=["quantity", "unit_price"],
        allowed_expression={
            "op": "multiply",
            "left": {"column": "quantity"},
            "right": {"column": "unit_price"},
        },
        output_semantic_role="additive_measure",
        source_reference="backend/app/tools/revenue.py#Rules-1-2;sha256:f9bfdeea5fbd4ddd15961a53ecf76c929fd9688c7903c026bf4b2cc772fb5f63",
    ),
    (
        "520172097d69e40cf729714342d8d6ce7a7763670b5d13f6d673aaf1af0359cd",
        "revenue",
    ): MetricDefinitionContext(
        display_name="Revenue",
        metric_key="revenue",
        definition_version="revenue-v1",
        dataset_sha256="520172097d69e40cf729714342d8d6ce7a7763670b5d13f6d673aaf1af0359cd",
        manifest_version="insightflow-bench-v1.0-semantic-v1",
        exact_column_references=["Total Revenue"],
        allowed_expression={"column": "Total Revenue"},
        output_semantic_role="additive_measure",
        source_reference="backend/app/tools/revenue.py#Rules-1-2;sha256:f9bfdeea5fbd4ddd15961a53ecf76c929fd9688c7903c026bf4b2cc772fb5f63",
    ),
}
REGISTERED_METRIC_KEYS = frozenset(metric_key for _, metric_key in _REGISTRY)


def _expression_columns(expression: object) -> list[str] | None:
    """Validate the restricted V2.8.1 expression shape and return column refs."""
    if not isinstance(expression, dict):
        return None
    if set(expression) == {"column"}:
        column = expression["column"]
        return [column] if isinstance(column, str) and column else None
    if set(expression) == {"literal"}:
        value = expression["literal"]
        return [] if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) else None
    if set(expression) == {"op", "value"} and expression["op"] == "negate":
        return _expression_columns(expression["value"])
    if set(expression) == {"op", "left", "right"} and expression["op"] in {"add", "subtract", "multiply", "divide"}:
        left = _expression_columns(expression["left"])
        right = _expression_columns(expression["right"])
        return None if left is None or right is None else left + right
    return None


def validate_metric_definition(
    definition: MetricDefinitionContext,
    *,
    dataset_sha256: str,
    manifest: DatasetSemanticManifest,
    physical_columns: list[str] | tuple[str, ...],
) -> bool:
    """Check every registry binding without resolving names or aliases."""
    if not isinstance(dataset_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", dataset_sha256):
        return False
    if definition.dataset_sha256 != dataset_sha256 or manifest.dataset_sha256 != dataset_sha256:
        return False
    canonical = _REGISTRY.get((dataset_sha256, definition.metric_key or ""))
    if canonical is None or definition.model_dump(mode="json") != canonical.model_dump(mode="json"):
        return False
    if definition.manifest_version != manifest.manifest_version:
        return False
    if not definition.metric_key or not definition.definition_version or not definition.source_reference:
        return False
    if definition.output_semantic_role not in ALLOWED_ROLES:
        return False
    references = definition.exact_column_references
    if not references or len(references) != len(set(references)):
        return False
    if any(reference not in physical_columns or reference not in manifest.columns for reference in references):
        return False
    expression_columns = _expression_columns(definition.allowed_expression)
    if expression_columns is None or expression_columns != references:
        return False
    if definition.output_semantic_role == "additive_measure" and any(
        manifest.role_for(reference) not in {"additive_measure", "non_additive_measure"}
        for reference in references
    ):
        return False
    return True


def resolve_metric_definitions(
    *,
    dataset_sha256: str,
    manifest: DatasetSemanticManifest | None,
    physical_columns: list[str] | tuple[str, ...],
) -> list[MetricDefinitionContext]:
    """Return validated definitions for a known snapshot, or none for unknown data."""
    if manifest is None:
        return []
    definitions = [
        definition.model_copy(deep=True)
        for (registered_hash, _), definition in _REGISTRY.items()
        if registered_hash == dataset_sha256
    ]
    for definition in definitions:
        if not validate_metric_definition(
            definition,
            dataset_sha256=dataset_sha256,
            manifest=manifest,
            physical_columns=physical_columns,
        ):
            raise MetricRegistryError("A project metric definition does not match the trusted dataset manifest.")
    return definitions


def metric_definition_for_key(
    dataset_sha256: str, metric_key: str
) -> MetricDefinitionContext | None:
    """Return a defensive copy of a statically registered definition by key."""
    entry = _REGISTRY.get((dataset_sha256, metric_key))
    return entry.model_copy(deep=True) if entry is not None else None


def is_registered_metric_definition(definition: MetricDefinitionContext) -> bool:
    """Return true only when all supplied fields equal the application registry entry."""
    canonical = _REGISTRY.get((definition.dataset_sha256 or "", definition.metric_key or ""))
    return canonical is not None and definition.model_dump(mode="json") == canonical.model_dump(mode="json")


def validate_authoritative_planner_context(
    *,
    dataset_sha256: str | None,
    manifest_version: str | None,
    columns: list[str],
    semantic_roles: dict[str, str],
    definitions: list[MetricDefinitionContext] | tuple[MetricDefinitionContext, ...],
) -> bool:
    """Validate planner context against registry-owned snapshot and metric entries."""
    if any(not is_registered_metric_definition(item) for item in definitions):
        return False
    hashes = {item.dataset_sha256 for item in definitions}
    if len(hashes) > 1 or (hashes and dataset_sha256 not in hashes):
        return False
    manifest = semantic_manifest_for_snapshot(dataset_sha256 or "")
    if manifest is None or manifest.manifest_version != manifest_version:
        return False
    if set(columns) != set(manifest.columns) or semantic_roles != manifest.columns:
        return False
    return all(
        validate_metric_definition(
            item,
            dataset_sha256=dataset_sha256 or "",
            manifest=manifest,
            physical_columns=columns,
        )
        for item in definitions
    )
