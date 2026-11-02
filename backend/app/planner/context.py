"""Build exact, trusted planner context outside the planner service."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from app.planner.metric_registry import resolve_metric_definitions
from app.planner.schemas import DatasetContext, MetricDefinitionContext, PlannerRequest
from app.planner.semantic_manifest import load_semantic_manifest, sha256_file


@dataclass(frozen=True)
class PlannerContextBundle:
    request: PlannerRequest
    authorized_metric_definitions: tuple[MetricDefinitionContext, ...]


def build_planner_context(question: str, dataset_path: Path) -> PlannerContextBundle:
    """Read only CSV headers and trusted manifest metadata; never read benchmark labels."""
    with dataset_path.open("r", encoding="utf-8", newline="") as source:
        columns = next(csv.reader(source), None)
    if not columns or len(columns) != len(set(columns)) or any(not column for column in columns):
        raise ValueError("The dataset does not have a valid unique physical CSV header.")

    dataset_sha256 = sha256_file(dataset_path)
    manifest = load_semantic_manifest(dataset_path)
    roles = {column: manifest.role_for(column) for column in columns} if manifest else {
        column: "unknown" for column in columns
    }
    definitions = resolve_metric_definitions(
        dataset_sha256=dataset_sha256,
        manifest=manifest,
        physical_columns=columns,
    )
    dataset_context = DatasetContext(
        columns=columns,
        semantic_roles=roles,
        dataset_sha256=dataset_sha256,
        manifest_version=manifest.manifest_version if manifest else None,
    )
    request = PlannerRequest(
        question=question,
        dataset_context=dataset_context,
        metric_definitions=definitions,
    )
    return PlannerContextBundle(request, tuple(definitions))
