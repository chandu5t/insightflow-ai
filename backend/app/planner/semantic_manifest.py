"""Versioned, snapshot-bound semantic metadata for V2.8.1 execution."""

from __future__ import annotations

import hashlib
import csv
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal

SemanticRole = Literal[
    "identifier", "categorical", "additive_measure", "non_additive_measure",
    "date", "boolean", "unknown",
]
ALLOWED_ROLES = frozenset({
    "identifier", "categorical", "additive_measure", "non_additive_measure",
    "date", "boolean", "unknown",
})


@dataclass(frozen=True)
class DatasetSemanticManifest:
    manifest_version: str
    dataset_sha256: str
    columns: dict[str, SemanticRole]
    additive_products: tuple[tuple[str, str], ...] = ()

    def role_for(self, column: str) -> SemanticRole:
        return self.columns.get(column, "unknown")


# Explicit manifests bind these exact reviewed schemas to frozen dataset bytes.
# No role is inferred from a name or observed values at runtime.
_MANIFESTS: dict[str, DatasetSemanticManifest] = {
    "eae24ee4e675e8b99d6f0893a87c4b84717a0e2eb876a7f19654f88e46837d54":
        DatasetSemanticManifest(
            "insightflow-bench-v1.0-semantic-v1",
            "eae24ee4e675e8b99d6f0893a87c4b84717a0e2eb876a7f19654f88e46837d54",
            {
                "order_id": "identifier", "region": "categorical", "product": "categorical",
                "category": "categorical", "order_date": "date", "quantity": "additive_measure",
                "unit_price": "non_additive_measure",
            },
            (("quantity", "unit_price"),),
        ),
    "520172097d69e40cf729714342d8d6ce7a7763670b5d13f6d673aaf1af0359cd":
        DatasetSemanticManifest(
            "insightflow-bench-v1.0-semantic-v1",
            "520172097d69e40cf729714342d8d6ce7a7763670b5d13f6d673aaf1af0359cd",
            {
                "Order ID": "identifier", "Product Name": "categorical",
                "Quantity": "additive_measure", "Total Revenue": "additive_measure",
                "Region": "categorical",
            },
        ),
}


def semantic_manifest_for_snapshot(dataset_sha256: str) -> DatasetSemanticManifest | None:
    """Return the application-reviewed manifest bound to a snapshot hash."""
    manifest = _MANIFESTS.get(dataset_sha256)
    return manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_semantic_manifest(
    manifest: DatasetSemanticManifest,
    *,
    dataset_sha256: str,
    physical_columns: list[str] | tuple[str, ...],
) -> bool:
    """Validate a reviewed manifest against exact snapshot bytes and CSV headers."""
    if (
        not isinstance(manifest.manifest_version, str)
        or not manifest.manifest_version
        or not isinstance(manifest.dataset_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", manifest.dataset_sha256)
        or not isinstance(dataset_sha256, str)
    ):
        return False
    if dataset_sha256 != manifest.dataset_sha256:
        return False
    if any(not isinstance(column, str) or not column for column in physical_columns):
        return False
    if len(physical_columns) != len(set(physical_columns)):
        return False
    if set(manifest.columns) != set(physical_columns):
        return False
    if any(role not in ALLOWED_ROLES for role in manifest.columns.values()):
        return False
    for pair in manifest.additive_products:
        if not isinstance(pair, tuple) or len(pair) != 2:
            return False
        left, right = pair
        if not isinstance(left, str) or not isinstance(right, str):
            return False
        if left not in manifest.columns or right not in manifest.columns:
            return False
        if manifest.columns[left] != "additive_measure":
            return False
        if manifest.columns[right] not in {"additive_measure", "non_additive_measure"}:
            return False
    return True


def load_semantic_manifest(path: Path) -> DatasetSemanticManifest | None:
    """Load only a code-reviewed manifest bound to the exact file snapshot."""
    snapshot_hash = sha256_file(path)
    manifest = _MANIFESTS.get(snapshot_hash)
    if manifest is None:
        return None
    with path.open("r", encoding="utf-8", newline="") as source:
        physical_columns = next(csv.reader(source), None)
    if not physical_columns or not validate_semantic_manifest(
        manifest, dataset_sha256=snapshot_hash, physical_columns=physical_columns
    ):
        return None
    return manifest
