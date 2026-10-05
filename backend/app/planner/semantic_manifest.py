"""Versioned, snapshot-bound semantic metadata for V2.8.1 execution."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_semantic_manifest(path: Path) -> DatasetSemanticManifest | None:
    """Load only a code-reviewed manifest bound to the exact file snapshot."""
    snapshot_hash = sha256_file(path)
    manifest = _MANIFESTS.get(snapshot_hash)
    if manifest is None or manifest.dataset_sha256 != snapshot_hash:
        return None
    if not manifest.manifest_version or any(role not in ALLOWED_ROLES for role in manifest.columns.values()):
        return None
    return manifest
