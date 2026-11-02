"""Versioned, snapshot-bound semantic metadata for V2.8.1 execution."""

from __future__ import annotations

import hashlib
import csv
import io
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
    "677847de4659edd0da00411ee20d7d1eeaf7e2e6dab922bdf66681e682e93061":
        DatasetSemanticManifest(
            "insightflow-bench-v1.0-semantic-v1",
            "677847de4659edd0da00411ee20d7d1eeaf7e2e6dab922bdf66681e682e93061",
            {
                "order_id": "identifier", "region": "categorical", "product": "categorical",
                "category": "categorical", "order_date": "date", "quantity": "additive_measure",
                "unit_price": "non_additive_measure",
            },
            (("quantity", "unit_price"),),
        ),
    "76256f36b1440fd9d5a5e2ba9832ff89b621ba33cdf8442b36978ab85a139ced":
        DatasetSemanticManifest(
            "insightflow-bench-v1.0-semantic-v1",
            "76256f36b1440fd9d5a5e2ba9832ff89b621ba33cdf8442b36978ab85a139ced",
            {
                "Order ID": "identifier", "Product Name": "categorical",
                "Quantity": "additive_measure", "Total Revenue": "additive_measure",
                "Region": "categorical",
            },
        ),
}

_REVIEWED_SNAPSHOT_PATHS: dict[str, Path] = {
    "677847de4659edd0da00411ee20d7d1eeaf7e2e6dab922bdf66681e682e93061": (
        Path(__file__).resolve().parents[2]
        / "data/evaluation/insightflow_bench_v1.0/datasets/module3_sales.csv"
    ),
    "76256f36b1440fd9d5a5e2ba9832ff89b621ba33cdf8442b36978ab85a139ced": (
        Path(__file__).resolve().parents[2]
        / "data/evaluation/insightflow_bench_v1.0/datasets/module3_sales_direct.csv"
    ),
}


def semantic_manifest_for_snapshot(dataset_sha256: str) -> DatasetSemanticManifest | None:
    """Return the application-reviewed manifest bound to a snapshot hash."""
    manifest = _MANIFESTS.get(dataset_sha256)
    return manifest


def reviewed_csv_snapshot_for_upload(
    upload_bytes: bytes,
) -> tuple[bytes, DatasetSemanticManifest] | None:
    """Match a CSV upload to a reviewed snapshot before lossy upload normalization.

    Exact bytes match directly. Alternate CSV line endings or quoting match only when the
    complete parsed record sequence equals the reviewed snapshot: exact headers/order,
    every cell, duplicate rows, and missing cells. The returned bytes are always the
    hash-verified reviewed snapshot, never the client-provided representation.
    """
    for expected_hash, snapshot_path in _REVIEWED_SNAPSHOT_PATHS.items():
        manifest = _MANIFESTS.get(expected_hash)
        if manifest is None or not snapshot_path.is_file():
            continue
        try:
            snapshot_bytes = snapshot_path.read_bytes()
        except OSError:
            continue
        if hashlib.sha256(snapshot_bytes).hexdigest() != expected_hash:
            continue
        try:
            with snapshot_path.open("r", encoding="utf-8", newline="") as source:
                physical_columns = next(csv.reader(source), None)
        except (OSError, UnicodeDecodeError, csv.Error):
            continue
        if not physical_columns or not validate_semantic_manifest(
            manifest,
            dataset_sha256=expected_hash,
            physical_columns=physical_columns,
        ):
            continue
        if upload_bytes == snapshot_bytes or _csv_records_equal(upload_bytes, snapshot_bytes):
            return snapshot_bytes, manifest
    return None


def _csv_records_equal(left_bytes: bytes, right_bytes: bytes) -> bool:
    """Compare complete UTF-8 CSV records without header/value normalization."""
    sentinel = object()
    try:
        left_text = left_bytes.decode("utf-8")
        right_text = right_bytes.decode("utf-8")
        left_reader = csv.reader(io.StringIO(left_text, newline=""), strict=True)
        right_reader = csv.reader(io.StringIO(right_text, newline=""), strict=True)
        while True:
            left_row = next(left_reader, sentinel)
            right_row = next(right_reader, sentinel)
            if left_row is sentinel or right_row is sentinel:
                return left_row is sentinel and right_row is sentinel
            if left_row != right_row:
                return False
    except (UnicodeDecodeError, csv.Error):
        return False


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
