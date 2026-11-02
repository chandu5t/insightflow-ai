"""Persistence and serialization of machine-readable evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.evaluation.schemas import (
    AggregateMetrics,
    CaseEvaluationResult,
    ErrorRecord,
    ExperimentConfig,
    ExperimentManifest,
    ReproducibilityMetadata,
)


def persist_evaluation_artifacts(
    output_dir: Path | str,
    manifest: ExperimentManifest,
    case_results: list[CaseEvaluationResult],
    aggregate: AggregateMetrics,
    errors: list[ErrorRecord],
    reproducibility: ReproducibilityMetadata,
    ablation_results: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Persist all required machine-readable artifacts into the target run directory."""
    base_dir = Path(output_dir) / manifest.config.experiment_id
    base_dir.mkdir(parents=True, exist_ok=False)

    files: dict[str, Path] = {}

    # 1. Manifest
    manifest_path = base_dir / "experiment_manifest.json"
    manifest_dict = manifest.model_dump(mode="json")
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest_dict, f, indent=2, sort_keys=True, allow_nan=False)
    files["manifest"] = manifest_path

    # 2. Case Results
    case_results_path = base_dir / "case_results.json"
    cases_dict = [c.model_dump(mode="json") for c in case_results]
    with case_results_path.open("w", encoding="utf-8") as f:
        json.dump(cases_dict, f, indent=2, sort_keys=True, allow_nan=False)
    files["case_results"] = case_results_path

    # 3. Aggregate Results
    aggregate_path = base_dir / "aggregate_results.json"
    agg_dict = aggregate.model_dump(mode="json")
    with aggregate_path.open("w", encoding="utf-8") as f:
        json.dump(agg_dict, f, indent=2, sort_keys=True, allow_nan=False)
    files["aggregate_results"] = aggregate_path

    # 4. Error Results
    errors_path = base_dir / "error_results.json"
    err_dict = [e.model_dump(mode="json") for e in errors]
    with errors_path.open("w", encoding="utf-8") as f:
        json.dump(err_dict, f, indent=2, sort_keys=True, allow_nan=False)
    files["error_results"] = errors_path

    # 5. Ablation Results
    ablation_path = base_dir / "ablation_results.json"
    with ablation_path.open("w", encoding="utf-8") as f:
        json.dump(ablation_results or {}, f, indent=2, sort_keys=True, allow_nan=False)
    files["ablation_results"] = ablation_path

    # 6. Reproducibility Metadata
    reprod_path = base_dir / "reproducibility.json"
    reprod_dict = reproducibility.model_dump(mode="json")
    with reprod_path.open("w", encoding="utf-8") as f:
        json.dump(reprod_dict, f, indent=2, sort_keys=True, allow_nan=False)
    files["reproducibility"] = reprod_path

    return files
