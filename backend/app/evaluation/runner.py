"""Experiment execution engine coordinating loading, execution, aggregation, and artifact persistence."""

from __future__ import annotations

import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from uuid import UUID

from app.core.config import Settings, get_settings
from app.evaluation.ablations import InjectedError
from app.evaluation.artifacts import persist_evaluation_artifacts
from app.evaluation.benchmark import extract_ground_truth, load_benchmark, sanitize_case_for_execution
from app.evaluation.metrics import aggregate_metrics
from app.evaluation.reporting import generate_markdown_report
from app.evaluation.schemas import (
    AggregateMetrics,
    BenchmarkCase,
    CaseEvaluationResult,
    ErrorRecord,
    ExperimentConfig,
    ExperimentManifest,
    ReproducibilityMetadata,
)
from app.evaluation.service import evaluate_single_case
from app.planner.service import PlannerModel
from app.services.dataset_repository import DatasetRepository
from app.services.metric_retriever import MetricRetriever
from app.services.metric_retriever import KnowledgeBaseMetricRetriever, StubMetricRetriever


class EvaluationRunner:
    """The central V2.7 evaluation orchestrator."""

    def __init__(
        self,
        config: ExperimentConfig,
        repository: DatasetRepository,
        retriever: MetricRetriever | None = None,
        planner_model: PlannerModel | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.config = config
        self.repository = repository
        self.retriever = retriever
        self.planner_model = planner_model
        self.settings = settings or get_settings()

    def run_suite(
        self,
        cases: Sequence[BenchmarkCase],
        dataset_mapping: dict[str, str | UUID] | None = None,
        error_injections: dict[str, InjectedError] | None = None,
    ) -> tuple[ExperimentManifest, AggregateMetrics, list[CaseEvaluationResult], list[ErrorRecord]]:
        if not cases:
            raise ValueError("The benchmark contains no cases to evaluate.")
        selected_set = set(self.config.case_selection or [])
        case_ids = {case.id for case in cases}
        unknown_cases = selected_set - case_ids
        if unknown_cases:
            raise ValueError(f"Unknown case_selection IDs: {sorted(unknown_cases)}")
        if self.config.case_selection is not None and not selected_set:
            raise ValueError("case_selection must not be empty when supplied.")

        run_dir = Path(self.config.output_dir) / self.config.experiment_id
        if run_dir.exists():
            raise FileExistsError(f"Experiment output already exists: {run_dir}")

        development_fixture = self.config.evaluation_configuration.get("development_fixture") is True
        if development_fixture and self.config.benchmark_id.casefold() == "insightflow_bench":
            raise ValueError("A development fixture cannot identify itself as official InsightFlow-Bench.")
        actual_model_name = getattr(self.planner_model, "model_name", None)
        if not development_fixture and (
            self.config.model_name == "mock-planner-model"
            or actual_model_name == "mock-planner-model"
            or (actual_model_name is not None and actual_model_name != self.config.model_name)
        ):
            raise ValueError("Mock or mismatched planner models are permitted only for an explicitly marked development fixture.")
        requires_rag = self.config.condition.value in {
            "system_d_full_v2", "ablation_a1_no_planner",
            "ablation_a2_no_verification", "ablation_a3_no_correction",
        }
        if requires_rag:
            declared_retriever = self.config.evaluation_configuration.get("retriever_mode")
            if development_fixture and declared_retriever != "stub":
                raise ValueError("A development fixture must explicitly record retriever_mode='stub'.")
            if not development_fixture and declared_retriever != "knowledge_base":
                raise ValueError("A research run must explicitly record retriever_mode='knowledge_base'.")
            if isinstance(self.retriever, StubMetricRetriever) and not development_fixture:
                raise ValueError("A stub retriever is permitted only for an explicitly marked development fixture.")
            if not development_fixture and not isinstance(self.retriever, KnowledgeBaseMetricRetriever):
                raise ValueError("This condition requires the existing knowledge-base retriever.")

        start_time_iso = datetime.now(timezone.utc).isoformat()
        start_clock = time.perf_counter()

        # Filter cases if case_selection is set
        if self.config.case_selection:
            selected_set = set(self.config.case_selection)
            active_cases = [c for c in cases if c.id in selected_set]
        else:
            active_cases = list(cases)

        results: list[CaseEvaluationResult] = []
        all_errors: list[ErrorRecord] = []
        dataset_mapping = dataset_mapping or self.config.evaluation_configuration.get("dataset_mapping", {})
        error_injections = error_injections or {}
        unknown_injections = set(error_injections) - {case.id for case in active_cases}
        if unknown_injections:
            raise ValueError(f"Error injections reference unselected/unknown cases: {sorted(unknown_injections)}")

        missing_datasets = sorted({case.dataset for case in active_cases if case.dataset not in dataset_mapping})
        if missing_datasets:
            raise ValueError(f"No explicit dataset ID mapping supplied for: {missing_datasets}")

        for case in active_cases:
            case_input = sanitize_case_for_execution(case)
            ground_truth = extract_ground_truth(case)
            dataset_id = dataset_mapping[case.dataset]
            injection = error_injections.get(case.id)

            case_res = evaluate_single_case(
                case_input=case_input,
                ground_truth=ground_truth,
                condition=self.config.condition,
                dataset_id=dataset_id,
                repository=self.repository,
                retriever=self.retriever,
                planner_model=self.planner_model,
                settings=self.settings,
                error_injection=injection,
                run_id=f"{self.config.experiment_id}-case-{case.id}",
                numerical_tolerance=self.config.evaluation_configuration.get("tolerance"),
                metric_version=self.config.metric_version,
            )
            results.append(case_res)
            all_errors.extend(case_res.errors)

        duration = time.perf_counter() - start_clock
        end_time_iso = datetime.now(timezone.utc).isoformat()

        # Compute aggregate metrics
        aggregate = aggregate_metrics(results, metric_version=self.config.metric_version)

        # Build reproducibility metadata
        try:
            git_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
                cwd=Path(__file__).resolve().parents[2],
            ).stdout.strip()
            worktree_clean = not subprocess.run(
                ["git", "status", "--porcelain"], check=True, capture_output=True, text=True,
                cwd=Path(__file__).resolve().parents[2],
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            git_commit, worktree_clean = "unavailable", False

        reproducibility = ReproducibilityMetadata(
            benchmark_version=self.config.benchmark_version,
            dataset_version=self.config.dataset_version,
            model_name=self.config.model_name,
            model_version=self.config.model_version,
            model_configuration=self.config.model_configuration,
            prompt_configuration_version=self.config.prompt_configuration_version,
            software_version=self.config.system_version,
            experiment_id=self.config.experiment_id,
            run_id=f"run-{self.config.experiment_id}",
            random_seed=self.config.random_seed,
            evaluation_configuration=self.config.evaluation_configuration,
            git_commit=git_commit,
            runtime_environment={
                "python_version": sys.version,
                "platform": platform.platform(),
                "machine": platform.machine(),
                "worktree_clean": worktree_clean,
                "reproducible": worktree_clean,
            },
        )

        manifest = ExperimentManifest(
            config=self.config,
            reproducibility=reproducibility,
            total_cases_evaluated=len(active_cases),
            start_time=start_time_iso,
            end_time=end_time_iso,
            duration_seconds=round(duration, 3),
            output_files={
                name: str(run_dir / filename)
                for name, filename in {
                    "manifest": "experiment_manifest.json",
                    "case_results": "case_results.json",
                    "aggregate_results": "aggregate_results.json",
                    "error_results": "error_results.json",
                    "ablation_results": "ablation_results.json",
                    "reproducibility": "reproducibility.json",
                    "report": "report.md",
                }.items()
            },
        )

        # Persist artifacts
        out_files = persist_evaluation_artifacts(
            output_dir=self.config.output_dir,
            manifest=manifest,
            case_results=results,
            aggregate=aggregate,
            errors=all_errors,
            reproducibility=reproducibility,
            ablation_results={
                "condition": self.config.condition.value,
                "configuration": {
                    "disable_planner": self.config.condition.value == "ablation_a1_no_planner",
                    "disable_verification": self.config.condition.value == "ablation_a2_no_verification",
                    "disable_self_correction": self.config.condition.value == "ablation_a3_no_correction",
                    "disable_rag": self.config.condition.value == "ablation_a4_no_rag",
                },
                "aggregate_metrics": aggregate.model_dump(mode="json"),
                "case_ids": [result.case_id for result in results],
            },
        )

        # Generate human-readable report
        report_path = Path(self.config.output_dir) / self.config.experiment_id / "report.md"
        generate_markdown_report(self.config, aggregate, all_errors, reproducibility, output_path=report_path)
        out_files["report"] = report_path

        return manifest, aggregate, results, all_errors
