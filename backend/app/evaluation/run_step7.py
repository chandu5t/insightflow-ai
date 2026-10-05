"""Local entry point for the frozen Step 7 official evaluation.

Run from backend with ``.venv\\Scripts\\python.exe -m app.evaluation.run_step7``.
This module only wires existing V2.7 evaluation components to the frozen
InsightFlow-Bench artifacts; it does not define new evaluation semantics.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

from app.core.config import BACKEND_DIR, Settings, get_settings
from app.evaluation.ablations import InjectedError
from app.evaluation.official_benchmark import (
    BENCHMARK_DIRECTORY,
    OfficialBenchmarkDatasetRepository,
    load_official_benchmark,
    verify_official_benchmark_hashes,
)
from app.evaluation.runner import EvaluationRunner
from app.evaluation.schemas import ExperimentConfig, SystemCondition
from app.api.analysis_routes import get_metric_retriever
from app.services.gemini_client import GeminiClient
from app.services.metric_retriever import KnowledgeBaseMetricRetriever


OUTPUT_DIRECTORY = BACKEND_DIR / "data" / "evaluation" / "official_runs_v1.0"
FREEZE_DIRECTORY = BENCHMARK_DIRECTORY.parent / "step7_v1.0"
SCHEDULE_PATH = FREEZE_DIRECTORY / "error_injection_schedule_v1.0.json"
SUPPORTED_CONDITIONS = {
    "baseline_a": SystemCondition.BASELINE_A_LLM_ONLY,
    "baseline_c_v1": SystemCondition.BASELINE_C_V1,
    "system_d_full_v2": SystemCondition.SYSTEM_D_FULL_V2,
    "ablation_a2_no_verification": SystemCondition.ABLATION_A2_NO_VERIFICATION,
    "ablation_a3_no_correction": SystemCondition.ABLATION_A3_NO_CORRECTION,
    "ablation_a4_no_rag": SystemCondition.ABLATION_A4_NO_RAG,
}
FROZEN_ORDER = tuple(SUPPORTED_CONDITIONS)
FROZEN_NA = {
    "baseline_b": "No frozen canonical mapping from V1 tool labels to V2.2 operation labels.",
    "A1": "V2.3 requires a validated AnalysisPlan and does not derive one without V2.2.",
    "A5": "Alias of System D; it is not an additional independent observation.",
}


def build_experiment_config(
    condition: SystemCondition,
    *,
    settings: Settings,
    dataset_mapping: dict[str, str],
    freeze: dict,
    case_selection: list[str] | None = None,
    run_suffix: str | None = None,
) -> ExperimentConfig:
    """Build one run config from the frozen metadata and existing schema."""
    condition_key = condition.value
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    suffix = run_suffix or uuid4().hex[:8]
    retriever_mode = (
        "not_applicable" if condition == SystemCondition.BASELINE_A_LLM_ONLY
        else "disabled_by_ablation" if condition == SystemCondition.ABLATION_A4_NO_RAG
        else "knowledge_base"
    )
    benchmark = freeze["benchmark"]
    evaluation_configuration = {
        "retriever_mode": retriever_mode,
        "dataset_mapping": dataset_mapping,
        "frozen_benchmark_sha256": benchmark["sha256"],
        "ground_truth_version": benchmark["ground_truth_version"],
        "official_benchmark_adapter": "separated_questions_and_ground_truth_v1",
        "development_fixture": False,
    }
    return ExperimentConfig(
        experiment_id=f"step7_{condition_key}_{timestamp}_{suffix}",
        benchmark_id=benchmark["benchmark_id"],
        benchmark_version=benchmark["benchmark_version"],
        dataset_version=benchmark["dataset_version"],
        condition=condition,
        baseline_id=condition_key if condition_key.startswith("baseline_") else None,
        ablation_id=condition_key if condition_key.startswith("ablation_") else None,
        model_name=settings.gemini_model,
        case_selection=case_selection,
        evaluation_configuration=evaluation_configuration,
        statistical_procedure=json.dumps(freeze["statistical_procedure"], sort_keys=True),
        output_dir=str(OUTPUT_DIRECTORY),
    )


def load_injection_schedule() -> tuple[list[str], dict[str, InjectedError]]:
    """Adapt the frozen four-case schedule to the existing runner interface."""
    schedule = json.loads(SCHEDULE_PATH.read_text(encoding="utf-8"))
    if schedule.get("condition") != SystemCondition.SYSTEM_D_FULL_V2.value:
        raise ValueError("Frozen error-injection schedule condition is invalid.")
    if schedule.get("error_type") != "incorrect_intermediate_result":
        raise ValueError("Frozen error-injection schedule type is unsupported.")
    value = schedule["injected_value"]
    selected: list[str] = []
    injections: dict[str, InjectedError] = {}
    for item in schedule["scheduled_cases"]:
        case_id = item["case_id"]
        if item.get("repetition_count") != 1 or case_id in injections:
            raise ValueError("Frozen error-injection schedule is malformed.")
        selected.append(case_id)
        injections[case_id] = InjectedError(
            error_type="incorrect_intermediate_result",
            target_step_id=None,
            injected_value=value,
        )
    if len(selected) != 4:
        raise ValueError("Expected the frozen four-case error-injection schedule.")
    return selected, injections


def _write_status(path: Path, status: dict) -> None:
    path.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")


def run(only_condition: str = "all", *, postgres_host: str = "127.0.0.1", postgres_port: int = 5433) -> int:
    # Match the project's existing dotenv pattern before Settings or providers
    # are initialized. Values are consumed in-process and never printed.
    load_dotenv(dotenv_path=BACKEND_DIR / ".env", override=False)
    get_settings.cache_clear()
    base_settings = get_settings()
    settings = base_settings.model_copy(update={
        "storage_backend": "postgres",
        "postgres_host": postgres_host,
        "postgres_port": postgres_port,
    })
    if not settings.gemini_api_key_value:
        raise RuntimeError("GEMINI_API_KEY is not configured; no experiment was started.")

    freeze = verify_official_benchmark_hashes()
    cases = load_official_benchmark()
    repository = OfficialBenchmarkDatasetRepository()
    dataset_mapping = repository.dataset_mapping
    retriever = get_metric_retriever(settings)
    if not isinstance(retriever, KnowledgeBaseMetricRetriever):
        raise RuntimeError("The configured project settings did not create the required knowledge-base retriever.")
    planner_model = GeminiClient(
        api_key=settings.gemini_api_key_value,
        model=settings.gemini_model,
        timeout_seconds=settings.gemini_timeout_seconds,
    )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    status_path = OUTPUT_DIRECTORY / f"step7_execution_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{uuid4().hex[:8]}.json"
    status = {
        "benchmark_id": freeze["benchmark"]["benchmark_id"],
        "benchmark_version": freeze["benchmark"]["benchmark_version"],
        "question_count": len(cases),
        "ground_truth_version": freeze["benchmark"]["ground_truth_version"],
        "dataset_version": freeze["benchmark"]["dataset_version"],
        "conditions": {},
        "na_decisions": FROZEN_NA,
        "status": "running",
    }
    _write_status(status_path, status)

    selected = FROZEN_ORDER if only_condition == "all" else (only_condition,)
    completed: dict[str, str] = {}
    for key in selected:
        condition = SUPPORTED_CONDITIONS[key]
        config = build_experiment_config(
            condition,
            settings=settings,
            dataset_mapping=dataset_mapping,
            freeze=freeze,
        )
        try:
            runner = EvaluationRunner(config, repository, retriever=retriever, planner_model=planner_model, settings=settings)
            manifest, _, results, _ = runner.run_suite(cases, dataset_mapping=dataset_mapping)
            completed[key] = manifest.config.experiment_id
            status["conditions"][key] = {
                "status": "artifacts_written",
                "experiment_id": manifest.config.experiment_id,
                "case_results": len(results),
            }
        except Exception as exc:
            status["conditions"][key] = {
                "status": "blocked",
                "error_type": type(exc).__name__,
            }
        _write_status(status_path, status)

        if key == "system_d_full_v2" and key in completed:
            try:
                injection_cases, injections = load_injection_schedule()
                injection_config = build_experiment_config(
                    condition,
                    settings=settings,
                    dataset_mapping=dataset_mapping,
                    freeze=freeze,
                    case_selection=injection_cases,
                    run_suffix="error_injection",
                )
                injection_runner = EvaluationRunner(
                    injection_config, repository, retriever=retriever,
                    planner_model=planner_model, settings=settings,
                )
                injection_manifest, _, injection_results, _ = injection_runner.run_suite(
                    cases, dataset_mapping=dataset_mapping, error_injections=injections,
                )
                status["conditions"]["system_d_error_injection"] = {
                    "status": "artifacts_written",
                    "experiment_id": injection_manifest.config.experiment_id,
                    "case_results": len(injection_results),
                    "schedule_id": "InsightFlow-Bench-v1.0-error-injection-schedule-v1.0",
                }
            except Exception as exc:
                status["conditions"]["system_d_error_injection"] = {
                    "status": "blocked",
                    "error_type": type(exc).__name__,
                }
            _write_status(status_path, status)

    if only_condition == "all":
        status["conditions"]["baseline_b"] = {"status": "not_applicable", "reason": FROZEN_NA["baseline_b"]}
        status["conditions"]["A1"] = {"status": "not_applicable", "reason": FROZEN_NA["A1"]}
        status["conditions"]["A5"] = {
            "status": "alias",
            "reason": FROZEN_NA["A5"],
            "system_d_experiment_id": completed.get("system_d_full_v2"),
        }
    status["status"] = "completed_with_blocked_conditions" if any(
        item.get("status") == "blocked" for item in status["conditions"].values()
    ) else "completed"
    _write_status(status_path, status)
    print(f"Step 7 status: {status['status']}")
    print(f"Status artifact: {status_path}")
    for key, value in status["conditions"].items():
        print(f"{key}: {value['status']}")
    return 1 if status["status"] == "completed_with_blocked_conditions" else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run frozen InsightFlow-Bench Step 7 conditions locally.")
    parser.add_argument("--condition", choices=("all", *FROZEN_ORDER), default="all")
    parser.add_argument("--postgres-host", default="127.0.0.1")
    parser.add_argument("--postgres-port", type=int, default=5433)
    args = parser.parse_args(argv)
    try:
        return run(args.condition, postgres_host=args.postgres_host, postgres_port=args.postgres_port)
    except Exception as exc:
        # Avoid including provider/config exception messages, which may contain
        # sensitive request context. The exception class is sufficient here.
        print(f"Step 7 preflight failed: {type(exc).__name__}; no further details logged.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
