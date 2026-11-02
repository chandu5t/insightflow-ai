"""CLI-first evaluation commands matching Contract v1.1 §23."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.core.config import get_settings
from app.evaluation.benchmark import load_benchmark, validate_benchmark_suite
from app.evaluation.config import load_experiment_config
from app.evaluation.reporting import generate_markdown_report
from app.evaluation.runner import EvaluationRunner
from app.evaluation.schemas import AggregateMetrics, ErrorRecord, ExperimentConfig, ReproducibilityMetadata
from app.services.dataset_repository import JsonDatasetRepository
from app.api.analysis_routes import get_metric_retriever
from app.services.gemini_client import GeminiClient
from app.evaluation.service import MockPlannerModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.evaluation.cli",
        description="InsightFlow AI V2.7 Experimental Evaluation CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. validate-benchmark
    val_bench = subparsers.add_parser("validate-benchmark", help="Validate benchmark file or directory.")
    val_bench.add_argument("--benchmark", required=True, help="Path to benchmark JSON file.")

    # 2. validate-config
    val_conf = subparsers.add_parser("validate-config", help="Validate experiment configuration file.")
    val_conf.add_argument("--config", required=True, help="Path to experiment configuration JSON.")

    # 3. evaluate
    eval_cmd = subparsers.add_parser("evaluate", help="Execute an evaluation experiment.")
    eval_cmd.add_argument("--config", required=True, help="Path to experiment configuration JSON.")
    eval_cmd.add_argument("--benchmark", required=True, help="Path to benchmark JSON.")
    eval_cmd.add_argument("--output-dir", default=None, help="Directory to save artifacts.")

    # 4. summarize
    summ_cmd = subparsers.add_parser("summarize", help="Generate a human-readable summary from run artifacts.")
    summ_cmd.add_argument("--run-dir", required=True, help="Path to run artifacts directory.")

    return parser


def cmd_validate_benchmark(benchmark_path: str) -> int:
    try:
        cases = load_benchmark(benchmark_path)
        print(f"PASS: Validated {len(cases)} benchmark cases from '{benchmark_path}'.")
        return 0
    except Exception as exc:
        print(f"FAIL: Benchmark validation failed: {exc}", file=sys.stderr)
        return 1


def cmd_validate_config(config_path: str) -> int:
    try:
        config = load_experiment_config(config_path)
        print(f"PASS: Validated experiment config '{config.experiment_id}' (condition: {config.condition.value}).")
        return 0
    except Exception as exc:
        print(f"FAIL: Configuration validation failed: {exc}", file=sys.stderr)
        return 1


def cmd_evaluate(config_path: str, benchmark_path: str, output_dir: str | None = None) -> int:
    try:
        config = load_experiment_config(config_path)
        if output_dir:
            config.output_dir = output_dir

        cases = load_benchmark(benchmark_path)
        settings = get_settings()
        repo = JsonDatasetRepository(settings.upload_dir)
        retriever = get_metric_retriever(settings)
        if config.model_name == "mock-planner-model":
            planner_model = MockPlannerModel()
        else:
            if config.model_name != settings.gemini_model:
                raise ValueError("The configured model name does not match the available Gemini client configuration.")
            if not settings.gemini_api_key_value:
                raise ValueError("The configured evaluation model is unavailable; no live result was generated.")
            planner_model = GeminiClient(
                api_key=settings.gemini_api_key_value,
                model=settings.gemini_model,
                timeout_seconds=settings.gemini_timeout_seconds,
            )

        runner = EvaluationRunner(config, repo, retriever=retriever, planner_model=planner_model, settings=settings)
        dataset_mapping = config.evaluation_configuration.get("dataset_mapping", {})
        manifest, aggregate, case_results, errors = runner.run_suite(cases, dataset_mapping=dataset_mapping)

        print(f"SUCCESS: Evaluated {len(case_results)} cases for experiment '{config.experiment_id}'.")
        print(f"Task Success Rate: {aggregate.task_success_rate:.2%}")
        if aggregate.numerical_accuracy is not None:
            print(f"Numerical Accuracy: {aggregate.numerical_accuracy:.2%}")
        print(f"Artifacts saved to: {Path(config.output_dir) / config.experiment_id}")
        return 0
    except Exception as exc:
        print(f"FAIL: Evaluation execution failed: {exc}", file=sys.stderr)
        return 1


def cmd_summarize(run_dir: str) -> int:
    try:
        dir_path = Path(run_dir)
        manifest_p = dir_path / "experiment_manifest.json"
        agg_p = dir_path / "aggregate_results.json"
        err_p = dir_path / "error_results.json"
        rep_p = dir_path / "reproducibility.json"

        if not all(p.exists() for p in (manifest_p, agg_p, err_p, rep_p)):
            print(f"FAIL: Missing required artifact files in '{run_dir}'.", file=sys.stderr)
            return 1

        with manifest_p.open("r", encoding="utf-8") as f:
            manifest_data = json.load(f)
            config = ExperimentConfig.model_validate(manifest_data["config"])

        with agg_p.open("r", encoding="utf-8") as f:
            aggregate = AggregateMetrics.model_validate(json.load(f))

        with err_p.open("r", encoding="utf-8") as f:
            errors = [ErrorRecord.model_validate(e) for e in json.load(f)]

        with rep_p.open("r", encoding="utf-8") as f:
            reproducibility = ReproducibilityMetadata.model_validate(json.load(f))

        report = generate_markdown_report(config, aggregate, errors, reproducibility)
        print(report)
        return 0
    except Exception as exc:
        print(f"FAIL: Summarize failed: {exc}", file=sys.stderr)
        return 1


def main(args: list[str] | None = None) -> int:
    parser = build_parser()
    parsed = parser.parse_args(args)

    if parsed.command == "validate-benchmark":
        return cmd_validate_benchmark(parsed.benchmark)
    elif parsed.command == "validate-config":
        return cmd_validate_config(parsed.config)
    elif parsed.command == "evaluate":
        return cmd_evaluate(parsed.config, parsed.benchmark, parsed.output_dir)
    elif parsed.command == "summarize":
        return cmd_summarize(parsed.run_dir)
    return 1


if __name__ == "__main__":
    sys.exit(main())
