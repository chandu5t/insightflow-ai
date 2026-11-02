"""Unit tests for artifact persistence and reporting format."""

from pathlib import Path

from app.evaluation.artifacts import persist_evaluation_artifacts
from app.evaluation.reporting import generate_markdown_report
from app.evaluation.schemas import (
    AggregateMetrics,
    CaseEvaluationResult,
    ErrorRecord,
    ExperimentConfig,
    ExperimentManifest,
    ReproducibilityMetadata,
)


def test_persist_evaluation_artifacts(tmp_path: Path):
    config = ExperimentConfig(experiment_id="art_test_01")
    repro = ReproducibilityMetadata(
        benchmark_version="1.0",
        dataset_version="1.0",
        model_name="mock",
        model_version="v1",
        model_configuration={},
        prompt_configuration_version="v1",
        software_version="2.7.0",
        experiment_id="art_test_01",
        run_id="run-1",
        random_seed=42,
        evaluation_configuration={},
    )
    manifest = ExperimentManifest(
        config=config,
        reproducibility=repro,
        total_cases_evaluated=1,
        start_time="2026-10-04T12:00:00Z",
        end_time="2026-10-04T12:00:01Z",
        duration_seconds=1.0,
    )
    case_res = CaseEvaluationResult(
        case_id="Q1",
        condition="system_d_full_v2",
        run_id="run-1",
        success=True,
    )
    agg = AggregateMetrics(
        total_cases=1,
        completed_cases=1,
        failed_cases=0,
        task_success_rate=1.0,
    )
    err = ErrorRecord(
        case_id="Q1",
        stage="planner",
        error_category="E1_intent_misunderstanding",
        expected_behavior="Valid plan",
        observed_behavior="Invalid plan",
    )

    files = persist_evaluation_artifacts(
        output_dir=tmp_path,
        manifest=manifest,
        case_results=[case_res],
        aggregate=agg,
        errors=[err],
        reproducibility=repro,
    )

    assert len(files) == 6
    for path in files.values():
        assert path.exists()

    # Generate Markdown report
    report_text = generate_markdown_report(config, agg, [err], repro, output_path=tmp_path / "report.md")
    assert "InsightFlow AI V2.7 Evaluation Report" in report_text
    assert "M1" in report_text
    assert "M10" in report_text
    assert "E1_intent_misunderstanding" in report_text
