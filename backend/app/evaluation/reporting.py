"""Human-readable evaluation report generator matching Contract v1.1 §19."""

from __future__ import annotations

from pathlib import Path

from app.evaluation.schemas import (
    AggregateMetrics,
    ErrorRecord,
    ExperimentConfig,
    ReproducibilityMetadata,
)
from app.evaluation.metrics import build_aggregate_metric_records


_OFFICIAL_METRIC_NAMES = {
    "M1": "Numerical Accuracy",
    "M2": "Task Success Rate",
    "M3": "Planning Accuracy",
    "M4": "Tool Selection Accuracy",
    "M5": "Verification Detection Rate",
    "M6": "Correction Success Rate",
    "M7": "Hallucination / Unsupported-Claim Rate",
    "M8": "Explanation Groundedness",
    "M9": "Latency",
    "M10": "LLM/API Usage",
}


def _display_value(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float) and 0 <= value <= 1:
        return f"{value:.2%}"
    if isinstance(value, dict):
        return "; ".join(f"{key}={item if item is not None else 'N/A'}" for key, item in value.items())
    return str(value)


def generate_markdown_report(
    config: ExperimentConfig,
    aggregate: AggregateMetrics,
    errors: list[ErrorRecord],
    reproducibility: ReproducibilityMetadata,
    output_path: Path | str | None = None,
) -> str:
    """Generate an objective, auditable Markdown evaluation summary report."""
    planning_latency_str = f"{aggregate.mean_planning_latency_ms:.1f} ms" if aggregate.mean_planning_latency_ms is not None else "N/A"
    analysis_latency_str = f"{aggregate.mean_analysis_latency_ms:.1f} ms" if aggregate.mean_analysis_latency_ms is not None else "N/A"
    verification_latency_str = f"{aggregate.mean_verification_latency_ms:.1f} ms" if aggregate.mean_verification_latency_ms is not None else "N/A"
    correction_latency_str = f"{aggregate.mean_correction_latency_ms:.1f} ms" if aggregate.mean_correction_latency_ms is not None else "N/A"

    metric_records = aggregate.metric_records or build_aggregate_metric_records(aggregate, config.metric_version)
    official_records = {record.metric_id: record for record in metric_records if record.metric_id in _OFFICIAL_METRIC_NAMES}
    report_lines = [
        f"# InsightFlow AI V2.7 Evaluation Report — {config.experiment_id}",
        "",
        "## 1. Experiment Overview",
        f"- **Experiment ID:** `{config.experiment_id}`",
        f"- **Benchmark:** `{config.benchmark_id}` (v{config.benchmark_version})",
        f"- **Condition:** `{config.condition.value}`",
        f"- **Model:** `{config.model_name}` ({config.model_version})",
        f"- **Random Seed:** `{config.random_seed}`",
        f"- **Total Cases Evaluated:** {aggregate.total_cases}",
        f"- **Completed Cases:** {aggregate.completed_cases}",
        f"- **Failed Cases:** {aggregate.failed_cases}",
        "",
        "## 2. Official Research Metrics (M1–M10)",
        "",
        "| Metric ID | Name | Status | Denominator | Value | Unavailability reason |",
        "|---|---|---|---:|---|---|",
        *[
            f"| {metric_id} | {name} | {official_records[metric_id].status if metric_id in official_records else 'unavailable'} | "
            f"{official_records[metric_id].denominator if metric_id in official_records and official_records[metric_id].denominator is not None else 'N/A'} | "
            f"{_display_value(official_records[metric_id].value) if metric_id in official_records else 'N/A'} | "
            f"{official_records[metric_id].unavailable_reason or '' if metric_id in official_records else 'Metric record unavailable'} |"
            for metric_id, name in _OFFICIAL_METRIC_NAMES.items()
        ],
        "",
        "Official M3 Planning Accuracy is aggregated only from its official case-level result, which compares the observed plan with the frozen expected operation sequence where that annotation exists. No dependency graph is inferred. Dependency-evaluability and the separate operation-sequence diagnostic records are labeled diagnostic and are excluded from M3 aggregation.",
        "",
        f"M1 Numerical Accuracy: numerator={aggregate.numerical_correct_count}; denominator={aggregate.numerical_evaluated_count}; excluded/non-applicable={aggregate.numerical_excluded_count}. Applicability rule: only finite numeric scalars or non-empty all-numeric lists/mappings are M1 targets; booleans, text, mixed categorical/numeric structures, clarification/unsupported outcomes, and targets without an applicable numerical comparison are excluded.",
        "",
        "M4 is unavailable for Baseline B because V2.1 and Contract v1.1 define no canonical mapping from V1 tool names to V2.2 operation labels.",
        "",
        "### Stage Latencies (M9 Breakdown)",
        f"- Mean Planning Latency: {planning_latency_str}",
        f"- Mean Analysis/Execution Latency: {analysis_latency_str}",
        f"- Mean Verification Latency: {verification_latency_str}",
        f"- Mean Self-Correction Latency: {correction_latency_str}",
        "",
        "## 3. Error Taxonomy Distribution (E1–E15)",
        "",
    ]

    if not errors:
        report_lines.append("No errors recorded during this evaluation run.")
    else:
        from collections import Counter
        cat_counts = Counter(err.error_category for err in errors)
        report_lines.extend([
            "| Error Category | Count | Example Subcategory |",
            "|---|---|---|",
        ])
        for cat, cnt in sorted(cat_counts.items()):
            sub = next((e.error_subcategory for e in errors if e.error_category == cat), "")
            report_lines.append(f"| `{cat}` | {cnt} | `{sub}` |")

    report_lines.extend([
        "",
        "## 4. Reproducibility Metadata",
        f"- **Git Commit / Version:** `{reproducibility.git_commit}` ({reproducibility.software_version})",
        f"- **Execution Timestamp:** `{reproducibility.timestamp}`",
        f"- **Prompt Configuration:** `{reproducibility.prompt_configuration_version}`",
        f"- **Dataset Version:** `{reproducibility.dataset_version}`",
        f"- **Reproducible:** `{reproducibility.runtime_environment.get('reproducible', False)}`",
        "",
        "---",
        "*This report records software evaluation outputs; it does not claim research improvement or statistical significance.*",
        "*Metrics without sufficient observed data are reported as N/A.*",
    ])

    content = "\n".join(report_lines) + "\n"

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            f.write(content)

    return content
