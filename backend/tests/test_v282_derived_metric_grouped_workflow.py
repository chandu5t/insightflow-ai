from pathlib import Path
from uuid import uuid4

import pandas as pd

from app.multi_agent.agents import AnalysisAgent
from app.multi_agent.schemas import AgentResult, MultiAgentRequest, MultiAgentResult
from app.multi_agent.workflow import run_multi_agent_workflow
from app.planner.context import build_planner_context
from app.planner.prompts import SYSTEM_PROMPT
from app.planner.schemas import AnalysisPlan, PlanStep
from app.planner.validator import validate_plan
from app.services.metric_retriever import StubMetricRetriever
from app.verification.service import verify_execution


DATASET = (
    Path(__file__).parents[1]
    / "data/evaluation/insightflow_bench_v1.0/datasets/module3_sales.csv"
)


class Repository:
    def get_csv_path(self, _dataset_id):
        return DATASET

    def get(self, _dataset_id):
        return object()


def revenue_plan() -> AnalysisPlan:
    return AnalysisPlan(
        intent="total_revenue",
        reasoning_type="multi_step",
        steps=[
            PlanStep(
                step_id="derive_revenue", operation="derive_metric", description="Derive authorized revenue",
                inputs=["quantity", "unit_price"],
                parameters={
                    "formula": {"op": "multiply", "left": {"column": "quantity"},
                                "right": {"column": "unit_price"}},
                    "output_name": "revenue",
                },
            ),
            PlanStep(
                step_id="group_region", operation="group_by", description="Group revenue rows by region",
                inputs=["region"], parameters={"column": "region"}, depends_on=["derive_revenue"],
            ),
            PlanStep(
                step_id="sum_revenue", operation="aggregate", description="Sum revenue by region",
                inputs=["revenue"], parameters={"function": "sum"}, depends_on=["group_region"],
            ),
        ],
    )


def test_trusted_context_accepts_only_physical_sources_and_authorized_revenue():
    bundle = build_planner_context("Show revenue by region", DATASET)
    assert bundle.request.dataset_context is not None
    assert "revenue" not in bundle.request.dataset_context.columns
    assert bundle.authorized_metric_definitions[0].metric_key == "revenue"
    assert validate_plan(
        revenue_plan(),
        dataset_columns=bundle.request.dataset_context.columns,
        semantic_roles=bundle.request.dataset_context.semantic_roles,
        authorized_metric_definitions=bundle.authorized_metric_definitions,
    ) == []

    bad = AnalysisPlan(
        intent="total_revenue", reasoning_type="simple",
        steps=[PlanStep(step_id="select", operation="select_columns", description="Select revenue",
                        inputs=["region", "revenue"], parameters={"columns": ["region", "revenue"]})],
    )
    assert any(error.code == "UNKNOWN_PHYSICAL_COLUMN" for error in validate_plan(
        bad, dataset_columns=bundle.request.dataset_context.columns,
        semantic_roles=bundle.request.dataset_context.semantic_roles,
        authorized_metric_definitions=bundle.authorized_metric_definitions,
    ))


def test_grouped_derived_revenue_preserves_rows_and_is_independently_verified():
    dataset_id = uuid4()
    plan = revenue_plan()
    execution = run_multi_agent_workflow(
        MultiAgentRequest(question="Show revenue by region", dataset_id=dataset_id, analysis_plan=plan),
        Repository(), StubMetricRetriever(),
    )
    assert execution.workflow_status == "completed", execution.errors
    assert execution.final_result["kind"] == "grouped"
    assert any(item["key"] == "(missing)" for item in execution.final_result["groups"])

    source = pd.read_csv(DATASET)
    source["expected_revenue"] = source["quantity"] * source["unit_price"]
    source["region_key"] = source["region"].where(source["region"].notna(), "(missing)")
    expected = [
        {"key": str(key), "value": float(group["expected_revenue"].sum())}
        for key, group in source.groupby("region_key", sort=True, dropna=False)
    ]
    assert execution.final_result["groups"] == expected
    # The exact-duplicate source row remains part of the totals.
    assert sum(group["value"] for group in execution.final_result["groups"]) == float(source["expected_revenue"].sum())

    verified = verify_execution(plan, execution, repository=Repository(), dataset_id=dataset_id)
    assert verified.status == "passed", verified.model_dump(mode="json")
    semantic_checks = [check for check in verified.checks if check.name == "semantic_operation_verification"]
    assert [check.status for check in semantic_checks] == ["passed", "passed", "passed"]

    tampered = execution.model_copy(deep=True)
    tampered.final_result["groups"][0]["value"] += 1
    aggregate_agent = next(item for item in tampered.agent_results if item.step_id == "sum_revenue")
    aggregate_agent.result["groups"][0]["value"] += 1
    rejected = verify_execution(plan, tampered, repository=Repository(), dataset_id=dataset_id)
    aggregate_check = next(
        check for check in rejected.checks
        if check.name == "semantic_operation_verification" and check.step_ids == ["sum_revenue"]
    )
    assert aggregate_check.status == "failed"


def test_selected_table_to_derived_grouped_aggregate_is_independently_verified():
    """Regression for the live run: lineage verification must receive the plan index."""
    dataset_id = uuid4()
    canonical = revenue_plan()
    steps = [
        PlanStep(
            step_id="select_sales", operation="select_columns", description="Select metric sources",
            inputs=["quantity", "unit_price", "region"],
            parameters={"columns": ["quantity", "unit_price", "region"]},
        ),
        canonical.steps[0].model_copy(update={"depends_on": ["select_sales"]}),
        canonical.steps[1],
        canonical.steps[2],
    ]
    plan = AnalysisPlan(
        intent=canonical.intent, reasoning_type=canonical.reasoning_type, steps=steps,
    )
    execution = run_multi_agent_workflow(
        MultiAgentRequest(question="Show revenue by region", dataset_id=dataset_id, analysis_plan=plan),
        Repository(), StubMetricRetriever(),
    )

    assert execution.workflow_status == "completed", execution.errors
    assert execution.final_result["groups"] == [
        {"key": "(missing)", "value": 2500.0},
        {"key": "East", "value": 72000.0},
        {"key": "North", "value": 106500.0},
        {"key": "South", "value": 75000.0},
    ]
    verification = verify_execution(plan, execution, repository=Repository(), dataset_id=dataset_id)
    assert verification.status == "passed", verification.model_dump(mode="json")
    assert [check.status for check in verification.checks
            if check.name == "semantic_operation_verification"] == ["passed", "passed", "passed"]


def test_redundant_transitive_group_dependency_remains_invalid_and_prompt_is_explicit():
    bundle = build_planner_context("Show revenue by region", DATASET)
    canonical = revenue_plan()
    steps = [
        PlanStep(
            step_id="select_sales", operation="select_columns", description="Select metric sources",
            inputs=["quantity", "unit_price", "region"],
            parameters={"columns": ["quantity", "unit_price", "region"]},
        ),
        canonical.steps[0].model_copy(update={"depends_on": ["select_sales"]}),
        canonical.steps[1].model_copy(update={"depends_on": ["select_sales", "derive_revenue"]}),
        canonical.steps[2],
    ]
    invalid = AnalysisPlan(
        intent=canonical.intent, reasoning_type=canonical.reasoning_type, steps=steps,
    )
    errors = validate_plan(
        invalid,
        dataset_columns=bundle.request.dataset_context.columns,
        semantic_roles=bundle.request.dataset_context.semantic_roles,
        authorized_metric_definitions=bundle.authorized_metric_definitions,
    )

    assert any(error.code == "INVALID_DERIVED_METRIC_DEPENDENCY" for error in errors)
    assert "group_by.depends_on contains only the derive_metric step ID" in SYSTEM_PROMPT
    assert "aggregate.depends_on contains" in SYSTEM_PROMPT


def test_group_by_missing_physical_group_column_input_is_rejected_before_execution():
    bundle = build_planner_context("Show revenue by region", DATASET)
    canonical = revenue_plan()
    invalid_group = canonical.steps[1].model_copy(update={"inputs": ["revenue"]})
    invalid = AnalysisPlan(
        intent=canonical.intent,
        reasoning_type=canonical.reasoning_type,
        steps=[canonical.steps[0], invalid_group, canonical.steps[2]],
    )

    errors = validate_plan(
        invalid,
        dataset_columns=bundle.request.dataset_context.columns,
        semantic_roles=bundle.request.dataset_context.semantic_roles,
        authorized_metric_definitions=bundle.authorized_metric_definitions,
    )

    assert any(error.code == "INVALID_GROUP_BY_INPUT" for error in errors)
    assert "MUST also be" in SYSTEM_PROMPT and "group_by.inputs" in SYSTEM_PROMPT


def test_scalar_derived_revenue_aggregate_completes_and_verifies():
    dataset_id = uuid4()
    derivation = revenue_plan().steps[0]
    scalar_plan = AnalysisPlan(
        intent="total_revenue",
        reasoning_type="multi_step",
        steps=[
            derivation,
            PlanStep(
                step_id="sum_revenue",
                operation="aggregate",
                description="Sum authorized revenue",
                inputs=["revenue"],
                parameters={"function": "sum"},
                depends_on=["derive_revenue"],
            ),
        ],
    )

    execution = run_multi_agent_workflow(
        MultiAgentRequest(
            question="What is the total revenue?",
            dataset_id=dataset_id,
            analysis_plan=scalar_plan,
        ),
        Repository(),
        StubMetricRetriever(),
    )

    assert execution.workflow_status == "completed", execution.errors
    assert execution.final_result == {"kind": "scalar", "value": 256000.0}
    verification = verify_execution(
        scalar_plan, execution, repository=Repository(), dataset_id=dataset_id
    )
    assert verification.status == "passed", verification.model_dump(mode="json")
    assert [check.status for check in verification.checks
            if check.name == "semantic_operation_verification"] == ["passed", "passed"]


def test_ordinary_series_aggregate_preserves_trusted_dependency_role():
    dataset_id = uuid4()
    source = AgentResult(
        agent_name="analysis",
        step_id="source_metric",
        status="completed",
        result=[100.0, 156000.0],
        metadata={
            "operation": "derive_metric",
            "semantic_role": "additive_measure",
            "lineage": [{"kind": "column", "name": "revenue"}],
        },
    )
    state = {
        "dataset_id": dataset_id,
        "step_results": {"source_metric": pd.Series([100.0, 156000.0])},
        "agent_outputs": [source],
    }
    step = PlanStep(
        step_id="sum_series",
        operation="aggregate",
        description="Sum trusted series",
        inputs=["revenue"],
        parameters={"function": "sum"},
        depends_on=["source_metric"],
    )

    result, _ = AnalysisAgent().run(step, state)

    assert result.status == "completed"
    assert result.result == {"kind": "scalar", "value": 156100.0}
