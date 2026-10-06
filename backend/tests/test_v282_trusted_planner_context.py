"""Focused tests for V2.8.2 trusted snapshot and metric context."""

from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.planner.context import build_planner_context
from app.planner.metric_registry import (
    METRIC_REGISTRY_VERSION,
    metric_definition_for_key,
    resolve_metric_definitions,
    validate_metric_definition,
)
from app.planner.schemas import AnalysisPlan
from app.planner.semantic_manifest import (
    load_semantic_manifest,
    sha256_file,
    validate_semantic_manifest,
)
from app.planner.validator import validate_plan

BENCHMARK_DIR = Path(__file__).parents[1] / "data" / "evaluation" / "insightflow_bench_v1.0"
SALES = BENCHMARK_DIR / "datasets" / "module3_sales.csv"
DIRECT = BENCHMARK_DIR / "datasets" / "module3_sales_direct.csv"


def revenue_plan(formula, *, output_name="revenue", inputs=None):
    return AnalysisPlan.model_validate({
        "intent": "total_revenue",
        "reasoning_type": "multi_step",
        "steps": [{
            "step_id": "derive",
            "operation": "derive_metric",
            "description": "Derive the registered metric",
            "inputs": inputs if inputs is not None else ["quantity", "unit_price"],
            "parameters": {"formula": formula, "output_name": output_name},
            "depends_on": [],
        }],
    })


def test_manifest_resolves_exact_sales_and_direct_snapshots_and_roles():
    sales = load_semantic_manifest(SALES)
    direct = load_semantic_manifest(DIRECT)

    assert sales is not None and direct is not None
    assert sales.role_for("order_date") == "date"
    assert sales.role_for("not_a_column") == "unknown"
    assert direct.role_for("Total Revenue") == "additive_measure"
    assert direct.role_for("total_revenue") == "unknown"


def test_manifest_validation_rejects_hash_column_and_role_mismatches():
    manifest = load_semantic_manifest(SALES)
    assert manifest is not None
    with SALES.open("r", encoding="utf-8", newline="") as file:
        import csv
        headers = next(csv.reader(file))

    assert validate_semantic_manifest(
        manifest, dataset_sha256=sha256_file(SALES), physical_columns=headers
    )
    assert not validate_semantic_manifest(
        manifest, dataset_sha256="0" * 64, physical_columns=headers
    )
    invalid_role = replace(manifest, columns={**manifest.columns, "quantity": "imaginary_role"})
    assert not validate_semantic_manifest(
        invalid_role, dataset_sha256=sha256_file(SALES), physical_columns=headers
    )
    assert not validate_semantic_manifest(
        manifest, dataset_sha256=sha256_file(SALES), physical_columns=[*headers, "category"]
    )


def test_unknown_or_modified_snapshot_has_no_trusted_manifest(tmp_path):
    unknown = tmp_path / "unknown.csv"
    unknown.write_bytes(SALES.read_bytes() + b"\n")
    assert load_semantic_manifest(unknown) is None


def test_registry_resolves_only_snapshot_bound_revenue_definitions():
    sales_manifest = load_semantic_manifest(SALES)
    direct_manifest = load_semantic_manifest(DIRECT)
    assert sales_manifest is not None and direct_manifest is not None
    with SALES.open("r", encoding="utf-8", newline="") as file:
        import csv
        sales_headers = next(csv.reader(file))
    with DIRECT.open("r", encoding="utf-8", newline="") as file:
        import csv
        direct_headers = next(csv.reader(file))

    sales_defs = resolve_metric_definitions(
        dataset_sha256=sha256_file(SALES), manifest=sales_manifest, physical_columns=sales_headers
    )
    direct_defs = resolve_metric_definitions(
        dataset_sha256=sha256_file(DIRECT), manifest=direct_manifest, physical_columns=direct_headers
    )
    assert METRIC_REGISTRY_VERSION == "insightflow-metric-registry-v1"
    assert len(sales_defs) == len(direct_defs) == 1
    assert sales_defs[0].allowed_expression == {
        "op": "multiply", "left": {"column": "quantity"}, "right": {"column": "unit_price"}
    }
    assert sales_defs[0].display_name == "Revenue"
    assert sales_defs[0].metric_key == "revenue"
    assert sales_defs[0].definition_version == "revenue-v1"
    assert direct_defs[0].exact_column_references == ["Total Revenue"]
    assert direct_defs[0].allowed_expression == {"column": "Total Revenue"}
    assert direct_defs[0].output_semantic_role == "additive_measure"
    assert metric_definition_for_key("0" * 64, "revenue") is None


def test_registry_rejects_header_alias_or_role_mismatch():
    manifest = load_semantic_manifest(DIRECT)
    assert manifest is not None
    definition = metric_definition_for_key(sha256_file(DIRECT), "revenue")
    assert definition is not None
    aliased = definition.model_copy(update={"exact_column_references": ["total_revenue"]})
    with DIRECT.open("r", encoding="utf-8", newline="") as file:
        import csv
        headers = next(csv.reader(file))
    assert not validate_metric_definition(
        aliased, dataset_sha256=sha256_file(DIRECT), manifest=manifest, physical_columns=headers
    )
    wrong_hash = definition.model_copy(update={"dataset_sha256": "0" * 64})
    assert not validate_metric_definition(
        wrong_hash, dataset_sha256=sha256_file(DIRECT), manifest=manifest, physical_columns=headers
    )
    wrong_role = definition.model_copy(update={"output_semantic_role": "categorical"})
    assert not validate_metric_definition(
        wrong_role, dataset_sha256=sha256_file(DIRECT), manifest=manifest, physical_columns=headers
    )
    wrong_expression = definition.model_copy(update={"allowed_expression": {"column": "Quantity"}})
    assert not validate_metric_definition(
        wrong_expression, dataset_sha256=sha256_file(DIRECT), manifest=manifest, physical_columns=headers
    )


def test_authorized_expression_is_exact_and_unauthorized_product_is_rejected():
    bundle = build_planner_context("revenue", SALES)
    definition = bundle.authorized_metric_definitions[0]
    authorized = revenue_plan(definition.allowed_expression)
    unauthorized = revenue_plan({
        "op": "multiply", "left": {"column": "quantity"}, "right": {"literal": 2}
    })
    aliased_product = revenue_plan(
        definition.allowed_expression,
        output_name="other_metric",
    )
    assert validate_plan(authorized, authorized_metric_definitions=bundle.authorized_metric_definitions) == []
    assert "UNAUTHORIZED_METRIC_EXPRESSION" in {
        error.code for error in validate_plan(
            unauthorized, authorized_metric_definitions=bundle.authorized_metric_definitions
        )
    }
    assert "UNAUTHORIZED_METRIC_EXPRESSION" in {
        error.code for error in validate_plan(
            aliased_product, authorized_metric_definitions=bundle.authorized_metric_definitions
        )
    }
    assert "UNAUTHORIZED_METRIC_EXPRESSION" in {
        error.code for error in validate_plan(authorized, authorized_metric_definitions=())
    }


def test_direct_revenue_plan_binds_to_exact_physical_header():
    bundle = build_planner_context("revenue", DIRECT)
    expression = bundle.authorized_metric_definitions[0].allowed_expression
    plan = revenue_plan(expression, inputs=["Total Revenue"])
    assert validate_plan(plan, authorized_metric_definitions=bundle.authorized_metric_definitions) == []
    aliased = revenue_plan({"column": "total_revenue"}, inputs=["total_revenue"])
    codes = {error.code for error in validate_plan(
        aliased, authorized_metric_definitions=bundle.authorized_metric_definitions
    )}
    assert "UNAUTHORIZED_METRIC_EXPRESSION" in codes


def test_planner_context_contains_exact_headers_roles_and_only_registry_metadata(monkeypatch):
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path.name in {"questions.json", "ground_truth.json"}:
            raise AssertionError("Planner context attempted to read benchmark labels.")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    bundle = build_planner_context("What is revenue?", DIRECT)
    dumped = bundle.request.model_dump(mode="json")
    context = dumped["dataset_context"]

    assert context["columns"] == ["Order ID", "Product Name", "Region", "Quantity", "Total Revenue"]
    assert context["semantic_roles"]["Total Revenue"] == "additive_measure"
    assert "total_revenue" not in context["columns"]
    assert len(dumped["metric_definitions"]) == 1
    assert "case_id" not in dumped and "ground_truth" not in dumped and "expected_answer" not in dumped
    definition = dumped["metric_definitions"][0]
    assert set(definition) == {
        "display_name", "metric_key", "definition_version", "dataset_sha256",
        "manifest_version", "exact_column_references", "allowed_expression",
        "output_semantic_role", "source_reference",
    }
    assert definition["dataset_sha256"] == context["dataset_sha256"]
    assert definition["manifest_version"] == context["manifest_version"]
    assert definition["exact_column_references"] == ["Total Revenue"]


def test_metric_context_cannot_be_selected_by_case_or_question():
    first = build_planner_context("What is revenue?", SALES)
    second = build_planner_context("Different question text", SALES)
    assert first.authorized_metric_definitions == second.authorized_metric_definitions
    assert all("IFB-" not in item.source_reference for item in first.authorized_metric_definitions)


def test_unknown_snapshot_context_has_exact_columns_but_unknown_roles_and_no_definitions(tmp_path):
    custom = tmp_path / "custom.csv"
    custom.write_text("Revenue,Quantity\n5,1\n", encoding="utf-8")
    bundle = build_planner_context("What is revenue?", custom)
    assert bundle.request.dataset_context.columns == ["Revenue", "Quantity"]
    assert set(bundle.request.dataset_context.semantic_roles.values()) == {"unknown"}
    assert bundle.request.metric_definitions == []
    assert bundle.authorized_metric_definitions == ()


def test_planner_context_schema_rejects_benchmark_or_ground_truth_fields():
    from app.planner.schemas import PlannerRequest

    with pytest.raises(ValidationError):
        PlannerRequest.model_validate({
            "question": "What is revenue?",
            "dataset_context": {"columns": ["revenue"], "ground_truth": 123},
        })


def test_v24_verifies_direct_metric_against_exact_registered_header(tmp_path):
    import pandas as pd
    from uuid import UUID

    from app.multi_agent.schemas import AgentResult
    from app.verification.service import verify_execution

    dataset_id = UUID("12345678-1234-5678-1234-567812345678")

    class Repository:
        def get_csv_path(self, _dataset_id):
            return DIRECT

        def get(self, _dataset_id):
            return object()

    expression = {"column": "Total Revenue"}
    plan = revenue_plan(expression, inputs=["Total Revenue"])
    data_agent = AgentResult(
        agent_name="data_understanding", status="completed", result={},
        metadata={
            "dataset_id": str(dataset_id),
            "semantic_manifest_version": "insightflow-bench-v1.0-semantic-v1",
            "dataset_sha256": sha256_file(DIRECT),
        },
    )
    analysis_agent = AgentResult(
        agent_name="analysis", step_id="derive", status="completed",
        result=pd.read_csv(DIRECT)["Total Revenue"].astype(float).tolist(),
        metadata={
            "operation": "derive_metric", "semantic_role": "additive_measure", "output_name": "revenue",
            "lineage": [{"kind": "column", "name": "Total Revenue"}],
        },
    )
    workflow_result = {
        "workflow_status": "completed", "executed_steps": ["derive"],
        "step_statuses": {"derive": "completed"},
        "agent_results": [data_agent.model_dump(mode="json"), analysis_agent.model_dump(mode="json")],
        "final_result": analysis_agent.result, "errors": [],
    }
    verified = verify_execution(plan, workflow_result, repository=Repository(), dataset_id=dataset_id)
    metric_check = next(check for check in verified.checks
                        if check.name == "semantic_operation_verification")
    assert metric_check.status == "passed"
    assert metric_check.metadata["metric_key"] == "revenue"
    assert metric_check.metadata["metric_column_references"] == ["Total Revenue"]
    assert metric_check.metadata["metric_dataset_sha256"] == sha256_file(DIRECT)


def test_v23_revalidates_direct_metric_against_snapshot_registry():
    from datetime import datetime, timezone
    from uuid import UUID

    from app.multi_agent.schemas import MultiAgentRequest
    from app.multi_agent.workflow import V23Workflow
    from app.schemas.dataset_schema import DatasetMetadata
    from app.services.metric_retriever import StubMetricRetriever

    dataset_id = UUID("12345678-1234-5678-1234-567812345678")

    class Repository:
        def get_csv_path(self, _dataset_id):
            return DIRECT

        def get(self, _dataset_id):
            return DatasetMetadata(
                dataset_id=dataset_id, filename=DIRECT.name, source_format="csv",
                row_count=0, column_count=5, column_names=["Order ID", "Product Name", "Region", "Quantity", "Total Revenue"],
                original_size_bytes=DIRECT.stat().st_size, uploaded_at=datetime.now(timezone.utc),
            )

    allowed = metric_definition_for_key(sha256_file(DIRECT), "revenue")
    assert allowed is not None

    def prepare(expression, inputs):
        plan = revenue_plan(expression, inputs=inputs)
        request = MultiAgentRequest(question="What is revenue?", dataset_id=dataset_id, analysis_plan=plan)
        workflow = V23Workflow(Repository(), StubMetricRetriever())
        return workflow.prepare(workflow.initial_state(request))

    valid = prepare(allowed.allowed_expression, ["Total Revenue"])
    assert valid["workflow_status"] == "running"

    invalid = prepare({"column": "Quantity"}, ["Quantity"])
    assert invalid["workflow_status"] == "failed"
    assert invalid["errors"][0].category == "invalid_plan"
    assert invalid["errors"][0].code == "UNAUTHORIZED_METRIC_EXPRESSION"
