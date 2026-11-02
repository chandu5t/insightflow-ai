import csv
import hashlib
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.errors import AppError, ErrorCode
from app.api.analysis_routes import get_metric_retriever
from app.api.dataset_routes import get_dataset_repository
from app.api.planner_routes import get_planner_model
from app.main import app
from app.multi_agent import demo_workflow
from app.multi_agent.schemas import AgentResult, MultiAgentResult
from app.planner.schemas import AnalysisPlan, PlanStep, PlannerMetadata, PlannerResponse
from app.planner.context import build_planner_context
from app.planner.semantic_manifest import sha256_file
from app.self_correction.schemas import CorrectionResponse
from app.verification.schemas import VerificationError, VerificationResult
from app.visualization.schemas import VisualizationError, VisualizationResponse
from app.services.metric_retriever import StubMetricRetriever
from tests.helpers import csv_upload


DATASET = (
    Path(__file__).parents[1]
    / "data/evaluation/insightflow_bench_v1.0/datasets/module3_sales.csv"
)


class Repository:
    def get_csv_path(self, _dataset_id):
        return "unused.csv"


def test_demo_planning_error_exposes_only_sanitized_provider_diagnostics():
    error = AppError(
        ErrorCode.LLM_PROVIDER_ERROR,
        "The planner model request failed.",
        502,
        {
            "failure_category": "LLM_PROVIDER_ERROR",
            "provider_diagnostics": {
                "category": "permission",
                "http_status": 403,
                "exception_types": ["ClientError"],
                "message": "private provider response text",
                "authorization": "SECRET",
            },
        },
    )

    safe = demo_workflow._safe_error("planning", error)

    assert safe["provider_category"] == "permission"
    assert safe["provider_http_status"] == "403"
    assert safe["provider_exception_types"] == "ClientError"
    assert "private provider response text" not in str(safe)
    assert "SECRET" not in str(safe)


def plan():
    return AnalysisPlan(intent="count_orders", reasoning_type="simple", steps=[
        PlanStep(step_id="count", operation="count", description="Count source rows")
    ])


def grouped_revenue_plan():
    return AnalysisPlan(intent="total_revenue", reasoning_type="multi_step", steps=[
        PlanStep(step_id="derive_revenue", operation="derive_metric", description="Derive revenue",
                 inputs=["quantity", "unit_price"], parameters={
                     "formula": {"op": "multiply", "left": {"column": "quantity"},
                                 "right": {"column": "unit_price"}},
                     "output_name": "revenue",
                 }),
        PlanStep(step_id="group_region", operation="group_by", description="Group by region",
                 inputs=["region"], parameters={"column": "region"}, depends_on=["derive_revenue"]),
        PlanStep(step_id="sum_revenue", operation="aggregate", description="Revenue by region",
                 inputs=["revenue"], parameters={"function": "sum"}, depends_on=["group_region"]),
    ])


def planner_response(value):
    return PlannerResponse(
        plan=value, valid=True, metadata=PlannerMetadata(
            request_id="plan-test", planner_model="mock", retry_used=False, latency_ms=0,
        ),
    )


def verify(status):
    return VerificationResult(status=status, checks=[], metadata={"request_id": f"verify-{status}", "tolerance_absolute": 1e-9})


def execution(value=1):
    agent = AgentResult(agent_name="analysis", step_id="count", status="completed", result=value,
                        metadata={"operation": "count", "semantic_role": "unknown", "lineage": [{"kind": "dataset", "dataset_id": "id"}]})
    return MultiAgentResult(workflow_status="completed", executed_steps=["count"],
                            step_statuses={"count": "completed"}, agent_results=[agent], final_result=value)


def install_planner(monkeypatch, value):
    monkeypatch.setattr(demo_workflow, "build_planner_context", lambda *_: SimpleNamespace(
        request=object(), authorized_metric_definitions=(),
    ))
    monkeypatch.setattr(demo_workflow, "create_plan", lambda *_, **__: planner_response(value))


def test_demo_flow_visualizes_only_after_passed_verification(monkeypatch):
    dataset_id = uuid4()
    value = plan()
    install_planner(monkeypatch, value)
    executed = execution()
    monkeypatch.setattr(demo_workflow, "run_multi_agent_workflow", lambda *_: executed)
    monkeypatch.setattr(demo_workflow, "verify_execution", lambda *_args, **_kwargs: verify("passed"))
    calls = []
    monkeypatch.setattr(demo_workflow, "visualize", lambda request: calls.append(request) or VisualizationResponse(
        status="unsupported", errors=[VisualizationError(category="unsupported_visualization", code="SCALAR", message="Scalar results have no chart fields.")]
    ))

    result = demo_workflow.run_demo_workflow(
        question="How many rows?", dataset_id=dataset_id, repository=Repository(),
        retriever=object(), planner_model=object(),
    )

    assert result["status"] == "completed"
    assert result["stages"]["verification"] == "passed"
    assert result["stages"]["visualization"] == "unsupported"
    assert len(calls) == 1
    assert calls[0].analysis_plan == value.model_dump(mode="json")
    assert calls[0].execution_result == executed.model_dump(mode="json")
    assert calls[0].verification_result["status"] == "passed"


def test_unsupported_plan_stops_before_execution_and_visualization(monkeypatch):
    value = AnalysisPlan(intent="unsupported_analysis", reasoning_type="unsupported", steps=[], unsupported_reason="No safe operation.")
    install_planner(monkeypatch, value)
    monkeypatch.setattr(demo_workflow, "run_multi_agent_workflow", lambda *_: (_ for _ in ()).throw(AssertionError("must not execute")))
    monkeypatch.setattr(demo_workflow, "visualize", lambda *_: (_ for _ in ()).throw(AssertionError("must not visualize")))

    result = demo_workflow.run_demo_workflow(
        question="Unsupported?", dataset_id=uuid4(), repository=Repository(),
        retriever=object(), planner_model=object(),
    )

    assert result["status"] == "unsupported"
    assert result["stages"]["planning"] == "unsupported"
    assert result["stages"]["execution"] == "pending"
    assert result["visualization_result"] is None


def test_correction_keeps_original_artifacts_at_visualization_boundary(monkeypatch):
    dataset_id = uuid4()
    original_plan = plan()
    revised_plan = original_plan.model_copy(deep=True)
    revised_execution = execution(2)
    failed = verify("failed")
    passed = verify("passed")
    install_planner(monkeypatch, original_plan)
    original_execution = execution()
    monkeypatch.setattr(demo_workflow, "run_multi_agent_workflow", lambda *_: original_execution)
    monkeypatch.setattr(demo_workflow, "verify_execution", lambda *_args, **_kwargs: failed)
    correction = CorrectionResponse(
        terminal_status="corrected", question="How many rows?", dataset_id=dataset_id,
        original_plan=original_plan, final_plan=revised_plan,
        original_execution_result=original_execution, final_execution_result=revised_execution,
        original_verification_result=failed, final_verification_result=passed,
        attempts_used=1,
    )
    monkeypatch.setattr(demo_workflow, "correct_execution", lambda *_: correction)
    requests = []
    monkeypatch.setattr(demo_workflow, "visualize", lambda request: requests.append(request) or VisualizationResponse(
        status="unsupported", errors=[VisualizationError(category="unsupported_visualization", code="SCALAR", message="No chart.")]
    ))

    result = demo_workflow.run_demo_workflow(
        question="How many rows?", dataset_id=dataset_id, repository=Repository(),
        retriever=object(), planner_model=object(),
    )

    assert result["stages"]["correction"] == "corrected"
    assert result["final_plan"] == revised_plan.model_dump(mode="json")
    assert len(requests) == 1
    assert requests[0].analysis_plan == original_plan.model_dump(mode="json")
    assert requests[0].execution_result == original_execution.model_dump(mode="json")
    assert requests[0].verification_result["status"] == "failed"
    assert requests[0].correction_result["final_plan"] == revised_plan.model_dump(mode="json")


def test_workflow_api_runs_real_execution_verification_and_visualization_with_mock_planner():
    dataset_id = uuid4()

    class DemoRepository:
        def get_csv_path(self, _dataset_id):
            return DATASET

        def get(self, _dataset_id):
            return object()

    class DeterministicPlanner:
        model_name = "test-planner"

        def generate_json(self, *, system_prompt, user_prompt):
            return grouped_revenue_plan().model_dump_json()

    app.dependency_overrides[get_dataset_repository] = DemoRepository
    app.dependency_overrides[get_metric_retriever] = StubMetricRetriever
    app.dependency_overrides[get_planner_model] = DeterministicPlanner
    try:
        response = TestClient(app).post("/analysis/workflow", json={
            "question": "Show revenue by region", "dataset_id": str(dataset_id),
        })
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["stages"]["planning"] == "completed"
    assert payload["stages"]["execution"] == "completed"
    assert payload["stages"]["verification"] == "passed"
    assert payload["verification_result"]["status"] == "passed"
    assert payload["visualization_result"]["status"] == "generated"
    assert payload["visualization_result"]["visualization"]["chart_type"] == "bar"
    assert payload["visualization_result"]["visualization"]["traceability"]["verification_status"] == "passed"


@pytest.mark.parametrize("windows_line_endings", [False, True])
def test_upload_of_reviewed_sales_snapshot_keeps_trust_and_completes_exact_demo_workflow(
    make_client, upload_dir, windows_line_endings,
):
    class DeterministicPlanner:
        model_name = "test-planner"
        user_prompt = None

        def generate_json(self, *, system_prompt, user_prompt):
            self.user_prompt = user_prompt
            return grouped_revenue_plan().model_dump_json()

    planner = DeterministicPlanner()
    app.dependency_overrides[get_metric_retriever] = lambda: StubMetricRetriever()
    app.dependency_overrides[get_planner_model] = lambda: planner
    client = make_client()
    source_bytes = DATASET.read_bytes()
    upload_bytes = source_bytes.replace(b"\n", b"\r\n") if windows_line_endings else source_bytes
    try:
        uploaded = client.post(
            "/datasets/upload",
            files=csv_upload(upload_bytes, "module3_sales.csv"),
        )
        assert uploaded.status_code == 201, uploaded.text
        dataset_id = uploaded.json()["dataset_id"]
        assert uploaded.json()["row_count"] == 13
        assert uploaded.json()["column_count"] == 7
        stored_path = upload_dir / f"{dataset_id}.csv"
        assert stored_path.read_bytes() == source_bytes
        expected_hash = "677847de4659edd0da00411ee20d7d1eeaf7e2e6dab922bdf66681e682e93061"
        assert sha256_file(stored_path) == expected_hash
        context = build_planner_context("Show revenue by region.", stored_path)
        assert context.request.dataset_context.dataset_sha256 == expected_hash
        assert context.request.dataset_context.manifest_version == "insightflow-bench-v1.0-semantic-v1"
        assert len(context.authorized_metric_definitions) == 1
        metric = context.authorized_metric_definitions[0]
        assert metric.metric_key == "revenue"
        assert metric.exact_column_references == ["quantity", "unit_price"]
        assert metric.allowed_expression == {
            "op": "multiply", "left": {"column": "quantity"},
            "right": {"column": "unit_price"},
        }
        with stored_path.open(encoding="utf-8", newline="") as stored_csv:
            stored_rows = list(csv.DictReader(stored_csv))
        assert len(stored_rows) == 13
        assert len(stored_rows) - len({tuple(sorted(row.items())) for row in stored_rows}) == 1
        assert sum(not (row["region"] or "").strip() for row in stored_rows) == 1

        response = client.post("/analysis/workflow", json={
            "question": "Show revenue by region.", "dataset_id": dataset_id,
        })
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["stages"]["planning"] == "completed"
    assert payload["planner_response"]["valid"] is True
    assert payload["planner_response"]["validation_errors"] == []
    import json
    planner_payload = json.loads(planner.user_prompt)
    assert planner_payload["metric_definitions"][0]["allowed_expression"] == {
        "op": "multiply", "left": {"column": "quantity"},
        "right": {"column": "unit_price"},
    }
    assert payload["stages"]["execution"] == "completed"
    assert payload["final_verification_result"]["status"] == "passed"
    expected = {
        "East": 72000.0, "North": 106500.0, "South": 75000.0,
        "(missing)": 2500.0,
    }
    assert {item["key"]: item["value"] for item in payload["final_execution_result"]["final_result"]["groups"]} == expected
    assert payload["visualization_result"]["visualization"]["data"] == [
        {"key": key, "value": expected[key]} for key in sorted(expected)
    ]


def test_modified_sales_upload_with_same_filename_and_headers_remains_untrusted(make_client, upload_dir):
    app.dependency_overrides[get_metric_retriever] = lambda: StubMetricRetriever()
    client = make_client()
    modified = DATASET.read_bytes().replace(b"North", b"NORTH", 1)
    try:
        uploaded = client.post(
            "/datasets/upload",
            files=csv_upload(modified, "module3_sales.csv"),
        )
    finally:
        app.dependency_overrides.clear()

    assert uploaded.status_code == 201, uploaded.text
    stored_path = upload_dir / f"{uploaded.json()['dataset_id']}.csv"
    assert hashlib.sha256(stored_path.read_bytes()).hexdigest() != (
        "677847de4659edd0da00411ee20d7d1eeaf7e2e6dab922bdf66681e682e93061"
    )
    context = build_planner_context("Show revenue by region.", stored_path)
    assert context.request.dataset_context.manifest_version is None
    assert context.authorized_metric_definitions == ()


def test_workflow_accepts_unregistered_dataset_without_claiming_trusted_manifest(tmp_path):
    dataset_path = tmp_path / "uploaded_sales.csv"
    dataset_path.write_text("item,amount\na,3\nb,4\n", encoding="utf-8")
    dataset_id = uuid4()

    class DemoRepository:
        def get_csv_path(self, _dataset_id):
            return dataset_path

        def get(self, _dataset_id):
            return object()

    count_plan = AnalysisPlan(intent="count_orders", reasoning_type="simple", steps=[
        PlanStep(step_id="count_rows", operation="count", description="Count source rows")
    ])

    class DeterministicPlanner:
        model_name = "test-planner"

        def generate_json(self, *, system_prompt, user_prompt):
            return count_plan.model_dump_json()

    app.dependency_overrides[get_dataset_repository] = DemoRepository
    app.dependency_overrides[get_metric_retriever] = StubMetricRetriever
    app.dependency_overrides[get_planner_model] = DeterministicPlanner
    try:
        response = TestClient(app).post("/analysis/workflow", json={
            "question": "How many rows?", "dataset_id": str(dataset_id),
        })
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["stages"]["planning"] == "completed"
    assert payload["stages"]["execution"] == "completed"
    assert payload["final_execution_result"]["final_result"] == 2
