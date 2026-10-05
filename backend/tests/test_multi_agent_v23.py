"""Focused tests for the additive V2.3 multi-agent workflow."""

from uuid import UUID

import pandas as pd
import pytest

from app.multi_agent.agents import AnalysisAgent, DataUnderstandingAgent, KnowledgeAgent
from app.multi_agent.routing import AGENT_REGISTRY, OPERATION_AGENT, agent_for_operation, registered_agent_name
from app.multi_agent.schemas import MultiAgentRequest
from app.multi_agent.workflow import V23Workflow, run_multi_agent_workflow
from app.planner.operations import OPERATION_REGISTRY
from app.planner.schemas import AnalysisPlan, PlanStep
from app.services.dataset_repository import JsonDatasetRepository
from app.services.metric_retriever import StubMetricRetriever


class MemoryRepository:
    def __init__(self, tmp_path):
        self.repo = JsonDatasetRepository(tmp_path)
        self.dataset_id = UUID("12345678-1234-5678-1234-567812345678")
        self.frame = pd.DataFrame({"region": ["East", "West"], "value": ["10", "20"]})
        self.path = tmp_path / f"{self.dataset_id}.csv"
        self.frame.to_csv(self.path, index=False)
        from app.schemas.dataset_schema import DatasetMetadata
        from datetime import datetime, timezone
        self.metadata = DatasetMetadata(dataset_id=self.dataset_id, filename="data.csv", source_format="csv",
                                        row_count=2, column_count=2, column_names=["region", "value"],
                                        original_size_bytes=10, uploaded_at=datetime.now(timezone.utc))

    def get(self, dataset_id):
        assert dataset_id == self.dataset_id
        return self.metadata

    def get_csv_path(self, dataset_id):
        assert dataset_id == self.dataset_id
        return self.path


def make_plan(*steps, intent="count_orders", reasoning_type="multi_step"):
    return AnalysisPlan(intent=intent, reasoning_type=reasoning_type, steps=list(steps))


def request(repo, plan):
    return MultiAgentRequest(question="Count values", dataset_id=repo.dataset_id, analysis_plan=plan)


def test_request_validation_and_state_statuses(tmp_path):
    repo = MemoryRepository(tmp_path)
    plan = make_plan()
    req = request(repo, plan)
    state = V23Workflow(repo, StubMetricRetriever()).initial_state(req)
    assert state["workflow_status"] == "pending"
    assert set(AGENT_REGISTRY) == {"data_understanding", "analysis", "knowledge"}
    with pytest.raises(ValueError):
        MultiAgentRequest(question="  ", dataset_id=repo.dataset_id, analysis_plan=plan)


def test_all_fourteen_operations_have_controlled_analysis_route():
    assert set(OPERATION_AGENT) == set(OPERATION_REGISTRY)
    assert len(OPERATION_AGENT) == 14
    assert set(OPERATION_AGENT.values()) == {"analysis"}
    for operation in OPERATION_REGISTRY:
        assert agent_for_operation(operation) == "analysis"
    with pytest.raises(ValueError):
        agent_for_operation("execute_python")
    with pytest.raises(ValueError):
        registered_agent_name("planner")


def test_data_understanding_agent_loads_context_deterministically(tmp_path):
    repo = MemoryRepository(tmp_path)
    workflow = V23Workflow(repo, StubMetricRetriever())
    result, updates = workflow.data_agent.run(workflow.initial_state(request(repo, make_plan())))
    assert result.status == "completed"
    assert updates["dataset_context"] == {"columns": ["region", "value"], "row_count": 2, "column_count": 2}


def test_analysis_agent_executes_supported_operations_and_rejects_undefined_semantics(tmp_path):
    repo = MemoryRepository(tmp_path)
    workflow = V23Workflow(repo, StubMetricRetriever())
    state = workflow.initial_state(request(repo, make_plan()))
    _, updates = workflow.data_agent.run(state)
    state.update(updates)
    select = PlanStep(step_id="s1", operation="select_columns", description="select", parameters={"columns": ["value"]})
    first, raw = workflow.analysis_agent.run(select, state)
    assert first.status == "completed"
    assert list(raw.columns) == ["value"]
    state["step_results"] = {"s1": raw}
    count = PlanStep(step_id="s2", operation="count", description="count", inputs=["rows"], depends_on=["s1"])
    second, result = workflow.analysis_agent.run(count, state)
    assert second.status == "completed" and result == 2
    unsafe = PlanStep(step_id="s3", operation="derive_metric", description="formula", parameters={
        "formula": "__import__('os').system('whoami')", "output_name": "bad"})
    rejected, _ = workflow.analysis_agent.run(unsafe, state)
    assert rejected.status == "failed"
    assert rejected.error.category == "agent_failure"


def test_named_inputs_are_not_treated_as_step_ids_and_dependencies_resolve_results(tmp_path):
    repo = MemoryRepository(tmp_path)
    plan = make_plan(
        PlanStep(step_id="select", operation="select_columns", description="select columns",
                 inputs=["region", "value"], parameters={"columns": ["region", "value"]}),
        PlanStep(step_id="distinct", operation="distinct_count", description="count unique regions",
                 inputs=["region"], parameters={"column": "region"}, depends_on=["select"]),
    )
    result = run_multi_agent_workflow(request(repo, plan), repo, StubMetricRetriever())
    assert result.workflow_status == "completed"
    assert result.executed_steps == ["select", "distinct"]
    assert result.final_result == 2
    assert result.step_statuses == {"select": "completed", "distinct": "completed"}

    state = V23Workflow(repo, StubMetricRetriever()).initial_state(request(repo, plan))
    state["step_results"] = {"left": 9, "right": 4}
    difference = PlanStep(step_id="difference", operation="calculate_difference", description="subtract",
                          inputs=["left_value", "right_value"], depends_on=["left", "right"])
    difference_result, value = AnalysisAgent().run(difference, state)
    assert difference_result.status == "completed" and value == 5


def test_count_without_column_counts_rows_and_count_column_fails_safely(tmp_path):
    repo = MemoryRepository(tmp_path)
    workflow = V23Workflow(repo, StubMetricRetriever())
    state = workflow.initial_state(request(repo, make_plan()))
    _, updates = workflow.data_agent.run(state)
    state.update(updates)
    state["frame"] = pd.DataFrame({"value": ["1", None]})
    no_column = PlanStep(step_id="count", operation="count", description="count rows")
    result, value = workflow.analysis_agent.run(no_column, state)
    assert result.status == "completed" and value == 2

    with_column = PlanStep(step_id="count_col", operation="count", description="count values",
                           parameters={"column": "value"})
    for column_values in (["1", "2"], ["1", None]):
        state["frame"] = pd.DataFrame({"value": column_values})
        failed, _ = workflow.analysis_agent.run(with_column, state)
        assert failed.status == "failed"
        assert "null/value semantics" in failed.error.message


def test_group_by_retains_missing_label_and_distinct_count_fails_safely(tmp_path):
    repo = MemoryRepository(tmp_path)
    workflow = V23Workflow(repo, StubMetricRetriever())
    state = workflow.initial_state(request(repo, make_plan()))
    state["frame"] = pd.DataFrame({"value": ["East", None]})
    group_step = PlanStep(step_id="group_by", operation="group_by", description="group",
                          inputs=["value"], parameters={"column": "value"})
    grouped, groups = workflow.analysis_agent.run(group_step, state)
    assert grouped.status == "completed" and "(missing)" in groups
    count_step = PlanStep(step_id="distinct_count", operation="distinct_count", description="distinct",
                          inputs=["value"], parameters={"column": "value"})
    result, _ = workflow.analysis_agent.run(count_step, state)
    assert result.status == "failed" and "missing" in result.error.message


def test_aggregate_fails_without_trusted_additive_semantics(tmp_path):
    repo = MemoryRepository(tmp_path)
    state = V23Workflow(repo, StubMetricRetriever()).initial_state(request(repo, make_plan()))
    state["frame"] = pd.DataFrame({"value": ["1", "2"]})
    step = PlanStep(step_id="sum", operation="aggregate", description="sum values",
                    inputs=["value"], parameters={"function": "sum"})
    result, _ = AnalysisAgent().run(step, state)
    assert result.status == "failed"
    assert "trusted" in result.error.message


@pytest.mark.parametrize(("operation", "parameters"), [
    ("filter_rows", {"conditions": [{"column": "value", "operator": "gt", "value": 1}]}),
    ("derive_metric", {"formula": "value * 2", "output_name": "double"}),
    ("calculate_percentage_difference", {}),
    ("compare_groups", {"left_group": "East", "right_group": "West"}),
])
def test_invalid_operation_parameters_fail_and_percentage_difference_is_frozen(tmp_path, operation, parameters):
    repo = MemoryRepository(tmp_path)
    state = V23Workflow(repo, StubMetricRetriever()).initial_state(request(repo, make_plan()))
    state["step_results"] = {"left": 10, "right": 5}
    step = PlanStep(step_id="unsupported", operation=operation, description="underdefined",
                    parameters=parameters, depends_on=["left", "right"] if operation == "calculate_percentage_difference" else [])
    result, _ = AnalysisAgent().run(step, state)
    if operation == "calculate_percentage_difference":
        assert result.status == "completed" and result.result == 100.0
    else:
        assert result.status == "failed"
        assert result.error.category == "agent_failure"


def test_rank_executes_without_ties_and_fails_when_tie_policy_is_needed(tmp_path):
    repo = MemoryRepository(tmp_path)
    state = V23Workflow(repo, StubMetricRetriever()).initial_state(request(repo, make_plan()))
    step = PlanStep(step_id="rank", operation="rank", description="rank values",
                    parameters={"by": "value", "order": "desc"}, depends_on=["source"])
    state["step_results"] = {"source": pd.DataFrame({"value": [3, 1, 2]})}
    result, value = AnalysisAgent().run(step, state)
    assert result.status == "completed"
    assert value["rank"].tolist() == [1, 2, 3]
    state["step_results"] = {"source": pd.DataFrame({"value": [3, 3, 1]})}
    tied, _ = AnalysisAgent().run(step, state)
    assert tied.status == "failed"
    assert "tied values" in tied.error.message


def test_knowledge_agent_uses_injected_retriever():
    class Retriever:
        def lookup(self, question):
            from app.services.metric_retriever import MetricDefinitionResult
            return MetricDefinitionResult(found=False, source="stub", is_stub=True, query=question,
                                          definition=None, message="not found")
    state = {"question": "Define revenue"}
    result = KnowledgeAgent(Retriever()).run(state)
    assert result.status == "completed" and result.agent_name == "knowledge"


def test_supervisor_runs_sequentially_and_records_results(tmp_path):
    repo = MemoryRepository(tmp_path)
    plan = make_plan(
        PlanStep(step_id="s1", operation="select_columns", description="select", parameters={"columns": ["value"]}),
        PlanStep(step_id="s2", operation="count", description="count", inputs=["s1"], depends_on=["s1"]),
    )
    result = run_multi_agent_workflow(request(repo, plan), repo, StubMetricRetriever())
    assert result.workflow_status == "completed"
    assert result.executed_steps == ["s1", "s2"]
    assert [item.agent_name for item in result.agent_results] == ["data_understanding", "analysis", "analysis"]
    assert result.final_result == 2


def test_invalid_plan_and_agent_failure_terminate_without_retry(tmp_path):
    repo = MemoryRepository(tmp_path)
    invalid = make_plan(PlanStep(step_id="s1", operation="unknown", description="bad"))
    rejected = run_multi_agent_workflow(request(repo, invalid), repo, StubMetricRetriever())
    assert rejected.workflow_status == "failed"
    assert rejected.errors[0].category == "invalid_plan"
    plan = make_plan(PlanStep(step_id="s1", operation="filter_rows", description="undefined",
                              parameters={"conditions": []}))
    failed = run_multi_agent_workflow(request(repo, plan), repo, StubMetricRetriever())
    assert failed.workflow_status == "failed"
    assert failed.errors[0].category == "invalid_plan"
    assert failed.agent_results == []


def test_unexpected_graph_exception_returns_workflow_failure_without_retry(tmp_path, monkeypatch):
    repo = MemoryRepository(tmp_path)
    workflow = V23Workflow(repo, StubMetricRetriever())
    calls = {"build": 0}

    def fail_build():
        calls["build"] += 1
        raise RuntimeError("internal detail must not escape")

    monkeypatch.setattr(workflow, "build_graph", fail_build)
    result = workflow.run(request(repo, make_plan(PlanStep(
        step_id="one", operation="count", description="count rows"))))
    assert result.workflow_status == "failed"
    assert len(result.errors) == 1 and result.errors[0].category == "workflow_failure"
    assert "internal detail" not in result.errors[0].message
    assert calls["build"] == 1
    assert result.step_statuses == {"one": "pending"}


def test_metric_definition_invokes_knowledge_only_for_structured_intent(tmp_path):
    repo = MemoryRepository(tmp_path)
    plan = make_plan(intent="metric_definition", reasoning_type="definition")
    result = run_multi_agent_workflow(request(repo, plan), repo, StubMetricRetriever())
    assert [item.agent_name for item in result.agent_results] == ["data_understanding", "knowledge"]


def test_v23_api_boundary_is_additive_and_returns_structured_result(client, tmp_path):
    from app.api.analysis_routes import get_metric_retriever
    from app.api.dataset_routes import get_dataset_repository
    from app.api.multi_agent_routes import router
    from app.main import app

    repo = MemoryRepository(tmp_path)
    app.dependency_overrides[get_dataset_repository] = lambda: repo
    app.dependency_overrides[get_metric_retriever] = lambda: StubMetricRetriever()
    body = {"question": "Count rows", "dataset_id": str(repo.dataset_id),
            "analysis_plan": {"intent": "count_orders", "reasoning_type": "simple", "steps": [
                {"step_id": "c1", "operation": "count", "description": "count rows"}]}}
    response = client.post("/analysis/multi-agent", json=body)
    assert response.status_code == 200
    assert response.json()["workflow_status"] == "completed"
    assert response.json()["final_result"] == 2
    assert any(route.path == "/analysis/multi-agent" for route in router.routes)
