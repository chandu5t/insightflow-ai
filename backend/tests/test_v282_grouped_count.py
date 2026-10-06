"""V2.8.2 grouped source-row count execution and verification tests."""

from datetime import datetime, timezone
from uuid import UUID

import pandas as pd
import pytest

from app.multi_agent.agents import AnalysisAgent, execute_operation
from app.multi_agent.schemas import AgentResult, MultiAgentRequest
from app.multi_agent.workflow import run_multi_agent_workflow
from app.planner.schemas import AnalysisPlan, PlanStep
from app.planner.semantic_manifest import DatasetSemanticManifest
from app.planner.validator import validate_plan
from app.services.dataset_repository import JsonDatasetRepository
from app.services.metric_retriever import StubMetricRetriever
from app.verification import service as verification_service
from app.verification.service import verify_execution


DATASET_ID = UUID("12345678-1234-5678-1234-567812345678")


def step(step_id, operation, *, inputs=None, parameters=None, depends_on=None):
    return PlanStep(
        step_id=step_id,
        operation=operation,
        description=operation,
        inputs=inputs or [],
        parameters=parameters or {},
        depends_on=depends_on or [],
    )


def make_group_count_state(frame):
    group_step = step("groups", "group_by", inputs=["region"], parameters={"column": "region"})
    count_step = step("counts", "count", depends_on=["groups"])
    state = {
        "dataset_id": DATASET_ID,
        "frame": frame,
        "step_results": {},
        "agent_outputs": [],
        "semantic_manifest": DatasetSemanticManifest(
            "test-v1", "0" * 64, {"region": "categorical"}
        ),
    }
    group_agent, groups = AnalysisAgent().run(group_step, state)
    state["agent_outputs"].append(group_agent)
    state["step_results"]["groups"] = groups
    return state, group_step, count_step, group_agent, groups


def test_grouped_count_counts_rows_in_sorted_groups_including_missing():
    frame = pd.DataFrame({"region": ["West", None, "East", "East"], "value": [1, 2, 3, 4]})
    state, _, count_step, group_agent, _ = make_group_count_state(frame)
    result, raw = AnalysisAgent().run(count_step, state)

    assert group_agent.status == "completed"
    assert result.status == "completed"
    assert raw == {
        "kind": "grouped_count",
        "groups": [
            {"key": "(missing)", "value": 1},
            {"key": "East", "value": 2},
            {"key": "West", "value": 1},
        ],
    }
    assert result.result == raw
    assert result.metadata["lineage"] == [{"kind": "step", "step_id": "groups"}]


def test_grouped_count_empty_input_returns_empty_group_list():
    state, _, count_step, group_agent, groups = make_group_count_state(
        pd.DataFrame({"region": pd.Series(dtype="object")})
    )
    assert group_agent.status == "completed" and groups == {}
    result, raw = AnalysisAgent().run(count_step, state)
    assert result.status == "completed"
    assert raw == {"kind": "grouped_count", "groups": []}


@pytest.mark.parametrize("bad_dependency", [
    ("missing", None, {}),
    ("groups", "failed", {"operation": "group_by"}),
    ("groups", "completed", {"operation": "count"}),
    ("groups", "completed", {"operation": "group_by"}),
])
def test_grouped_count_rejects_missing_failed_wrong_or_non_group_results(bad_dependency):
    dependency_id, status, metadata = bad_dependency
    state = {
        "dataset_id": DATASET_ID,
        "frame": pd.DataFrame({"region": ["East"]}),
        "step_results": {dependency_id: 1 if metadata.get("operation") != "group_by" else ["not a group map"]},
        "agent_outputs": [],
        "semantic_manifest": DatasetSemanticManifest("test-v1", "0" * 64, {"region": "categorical"}),
    }
    if status is not None:
        state["agent_outputs"].append(AgentResult(
            agent_name="analysis", step_id=dependency_id, status=status,
            result={}, metadata=metadata,
        ))
    count_step = step("counts", "count", depends_on=[dependency_id])
    result, raw = AnalysisAgent().run(count_step, state)
    assert result.status == "failed"
    assert result.error.category == "agent_failure"
    assert raw is None


def test_grouped_count_requires_exactly_one_group_by_dependency_in_plan_validation():
    plan = AnalysisPlan(
        intent="count_orders",
        reasoning_type="multi_step",
        steps=[
            step("g", "group_by", inputs=["region"], parameters={"column": "region"}),
            step("other", "count"),
            step("count", "count", depends_on=["g", "other"]),
        ],
    )
    assert "INVALID_GROUPED_COUNT_DEPENDENCY" in {
        error.code for error in validate_plan(plan)
    }

    wrong_target = AnalysisPlan(
        intent="count_orders", reasoning_type="multi_step",
        steps=[step("source", "count"), step("count", "count", depends_on=["source"])],
    )
    assert "INVALID_GROUPED_COUNT_DEPENDENCY" in {
        error.code for error in validate_plan(wrong_target)
    }


def test_multi_agent_workflow_executes_grouped_count_additively(tmp_path):
    path = tmp_path / f"{DATASET_ID}.csv"
    pd.DataFrame({"region": ["West", None, "East", "East"]}).to_csv(path, index=False)

    class Repository:
        def get(self, dataset_id):
            from app.schemas.dataset_schema import DatasetMetadata
            assert dataset_id == DATASET_ID
            return DatasetMetadata(
                dataset_id=DATASET_ID, filename="groups.csv", source_format="csv",
                row_count=4, column_count=1, column_names=["region"],
                original_size_bytes=path.stat().st_size,
                uploaded_at=datetime.now(timezone.utc),
            )

        def get_csv_path(self, dataset_id):
            assert dataset_id == DATASET_ID
            return path

    plan = AnalysisPlan(
        intent="count_orders", reasoning_type="multi_step",
        steps=[
            step("groups", "group_by", inputs=["region"], parameters={"column": "region"}),
            step("counts", "count", depends_on=["groups"]),
        ],
    )
    result = run_multi_agent_workflow(
        MultiAgentRequest(question="Count rows per region", dataset_id=DATASET_ID, analysis_plan=plan),
        Repository(), StubMetricRetriever(),
    )
    assert result.workflow_status == "completed"
    assert result.final_result == {
        "kind": "grouped_count",
        "groups": [
            {"key": "(missing)", "value": 1},
            {"key": "East", "value": 2},
            {"key": "West", "value": 1},
        ],
    }


def _verification_fixture(tmp_path, monkeypatch, *, filter_condition=None):
    csv_path = tmp_path / "grouped.csv"
    pd.DataFrame({"region": ["West", None, "East", "East"]}).to_csv(csv_path, index=False)
    manifest = DatasetSemanticManifest("grouped-test-v1", "0" * 64, {"region": "categorical"})
    monkeypatch.setattr(verification_service, "load_semantic_manifest", lambda _path: manifest)

    class Repository:
        def get_csv_path(self, _dataset_id):
            return csv_path

        def get(self, _dataset_id):
            return object()

    steps = []
    group_dependencies = []
    if filter_condition is not None:
        steps.append(step("filtered", "filter_rows", parameters={"conditions": [filter_condition]}))
        group_dependencies = ["filtered"]
    group_step = step(
        "groups", "group_by", inputs=["region"], parameters={"column": "region"},
        depends_on=group_dependencies,
    )
    steps.extend([group_step, step("counts", "count", depends_on=["groups"])])
    plan = AnalysisPlan(intent="count_orders", reasoning_type="multi_step", steps=steps)
    manifest_context = AgentResult(
        agent_name="data_understanding", status="completed", result={},
        metadata={"dataset_id": str(DATASET_ID), "semantic_manifest_version": manifest.manifest_version,
                  "dataset_sha256": manifest.dataset_sha256},
    )
    state = {
        "dataset_id": DATASET_ID,
        "frame": pd.read_csv(csv_path),
        "step_results": {},
        "agent_outputs": [manifest_context],
        "semantic_manifest": manifest,
    }
    for plan_step in plan.steps:
        agent, raw_result = AnalysisAgent().run(plan_step, state)
        state["agent_outputs"].append(agent)
        if agent.status == "completed":
            state["step_results"][plan_step.step_id] = raw_result
    count_agent = state["agent_outputs"][-1]
    expected = count_agent.result
    result = {
        "workflow_status": "completed",
        "executed_steps": [item.step_id for item in plan.steps],
        "step_statuses": {item.step_id: "completed" for item in plan.steps},
        "agent_results": state["agent_outputs"],
        "final_result": count_agent.result,
        "errors": [],
    }
    return plan, result, expected, Repository()


def test_v24_independently_verifies_grouped_count(tmp_path, monkeypatch):
    plan, result, expected, repository = _verification_fixture(tmp_path, monkeypatch)
    verified = verify_execution(plan, result, repository=repository, dataset_id=DATASET_ID)
    check = next(item for item in verified.checks if item.name == "numerical_verification"
                 and item.step_ids == ["groups", "counts"])
    assert check.status == "passed"
    assert check.expected == expected
    assert check.metadata["count_semantics"] == "grouped_source_row_count"


def test_v24_recomputes_filtered_group_count_and_rejects_unfiltered_count(tmp_path, monkeypatch):
    condition = {"column": "region", "operator": "eq", "value": "East"}
    plan, result, expected_filtered, repository = _verification_fixture(
        tmp_path, monkeypatch, filter_condition=condition
    )
    verified = verify_execution(plan, result, repository=repository, dataset_id=DATASET_ID)
    count_check = next(item for item in verified.checks if item.name == "numerical_verification"
                       and item.step_ids == ["groups", "counts"])
    group_check = next(item for item in verified.checks
                       if item.name == "semantic_operation_verification" and item.step_ids == ["groups"])
    assert count_check.status == "passed"
    assert count_check.expected == {
        "kind": "grouped_count", "groups": [{"key": "East", "value": 2}]
    }
    assert count_check.expected == expected_filtered
    assert group_check.status == "passed"

    unfiltered_plan, _, unfiltered_count, _ = _verification_fixture(tmp_path, monkeypatch)
    assert unfiltered_plan.steps[-1].step_id == "counts"
    result["agent_results"][-1].result = unfiltered_count
    result["final_result"] = unfiltered_count
    rejected = verify_execution(plan, result, repository=repository, dataset_id=DATASET_ID)
    rejected_count = next(item for item in rejected.checks if item.name == "numerical_verification"
                          and item.step_ids == ["groups", "counts"])
    assert rejected_count.status == "failed"


def test_v24_grouped_count_verifies_empty_filtered_input(tmp_path, monkeypatch):
    plan, result, expected, repository = _verification_fixture(
        tmp_path, monkeypatch,
        filter_condition={"column": "region", "operator": "eq", "value": "No such region"},
    )
    verified = verify_execution(plan, result, repository=repository, dataset_id=DATASET_ID)
    count_check = next(item for item in verified.checks if item.name == "numerical_verification"
                       and item.step_ids == ["groups", "counts"])
    assert expected == {"kind": "grouped_count", "groups": []}
    assert count_check.status == "passed"
    assert count_check.expected == {"kind": "grouped_count", "groups": []}


def test_v24_fails_on_incorrect_grouped_count_structure_or_values(tmp_path, monkeypatch):
    plan, result, _, repository = _verification_fixture(tmp_path, monkeypatch)
    result["agent_results"][-1].result = {"kind": "grouped_count", "groups": [{"key": "East", "value": 99}]}
    result["final_result"] = result["agent_results"][-1].result
    verified = verify_execution(plan, result, repository=repository, dataset_id=DATASET_ID)
    check = next(item for item in verified.checks if item.name == "numerical_verification"
                 and item.step_ids == ["groups", "counts"])
    assert check.status == "failed"


def test_v24_rejects_grouped_count_when_group_by_dependency_is_missing(tmp_path, monkeypatch):
    plan, result, _, repository = _verification_fixture(tmp_path, monkeypatch)
    result["agent_results"] = [item for item in result["agent_results"] if item.step_id != "groups"]
    verified = verify_execution(plan, result, repository=repository, dataset_id=DATASET_ID)
    check = next(item for item in verified.checks if item.name == "numerical_verification"
                 and item.step_ids == ["groups", "counts"])
    assert check.status == "failed"


def test_v24_rejects_grouped_count_when_group_by_input_is_not_a_table(tmp_path, monkeypatch):
    from app.planner.schemas import AnalysisPlan

    plan, result, _, repository = _verification_fixture(tmp_path, monkeypatch)
    source_count = step("source_count", "count")
    plan = AnalysisPlan(
        intent="count_orders", reasoning_type="multi_step",
        steps=[source_count,
               step("groups", "group_by", inputs=["region"], parameters={"column": "region"},
                    depends_on=["source_count"]),
               step("counts", "count", depends_on=["groups"])],
    )
    result["executed_steps"].insert(0, "source_count")
    result["step_statuses"]["source_count"] = "completed"
    result["step_statuses"]["groups"] = "completed"
    result["step_statuses"]["counts"] = "completed"
    result["agent_results"].insert(1, AgentResult(
        agent_name="analysis", step_id="source_count", status="completed", result=4,
        metadata={"operation": "count", "lineage": [{"kind": "dataset", "dataset_id": str(DATASET_ID)}]},
    ))
    verified = verify_execution(plan, result, repository=repository, dataset_id=DATASET_ID)
    count_check = next(item for item in verified.checks if item.name == "numerical_verification"
                       and item.step_ids == ["groups", "counts"])
    assert count_check.status == "failed"
