from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from app.multi_agent.agents import AnalysisAgent, execute_operation
from app.multi_agent.schemas import AgentResult, WorkflowError
from app.planner.semantic_manifest import DatasetSemanticManifest
from app.planner.schemas import AnalysisPlan, PlanStep
from app.planner.validator import validate_plan
from app.verification import service as verification_service
from app.verification.service import verify_execution


def step(step_id, operation, *, inputs=None, parameters=None, depends_on=None):
    return PlanStep(step_id=step_id, operation=operation, description=operation,
                    inputs=inputs or [], parameters=parameters or {}, depends_on=depends_on or [])


def state(frame, roles=None):
    manifest = DatasetSemanticManifest("test-v1", "0" * 64, roles or {})
    return {"dataset_id": uuid4(), "frame": frame, "step_results": {}, "agent_outputs": [],
            "semantic_manifest": manifest}


def test_filter_predicate_and_missing_predicates_preserve_order():
    frame = pd.DataFrame({"n": ["1", "", "3", "2"], "label": ["a", "b", "a", "c"]})
    frame["n"] = frame["n"].replace("", None)
    s = state(frame, {"n": "additive_measure", "label": "categorical"})
    out = execute_operation(step("f", "filter_rows", parameters={"conditions": [
        {"column": "n", "operator": "gte", "value": 2}]}), s)
    assert out["label"].tolist() == ["a", "c"]
    missing = execute_operation(step("m", "filter_rows", parameters={"conditions": [
        {"column": "n", "operator": "is_missing"}]}), s)
    assert missing["label"].tolist() == ["b"]


def test_filter_rejects_implicit_type_conversion_and_duplicate_mode_is_explicit():
    s = state(pd.DataFrame({"n": ["1", "2"]}), {"n": "additive_measure"})
    bad = step("bad", "filter_rows", parameters={"conditions": [
        {"column": "n", "operator": "eq", "value": "1"}]})
    with pytest.raises(ValueError, match="Numeric predicates"):
        execute_operation(bad, s)
    duplicate = execute_operation(step("d", "filter_rows", parameters={
        "mode": "exact_duplicate_rows", "keep": "first"}),
        state(pd.DataFrame({"x": [1, 1, 2]})))
    assert duplicate["x"].tolist() == [1, 2]


def test_derive_tree_is_bounded_and_rejects_zero_division():
    s = state(pd.DataFrame({"quantity": [2, 3], "unit_price": [5, 7]}),
              {"quantity": "additive_measure", "unit_price": "non_additive_measure"})
    s["semantic_manifest"] = DatasetSemanticManifest("test", "0" * 64,
        {"quantity": "additive_measure", "unit_price": "non_additive_measure"},
        (("quantity", "unit_price"),))
    formula = {"op": "multiply", "left": {"column": "quantity"}, "right": {"column": "unit_price"}}
    result = AnalysisAgent().run(step("d", "derive_metric", inputs=["quantity", "unit_price"],
        parameters={"formula": formula, "output_name": "revenue"}), s)[0]
    assert result.status == "completed" and result.result == [10.0, 21.0]
    assert result.metadata["semantic_role"] == "additive_measure"
    divzero = {"op": "divide", "left": {"column": "quantity"}, "right": {"literal": 0}}
    failed = AnalysisAgent().run(step("z", "derive_metric", inputs=["quantity"],
        parameters={"formula": divzero, "output_name": "bad"}), s)[0]
    assert failed.status == "failed"
    assert validate_plan(AnalysisPlan(intent="metric_definition", reasoning_type="definition", steps=[
        step("unsafe", "derive_metric", inputs=["quantity"],
             parameters={"formula": "__import__('os')", "output_name": "x"})]))


def test_derive_metric_can_reference_prior_numeric_step_and_dataset_column():
    s = state(pd.DataFrame({"quantity": [1, 2]}), {"quantity": "additive_measure"})
    prior = AgentResult(agent_name="analysis", step_id="offset", status="completed", result=2,
                        metadata={"operation": "count", "semantic_role": "unknown", "lineage": [{"kind": "step"}]})
    s["agent_outputs"] = [prior]
    s["step_results"]["offset"] = 2
    formula = {"op": "add", "left": {"column": "quantity"}, "right": {"step_id": "offset"}}
    plan = AnalysisPlan(intent="total_revenue", reasoning_type="multi_step", steps=[
        step("offset", "count"),
        step("derived", "derive_metric", inputs=["quantity"],
             parameters={"formula": formula, "output_name": "adjusted"}, depends_on=["offset"]),
    ])
    assert validate_plan(plan) == []
    result = AnalysisAgent().run(plan.steps[1], s)[0]
    assert result.status == "completed" and result.result == [3.0, 4.0]


def test_aggregate_requires_semantic_role_and_supports_grouped_sum():
    s = state(pd.DataFrame({"order_id": ["10", "11"], "sales": ["4", "5"], "region": ["N", "S"]}),
              {"order_id": "identifier", "sales": "additive_measure", "region": "categorical"})
    identifier = AnalysisAgent().run(step("i", "aggregate", inputs=["order_id"],
        parameters={"function": "sum"}), s)[0]
    assert identifier.status == "failed"
    summed = AnalysisAgent().run(step("s", "aggregate", inputs=["sales"],
        parameters={"function": "sum"}), s)[0]
    assert summed.status == "completed" and summed.result == {"kind": "scalar", "value": 9.0}
    groups = execute_operation(step("g", "group_by", inputs=["region"], parameters={"column": "region"}), s)
    s["step_results"]["g"] = groups
    aggregate = AnalysisAgent().run(step("ga", "aggregate", inputs=["sales"],
        parameters={"function": "sum"}, depends_on=["g"]), s)[0]
    assert aggregate.status == "completed"
    assert aggregate.result["groups"] == [{"key": "N", "value": 4.0}, {"key": "S", "value": 5.0}]


def test_group_by_missing_sentinel_and_percentage_difference():
    s = state(pd.DataFrame({"region": ["N", None], "v": [1, 2]}), {"region": "categorical"})
    groups = execute_operation(step("g", "group_by", inputs=["region"], parameters={"column": "region"}), s)
    assert "(missing)" in groups
    s["step_results"].update({"a": 110.0, "b": 100.0})
    result = AnalysisAgent().run(step("pct", "calculate_percentage_difference", depends_on=["a", "b"]), s)[0]
    assert result.status == "completed" and result.result == 10.0
    s["step_results"]["b"] = 0.0
    failed = AnalysisAgent().run(step("zero", "calculate_percentage_difference", depends_on=["a", "b"]), s)[0]
    assert failed.status == "failed"


def test_compare_groups_uses_only_explicit_prior_results():
    s = state(pd.DataFrame())
    source = AgentResult(agent_name="analysis", step_id="sales", status="completed",
                         result={"kind": "scalar", "value": 100.0},
                         metadata={"operation": "aggregate", "semantic_role": "additive_measure", "lineage": [{"kind": "dataset"}]})
    group = AgentResult(agent_name="analysis", step_id="group", status="completed",
                         result={"kind": "grouped", "groups": [{"key": "Accessories", "value": 10.0}]},
                         metadata={"operation": "aggregate", "semantic_role": "additive_measure", "lineage": [{"kind": "dataset"}]})
    s["agent_outputs"] = [source, group]
    s["step_results"].update({"sales": source.result, "group": group.result})
    parameters = {"left": {"step_id": "group", "selector": {"kind": "group_key", "key": "Accessories"}, "multiplier": 1},
                  "comparator": "lt",
                  "right": {"step_id": "sales", "selector": "scalar", "multiplier": .12}}
    result = AnalysisAgent().run(step("cmp", "compare_groups", parameters=parameters,
        depends_on=["group", "sales"]), s)[0]
    assert result.status == "completed" and result.result["result"] is True


def test_v24_independently_verifies_sum_without_ground_truth(tmp_path, monkeypatch):
    csv = tmp_path / "data.csv"
    csv.write_text("sales\n2\n3\n", encoding="utf-8")
    manifest = DatasetSemanticManifest("test-v1", "0" * 64, {"sales": "additive_measure"})
    monkeypatch.setattr(verification_service, "load_semantic_manifest", lambda _path: manifest)

    class Repository:
        def get_csv_path(self, _dataset_id): return csv
        def get(self, _dataset_id): return object()

    dataset_id = uuid4()
    plan = AnalysisPlan(intent="total_revenue", reasoning_type="simple", steps=[
        step("sum", "aggregate", inputs=["sales"], parameters={"function": "sum"})])
    agent = AgentResult(agent_name="analysis", step_id="sum", status="completed",
        result={"kind": "scalar", "value": 5.0},
        metadata={"operation": "aggregate", "semantic_role": "additive_measure",
                  "lineage": [{"kind": "column", "name": "sales"}]})
    workflow = {"workflow_status": "completed", "executed_steps": ["sum"], "step_statuses": {"sum": "completed"},
                "agent_results": [
                    AgentResult(agent_name="data_understanding", status="completed", result={}, metadata={
                        "dataset_id": str(dataset_id), "semantic_manifest_version": manifest.manifest_version,
                        "dataset_sha256": manifest.dataset_sha256}).model_dump(mode="json"),
                    agent.model_dump(mode="json")], "final_result": agent.result, "errors": []}
    verified = verify_execution(plan, workflow, repository=Repository(), dataset_id=dataset_id)
    check = next(c for c in verified.checks if c.name == "semantic_operation_verification")
    assert check.status == "passed" and verified.status == "passed"
    assert "ground_truth" not in Path(verification_service.__file__).read_text(encoding="utf-8")


def test_frozen_manifest_roles_match_the_exact_direct_csv_schema(tmp_path):
    dataset = Path(__file__).parents[1] / "data/evaluation/insightflow_bench_v1.0/datasets/module3_sales_direct.csv"
    from app.planner.semantic_manifest import load_semantic_manifest
    manifest = load_semantic_manifest(dataset)
    assert manifest is not None
    assert manifest.columns == {
        "Order ID": "identifier",
        "Product Name": "categorical",
        "Quantity": "additive_measure",
        "Total Revenue": "additive_measure",
        "Region": "categorical",
    }
    assert manifest.role_for("Order ID") == "identifier"
    assert manifest.role_for("order_id") == "unknown"
    assert manifest.role_for("product") == "unknown"
    assert manifest.role_for("region") == "unknown"
    assert manifest.role_for("category") == "unknown"
    from app.planner.semantic_manifest import load_semantic_manifest
    unknown = tmp_path / "unrecognized.csv"
    unknown.write_text("x\n1\n", encoding="utf-8")
    assert load_semantic_manifest(unknown) is None


def test_v24_independently_verifies_percentage_difference_and_comparison(tmp_path, monkeypatch):
    csv = tmp_path / "data.csv"
    csv.write_text("value,reference\n110,100\n", encoding="utf-8")
    manifest = DatasetSemanticManifest("test-v1", "0" * 64,
                                       {"value": "additive_measure", "reference": "additive_measure"})
    monkeypatch.setattr(verification_service, "load_semantic_manifest", lambda _path: manifest)

    class Repository:
        def get_csv_path(self, _dataset_id): return csv
        def get(self, _dataset_id): return object()

    dataset_id = uuid4()
    a = step("a", "aggregate", inputs=["value"], parameters={"function": "sum"})
    b = step("b", "aggregate", inputs=["reference"], parameters={"function": "sum"})
    pct = step("pct", "calculate_percentage_difference", depends_on=["a", "b"])
    plan = AnalysisPlan(intent="total_revenue", reasoning_type="multi_step", steps=[a, b, pct])
    ar = AgentResult(agent_name="analysis", step_id="a", status="completed",
        result={"kind": "scalar", "value": 110.0},
        metadata={"operation": "aggregate", "semantic_role": "additive_measure", "lineage": [{"kind": "column", "name": "value"}]})
    br = AgentResult(agent_name="analysis", step_id="b", status="completed",
        result={"kind": "scalar", "value": 100.0},
        metadata={"operation": "aggregate", "semantic_role": "additive_measure", "lineage": [{"kind": "column", "name": "reference"}]})
    pr = AgentResult(agent_name="analysis", step_id="pct", status="completed", result=10.0,
        metadata={"operation": "calculate_percentage_difference", "semantic_role": "non_additive_measure",
                  "lineage": [{"kind": "step", "step_id": "a"}, {"kind": "step", "step_id": "b"}]})
    data_context = AgentResult(agent_name="data_understanding", status="completed", result={}, metadata={
        "dataset_id": str(dataset_id), "semantic_manifest_version": manifest.manifest_version,
        "dataset_sha256": manifest.dataset_sha256}).model_dump(mode="json")
    raw = {"workflow_status": "completed", "executed_steps": ["a", "b", "pct"],
           "step_statuses": {"a": "completed", "b": "completed", "pct": "completed"},
           "agent_results": [data_context, *[x.model_dump(mode="json") for x in (ar, br, pr)]],
           "final_result": 10.0, "errors": []}
    verified = verify_execution(plan, raw, repository=Repository(), dataset_id=dataset_id)
    pct_check = next(c for c in verified.checks if c.name == "semantic_operation_verification" and c.step_ids == ["pct"])
    assert pct_check.status == "passed"

    comparison = step("cmp", "compare_groups", parameters={
        "left": {"step_id": "a", "selector": "scalar", "multiplier": 1},
        "comparator": "gt",
        "right": {"step_id": "b", "selector": "scalar", "multiplier": 1},
    }, depends_on=["a", "b"])
    plan = AnalysisPlan(intent="compare_groups", reasoning_type="multi_step", steps=[a, b, comparison])
    cmp_result = {"left_value": 110.0, "comparator": "gt", "right_value": 100.0, "result": True}
    cr = AgentResult(agent_name="analysis", step_id="cmp", status="completed", result=cmp_result,
        metadata={"operation": "compare_groups", "semantic_role": "boolean",
                  "lineage": [{"kind": "step", "step_id": "a"}, {"kind": "step", "step_id": "b"}]})
    raw.update({"executed_steps": ["a", "b", "cmp"],
                "step_statuses": {"a": "completed", "b": "completed", "cmp": "completed"},
                "agent_results": [data_context, *[x.model_dump(mode="json") for x in (ar, br, cr)]],
                "final_result": cmp_result})
    verified = verify_execution(plan, raw, repository=Repository(), dataset_id=dataset_id)
    cmp_check = next(c for c in verified.checks if c.name == "semantic_operation_verification" and c.step_ids == ["cmp"])
    assert cmp_check.status == "passed"


def test_v24_independently_verifies_filter_and_derived_metric(tmp_path, monkeypatch):
    csv = tmp_path / "data.csv"
    csv.write_text("quantity,unit_price,region,sales\n2,5,N,1\n3,7,S,2\n", encoding="utf-8")
    manifest = DatasetSemanticManifest("test-v1", "0" * 64,
        {"quantity": "additive_measure", "unit_price": "non_additive_measure",
         "region": "categorical", "sales": "additive_measure"}, (("quantity", "unit_price"),))
    monkeypatch.setattr(verification_service, "load_semantic_manifest", lambda _path: manifest)

    class Repository:
        def get_csv_path(self, _dataset_id): return csv
        def get(self, _dataset_id): return object()

    dataset_id = uuid4()
    du = AgentResult(agent_name="data_understanding", status="completed", result={}, metadata={
        "dataset_id": str(dataset_id), "semantic_manifest_version": manifest.manifest_version,
        "dataset_sha256": manifest.dataset_sha256}).model_dump(mode="json")
    filtered = {"columns": ["quantity", "unit_price", "region", "sales"],
                "rows": [{"quantity": "3", "unit_price": "7", "region": "S", "sales": "2"}]}
    filter_step = step("filter", "filter_rows", parameters={"conditions": [
        {"column": "sales", "operator": "gte", "value": 2}]})
    filter_agent = AgentResult(agent_name="analysis", step_id="filter", status="completed", result=filtered,
        metadata={"operation": "filter_rows", "lineage": [{"kind": "dataset", "dataset_id": str(dataset_id)}]})
    plan = AnalysisPlan(intent="filtered_revenue", reasoning_type="simple", steps=[filter_step])
    raw = {"workflow_status": "completed", "executed_steps": ["filter"],
           "step_statuses": {"filter": "completed"},
           "agent_results": [du, filter_agent.model_dump(mode="json")], "final_result": filtered, "errors": []}
    verified = verify_execution(plan, raw, repository=Repository(), dataset_id=dataset_id)
    assert next(c for c in verified.checks if c.name == "semantic_operation_verification").status == "passed"

    formula = {"op": "multiply", "left": {"column": "quantity"}, "right": {"column": "unit_price"}}
    derived_step = step("derive", "derive_metric", inputs=["quantity", "unit_price"],
        parameters={"formula": formula, "output_name": "revenue"})
    derived_agent = AgentResult(agent_name="analysis", step_id="derive", status="completed", result=[10.0, 21.0],
        metadata={"operation": "derive_metric", "semantic_role": "additive_measure", "output_name": "revenue",
                  "lineage": [{"kind": "column", "name": "quantity"}, {"kind": "column", "name": "unit_price"}]})
    plan = AnalysisPlan(intent="total_revenue", reasoning_type="simple", steps=[derived_step])
    raw.update({"executed_steps": ["derive"], "step_statuses": {"derive": "completed"},
                "agent_results": [du, derived_agent.model_dump(mode="json")], "final_result": [10.0, 21.0]})
    verified = verify_execution(plan, raw, repository=Repository(), dataset_id=dataset_id)
    assert next(c for c in verified.checks if c.name == "semantic_operation_verification").status == "passed"


def test_v24_independently_verifies_missing_group_and_grouped_aggregate(tmp_path, monkeypatch):
    csv = tmp_path / "data.csv"
    csv.write_text("region,sales\nN,2\n,3\n", encoding="utf-8")
    manifest = DatasetSemanticManifest("test-v1", "0" * 64,
                                       {"region": "categorical", "sales": "additive_measure"})
    monkeypatch.setattr(verification_service, "load_semantic_manifest", lambda _path: manifest)

    class Repository:
        def get_csv_path(self, _dataset_id): return csv
        def get(self, _dataset_id): return object()

    dataset_id = uuid4()
    groups = {
        "(missing)": {"columns": ["region", "sales"], "rows": [{"region": "(missing)", "sales": "3"}]},
        "N": {"columns": ["region", "sales"], "rows": [{"region": "N", "sales": "2"}]},
    }
    group_step = step("groups", "group_by", inputs=["region"], parameters={"column": "region"})
    aggregate_step = step("sum", "aggregate", inputs=["sales"], parameters={"function": "sum"},
                          depends_on=["groups"])
    group_agent = AgentResult(agent_name="analysis", step_id="groups", status="completed", result=groups,
        metadata={"operation": "group_by", "lineage": [{"kind": "dataset", "dataset_id": str(dataset_id)}]})
    aggregate_result = {"kind": "grouped", "groups": [{"key": "(missing)", "value": 3.0}, {"key": "N", "value": 2.0}]}
    aggregate_agent = AgentResult(agent_name="analysis", step_id="sum", status="completed", result=aggregate_result,
        metadata={"operation": "aggregate", "semantic_role": "additive_measure",
                  "lineage": [{"kind": "dataset", "dataset_id": str(dataset_id)},
                              {"kind": "step", "step_id": "groups"},
                              {"kind": "column", "name": "sales"}]})
    du = AgentResult(agent_name="data_understanding", status="completed", result={}, metadata={
        "dataset_id": str(dataset_id), "semantic_manifest_version": manifest.manifest_version,
        "dataset_sha256": manifest.dataset_sha256}).model_dump(mode="json")
    plan = AnalysisPlan(intent="total_revenue", reasoning_type="multi_step", steps=[group_step, aggregate_step])
    raw = {"workflow_status": "completed", "executed_steps": ["groups", "sum"],
           "step_statuses": {"groups": "completed", "sum": "completed"},
           "agent_results": [du, group_agent.model_dump(mode="json"), aggregate_agent.model_dump(mode="json")],
           "final_result": aggregate_result, "errors": []}
    verified = verify_execution(plan, raw, repository=Repository(), dataset_id=dataset_id)
    operation_checks = [c for c in verified.checks if c.name == "semantic_operation_verification"]
    assert [c.status for c in operation_checks] == ["passed", "passed"]
