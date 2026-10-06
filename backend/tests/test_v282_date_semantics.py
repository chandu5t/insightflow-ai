"""Focused V2.8.2 strict-date planning and execution tests."""

from uuid import UUID

import pandas as pd
import pytest

from app.multi_agent.agents import AnalysisAgent, execute_operation
from app.multi_agent.schemas import MultiAgentRequest
from app.multi_agent.workflow import run_multi_agent_workflow
from app.planner.date_semantics import InvalidDateValue, parse_strict_date
from app.planner.semantic_manifest import DatasetSemanticManifest
from app.planner.schemas import AnalysisPlan, PlanStep
from app.planner.validator import validate_plan
from app.services.dataset_repository import JsonDatasetRepository
from app.services.metric_retriever import StubMetricRetriever


def make_step(conditions):
    return PlanStep(
        step_id="filter",
        operation="filter_rows",
        description="Filter dates",
        parameters={"conditions": conditions},
    )


def make_plan(conditions):
    return AnalysisPlan(
        intent="time_based_revenue",
        reasoning_type="simple",
        steps=[make_step(conditions)],
    )


def make_state(frame, roles=None):
    manifest = DatasetSemanticManifest(
        "test-date-v1", "0" * 64,
        roles if roles is not None else {"order_date": "date"},
    )
    return {
        "dataset_id": UUID("12345678-1234-5678-1234-567812345678"),
        "frame": frame,
        "step_results": {},
        "agent_outputs": [],
        "semantic_manifest": manifest,
    }


def test_strict_date_parser_accepts_calendar_dates_and_rejects_noncanonical_values():
    assert parse_strict_date("2024-02-29").isoformat() == "2024-02-29"
    for value in ("2025-2-03", "2025-02-30", " 2025-02-03", "2025-02-03 ",
                  "2025-02-03T00:00:00", "03/02/2025", 20250203):
        with pytest.raises(InvalidDateValue):
            parse_strict_date(value)


@pytest.mark.parametrize("literal", [
    "2025-2-03", "2025-02-30", "2025-02-03T00:00:00", " 2025-02-03",
])
def test_planner_rejects_malformed_or_impossible_date_literals(literal):
    errors = validate_plan(
        make_plan([{"column": "order_date", "operator": "lte", "value": literal}]),
        semantic_roles={"order_date": "date"},
    )
    assert "INVALID_DATE_LITERAL" in {item.code for item in errors}


def test_planner_requires_trusted_date_role_and_rejects_reversed_ranges():
    conditions = [
        {"column": "order_date", "operator": "gte", "value": "2025-02-08"},
        {"column": "order_date", "operator": "lte", "value": "2025-02-03"},
    ]
    errors = validate_plan(make_plan(conditions), semantic_roles={"order_date": "unknown"})
    assert "UNTRUSTED_DATE_COLUMN" in {item.code for item in errors}

    errors = validate_plan(make_plan(conditions), semantic_roles={"order_date": "date"})
    assert "INVALID_DATE_RANGE" in {item.code for item in errors}


def test_planner_accepts_inclusive_range_and_half_open_calendar_month():
    inclusive = [
        {"column": "order_date", "operator": "gte", "value": "2025-01-05"},
        {"column": "order_date", "operator": "lte", "value": "2025-01-07"},
    ]
    january = [
        {"column": "order_date", "operator": "gte", "value": "2025-01-01"},
        {"column": "order_date", "operator": "lt", "value": "2025-02-01"},
    ]
    roles = {"order_date": "date"}
    assert validate_plan(make_plan(inclusive), semantic_roles=roles) == []
    assert validate_plan(make_plan(january), semantic_roles=roles) == []


def test_execution_uses_exact_on_inclusive_before_after_and_range_boundaries():
    frame = pd.DataFrame({"order_date": [
        "2025-01-04", "2025-01-05", "2025-01-07", "2025-01-08", None,
    ], "row": [0, 1, 2, 3, 4]})
    cases = [
        ([{"column": "order_date", "operator": "eq", "value": "2025-01-05"}], [1]),
        ([{"column": "order_date", "operator": "lte", "value": "2025-01-05"}], [0, 1]),
        ([{"column": "order_date", "operator": "gte", "value": "2025-01-07"}], [2, 3]),
        ([{"column": "order_date", "operator": "gte", "value": "2025-01-05"},
          {"column": "order_date", "operator": "lte", "value": "2025-01-07"}], [1, 2]),
        ([{"column": "order_date", "operator": "gte", "value": "2025-01-01"},
          {"column": "order_date", "operator": "lt", "value": "2025-02-01"}], [0, 1, 2, 3]),
        ([{"column": "order_date", "operator": "is_missing"}], [4]),
    ]
    for conditions, expected_rows in cases:
        output = execute_operation(make_step(conditions), make_state(frame))
        assert output["row"].tolist() == expected_rows


def test_execution_fails_atomically_on_invalid_nonmissing_source_date():
    frame = pd.DataFrame({"order_date": ["2025-01-01", "2025-1-02"], "row": [1, 2]})
    result, raw = AnalysisAgent().run(
        make_step([{"column": "order_date", "operator": "lte", "value": "2025-01-31"}]),
        make_state(frame),
    )
    assert result.status == "failed"
    assert result.error.category == "agent_failure"
    assert raw is None


def test_workflow_revalidates_date_role_against_loaded_trusted_manifest(tmp_path):
    class Repository:
        def __init__(self):
            self.delegate = JsonDatasetRepository(tmp_path)
            self.dataset_id = UUID("12345678-1234-5678-1234-567812345678")
            self.path = tmp_path / f"{self.dataset_id}.csv"
            self.path.write_text("order_date\n2025-01-01\n", encoding="utf-8")
            from app.schemas.dataset_schema import DatasetMetadata
            from datetime import datetime, timezone
            self.metadata = DatasetMetadata(
                dataset_id=self.dataset_id, filename="dates.csv", source_format="csv",
                row_count=1, column_count=1, column_names=["order_date"],
                original_size_bytes=self.path.stat().st_size,
                uploaded_at=datetime.now(timezone.utc),
            )

        def get(self, dataset_id):
            assert dataset_id == self.dataset_id
            return self.metadata

        def get_csv_path(self, dataset_id):
            assert dataset_id == self.dataset_id
            return self.path

    repository = Repository()
    request = MultiAgentRequest(
        question="Filter by date",
        dataset_id=repository.dataset_id,
        analysis_plan=make_plan([{
            "column": "order_date", "operator": "lte", "value": "2025-01-31",
        }]),
    )
    result = run_multi_agent_workflow(request, repository, StubMetricRetriever())
    assert result.workflow_status == "failed"
    assert result.errors[0].category == "invalid_plan"
    assert result.errors[0].code == "UNTRUSTED_DATE_COLUMN"
    assert result.executed_steps == []
