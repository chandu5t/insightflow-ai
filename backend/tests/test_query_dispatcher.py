"""Tests for the tool dispatcher."""

from pathlib import Path

import pytest

from app.core.errors import AppError, ErrorCode, ToolError
from app.schemas.profile_schema import MissingValueReport
from app.schemas.query_schema import QueryPlan
from app.schemas.tool_schema import AggregationResult, GroupingResult, RankingResult
from app.services import query_dispatcher
from app.services.query_dispatcher import TOOL_HANDLERS, dispatch
from app.utils.dataframe_utils import load_dataframe
from tests.frames import SALES_PATH, frame_from_csv


@pytest.fixture(scope="module")
def sales():
    return load_dataframe(SALES_PATH)


def plan(**fields) -> QueryPlan:
    return QueryPlan(confidence=0.9, **fields)


def test_total_revenue(sales) -> None:
    result = dispatch(sales, plan(intent="total_revenue", tool_name="revenue"))

    assert isinstance(result, AggregationResult) and result.metric == "total_revenue" and result.value == 256000.0


def test_aggregations(sales) -> None:
    def run(metric: str, aggregation: str) -> AggregationResult:
        return dispatch(sales, plan(intent="aggregate", tool_name="aggregation", metric=metric, aggregation=aggregation))

    assert run("unit_price", "average").value == pytest.approx(13807.692308, abs=1e-6)
    assert run("quantity", "max").value == 5.0 and run("quantity", "max").metric == "maximum"
    assert run("quantity", "min").value == 1.0
    assert run("quantity", "sum").value == 30.0
    assert run("records", "count").value == 13
    assert run("average_order_value", "average").value == 25600.0
    assert run("quantity", "count").value == 13


def test_grouping(sales) -> None:
    result = dispatch(sales, plan(intent="group", tool_name="grouping", metric="revenue", aggregation="sum", group_by="region"))

    assert isinstance(result, GroupingResult)
    assert {g.group: g.value for g in result.groups} == {"East": 72000.0, "North": 106500.0, "South": 75000.0}


def test_ranking_uses_the_limit_and_the_order(sales) -> None:
    result = dispatch(sales, plan(intent="rank", tool_name="ranking", metric="revenue", aggregation="sum",
                                  group_by="product", sort_order="asc", limit=2))

    assert isinstance(result, RankingResult) and result.n_returned == 2
    assert [i.group for i in result.items] == ["Keyboard", "Mouse"]


def test_ranking_default_limit_is_five() -> None:
    frame = frame_from_csv("g,v\na,1\nb,2\nc,3\nd,4\ne,5\nf,6\ng,7")
    result = dispatch(frame, plan(intent="rank", tool_name="ranking", metric="records", aggregation="count",
                                  group_by="g", sort_order="desc"))

    assert len(result.items) == 5


def test_missing_values(sales) -> None:
    result = dispatch(sales, plan(intent="missing_values", tool_name="missing_values"))

    assert isinstance(result, MissingValueReport) and result.total_missing_cells == 1


def test_unknown_tool_is_refused(sales) -> None:
    bad = QueryPlan.model_construct(intent="aggregate", tool_name="run_python", metric="revenue", aggregation="sum")

    with pytest.raises(AppError) as info:
        dispatch(sales, bad)

    assert info.value.code == ErrorCode.UNSUPPORTED_TOOL


@pytest.mark.parametrize(
    "fields",
    [
        dict(intent="group", tool_name="grouping", metric="revenue", aggregation="sum", group_by=None),
        dict(intent="aggregate", tool_name="aggregation", metric=None, aggregation="sum"),
        dict(intent="aggregate", tool_name="aggregation", metric="revenue", aggregation="median"),
        dict(intent="group", tool_name="grouping", metric="profit", aggregation="sum", group_by="region"),
    ],
)
def test_invalid_arguments_are_refused(sales, fields) -> None:
    with pytest.raises(AppError) as info:
        dispatch(sales, QueryPlan.model_construct(**fields))

    assert info.value.code == ErrorCode.INVALID_QUERY_PLAN


def test_tool_errors_pass_through_unchanged() -> None:
    with pytest.raises(ToolError) as info:
        dispatch(frame_from_csv("product\na"), plan(intent="total_revenue", tool_name="revenue"))

    assert info.value.code == ErrorCode.INSUFFICIENT_DATA


def test_only_the_five_approved_tools_exist() -> None:
    assert set(TOOL_HANDLERS) == {"revenue", "aggregation", "grouping", "ranking", "missing_values"}


def test_the_dispatcher_source_has_no_dynamic_execution() -> None:
    source = Path(query_dispatcher.__file__).read_text(encoding="utf-8")

    for forbidden in ("eval(", "exec(", "getattr(", "__import__", "importlib", "subprocess", "os.system"):
        assert forbidden not in source