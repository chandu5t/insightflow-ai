"""Tests for how the LangGraph workflow is built."""

from langgraph.graph import END, START

from app.workflow.graph import NODE_NAMES, build_query_graph, build_state_graph
from app.workflow.state import WorkflowState
from tests.workflow_helpers import make_deps


def test_all_required_nodes_are_registered() -> None:
    builder = build_state_graph(make_deps())

    assert set(builder.nodes) == set(NODE_NAMES)
    assert set(NODE_NAMES) == {"classify", "route", "execute", "validate", "explain", "metric_definition", "respond"}


def test_entry_point_and_fixed_edges() -> None:
    edges = build_state_graph(make_deps()).edges

    assert (START, "classify") in edges
    for edge in [("classify", "route"), ("explain", "respond"), ("metric_definition", "respond"), ("respond", END)]:
        assert edge in edges


def test_conditional_routing_is_registered_after_route_execute_and_validate() -> None:
    assert set(build_state_graph(make_deps()).branches) == {"route", "execute", "validate"}


def test_the_graph_compiles() -> None:
    graph = build_query_graph(make_deps())

    assert callable(graph.invoke)


def test_state_keys_never_collide_with_node_names() -> None:
    # LangGraph refuses a node whose name is also a state key.
    assert not set(WorkflowState.__annotations__) & set(NODE_NAMES)