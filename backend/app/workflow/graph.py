"""Builds the LangGraph workflow. This file only wires nodes together. It has no business logic."""

from langgraph.graph import END, START, StateGraph

from app.workflow.nodes import (
    WorkflowDependencies,
    make_nodes,
    route_after_execute,
    route_after_route,
    route_after_validate,
)
from app.workflow.state import WorkflowState

NODE_NAMES = ("classify", "route", "execute", "validate", "explain", "metric_definition", "respond")


def build_state_graph(deps: WorkflowDependencies) -> StateGraph:
    """Create the graph (not compiled yet). Tests inspect it before compiling."""
    builder = StateGraph(WorkflowState)
    for name, node in make_nodes(deps).items():
        builder.add_node(name, node)

    builder.add_edge(START, "classify")
    builder.add_edge("classify", "route")
    builder.add_conditional_edges(
        "route",
        route_after_route,
        {
            "tool": "execute",
            "metric_definition": "metric_definition",
            "unsupported": "respond",
            "invalid": "respond",
        },
    )
    builder.add_conditional_edges(
        "execute", route_after_execute, {"validate": "validate", "respond": "respond"}
    )
    builder.add_conditional_edges(
        "validate", route_after_validate, {"explain": "explain", "respond": "respond"}
    )
    builder.add_edge("explain", "respond")
    builder.add_edge("metric_definition", "respond")
    builder.add_edge("respond", END)
    return builder


def build_query_graph(deps: WorkflowDependencies):
    """Create and compile the workflow."""
    return build_state_graph(deps).compile()


def run_query_workflow(deps: WorkflowDependencies, initial_state: WorkflowState) -> WorkflowState:
    """Run one question through the workflow and return the final state."""
    return build_query_graph(deps).invoke(initial_state)