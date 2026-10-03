"""Deterministic V2.2 operation and V2.3 agent routing."""

from app.multi_agent.schemas import AgentName
from app.planner.operations import OPERATION_REGISTRY

AGENT_REGISTRY: dict[AgentName, str] = {
    "data_understanding": "data_understanding",
    "analysis": "analysis",
    "knowledge": "knowledge",
}
OPERATION_AGENT: dict[str, AgentName] = {operation: "analysis" for operation in OPERATION_REGISTRY}


def agent_for_operation(operation: str) -> AgentName:
    try:
        agent = OPERATION_AGENT[operation]
    except KeyError as exc:
        raise ValueError(f"Unknown V2.2 operation: {operation}") from exc
    if agent not in AGENT_REGISTRY:
        raise ValueError(f"Unknown V2.3 agent: {agent}")
    return agent


def registered_agent_name(agent: str) -> AgentName:
    """Validate an agent identifier against the exact controlled registry."""
    if agent not in AGENT_REGISTRY:
        raise ValueError(f"Unknown V2.3 agent: {agent}")
    return agent  # type: ignore[return-value]
