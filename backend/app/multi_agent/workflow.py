"""Separate, sequential LangGraph workflow for V2.3 multi-agent execution."""

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.multi_agent.agents import AnalysisAgent, DataUnderstandingAgent, KnowledgeAgent
from app.multi_agent.routing import AGENT_REGISTRY, agent_for_operation
from app.multi_agent.schemas import (
    AgentResult, MultiAgentRequest, MultiAgentResult, WorkflowError, WorkflowState,
)
from app.planner.validator import validate_plan
from app.services.dataset_repository import DatasetRepository
from app.services.metric_retriever import MetricRetriever


class V23Workflow:
    def __init__(self, repository: DatasetRepository, retriever: MetricRetriever) -> None:
        self.data_agent = DataUnderstandingAgent(repository)
        self.analysis_agent = AnalysisAgent()
        self.knowledge_agent = KnowledgeAgent(retriever)
        self.agent_registry = {
            "data_understanding": self.data_agent,
            "analysis": self.analysis_agent,
            "knowledge": self.knowledge_agent,
        }
        if set(self.agent_registry) != set(AGENT_REGISTRY):
            raise ValueError("V2.3 agent registry does not match the controlled registry.")

    def initial_state(self, request: MultiAgentRequest) -> WorkflowState:
        return {
            "question": request.question,
            "dataset_id": request.dataset_id,
            "analysis_plan": request.analysis_plan,
            "current_step": None,
            "completed_steps": [],
            "step_statuses": {step.step_id: "pending" for step in request.analysis_plan.steps},
            "step_results": {},
            "agent_outputs": [],
            "workflow_status": "pending",
            "errors": [],
            "routing_agent": None,
        }

    def prepare(self, state: WorkflowState) -> dict[str, Any]:
        result, updates = self.data_agent.run(state)
        if result.status == "failed":
            return {**updates, "agent_outputs": [result], "errors": [result.error], "workflow_status": "failed"}
        return {**updates, "agent_outputs": [result], "workflow_status": "running"}

    def supervisor(self, state: WorkflowState) -> dict[str, Any]:
        if state["workflow_status"] == "failed":
            return {"current_step": None, "routing_agent": None}
        plan = state["analysis_plan"]
        index = len(state["completed_steps"])
        if index < len(plan.steps):
            step = plan.steps[index]
            try:
                agent = agent_for_operation(step.operation)
            except ValueError:
                error = WorkflowError(category="invalid_route", code="INVALID_ROUTE",
                                      message="The plan step has no registered agent route.", step_id=step.step_id)
                statuses = {**state["step_statuses"], step.step_id: "failed"}
                return {"current_step": None, "routing_agent": None, "step_statuses": statuses,
                        "workflow_status": "failed", "errors": [*state["errors"], error]}
            statuses = {**state["step_statuses"], step.step_id: "running"}
            return {"current_step": step.step_id, "workflow_status": "running", "routing_agent": agent,
                    "step_statuses": statuses}
        if plan.intent == "metric_definition":
            return {"current_step": "__knowledge__", "routing_agent": "knowledge"}
        return {"current_step": None, "routing_agent": None}

    def analysis(self, state: WorkflowState) -> dict[str, Any]:
        step = next((item for item in state["analysis_plan"].steps if item.step_id == state["current_step"]), None)
        if step is None:
            error = WorkflowError(category="invalid_state", code="CURRENT_STEP_MISSING",
                                  message="The current plan step is unavailable.")
            return {"workflow_status": "failed", "errors": [*state["errors"], error]}
        result, raw_result = self.analysis_agent.run(step, state)
        if result.status == "failed":
            statuses = {**state["step_statuses"], step.step_id: "failed"}
            return {"agent_outputs": [*state["agent_outputs"], result], "workflow_status": "failed",
                    "errors": [*state["errors"], result.error], "step_statuses": statuses}
        return {
            "agent_outputs": [*state["agent_outputs"], result],
            "step_results": {**state["step_results"], step.step_id: raw_result},
            "completed_steps": [*state["completed_steps"], step.step_id],
            "step_statuses": {**state["step_statuses"], step.step_id: "completed"},
        }

    def knowledge(self, state: WorkflowState) -> dict[str, Any]:
        if state["analysis_plan"].intent != "metric_definition":
            error = WorkflowError(category="invalid_route", code="INVALID_ROUTE",
                                  message="Knowledge retrieval was not requested by structured plan context.")
            return {"workflow_status": "failed", "errors": [*state["errors"], error]}
        result = self.knowledge_agent.run(state)
        if result.status == "failed":
            return {"agent_outputs": [*state["agent_outputs"], result], "workflow_status": "failed",
                    "errors": [*state["errors"], result.error]}
        return {"agent_outputs": [*state["agent_outputs"], result], "final_result": result.result}

    @staticmethod
    def after_prepare(state: WorkflowState) -> str:
        return "finish" if state["workflow_status"] == "failed" else "supervisor"

    @staticmethod
    def after_supervisor(state: WorkflowState) -> str:
        if state["workflow_status"] == "failed":
            return "finish"
        if state.get("routing_agent") == "analysis":
            return "analysis"
        if state.get("routing_agent") == "knowledge":
            return "knowledge"
        return "finish"

    @staticmethod
    def after_analysis(state: WorkflowState) -> str:
        return "finish" if state["workflow_status"] == "failed" else "supervisor"

    def build_graph(self):
        graph = StateGraph(WorkflowState)
        graph.add_node("prepare", self.prepare)
        graph.add_node("supervisor", self.supervisor)
        graph.add_node("analysis", self.analysis)
        graph.add_node("knowledge", self.knowledge)
        graph.add_node("finish", self.finish)
        graph.add_edge(START, "prepare")
        graph.add_conditional_edges("prepare", self.after_prepare, {"supervisor": "supervisor", "finish": "finish"})
        graph.add_conditional_edges("supervisor", self.after_supervisor,
                                   {"analysis": "analysis", "knowledge": "knowledge", "finish": "finish"})
        graph.add_conditional_edges("analysis", self.after_analysis, {"supervisor": "supervisor", "finish": "finish"})
        graph.add_edge("knowledge", "finish")
        graph.add_edge("finish", END)
        return graph.compile()

    @staticmethod
    def finish(state: WorkflowState) -> dict[str, Any]:
        if state["workflow_status"] != "failed":
            return {"workflow_status": "completed"}
        return {}

    def run(self, request: MultiAgentRequest) -> MultiAgentResult:
        errors = validate_plan(request.analysis_plan)
        if errors:
            return MultiAgentResult(
                workflow_status="failed",
                errors=[WorkflowError(category="invalid_plan", code=item.code, message=item.message,
                                      step_id=item.step_id) for item in errors],
            )
        current_state = self.initial_state(request)
        try:
            for current_state in self.build_graph().stream(current_state, stream_mode="values"):
                pass
        except Exception:
            statuses = dict(current_state["step_statuses"])
            current_step = current_state.get("current_step")
            if current_step in statuses and statuses[current_step] == "running":
                statuses[current_step] = "failed"
            return MultiAgentResult(
                workflow_status="failed",
                executed_steps=current_state["completed_steps"],
                step_statuses=statuses,
                agent_results=current_state["agent_outputs"],
                errors=[WorkflowError(category="workflow_failure", code="GRAPH_EXECUTION_FAILED",
                                      message="The multi-agent workflow could not be completed.")],
            )
        final: WorkflowState = current_state
        outputs = final["agent_outputs"]
        if final.get("final_result") is None:
            analysis_outputs = [item for item in outputs if item.agent_name == "analysis" and item.status == "completed"]
            final_result = analysis_outputs[-1].result if analysis_outputs else None
        else:
            final_result = final["final_result"]
        return MultiAgentResult(
            workflow_status="failed" if final["workflow_status"] == "failed" else "completed",
            executed_steps=final["completed_steps"],
            step_statuses=final["step_statuses"],
            agent_results=outputs,
            final_result=final_result,
            errors=[error for error in final["errors"] if error is not None],
        )


def run_multi_agent_workflow(request: MultiAgentRequest, repository: DatasetRepository,
                             retriever: MetricRetriever) -> MultiAgentResult:
    return V23Workflow(repository, retriever).run(request)
