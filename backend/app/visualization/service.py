"""V2.6 validation and deterministic visualization-spec generation."""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from pydantic import ValidationError

from app.multi_agent.schemas import MultiAgentResult
from app.planner.schemas import AnalysisPlan
from app.planner.validator import validate_plan
from app.self_correction.schemas import CorrectionResponse
from app.verification.schemas import VerificationResult
from app.visualization.eligibility import UnsafeVisualizationData, UnsupportedVisualization, select_chart
from app.visualization.schemas import (
    VisualizationAxis,
    VisualizationError,
    VisualizationRequest,
    VisualizationResponse,
    VisualizationSeries,
    VisualizationSpec,
    VisualizationTraceability,
)

_EXECUTION_FIELDS = {"workflow_status", "executed_steps", "step_statuses", "agent_results", "final_result", "errors"}
_AGENT_FIELDS = {"agent_name", "step_id", "status", "result", "metadata", "error"}


class ArtifactMismatch(ValueError):
    pass


def _error(category: str, code: str, message: str) -> VisualizationError:
    return VisualizationError(category=category, code=code, message=message)


def _failed(category: str, code: str, message: str) -> VisualizationResponse:
    return VisualizationResponse(status="failed", errors=[_error(category, code, message)])


def _unsupported(code: str, message: str, category: str = "unsupported_visualization") -> VisualizationResponse:
    return VisualizationResponse(status="unsupported", errors=[_error(category, code, message)])


def _require_execution_fields(raw: Any) -> None:
    if isinstance(raw, MultiAgentResult):
        raw = raw.model_dump(mode="python")
    if not isinstance(raw, dict) or not _EXECUTION_FIELDS.issubset(raw):
        raise ValueError("The execution result is missing required V2.3 fields.")
    agents = raw.get("agent_results")
    if not isinstance(agents, list) or any(not isinstance(item, dict) or not _AGENT_FIELDS.issubset(item) for item in agents):
        raise ValueError("The execution result contains a malformed agent result.")


def _dump(model: Any) -> Any:
    return model.model_dump(mode="json")


def _validate_artifacts(request: VisualizationRequest) -> tuple[
    AnalysisPlan, MultiAgentResult, VerificationResult, str | None
]:
    plan = AnalysisPlan.model_validate(request.analysis_plan)
    execution = MultiAgentResult.model_validate(request.execution_result)
    verification = VerificationResult.model_validate(request.verification_result)
    _require_execution_fields(request.execution_result)
    if validate_plan(plan):
        raise ValueError("The supplied plan is not valid under V2.2 validation.")
    correction_status: str | None = None

    if request.correction_result is None:
        selected_plan, selected_execution, selected_verification = plan, execution, verification
    else:
        correction = CorrectionResponse.model_validate(request.correction_result)
        _require_execution_fields(request.execution_result)
        _require_execution_fields(correction.final_execution_result)
        if request.question != correction.question or request.dataset_id != correction.dataset_id:
            raise ArtifactMismatch("Request question or dataset ID does not match correction traceability.")
        if (
            _dump(plan) != _dump(correction.original_plan)
            or _dump(execution) != _dump(correction.original_execution_result)
            or _dump(verification) != _dump(correction.original_verification_result)
        ):
            raise ArtifactMismatch("Top-level original artifacts do not match the correction response originals.")
        selected_plan = correction.final_plan
        selected_execution = correction.final_execution_result
        selected_verification = correction.final_verification_result
        correction_status = correction.terminal_status
        if correction_status not in {"corrected", "not_corrected"}:
            raise UnsupportedVisualization(
                "CORRECTION_STATUS_UNSUITABLE", "The correction result is valid but is not eligible for visualization."
            )

    if validate_plan(selected_plan):
        raise ValueError("The authoritative plan is not valid under V2.2 validation.")
    if selected_verification.status != "passed":
        raise UnsupportedVisualization(
            "VERIFICATION_NOT_PASSED", "Only an authoritative V2.4 passed result can be visualized."
        )
    if selected_execution.workflow_status != "completed" or selected_execution.errors:
        raise ArtifactMismatch("The authoritative execution is not a clean completed workflow.")
    plan_steps = {step.step_id: step for step in selected_plan.steps}
    plan_ids = list(plan_steps)
    if selected_execution.executed_steps != plan_ids:
        raise ArtifactMismatch("The executed step order does not match the authoritative plan.")
    if set(selected_execution.step_statuses) != set(plan_steps):
        raise ArtifactMismatch("The execution step-status keys do not match the authoritative plan.")
    if len(set(selected_execution.executed_steps)) != len(selected_execution.executed_steps):
        raise ArtifactMismatch("The execution result contains duplicate step IDs.")
    if any(step_id not in plan_steps for step_id in selected_execution.executed_steps):
        raise ArtifactMismatch("The execution contains a step absent from the authoritative plan.")
    if any(selected_execution.step_statuses.get(step.step_id) != "completed" for step in selected_plan.steps):
        raise ArtifactMismatch("The authoritative plan and execution step statuses do not agree.")
    if any(step.step_id not in selected_verification.verified_steps for step in selected_plan.steps):
        raise ArtifactMismatch("The authoritative verification does not list every planned step as verified.")
    for step in selected_plan.steps:
        matching = [
            agent for agent in selected_execution.agent_results
            if agent.agent_name == "analysis" and agent.step_id == step.step_id and agent.status == "completed"
        ]
        if len(matching) != 1 or matching[0].metadata.get("operation") != step.operation:
            raise ArtifactMismatch("A planned step does not have exactly one matching successful analysis result.")
    return selected_plan, selected_execution, selected_verification, correction_status


def visualize(request: VisualizationRequest) -> VisualizationResponse:
    """Use the existing verified output only; never load or calculate dataset data."""
    try:
        plan, execution, verification, correction_status = _validate_artifacts(request)
    except UnsupportedVisualization as exc:
        return _unsupported(exc.code, str(exc), exc.category)
    except ArtifactMismatch as exc:
        return _failed("visualization_data_mismatch", "ARTIFACT_MISMATCH", str(exc))
    except (ValidationError, TypeError, ValueError):
        return _failed(
            "invalid_visualization_input", "INVALID_VISUALIZATION_INPUT",
            "The supplied analytical artifacts are malformed or fail V2.2 plan validation.",
        )

    try:
        selected = select_chart(plan, execution, verification.verified_steps)
        step = selected["step"]
        chart_type = selected["chart_type"]
        x_field, y_field = selected["x_field"], selected["y_field"]
        rows = selected["rows"]
        x_type = "temporal" if selected.get("ordering") == "explicit_time_values" else (
            "ordered" if selected.get("ordered") else "categorical" if chart_type in {"bar", "pie"} else "numeric"
        )
        y_type = "numeric" if chart_type != "scatter" else "numeric"
        spec = VisualizationSpec(
            visualization_id=f"viz-{uuid4()}",
            chart_type=chart_type,
            title=step.description,
            description=f"{chart_type.title()} visualization of the verified {step.operation} result.",
            source_step_id=step.step_id,
            x_axis=VisualizationAxis(field=x_field, label=x_field, data_type=x_type),
            y_axis=VisualizationAxis(field=y_field, label=y_field, data_type=y_type),
            series=[VisualizationSeries(name=field, field=field) for field in ([x_field, y_field] if chart_type == "scatter" else [y_field])],
            data=rows,
            metadata={
                "chart_type": chart_type,
                "source_operation": step.operation,
                "source_step_id": step.step_id,
                "input_row_count": None,
                "output_row_count": len(rows),
                "eligibility_decision": "eligible",
                "transformation_metadata": selected.get("ordering"),
                "verification_status": verification.status,
                "correction_status": correction_status,
                "part_to_whole_id": selected.get("whole_id"),
                "part_to_whole_whole_value": selected.get("whole_value"),
            },
            traceability=VisualizationTraceability(
                question=request.question,
                dataset_id=request.dataset_id,
                source_step_id=step.step_id,
                source_operation=step.operation,
                verification_status=verification.status,
                correction_status=correction_status,
            ),
        )
        return VisualizationResponse(
            status="generated",
            visualization=spec,
            metadata={"source_step_id": step.step_id, "chart_type": chart_type},
        )
    except UnsupportedVisualization as exc:
        return _unsupported(exc.code, str(exc), exc.category)
    except UnsafeVisualizationData:
        return _failed(
            "visualization_data_mismatch", "UNSAFE_RESULT_VALUE",
            "A result value cannot be represented safely in the visualization specification.",
        )
    except (ValidationError, TypeError, ValueError):
        return _failed(
            "invalid_visualization_schema", "INVALID_VISUALIZATION_SCHEMA",
            "The visualization specification could not be validated.",
        )
    except Exception:
        return _failed(
            "visualization_generation_failure", "VISUALIZATION_GENERATION_FAILURE",
            "Visualization generation failed safely.",
        )
