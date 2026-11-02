"""Composition of the existing V2.2–V2.6 services for the dashboard demo."""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from app.core.errors import AppError
from app.multi_agent.schemas import MultiAgentRequest
from app.planner.context import build_planner_context
from app.planner.schemas import PlannerResponse
from app.planner.service import PlannerModel, create_plan
from app.self_correction.schemas import CorrectionRequest, CorrectionResponse
from app.self_correction.service import correct_execution
from app.services.dataset_repository import DatasetRepository
from app.services.metric_retriever import MetricRetriever
from app.verification.schemas import VerificationResult
from app.verification.service import verify_execution
from app.visualization.schemas import VisualizationRequest, VisualizationResponse
from app.visualization.service import visualize
from app.multi_agent.workflow import run_multi_agent_workflow

logger = logging.getLogger(__name__)


def _safe_error(stage: str, exc: Exception) -> dict[str, str]:
    """Return actionable exception identity without leaking request/provider data."""
    if isinstance(exc, AppError):
        category = exc.details.get("failure_category")
        logger.error(
            "demo_workflow_stage_failed stage=%s code=%s category=%s",
            stage,
            exc.code.value,
            category if isinstance(category, str) else "unclassified",
        )
        return {
            "stage": stage,
            "code": exc.code.value,
            "message": exc.message,
            **({"category": str(category)} if isinstance(category, str) else {}),
            **_safe_provider_diagnostics(exc),
        }
    logger.error(
        "demo_workflow_stage_failed stage=%s exception_type=%s",
        stage,
        type(exc).__name__,
    )
    return {
        "stage": stage,
        "code": f"{stage.upper()}_FAILED",
        "message": f"{stage.replace('_', ' ').capitalize()} failed ({type(exc).__name__}). Check the service configuration and retry.",
    }


def _safe_provider_diagnostics(exc: AppError) -> dict[str, str]:
    """Forward only the planner's allowlisted status/category/class metadata."""
    diagnostics = exc.details.get("provider_diagnostics")
    if not isinstance(diagnostics, dict):
        return {}
    safe: dict[str, str] = {}
    category = diagnostics.get("category")
    if isinstance(category, str) and category.isidentifier():
        safe["provider_category"] = category
    status = diagnostics.get("http_status")
    if isinstance(status, int) and not isinstance(status, bool) and 100 <= status <= 599:
        safe["provider_http_status"] = str(status)
    exception_types = diagnostics.get("exception_types")
    if (
        isinstance(exception_types, list)
        and len(exception_types) <= 6
        and all(isinstance(name, str) and name.isidentifier() for name in exception_types)
    ):
        safe["provider_exception_types"] = ">".join(exception_types)
    return safe


def run_demo_workflow(
    *,
    question: str,
    dataset_id: UUID,
    repository: DatasetRepository,
    retriever: MetricRetriever,
    planner_model: PlannerModel,
) -> dict[str, Any]:
    """Run planning, controlled execution, verification, bounded correction, then visualization.

    This composes the existing versioned services. It does not access evaluation labels,
    alter the V1 route, or move orchestration into the visualization endpoint.
    """
    response: dict[str, Any] = {
        "status": "failed",
        "question": question,
        "dataset_id": str(dataset_id),
        "stages": {
            "planning": "pending",
            "execution": "pending",
            "verification": "pending",
            "correction": "not_needed",
            "visualization": "not_ready",
        },
        "planner_response": None,
        "analysis_plan": None,
        "execution_result": None,
        "verification_result": None,
        "correction_result": None,
        "final_plan": None,
        "final_execution_result": None,
        "final_verification_result": None,
        "visualization_result": None,
        "errors": [],
    }

    try:
        context = build_planner_context(question, repository.get_csv_path(dataset_id))
        dataset_context = getattr(context.request, "dataset_context", None)
        has_trusted_semantics = bool(
            dataset_context
            and (
                dataset_context.manifest_version
                or any(role != "unknown" for role in dataset_context.semantic_roles.values())
            )
        )
        planned: PlannerResponse = create_plan(
            context.request,
            llm_client=planner_model,
            # Match POST /analysis/plan: an unregistered dataset has no authority
            # to assert registry metadata, so pass None and validate its physical
            # schema only. Trusted manifests still require explicit registry validation.
            authorized_metric_definitions=(
                context.authorized_metric_definitions if has_trusted_semantics else None
            ),
        )
        response["planner_response"] = planned.model_dump(mode="json")
        response["analysis_plan"] = planned.plan.model_dump(mode="json")
        if not planned.valid:
            response["stages"]["planning"] = "failed"
            response["errors"] = [
                {"stage": "planning", "code": error.code, "message": error.message}
                for error in planned.validation_errors
            ]
            return response
        response["stages"]["planning"] = (
            "unsupported" if planned.plan.intent == "unsupported_analysis" else "completed"
        )
        response["final_plan"] = planned.plan.model_dump(mode="json")
        if planned.plan.intent == "unsupported_analysis":
            response["status"] = "unsupported"
            response["errors"] = [{
                "stage": "planning", "code": "UNSUPPORTED_ANALYSIS",
                "message": planned.plan.unsupported_reason or "The requested analysis is unsupported.",
            }]
            return response
    except Exception as exc:
        response["stages"]["planning"] = "failed"
        response["errors"] = [_safe_error("planning", exc)]
        return response

    plan = planned.plan
    original_plan = plan
    try:
        execution = run_multi_agent_workflow(
            MultiAgentRequest(question=question, dataset_id=dataset_id, analysis_plan=plan),
            repository,
            retriever,
        )
        response["execution_result"] = execution.model_dump(mode="json")
        response["final_execution_result"] = execution.model_dump(mode="json")
        original_execution = execution
        response["stages"]["execution"] = (
            "completed" if execution.workflow_status == "completed" else "failed"
        )
        if execution.errors:
            response["errors"].extend(
                {"stage": "execution", **error.model_dump(mode="json")}
                for error in execution.errors
            )
    except Exception as exc:
        response["stages"]["execution"] = "failed"
        response["errors"].append(_safe_error("execution", exc))
        return response

    try:
        verification: VerificationResult = verify_execution(
            plan, execution, repository=repository, dataset_id=dataset_id
        )
        response["verification_result"] = verification.model_dump(mode="json")
        response["final_verification_result"] = verification.model_dump(mode="json")
        original_verification = verification
        response["stages"]["verification"] = verification.status
        response["errors"].extend(
            {"stage": "verification", **error.model_dump(mode="json")}
            for error in verification.errors
        )
    except Exception as exc:
        response["stages"]["verification"] = "failed"
        response["errors"].append(_safe_error("verification", exc))
        return response

    if verification.status == "failed":
        response["stages"]["correction"] = "running"
        try:
            correction: CorrectionResponse = correct_execution(
                CorrectionRequest(
                    question=question,
                    dataset_id=dataset_id,
                    analysis_plan=plan,
                    execution_result=execution,
                    verification_result=verification,
                    correction_budget=2,
                ),
                repository,
                retriever,
            )
            response["correction_result"] = correction.model_dump(mode="json")
            response["stages"]["correction"] = correction.terminal_status
            response["final_plan"] = correction.final_plan.model_dump(mode="json")
            response["final_execution_result"] = correction.final_execution_result.model_dump(mode="json")
            response["final_verification_result"] = correction.final_verification_result.model_dump(mode="json")
            response["stages"]["verification"] = correction.final_verification_result.status
            response["errors"].extend(
                {"stage": "correction", **error.model_dump(mode="json")}
                for error in correction.errors
            )
            plan = correction.final_plan
            execution = correction.final_execution_result
            verification = correction.final_verification_result
        except Exception as exc:
            response["stages"]["correction"] = "failed"
            response["errors"].append(_safe_error("correction", exc))
            return response

    if verification.status != "passed" or execution.workflow_status != "completed":
        response["status"] = "unsupported" if verification.status == "unsupported" else "failed"
        response["stages"]["visualization"] = "blocked_by_verification"
        return response

    try:
        visualization: VisualizationResponse = visualize(VisualizationRequest(
            question=question,
            dataset_id=dataset_id,
            analysis_plan=original_plan.model_dump(mode="json"),
            execution_result=original_execution.model_dump(mode="json"),
            verification_result=original_verification.model_dump(mode="json"),
            correction_result=(
                response["correction_result"]
                if response["correction_result"] is not None else None
            ),
        ))
        response["visualization_result"] = visualization.model_dump(mode="json")
        response["stages"]["visualization"] = visualization.status
        if visualization.status == "generated":
            response["status"] = "completed"
        else:
            response["status"] = "completed"
            response["errors"].extend(
                {"stage": "visualization", **error.model_dump(mode="json")}
                for error in visualization.errors
            )
    except Exception as exc:
        response["status"] = "completed"
        response["stages"]["visualization"] = "failed"
        response["errors"].append(_safe_error("visualization", exc))
    return response
