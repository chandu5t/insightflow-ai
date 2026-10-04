"""V2 pipeline execution and stage-level evaluation service."""

from __future__ import annotations

import time
from typing import Any
from uuid import UUID

from app.core.config import Settings, get_settings
from app.evaluation.ablations import AblationConfig, InjectedError, apply_error_injection
from app.evaluation.baselines import BaselineARunner, BaselineBRunner, BaselineCRunner, CountingLlmClient
from app.evaluation.errors import classify_error
from app.evaluation.metrics import (
    build_case_metric_records,
    evaluate_numerical_accuracy,
    evaluate_planning_accuracy,
    evaluate_tool_selection_accuracy,
)
from app.evaluation.schemas import (
    CaseEvaluationResult,
    CaseStageResult,
    ErrorRecord,
    EvaluatorInput,
    GroundTruth,
    SystemCondition,
)
from app.multi_agent.schemas import MultiAgentRequest, MultiAgentResult
from app.multi_agent.workflow import run_multi_agent_workflow
from app.planner.schemas import AnalysisPlan, PlanStep, PlannerRequest
from app.planner.service import PlannerModel, create_plan
from app.self_correction.schemas import CorrectionRequest, CorrectionResponse
from app.self_correction.service import correct_execution
from app.services.dataset_repository import DatasetRepository
from app.services.gemini_client import LlmClient
from app.services.metric_retriever import MetricRetriever, StubMetricRetriever
from app.verification.schemas import VerificationRequest, VerificationResult
from app.verification.service import verify_execution
from app.visualization.schemas import VisualizationRequest
from app.visualization.service import visualize


class MockPlannerModel:
    """Deterministic fallback planner model for testing without live LLM APIs."""

    model_name: str = "mock-planner-model"

    def __init__(self, plan_override: AnalysisPlan | None = None) -> None:
        self.plan_override = plan_override

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> str:
        if self.plan_override:
            return self.plan_override.model_dump_json()

        # Read only the question field; registry/context strings are not input intent.
        import json

        try:
            question = json.loads(user_prompt)["question"]
        except (json.JSONDecodeError, KeyError, TypeError):
            question = ""
        q_lower = question.lower()
        if "missing" in q_lower or "percentage" in q_lower or "compare" in q_lower:
            plan = {
                "intent": "unsupported_analysis",
                "reasoning_type": "unsupported",
                "steps": [],
                "unsupported_reason": "The deterministic evaluation planner does not model this request.",
            }
        elif "revenue" in q_lower:
            plan = {
                "intent": "total_revenue",
                "reasoning_type": "simple",
                "steps": [
                    {"step_id": "s1", "operation": "derive_metric", "description": "Calculate revenue", "inputs": ["quantity", "unit_price"], "parameters": {"formula": "quantity * unit_price", "output_name": "revenue"}, "depends_on": []}
                ],
            }
        elif "count" in q_lower or "how many rows" in q_lower or "how many records" in q_lower:
            plan = {
                "intent": "count_orders",
                "reasoning_type": "simple",
                "steps": [
                    {"step_id": "s1", "operation": "count", "description": "Count rows", "inputs": [], "parameters": {}, "depends_on": []}
                ],
            }
        else:
            plan = {
                "intent": "unsupported_analysis",
                "reasoning_type": "unsupported",
                "steps": [],
                "unsupported_reason": "Unsupported query type.",
            }
        return json.dumps(plan)


def evaluate_single_case(
    case_input: EvaluatorInput,
    ground_truth: GroundTruth,
    condition: SystemCondition,
    dataset_id: str | UUID,
    repository: DatasetRepository,
    retriever: MetricRetriever | None = None,
    planner_model: PlannerModel | None = None,
    settings: Settings | None = None,
    error_injection: InjectedError | None = None,
    run_id: str = "run-1",
    numerical_tolerance: float | None = None,
    metric_version: str = "v2.1",
) -> CaseEvaluationResult:
    """Execute and evaluate a single benchmark case under the specified condition."""
    settings = settings or get_settings()
    retriever = retriever or StubMetricRetriever()
    dataset_uuid = UUID(str(dataset_id))
    tolerance = numerical_tolerance if numerical_tolerance is not None else ground_truth.tolerance
    counted_model = CountingLlmClient(planner_model) if planner_model is not None else None

    # ---- 1. Handle Baselines A, B, and C ----
    if condition == SystemCondition.BASELINE_A_LLM_ONLY:
        runner_a = BaselineARunner(repository, settings, llm_client=counted_model, retriever=retriever)
        res = runner_a.run(case_input, dataset_uuid)
        num_ok = evaluate_numerical_accuracy(res["raw_answer"], ground_truth.value, tolerance)
        return _with_metric_records(CaseEvaluationResult(
            case_id=case_input.case_id,
            condition=condition.value,
            run_id=run_id,
            success=res["success"],
            numerical_correct=num_ok,
            latency_ms=res["latency_ms"],
            llm_calls=res["llm_calls"],
            input_tokens=res["input_tokens"],
            output_tokens=res["output_tokens"],
            stage_results=res["stage_results"],
            raw_answer=res["raw_answer"],
        ), metric_version)

    if condition == SystemCondition.BASELINE_B_TOOL_AUGMENTED:
        runner_b = BaselineBRunner(repository, settings, llm_client=counted_model, retriever=retriever)
        res = runner_b.run(case_input, dataset_uuid)
        num_ok = evaluate_numerical_accuracy(res["raw_answer"], ground_truth.value, tolerance)
        result = CaseEvaluationResult(
            case_id=case_input.case_id,
            condition=condition.value,
            run_id=run_id,
            success=res["success"],
            numerical_correct=num_ok,
            latency_ms=res["latency_ms"],
            llm_calls=res["llm_calls"],
            input_tokens=res["input_tokens"],
            output_tokens=res["output_tokens"],
            stage_results=res["stage_results"],
            raw_answer=res["raw_answer"],
        )
        result.metrics["metric_unavailable_reasons"] = {
            "M4": "V1 tool labels and V2.2 operation labels have no frozen canonical mapping."
        }
        return _with_metric_records(result, metric_version)

    if condition == SystemCondition.BASELINE_C_V1:
        runner_c = BaselineCRunner(repository, settings, llm_client=counted_model, retriever=retriever)
        res = runner_c.run(case_input, dataset_uuid)
        num_ok = evaluate_numerical_accuracy(res["raw_answer"], ground_truth.value, tolerance)
        return _with_metric_records(CaseEvaluationResult(
            case_id=case_input.case_id,
            condition=condition.value,
            run_id=run_id,
            success=res["success"],
            numerical_correct=num_ok,
            latency_ms=res["latency_ms"],
            llm_calls=res["llm_calls"],
            input_tokens=res["input_tokens"],
            output_tokens=res["output_tokens"],
            stage_results=res["stage_results"],
            raw_answer=res["raw_answer"],
        ), metric_version)

    # ---- 2. Full V2 and Ablations (A1–A4) ----
    ablation = AblationConfig(
        disable_planner=condition == SystemCondition.ABLATION_A1_NO_PLANNER,
        disable_verification=condition == SystemCondition.ABLATION_A2_NO_VERIFICATION,
        disable_self_correction=condition == SystemCondition.ABLATION_A3_NO_CORRECTION,
        disable_rag=condition == SystemCondition.ABLATION_A4_NO_RAG,
    )

    stage_results: list[CaseStageResult] = []
    errors: list[ErrorRecord] = []
    case_metrics: dict[str, Any] = {}

    total_start = time.perf_counter()
    if counted_model is None and not ablation.disable_planner:
        raise ValueError("V2 evaluation requires an explicitly configured planner model.")
    active_retriever = StubMetricRetriever() if ablation.disable_rag else retriever

    # --- Stage 1: Planning ---
    plan_start = time.perf_counter()
    plan: AnalysisPlan
    plan_correct = None

    if ablation.disable_planner:
        # A1: Bypass planner, create a single direct step
        plan = AnalysisPlan(
            intent="total_revenue",
            reasoning_type="simple",
            steps=[PlanStep(step_id="s1", operation="derive_metric", description="Direct metric", parameters={"metric": "revenue"})],
        )
        plan_lat = (time.perf_counter() - plan_start) * 1000
        direct_ops = [step.operation for step in plan.steps]
        if ground_truth.expected_operations:
            if len(ground_truth.expected_operations) > 1:
                case_metrics["planning_dependency_evaluability"] = "unavailable_no_dependency_ground_truth"
            operation_sequence_correct = evaluate_planning_accuracy(direct_ops, ground_truth.expected_operations)
            case_metrics["planning_operation_sequence_correct"] = operation_sequence_correct
            # The frozen benchmark does not define expected dependency graphs.
            # A single operation has no inter-step dependency to assess.
            if not operation_sequence_correct:
                plan_correct = False
            else:
                plan_correct = True
            case_metrics["tool_selection_correct"] = evaluate_tool_selection_accuracy(
                direct_ops if direct_ops else None, ground_truth.expected_operations
            )
        stage_results.append(CaseStageResult(stage_name="planner", status="skipped", latency_ms=plan_lat, output_summary="bypassed_planner"))
    else:
        try:
            req = PlannerRequest(question=case_input.question)
            resp = create_plan(req, llm_client=counted_model)
            plan = resp.plan
            plan_lat = (time.perf_counter() - plan_start) * 1000
            plan_status = "completed" if resp.valid else "failed"

            # Evaluate planning accuracy
            observed_ops = [step.operation for step in plan.steps]
            if ground_truth.expected_operations:
                if len(ground_truth.expected_operations) > 1:
                    case_metrics["planning_dependency_evaluability"] = "unavailable_no_dependency_ground_truth"
                operation_sequence_correct = evaluate_planning_accuracy(observed_ops, ground_truth.expected_operations)
                case_metrics["planning_operation_sequence_correct"] = operation_sequence_correct
                if not resp.valid or not operation_sequence_correct:
                    plan_correct = False
                else:
                    plan_correct = True
                tool_correct = evaluate_tool_selection_accuracy(observed_ops if observed_ops else None, ground_truth.expected_operations)
                case_metrics["tool_selection_correct"] = tool_correct

            stage_results.append(
                CaseStageResult(
                    stage_name="planner",
                    status=plan_status,
                    latency_ms=plan_lat,
                    output_summary={"intent": plan.intent, "steps": observed_ops},
                    errors=[err.model_dump() for err in resp.validation_errors],
                )
            )
            if not resp.valid:
                errors.append(classify_error("planner", "INVALID_PLAN", "Plan failed validation", case_id=case_input.case_id))
        except Exception as exc:
            plan_lat = (time.perf_counter() - plan_start) * 1000
            stage_results.append(CaseStageResult(stage_name="planner", status="failed", latency_ms=plan_lat, output_summary=str(exc)))
            errors.append(classify_error("planner", "PLANNER_FAILURE", str(exc), case_id=case_input.case_id))
            return _with_metric_records(CaseEvaluationResult(
                case_id=case_input.case_id,
                condition=condition.value,
                run_id=run_id,
                success=False,
                latency_ms=(time.perf_counter() - total_start) * 1000,
                llm_calls=counted_model.call_count if counted_model else 0,
                input_tokens=counted_model.input_tokens if counted_model else 0,
                output_tokens=counted_model.output_tokens if counted_model else 0,
                stage_results=stage_results,
                errors=errors,
            ), metric_version)

    case_metrics["planning_latency_ms"] = plan_lat

    # --- Stage 2: Multi-Agent Execution ---
    exec_start = time.perf_counter()
    exec_request = MultiAgentRequest(
        question=case_input.question,
        dataset_id=dataset_uuid,
        analysis_plan=plan,
    )
    multi_agent_res = run_multi_agent_workflow(exec_request, repository, active_retriever)
    exec_lat = (time.perf_counter() - exec_start) * 1000
    case_metrics["analysis_latency_ms"] = exec_lat

    stage_results.append(
        CaseStageResult(
            stage_name="execution",
            status="completed" if multi_agent_res.workflow_status == "completed" else "failed",
            latency_ms=exec_lat,
            output_summary=multi_agent_res.model_dump(mode="json"),
            errors=[err.model_dump() for err in multi_agent_res.errors],
        )
    )
    for workflow_error in multi_agent_res.errors:
        if workflow_error.category == "invalid_plan":
            error_stage, error_code = "planner", workflow_error.code
        elif workflow_error.category == "invalid_route":
            error_stage, error_code = "execution", "INVALID_OPERATION"
        else:
            error_stage, error_code = "execution", "TOOL_EXECUTION_FAILED"
        errors.append(classify_error(
            error_stage, error_code, workflow_error.message, case_id=case_input.case_id,
            expected="The validated plan executes through the controlled V2.3 workflow.",
            observed=workflow_error.category, evidence=workflow_error.code,
        ))

    # --- Optional Controlled Error Injection (for H3 / H4 testing) ---
    if error_injection:
        plan, multi_agent_res, applied = apply_error_injection(plan, multi_agent_res, error_injection)
        case_metrics["injected_errors_count"] = 1 if applied else 0
        case_metrics["injection_record"] = {
            **error_injection.model_dump(mode="json"),
            "applied": applied,
        }

    # --- Stage 3: Verification ---
    verif_start = time.perf_counter()
    verif_res: VerificationResult | None = None
    verification_status = None

    if ablation.disable_verification:
        verif_lat = None
        stage_results.append(CaseStageResult(stage_name="verification", status="skipped", latency_ms=0.0))
    else:
        verif_res = verify_execution(plan, multi_agent_res, repository=repository, dataset_id=dataset_uuid)
        verif_lat = (time.perf_counter() - verif_start) * 1000
        verification_status = verif_res.status

        # Evaluate M5: Verification Detection Rate on injected errors
        if error_injection and case_metrics.get("injected_errors_count", 0) > 0:
            detected = any(
                check.status == "failed"
                and error_injection.target_step_id is not None
                and error_injection.target_step_id in check.step_ids
                for check in verif_res.checks
            )
            case_metrics["detected_injected_errors_count"] = 1 if detected else 0
            case_metrics["injection_record"]["detected_by_verification"] = detected

        stage_results.append(
            CaseStageResult(
                stage_name="verification",
            status="completed" if verif_res.status == "passed" else "failed" if verif_res.status == "failed" else "unsupported",
                latency_ms=verif_lat,
                output_summary=verif_res.model_dump(mode="json"),
                errors=[err.model_dump() for err in verif_res.errors],
            )
        )
        if verif_res.status == "failed":
            errors.append(classify_error("verification", "VERIFICATION_FAILED", "Verification did not pass", case_id=case_input.case_id))
        elif verif_res.status == "unsupported":
            errors.append(classify_error(
                "verification", "UNSUPPORTED_QUESTION", "A required verification check was unsupported.",
                case_id=case_input.case_id, observed="unsupported",
                evidence=",".join(check.name for check in verif_res.unsupported_checks),
            ))

    case_metrics["verification_latency_ms"] = verif_lat
    if error_injection and case_metrics.get("injected_errors_count", 0) and "detected_injected_errors_count" not in case_metrics:
        case_metrics["detected_injected_errors_count"] = 0

    # --- Stage 4: Self-Correction ---
    corr_start = time.perf_counter()
    final_plan = plan
    final_execution_res = multi_agent_res
    final_verification_res = verif_res
    correction_res: CorrectionResponse | None = None
    correction_attempts_used = 0

    if verif_res and verif_res.status == "failed":
        case_metrics["detected_errors_count"] = 1

    if not ablation.disable_self_correction and verif_res and verif_res.status == "failed":
        corr_req = CorrectionRequest(
            question=case_input.question,
            dataset_id=dataset_uuid,
            analysis_plan=plan,
            execution_result=multi_agent_res,
            verification_result=verif_res,
            correction_budget=2,
        )
        corr_res = correct_execution(corr_req, repository, active_retriever)
        correction_res = corr_res
        corr_lat = (time.perf_counter() - corr_start) * 1000
        correction_attempts_used = corr_res.attempts_used

        if corr_res.terminal_status in {"corrected", "not_corrected"}:
            final_plan = corr_res.final_plan
            final_execution_res = corr_res.final_execution_result
            final_verification_res = corr_res.final_verification_result
            if corr_res.terminal_status == "corrected" and final_verification_res.status == "passed":
                case_metrics["corrected_errors_count"] = 1
            stage_results.append(CaseStageResult(
                stage_name="verification",
                status="completed" if final_verification_res.status == "passed" else "failed" if final_verification_res.status == "failed" else "unsupported",
                output_summary={"phase": "post_correction", "status": final_verification_res.status,
                                "verified_steps": final_verification_res.verified_steps},
                errors=[error.model_dump() for error in final_verification_res.errors],
                metadata={"correction_attempts": correction_attempts_used},
            ))

        for correction_error in corr_res.errors:
            code = correction_error.code.upper()
            stage = "diagnosis" if "DIAGNOSIS" in code else "self_correction"
            errors.append(classify_error(
                stage, code, correction_error.message, case_id=case_input.case_id,
                expected="Safe correction outcome under V2.5.",
                observed=corr_res.terminal_status,
                evidence=correction_error.code,
            ))

        stage_results.append(
            CaseStageResult(
                stage_name="self_correction",
                status="completed" if corr_res.terminal_status == "corrected" else "failed",
                latency_ms=corr_lat,
                output_summary=corr_res.model_dump(mode="json"),
                errors=[err.model_dump() for err in corr_res.errors],
            )
        )
    else:
        corr_lat = None
        if not ablation.disable_self_correction:
            stage_results.append(CaseStageResult(stage_name="self_correction", status="skipped", latency_ms=0.0))

    case_metrics["correction_latency_ms"] = corr_lat

    # --- Stage 5: Visualization ---
    if final_verification_res and final_verification_res.status == "passed":
        viz_start = time.perf_counter()
        viz_req = VisualizationRequest(
            question=case_input.question,
            dataset_id=dataset_uuid,
            analysis_plan=final_plan,
            execution_result=final_execution_res,
            verification_result=final_verification_res,
            correction_result=correction_res,
        )
        viz_res = visualize(viz_req)
        viz_lat = (time.perf_counter() - viz_start) * 1000
        stage_results.append(
            CaseStageResult(
                stage_name="visualization",
                status="completed" if viz_res.status == "generated" else "unsupported",
                latency_ms=viz_lat,
                output_summary=viz_res.model_dump(mode="json"),
            )
        )

    # --- Final Evaluator Outcome ---
    total_lat = (time.perf_counter() - total_start) * 1000
    final_output = final_execution_res.final_result
    num_correct = evaluate_numerical_accuracy(final_output, ground_truth.value, tolerance)

    # Claim-level hallucination/grounding requires an independently annotated
    # claim protocol. A numeric mismatch alone is not evidence of hallucination.
    case_metrics["explanations_evaluated_count"] = 0
    case_metrics["grounded_explanations_count"] = 0
    case_metrics["claims_evaluated_count"] = 0
    case_metrics["unsupported_claims_count"] = 0

    overall_success = final_execution_res.workflow_status == "completed" and (
        final_verification_res is None or final_verification_res.status == "passed"
    )

    return _with_metric_records(CaseEvaluationResult(
        case_id=case_input.case_id,
        condition=condition.value,
        run_id=run_id,
        success=overall_success,
        numerical_correct=num_correct,
        plan_correct=plan_correct,
        verification_status=final_verification_res.status if final_verification_res else None,
        correction_attempts=correction_attempts_used,
        latency_ms=total_lat,
        llm_calls=counted_model.call_count if counted_model else 0,
        input_tokens=counted_model.input_tokens if counted_model else 0,
        output_tokens=counted_model.output_tokens if counted_model else 0,
        metrics=case_metrics,
        errors=errors,
        stage_results=stage_results,
        raw_answer=final_output,
    ), metric_version)


def _with_metric_records(result: CaseEvaluationResult, metric_version: str) -> CaseEvaluationResult:
    result.metric_records = build_case_metric_records(result, metric_version)
    return result
