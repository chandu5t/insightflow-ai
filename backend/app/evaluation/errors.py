"""Official V2.1 Error Taxonomy (E1–E15) matching Contract v1.1 §10."""

from __future__ import annotations

from enum import Enum
from typing import Any

from app.evaluation.schemas import ErrorRecord


class ErrorCategory(str, Enum):
    """The 15 official V2.1 research error categories."""

    E1_INTENT_MISUNDERSTANDING = "E1_intent_misunderstanding"
    E2_SCHEMA_MISUNDERSTANDING = "E2_schema_misunderstanding"
    E3_METRIC_DEFINITION_ERROR = "E3_metric_definition_error"
    E4_WRONG_COLUMN_SELECTION = "E4_wrong_column_selection"
    E5_WRONG_ANALYTICAL_OPERATION = "E5_wrong_analytical_operation"
    E6_NUMERICAL_CALCULATION_ERROR = "E6_numerical_calculation_error"
    E7_INTERMEDIATE_RESULT_ERROR = "E7_intermediate_result_error"
    E8_FILTERING_RANKING_ERROR = "E8_filtering_ranking_error"
    E9_TOOL_SELECTION_ERROR = "E9_tool_selection_error"
    E10_VERIFICATION_FAILURE = "E10_verification_failure"
    E11_ERROR_DIAGNOSIS_FAILURE = "E11_error_diagnosis_failure"
    E12_CORRECTION_FAILURE = "E12_correction_failure"
    E13_EXPLANATION_GROUNDING_FAILURE = "E13_explanation_grounding_failure"
    E14_UNSUPPORTED_ANSWER_FAILURE = "E14_unsupported_answer_failure"
    E15_SYSTEM_TOOL_EXECUTION_FAILURE = "E15_system_tool_execution_failure"


_CODE_MAP: dict[str, ErrorCategory] = {
    # Planner / Intent / Schema
    "INVALID_INTENT": ErrorCategory.E1_INTENT_MISUNDERSTANDING,
    "UNSUPPORTED_QUESTION": ErrorCategory.E14_UNSUPPORTED_ANSWER_FAILURE,
    "SCHEMA_ERROR": ErrorCategory.E2_SCHEMA_MISUNDERSTANDING,
    "MISSING_COLUMN": ErrorCategory.E4_WRONG_COLUMN_SELECTION,
    "AMBIGUOUS_COLUMN": ErrorCategory.E4_WRONG_COLUMN_SELECTION,
    "UNSUPPORTED_OPERATION": ErrorCategory.E5_WRONG_ANALYTICAL_OPERATION,
    "INVALID_OPERATION": ErrorCategory.E5_WRONG_ANALYTICAL_OPERATION,
    "UNSUPPORTED_TOOL": ErrorCategory.E9_TOOL_SELECTION_ERROR,
    # Numerical / intermediate
    "NUMERICAL_MISMATCH": ErrorCategory.E6_NUMERICAL_CALCULATION_ERROR,
    "INTERMEDIATE_MISMATCH": ErrorCategory.E7_INTERMEDIATE_RESULT_ERROR,
    "FILTER_ERROR": ErrorCategory.E8_FILTERING_RANKING_ERROR,
    "RANK_ERROR": ErrorCategory.E8_FILTERING_RANKING_ERROR,
    # Verification
    "VERIFICATION_FAILED": ErrorCategory.E10_VERIFICATION_FAILURE,
    "VERIFICATION_NOT_PASSED": ErrorCategory.E10_VERIFICATION_FAILURE,
    "UNSUPPORTED_CLAIM": ErrorCategory.E13_EXPLANATION_GROUNDING_FAILURE,
    "EXPLANATION_NOT_GROUNDED": ErrorCategory.E13_EXPLANATION_GROUNDING_FAILURE,
    "UNGROUNDED_NUMBER": ErrorCategory.E13_EXPLANATION_GROUNDING_FAILURE,
    # Self-Correction
    "CORRECTION_FAILED": ErrorCategory.E12_CORRECTION_FAILURE,
    "CORRECTION_NOT_SUPPORTED": ErrorCategory.E12_CORRECTION_FAILURE,
    "CORRECTION_BUDGET_EXHAUSTED": ErrorCategory.E12_CORRECTION_FAILURE,
    "DIAGNOSIS_FAILED": ErrorCategory.E11_ERROR_DIAGNOSIS_FAILURE,
    # Metric definition / RAG
    "METRIC_NOT_FOUND": ErrorCategory.E3_METRIC_DEFINITION_ERROR,
    "DEFINITION_NOT_AVAILABLE": ErrorCategory.E3_METRIC_DEFINITION_ERROR,
    # System / Execution
    "TOOL_EXECUTION_FAILED": ErrorCategory.E15_SYSTEM_TOOL_EXECUTION_FAILURE,
    "GRAPH_EXECUTION_FAILED": ErrorCategory.E15_SYSTEM_TOOL_EXECUTION_FAILURE,
    "INTERNAL_ERROR": ErrorCategory.E15_SYSTEM_TOOL_EXECUTION_FAILURE,
    "DATASET_UNREADABLE": ErrorCategory.E15_SYSTEM_TOOL_EXECUTION_FAILURE,
}


def classify_error(
    stage: str,
    code: str,
    message: str,
    case_id: str = "",
    expected: str = "",
    observed: str = "",
    evidence: str = "",
    recoverable: bool = False,
    corrected: bool = False,
) -> ErrorRecord:
    """Map observed pipeline errors deterministically to the official E1–E15 taxonomy."""
    code_upper = code.strip().upper()
    category = _CODE_MAP.get(code_upper)

    if category is None:
        # Fallback based on stage
        stage_lower = stage.strip().lower()
        if "plan" in stage_lower:
            category = ErrorCategory.E1_INTENT_MISUNDERSTANDING
        elif "verif" in stage_lower:
            category = ErrorCategory.E10_VERIFICATION_FAILURE
        elif "correct" in stage_lower:
            category = ErrorCategory.E12_CORRECTION_FAILURE
        elif "knowledge" in stage_lower or "rag" in stage_lower:
            category = ErrorCategory.E3_METRIC_DEFINITION_ERROR
        elif "explain" in stage_lower:
            category = ErrorCategory.E13_EXPLANATION_GROUNDING_FAILURE
        else:
            category = ErrorCategory.E15_SYSTEM_TOOL_EXECUTION_FAILURE

    return ErrorRecord(
        case_id=case_id,
        stage=stage,
        error_category=category.value,
        error_subcategory=code,
        expected_behavior=expected or "Successful stage completion.",
        observed_behavior=observed or message,
        recoverable=recoverable,
        corrected=corrected,
        evidence=evidence,
    )
