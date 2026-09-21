"""Answers a question: classify -> run one approved tool -> validate the result -> explain."""

import logging

from app.core.config import Settings
from app.core.errors import AppError, ErrorCode, ToolError
from app.schemas.query_schema import QueryErrorInfo, QueryResponse
from app.schemas.tool_schema import AggregationResult, GroupingResult, RankingResult
from app.services.dataset_repository import DatasetRepository
from app.services.explainer import explain, uses_money
from app.services.gemini_classifier import classify_question
from app.services.gemini_client import LlmClient
from app.services.query_dispatcher import dispatch
from app.services.result_validator import validate_result
from app.utils.dataframe_utils import load_dataframe
from app.utils.upload_validator import parse_dataset_id

logger = logging.getLogger(__name__)

SUPPORTED_HELP = (
    "You can ask for: the total revenue, the average order value, the sum, average, minimum, maximum or "
    "count of quantity or unit price, the number of records, revenue or quantity by a column such as "
    "region or product, the top or bottom groups, and missing values."
)
DEFINITION_MESSAGE = (
    "Business metric definitions are not available yet. They will be added in Module 7. "
    "For now, ask a calculation question such as 'What is the total revenue?'."
)
NO_FILTERS_NOTE = "No filters were applied. The answer covers every row of the dataset."
_TOOL_NOTES = (AggregationResult, GroupingResult, RankingResult)


def _clean_question(question: str, max_length: int) -> str:
    """Collapse spaces and remove control characters. Reject empty or too long questions."""
    cleaned = "".join(ch for ch in " ".join(question.split()) if ch.isprintable())
    if not cleaned:
        raise AppError(ErrorCode.EMPTY_QUESTION, "Please type a question.", status_code=422)
    if len(cleaned) > max_length:
        raise AppError(
            ErrorCode.QUESTION_TOO_LONG,
            f"The question is longer than {max_length} characters.",
            status_code=422,
            details={"max_length": max_length},
        )
    return cleaned


def _calculation_method(result: object) -> str:
    if isinstance(result, AggregationResult):
        if result.order_count_method:
            return f"{result.calculation_method}; {result.order_count_method}"
        return result.calculation_method
    if isinstance(result, (GroupingResult, RankingResult)):
        return result.calculation_method
    return "empty_cell_count"


def _error_info(code: ErrorCode, message: str, details: dict | None = None) -> QueryErrorInfo:
    return QueryErrorInfo(code=code.value, message=message, details=details or {})


def answer_question(
    *,
    raw_dataset_id: str,
    question: str,
    settings: Settings,
    repository: DatasetRepository,
    llm_client: LlmClient,
) -> QueryResponse:
    """Answer one question about a stored dataset. Raises AppError for a bad id, question or dataset."""
    dataset_id = parse_dataset_id(raw_dataset_id)
    clean = _clean_question(question, settings.max_question_length)
    repository.get(dataset_id)  # raises DATASET_NOT_FOUND (404) for an unknown dataset
    frame = load_dataframe(repository.get_csv_path(dataset_id))

    classification = classify_question(clean, [str(name) for name in frame.columns], llm_client, settings)
    plan = classification.plan

    def respond(status: str, **extra: object) -> QueryResponse:
        response = QueryResponse(
            status=status,  # type: ignore[arg-type]
            question=clean,
            dataset_id=dataset_id,
            classifier=classification.info,
            query_plan=plan,
            **extra,  # type: ignore[arg-type]
        )
        # Only technical facts are logged. The question text is not.
        logger.info(
            "Question answered: dataset=%s classifier=%s intent=%s status=%s question_length=%d",
            dataset_id, classification.info.used, plan.intent, status, len(clean),
        )
        return response

    if plan.intent == "definition":
        return respond(
            "unsupported",
            message=DEFINITION_MESSAGE,
            error=_error_info(ErrorCode.UNSUPPORTED_QUESTION, DEFINITION_MESSAGE,
                              {"reason": "definition_not_available"}),
        )
    if plan.intent == "unsupported":
        message = f"I can't answer that yet. {plan.reasoning} {SUPPORTED_HELP}".replace("  ", " ")
        return respond(
            "unsupported",
            message=message,
            error=_error_info(ErrorCode.UNSUPPORTED_QUESTION, message, {"reason": "unsupported_question"}),
        )

    try:
        result = dispatch(frame, plan)
    except ToolError as exc:
        status = "error" if exc.code == ErrorCode.INVALID_PARAMETER else "insufficient_data"
        return respond(status, message=exc.message, error=_error_info(exc.code, exc.message, exc.details))
    except AppError as exc:  # invalid plan or unknown tool
        return respond("error", message=exc.message, error=_error_info(exc.code, exc.message, exc.details))

    validation = validate_result(frame, plan, result)
    if validation.status == "failed":
        message = "The result could not be validated, so it is not shown."
        failed_checks = [check.name for check in validation.checks if not check.passed]
        return respond(
            "error",
            validation=validation,
            message=message,
            error=_error_info(ErrorCode.RESULT_VALIDATION_FAILED, message, {"failed_checks": failed_checks}),
        )

    notes = result.notes if isinstance(result, _TOOL_NOTES) else []
    assumptions = [NO_FILTERS_NOTE, *classification.assumptions, *notes]
    if uses_money(plan):
        assumptions.append(
            f"Amounts are shown with the symbol '{settings.currency_symbol}'. The data file does not contain a currency."
        )
    return respond(
        "success",
        tool_used=result.tool,  # type: ignore[attr-defined]
        result=result,
        explanation=explain(plan, result, settings.currency_symbol),
        calculation_method=_calculation_method(result),
        assumptions=assumptions,
        validation=validation,
    )