"""Question-to-validated-plan service. No dataset access or plan execution occurs here."""

import json
import logging
import time
import uuid
from typing import Protocol

from pydantic import ValidationError

from app.core.config import Settings
from app.core.errors import AppError, ErrorCode
from app.planner.normalizer import normalize_plan_payload
from app.planner.metric_registry import validate_authoritative_planner_context
from app.planner.prompts import SYSTEM_PROMPT, build_user_prompt
from app.planner.schemas import AnalysisPlan, MetricDefinitionContext, PlannerRequest, PlannerResponse, PlannerMetadata
from app.planner.validator import validate_plan
from app.services.gemini_client import GeminiNotConfiguredError, GeminiRequestError

logger = logging.getLogger(__name__)


class PlannerModel(Protocol):
    model_name: str

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> str: ...


def _safe_log(event: dict[str, object]) -> None:
    """Emit JSON application logs; caller never supplies provider response or secrets."""
    logger.info("planner_event %s", json.dumps(event, ensure_ascii=False, default=str, separators=(",", ":")))


def _parse_generated_plan(text: str) -> AnalysisPlan:
    payload = json.loads(text)
    normalized = normalize_plan_payload(payload)
    return AnalysisPlan.model_validate(normalized)


def _provider_failure_details(exc: Exception) -> dict[str, object] | None:
    """Expose only normalized Gemini diagnostics; never provider text or request data."""
    if not isinstance(exc, GeminiRequestError):
        return None
    details: dict[str, object] = {}
    if isinstance(exc.provider_category, str):
        details["category"] = exc.provider_category
    if isinstance(exc.provider_http_status, int):
        details["http_status"] = exc.provider_http_status
    if exc.provider_exception_types:
        details["exception_types"] = list(exc.provider_exception_types[:6])
    return details or None


def _is_retryable_provider_failure(exc: Exception) -> bool:
    """Retry transient failures once; permanent HTTP responses are not retryable."""
    if not isinstance(exc, GeminiRequestError):
        return False
    status = exc.provider_http_status
    if status is not None:
        return status in {408, 425, 429, 500, 502, 503, 504}
    if exc.provider_category is not None:
        return exc.provider_category in {
            "connection", "timeout", "provider_server_error", "quota_or_rate_limit"
        }
    # Preserve the existing single retry for normalized errors without provider metadata.
    return True


def create_plan(
    request: PlannerRequest,
    *,
    llm_client: PlannerModel,
    authorized_metric_definitions: list[MetricDefinitionContext] | tuple[MetricDefinitionContext, ...] | None = None,
) -> PlannerResponse:
    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    retry_used = False
    failure_category: str | None = None
    context_has_trusted_semantics = bool(
        request.dataset_context
        and (
            request.dataset_context.manifest_version
            or any(role != "unknown" for role in request.dataset_context.semantic_roles.values())
        )
    )
    if (request.metric_definitions or context_has_trusted_semantics) and authorized_metric_definitions is None:
        raise ValueError("Semantic planner context must be validated against the application-owned registry before planning.")
    if authorized_metric_definitions is not None:
        request_definitions = request.metric_definitions or []
        context = request.dataset_context
        if (
            list(authorized_metric_definitions) != request_definitions
            or context is None
            or not validate_authoritative_planner_context(
                dataset_sha256=context.dataset_sha256,
                manifest_version=context.manifest_version,
                columns=context.columns,
                semantic_roles=context.semantic_roles,
                definitions=authorized_metric_definitions,
            )
        ):
            raise ValueError("Planner metric context must exactly match its validated registry entries.")
    prompt = build_user_prompt(request)

    for attempt in range(2):
        try:
            raw = llm_client.generate_json(system_prompt=SYSTEM_PROMPT, user_prompt=prompt)
        except GeminiNotConfiguredError as exc:
            failure_category = "LLM_PROVIDER_ERROR"
            latency = (time.perf_counter() - started) * 1000
            _log_failure(request, request_id, llm_client.model_name, retry_used, failure_category, latency)
            raise AppError(ErrorCode.PLANNER_GENERATION_ERROR, "The planner model is not configured.", 503,
                           {"failure_category": failure_category, "request_id": request_id}) from exc
        except Exception as exc:
            # GeminiRequestError is the client's normalized provider/network failure.
            # Unexpected adapter errors are made safe but are not retried as transient.
            invalid_ca_override = (
                isinstance(exc, GeminiRequestError)
                and "SSL_CERT_FILE is invalid" in exc.reason
            )
            failure_category = (
                "SSL_CERT_FILE_INVALID" if invalid_ca_override
                else "PLANNER_TIMEOUT" if isinstance(exc, GeminiRequestError) and "timeout" in exc.reason.lower()
                else "LLM_PROVIDER_ERROR"
            )
            if attempt == 0 and _is_retryable_provider_failure(exc) and not invalid_ca_override:
                retry_used = True
                continue
            latency = (time.perf_counter() - started) * 1000
            _log_failure(request, request_id, llm_client.model_name, retry_used, failure_category, latency)
            code = ErrorCode.PLANNER_TIMEOUT if failure_category == "PLANNER_TIMEOUT" else ErrorCode.LLM_PROVIDER_ERROR
            status = 504 if failure_category == "PLANNER_TIMEOUT" else 503 if invalid_ca_override else 502
            message = (
                "The SSL_CERT_FILE override is invalid. Unset the stale process-level value or point it to a readable trusted CA bundle; TLS verification remains enabled."
                if invalid_ca_override else "The planner model request failed."
            )
            provider_diagnostics = _provider_failure_details(exc)
            details: dict[str, object] = {
                "failure_category": failure_category,
                "request_id": request_id,
            }
            if provider_diagnostics is not None:
                details["provider_diagnostics"] = provider_diagnostics
            raise AppError(code, message, status,
                           details) from exc

        try:
            plan = _parse_generated_plan(raw)
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
            failure_category = "PLAN_SCHEMA_ERROR"
            if attempt == 0:
                retry_used = True
                continue
            latency = (time.perf_counter() - started) * 1000
            _log_failure(request, request_id, llm_client.model_name, retry_used, failure_category, latency)
            raise AppError(ErrorCode.PLAN_SCHEMA_ERROR, "The planner did not return a usable structured plan.", 502,
                           {"failure_category": failure_category, "request_id": request_id}) from exc

        validation_errors = validate_plan(
            plan,
            authorized_metric_definitions=authorized_metric_definitions,
            semantic_roles=(
                request.dataset_context.semantic_roles
                if request.dataset_context is not None
                else {}
            ),
            dataset_columns=(
                request.dataset_context.columns
                if request.dataset_context is not None
                else None
            ),
        )
        latency = (time.perf_counter() - started) * 1000
        valid = not validation_errors
        response = PlannerResponse(
            plan=plan,
            valid=valid,
            validation_errors=validation_errors,
            metadata=PlannerMetadata(
                request_id=request_id,
                planner_model=llm_client.model_name,
                retry_used=retry_used,
                latency_ms=round(latency, 3),
            ),
        )
        _safe_log({
            "event": "planner_completed",
            "request_id": request_id,
            "question": request.question,
            "planner_model": llm_client.model_name,
            "planner_version": "v2.2",
            "plan": plan.model_dump(mode="json"),
            "valid": valid,
            "validation_errors": [error.model_dump(mode="json") for error in validation_errors],
            "retry_used": retry_used,
            "failure_category": "PLAN_VALIDATION_ERROR" if not valid else None,
            "execution_time_ms": round(latency, 3),
            "token_usage": None,
        })
        return response

    raise AssertionError("The planner retry loop must return or raise.")


def _log_failure(request: PlannerRequest, request_id: str, model: str, retry_used: bool,
                 category: str, latency: float) -> None:
    _safe_log({
        "event": "planner_failed",
        "request_id": request_id,
        "question": request.question,
        "planner_model": model,
        "planner_version": "v2.2",
        "validation_result": None,
        "validation_errors": [],
        "retry_used": retry_used,
        "failure_category": category,
        "execution_time_ms": round(latency, 3),
        "token_usage": None,
    })
