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
from app.planner.prompts import SYSTEM_PROMPT, build_user_prompt
from app.planner.schemas import AnalysisPlan, PlannerRequest, PlannerResponse, PlannerMetadata
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


def create_plan(
    request: PlannerRequest,
    *,
    llm_client: PlannerModel,
) -> PlannerResponse:
    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    retry_used = False
    failure_category: str | None = None
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
            failure_category = "PLANNER_TIMEOUT" if isinstance(exc, GeminiRequestError) and "timeout" in exc.reason.lower() else "LLM_PROVIDER_ERROR"
            if attempt == 0 and isinstance(exc, GeminiRequestError):
                retry_used = True
                continue
            latency = (time.perf_counter() - started) * 1000
            _log_failure(request, request_id, llm_client.model_name, retry_used, failure_category, latency)
            code = ErrorCode.PLANNER_TIMEOUT if failure_category == "PLANNER_TIMEOUT" else ErrorCode.LLM_PROVIDER_ERROR
            status = 504 if failure_category == "PLANNER_TIMEOUT" else 502
            raise AppError(code, "The planner model request failed.", status,
                           {"failure_category": failure_category, "request_id": request_id}) from exc

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

        validation_errors = validate_plan(plan)
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
