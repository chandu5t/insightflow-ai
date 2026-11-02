"""Thin wrapper around the Google Gemini SDK. This is the ONLY file that imports the SDK.

If Google changes the SDK, only this file needs to change. Every failure becomes a small,
safe error, and the caller then uses the rule-based classifier.
"""

import logging
import os
import ssl
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Protocol

from app.core.errors import ErrorCode

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LlmCallResult:
    """Provider output with only usage values explicitly returned by the API."""

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class LlmError(Exception):
    """Base error. `code` is a Gemini error code, `reason` is a short text that is safe to log."""

    code: ErrorCode = ErrorCode.GEMINI_REQUEST_FAILED

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class GeminiNotConfiguredError(LlmError):
    code = ErrorCode.GEMINI_NOT_CONFIGURED


class GeminiRequestError(LlmError):
    code = ErrorCode.GEMINI_REQUEST_FAILED

    def __init__(
        self,
        reason: str,
        *,
        provider_category: str | None = None,
        provider_http_status: int | None = None,
        provider_exception_types: tuple[str, ...] = (),
        provider_google_status: str | None = None,
        provider_google_reason: str | None = None,
        provider_google_domain: str | None = None,
        provider_auth_diagnosis: str | None = None,
    ) -> None:
        super().__init__(reason)
        # Safe metadata only: never retain provider messages, request data, or headers.
        self.provider_category = provider_category
        self.provider_http_status = provider_http_status
        self.provider_exception_types = provider_exception_types
        self.provider_google_status = provider_google_status
        self.provider_google_reason = provider_google_reason
        self.provider_google_domain = provider_google_domain
        self.provider_auth_diagnosis = provider_auth_diagnosis


class LlmClient(Protocol):
    """What the classifier needs. Tests use a fake with the same two members."""

    model_name: str

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> str: ...


class GeminiClient:
    def __init__(self, *, api_key: str, model: str, timeout_seconds: float) -> None:
        self._api_key = api_key
        self.model_name = model
        self._timeout_seconds = timeout_seconds

    def __repr__(self) -> str:  # never show the key
        return f"GeminiClient(model={self.model_name!r})"

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> str:
        """Ask Gemini for a JSON reply. Raises GeminiNotConfiguredError or GeminiRequestError."""
        if not self._api_key:
            raise GeminiNotConfiguredError("GEMINI_API_KEY is not set")
        try:
            text = self._call_sdk(system_prompt, user_prompt)
        except ImportError as exc:
            _log_provider_failure(self.model_name, "generate_json", exc)
            raise GeminiNotConfiguredError("the google-genai package is not installed") from exc
        except TimeoutError as exc:
            _log_provider_failure(self.model_name, "generate_json", exc)
            raise GeminiRequestError(
                "timeout",
                provider_category="timeout",
                provider_exception_types=(type(exc).__name__,),
            ) from exc
        except GeminiRequestError as exc:
            _log_provider_failure(self.model_name, "generate_json", exc)
            raise
        except Exception as exc:
            category, status, exception_types = _provider_failure_details(exc)
            google_details = _google_error_details(exc)
            _log_provider_failure(self.model_name, "generate_json", exc)
            raise GeminiRequestError(
                type(exc).__name__,
                provider_category=category,
                provider_http_status=status,
                provider_exception_types=exception_types,
                **google_details,
            ) from exc
        if not text or not text.strip():
            raise GeminiRequestError("empty response")
        return text

    def generate_json_with_usage(self, *, system_prompt: str, user_prompt: str) -> LlmCallResult:
        """Return response text and provider-reported token counts, when supplied."""
        if not self._api_key:
            raise GeminiNotConfiguredError("GEMINI_API_KEY is not set")
        try:
            result = self._call_sdk_with_usage(system_prompt, user_prompt)
        except ImportError as exc:
            _log_provider_failure(self.model_name, "generate_json_with_usage", exc)
            raise GeminiNotConfiguredError("the google-genai package is not installed") from exc
        except TimeoutError as exc:
            _log_provider_failure(self.model_name, "generate_json_with_usage", exc)
            raise GeminiRequestError(
                "timeout",
                provider_category="timeout",
                provider_exception_types=(type(exc).__name__,),
            ) from exc
        except GeminiRequestError as exc:
            _log_provider_failure(self.model_name, "generate_json_with_usage", exc)
            raise
        except Exception as exc:
            category, status, exception_types = _provider_failure_details(exc)
            google_details = _google_error_details(exc)
            _log_provider_failure(self.model_name, "generate_json_with_usage", exc)
            raise GeminiRequestError(
                type(exc).__name__,
                provider_category=category,
                provider_http_status=status,
                provider_exception_types=exception_types,
                **google_details,
            ) from exc
        if not result.text or not result.text.strip():
            raise GeminiRequestError("empty response")
        return result

    def _call_sdk(self, system_prompt: str, user_prompt: str) -> str | None:
        return self._generate_content(system_prompt, user_prompt).text

    def _call_sdk_with_usage(self, system_prompt: str, user_prompt: str) -> LlmCallResult:
        response = self._generate_content(system_prompt, user_prompt)
        usage = getattr(response, "usage_metadata", None)
        return LlmCallResult(
            text=response.text or "",
            input_tokens=_provider_token_count(usage, "prompt_token_count"),
            output_tokens=_provider_token_count(usage, "candidates_token_count"),
        )

    def _generate_content(self, system_prompt: str, user_prompt: str) -> Any:
        _validate_ssl_cert_override()
        from google import genai
        from google.genai import types

        # google-genai otherwise defaults to certifi's bundle. On Windows, the
        # Python default context also trusts the OS certificate store, which may
        # contain enterprise inspection roots. Keep normal chain and hostname
        # verification enabled while passing that context explicitly to HTTPX.
        ssl_context = ssl.create_default_context()
        if ssl_context.verify_mode != ssl.CERT_REQUIRED or not ssl_context.check_hostname:
            raise GeminiRequestError("a verified TLS certificate context is unavailable")

        client = genai.Client(
            api_key=self._api_key,
            http_options=types.HttpOptions(
                timeout=int(self._timeout_seconds * 1000),
                client_args={"verify": ssl_context},
            ),
        )
        response = client.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            # JSON mode and prompt configuration are shared by normal and usage-aware calls.
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
            ),
        )
        return response


def _validate_ssl_cert_override() -> None:
    """Fail early with a safe hint when SSL_CERT_FILE is stale or not a CA bundle.

    The path and contents are intentionally never logged or included in exceptions.
    TLS verification remains enabled in the Google GenAI/HTTPX transport.
    """
    configured_path = os.environ.get("SSL_CERT_FILE")
    if not configured_path:
        return
    try:
        if not Path(configured_path).is_file():
            raise OSError("not a file")
        ssl.create_default_context(cafile=configured_path)
    except (OSError, ssl.SSLError, ValueError) as exc:
        raise GeminiRequestError(
            "SSL_CERT_FILE is invalid; unset the stale override or configure a readable trusted CA bundle."
        ) from exc


def _provider_token_count(usage: object | None, field: str) -> int | None:
    if isinstance(usage, dict):
        value = usage.get(field)
    else:
        value = getattr(usage, field, None) if usage is not None else None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _log_provider_failure(model: str, operation: str, error: Exception) -> None:
    """Log only safe provider diagnostics; never log exception text or request data."""
    category, status, chain = _provider_failure_details(error)
    google_details = _google_error_details(error)
    logger.warning(
        "Gemini request failed model=%s operation=%s category=%s http_status=%s "
        "exception_types=%s google_status=%s google_reason=%s google_domain=%s auth_diagnosis=%s",
        model,
        operation,
        category,
        status if status is not None else "unavailable",
        ">".join(chain),
        google_details.get("provider_google_status") or "unavailable",
        google_details.get("provider_google_reason") or "unavailable",
        google_details.get("provider_google_domain") or "unavailable",
        google_details.get("provider_auth_diagnosis") or "unclassified",
    )


_GOOGLE_STATUS_ALLOWLIST = {
    "UNAUTHENTICATED", "PERMISSION_DENIED", "INVALID_ARGUMENT", "NOT_FOUND",
    "RESOURCE_EXHAUSTED", "FAILED_PRECONDITION", "UNAVAILABLE", "INTERNAL",
}
_GOOGLE_REASON_ALLOWLIST = {
    "API_KEY_INVALID", "API_KEY_EXPIRED", "API_KEY_SERVICE_BLOCKED",
    "API_KEY_IP_ADDRESS_BLOCKED", "API_KEY_HTTP_REFERRER_BLOCKED",
    "API_KEY_ANDROID_APP_BLOCKED", "API_KEY_IOS_APP_BLOCKED", "SERVICE_DISABLED",
    "CONSUMER_INVALID", "ACCESS_NOT_CONFIGURED", "MODEL_NOT_FOUND",
    "MODEL_ACCESS_DENIED", "MODEL_NOT_SUPPORTED", "UNAUTHENTICATED",
    "PERMISSION_DENIED",
}


def _google_error_details(error: Exception) -> dict[str, str]:
    """Extract only allowlisted Google RPC status/reason fields from GenAI errors."""
    result: dict[str, str] = {}
    pending: Exception | None = error
    seen: set[int] = set()
    while pending is not None and id(pending) not in seen:
        seen.add(id(pending))
        status = getattr(pending, "status", None)
        if isinstance(status, str) and status in _GOOGLE_STATUS_ALLOWLIST:
            result.setdefault("provider_google_status", status)

        payload = getattr(pending, "details", None)
        if isinstance(payload, dict):
            error_payload = payload.get("error")
            record = error_payload if isinstance(error_payload, dict) else payload
            payload_status = record.get("status")
            if isinstance(payload_status, str) and payload_status in _GOOGLE_STATUS_ALLOWLIST:
                result.setdefault("provider_google_status", payload_status)
            details = record.get("details")
            if isinstance(details, list):
                for item in details:
                    if not isinstance(item, dict):
                        continue
                    if item.get("@type") != "type.googleapis.com/google.rpc.ErrorInfo":
                        continue
                    reason = item.get("reason")
                    domain = item.get("domain")
                    if isinstance(reason, str) and reason in _GOOGLE_REASON_ALLOWLIST:
                        result.setdefault("provider_google_reason", reason)
                    # Project identifiers and arbitrary metadata are deliberately omitted.
                    if domain == "googleapis.com":
                        result.setdefault("provider_google_domain", domain)
        pending = pending.__cause__ or pending.__context__

    status = result.get("provider_google_status")
    reason = result.get("provider_google_reason")
    diagnosis = None
    if reason in {"API_KEY_INVALID", "API_KEY_EXPIRED"}:
        diagnosis = "invalid_or_expired_api_key"
    elif reason in {
        "API_KEY_SERVICE_BLOCKED", "API_KEY_IP_ADDRESS_BLOCKED",
        "API_KEY_HTTP_REFERRER_BLOCKED", "API_KEY_ANDROID_APP_BLOCKED",
        "API_KEY_IOS_APP_BLOCKED",
    }:
        diagnosis = "api_key_restriction"
    elif status == "UNAUTHENTICATED" or reason == "UNAUTHENTICATED":
        diagnosis = "authentication"
    elif reason in {"SERVICE_DISABLED", "CONSUMER_INVALID", "ACCESS_NOT_CONFIGURED"}:
        diagnosis = "project_or_api_access"
    elif reason in {"MODEL_NOT_FOUND", "MODEL_ACCESS_DENIED", "MODEL_NOT_SUPPORTED"}:
        diagnosis = "model_access"
    elif status == "PERMISSION_DENIED":
        diagnosis = "permission_denied_unspecified"
    if diagnosis is not None:
        result["provider_auth_diagnosis"] = diagnosis
    return result


def _provider_failure_details(error: Exception) -> tuple[str, int | None, tuple[str, ...]]:
    """Extract bounded exception identities and numeric HTTP status; never messages."""
    chain: list[str] = []
    status: int | None = None
    pending: Exception | None = error
    seen: set[int] = set()
    while pending is not None and id(pending) not in seen and len(chain) < 6:
        seen.add(id(pending))
        chain.append(type(pending).__name__)
        if status is None:
            for attribute in ("status_code", "code"):
                candidate = getattr(pending, attribute, None)
                if isinstance(candidate, int) and not isinstance(candidate, bool) and 100 <= candidate <= 599:
                    status = candidate
                    break
        pending = pending.__cause__ or pending.__context__

    normalized = " ".join(chain).lower()
    if isinstance(error, GeminiRequestError) and "ssl_cert_file is invalid" in error.reason.lower():
        category = "tls_configuration"
    elif isinstance(error, GeminiRequestError) and "timeout" in error.reason.lower():
        category = "timeout"
    elif status == 429:
        category = "quota_or_rate_limit"
    elif status == 401:
        category = "authentication"
    elif status == 403:
        category = "permission"
    elif status == 400:
        category = "invalid_request"
    elif status is not None and status >= 500:
        category = "provider_server_error"
    elif "ssl" in normalized or "certificate" in normalized:
        category = "tls"
    elif "timeout" in normalized or "deadline" in normalized:
        category = "timeout"
    elif "connect" in normalized or "network" in normalized:
        category = "connection"
    elif isinstance(error, GeminiNotConfiguredError) or isinstance(error, ImportError):
        category = "client_configuration"
    else:
        category = "provider_or_client_error"

    return category, status, tuple(chain)
