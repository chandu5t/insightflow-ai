"""Thin wrapper around the Google Gemini SDK. This is the ONLY file that imports the SDK.

If Google changes the SDK, only this file needs to change. Every failure becomes a small,
safe error, and the caller then uses the rule-based classifier.
"""

import logging
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
            raise GeminiNotConfiguredError("the google-genai package is not installed") from exc
        except TimeoutError as exc:
            raise GeminiRequestError("timeout") from exc
        except Exception as exc:
            # Only the exception TYPE is kept. The message could contain sensitive text.
            logger.warning("Gemini request failed (%s)", type(exc).__name__)
            raise GeminiRequestError(type(exc).__name__) from exc
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
            raise GeminiNotConfiguredError("the google-genai package is not installed") from exc
        except TimeoutError as exc:
            raise GeminiRequestError("timeout") from exc
        except Exception as exc:
            logger.warning("Gemini request failed (%s)", type(exc).__name__)
            raise GeminiRequestError(type(exc).__name__) from exc
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
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=self._api_key,
            http_options=types.HttpOptions(timeout=int(self._timeout_seconds * 1000)),
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


def _provider_token_count(usage: object | None, field: str) -> int | None:
    if isinstance(usage, dict):
        value = usage.get(field)
    else:
        value = getattr(usage, field, None) if usage is not None else None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value
