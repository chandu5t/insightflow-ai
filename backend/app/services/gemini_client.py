"""Thin wrapper around the Google Gemini SDK. This is the ONLY file that imports the SDK.

If Google changes the SDK, only this file needs to change. Every failure becomes a small,
safe error, and the caller then uses the rule-based classifier.
"""

import logging
from typing import Protocol

from app.core.errors import ErrorCode

logger = logging.getLogger(__name__)


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

    def _call_sdk(self, system_prompt: str, user_prompt: str) -> str | None:
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=self._api_key,
            http_options=types.HttpOptions(timeout=int(self._timeout_seconds * 1000)),  # milliseconds
        )
        response = client.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            # JSON mode only. We check the structure ourselves with Pydantic, so this keeps working
            # even if Google changes how schemas are configured. No temperature: it is deprecated.
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
            ),
        )
        return response.text