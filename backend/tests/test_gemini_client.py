"""Tests for the Gemini wrapper. The SDK call itself is replaced, so there is no network."""

import ssl

import pytest
from google import genai
from google.genai.errors import APIError

from app.core.errors import ErrorCode
from app.services.gemini_client import (
    GeminiClient,
    GeminiNotConfiguredError,
    GeminiRequestError,
    LlmCallResult,
    _provider_token_count,
)


def make(key: str = "sk-SECRET-123") -> GeminiClient:
    return GeminiClient(api_key=key, model="m", timeout_seconds=5)


def ask(client: GeminiClient) -> str:
    return client.generate_json(system_prompt="s", user_prompt="u")


def test_missing_key_is_not_configured_and_the_sdk_is_not_touched(monkeypatch) -> None:
    client = make(key="")
    monkeypatch.setattr(client, "_call_sdk", lambda system_prompt, user_prompt: pytest.fail("SDK was called"))

    with pytest.raises(GeminiNotConfiguredError) as info:
        ask(client)

    assert info.value.code == ErrorCode.GEMINI_NOT_CONFIGURED


def test_text_is_returned(monkeypatch) -> None:
    client = make()
    monkeypatch.setattr(client, "_call_sdk", lambda system_prompt, user_prompt: '{"a": 1}')

    assert ask(client) == '{"a": 1}'


def test_missing_sdk_package_means_not_configured(monkeypatch) -> None:
    client = make()

    def missing(system_prompt: str, user_prompt: str) -> str:
        raise ImportError("no module named google")

    monkeypatch.setattr(client, "_call_sdk", missing)

    with pytest.raises(GeminiNotConfiguredError):
        ask(client)


def test_api_failure_hides_the_original_message(monkeypatch) -> None:
    client = make()

    def broken(system_prompt: str, user_prompt: str) -> str:
        raise RuntimeError("request with key sk-SECRET-123 failed")

    monkeypatch.setattr(client, "_call_sdk", broken)

    with pytest.raises(GeminiRequestError) as info:
        ask(client)

    assert info.value.code == ErrorCode.GEMINI_REQUEST_FAILED
    assert info.value.reason == "RuntimeError" and "sk-SECRET-123" not in str(info.value)


def test_provider_failure_logs_safe_status_and_category_without_exception_text(monkeypatch, caplog) -> None:
    client = make()

    class RateLimitError(Exception):
        status_code = 429

    def limited(system_prompt: str, user_prompt: str) -> str:
        raise RateLimitError("private response text with credential sk-SECRET-123")

    monkeypatch.setattr(client, "_call_sdk", limited)
    with caplog.at_level("WARNING", logger="app.services.gemini_client"):
        with pytest.raises(GeminiRequestError):
            ask(client)

    assert "category=quota_or_rate_limit" in caplog.text
    assert "http_status=429" in caplog.text
    assert "RateLimitError" in caplog.text
    assert "private response text" not in caplog.text
    assert "sk-SECRET-123" not in caplog.text


def test_provider_http_status_is_preserved_as_safe_exception_metadata(monkeypatch) -> None:
    client = make()

    class PermissionDenied(Exception):
        code = 403

    def denied(system_prompt: str, user_prompt: str) -> str:
        raise PermissionDenied("private provider payload and sk-SECRET-123")

    monkeypatch.setattr(client, "_call_sdk", denied)
    with pytest.raises(GeminiRequestError) as info:
        ask(client)

    assert info.value.provider_category == "permission"
    assert info.value.provider_http_status == 403
    assert info.value.provider_exception_types == ("PermissionDenied",)
    assert "private provider payload" not in str(info.value)
    assert "sk-SECRET-123" not in str(info.value)


@pytest.mark.parametrize(
    ("reason", "expected_diagnosis"),
    [
        ("API_KEY_SERVICE_BLOCKED", "api_key_restriction"),
        ("API_KEY_INVALID", "invalid_or_expired_api_key"),
        ("UNAUTHENTICATED", "authentication"),
        ("SERVICE_DISABLED", "project_or_api_access"),
        ("MODEL_ACCESS_DENIED", "model_access"),
    ],
)
def test_google_error_info_logs_only_allowlisted_auth_diagnostics(
    monkeypatch, caplog, reason, expected_diagnosis
) -> None:
    client = make()
    error = APIError(403, {
        "error": {
            "status": "PERMISSION_DENIED",
            "message": "provider message containing sk-SECRET-123",
            "details": [{
                "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                "reason": reason,
                "domain": "googleapis.com",
                "metadata": {"consumer": "projects/private-project-123", "credential": "sk-SECRET-123"},
            }],
        }
    })

    def denied(system_prompt: str, user_prompt: str) -> str:
        raise error

    monkeypatch.setattr(client, "_call_sdk", denied)
    with caplog.at_level("WARNING", logger="app.services.gemini_client"):
        with pytest.raises(GeminiRequestError) as info:
            ask(client)

    assert info.value.provider_google_status == "PERMISSION_DENIED"
    assert info.value.provider_google_reason == reason
    assert info.value.provider_google_domain == "googleapis.com"
    assert info.value.provider_auth_diagnosis == expected_diagnosis
    assert f"google_reason={reason}" in caplog.text
    assert f"auth_diagnosis={expected_diagnosis}" in caplog.text
    assert "private-project-123" not in caplog.text
    assert "credential" not in caplog.text
    assert "provider message" not in caplog.text
    assert "raw secret-bearing" not in caplog.text
    assert "sk-SECRET-123" not in caplog.text


def test_unrecognized_google_error_reason_and_domain_are_not_logged(monkeypatch, caplog) -> None:
    client = make()
    error = APIError(403, {
        "error": {
            "status": "PERMISSION_DENIED",
            "details": [{
                "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                "reason": "PRIVATE_REASON_WITH_SECRET",
                "domain": "private-project.example",
            }],
        }
    })

    monkeypatch.setattr(
        client, "_call_sdk",
        lambda system_prompt, user_prompt: (_ for _ in ()).throw(error),
    )
    with caplog.at_level("WARNING", logger="app.services.gemini_client"):
        with pytest.raises(GeminiRequestError) as info:
            ask(client)

    assert info.value.provider_google_status == "PERMISSION_DENIED"
    assert info.value.provider_google_reason is None
    assert info.value.provider_google_domain is None
    assert info.value.provider_auth_diagnosis == "permission_denied_unspecified"
    assert "PRIVATE_REASON_WITH_SECRET" not in caplog.text
    assert "private-project.example" not in caplog.text


def test_timeout_is_a_request_error(monkeypatch) -> None:
    client = make()

    def slow(system_prompt: str, user_prompt: str) -> str:
        raise TimeoutError()

    monkeypatch.setattr(client, "_call_sdk", slow)

    with pytest.raises(GeminiRequestError) as info:
        ask(client)

    assert info.value.reason == "timeout"


def test_invalid_ssl_cert_override_fails_safely_before_provider_request(monkeypatch) -> None:
    client = make()
    marker_path = "C:/private/ca-bundle-SECRET.pem"
    monkeypatch.setenv("SSL_CERT_FILE", marker_path)
    monkeypatch.setattr(
        client, "_call_sdk",
        lambda system_prompt, user_prompt: client._generate_content(system_prompt, user_prompt).text,
    )
    with pytest.raises(GeminiRequestError) as info:
        ask(client)

    assert "SSL_CERT_FILE is invalid" in str(info.value)
    assert marker_path not in str(info.value)
    assert "SECRET" not in str(info.value)


def test_sdk_uses_verified_python_default_trust_context(monkeypatch) -> None:
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    captured = {}

    class Models:
        @staticmethod
        def generate_content(**kwargs):
            return type("Response", (), {"text": '{"ok": true}'})()

    class FakeSdkClient:
        models = Models()

        def __init__(self, *, api_key, http_options):
            # Retain only options for assertions; never inspect or display api_key.
            captured["http_options"] = http_options

    monkeypatch.setattr(genai, "Client", FakeSdkClient)
    client = make()

    response = client._generate_content("system", "user")

    assert response.text == '{"ok": true}'
    options = captured["http_options"]
    assert options.timeout == 5000
    context = options.client_args["verify"]
    assert isinstance(context, ssl.SSLContext)
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True


@pytest.mark.parametrize("reply", [None, "", "   "])
def test_empty_reply_is_a_request_error(monkeypatch, reply) -> None:
    client = make()
    monkeypatch.setattr(client, "_call_sdk", lambda system_prompt, user_prompt: reply)

    with pytest.raises(GeminiRequestError):
        ask(client)


def test_the_key_is_never_shown_in_repr() -> None:
    assert "sk-SECRET-123" not in repr(make())


def test_usage_metadata_is_exposed_only_when_provider_supplies_counts(monkeypatch) -> None:
    client = make()
    monkeypatch.setattr(
        client,
        "_call_sdk_with_usage",
        lambda system_prompt, user_prompt: LlmCallResult(
            text='{"a": 1}', input_tokens=23, output_tokens=7
        ),
    )
    result = client.generate_json_with_usage(system_prompt="s", user_prompt="u")
    assert result.text == '{"a": 1}'
    assert result.input_tokens == 23
    assert result.output_tokens == 7


def test_usage_metadata_parser_does_not_estimate_missing_or_invalid_values() -> None:
    class Usage:
        prompt_token_count = 23
        candidates_token_count = None

    assert _provider_token_count(Usage(), "prompt_token_count") == 23
    assert _provider_token_count(Usage(), "candidates_token_count") is None
    assert _provider_token_count({"prompt_token_count": "23"}, "prompt_token_count") is None
