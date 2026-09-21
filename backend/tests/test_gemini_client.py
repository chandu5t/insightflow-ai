"""Tests for the Gemini wrapper. The SDK call itself is replaced, so there is no network."""

import pytest

from app.core.errors import ErrorCode
from app.services.gemini_client import GeminiClient, GeminiNotConfiguredError, GeminiRequestError


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


def test_timeout_is_a_request_error(monkeypatch) -> None:
    client = make()

    def slow(system_prompt: str, user_prompt: str) -> str:
        raise TimeoutError()

    monkeypatch.setattr(client, "_call_sdk", slow)

    with pytest.raises(GeminiRequestError) as info:
        ask(client)

    assert info.value.reason == "timeout"


@pytest.mark.parametrize("reply", [None, "", "   "])
def test_empty_reply_is_a_request_error(monkeypatch, reply) -> None:
    client = make()
    monkeypatch.setattr(client, "_call_sdk", lambda system_prompt, user_prompt: reply)

    with pytest.raises(GeminiRequestError):
        ask(client)


def test_the_key_is_never_shown_in_repr() -> None:
    assert "sk-SECRET-123" not in repr(make())