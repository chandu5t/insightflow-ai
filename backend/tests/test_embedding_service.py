"""Tests for GeminiEmbeddingService. The SDK call is monkeypatched; no network is used."""

import pytest

from app.services.embedding_service import GeminiEmbeddingService, _normalize
from app.services.gemini_client import GeminiNotConfiguredError, GeminiRequestError


def make(key: str = "sk-test") -> GeminiEmbeddingService:
    return GeminiEmbeddingService(api_key=key, model="gemini-embedding-001", dimensions=4, timeout_seconds=5)


def test_normalize_produces_a_unit_vector() -> None:
    result = _normalize([3.0, 4.0])
    assert result == pytest.approx([0.6, 0.8])


def test_missing_key_raises_not_configured_without_touching_the_sdk() -> None:
    service = make(key="")
    with pytest.raises(GeminiNotConfiguredError):
        service.embed_query("what is revenue")


def test_embed_query_normalizes_the_returned_vector(monkeypatch) -> None:
    service = make()
    monkeypatch.setattr(service, "_call_sdk", lambda texts, task_type: [[3.0, 4.0, 0.0, 0.0]])

    result = service.embed_query("average order value")

    assert result == pytest.approx([0.6, 0.8, 0.0, 0.0])


def test_embed_documents_calls_the_sdk_once_for_the_whole_batch(monkeypatch) -> None:
    service = make()
    calls = []

    def fake_call(texts, task_type):
        calls.append((texts, task_type))
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    monkeypatch.setattr(service, "_call_sdk", fake_call)

    results = service.embed_documents(["a", "b", "c"])

    assert len(results) == 3 and calls == [(["a", "b", "c"], "RETRIEVAL_DOCUMENT")]


def test_embed_query_uses_retrieval_query_task_type(monkeypatch) -> None:
    service = make()
    captured = {}
    monkeypatch.setattr(
        service, "_call_sdk",
        lambda texts, task_type: captured.update(task_type=task_type) or [[1.0, 0.0, 0.0, 0.0]],
    )

    service.embed_query("q")

    assert captured["task_type"] == "RETRIEVAL_QUERY"


def test_empty_documents_list_returns_empty_without_calling_the_sdk(monkeypatch) -> None:
    service = make()
    monkeypatch.setattr(service, "_call_sdk", lambda *a: pytest.fail("should not be called"))

    assert service.embed_documents([]) == []


def test_sdk_failure_becomes_a_request_error(monkeypatch) -> None:
    service = make()

    def broken(texts, task_type):
        raise RuntimeError("sk-test leaked here")

    monkeypatch.setattr(service, "_call_sdk", broken)

    with pytest.raises(GeminiRequestError) as info:
        service.embed_query("q")
    assert "sk-test" not in str(info.value)