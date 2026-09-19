"""Tests for the standard error format."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.errors import register_exception_handlers


def test_unknown_route_uses_standard_error_format(client: TestClient) -> None:
    response = client.get("/nothing-here")

    assert response.status_code == 404
    assert response.json() == {"error": "Not Found", "code": "NOT_FOUND", "details": {}}


def test_wrong_method_uses_standard_error_format_and_keeps_allow_header(client: TestClient) -> None:
    response = client.post("/health")

    assert response.status_code == 405
    assert response.json()["code"] == "METHOD_NOT_ALLOWED"
    assert "allow" in response.headers


def test_unexpected_error_returns_safe_500_message() -> None:
    small_app = FastAPI()
    register_exception_handlers(small_app)

    @small_app.get("/boom")
    def boom() -> None:
        raise RuntimeError("secret internal detail: C:\\private\\path")

    response = TestClient(small_app, raise_server_exceptions=False).get("/boom")

    assert response.status_code == 500
    assert response.json()["code"] == "INTERNAL_ERROR"
    assert "secret" not in response.text and "private" not in response.text