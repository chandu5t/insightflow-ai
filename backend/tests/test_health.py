"""Tests for the health endpoint, CORS behaviour, and settings parsing."""

from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings


def test_health_returns_200_and_healthy_status(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_health_response_contains_expected_fields(client: TestClient) -> None:
    body = client.get("/health").json()

    assert set(body.keys()) == {"status", "app", "version", "environment"}
    assert body["app"] == "InsightFlow AI"


def test_health_rejects_post_method(client: TestClient) -> None:
    response = client.post("/health")

    assert response.status_code == 405


def test_unknown_route_returns_404(client: TestClient) -> None:
    response = client.get("/this-route-does-not-exist")

    assert response.status_code == 404


def test_cors_allows_configured_frontend_origin(client: TestClient) -> None:
    allowed_origin = get_settings().cors_origins_list[0]

    response = client.get("/health", headers={"Origin": allowed_origin})

    assert response.headers.get("access-control-allow-origin") == allowed_origin


def test_cors_does_not_allow_unknown_origin(client: TestClient) -> None:
    response = client.get("/health", headers={"Origin": "http://evil.example.com"})

    assert "access-control-allow-origin" not in response.headers


def test_cors_origins_setting_is_split_into_clean_list() -> None:
    settings = Settings(cors_origins="http://a.com, http://b.com ,")

    assert settings.cors_origins_list == ["http://a.com", "http://b.com"]