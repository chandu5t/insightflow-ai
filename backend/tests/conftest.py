"""Shared Pytest fixtures."""

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.main import app


@pytest.fixture
def upload_dir(tmp_path: Path) -> Path:
    """A temporary upload folder, so tests never touch the real data/uploads folder."""
    return tmp_path / "uploads"


@pytest.fixture
def make_client(upload_dir: Path) -> Iterator[Callable[..., TestClient]]:
    """Build a test client with custom settings, e.g. make_client(max_rows=3)."""

    def _make_client(**setting_overrides: object) -> TestClient:
        # Tests must never use a real Gemini key that may exist in backend/.env.
        setting_overrides.setdefault("gemini_api_key", None)
        test_settings = Settings(upload_dir=upload_dir, **setting_overrides)
        app.dependency_overrides[get_settings] = lambda: test_settings
        return TestClient(app)

    yield _make_client
    app.dependency_overrides.clear()


@pytest.fixture
def client(make_client: Callable[..., TestClient]) -> TestClient:
    """A test client that calls the API in memory (no server needed)."""
    return make_client()