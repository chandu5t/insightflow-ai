"""Shared Pytest fixtures."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    """A test client that calls the API in memory (no server needed)."""
    return TestClient(app)