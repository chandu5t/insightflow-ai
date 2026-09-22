"""End-to-end integration test: the FastAPI app itself, talking to a real PostgreSQL database.

Skips automatically if PostgreSQL is not reachable (see tests/db_fixtures.py).
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.main import app
from tests.db_fixtures import clean_postgres_tables, postgres_engine, postgres_settings  # noqa: F401
from tests.frames import SALES_PATH


@pytest.fixture
def postgres_client(postgres_engine, postgres_settings, tmp_path: Path, clean_postgres_tables):
    test_settings = Settings(
        gemini_api_key=None, upload_dir=tmp_path, storage_backend="postgres",
        postgres_host=postgres_settings.postgres_host, postgres_port=postgres_settings.postgres_port,
        postgres_user=postgres_settings.postgres_user, postgres_password=postgres_settings.postgres_password,
        postgres_db=postgres_settings.postgres_db,
    )
    app.dependency_overrides[get_settings] = lambda: test_settings
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_upload_then_profile_through_the_real_api_with_postgres(postgres_client: TestClient) -> None:
    with SALES_PATH.open("rb") as file:
        response = postgres_client.post(
            "/datasets/upload", files={"file": ("module3_sales.csv", file, "text/csv")}
        )
    assert response.status_code == 201, response.text
    dataset_id = response.json()["dataset_id"]

    profile = postgres_client.get(f"/datasets/{dataset_id}/profile")
    assert profile.status_code == 200
    assert profile.json()["row_count"] == 13


def test_unknown_dataset_is_still_a_404_with_postgres(postgres_client: TestClient) -> None:
    response = postgres_client.get("/datasets/123e4567-e89b-42d3-a456-426614174000/profile")

    assert response.status_code == 404
    assert response.json()["code"] == "DATASET_NOT_FOUND"


def test_a_fresh_client_still_finds_the_dataset(postgres_client: TestClient) -> None:
    """A NEW TestClient (a new dependency-injected repository per request) still finds
    the row, because the data lives in PostgreSQL, not in process memory -- the closest
    thing to a restart proof at the API layer. See Section 14 for the full, real,
    process-restart persistence test.
    """
    with SALES_PATH.open("rb") as file:
        upload = postgres_client.post(
            "/datasets/upload", files={"file": ("module3_sales.csv", file, "text/csv")}
        )
    dataset_id = upload.json()["dataset_id"]

    fresh_client = TestClient(app)  # shares the same app.dependency_overrides set above
    response = fresh_client.get(f"/datasets/{dataset_id}/profile")

    assert response.status_code == 200