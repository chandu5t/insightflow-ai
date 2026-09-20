"""API tests for GET /datasets/{id}/profile (and regression tests for the Module 2 routes)."""

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.schemas.profile_schema import DatasetProfile
from tests.frames import DIRECT_PATH, SALES_PATH
from tests.helpers import XLSX_MIME, csv_upload, make_xlsx

UNKNOWN_ID = "123e4567-e89b-42d3-a456-426614174000"
MakeClient = Callable[..., TestClient]


def upload_file(client: TestClient, path=SALES_PATH) -> str:
    response = client.post("/datasets/upload", files=csv_upload(path.read_bytes(), path.name))
    assert response.status_code == 201, response.text
    return response.json()["dataset_id"]


def test_profile_success_and_schema(client: TestClient) -> None:
    dataset_id = upload_file(client)

    response = client.get(f"/datasets/{dataset_id}/profile")

    assert response.status_code == 200
    profile = DatasetProfile.model_validate(response.json())  # matches the documented schema
    assert str(profile.dataset_id) == dataset_id and profile.filename == "module3_sales.csv"
    assert profile.source_format == "csv"
    assert (profile.row_count, profile.column_count, profile.duplicate_row_count) == (13, 7, 1)
    assert profile.total_missing_cells == 1


def test_profile_response_fields() -> None:
    assert set(DatasetProfile.model_fields) == {
        "row_count", "column_count", "column_names", "duplicate_row_count", "total_missing_cells",
        "columns", "column_mapping", "warnings", "dataset_id", "filename", "source_format",
    }


def test_profile_columns_and_statistics(client: TestClient) -> None:
    body = client.get(f"/datasets/{upload_file(client)}/profile").json()
    columns = {column["name"]: column for column in body["columns"]}

    assert columns["quantity"]["data_type"] == "integer"
    assert columns["quantity"]["numeric_stats"]["max"] == 5.0
    assert columns["region"]["missing_count"] == 1 and columns["region"]["missing_percentage"] == 7.69
    assert columns["product"]["categorical"]["unique_count"] == 5
    assert columns["order_date"]["datetime_stats"] == {"min": "2025-01-05", "max": "2025-01-14"}
    assert body["column_mapping"]["missing"] == ["revenue"]


def test_profile_is_json_safe_deterministic_and_has_no_rows_or_paths(client: TestClient, upload_dir: Path) -> None:
    dataset_id = upload_file(client)

    first = client.get(f"/datasets/{dataset_id}/profile")
    second = client.get(f"/datasets/{dataset_id}/profile")

    assert first.json() == second.json()
    json.loads(first.text, parse_constant=lambda name: pytest.fail(f"non-JSON value {name}"))
    assert "file_path" not in first.text and str(upload_dir) not in first.text and "uploads" not in first.text
    assert "rows" not in first.json() and "preview_rows" not in first.json()


def test_profile_limits_come_from_settings(make_client: MakeClient) -> None:
    client = make_client(profile_top_values=2, profile_value_max_length=10)
    body = client.get(f"/datasets/{upload_file(client)}/profile").json()
    product = next(column for column in body["columns"] if column["name"] == "product")

    assert len(product["categorical"]["top_values"]) == 2
    assert product["categorical"]["values_truncated"] is True and product["categorical"]["unique_count"] == 5


def test_profile_of_an_xlsx_upload(client: TestClient) -> None:
    content = make_xlsx([["Product", "Qty"], ["Pen", 5], ["Book", 7]])
    dataset_id = client.post("/datasets/upload", files={"file": ("s.xlsx", content, XLSX_MIME)}).json()["dataset_id"]

    body = client.get(f"/datasets/{dataset_id}/profile").json()

    assert body["source_format"] == "xlsx" and body["row_count"] == 2
    assert body["column_mapping"]["resolved"] == {"product": "Product", "quantity": "Qty"}


def test_profile_of_the_direct_revenue_dataset(client: TestClient) -> None:
    body = client.get(f"/datasets/{upload_file(client, DIRECT_PATH)}/profile").json()

    assert body["column_mapping"]["resolved"]["revenue"] == "Total Revenue"
    assert body["column_mapping"]["resolved"]["order_id"] == "Order ID"


def test_profile_reports_ambiguous_mapping_as_a_warning(client: TestClient) -> None:
    dataset_id = client.post("/datasets/upload", files=csv_upload(b"revenue,sales\n1,2\n")).json()["dataset_id"]

    body = client.get(f"/datasets/{dataset_id}/profile").json()

    assert body["column_mapping"]["ambiguous"] == {"revenue": ["revenue", "sales"]}
    assert any("More than one column could be 'revenue'" in warning for warning in body["warnings"])


def test_profile_of_an_unknown_dataset_is_404(client: TestClient) -> None:
    response = client.get(f"/datasets/{UNKNOWN_ID}/profile")

    assert response.status_code == 404
    assert response.json() == {"error": "Dataset not found.", "code": "DATASET_NOT_FOUND", "details": {}}


@pytest.mark.parametrize("bad_id", ["not-a-uuid", "dataset_001", "..%5C..%5Csecret", "%2e%2e"])
def test_profile_with_an_invalid_id_is_422(client: TestClient, bad_id: str) -> None:
    response = client.get(f"/datasets/{bad_id}/profile")

    assert response.status_code == 422
    assert response.json()["code"] == "INVALID_DATASET_ID"


def test_profile_rejects_other_methods(client: TestClient) -> None:
    response = client.post(f"/datasets/{UNKNOWN_ID}/profile")

    assert response.status_code == 405 and response.json()["code"] == "METHOD_NOT_ALLOWED"


# ---- regression: Module 1 and 2 routes still work -------------------------------------------------
def test_existing_routes_still_work(client: TestClient) -> None:
    dataset_id = upload_file(client)

    assert client.get("/health").status_code == 200
    preview = client.get(f"/datasets/{dataset_id}/preview")
    assert preview.status_code == 200 and len(preview.json()["preview_rows"]) == 5
    assert client.get(f"/datasets/{dataset_id}/preview", params={"rows": 21}).status_code == 422
    assert client.post("/datasets/upload", files=csv_upload(b"")).json()["code"] == "EMPTY_FILE"