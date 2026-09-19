"""API tests for POST /datasets/upload and GET /datasets/{id}/preview."""

import csv
import json
from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.helpers import SALES_CSV, XLSX_MIME, csv_upload, make_csv, make_xlsx

MakeClient = Callable[..., TestClient]
UNKNOWN_ID = "123e4567-e89b-42d3-a456-426614174000"


def upload_csv(client: TestClient, content: bytes = SALES_CSV, filename: str = "sales.csv") -> dict:
    response = client.post("/datasets/upload", files=csv_upload(content, filename))
    assert response.status_code == 201, response.text
    return response.json()


def assert_error(response, status: int, code: str) -> dict:
    assert response.status_code == status, response.text
    body = response.json()
    assert set(body) == {"error", "code", "details"}  # the standard error format
    assert body["code"] == code
    assert body["error"]
    return body


# ============================== Upload: success ================================
def test_valid_csv_upload(client: TestClient) -> None:
    body = upload_csv(client)

    assert body["filename"] == "sales.csv"
    assert body["source_format"] == "csv"
    assert body["row_count"] == 3
    assert body["column_count"] == 6
    assert body["column_names"][:3] == ["order_id", "order_date", "product"]
    assert body["status"] == "uploaded"
    assert body["warnings"] == []
    assert "file_path" not in body and "path" not in json.dumps(body).lower()


def test_valid_xlsx_upload(client: TestClient) -> None:
    content = make_xlsx([["product", "qty"], ["Pen", 5], ["Book", 7]])

    response = client.post("/datasets/upload", files={"file": ("stock.xlsx", content, XLSX_MIME)})

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["source_format"] == "xlsx"
    assert body["row_count"] == 2
    assert body["column_names"] == ["product", "qty"]


def test_xlsx_with_two_sheets_returns_warning(client: TestClient) -> None:
    content = make_xlsx([["a"], [1]], extra_sheet_rows=[["b"], [2]])

    response = client.post("/datasets/upload", files={"file": ("book.xlsx", content, XLSX_MIME)})

    assert response.status_code == 201
    assert "Only the first sheet" in response.json()["warnings"][0]


def test_dataset_ids_are_unique_uuids(client: TestClient) -> None:
    first = upload_csv(client)["dataset_id"]
    second = upload_csv(client)["dataset_id"]

    assert first != second
    assert len(first) == 36


# ============================== Upload: rejections =============================
@pytest.mark.parametrize("filename", ["old.xls", "macro.xlsm"])
def test_xls_and_xlsm_are_rejected(client: TestClient, filename: str) -> None:
    response = client.post("/datasets/upload", files=csv_upload(b"data", filename, "application/octet-stream"))

    assert_error(response, 400, "UNSUPPORTED_FILE_TYPE")


@pytest.mark.parametrize("filename", ["notes.txt", "virus.exe", "data.json", "noextension"])
def test_unsupported_extensions_are_rejected(client: TestClient, filename: str) -> None:
    response = client.post("/datasets/upload", files=csv_upload(b"a,b\n1,2\n", filename))

    assert_error(response, 400, "UNSUPPORTED_FILE_TYPE")


def test_wrong_mime_type_is_rejected(client: TestClient) -> None:
    response = client.post("/datasets/upload", files=csv_upload(SALES_CSV, "sales.csv", "image/png"))

    assert_error(response, 400, "UNSUPPORTED_FILE_TYPE")


def test_empty_file_is_rejected(client: TestClient) -> None:
    response = client.post("/datasets/upload", files=csv_upload(b""))

    assert_error(response, 400, "EMPTY_FILE")


def test_corrupted_xlsx_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/datasets/upload", files={"file": ("broken.xlsx", b"PK\x03\x04garbage", XLSX_MIME)}
    )

    assert_error(response, 400, "CORRUPTED_FILE")


def test_text_file_named_xlsx_is_rejected(client: TestClient) -> None:
    response = client.post("/datasets/upload", files={"file": ("fake.xlsx", b"a,b\n1,2\n", XLSX_MIME)})

    assert_error(response, 400, "CORRUPTED_FILE")


def test_missing_headers_are_rejected(client: TestClient) -> None:
    response = client.post("/datasets/upload", files=csv_upload(b"a,,c\n1,2,3\n"))

    body = assert_error(response, 400, "MISSING_HEADERS")
    assert body["details"]["blank_columns"] == [2]


def test_duplicate_columns_are_rejected(client: TestClient) -> None:
    response = client.post("/datasets/upload", files=csv_upload(b"a,A\n1,2\n"))

    assert_error(response, 400, "DUPLICATE_COLUMNS")


def test_file_with_no_data_rows_is_rejected(client: TestClient) -> None:
    response = client.post("/datasets/upload", files=csv_upload(b"a,b,c\n"))

    assert_error(response, 400, "NO_DATA_ROWS")


def test_invalid_encoding_is_rejected(client: TestClient) -> None:
    response = client.post("/datasets/upload", files=csv_upload(b"a,b\n\x81,2\n"))

    assert_error(response, 400, "INVALID_ENCODING")


def test_file_larger_than_size_limit_is_rejected(make_client: MakeClient) -> None:
    client = make_client(max_upload_size_mb=0.001)  # about 1 KB

    response = client.post("/datasets/upload", files=csv_upload(b"a,b\n" + b"1,2\n" * 1000))

    body = assert_error(response, 413, "FILE_TOO_LARGE")
    assert body["details"]["max_bytes"] == 1048


def test_file_within_size_limit_is_accepted(make_client: MakeClient) -> None:
    client = make_client(max_upload_size_mb=0.001)

    upload_csv(client, b"a,b\n1,2\n")


def test_row_and_column_limits(make_client: MakeClient) -> None:
    client = make_client(max_rows=2, max_columns=3)

    too_many_rows = client.post("/datasets/upload", files=csv_upload(make_csv([["a"], [1], [2], [3]])))
    too_many_columns = client.post("/datasets/upload", files=csv_upload(make_csv([list("abcd"), [1, 2, 3, 4]])))

    assert_error(too_many_rows, 400, "TOO_MANY_ROWS")
    assert_error(too_many_columns, 400, "TOO_MANY_COLUMNS")


def test_missing_file_field_returns_validation_error(client: TestClient) -> None:
    response = client.post("/datasets/upload", data={"other": "value"})

    body = assert_error(response, 422, "VALIDATION_ERROR")
    assert body["details"]["errors"][0]["field"].endswith("file")


def test_rejected_upload_leaves_no_files_behind(client: TestClient, upload_dir: Path) -> None:
    client.post("/datasets/upload", files=csv_upload(b"a,,c\n1,2,3\n"))
    client.post("/datasets/upload", files=csv_upload(b"a,b\n"))

    assert list(upload_dir.iterdir()) == []


# ============================== Metadata and storage ===========================
def test_dataset_metadata_and_clean_csv_are_stored(client: TestClient, upload_dir: Path) -> None:
    body = upload_csv(client)
    dataset_id = body["dataset_id"]

    metadata = json.loads((upload_dir / f"{dataset_id}.meta.json").read_text(encoding="utf-8"))
    assert metadata["dataset_id"] == dataset_id
    assert metadata["filename"] == "sales.csv"
    assert metadata["source_format"] == "csv"
    assert metadata["row_count"] == 3
    assert metadata["column_count"] == 6
    assert metadata["original_size_bytes"] == len(SALES_CSV)
    assert "uploaded_at" in metadata
    assert "file_path" not in metadata

    with (upload_dir / f"{dataset_id}.csv").open(encoding="utf-8", newline="") as file:
        stored_rows = list(csv.reader(file))
    assert len(stored_rows) == 4  # header + 3 data rows
    assert sorted(path.name for path in upload_dir.iterdir()) == [f"{dataset_id}.csv", f"{dataset_id}.meta.json"]


def test_xlsx_is_stored_as_csv(client: TestClient, upload_dir: Path) -> None:
    content = make_xlsx([["a", "b"], [1, "x"]])
    response = client.post("/datasets/upload", files={"file": ("book.xlsx", content, XLSX_MIME)})
    dataset_id = response.json()["dataset_id"]

    stored = (upload_dir / f"{dataset_id}.csv").read_text(encoding="utf-8")

    assert stored.splitlines() == ["a,b", "1,x"]
    assert not list(upload_dir.glob("*.xlsx"))


# ============================== Path traversal =================================
@pytest.mark.parametrize("filename", ["../../evil.csv", "..\\..\\evil.csv", "/etc/passwd.csv"])
def test_path_traversal_in_filename_is_neutralised(
    client: TestClient, upload_dir: Path, filename: str
) -> None:
    body = upload_csv(client, filename=filename)

    assert "/" not in body["filename"] and "\\" not in body["filename"] and ".." not in body["filename"]
    # Only our two files exist, inside the upload folder, named after the dataset id.
    assert sorted(path.name for path in upload_dir.iterdir()) == [
        f"{body['dataset_id']}.csv",
        f"{body['dataset_id']}.meta.json",
    ]
    assert not (upload_dir.parent / "evil.csv").exists()
    assert not (upload_dir.parent.parent / "evil.csv").exists()


@pytest.mark.parametrize("bad_id", ["..%5C..%5Csecret", "not-a-uuid", "dataset_001", "%2e%2e", "..%2F..%2Fetc%2Fpasswd"])
def test_invalid_or_traversal_dataset_id_is_rejected(client: TestClient, bad_id: str) -> None:
    response = client.get(f"/datasets/{bad_id}/preview")

    # 422 for a malformed id. 404 is also safe, if the URL never reaches our route.
    assert response.status_code in {404, 422}
    if response.status_code == 422:
        assert response.json()["code"] == "INVALID_DATASET_ID"


def test_unknown_dataset_returns_404(client: TestClient) -> None:
    response = client.get(f"/datasets/{UNKNOWN_ID}/preview")

    assert_error(response, 404, "DATASET_NOT_FOUND")


# ============================== Preview ========================================
def test_preview_returns_five_rows_by_default(client: TestClient) -> None:
    content = make_csv([["n", "v"]] + [[i, f"row{i}"] for i in range(1, 31)])
    dataset_id = upload_csv(client, content)["dataset_id"]

    body = client.get(f"/datasets/{dataset_id}/preview").json()

    assert len(body["preview_rows"]) == 5
    assert body["requested_rows"] == 5
    assert body["preview_rows"][0] == ["1", "row1"]
    assert body["row_count"] == 30
    assert body["column_count"] == 2
    assert body["column_names"] == ["n", "v"]
    assert body["filename"] == "sales.csv"
    assert body["source_format"] == "csv"
    assert body["cells_truncated"] is False


def test_preview_allows_up_to_twenty_rows(client: TestClient) -> None:
    content = make_csv([["n"]] + [[i] for i in range(1, 31)])
    dataset_id = upload_csv(client, content)["dataset_id"]

    body = client.get(f"/datasets/{dataset_id}/preview", params={"rows": 20}).json()

    assert len(body["preview_rows"]) == 20


def test_preview_over_the_maximum_is_rejected(client: TestClient) -> None:
    dataset_id = upload_csv(client)["dataset_id"]

    response = client.get(f"/datasets/{dataset_id}/preview", params={"rows": 21})

    body = assert_error(response, 422, "INVALID_PREVIEW_ROWS")
    assert body["details"]["max_rows"] == 20


@pytest.mark.parametrize("rows", [0, -1])
def test_preview_with_zero_or_negative_rows_is_rejected(client: TestClient, rows: int) -> None:
    dataset_id = upload_csv(client)["dataset_id"]

    response = client.get(f"/datasets/{dataset_id}/preview", params={"rows": rows})

    assert_error(response, 422, "INVALID_PREVIEW_ROWS")


def test_preview_with_non_number_rows_is_a_validation_error(client: TestClient) -> None:
    dataset_id = upload_csv(client)["dataset_id"]

    response = client.get(f"/datasets/{dataset_id}/preview", params={"rows": "many"})

    assert_error(response, 422, "VALIDATION_ERROR")


def test_preview_shows_fewer_rows_when_the_file_is_small(client: TestClient) -> None:
    dataset_id = upload_csv(client)["dataset_id"]  # only 3 data rows

    body = client.get(f"/datasets/{dataset_id}/preview", params={"rows": 10}).json()

    assert len(body["preview_rows"]) == 3


def test_preview_limits_come_from_settings(make_client: MakeClient) -> None:
    client = make_client(preview_default_rows=2, preview_max_rows=3)
    dataset_id = upload_csv(client)["dataset_id"]

    assert len(client.get(f"/datasets/{dataset_id}/preview").json()["preview_rows"]) == 2
    assert_error(client.get(f"/datasets/{dataset_id}/preview", params={"rows": 4}), 422, "INVALID_PREVIEW_ROWS")


def test_long_cells_are_truncated(client: TestClient) -> None:
    content = make_csv([["note"], ["x" * 500], ["short"]])
    dataset_id = upload_csv(client, content)["dataset_id"]

    body = client.get(f"/datasets/{dataset_id}/preview").json()

    long_cell = body["preview_rows"][0][0]
    assert len(long_cell) == body["max_cell_length"] == 100
    assert long_cell.endswith("…")
    assert body["preview_rows"][1] == ["short"]
    assert body["cells_truncated"] is True


def test_empty_cells_are_returned_as_null(client: TestClient) -> None:
    dataset_id = upload_csv(client, b"a,b\n1,\n,2\n")["dataset_id"]

    body = client.get(f"/datasets/{dataset_id}/preview").json()

    assert body["preview_rows"] == [["1", None], [None, "2"]]


def test_xlsx_preview_works(client: TestClient) -> None:
    content = make_xlsx([["product", "qty"], ["Pen", 5]])
    response = client.post("/datasets/upload", files={"file": ("s.xlsx", content, XLSX_MIME)})

    body = client.get(f"/datasets/{response.json()['dataset_id']}/preview").json()

    assert body["source_format"] == "xlsx"
    assert body["preview_rows"] == [["Pen", "5"]]