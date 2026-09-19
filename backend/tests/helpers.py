"""Small helpers to create test files in memory."""

import io
from collections.abc import Sequence
from typing import Any

from openpyxl import Workbook

SALES_CSV = (
    "order_id,order_date,product,quantity,unit_price,region\n"
    "1001,2025-01-05,Laptop,2,55000,North\n"
    "1002,2025-01-06,Mouse,10,500,South\n"
    "1003,2025-01-07,Keyboard,5,1500,East\n"
).encode("utf-8")


def make_csv(rows: Sequence[Sequence[object]]) -> bytes:
    return ("\n".join(",".join(str(cell) for cell in row) for row in rows) + "\n").encode("utf-8")


def make_xlsx(
    rows: Sequence[Sequence[Any]],
    extra_sheet_rows: Sequence[Sequence[Any]] | None = None,
) -> bytes:
    """Create an .xlsx file in memory. The first sheet gets `rows`."""
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Sales"
    for row in rows:
        sheet.append(list(row))
    if extra_sheet_rows is not None:
        second = workbook.create_sheet("Other")
        for row in extra_sheet_rows:
            second.append(list(row))
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def csv_upload(content: bytes, filename: str = "sales.csv", content_type: str = "text/csv") -> dict:
    """Arguments for client.post(..., files=...)."""
    return {"file": (filename, content, content_type)}


XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"