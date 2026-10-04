"""Fixtures used only by the V2.7 evaluation tests."""

from datetime import datetime, timezone
from uuid import UUID

from app.schemas.dataset_schema import DatasetMetadata
from tests.helpers import SALES_CSV


def seed_evaluation_dataset(repo, dataset_id: UUID, *, content: bytes = SALES_CSV, filename: str = "sales_01.csv") -> UUID:
    """Seed the existing dataset repository through its public V1 interface."""
    temp_path = repo.new_temp_path(dataset_id)
    temp_path.write_bytes(content)
    metadata = DatasetMetadata(
        dataset_id=dataset_id,
        filename=filename,
        source_format="csv",
        row_count=3,
        column_count=6,
        column_names=["order_id", "order_date", "product", "quantity", "unit_price", "region"],
        original_size_bytes=len(content),
        uploaded_at=datetime.now(timezone.utc),
    )
    repo.save(metadata, temp_path)
    return dataset_id
