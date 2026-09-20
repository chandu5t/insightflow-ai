"""Builds the dataset profile: repository -> Pandas -> profiling tool."""

import logging

from app.core.config import Settings
from app.schemas.profile_schema import DatasetProfile
from app.services.dataset_repository import DatasetRepository
from app.tools.profiling_tool import profile_dataframe
from app.utils.dataframe_utils import load_dataframe
from app.utils.upload_validator import parse_dataset_id

logger = logging.getLogger(__name__)


def get_dataset_profile(
    *,
    raw_dataset_id: str,
    settings: Settings,
    repository: DatasetRepository,
) -> DatasetProfile:
    """Return the profile of a stored dataset (same errors as the preview: 422 and 404)."""
    dataset_id = parse_dataset_id(raw_dataset_id)
    metadata = repository.get(dataset_id)
    frame = load_dataframe(repository.get_csv_path(dataset_id))

    profile = profile_dataframe(
        frame,
        top_values=settings.profile_top_values,
        value_max_length=settings.profile_value_max_length,
    )
    # Log only technical facts, never values.
    logger.info(
        "Dataset profiled: id=%s rows=%d columns=%d",
        dataset_id,
        profile.row_count,
        profile.column_count,
    )
    return DatasetProfile(
        dataset_id=metadata.dataset_id,
        filename=metadata.filename,
        source_format=metadata.source_format,
        **profile.model_dump(),
    )