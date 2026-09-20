"""Routes for uploading a dataset and previewing it. Logic is in dataset_service.py."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, Query, UploadFile

from app.core.config import Settings, get_settings
from app.schemas.dataset_schema import DatasetPreview, DatasetSummary
from app.schemas.error_schema import ErrorResponse
from app.schemas.profile_schema import DatasetProfile
from app.services import dataset_service, profiling_service
from app.services.dataset_repository import DatasetRepository, JsonDatasetRepository

router = APIRouter(prefix="/datasets", tags=["Datasets"])


def get_dataset_repository(settings: Settings = Depends(get_settings)) -> DatasetRepository:
    """Give routes the storage to use. Module 6 will return a PostgreSQL version here."""
    return JsonDatasetRepository(settings.upload_dir)


@router.post(
    "/upload",
    response_model=DatasetSummary,
    status_code=201,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
)
def upload_dataset(
    file: Annotated[UploadFile, File(description="A .csv or .xlsx file")],
    settings: Settings = Depends(get_settings),
    repository: DatasetRepository = Depends(get_dataset_repository),
) -> DatasetSummary:
    """Upload a CSV or XLSX file. It is validated, converted to CSV and stored."""
    return dataset_service.ingest_upload(
        filename=file.filename,
        content_type=file.content_type,
        stream=file.file,
        settings=settings,
        repository=repository,
    )


@router.get(
    "/{dataset_id}/preview",
    response_model=DatasetPreview,
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
def preview_dataset(
    dataset_id: str,
    rows: Annotated[int | None, Query(description="Number of rows to show")] = None,
    settings: Settings = Depends(get_settings),
    repository: DatasetRepository = Depends(get_dataset_repository),
) -> DatasetPreview:
    """Return the column names and the first rows of a dataset (default 5, maximum 20)."""
    return dataset_service.get_dataset_preview(
        raw_dataset_id=dataset_id,
        rows=rows,
        settings=settings,
        repository=repository,
    )


@router.get(
    "/{dataset_id}/profile",
    response_model=DatasetProfile,
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
def profile_dataset(
    dataset_id: str,
    settings: Settings = Depends(get_settings),
    repository: DatasetRepository = Depends(get_dataset_repository),
) -> DatasetProfile:
    """Return the profile of a dataset: types, missing values, statistics and limited value lists."""
    return profiling_service.get_dataset_profile(
        raw_dataset_id=dataset_id,
        settings=settings,
        repository=repository,
    )