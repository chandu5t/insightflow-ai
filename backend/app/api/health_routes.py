"""Health check route."""

from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.schemas.health_schema import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def health_check(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Confirm that the backend is running."""
    return HealthResponse(
        status="healthy",
        app=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
    )