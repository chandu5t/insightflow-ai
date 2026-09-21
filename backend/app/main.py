"""FastAPI application entry point."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.dataset_routes import router as dataset_router
from app.api.analysis_routes import router as analysis_router
from app.api.health_routes import router as health_router
from app.core.config import get_settings
from app.core.errors import register_exception_handlers
from app.core.logging_config import setup_logging

settings = get_settings()
setup_logging(settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "AI-powered business data analysis assistant. "
        "Python tools calculate; the LLM explains validated results."
    ),
)

# CORS: allow only the configured frontend origins to call this API from a browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

register_exception_handlers(app)

app.include_router(health_router)
app.include_router(dataset_router)
app.include_router(analysis_router)

logger.info(
    "%s v%s configured (environment=%s, allowed origins=%s)",
    settings.app_name,
    settings.app_version,
    settings.environment,
    settings.cors_origins_list,
)