"""Application settings loaded from environment variables and backend/.env."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# config.py lives in backend/app/core/, so parents[2] is the backend/ folder.
BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "InsightFlow AI"
    app_version: str = "0.1.0"
    environment: str = "development"
    log_level: str = "INFO"
    # Stored as a comma-separated string (simple to write in .env).
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origins_list(self) -> list[str]:
        """Return CORS origins as a clean list, ignoring blanks and extra spaces."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Create the Settings object once and reuse it everywhere."""
    return Settings()