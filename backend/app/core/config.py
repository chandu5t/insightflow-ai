"""Application settings loaded from environment variables and backend/.env."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# config.py lives in backend/app/core/, so parents[2] is the backend/ folder.
BACKEND_DIR = Path(__file__).resolve().parents[2]

BYTES_PER_MB = 1024 * 1024


class Settings(BaseSettings):
    app_name: str = "InsightFlow AI"
    app_version: str = "0.1.0"
    environment: str = "development"
    log_level: str = "INFO"
    # Stored as a comma-separated string (simple to write in .env).
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # ---- Data ingestion (Module 2) ----
    # Where uploaded datasets are stored. A relative path is resolved from backend/.
    upload_dir: Path = BACKEND_DIR / "data" / "uploads"
    # Maximum size of the uploaded file itself.
    max_upload_size_mb: float = Field(default=10, gt=0)
    # Safety limit for an .xlsx file after it is unzipped (protects against zip bombs).
    max_xlsx_uncompressed_mb: float = Field(default=100, gt=0)
    # Limits on the table inside the file.
    max_rows: int = Field(default=100_000, gt=0)
    max_columns: int = Field(default=100, gt=0)
    # Preview endpoint limits.
    preview_default_rows: int = Field(default=5, gt=0)
    preview_max_rows: int = Field(default=20, gt=0)
    preview_max_cell_length: int = Field(default=100, ge=10)
    # ---- Profiling (Module 3) ----
    # Number of most common values shown per column.
    profile_top_values: int = Field(default=10, gt=0)
    # Maximum length of values displayed in the profile.
    profile_value_max_length: int = Field(default=50, ge=10)

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("upload_dir")
    @classmethod
    def make_upload_dir_absolute(cls, value: Path) -> Path:
        """Resolve a relative UPLOAD_DIR from the backend folder, not the current folder."""
        return value if value.is_absolute() else BACKEND_DIR / value

    @model_validator(mode="after")
    def check_preview_limits(self) -> "Settings":
        if self.preview_default_rows > self.preview_max_rows:
            raise ValueError("PREVIEW_DEFAULT_ROWS cannot be greater than PREVIEW_MAX_ROWS")
        return self

    @property
    def cors_origins_list(self) -> list[str]:
        """Return CORS origins as a clean list, ignoring blanks and extra spaces."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def max_upload_size_bytes(self) -> int:
        return int(self.max_upload_size_mb * BYTES_PER_MB)

    @property
    def max_xlsx_uncompressed_bytes(self) -> int:
        return int(self.max_xlsx_uncompressed_mb * BYTES_PER_MB)


@lru_cache
def get_settings() -> Settings:
    """Create the Settings object once and reuse it everywhere."""
    return Settings()