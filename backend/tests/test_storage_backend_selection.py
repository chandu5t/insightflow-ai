"""Unit tests: STORAGE_BACKEND selects the right repository WITHOUT needing a real database.

get_dataset_repository()/get_analysis_history_repository() only CONSTRUCT a Postgres
repository when storage_backend == "postgres"; construction does not open a connection
(SQLAlchemy engines/sessions are lazy), so these are safe in the ordinary unit-test suite.
"""

from app.api.analysis_routes import get_analysis_history_repository
from app.api.dataset_routes import get_dataset_repository
from app.core.config import Settings
from app.services.analysis_history_repository import (
    NullAnalysisHistoryRepository,
    PostgresAnalysisHistoryRepository,
)
from app.services.dataset_repository import JsonDatasetRepository
from app.services.postgres_dataset_repository import PostgresDatasetRepository


def test_default_storage_backend_is_json(tmp_path) -> None:
    settings = Settings(gemini_api_key=None, upload_dir=tmp_path)

    assert settings.storage_backend == "json"
    assert isinstance(get_dataset_repository(settings), JsonDatasetRepository)
    assert isinstance(get_analysis_history_repository(settings), NullAnalysisHistoryRepository)


def test_postgres_backend_selects_the_postgres_repository_without_connecting(tmp_path) -> None:
    settings = Settings(gemini_api_key=None, upload_dir=tmp_path, storage_backend="postgres")

    assert isinstance(get_dataset_repository(settings), PostgresDatasetRepository)
    assert isinstance(get_analysis_history_repository(settings), PostgresAnalysisHistoryRepository)


def test_database_url_is_built_from_the_individual_settings() -> None:
    settings = Settings(
        gemini_api_key=None, storage_backend="postgres", postgres_user="u",
        postgres_password="p", postgres_host="h", postgres_port=1234, postgres_db="d",
    )

    assert settings.database_url == "postgresql+psycopg://u:p@h:1234/d"


def test_password_is_never_shown_in_settings_repr() -> None:
    settings = Settings(gemini_api_key=None, postgres_password="sk-SECRET")

    assert "sk-SECRET" not in repr(settings) and "sk-SECRET" not in str(settings)