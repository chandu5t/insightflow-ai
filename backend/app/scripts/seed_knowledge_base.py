"""Run this once (or any time the seed definitions change) to populate knowledge_documents.

Usage (from backend/, venv active):
    python -m app.scripts.seed_knowledge_base
"""

import sys

from app.core.config import get_settings
from app.db.engine import get_session_factory
from app.services.embedding_service import GeminiEmbeddingService
from app.services.knowledge_seed_service import seed_knowledge_base


def main() -> None:
    settings = get_settings()
    if settings.storage_backend != "postgres":
        print("STORAGE_BACKEND is not 'postgres'. Seeding requires PostgreSQL.", file=sys.stderr)
        raise SystemExit(1)
    if not settings.gemini_api_key_value:
        print("GEMINI_API_KEY is not set. Seeding requires it to generate embeddings.", file=sys.stderr)
        raise SystemExit(1)

    embedding_service = GeminiEmbeddingService(
        api_key=settings.gemini_api_key_value,
        model=settings.gemini_embedding_model,
        dimensions=settings.knowledge_embedding_dimensions,
        timeout_seconds=settings.gemini_timeout_seconds,
    )
    result = seed_knowledge_base(get_session_factory(settings), embedding_service)
    print(f"Seeding complete: {result}")


if __name__ == "__main__":
    main()