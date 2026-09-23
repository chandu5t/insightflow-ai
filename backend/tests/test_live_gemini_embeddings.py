"""LIVE Gemini embedding checks. Real network calls. Skipped automatically without a key.

Not part of the normal test run's expectations -- run explicitly and only when you
want to confirm the real API still behaves as documented.
"""

import math

import pytest

from app.core.config import get_settings
from app.services.embedding_service import GeminiEmbeddingService

pytestmark = pytest.mark.skipif(
    not get_settings().gemini_api_key_value, reason="GEMINI_API_KEY is not set; skipping live Gemini checks."
)


@pytest.fixture
def live_service():
    settings = get_settings()
    return GeminiEmbeddingService(
        api_key=settings.gemini_api_key_value, model=settings.gemini_embedding_model,
        dimensions=settings.knowledge_embedding_dimensions, timeout_seconds=settings.gemini_timeout_seconds,
    )


def test_embed_query_returns_a_normalized_vector_of_the_configured_size(live_service) -> None:
    vector = live_service.embed_query("What is average order value?")

    assert len(vector) == get_settings().knowledge_embedding_dimensions
    norm = math.sqrt(sum(v * v for v in vector))
    assert norm == pytest.approx(1.0, abs=1e-3)


def test_similar_texts_are_closer_than_unrelated_texts(live_service) -> None:
    a, b, c = live_service.embed_documents([
        "Average order value is total revenue divided by number of orders.",
        "AOV measures how much a customer spends per order on average.",
        "The weather today is sunny with a light breeze.",
    ])

    def cosine(x, y):
        return sum(p * q for p, q in zip(x, y))  # unit vectors -> dot product == cosine similarity

    assert cosine(a, b) > cosine(a, c)