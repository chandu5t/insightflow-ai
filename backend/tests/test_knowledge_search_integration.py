"""Integration test: the real pgvector similarity query against seeded, fake-embedded rows."""

from sqlalchemy import text

from app.services.knowledge_search_service import KnowledgeSearchService
from app.services.knowledge_seed_data import DEFAULT_SOURCE, SEED_DEFINITIONS
from app.services.knowledge_seed_service import seed_knowledge_base
from tests.db_fixtures import clean_postgres_tables, postgres_engine, postgres_session_factory, postgres_settings  # noqa: F401
from tests.fake_embeddings import FakeEmbeddings

AOV_TEXT = SEED_DEFINITIONS[0]["definition"]  # "average-order-value"


def _clean_knowledge_table(postgres_engine) -> None:
    with postgres_engine.begin() as connection:
        connection.execute(text("DELETE FROM knowledge_documents"))


def _seeded_embeddings() -> FakeEmbeddings:
    # The AOV definition and a query worded closely to it get a near-identical vector;
    # everything else gets an unrelated, hash-derived vector -- deterministic and
    # controllable for a precise ranking assertion.
    shared = [1.0, 0.0, 0.0, 0.0] + [0.0] * 764
    return FakeEmbeddings(
        vectors={AOV_TEXT: shared, "What does average order value mean?": shared},
        dimensions=768,
    )


def test_a_closely_worded_query_ranks_the_matching_definition_first(
    postgres_engine, postgres_session_factory, clean_postgres_tables
) -> None:
    _clean_knowledge_table(postgres_engine)
    embeddings = _seeded_embeddings()
    seed_knowledge_base(postgres_session_factory, embeddings)
    service = KnowledgeSearchService(postgres_session_factory, embeddings, top_k_limit=3, threshold=0.6)

    results = service.search("What does average order value mean?")

    assert results and results[0].slug == "average-order-value"
    assert results[0].similarity > 0.99  # identical vectors -> cosine distance ~0
    assert results[0].source == DEFAULT_SOURCE


def test_at_most_three_results_are_returned(
    postgres_engine, postgres_session_factory, clean_postgres_tables
) -> None:
    _clean_knowledge_table(postgres_engine)
    embeddings = _seeded_embeddings()
    seed_knowledge_base(postgres_session_factory, embeddings)
    service = KnowledgeSearchService(postgres_session_factory, embeddings, top_k_limit=3, threshold=-1.0)

    assert len(service.search("What does average order value mean?")) <= 3


def test_an_unrelated_query_returns_nothing_above_a_high_threshold(
    postgres_engine, postgres_session_factory, clean_postgres_tables
) -> None:
    _clean_knowledge_table(postgres_engine)
    embeddings = _seeded_embeddings()
    seed_knowledge_base(postgres_session_factory, embeddings)
    service = KnowledgeSearchService(postgres_session_factory, embeddings, top_k_limit=3, threshold=0.95)

    assert service.search("completely unrelated random text about weather") == []