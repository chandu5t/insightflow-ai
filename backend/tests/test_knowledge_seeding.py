"""Integration tests for idempotent seeding. Real PostgreSQL, fake (network-free) embeddings."""

from sqlalchemy import select

from app.db.models import KnowledgeDocumentRow
from app.services.knowledge_seed_data import SEED_DEFINITIONS
from app.services.knowledge_seed_service import seed_knowledge_base
from tests.db_fixtures import clean_postgres_tables, postgres_engine, postgres_session_factory, postgres_settings  # noqa: F401
from tests.fake_embeddings import FakeEmbeddings


def _clean_knowledge_table(postgres_engine) -> None:
    from sqlalchemy import text
    with postgres_engine.begin() as connection:
        connection.execute(text("DELETE FROM knowledge_documents"))


def test_first_run_creates_all_definitions_with_no_duplicates(
    postgres_engine, postgres_session_factory, clean_postgres_tables
) -> None:
    _clean_knowledge_table(postgres_engine)
    embeddings = FakeEmbeddings(dimensions=768)

    result = seed_knowledge_base(postgres_session_factory, embeddings)

    assert result == {"created": len(SEED_DEFINITIONS), "updated": 0, "skipped": 0}
    with postgres_session_factory() as session:
        assert session.query(KnowledgeDocumentRow).count() == len(SEED_DEFINITIONS)


def test_second_run_skips_unchanged_rows_and_makes_no_embedding_calls(
    postgres_engine, postgres_session_factory, clean_postgres_tables
) -> None:
    _clean_knowledge_table(postgres_engine)
    embeddings = FakeEmbeddings(dimensions=768)
    seed_knowledge_base(postgres_session_factory, embeddings)

    call_count = {"n": 0}
    original = embeddings.embed_documents

    def counting_embed(texts):
        call_count["n"] += 1
        return original(texts)

    embeddings.embed_documents = counting_embed
    result = seed_knowledge_base(postgres_session_factory, embeddings)

    assert result == {"created": 0, "updated": 0, "skipped": len(SEED_DEFINITIONS)}
    assert call_count["n"] == 0
    with postgres_session_factory() as session:
        assert session.query(KnowledgeDocumentRow).count() == len(SEED_DEFINITIONS)  # still no duplicates


def test_a_changed_definition_is_updated_not_duplicated(
    postgres_engine, postgres_session_factory, clean_postgres_tables
) -> None:
    _clean_knowledge_table(postgres_engine)
    embeddings = FakeEmbeddings(dimensions=768)
    seed_knowledge_base(postgres_session_factory, embeddings)

    with postgres_session_factory() as session, session.begin():
        row = session.execute(
            select(KnowledgeDocumentRow).where(KnowledgeDocumentRow.slug == "average-order-value")
        ).scalar_one()
        row.definition = "An intentionally different definition text."

    result = seed_knowledge_base(postgres_session_factory, embeddings)

    assert result["updated"] == 1
    with postgres_session_factory() as session:
        assert session.query(KnowledgeDocumentRow).count() == len(SEED_DEFINITIONS)