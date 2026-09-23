import uuid

from app.core.config import get_settings
from app.db.models import KnowledgeDocumentRow


def test_table_name_and_columns() -> None:
    assert KnowledgeDocumentRow.__tablename__ == "knowledge_documents"
    columns = {c.name for c in KnowledgeDocumentRow.__table__.columns}
    assert columns == {
        "id", "slug", "name", "definition", "category", "source",
        "embedding", "doc_metadata", "created_at", "updated_at",
    }


def test_slug_is_unique_and_indexed() -> None:
    slug_column = KnowledgeDocumentRow.__table__.columns["slug"]
    assert slug_column.unique and slug_column.index


def test_embedding_dimension_matches_settings() -> None:
    settings = get_settings()
    assert KnowledgeDocumentRow.__table__.columns["embedding"].type.dim == settings.knowledge_embedding_dimensions


def test_can_be_instantiated_without_a_database() -> None:
    row = KnowledgeDocumentRow(
        slug="test-metric", name="Test Metric", definition="A test.", category="test",
        source="test", embedding=[0.1, 0.2, 0.3, 0.4],
    )
    assert row.slug == "test-metric"