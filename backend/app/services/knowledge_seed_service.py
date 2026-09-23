"""Idempotent seeding of the knowledge_documents table (Module 7, D-063).

Idempotency rule: a row is (re-)embedded only if it is NEW or its definition TEXT
changed since last seeded. An unchanged row is skipped entirely -- no database write,
no Gemini API call -- so running this repeatedly costs nothing after the first run.
"""

import logging
from collections.abc import Callable

from langchain_core.embeddings import Embeddings
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import KnowledgeDocumentRow
from app.services.knowledge_seed_data import DEFAULT_SOURCE, SEED_DEFINITIONS

logger = logging.getLogger(__name__)


def seed_knowledge_base(
    session_factory: Callable[[], Session], embedding_service: Embeddings
) -> dict[str, int]:
    """Insert/update the seed definitions. Returns counts: created, updated, skipped."""
    created = updated = skipped = 0
    with session_factory() as session, session.begin():
        for entry in SEED_DEFINITIONS:
            existing = session.execute(
                select(KnowledgeDocumentRow).where(KnowledgeDocumentRow.slug == entry["slug"])
            ).scalar_one_or_none()

            if existing is not None and existing.definition == entry["definition"]:
                skipped += 1
                continue

            [embedding] = embedding_service.embed_documents([entry["definition"]])

            if existing is None:
                session.add(
                    KnowledgeDocumentRow(
                        slug=entry["slug"], name=entry["name"], definition=entry["definition"],
                        category=entry["category"], source=DEFAULT_SOURCE, embedding=embedding,
                    )
                )
                created += 1
            else:
                existing.name = entry["name"]
                existing.definition = entry["definition"]
                existing.category = entry["category"]
                existing.embedding = embedding
                updated += 1

    logger.info("Knowledge base seeded: created=%d updated=%d skipped=%d", created, updated, skipped)
    return {"created": created, "updated": updated, "skipped": skipped}