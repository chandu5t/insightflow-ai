"""Top-K similarity search over knowledge_documents, using explicit SQL (not ORM query building)."""

import logging
from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.embeddings import Embeddings
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.errors import AppError, ErrorCode

logger = logging.getLogger(__name__)

# `embedding <=> CAST(:query_embedding AS vector)` is pgvector's cosine DISTANCE
# (0 = identical, 2 = opposite). We convert to a "higher is better" similarity score
# via 1 - distance, which lands in roughly [0, 1] for related business text.
# The query embedding is bound as a PARAMETER (a pgvector text literal), never
# interpolated into the SQL string -- this is the parameterized-query requirement.
_SEARCH_SQL = text(
    """
    SELECT id, slug, name, definition, category, source,
           1 - (embedding <=> CAST(:query_embedding AS vector)) AS similarity
    FROM knowledge_documents
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> CAST(:query_embedding AS vector)
    LIMIT :top_k
    """
)


def _to_pgvector_literal(values: list[float]) -> str:
    """`[0.1, -0.2, ...]` -> `'[0.1,-0.2,...]'`, the text format pgvector's CAST expects."""
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


@dataclass(frozen=True)
class KnowledgeSearchResult:
    id: str
    slug: str
    name: str
    definition: str
    category: str
    source: str
    similarity: float


class KnowledgeSearchService:
    def __init__(
        self,
        session_factory: Callable[[], Session],
        embedding_service: Embeddings,
        *,
        top_k_limit: int,
        threshold: float,
    ) -> None:
        self._session_factory = session_factory
        self._embedding_service = embedding_service
        self._top_k_limit = top_k_limit
        self._threshold = threshold

    def search(self, query: str, *, top_k: int | None = None) -> list[KnowledgeSearchResult]:
        """Returns 0-3 results, already filtered to the similarity threshold, best first.

        Threshold filtering happens in Python, after the SQL LIMIT: the result set is
        at most 3 rows, so filtering here is simpler than repeating the distance
        expression in a SQL WHERE clause (which cannot reference a SELECT alias
        directly), with no meaningful cost at this scale.
        """
        requested = min(top_k or self._top_k_limit, self._top_k_limit)
        query_embedding = self._embedding_service.embed_query(query)  # GeminiRequestError propagates

        try:
            with self._session_factory() as session:
                rows = session.execute(
                    _SEARCH_SQL,
                    {"query_embedding": _to_pgvector_literal(query_embedding), "top_k": requested},
                ).mappings().all()
        except SQLAlchemyError as exc:
            raise AppError(
                ErrorCode.DATABASE_ERROR,
                "Knowledge search failed because the database is unavailable.",
                status_code=503,
            ) from exc
        
        logger.info(
            "Knowledge search raw similarities: %s",
            [
                {
                    "slug": row["slug"],
                    "similarity": float(row["similarity"]),
                }
                for row in rows
            ],
        )
        return [
            KnowledgeSearchResult(
                id=str(row["id"]), slug=row["slug"], name=row["name"], definition=row["definition"],
                category=row["category"], source=row["source"], similarity=float(row["similarity"]),
            )
            for row in rows
            if row["similarity"] >= self._threshold
        ]