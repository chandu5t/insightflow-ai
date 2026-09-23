"""Tests for the ranking/threshold logic, isolated from the database via a fake session."""

import pytest

from app.core.errors import AppError
from app.services.knowledge_search_service import KnowledgeSearchService, _to_pgvector_literal
from tests.fake_embeddings import FakeEmbeddings


def test_to_pgvector_literal_format() -> None:
    assert _to_pgvector_literal([0.1, -0.2, 1.0]) == "[0.1,-0.2,1.0]"


class _FakeConnection:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def execute(self, statement, params):
        class _Result:
            def __init__(self, rows):
                self._rows = rows

            def mappings(self):
                return self

            def all(self):
                return self._rows

        return _Result(self._rows)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def make_service(rows, threshold=0.6, top_k_limit=3):
    embeddings = FakeEmbeddings(vectors={"q": [1.0, 0.0, 0.0, 0.0]}, dimensions=4)
    return KnowledgeSearchService(
        lambda: _FakeConnection(rows), embeddings, top_k_limit=top_k_limit, threshold=threshold
    )


def test_results_below_threshold_are_filtered_out() -> None:
    rows = [
        {"id": "1", "slug": "aov", "name": "AOV", "definition": "d", "category": "c", "source": "s", "similarity": 0.9},
        {"id": "2", "slug": "cac", "name": "CAC", "definition": "d", "category": "c", "source": "s", "similarity": 0.3},
    ]
    results = make_service(rows, threshold=0.6).search("q")

    assert [r.slug for r in results] == ["aov"]


def test_empty_database_returns_no_results() -> None:
    assert make_service([]).search("q") == []


def test_top_k_is_capped_at_the_configured_limit() -> None:
    service = make_service([], top_k_limit=3)
    # searching with top_k=10 must still be capped -- verified via the bound parameter
    calls = {}
    original_connection = service._session_factory

    class Spy:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, statement, params):
            calls.update(params)

            class R:
                def mappings(self):
                    return self

                def all(self):
                    return []

            return R()

    service._session_factory = lambda: Spy()
    service.search("q", top_k=10)

    assert calls["top_k"] == 3


def test_database_failure_becomes_an_app_error() -> None:
    from sqlalchemy.exc import SQLAlchemyError

    class BrokenConnection:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, *a, **k):
            raise SQLAlchemyError("connection refused")

    embeddings = FakeEmbeddings(vectors={"q": [1.0, 0.0, 0.0, 0.0]}, dimensions=4)
    service = KnowledgeSearchService(lambda: BrokenConnection(), embeddings, top_k_limit=3, threshold=0.6)

    with pytest.raises(AppError) as info:
        service.search("q")
    assert info.value.status_code == 503