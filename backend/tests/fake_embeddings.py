"""Deterministic, network-free embeddings for tests."""

import hashlib
import math
import random

from langchain_core.embeddings import Embeddings


class FakeEmbeddings(Embeddings):
    """Deterministic embeddings for isolated and API tests."""

    def __init__(
        self,
        vectors: dict[str, list[float]] | None = None,
        dimensions: int = 768,
    ) -> None:
        self.dimensions = dimensions
        self._vectors = vectors or {}

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        normalized_text = text.lower().strip()

        if text in self._vectors:
            return self._normalize(self._vectors[text])

        # Match only the AOV definition itself, not CLV's reference to AOV.
        if normalized_text.startswith(
            "average order value is the average amount spent"
        ):
            return self._normalize(
                [1.0] + [0.0] * (self.dimensions - 1)
            )

        seed = int(
            hashlib.sha256(text.encode()).hexdigest(),
            16,
        ) % (2**32)

        rng = random.Random(seed)

        return self._normalize(
            [
                rng.uniform(-1, 1)
                for _ in range(self.dimensions)
            ]
        )

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vector))

        return (
            [value / norm for value in vector]
            if norm
            else vector
        )