"""
Gemini embedding service, implementing the LangChain Embeddings interface.

Reuses the same google-genai SDK style as gemini_client.py.
"""

import logging
import math

from langchain_core.embeddings import Embeddings

from app.services.gemini_client import (
    GeminiNotConfiguredError,
    GeminiRequestError,
)

logger = logging.getLogger(__name__)


def _normalize(vector: list[float]) -> list[float]:
    """L2-normalize the vector for cosine similarity."""
    norm = math.sqrt(sum(component * component for component in vector))
    return [component / norm for component in vector] if norm else vector


class GeminiEmbeddingService(Embeddings):
    """Gemini embedding service for documents and queries."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        dimensions: int,
        timeout_seconds: float,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._dimensions = dimensions
        self._timeout_seconds = timeout_seconds

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate normalized embeddings for multiple documents."""
        if not texts:
            return []

        try:
            values = self._call_sdk(
                texts,
                task_type="RETRIEVAL_DOCUMENT",
            )
        except GeminiNotConfiguredError:
            raise
        except GeminiRequestError:
            raise
        except Exception as exc:
            logger.warning(
                "Gemini document embedding failed (%s)",
                type(exc).__name__,
            )
            raise GeminiRequestError(
                "Gemini embedding request failed."
            ) from exc

        return [_normalize(vector) for vector in values]

    def embed_query(self, text: str) -> list[float]:
        """Generate a normalized embedding for a search query."""
        try:
            [values] = self._call_sdk(
                [text],
                task_type="RETRIEVAL_QUERY",
            )
        except GeminiNotConfiguredError:
            raise
        except GeminiRequestError:
            raise
        except Exception as exc:
            logger.warning(
                "Gemini query embedding failed (%s)",
                type(exc).__name__,
            )
            raise GeminiRequestError(
                "Gemini embedding request failed."
            ) from exc

        return _normalize(values)

    def _call_sdk(
        self,
        texts: list[str],
        *,
        task_type: str,
    ) -> list[list[float]]:
        """Call the Google Gemini embedding SDK."""
        if not self._api_key:
            raise GeminiNotConfiguredError(
                "GEMINI_API_KEY is not set"
            )

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(
                api_key=self._api_key,
                http_options=types.HttpOptions(
                    timeout=int(self._timeout_seconds * 1000)
                ),
            )

            response = client.models.embed_content(
                model=self._model,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self._dimensions,
                ),
            )

        except ImportError as exc:
            raise GeminiNotConfiguredError(
                "the google-genai package is not installed"
            ) from exc

        except TimeoutError as exc:
            raise GeminiRequestError("timeout") from exc

        except Exception as exc:
            logger.warning(
                "Gemini embedding request failed (%s)",
                type(exc).__name__,
            )
            raise GeminiRequestError(
                "Gemini embedding request failed."
            ) from exc

        values = [
            list(embedding.values)
            for embedding in response.embeddings
        ]

        if len(values) != len(texts) or any(not vector for vector in values):
            raise GeminiRequestError(
                "empty or incomplete embedding response"
            )

        return values