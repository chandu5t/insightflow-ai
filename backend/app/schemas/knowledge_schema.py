"""Pydantic models for POST /knowledge/search."""

from pydantic import BaseModel, ConfigDict, Field


class KnowledgeSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=300)
    top_k: int | None = Field(default=None, ge=1, le=3)  # hard cap of 3, regardless of settings


class KnowledgeSearchResultItem(BaseModel):
    id: str
    slug: str
    name: str
    definition: str
    category: str
    source: str
    similarity: float


class KnowledgeSearchResponse(BaseModel):
    query: str
    results: list[KnowledgeSearchResultItem]
    fallback: bool  # True when no result passed the similarity threshold