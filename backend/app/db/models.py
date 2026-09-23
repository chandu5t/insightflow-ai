"""SQLAlchemy models for dataset metadata and analysis history (Module 6).

PostgreSQL stores METADATA only. The CSV files themselves stay on local disk,
exactly as in JsonDatasetRepository (see docs/DECISIONS.md D-057).
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from app.db.base import Base


class DatasetRow(Base):
    """One row per uploaded dataset. Mirrors DatasetMetadata (dataset_schema.py) field-for-field."""

    __tablename__ = "datasets"

    dataset_id: Mapped[uuid.UUID] = mapped_column(PostgresUUID(as_uuid=True), primary_key=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    source_format: Mapped[str] = mapped_column(String(10), nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    column_count: Mapped[int] = mapped_column(Integer, nullable=False)
    # JSONB, not a child table: column_names is always read/written as one whole list
    # (nothing filters datasets BY an individual column name), so a join buys nothing.
    column_names: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    original_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="uploaded")
    warnings: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    history: Mapped[list["AnalysisHistoryRow"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )


class AnalysisHistoryRow(Base):
    """One row per answered question. `response` is the FULL QueryResponse, JSON-encoded.

    Storing the whole response as JSONB (instead of one column per field) is deliberate:
    QueryResponse.result is a discriminated union (AggregationResult | GroupingResult |
    RankingResult | MissingValueReport) whose shape depends on which tool answered --
    a fixed relational schema would need a different table per tool. JSONB keeps this
    simple and still lets PostgreSQL query INSIDE the JSON later if ever needed
    (e.g. `response -> 'status'`), unlike a plain TEXT column.
    """

    __tablename__ = "analysis_history"

    id: Mapped[uuid.UUID] = mapped_column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("datasets.dataset_id", ondelete="CASCADE"),
        nullable=False,
        index=True,  # every history lookup filters "for this dataset"
    )
    # TEXT, not String(n): max_question_length (Settings) is already enforced before this
    # is ever written, so the limit is not hard-coded a second time here.
    question: Mapped[str] = mapped_column(Text, nullable=False)
    intent: Mapped[str | None] = mapped_column(String(50), nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    response: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC), index=True
    )  # supports "most recent questions first" -- the natural way to view history

    dataset: Mapped["DatasetRow"] = relationship(back_populates="history")

class KnowledgeDocumentRow(Base):
    """One row per business-metric definition (Module 7)."""

    __tablename__ = "knowledge_documents"

    id: Mapped[uuid.UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    slug: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
    )

    name: Mapped[str] = mapped_column(String(200), nullable=False)

    definition: Mapped[str] = mapped_column(Text, nullable=False)

    category: Mapped[str] = mapped_column(String(50), nullable=False)

    source: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
        default="InsightFlow AI built-in knowledge base",
    )

    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(768),
        nullable=True,
    )

    doc_metadata: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )