from datetime import datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import DateTime, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class EvaluationDataset(Base):
    """A user-owned collection of approved RAG evaluation cases."""

    __tablename__ = "evaluation_datasets"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    user_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    original_file_name: Mapped[str] = mapped_column(String, nullable=False)
    cases: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    document_ids: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    case_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        index=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

