import re
from typing import Any

from rank_bm25 import BM25Okapi
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.document import Document
from app.models.document_chunk import DocumentChunk


def tokenize_text(text: str) -> list[str]:
    """
    Simple tokenizer for BM25 keyword retrieval.

    This keeps Day 8 intentionally simple:
    - lowercase
    - remove basic punctuation
    - split on whitespace
    """

    if not text:
        return []

    clean_text = text.lower()
    clean_text = re.sub(r"[^a-z0-9\s]", " ", clean_text)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()

    if not clean_text:
        return []

    return clean_text.split()


def bm25_search(
    db: Session,
    user_id: str,
    query: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Search user-owned document chunks using BM25 keyword ranking.

    This is a foundation/debug retriever only.
    It does not generate final LLM answers.
    It does not combine vector search.
    """

    if not user_id or not user_id.strip():
        raise ValueError("user_id is required")

    if not query or not query.strip():
        raise ValueError("query is required")

    if top_k <= 0:
        top_k = 5

    if top_k > 50:
        top_k = 50

    clean_user_id = user_id.strip()
    clean_query = query.strip()

    # PostgreSQL maintains and searches its lexical representation inside the
    # database. This avoids fetching and rebuilding the user's whole BM25
    # corpus for every production query. SQLite retains the Python fallback so
    # unit tests and lightweight local development remain portable.
    if db.bind is not None and db.bind.dialect.name == "postgresql":
        document = func.to_tsvector(
            "english",
            func.coalesce(DocumentChunk.search_text, DocumentChunk.chunk_text),
        )
        query_vector = func.websearch_to_tsquery("english", clean_query)
        rank = func.ts_rank_cd(document, query_vector).label("bm25_score")
        rows = (
            db.query(
                DocumentChunk,
                Document.original_file_name.label("document_name"),
                Document.category.label("category"),
                rank,
            )
            .join(Document, DocumentChunk.document_id == Document.id)
            .filter(DocumentChunk.user_id == clean_user_id)
            .filter(DocumentChunk.status.in_(["created", "embedded"]))
            .filter(Document.status == "embedded")
            .filter(document.op("@@")(query_vector))
            .order_by(rank.desc())
            .limit(top_k)
            .all()
        )
        return [
            {
                "chunk_id": str(row.DocumentChunk.id),
                "document_id": str(row.DocumentChunk.document_id),
                "user_id": row.DocumentChunk.user_id,
                "chunk_text": row.DocumentChunk.chunk_text,
                "chunk_index": row.DocumentChunk.chunk_index,
                "token_count": row.DocumentChunk.token_count,
                "page_number": row.DocumentChunk.page_number,
                "section_title": row.DocumentChunk.section_title,
                "summary": row.DocumentChunk.summary,
                "keywords": row.DocumentChunk.keywords or [],
                "hypothetical_questions": row.DocumentChunk.hypothetical_questions or [],
                "structure_types": row.DocumentChunk.structure_types or [],
                "document_name": row.document_name,
                "category": row.category,
                "bm25_score": float(row.bm25_score),
            }
            for row in rows
        ]

    rows = (
        db.query(
            DocumentChunk,
            Document.original_file_name.label("document_name"),
            Document.category.label("category"),
        )
        .join(Document, DocumentChunk.document_id == Document.id)
        .filter(DocumentChunk.user_id == clean_user_id)
        .filter(DocumentChunk.status.in_(["created", "embedded"]))
        .filter(DocumentChunk.chunk_text.isnot(None))
        .filter(Document.status == "embedded")
        .all()
    )

    if not rows:
        return []

    tokenized_corpus: list[list[str]] = []
    valid_rows = []

    for row in rows:
        chunk = row[0]

        searchable_text = getattr(chunk, "search_text", None) or chunk.chunk_text

        if not searchable_text or not searchable_text.strip():
            continue

        tokens = tokenize_text(searchable_text)

        if not tokens:
            continue

        tokenized_corpus.append(tokens)
        valid_rows.append(row)

    if not valid_rows:
        return []

    query_tokens = tokenize_text(clean_query)

    if not query_tokens:
        return []

    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query_tokens)

    scored_results = []

    for row, score in zip(valid_rows, scores):
        chunk = row[0]
        document_name = row[1]
        category = row[2]

        scored_results.append(
            {
                "chunk_id": str(chunk.id),
                "document_id": str(chunk.document_id),
                "user_id": chunk.user_id,
                "chunk_text": chunk.chunk_text,
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count,
                "page_number": chunk.page_number,
                "section_title": chunk.section_title,
                "summary": getattr(chunk, "summary", None),
                "keywords": getattr(chunk, "keywords", None) or [],
                "hypothetical_questions": getattr(chunk, "hypothetical_questions", None) or [],
                "structure_types": getattr(chunk, "structure_types", None) or [],
                "document_name": document_name,
                "category": category,
                "bm25_score": float(score),
            }
        )

    scored_results.sort(
        key=lambda item: item["bm25_score"],
        reverse=True,
    )

    return scored_results[:top_k]
