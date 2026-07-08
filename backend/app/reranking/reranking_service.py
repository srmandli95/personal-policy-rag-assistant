from typing import Any
from functools import lru_cache

from app.config.settings import settings
from app.reranking.cross_encoder_reranker import CrossEncoderReranker
from app.retrieval.hybrid_retriever import hybrid_search
from app.retrieval.retrieval_settings import (
    RetrievalSettings,
    validate_retrieval_settings,
)


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderReranker:
    """Return the process-wide reranker instead of loading it per query."""
    return CrossEncoderReranker(model_name=settings.RERANKER_MODEL_NAME)


def rerank_hybrid_results(
    db,
    user_id: str,
    query: str,
    top_k: int = 8,
    hybrid_top_k: int = 20,
    vector_top_k: int = 20,
    bm25_top_k: int = 20,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> list[dict[str, Any]]:
    """
    Run hybrid retrieval first, then rerank candidate chunks using a local
    cross-encoder model.

    This is still a foundation/debugging service, not final answer generation.
    """

    if not user_id or not user_id.strip():
        raise ValueError("user_id is required")

    if not query or not query.strip():
        raise ValueError("query is required")

    retrieval_settings = validate_retrieval_settings(
        RetrievalSettings(
            vector_top_k=vector_top_k,
            bm25_top_k=bm25_top_k,
            hybrid_top_k=hybrid_top_k,
            rerank_top_k=top_k,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
        )
    )

    candidates = hybrid_search(
        db=db,
        user_id=user_id,
        query=query,
        top_k=retrieval_settings.hybrid_top_k,
        vector_top_k=retrieval_settings.vector_top_k,
        bm25_top_k=retrieval_settings.bm25_top_k,
        vector_weight=retrieval_settings.vector_weight,
        bm25_weight=retrieval_settings.bm25_weight,
    )

    if not candidates:
        return []

    reranker = get_reranker()

    return reranker.rerank(
        query=query,
        candidates=candidates,
        top_k=retrieval_settings.rerank_top_k,
    )
