from typing import Any, TypedDict


class RAGState(TypedDict, total=False):
    """Typed state passed between RAG graph nodes."""
    db: Any

    user_id: str
    question: str
    rewritten_question: str | None
    chat_history: list[dict[str, Any]]

    top_k: int
    hybrid_top_k: int
    vector_top_k: int
    bm25_top_k: int
    rerank_top_k: int
    vector_weight: float
    bm25_weight: float
    min_reranker_score: float | None
    retrieval_attempts: int
    max_retrieval_attempts: int
    generation_attempts: int
    max_generation_attempts: int
    generation_feedback: str | None

    user_context: dict[str, Any] | None

    evidence_chunks: list[dict[str, Any]]
    evidence_sufficient: bool | None
    evidence_sufficiency_reason: str | None

    generated_answer: str | None
    citations: list[dict[str, Any]]

    validation_status: str | None
    validation_reason: str | None
    grounding_status: str | None
    grounding_reason: str | None
    unsupported_claims: list[str]

    final_answer: str | None
    final_response: dict[str, Any] | None

    model_name: str | None
    status: str | None
    error: str | None
