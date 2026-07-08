from copy import deepcopy
from typing import Any
from threading import BoundedSemaphore

from sentence_transformers import CrossEncoder
from app.config.settings import settings


_inference_slots = BoundedSemaphore(max(1, settings.INFERENCE_MAX_CONCURRENCY))


class CrossEncoderReranker:
    """
    Local cross-encoder reranker.

    It scores query/chunk_text pairs and returns candidates sorted by
    reranker_score descending.

    This is used after hybrid retrieval to improve the final chunk ordering.
    """

    def __init__(self, model_name: str):
        """Load the cross-encoder model used for reranking."""
        if not model_name or not model_name.strip():
            raise ValueError("model_name is required")

        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 8,
    ) -> list[dict[str, Any]]:
        """Score and sort candidate chunks by relevance to a query."""
        if not query or not query.strip():
            raise ValueError("query is required")

        if not candidates:
            return []

        safe_top_k = max(1, top_k)

        pairs = [
            (
                query,
                str(candidate.get("chunk_text") or ""),
            )
            for candidate in candidates
        ]

        with _inference_slots:
            scores = self.model.predict(pairs)

        reranked_results: list[dict[str, Any]] = []

        for candidate, score in zip(candidates, scores):
            copied_candidate = deepcopy(candidate)

            copied_candidate["reranker_score"] = float(score)
            copied_candidate["reranker_model_name"] = self.model_name

            reranked_results.append(copied_candidate)

        reranked_results.sort(
            key=lambda item: item["reranker_score"],
            reverse=True,
        )

        return reranked_results[:safe_top_k]
