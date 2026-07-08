import json
import re
from typing import Any

from app.generation.citation_guard import (
    build_citations_from_evidence,
    check_evidence_sufficiency,
    validate_answer_support,
)
from app.generation.llm_client import LLMRateLimitError, get_aux_llm_client, get_llm_client
from app.generation.prompt_builder import (
    build_answer_regeneration_prompt,
    build_answer_verification_prompt,
    build_answer_prompt,
    build_query_rewrite_prompt,
    build_retrieval_retry_prompt,
    get_refusal_message,
    strip_generated_source_metadata,
)
from app.graph.state import RAGState
from app.reranking.reranking_service import rerank_hybrid_results
from app.utils.logger import get_logger


logger = get_logger(__name__)


def _extract_llm_text(response: Any) -> str:
    """Extract plain text from supported LLM response shapes."""
    if response is None:
        return ""

    if isinstance(response, str):
        return response

    if hasattr(response, "content"):
        return str(response.content)

    if isinstance(response, dict):
        if "content" in response:
            return str(response["content"])

        if "text" in response:
            return str(response["text"])

        if "answer" in response:
            return str(response["answer"])

    return str(response)


def _parse_verification_response(response_text: str) -> dict[str, Any]:
    """Parse the grounding verifier JSON response."""
    if not response_text or not response_text.strip():
        return {
            "status": "unsupported",
            "reason": "Answer verifier returned an empty response.",
            "unsupported_claims": [],
        }

    cleaned_text = response_text.strip()
    fenced_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        cleaned_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced_match:
        cleaned_text = fenced_match.group(1).strip()

    try:
        parsed = json.loads(cleaned_text)
    except json.JSONDecodeError:
        return {
            "status": "unsupported",
            "reason": "Answer verifier returned invalid JSON.",
            "unsupported_claims": [],
        }

    status = str(parsed.get("status") or "").strip().lower()
    if status != "supported":
        status = "unsupported"

    unsupported_claims = parsed.get("unsupported_claims") or []
    if not isinstance(unsupported_claims, list):
        unsupported_claims = [str(unsupported_claims)]

    return {
        "status": status,
        "reason": str(parsed.get("reason") or "Answer grounding verification completed."),
        "unsupported_claims": [
            str(claim)
            for claim in unsupported_claims
            if str(claim).strip()
        ],
    }


def load_user_context_node(state: RAGState) -> RAGState:
    """Attach basic user context to the RAG graph state."""
    logger.debug("RAG node load_user_context: user_id=%s", state.get("user_id"))
    state["user_context"] = {
        "user_id": state.get("user_id"),
    }

    return state


def rewrite_query_node(state: RAGState) -> RAGState:
    """Rewrite the user question before retrieval when possible."""
    original_question = state.get("question") or ""
    chat_history = state.get("chat_history") or []

    try:
        logger.debug(
            "RAG node rewrite_query started: user_id=%s question_length=%s history_turns=%s",
            state.get("user_id"),
            len(original_question),
            len(chat_history),
        )
        prompt = build_query_rewrite_prompt(
            question=original_question,
            chat_history=chat_history,
        )
        llm_client = get_llm_client()
        response = llm_client.generate(prompt)
        rewritten_question = _extract_llm_text(response).strip()

        if not rewritten_question:
            rewritten_question = original_question

        state["rewritten_question"] = rewritten_question
        logger.debug(
            "RAG node rewrite_query completed: user_id=%s rewritten_length=%s",
            state.get("user_id"),
            len(rewritten_question),
        )

    except Exception as exc:
        state["rewritten_question"] = original_question
        state["error"] = f"Query rewrite failed and fell back to original question: {exc}"
        logger.warning(
            "RAG node rewrite_query failed; using original question: user_id=%s error=%s",
            state.get("user_id"),
            exc,
        )

    return state


async def rewrite_query_node_async(state: RAGState) -> RAGState:
    """Async provider-I/O variant used by the API workflow."""
    original_question = state.get("question") or ""
    chat_history = state.get("chat_history") or []
    if not chat_history:
        state["rewritten_question"] = original_question
        return state
    try:
        prompt = build_query_rewrite_prompt(
            question=original_question,
            chat_history=chat_history,
        )
        rewritten = _extract_llm_text(await get_aux_llm_client().agenerate(prompt)).strip()
        state["rewritten_question"] = rewritten or original_question
    except Exception as exc:
        state["rewritten_question"] = original_question
        state["error"] = f"Query rewrite failed and fell back to original question: {exc}"
    return state


def refine_retrieval_query_node(state: RAGState) -> RAGState:
    """Create an alternative query after an insufficient retrieval attempt."""
    question = state.get("question") or ""
    previous_query = state.get("rewritten_question") or question
    next_attempt = state.get("retrieval_attempts", 0) + 1

    try:
        prompt = build_retrieval_retry_prompt(
            question=question,
            previous_query=previous_query,
            attempt=next_attempt,
        )
        response = get_llm_client().generate(prompt)
        refined_query = _extract_llm_text(response).strip()
        state["rewritten_question"] = refined_query or question
        logger.info(
            "RAG retrieval retry query prepared: user_id=%s next_attempt=%s query_length=%s",
            state.get("user_id"),
            next_attempt,
            len(state["rewritten_question"] or ""),
        )
    except Exception as exc:
        state["rewritten_question"] = question
        state["error"] = f"Retrieval retry query refinement failed: {exc}"
        logger.warning(
            "RAG retrieval retry query refinement failed; using original question: user_id=%s error=%s",
            state.get("user_id"),
            exc,
        )

    return state


async def refine_retrieval_query_node_async(state: RAGState) -> RAGState:
    question = state.get("question") or ""
    previous_query = state.get("rewritten_question") or question
    try:
        prompt = build_retrieval_retry_prompt(
            question=question,
            previous_query=previous_query,
            attempt=state.get("retrieval_attempts", 0) + 1,
        )
        refined = _extract_llm_text(await get_aux_llm_client().agenerate(prompt)).strip()
        state["rewritten_question"] = refined or question
    except Exception as exc:
        state["rewritten_question"] = question
        state["error"] = f"Retrieval retry query refinement failed: {exc}"
    return state


def retrieve_and_rerank_node(state: RAGState) -> RAGState:
    """Retrieve, combine, and rerank candidate evidence chunks."""
    db = state["db"]
    user_id = state["user_id"]

    retrieval_query = state.get("rewritten_question") or state["question"]
    logger.debug(
        "RAG node retrieve_and_rerank started: user_id=%s query_length=%s",
        user_id,
        len(retrieval_query or ""),
    )

    top_k = state.get("top_k", 5)
    hybrid_top_k = state.get("hybrid_top_k", 20)
    vector_top_k = state.get("vector_top_k", 20)
    bm25_top_k = state.get("bm25_top_k", 20)
    rerank_top_k = state.get("rerank_top_k", 8)
    vector_weight = state.get("vector_weight", 0.6)
    bm25_weight = state.get("bm25_weight", 0.4)

    state["retrieval_attempts"] = state.get("retrieval_attempts", 0) + 1
    evidence_chunks = rerank_hybrid_results(
        db=db,
        user_id=user_id,
        query=retrieval_query,
        top_k=rerank_top_k,
        hybrid_top_k=hybrid_top_k,
        vector_top_k=vector_top_k,
        bm25_top_k=bm25_top_k,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
    )

    state["evidence_chunks"] = evidence_chunks[:top_k]
    logger.info(
        "RAG node retrieve_and_rerank completed: user_id=%s attempt=%s retrieved=%s selected=%s",
        user_id,
        state["retrieval_attempts"],
        len(evidence_chunks),
        len(state["evidence_chunks"]),
    )

    return state


def check_evidence_sufficiency_node(state: RAGState) -> RAGState:
    """Assess whether retrieved evidence can support an answer."""
    evidence_chunks = state.get("evidence_chunks", [])
    min_reranker_score = state.get("min_reranker_score")

    result = check_evidence_sufficiency(
        evidence_chunks=evidence_chunks,
        min_evidence_chunks=1,
        min_reranker_score=min_reranker_score,
    )

    evidence_sufficient = bool(result["evidence_sufficient"])
    reason = str(result["reason"])
    logger.info(
        "RAG node evidence sufficiency checked: user_id=%s sufficient=%s evidence_chunks=%s reason=%s",
        state.get("user_id"),
        evidence_sufficient,
        len(evidence_chunks),
        reason,
    )

    state["evidence_sufficient"] = evidence_sufficient
    state["evidence_sufficiency_reason"] = reason

    return state


def generate_answer_node(state: RAGState) -> RAGState:
    """Generate or refuse an answer from the current graph state."""
    if state.get("status") == "refused" or state.get("evidence_sufficient") is False:
        refusal_message = state.get("generated_answer") or get_refusal_message()
        logger.info("RAG node generate_answer skipped with refusal: user_id=%s", state.get("user_id"))

        state["generated_answer"] = refusal_message
        state["final_answer"] = refusal_message
        state["citations"] = state.get("citations", [])
        state["status"] = "refused"

        return state

    question = state["question"]
    evidence_chunks = state.get("evidence_chunks", [])
    previous_answer = state.get("generated_answer") or ""
    generation_attempt = state.get("generation_attempts", 0) + 1
    state["generation_attempts"] = generation_attempt

    citations = build_citations_from_evidence(evidence_chunks)
    logger.debug(
        "RAG node generate_answer started: user_id=%s evidence_chunks=%s citations=%s",
        state.get("user_id"),
        len(evidence_chunks),
        len(citations),
    )
    if generation_attempt > 1:
        prompt = build_answer_regeneration_prompt(
            question=question,
            evidence_chunks=evidence_chunks,
            previous_answer=previous_answer,
            feedback=state.get("generation_feedback") or "The previous answer was not fully supported.",
        )
    else:
        prompt = build_answer_prompt(
            question=question,
            evidence_chunks=evidence_chunks,
        )

    llm_client = get_llm_client()
    generated_answer = strip_generated_source_metadata(
        _extract_llm_text(llm_client.generate(prompt))
    )

    state["generated_answer"] = generated_answer
    state["final_answer"] = generated_answer
    state["citations"] = citations
    state["status"] = "answered"
    state["grounding_status"] = None
    state["grounding_reason"] = None
    state["unsupported_claims"] = []
    state["validation_status"] = None
    state["validation_reason"] = None
    logger.info(
        "RAG node generate_answer completed: user_id=%s attempt=%s answer_length=%s citations=%s",
        state.get("user_id"),
        generation_attempt,
        len(generated_answer),
        len(citations),
    )

    return state


async def generate_answer_node_async(state: RAGState) -> RAGState:
    if state.get("status") == "refused" or state.get("evidence_sufficient") is False:
        return generate_answer_node(state)
    question = state["question"]
    evidence_chunks = state.get("evidence_chunks", [])
    attempt = state.get("generation_attempts", 0) + 1
    state["generation_attempts"] = attempt
    citations = build_citations_from_evidence(evidence_chunks)
    if attempt > 1:
        prompt = build_answer_regeneration_prompt(
            question=question,
            evidence_chunks=evidence_chunks,
            previous_answer=state.get("generated_answer") or "",
            feedback=state.get("generation_feedback") or "The previous answer was not fully supported.",
        )
    else:
        prompt = build_answer_prompt(question=question, evidence_chunks=evidence_chunks)
    answer = strip_generated_source_metadata(
        _extract_llm_text(await get_llm_client().agenerate(prompt))
    )
    state.update({
        "generated_answer": answer,
        "final_answer": answer,
        "citations": citations,
        "status": "answered",
        "grounding_status": None,
        "grounding_reason": None,
        "unsupported_claims": [],
        "validation_status": None,
        "validation_reason": None,
    })
    return state


def verify_answer_grounding_node(state: RAGState) -> RAGState:
    """Verify that the generated answer is supported by retrieved evidence."""
    if state.get("status") == "refused":
        state["grounding_status"] = state.get("grounding_status") or "unsupported"
        state["grounding_reason"] = state.get("grounding_reason") or "Answer was already refused."
        state["unsupported_claims"] = state.get("unsupported_claims", [])

        return state

    generated_answer = state.get("generated_answer") or ""
    evidence_chunks = state.get("evidence_chunks", [])

    try:
        prompt = build_answer_verification_prompt(
            question=state["question"],
            answer=generated_answer,
            evidence_chunks=evidence_chunks,
        )

        llm_client = get_llm_client()
        verifier_response = _extract_llm_text(llm_client.generate(prompt))
        verification_result = _parse_verification_response(verifier_response)
    except LLMRateLimitError:
        raise
    except Exception as exc:
        verification_result = {
            "status": "unsupported",
            "reason": f"Answer grounding verification failed: {exc}",
            "unsupported_claims": [],
        }

    state["grounding_status"] = verification_result["status"]
    state["grounding_reason"] = verification_result["reason"]
    state["unsupported_claims"] = verification_result["unsupported_claims"]
    logger.info(
        "RAG node verify_answer_grounding completed: user_id=%s grounding=%s reason=%s",
        state.get("user_id"),
        state["grounding_status"],
        state["grounding_reason"],
    )

    return state


async def verify_answer_grounding_node_async(state: RAGState) -> RAGState:
    if state.get("status") == "refused":
        return verify_answer_grounding_node(state)
    try:
        prompt = build_answer_verification_prompt(
            question=state["question"],
            answer=state.get("generated_answer") or "",
            evidence_chunks=state.get("evidence_chunks", []),
        )
        result = _parse_verification_response(
            _extract_llm_text(await get_aux_llm_client().agenerate(prompt))
        )
    except LLMRateLimitError:
        raise
    except Exception as exc:
        result = {
            "status": "unsupported",
            "reason": f"Answer grounding verification failed: {exc}",
            "unsupported_claims": [],
        }
    state["grounding_status"] = result["status"]
    state["grounding_reason"] = result["reason"]
    state["unsupported_claims"] = result["unsupported_claims"]
    return state


def validate_citations_node(state: RAGState) -> RAGState:
    """Validate generated citations against retrieved evidence."""
    if state.get("status") == "refused":
        state["validation_status"] = state.get("validation_status") or "unsupported"
        state["validation_reason"] = state.get("validation_reason") or "Evidence was insufficient."

        return state

    validation_result = validate_answer_support(
        answer=state.get("generated_answer"),
        citations=state.get("citations", []),
        evidence_chunks=state.get("evidence_chunks", []),
    )

    state["validation_status"] = validation_result["validation_status"]
    state["validation_reason"] = validation_result["validation_reason"]
    logger.info(
        "RAG node validate_citations completed: user_id=%s validation=%s reason=%s",
        state.get("user_id"),
        state["validation_status"],
        state["validation_reason"],
    )

    return state


def prepare_generation_retry_node(state: RAGState) -> RAGState:
    """Preserve validation feedback before regenerating an answer."""
    unsupported_claims = state.get("unsupported_claims") or []
    feedback_parts = [
        state.get("grounding_reason"),
        state.get("validation_reason"),
    ]
    if unsupported_claims:
        feedback_parts.append("Unsupported claims: " + "; ".join(unsupported_claims))

    state["generation_feedback"] = " ".join(
        str(part).strip()
        for part in feedback_parts
        if part and str(part).strip()
    ) or "The previous answer was not fully supported by the evidence."
    logger.info(
        "RAG generation retry prepared: user_id=%s next_attempt=%s",
        state.get("user_id"),
        state.get("generation_attempts", 0) + 1,
    )
    return state


def refuse_answer_node(state: RAGState) -> RAGState:
    """Finalize a refusal after retry attempts are exhausted."""
    refusal_message = get_refusal_message()
    reason = (
        state.get("validation_reason")
        or state.get("grounding_reason")
        or state.get("evidence_sufficiency_reason")
        or "Retry attempts were exhausted without a supported answer."
    )
    state["generated_answer"] = refusal_message
    state["final_answer"] = refusal_message
    state["citations"] = []
    state["validation_status"] = "unsupported"
    state["validation_reason"] = reason
    state["status"] = "refused"
    logger.warning(
        "RAG workflow refused after retries: user_id=%s retrieval_attempts=%s generation_attempts=%s reason=%s",
        state.get("user_id"),
        state.get("retrieval_attempts", 0),
        state.get("generation_attempts", 0),
        reason,
    )
    return state


def route_after_evidence_check(state: RAGState) -> str:
    """Route sufficient evidence to generation or retry/refuse weak retrieval."""
    if state.get("evidence_sufficient"):
        return "generate"
    if state.get("retrieval_attempts", 0) < state.get("max_retrieval_attempts", 1):
        return "retry"
    return "refuse"


def route_after_grounding_check(state: RAGState) -> str:
    """Route grounded answers to citation validation or retry/refuse."""
    if state.get("grounding_status") == "supported":
        return "validate"
    if state.get("generation_attempts", 0) < state.get("max_generation_attempts", 1):
        return "retry"
    return "refuse"


def route_after_citation_check(state: RAGState) -> str:
    """Return supported answers or retry/refuse invalid citations."""
    if state.get("validation_status") == "supported":
        return "final"
    if state.get("generation_attempts", 0) < state.get("max_generation_attempts", 1):
        return "retry"
    return "refuse"


def final_response_node(state: RAGState) -> RAGState:
    """Build the final API response from graph state."""
    final_answer = state.get("final_answer") or state.get("generated_answer") or get_refusal_message()
    evidence_chunks = state.get("evidence_chunks", [])

    final_response = {
        "user_id": state.get("user_id"),
        "question": state.get("question"),
        "rewritten_question": state.get("rewritten_question"),
        "answer": final_answer,
        "citations": state.get("citations", []),

        "evidence_chunks": evidence_chunks,
        "evidence_chunk_count": len(evidence_chunks),

        "validation_status": state.get("validation_status"),
        "validation_reason": state.get("validation_reason"),
        "grounding_status": state.get("grounding_status"),
        "grounding_reason": state.get("grounding_reason"),
        "unsupported_claims": state.get("unsupported_claims", []),
        "evidence_sufficient": state.get("evidence_sufficient"),
        "evidence_sufficiency_reason": state.get("evidence_sufficiency_reason"),
        "model_name": state.get("model_name"),
        "status": state.get("status"),
        "retrieval_attempts": state.get("retrieval_attempts", 0),
        "generation_attempts": state.get("generation_attempts", 0),
    }

    state["final_answer"] = final_answer
    state["final_response"] = final_response
    logger.debug(
        "RAG node final_response completed: user_id=%s status=%s",
        state.get("user_id"),
        final_response["status"],
    )

    return state
