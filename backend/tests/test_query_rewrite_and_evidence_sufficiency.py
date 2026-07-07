import pytest

from app.generation.citation_guard import check_evidence_sufficiency
from app.graph.nodes import (
    check_evidence_sufficiency_node,
    generate_answer_node,
    retrieve_and_rerank_node,
    rewrite_query_node,
    verify_answer_grounding_node,
)
from app.graph.rag_graph import run_rag_workflow


class FakeLLM:
    def __init__(self, response: str | None = None, should_raise: bool = False):
        self.response = response
        self.should_raise = should_raise
        self.called = False
        self.prompt = None

    def generate(self, prompt: str):
        self.called = True
        self.prompt = prompt

        if self.should_raise:
            raise RuntimeError("LLM failed")

        return self.response


def test_rewrite_query_node_stores_rewritten_question(monkeypatch):
    fake_llm = FakeLLM("late payment consequences DTE Energy bill")

    monkeypatch.setattr(
        "app.graph.nodes.get_llm_client",
        lambda: fake_llm,
    )

    state = {
        "question": "can you explain me the consequences for the late payment of my bill for DTE Energy?"
    }

    result = rewrite_query_node(state)

    assert result["rewritten_question"] == "late payment consequences DTE Energy bill"
    assert fake_llm.called is True


def test_rewrite_query_node_uses_chat_history_for_follow_up(monkeypatch):
    fake_llm = FakeLLM("vision deductible for my health insurance plan")

    monkeypatch.setattr(
        "app.graph.nodes.get_llm_client",
        lambda: fake_llm,
    )

    state = {
        "question": "What about vision?",
        "chat_history": [
            {
                "question": "What is my dental deductible?",
                "answer": "The dental deductible is $50 according to the evidence.",
            }
        ],
    }

    result = rewrite_query_node(state)

    assert result["rewritten_question"] == "vision deductible for my health insurance plan"
    assert "Prior chat context:" in fake_llm.prompt
    assert "What is my dental deductible?" in fake_llm.prompt
    assert "Current question:" in fake_llm.prompt
    assert "What about vision?" in fake_llm.prompt


def test_rewrite_query_node_falls_back_when_llm_returns_empty(monkeypatch):
    fake_llm = FakeLLM("   ")

    monkeypatch.setattr(
        "app.graph.nodes.get_llm_client",
        lambda: fake_llm,
    )

    state = {
        "question": "Does my health insurance cover urgent care?"
    }

    result = rewrite_query_node(state)

    assert result["rewritten_question"] == "Does my health insurance cover urgent care?"


def test_rewrite_query_node_falls_back_when_llm_raises(monkeypatch):
    fake_llm = FakeLLM(should_raise=True)

    monkeypatch.setattr(
        "app.graph.nodes.get_llm_client",
        lambda: fake_llm,
    )

    state = {
        "question": "What happens if I miss a payment?"
    }

    result = rewrite_query_node(state)

    assert result["rewritten_question"] == "What happens if I miss a payment?"
    assert "Query rewrite failed" in result["error"]


def test_retrieve_and_rerank_node_uses_rewritten_question(monkeypatch):
    captured = {}

    def fake_rerank_hybrid_results(
        db,
        user_id,
        query,
        top_k,
        hybrid_top_k,
        vector_top_k,
        bm25_top_k,
        vector_weight,
        bm25_weight,
    ):
        captured["query"] = query

        return [
            {
                "chunk_id": "chunk-1",
                "chunk_text": "Late payments may result in a late fee.",
                "reranker_score": 0.9,
            }
        ]

    monkeypatch.setattr(
        "app.graph.nodes.rerank_hybrid_results",
        fake_rerank_hybrid_results,
    )

    state = {
        "db": object(),
        "user_id": "local-user-123",
        "question": "original question",
        "rewritten_question": "rewritten search query",
        "top_k": 5,
        "hybrid_top_k": 20,
        "vector_top_k": 20,
        "bm25_top_k": 20,
    }

    result = retrieve_and_rerank_node(state)

    assert captured["query"] == "rewritten search query"
    assert len(result["evidence_chunks"]) == 1


def test_retrieve_and_rerank_node_reranks_then_slices_final_evidence(monkeypatch):
    captured = {}

    def fake_rerank_hybrid_results(**kwargs):
        captured.update(kwargs)
        return [{"chunk_id": f"chunk-{index}"} for index in range(8)]

    monkeypatch.setattr(
        "app.graph.nodes.rerank_hybrid_results",
        fake_rerank_hybrid_results,
    )

    state = {
        "db": object(),
        "user_id": "local-user-123",
        "question": "question",
        "top_k": 3,
        "rerank_top_k": 8,
        "vector_weight": 0.7,
        "bm25_weight": 0.3,
    }

    result = retrieve_and_rerank_node(state)

    assert captured["top_k"] == 8
    assert captured["vector_weight"] == 0.7
    assert captured["bm25_weight"] == 0.3
    assert len(result["evidence_chunks"]) == 3


def test_evidence_sufficiency_fails_when_no_chunks():
    result = check_evidence_sufficiency([])

    assert result["evidence_sufficient"] is False
    assert result["reason"] == "No evidence chunks were retrieved."


def test_evidence_sufficiency_fails_when_chunk_count_below_minimum():
    result = check_evidence_sufficiency(
        evidence_chunks=[
            {
                "chunk_id": "chunk-1",
                "chunk_text": "Some evidence",
            }
        ],
        min_evidence_chunks=2,
    )

    assert result["evidence_sufficient"] is False
    assert "at least 2" in result["reason"]


def test_evidence_sufficiency_passes_when_enough_chunks_exist():
    result = check_evidence_sufficiency(
        evidence_chunks=[
            {
                "chunk_id": "chunk-1",
                "chunk_text": "Some evidence",
            }
        ],
        min_evidence_chunks=1,
    )

    assert result["evidence_sufficient"] is True


def test_evidence_sufficiency_fails_when_min_reranker_score_not_met():
    result = check_evidence_sufficiency(
        evidence_chunks=[
            {
                "chunk_id": "chunk-1",
                "chunk_text": "Weak evidence",
                "reranker_score": 0.2,
            }
        ],
        min_evidence_chunks=1,
        min_reranker_score=0.7,
    )

    assert result["evidence_sufficient"] is False
    assert "minimum reranker score" in result["reason"]


def test_evidence_sufficiency_passes_when_at_least_one_chunk_meets_score():
    result = check_evidence_sufficiency(
        evidence_chunks=[
            {
                "chunk_id": "chunk-1",
                "chunk_text": "Weak evidence",
                "reranker_score": 0.2,
            },
            {
                "chunk_id": "chunk-2",
                "chunk_text": "Strong evidence",
                "reranker_score": 0.8,
            },
        ],
        min_evidence_chunks=1,
        min_reranker_score=0.7,
    )

    assert result["evidence_sufficient"] is True


def test_check_evidence_sufficiency_node_records_weak_evidence_for_routing():
    state = {
        "evidence_chunks": [],
        "min_reranker_score": None,
    }

    result = check_evidence_sufficiency_node(state)

    assert result["evidence_sufficient"] is False
    assert result["evidence_sufficiency_reason"] == "No evidence chunks were retrieved."
    assert "status" not in result


def test_generate_answer_node_skips_llm_when_status_refused(monkeypatch):
    fake_llm = FakeLLM("This should not be called")

    monkeypatch.setattr(
        "app.graph.nodes.get_llm_client",
        lambda: fake_llm,
    )

    state = {
        "question": "Does this exist?",
        "status": "refused",
        "evidence_sufficient": False,
        "generated_answer": "Refusal answer",
        "final_answer": "Refusal answer",
        "citations": [],
    }

    result = generate_answer_node(state)

    assert result["generated_answer"] == "Refusal answer"
    assert result["status"] == "refused"
    assert fake_llm.called is False


def test_verify_answer_grounding_node_keeps_supported_answer(monkeypatch):
    fake_llm = FakeLLM(
        '{"status": "supported", "reason": "The answer matches the evidence.", "unsupported_claims": []}'
    )

    monkeypatch.setattr(
        "app.graph.nodes.get_llm_client",
        lambda: fake_llm,
    )

    state = {
        "user_id": "local-user-123",
        "question": "What is my deductible?",
        "generated_answer": "Your deductible is $50.",
        "evidence_chunks": [
            {
                "chunk_id": "chunk-1",
                "chunk_text": "The deductible is $50.",
            }
        ],
        "status": "answered",
    }

    result = verify_answer_grounding_node(state)

    assert result["grounding_status"] == "supported"
    assert result["status"] == "answered"
    assert result["generated_answer"] == "Your deductible is $50."


def test_verify_answer_grounding_node_records_unsupported_answer_for_routing(monkeypatch):
    fake_llm = FakeLLM(
        '{"status": "unsupported", "reason": "The evidence says $50, not $500.", '
        '"unsupported_claims": ["Your deductible is $500."]}'
    )

    monkeypatch.setattr(
        "app.graph.nodes.get_llm_client",
        lambda: fake_llm,
    )

    state = {
        "user_id": "local-user-123",
        "question": "What is my deductible?",
        "generated_answer": "Your deductible is $500.",
        "final_answer": "Your deductible is $500.",
        "citations": [{"chunk_id": "chunk-1"}],
        "evidence_chunks": [
            {
                "chunk_id": "chunk-1",
                "chunk_text": "The deductible is $50.",
            }
        ],
        "status": "answered",
    }

    result = verify_answer_grounding_node(state)

    assert result["grounding_status"] == "unsupported"
    assert result["status"] == "answered"
    assert result["citations"] == [{"chunk_id": "chunk-1"}]
    assert result["final_answer"] == "Your deductible is $500."
    assert result["unsupported_claims"] == ["Your deductible is $500."]


def test_run_rag_workflow_includes_rewrite_and_evidence_fields(monkeypatch):
    rewrite_llm = FakeLLM("urgent care health insurance coverage")
    answer_llm = FakeLLM("Yes, urgent care is covered according to the evidence.")
    verifier_llm = FakeLLM(
        '{"status": "supported", "reason": "The answer is supported by the evidence.", "unsupported_claims": []}'
    )

    llm_calls = [rewrite_llm, answer_llm, verifier_llm]

    def fake_get_llm_client():
        return llm_calls.pop(0)

    def fake_rerank_hybrid_results(
        db,
        user_id,
        query,
        top_k,
        hybrid_top_k,
        vector_top_k,
        bm25_top_k,
        vector_weight,
        bm25_weight,
    ):
        return [
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "document_name": "sample_health_policy.txt",
                "category": "health_insurance",
                "page_number": None,
                "section_title": "Urgent Care",
                "chunk_index": 0,
                "chunk_text": "Urgent care visits are covered under this plan.",
                "reranker_score": 0.95,
                "hybrid_score": 0.88,
            }
        ]

    monkeypatch.setattr(
        "app.graph.nodes.get_llm_client",
        fake_get_llm_client,
    )

    monkeypatch.setattr(
        "app.graph.nodes.rerank_hybrid_results",
        fake_rerank_hybrid_results,
    )

    result = run_rag_workflow(
        db=object(),
        user_id="local-user-123",
        question="Does my health insurance cover urgent care?",
        top_k=5,
        hybrid_top_k=20,
        vector_top_k=20,
        bm25_top_k=20,
    )

    assert result["rewritten_question"] == "urgent care health insurance coverage"
    assert result["evidence_sufficient"] is True
    assert result["evidence_sufficiency_reason"] == "Retrieved evidence is sufficient for answer generation."
    assert result["grounding_status"] == "supported"
    assert result["status"] == "answered"
    assert result["validation_status"] == "supported"
    assert len(result["citations"]) == 1
