import pytest

from app.graph.rag_graph import build_rag_graph, run_rag_workflow


class FakeLLM:
    def __init__(self, response: str):
        self.response = response
        self.called = False

    def generate(self, prompt: str):
        self.called = True
        return self.response


def test_build_rag_graph_compiles():
    graph = build_rag_graph()

    assert graph is not None


def test_run_rag_workflow_returns_final_response(monkeypatch):
    rewrite_llm = FakeLLM("late payment DTE Energy consequences")
    answer_llm = FakeLLM("A late payment may result in a late payment charge based on the evidence.")
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
        assert query == "late payment DTE Energy consequences"
        assert top_k == 8
        assert vector_weight == 0.6
        assert bm25_weight == 0.4

        return [
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "document_name": "yourguide.pdf",
                "category": "utility_policy",
                "page_number": 10,
                "section_title": "Billing",
                "chunk_index": 3,
                "chunk_text": "A late payment charge may apply when payment is received after the due date.",
                "reranker_score": 0.91,
                "hybrid_score": 0.83,
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

    response = run_rag_workflow(
        db=object(),
        user_id="local-user-123",
        question="can you explain me the consequences for the late payment of my bill for DTE Energy?",
        top_k=5,
        hybrid_top_k=20,
        vector_top_k=20,
        bm25_top_k=20,
    )

    assert response["user_id"] == "local-user-123"
    assert response["question"] == "can you explain me the consequences for the late payment of my bill for DTE Energy?"
    assert response["rewritten_question"] == "late payment DTE Energy consequences"
    assert response["evidence_sufficient"] is True
    assert response["grounding_status"] == "supported"
    assert response["status"] == "answered"
    assert response["validation_status"] == "supported"
    assert len(response["citations"]) == 1
    assert response["retrieval_attempts"] == 1
    assert response["generation_attempts"] == 1


def test_rag_graph_retries_retrieval_with_refined_query(monkeypatch):
    llm_calls = [
        FakeLLM("initial policy wording"),
        FakeLLM("alternative contract terminology"),
        FakeLLM("The policy provides coverage."),
        FakeLLM('{"status": "supported", "reason": "Supported.", "unsupported_claims": []}'),
    ]
    queries = []

    monkeypatch.setattr("app.graph.nodes.get_llm_client", lambda: llm_calls.pop(0))

    def fake_retrieval(**kwargs):
        queries.append(kwargs["query"])
        if len(queries) == 1:
            return []
        return [{
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "document_name": "policy.pdf",
            "chunk_text": "The policy provides coverage.",
            "reranker_score": 0.9,
        }]

    monkeypatch.setattr("app.graph.nodes.rerank_hybrid_results", fake_retrieval)

    response = run_rag_workflow(
        db=object(),
        user_id="user-1",
        question="Is this covered?",
        max_retrieval_attempts=2,
    )

    assert queries == ["initial policy wording", "alternative contract terminology"]
    assert response["retrieval_attempts"] == 2
    assert response["generation_attempts"] == 1
    assert response["status"] == "answered"


def test_rag_graph_regenerates_after_unsupported_grounding(monkeypatch):
    llm_calls = [
        FakeLLM("deductible amount"),
        FakeLLM("The deductible is $500."),
        FakeLLM(
            '{"status": "unsupported", "reason": "The evidence says $50.", '
            '"unsupported_claims": ["The deductible is $500."]}'
        ),
        FakeLLM("The deductible is $50."),
        FakeLLM('{"status": "supported", "reason": "Supported.", "unsupported_claims": []}'),
    ]
    prompts = []

    def fake_llm_client():
        client = llm_calls.pop(0)
        original_generate = client.generate

        def capture(prompt):
            prompts.append(prompt)
            return original_generate(prompt)

        client.generate = capture
        return client

    monkeypatch.setattr("app.graph.nodes.get_llm_client", fake_llm_client)
    monkeypatch.setattr(
        "app.graph.nodes.rerank_hybrid_results",
        lambda **_: [{
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "document_name": "policy.pdf",
            "chunk_text": "The deductible is $50.",
            "reranker_score": 0.95,
        }],
    )

    response = run_rag_workflow(
        db=object(),
        user_id="user-1",
        question="What is the deductible?",
        max_generation_attempts=2,
    )

    assert response["answer"] == "The deductible is $50."
    assert response["generation_attempts"] == 2
    assert response["grounding_status"] == "supported"
    assert response["validation_status"] == "supported"
    assert response["status"] == "answered"
    assert any("The evidence says $50." in prompt for prompt in prompts)


def test_rag_graph_refuses_after_retrieval_attempts_are_exhausted(monkeypatch):
    llm_calls = [FakeLLM("first query"), FakeLLM("second query")]
    monkeypatch.setattr("app.graph.nodes.get_llm_client", lambda: llm_calls.pop(0))
    monkeypatch.setattr("app.graph.nodes.rerank_hybrid_results", lambda **_: [])

    response = run_rag_workflow(
        db=object(),
        user_id="user-1",
        question="Is this covered?",
        max_retrieval_attempts=2,
    )

    assert response["status"] == "refused"
    assert response["retrieval_attempts"] == 2
    assert response["generation_attempts"] == 0
    assert response["citations"] == []


def test_rag_graph_refuses_after_generation_attempts_are_exhausted(monkeypatch):
    unsupported = (
        '{"status": "unsupported", "reason": "The answer is not supported.", '
        '"unsupported_claims": ["Unsupported claim"]}'
    )
    llm_calls = [
        FakeLLM("coverage query"),
        FakeLLM("Unsupported first answer."),
        FakeLLM(unsupported),
        FakeLLM("Unsupported second answer."),
        FakeLLM(unsupported),
    ]
    monkeypatch.setattr("app.graph.nodes.get_llm_client", lambda: llm_calls.pop(0))
    monkeypatch.setattr(
        "app.graph.nodes.rerank_hybrid_results",
        lambda **_: [{
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "document_name": "policy.pdf",
            "chunk_text": "The policy has limited evidence.",
            "reranker_score": 0.9,
        }],
    )

    response = run_rag_workflow(
        db=object(),
        user_id="user-1",
        question="Is this covered?",
        max_generation_attempts=2,
    )

    assert response["status"] == "refused"
    assert response["retrieval_attempts"] == 1
    assert response["generation_attempts"] == 2
    assert response["validation_status"] == "unsupported"
    assert response["citations"] == []


@pytest.mark.parametrize("field", ["max_retrieval_attempts", "max_generation_attempts"])
@pytest.mark.parametrize("value", [0, 6, True])
def test_rag_graph_rejects_invalid_attempt_limits(field, value):
    with pytest.raises(ValueError, match="integer between 1 and 5"):
        run_rag_workflow(
            db=object(),
            user_id="user-1",
            question="Question",
            **{field: value},
        )
