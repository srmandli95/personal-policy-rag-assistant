from types import SimpleNamespace
import pytest

from app.embeddings import embedding_service
from app.reranking import reranking_service
from app.generation import llm_client
from app.generation.prompt_builder import build_evidence_context
from app.graph.nodes import rewrite_query_node_async


def test_embedding_service_is_cached(monkeypatch):
    embedding_service.get_embedding_service.cache_clear()
    created = []
    monkeypatch.setattr(
        embedding_service,
        "LocalEmbeddingService",
        lambda model_name: created.append(model_name) or SimpleNamespace(model_name=model_name),
    )
    first = embedding_service.get_embedding_service()
    second = embedding_service.get_embedding_service()
    assert first is second
    assert len(created) == 1
    embedding_service.get_embedding_service.cache_clear()


def test_reranker_is_cached(monkeypatch):
    reranking_service.get_reranker.cache_clear()


@pytest.mark.asyncio
async def test_async_rewrite_skips_provider_without_history(monkeypatch):
    def fail_if_called():
        raise AssertionError("auxiliary model should not be called")

    monkeypatch.setattr("app.graph.nodes.get_aux_llm_client", fail_if_called)
    state = {"question": "What is covered?", "chat_history": []}
    result = await rewrite_query_node_async(state)
    assert result["rewritten_question"] == "What is covered?"


def test_auxiliary_client_uses_auxiliary_model(monkeypatch):
    llm_client.get_llm_client.cache_clear()
    captured = {}

    def fake_client(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(llm_client, "OpenAILLMClient", fake_client)
    llm_client.get_aux_llm_client()
    assert captured["model_name"] == llm_client.settings.OPENAI_AUX_MODEL_NAME
    assert captured["max_output_tokens"] == llm_client.settings.OPENAI_AUX_MAX_TOKENS
    llm_client.get_llm_client.cache_clear()


def test_evidence_context_is_bounded(monkeypatch):
    monkeypatch.setattr("app.generation.prompt_builder.settings.PROMPT_MAX_EVIDENCE_CHARS_PER_CHUNK", 20)
    monkeypatch.setattr("app.generation.prompt_builder.settings.PROMPT_MAX_EVIDENCE_CHARS", 200)
    context = build_evidence_context([
        {"chunk_id": "one", "document_name": "policy.txt", "chunk_text": "x" * 100}
    ])
    assert "x" * 20 in context
    assert "x" * 21 not in context
    created = []
    monkeypatch.setattr(
        reranking_service,
        "CrossEncoderReranker",
        lambda model_name: created.append(model_name) or SimpleNamespace(model_name=model_name),
    )
    first = reranking_service.get_reranker()
    second = reranking_service.get_reranker()
    assert first is second
    assert len(created) == 1
    reranking_service.get_reranker.cache_clear()
