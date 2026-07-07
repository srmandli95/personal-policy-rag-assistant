from functools import lru_cache
from pathlib import Path
import re
from typing import Any

import yaml


PROMPTS_PATH = Path(__file__).resolve().parents[1] / "config" / "prompts.yaml"


@lru_cache
def load_prompts() -> dict[str, str]:
    """Load prompt templates from the configured YAML file."""
    with PROMPTS_PATH.open("r", encoding="utf-8") as file:
        prompts = yaml.safe_load(file) or {}

    return prompts


def build_chat_history_context(chat_history: list[dict[str, Any]], max_answer_chars: int = 500) -> str:
    """Format prior chat messages for contextual query rewriting."""
    if not chat_history:
        return "No prior chat context."

    history_lines: list[str] = []

    for index, message in enumerate(chat_history, start=1):
        question = str(message.get("question") or "").strip()
        answer = str(message.get("answer") or "").strip()

        if len(answer) > max_answer_chars:
            answer = answer[:max_answer_chars].rstrip() + "..."

        parts = [f"Turn {index}"]
        if question:
            parts.append(f"User: {question}")
        if answer:
            parts.append(f"Assistant: {answer}")

        history_lines.append("\n".join(parts))

    return "\n\n".join(history_lines)


def build_query_rewrite_prompt(
    question: str,
    chat_history: list[dict[str, Any]] | None = None,
) -> str:
    """Build a prompt for rewriting a user query."""
    prompts = load_prompts()
    if chat_history:
        template = prompts.get("contextual_query_rewrite_prompt") or prompts["query_rewrite_prompt"]
        return template.format(
            question=question,
            chat_history=build_chat_history_context(chat_history),
        )

    template = prompts["query_rewrite_prompt"]
    return template.format(question=question)


def build_retrieval_retry_prompt(
    question: str,
    previous_query: str,
    attempt: int,
) -> str:
    """Build a prompt for broadening an unsuccessful retrieval query."""
    template = load_prompts()["retrieval_retry_prompt"]
    return template.format(
        question=question,
        previous_query=previous_query,
        attempt=attempt,
    )


def build_evidence_context(evidence_chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks as evidence context."""
    if not evidence_chunks:
        return "No evidence chunks were retrieved."

    evidence_lines: list[str] = []

    for index, chunk in enumerate(evidence_chunks, start=1):
        chunk_text = chunk.get("chunk_text") or ""
        document_name = chunk.get("document_name") or chunk.get("original_file_name") or "Unknown document"
        page_number = chunk.get("page_number")
        section_title = chunk.get("section_title")
        chunk_id = chunk.get("chunk_id") or chunk.get("id")

        metadata_parts = [
            f"Source {index}",
            f"Document: {document_name}",
        ]

        if page_number is not None:
            metadata_parts.append(f"Page: {page_number}")

        if section_title:
            metadata_parts.append(f"Section: {section_title}")

        if chunk_id:
            metadata_parts.append(f"Chunk ID: {chunk_id}")

        evidence_lines.append(
            "\n".join(
                [
                    " | ".join(metadata_parts),
                    f"Content: {chunk_text}",
                ]
            )
        )

    return "\n\n".join(evidence_lines)


def build_answer_prompt(
    question: str,
    evidence_chunks: list[dict[str, Any]],
) -> str:
    """Build the grounded answer-generation prompt."""
    prompts = load_prompts()
    template = prompts["answer_generation_prompt"]

    evidence_context = build_evidence_context(evidence_chunks)

    return template.format(
        question=question,
        evidence_context=evidence_context,
    )


def build_answer_regeneration_prompt(
    question: str,
    evidence_chunks: list[dict[str, Any]],
    previous_answer: str,
    feedback: str,
) -> str:
    """Build a grounded regeneration prompt using validation feedback."""
    template = load_prompts()["answer_regeneration_prompt"]
    return template.format(
        question=question,
        evidence_context=build_evidence_context(evidence_chunks),
        previous_answer=previous_answer,
        feedback=feedback,
    )


def build_answer_verification_prompt(
    question: str,
    answer: str,
    evidence_chunks: list[dict[str, Any]],
) -> str:
    """Build the post-generation grounding verification prompt."""
    prompts = load_prompts()
    template = prompts["answer_verification_prompt"]

    evidence_context = build_evidence_context(evidence_chunks)

    return template.format(
        question=question,
        answer=answer,
        evidence_context=evidence_context,
    )


def get_refusal_message() -> str:
    """Return the configured refusal message."""
    prompts = load_prompts()

    return prompts.get(
        "refusal_message",
        "I could not find enough evidence in your uploaded documents to answer this question.",
    ).strip()


def strip_generated_source_metadata(answer: str) -> str:
    """Remove source metadata that belongs in the structured citations field."""
    cleaned_lines: list[str] = []

    for line in answer.strip().splitlines():
        stripped = line.strip()
        if re.match(r"^(sources?|citations?)\s*:", stripped, flags=re.IGNORECASE):
            continue
        if re.match(r"^(sources?|citations?)\s*$", stripped, flags=re.IGNORECASE):
            continue
        if re.search(r"\bchunk\s+id\s*:", stripped, flags=re.IGNORECASE):
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()
