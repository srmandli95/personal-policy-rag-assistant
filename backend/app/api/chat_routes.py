from typing import Any
import asyncio

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.auth.dependencies import get_current_user
from app.config.settings import settings
from app.db.database import SessionLocal, get_async_db
from app.graph.rag_graph import run_rag_workflow, run_rag_workflow_async
from app.generation.llm_client import LLMRateLimitError
from app.models.user import User
from app.retrieval.retrieval_settings import (
    RetrievalSettings,
    validate_retrieval_settings,
)
from app.schemas.chat_schema import (
    AskRequest,
    AskResponse,
    ChatMessageEvidenceResponse,
    ChatMessageResponse,
    ChatSessionDeleteResponse,
    ChatSessionDetailResponse,
    ChatSessionListResponse,
    ChatSessionResponse,
)
from app.repositories import chat_repository as chat_service
from app.utils.logger import get_logger


router = APIRouter(prefix="/chat", tags=["Chat"])
debug_router = APIRouter(tags=["Chat Debug"])
logger = get_logger(__name__)
CHAT_HISTORY_TURN_LIMIT = 6
rag_request_slots = asyncio.Semaphore(max(1, settings.API_MAX_CONCURRENT_RAG_REQUESTS))


def _run_rag_workflow_with_session(**kwargs: Any) -> dict[str, Any]:
    """Run the RAG workflow with the active synchronous database session."""
    db = SessionLocal()
    try:
        logger.debug(
            "RAG workflow thread started: user_id=%s top_k=%s",
            kwargs.get("user_id"),
            kwargs.get("top_k"),
        )
        return run_rag_workflow(db=db, **kwargs)
    finally:
        db.close()


def _to_chat_session_response(session: Any) -> ChatSessionResponse:
    """Convert a chat session model into its API response shape."""
    return ChatSessionResponse(
        session_id=session.id,
        user_id=session.user_id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


def _to_chat_message_response(message: Any) -> ChatMessageResponse:
    """Convert a chat message model into its API response shape."""
    return ChatMessageResponse(
        message_id=message.id,
        session_id=message.session_id,
        user_id=message.user_id,
        question=message.question,
        rewritten_question=message.rewritten_question,
        answer=message.answer,
        citations=message.citations or [],
        evidence_chunk_count=message.evidence_chunk_count or 0,
        model_name=message.model_name,
        status=message.status,
        validation_status=message.validation_status,
        validation_reason=message.validation_reason,
        evidence_sufficient=message.evidence_sufficient,
        evidence_sufficiency_reason=message.evidence_sufficiency_reason,
        created_at=message.created_at,
    )


def _to_rewrite_history(messages: list[Any], limit: int = CHAT_HISTORY_TURN_LIMIT) -> list[dict[str, str]]:
    """Convert recent persisted messages into bounded query-rewrite context."""
    recent_messages = messages[-limit:] if limit > 0 else []

    return [
        {
            "question": message.question,
            "answer": message.answer or "",
        }
        for message in recent_messages
        if getattr(message, "question", None)
    ]


@router.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    async_db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user),
) -> AskResponse:
    """
    Generate an answer for the authenticated user using the RAG workflow.

    The JWT identity is used for authorization, chat-session ownership, and
    message persistence. Any legacy `user_id` supplied by the client is ignored.
    """
    if not request.question or not request.question.strip():
        logger.warning("Chat ask rejected: empty question")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="question is required",
        )

    user_id = str(current_user.id)
    question = request.question.strip()
    logger.info(
        "Chat ask received: user_id=%s session_id=%s question_length=%s",
        user_id,
        request.session_id,
        len(question),
    )

    try:
        retrieval_settings = validate_retrieval_settings(
            RetrievalSettings(
                top_k=request.top_k,
                hybrid_top_k=request.hybrid_top_k,
                vector_top_k=request.vector_top_k,
                bm25_top_k=request.bm25_top_k,
                rerank_top_k=request.rerank_top_k,
                vector_weight=request.vector_weight,
                bm25_weight=request.bm25_weight,
                min_reranker_score=request.min_reranker_score,
            ),
            require_final_top_k_within_rerank=True,
        )
    except LLMRateLimitError as exc:
        logger.warning("LLM provider capacity exhausted: user_id=%s", user_id)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="The answer service is temporarily busy. Please retry shortly.",
            headers={"Retry-After": "5"},
        ) from exc
    except ValueError as exc:
        logger.warning(
            "Chat ask rejected by retrieval settings: user_id=%s error=%s",
            user_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    try:
        chat_session = await chat_service.get_or_create_chat_session(
            db=async_db,
            user_id=user_id,
            session_id=request.session_id,
            question=question,
        )
    except ValueError as exc:
        logger.warning(
            "Chat ask rejected because session was not found: user_id=%s session_id=%s",
            user_id,
            request.session_id,
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    chat_history: list[dict[str, str]] = []
    try:
        previous_messages = await chat_service.get_chat_messages_by_session(
            db=async_db,
            session_id=chat_session.id,
            user_id=user_id,
        )
        chat_history = _to_rewrite_history(previous_messages)
    except Exception as exc:
        logger.warning(
            "Chat history unavailable for query rewrite; continuing without it: user_id=%s session_id=%s error=%s",
            user_id,
            chat_session.id,
            exc,
        )

    try:
        async with rag_request_slots:
            if settings.ASYNC_RAG_WORKFLOW:
                sync_db = SessionLocal()
                try:
                    result = await run_rag_workflow_async(
                        db=sync_db,
                        user_id=str(current_user.id),
                        question=question,
                        chat_history=chat_history,
                        top_k=retrieval_settings.top_k,
                        hybrid_top_k=retrieval_settings.hybrid_top_k,
                        vector_top_k=retrieval_settings.vector_top_k,
                        bm25_top_k=retrieval_settings.bm25_top_k,
                        rerank_top_k=retrieval_settings.rerank_top_k,
                        vector_weight=retrieval_settings.vector_weight,
                        bm25_weight=retrieval_settings.bm25_weight,
                        min_reranker_score=retrieval_settings.min_reranker_score,
                    )
                finally:
                    sync_db.close()
            else:
                result = await run_in_threadpool(
                    _run_rag_workflow_with_session,
                    user_id=str(current_user.id),
                    question=question,
                    chat_history=chat_history,
                    top_k=retrieval_settings.top_k,
                    hybrid_top_k=retrieval_settings.hybrid_top_k,
                    vector_top_k=retrieval_settings.vector_top_k,
                    bm25_top_k=retrieval_settings.bm25_top_k,
                    rerank_top_k=retrieval_settings.rerank_top_k,
                    vector_weight=retrieval_settings.vector_weight,
                    bm25_weight=retrieval_settings.bm25_weight,
                    min_reranker_score=retrieval_settings.min_reranker_score,
                )
    except ValueError as exc:
        logger.warning("RAG workflow rejected chat ask: user_id=%s error=%s", user_id, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception:
        logger.exception("RAG workflow failed unexpectedly: user_id=%s", user_id)
        raise

    chat_message = await chat_service.create_chat_message(
        db=async_db,
        session_id=chat_session.id,
        user_id=user_id,
        question=question,
        answer_response=result,
    )

    logger.info(
        "Chat ask completed: user_id=%s session_id=%s message_id=%s status=%s evidence_chunks=%s",
        user_id,
        chat_session.id,
        chat_message.id,
        result.get("status"),
        result.get("evidence_chunk_count") or len(result.get("evidence_chunks") or []),
    )
    return AskResponse(
        user_id=user_id,
        question=result.get("question") or question,
        rewritten_question=result.get("rewritten_question"),
        answer=result.get("answer") or "",
        citations=result.get("citations") or [],
        evidence_chunk_count=result.get("evidence_chunk_count") or len(
            result.get("evidence_chunks") or []
        ),
        model_name=result.get("model_name"),
        status=result.get("status"),
        validation_status=result.get("validation_status"),
        validation_reason=result.get("validation_reason"),
        grounding_status=result.get("grounding_status"),
        grounding_reason=result.get("grounding_reason"),
        unsupported_claims=result.get("unsupported_claims") or [],
        evidence_sufficient=result.get("evidence_sufficient"),
        evidence_sufficiency_reason=result.get("evidence_sufficiency_reason"),
        session_id=chat_session.id,
        message_id=chat_message.id,
    )


@debug_router.get("/messages/{message_id}/evidence", response_model=ChatMessageEvidenceResponse)
async def get_chat_message_evidence(
    message_id: str,
    async_db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user),
) -> ChatMessageEvidenceResponse:
    """Return stored retrieved evidence for an owned chat message."""
    user_id = str(current_user.id)
    message = await chat_service.get_chat_message_by_id(
        db=async_db,
        message_id=message_id,
        user_id=user_id,
    )

    if message is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat message not found",
        )

    retrieved_chunks = message.retrieved_chunks or []
    return ChatMessageEvidenceResponse(
        message_id=message.id,
        session_id=message.session_id,
        user_id=message.user_id,
        question=message.question,
        answer=message.answer,
        citations=message.citations or [],
        retrieved_chunks=retrieved_chunks,
        evidence_chunk_count=message.evidence_chunk_count or len(retrieved_chunks),
        created_at=message.created_at,
    )


@router.get("/sessions", response_model=ChatSessionListResponse)
async def list_chat_sessions(
    async_db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user),
) -> ChatSessionListResponse:
    """
    List chat sessions that belong to the authenticated user.

    The endpoint uses the JWT identity to scope the result set and does not
    trust any client-supplied user identifier.
    """
    user_id = str(current_user.id)
    logger.debug("Listing chat sessions: user_id=%s", user_id)

    sessions = await chat_service.get_chat_sessions_by_user(
        db=async_db,
        user_id=user_id,
    )

    return ChatSessionListResponse(
        user_id=user_id,
        sessions=[
            _to_chat_session_response(session)
            for session in sessions
        ],
    )


@router.delete(
    "/sessions/{session_id}",
    response_model=ChatSessionDeleteResponse,
)
async def delete_chat_session(
    session_id: str,
    async_db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user),
) -> ChatSessionDeleteResponse:
    """Delete an owned chat session and return its identifier."""
    user_id = str(current_user.id)
    logger.info(
        "Chat session delete requested: user_id=%s session_id=%s",
        user_id,
        session_id,
    )
    chat_session = await chat_service.delete_chat_session(
        db=async_db,
        session_id=session_id,
        user_id=user_id,
    )

    if chat_session is None:
        logger.warning(
            "Chat session delete requested for missing session: user_id=%s session_id=%s",
            user_id,
            session_id,
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found",
        )

    return ChatSessionDeleteResponse(
        session_id=chat_session.id,
        user_id=chat_session.user_id,
        message="Chat session deleted successfully",
    )


@router.get("/sessions/{session_id}", response_model=ChatSessionDetailResponse)
async def get_chat_session_detail(
    session_id: str,
    async_db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user),
) -> ChatSessionDetailResponse:
    """
    Return a chat session only when it belongs to the authenticated user.

    The endpoint uses the JWT identity for authorization and returns 404 when
    the requested session is owned by a different user.
    """
    user_id = str(current_user.id)
    logger.debug(
        "Chat session detail requested: user_id=%s session_id=%s",
        user_id,
        session_id,
    )

    chat_session = await chat_service.get_chat_session(
        db=async_db,
        session_id=session_id,
        user_id=user_id,
    )

    if chat_session is None:
        logger.warning(
            "Chat session detail requested for missing session: user_id=%s session_id=%s",
            user_id,
            session_id,
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found",
        )

    messages = await chat_service.get_chat_messages_by_session(
        db=async_db,
        session_id=session_id,
        user_id=user_id,
    )

    return ChatSessionDetailResponse(
        session_id=chat_session.id,
        user_id=chat_session.user_id,
        title=chat_session.title,
        messages=[
            _to_chat_message_response(message)
            for message in messages
        ],
    )


if settings.ENABLE_DEBUG_ENDPOINTS:
    router.add_api_route(
        "/messages/{message_id}/evidence",
        get_chat_message_evidence,
        methods=["GET"],
        response_model=ChatMessageEvidenceResponse,
    )
