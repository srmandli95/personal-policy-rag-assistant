from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.health_routes import router as health_router
from app.api.document_routes import router as document_router
from app.api.chat_routes import router as chat_router
from app.api.auth_routes import router as auth_router
from app.api.retrieval_routes import router as retrieval_router
from app.api.evaluation_routes import router as evaluation_router
from app.models.chat_session import ChatSession
from app.models.chat_message import ChatMessage
from app.models.document_processing_job import DocumentProcessingJob
from app.models.user import User
from app.models.evaluation_dataset import EvaluationDataset
from app.models.evaluation_run import EvaluationRun
from app.config.settings import settings
from app.db.database import Base, engine
from app.db.schema_migrations import ensure_document_chunk_metadata_columns
from app.models import Document, DocumentChunk


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Create database tables during application startup."""
    Base.metadata.create_all(bind=engine)
    ensure_document_chunk_metadata_columns(engine)
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    description=(
        "A reusable document RAG framework for uploading any document "
        "and asking citation-backed questions over its contents."
    ),
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(document_router)
app.include_router(chat_router)
app.include_router(auth_router)
app.include_router(evaluation_router)

if settings.ENABLE_DEBUG_ENDPOINTS and settings.ENABLE_RETRIEVAL_DEBUG_ENDPOINTS:
    app.include_router(retrieval_router)


@app.get("/")
def root() -> dict[str, str]:
    """Return a basic API health message."""
    return {
        "message": "Welcome to the RAG application backend",
        "docs": "/docs",
    }
