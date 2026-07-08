from pathlib import Path
from uuid import uuid4

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.auth.dependencies import get_current_user
from app.config.settings import settings
from app.db.database import SessionLocal, get_db
from app.ingestion.chunking_service import chunk_and_store_document_text
from app.ingestion.document_processing_service import process_document
from app.ingestion.embedding_indexing_service import embed_document_chunks
from app.ingestion.extraction_service import extract_and_store_document_text
from app.models.user import User
from app.schemas.document_schema import (
    DocumentChunkDetailResponse,
    DocumentChunkingResponse,
    DocumentChunkListResponse,
    DocumentDeleteResponse,
    DocumentDetailResponse,
    DocumentEmbeddingResponse,
    DocumentExtractionResponse,
    DocumentListResponse,
    DocumentMetadata,
    DocumentProcessingJobListResponse,
    DocumentProcessingJobResponse,
    DocumentProcessingResponse,
    DocumentUploadResponse,
)
from app.repositories.document_chunk_repository import (
    get_chunk_by_id,
    get_chunks_by_document_for_user,
)
from app.repositories.document_repository import (
    create_document_record,
    delete_document_completely,
    get_document_by_id,
    get_documents_by_user,
)
from app.repositories.document_processing_job_repository import (
    create_processing_job,
    get_processing_job_by_id,
    get_processing_jobs_by_document,
    get_latest_processing_job_for_document,
)
from app.storage.local import LocalStorageService
from app.utils.logger import get_logger


router = APIRouter(prefix="/documents", tags=["Documents"])
debug_router = APIRouter(tags=["Document Debug"])
logger = get_logger(__name__)


ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "text/markdown",
    "text/html",
}

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".html",
}

SUPPORTED_DOCUMENT_STATUSES = {
    "uploaded",
    "processing",
    "extracted",
    "chunked",
    "embedded",
    "failed",
}

SUPPORTED_DOCUMENT_CATEGORIES = {
    "health_insurance",
    "auto_insurance",
    "home_insurance",
    "mortgage",
    "hoa",
    "employer_benefits",
    "internet",
    "utility",
    "banking",
    "credit_card",
    "warranty",
    "travel",
    "general",
}


def _validate_category(category: str) -> str:
    """Return a trimmed category or raise when the upload category is blank."""
    clean_category = category.strip()

    if not clean_category:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="category is required",
        )

    return clean_category


def _validate_file(file: UploadFile) -> None:
    """Validate that an uploaded file has a supported name and content type."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="file is required",
        )

    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in ALLOWED_EXTENSIONS:
        logger.warning("Document upload rejected: unsupported extension %s", file_extension)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension: {file_extension}",
        )

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        logger.warning("Document upload rejected: unsupported content type %s", file.content_type)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported content type: {file.content_type}",
        )


async def _get_file_size_bytes(file: UploadFile) -> int:
    """Measure the spooled upload without copying its contents into memory."""
    def measure() -> int:
        current = file.file.tell()
        file.file.seek(0, 2)
        size = file.file.tell()
        file.file.seek(current)
        return size

    file_size_bytes = await run_in_threadpool(measure)

    max_upload_size_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    if file_size_bytes > max_upload_size_bytes:
        logger.warning(
            "Document upload rejected: file size %s exceeded limit %s",
            file_size_bytes,
            max_upload_size_bytes,
        )
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File size exceeds {settings.MAX_UPLOAD_SIZE_MB} MB limit",
        )

    await file.seek(0)

    return file_size_bytes


def _display_status(document_status: str, job=None) -> str:
    """Map internal document and job states to frontend display statuses."""
    if document_status == "embedded":
        return "ready"
    if document_status == "failed" or getattr(job, "status", None) == "failed":
        return "failed"

    current_step = getattr(job, "current_step", None)
    return {
        "validate": "validating",
        "extract": "extracting",
        "chunk": "chunking",
        "embed": "embedding",
    }.get(
        current_step,
        {
            "uploaded": "uploading",
            "processing": "validating",
            "extracted": "chunking",
            "chunked": "embedding",
        }.get(document_status, document_status),
    )


def _process_document_in_background(document_id: str, user_id: str, job_id: str) -> None:
    """Process an uploaded document in a fresh database session."""
    logger.info(
        "Background document processing started: document_id=%s user_id=%s job_id=%s",
        document_id,
        user_id,
        job_id,
    )
    db = SessionLocal()
    try:
        document = get_document_by_id(db=db, document_id=document_id, user_id=user_id)
        job = get_processing_job_by_id(db, job_id, user_id)
        if document is not None and job is not None:
            result = process_document(db=db, document=document, force=False, job=job)
            logger.info(
                "Background document processing finished: document_id=%s job_id=%s status=%s",
                document_id,
                job_id,
                result.status,
            )
        else:
            logger.warning(
                "Background document processing skipped: document_id=%s job_id=%s document_found=%s job_found=%s",
                document_id,
                job_id,
                document is not None,
                job is not None,
            )
    except Exception:
        logger.exception(
            "Background document processing failed unexpectedly: document_id=%s job_id=%s",
            document_id,
            job_id,
        )
        raise
    finally:
        db.close()


def _to_document_metadata(document, job=None) -> DocumentMetadata:
    """Convert a document model and optional job into API response metadata."""
    return DocumentMetadata(
        document_id=document.id,
        user_id=document.user_id,
        file_name=document.stored_file_name,
        original_file_name=document.original_file_name,
        content_type=document.content_type,
        file_size_bytes=document.file_size_bytes,
        category=document.category,
        storage_provider=document.storage_provider,
        storage_path=document.storage_path,
        status=document.status,
        display_status=_display_status(document.status, job),
        failure_reason=getattr(job, "error_message", None),
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


def _get_owned_document_or_404(db: Session, document_id: str, user_id: str):
    """Return an owned document or raise a 404."""
    document = get_document_by_id(
        db=db,
        document_id=document_id,
        user_id=user_id,
    )

    if document is None or getattr(document, "status", None) == "deleted":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    return document


def _to_chunk_detail_response(chunk) -> DocumentChunkDetailResponse:
    """Convert a chunk model into the public chunk detail response."""
    return DocumentChunkDetailResponse(
        chunk_id=str(chunk.id),
        document_id=str(chunk.document_id),
        user_id=chunk.user_id,
        chunk_text=chunk.chunk_text,
        chunk_index=chunk.chunk_index,
        token_count=chunk.token_count,
        page_number=chunk.page_number,
        section_title=chunk.section_title,
        summary=getattr(chunk, "summary", None),
        keywords=getattr(chunk, "keywords", None) or [],
        hypothetical_questions=getattr(chunk, "hypothetical_questions", None) or [],
        structure_types=getattr(chunk, "structure_types", None) or [],
        status=chunk.status,
        created_at=getattr(chunk, "created_at", None),
        updated_at=getattr(chunk, "updated_at", None),
    )


def _to_processing_job_response(job) -> DocumentProcessingJobResponse:
    """Convert a processing job model into its API response shape."""
    return DocumentProcessingJobResponse(
        job_id=str(job.id),
        document_id=str(job.document_id),
        user_id=job.user_id,
        status=job.status,
        force=bool(job.force),
        current_step=job.current_step,
        steps=job.steps or [],
        error_message=job.error_message,
        started_at=job.started_at,
        completed_at=job.completed_at,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    category: str = Form(...),
    file: UploadFile = File(...),
    user_id: str | None = Form(default=None),  # Legacy compatibility field; the JWT identity is authoritative.
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DocumentUploadResponse:
    """
    Upload a document and queue it for processing for the authenticated user.

    The JWT identity determines the document owner. Any legacy `user_id`
    form field is ignored so clients cannot assign ownership to another user.
    """
    authenticated_user_id = str(current_user.id)
    clean_category = _validate_category(category)

    _validate_file(file)
    file_size_bytes = await _get_file_size_bytes(file)

    storage_service = LocalStorageService()

    storage_document_id = str(uuid4())

    # UploadFile is already spooled by Starlette. Copy it in a worker thread in
    # fixed-size chunks so large files neither occupy the event loop nor get
    # materialized as a second in-memory byte string.
    storage_metadata = await run_in_threadpool(
        storage_service.save_file,
        file.file,
        authenticated_user_id,
        storage_document_id,
        file.filename,
    )
    file_size_bytes = int(storage_metadata.get("file_size_bytes", file_size_bytes))
    max_upload_size_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if file_size_bytes > max_upload_size_bytes:
        await run_in_threadpool(storage_service.delete_file, storage_metadata["storage_path"])
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File size exceeds {settings.MAX_UPLOAD_SIZE_MB} MB limit",
        )

    try:
        document = await run_in_threadpool(
            create_document_record,
            db=db,
            user_id=authenticated_user_id,
            original_file_name=file.filename,
            stored_file_name=storage_metadata["file_name"],
            category=clean_category,
            content_type=file.content_type or "application/octet-stream",
            file_size_bytes=file_size_bytes,
            storage_provider=storage_metadata["storage_provider"],
            storage_path=storage_metadata["storage_path"],
            status="uploaded",
        )
    except Exception:
        logger.exception(
            "Document record creation failed after file save: user_id=%s storage_path=%s",
            authenticated_user_id,
            storage_metadata["storage_path"],
        )
        storage_service.delete_file(storage_metadata["storage_path"])
        raise

    try:
        job = await run_in_threadpool(
            create_processing_job,
            db=db,
            document_id=str(document.id),
            user_id=authenticated_user_id,
        )
    except Exception:
        logger.exception(
            "Document processing job creation failed: document_id=%s user_id=%s",
            document.id,
            authenticated_user_id,
        )
        delete_document_completely(
            db=db,
            document_id=str(document.id),
            user_id=authenticated_user_id,
        )
        raise
    if settings.JOB_EXECUTION_MODE == "inline":
        background_tasks.add_task(
            _process_document_in_background,
            str(document.id),
            authenticated_user_id,
            str(job.id),
        )
    logger.info(
        "Document upload completed and processing queued: document_id=%s user_id=%s job_id=%s",
        document.id,
        authenticated_user_id,
        job.id,
    )

    metadata = _to_document_metadata(document, job)

    return DocumentUploadResponse(
        **metadata.model_dump(),
        job_id=str(job.id),
        message="Document uploaded and processing started",
    )


@router.get("", response_model=DocumentListResponse)
def list_documents(
    user_id: str | None = Query(default=None),  # Legacy compatibility field; the JWT identity is authoritative.
    status_filter: str | None = Query(default=None, alias="status"),
    category: str | None = Query(default=None),
    search: str | None = Query(default=None),
    ready_only: bool = Query(default=False),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DocumentListResponse:
    """
    List documents owned by the authenticated user.

    The JWT identity is used to scope the result set. Any legacy `user_id`
    query parameter is ignored.
    """
    authenticated_user_id = str(current_user.id)
    clean_status = status_filter.strip() if status_filter else None
    clean_category = category.strip() if category else None
    clean_search = search.strip() if search else None
    logger.debug(
        "Listing documents: user_id=%s status=%s category=%s search_present=%s ready_only=%s",
        authenticated_user_id,
        clean_status,
        clean_category,
        bool(clean_search),
        ready_only,
    )

    if clean_status and clean_status not in SUPPORTED_DOCUMENT_STATUSES:
        logger.warning("Document list rejected: unsupported status %s", clean_status)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported document status: {clean_status}. "
                f"Supported statuses: {', '.join(sorted(SUPPORTED_DOCUMENT_STATUSES))}"
            ),
        )

    if clean_category and clean_category not in SUPPORTED_DOCUMENT_CATEGORIES:
        logger.warning("Document list rejected: unsupported category %s", clean_category)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported document category: {clean_category}. "
                f"Supported categories: {', '.join(sorted(SUPPORTED_DOCUMENT_CATEGORIES))}"
            ),
        )

    documents = get_documents_by_user(
        db=db,
        user_id=authenticated_user_id,
        status=clean_status,
        category=clean_category,
        search=clean_search,
        ready_only=ready_only,
    )

    document_metadata = []
    for document in documents:
        latest_job = (
            get_latest_processing_job_for_document(
                db, str(document.id), authenticated_user_id
            )
            if document.status in {"processing", "failed"}
            else None
        )
        document_metadata.append(_to_document_metadata(document, latest_job))

    return DocumentListResponse(
        user_id=authenticated_user_id,
        documents=document_metadata,
        count=len(document_metadata),
    )


@debug_router.get("/chunks/{chunk_id}", response_model=DocumentChunkDetailResponse)
def get_document_chunk_detail(
    chunk_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DocumentChunkDetailResponse:
    """Return one owned document chunk without exposing its embedding vector."""
    user_id = str(current_user.id)
    chunk = get_chunk_by_id(
        db=db,
        chunk_id=chunk_id,
        user_id=user_id,
    )

    if chunk is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chunk not found",
        )

    return _to_chunk_detail_response(chunk)


@debug_router.get("/{document_id}/chunks", response_model=DocumentChunkListResponse)
def list_document_chunks(
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DocumentChunkListResponse:
    """Return chunks for one owned document in chunk order."""
    user_id = str(current_user.id)
    _get_owned_document_or_404(
        db=db,
        document_id=document_id,
        user_id=user_id,
    )

    chunks = get_chunks_by_document_for_user(
        db=db,
        document_id=document_id,
        user_id=user_id,
    )

    return DocumentChunkListResponse(
        document_id=document_id,
        user_id=user_id,
        chunks=[
            _to_chunk_detail_response(chunk)
            for chunk in chunks
        ],
    )


@debug_router.get("/processing-jobs/{job_id}", response_model=DocumentProcessingJobResponse)
def get_processing_job_detail(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DocumentProcessingJobResponse:
    """Return one processing job owned by the authenticated user."""
    user_id = str(current_user.id)
    job = get_processing_job_by_id(
        db=db,
        job_id=job_id,
        user_id=user_id,
    )

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Processing job not found",
        )

    return _to_processing_job_response(job)


@debug_router.get("/{document_id}/processing-jobs", response_model=DocumentProcessingJobListResponse)
def list_document_processing_jobs(
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DocumentProcessingJobListResponse:
    """Return processing jobs for one owned document."""
    user_id = str(current_user.id)
    _get_owned_document_or_404(
        db=db,
        document_id=document_id,
        user_id=user_id,
    )

    jobs = get_processing_jobs_by_document(
        db=db,
        document_id=document_id,
        user_id=user_id,
    )

    return DocumentProcessingJobListResponse(
        document_id=document_id,
        user_id=user_id,
        jobs=[
            _to_processing_job_response(job)
            for job in jobs
        ],
    )


@debug_router.get("/{document_id}", response_model=DocumentDetailResponse)
def get_document_detail(
    document_id: str,
    user_id: str | None = Query(default=None),  # Legacy compatibility field; the JWT identity is authoritative.
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DocumentDetailResponse:
    """Return one document owned by the authenticated user."""
    authenticated_user_id = str(current_user.id)
    document = _get_owned_document_or_404(
        db=db,
        document_id=document_id,
        user_id=authenticated_user_id,
    )
    latest_job = (
        get_latest_processing_job_for_document(
            db, str(document.id), authenticated_user_id
        )
        if document.status in {"processing", "failed"}
        else None
    )
    metadata = _to_document_metadata(document, latest_job)

    return DocumentDetailResponse(**metadata.model_dump())


@debug_router.post("/{document_id}/extract", response_model=DocumentExtractionResponse)
def extract_document(
    document_id: str,
    user_id: str | None = Query(default=None),  # Legacy compatibility field; the JWT identity is authoritative.
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DocumentExtractionResponse:
    """Extract text for an uploaded document owned by the authenticated user."""
    authenticated_user_id = str(current_user.id)
    document = get_document_by_id(
        db=db,
        document_id=document_id,
        user_id=authenticated_user_id,
    )

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    result = extract_and_store_document_text(db=db, document=document)
    return DocumentExtractionResponse(**result)


@debug_router.post("/{document_id}/embed", response_model=DocumentEmbeddingResponse)
def embed_document(
    document_id: str,
    user_id: str | None = Query(default=None),  # Legacy compatibility field; the JWT identity is authoritative.
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DocumentEmbeddingResponse:
    """Embed chunks for a chunked document owned by the authenticated user."""
    authenticated_user_id = str(current_user.id)
    document = get_document_by_id(
        db=db,
        document_id=document_id,
        user_id=authenticated_user_id,
    )

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    if document.status != "chunked":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document must be in chunked status before embedding",
        )

    result = embed_document_chunks(db=db, document=document)
    return DocumentEmbeddingResponse(**result)


@debug_router.post("/{document_id}/chunk", response_model=DocumentChunkingResponse)
def chunk_document(
    document_id: str,
    user_id: str | None = Query(default=None),  # Legacy compatibility field; the JWT identity is authoritative.
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DocumentChunkingResponse:
    """Run structure-aware chunking for an extracted document owned by the user."""
    authenticated_user_id = str(current_user.id)
    document = get_document_by_id(
        db=db,
        document_id=document_id,
        user_id=authenticated_user_id,
    )

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    if document.status != "extracted":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document must be extracted before chunking",
        )

    result = chunk_and_store_document_text(db=db, document=document)
    return DocumentChunkingResponse(**result)


@debug_router.post("/{document_id}/process", response_model=DocumentProcessingResponse)
def process_document_endpoint(
    document_id: str,
    user_id: str | None = Query(default=None),  # Legacy compatibility field; the JWT identity is authoritative.
    force: bool = Query(default=False),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DocumentProcessingResponse:
    """Synchronously run the full document preparation pipeline."""
    authenticated_user_id = str(current_user.id)
    document = get_document_by_id(
        db=db,
        document_id=document_id,
        user_id=authenticated_user_id,
    )

    if document is None or getattr(document, "status", None) == "deleted":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    result = process_document(db=db, document=document, force=force)
    return DocumentProcessingResponse.model_validate(result)


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
def delete_document(
    document_id: str,
    user_id: str | None = Query(default=None),  # Legacy compatibility field; the JWT identity is authoritative.
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DocumentDeleteResponse:
    """Permanently delete an owned document and all generated data."""
    authenticated_user_id = str(current_user.id)

    document = delete_document_completely(
        db=db,
        document_id=document_id,
        user_id=authenticated_user_id,
    )

    if document is None:
        logger.warning(
            "Document delete requested for missing document: document_id=%s user_id=%s",
            document_id,
            authenticated_user_id,
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    logger.info(
        "Document deleted: document_id=%s user_id=%s",
        document.id,
        document.user_id,
    )
    return DocumentDeleteResponse(
        document_id=document.id,
        user_id=document.user_id,
        status=document.status,
        message="Document deleted successfully",
    )


if settings.ENABLE_DEBUG_ENDPOINTS:
    router.add_api_route(
        "/chunks/{chunk_id}",
        get_document_chunk_detail,
        methods=["GET"],
        response_model=DocumentChunkDetailResponse,
    )
    router.add_api_route(
        "/{document_id}/chunks",
        list_document_chunks,
        methods=["GET"],
        response_model=DocumentChunkListResponse,
    )
    router.add_api_route(
        "/processing-jobs/{job_id}",
        get_processing_job_detail,
        methods=["GET"],
        response_model=DocumentProcessingJobResponse,
    )
    router.add_api_route(
        "/{document_id}/processing-jobs",
        list_document_processing_jobs,
        methods=["GET"],
        response_model=DocumentProcessingJobListResponse,
    )
    router.add_api_route(
        "/{document_id}",
        get_document_detail,
        methods=["GET"],
        response_model=DocumentDetailResponse,
    )
    router.add_api_route(
        "/{document_id}/extract",
        extract_document,
        methods=["POST"],
        response_model=DocumentExtractionResponse,
    )
    router.add_api_route(
        "/{document_id}/embed",
        embed_document,
        methods=["POST"],
        response_model=DocumentEmbeddingResponse,
    )
    router.add_api_route(
        "/{document_id}/chunk",
        chunk_document,
        methods=["POST"],
        response_model=DocumentChunkingResponse,
    )
    router.add_api_route(
        "/{document_id}/process",
        process_document_endpoint,
        methods=["POST"],
        response_model=DocumentProcessingResponse,
    )
