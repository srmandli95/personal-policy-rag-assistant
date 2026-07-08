from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    APP_NAME: str = "Document RAG"
    ENV: str = "local"
    ENABLE_DEBUG_ENDPOINTS: bool = False
    ENABLE_RETRIEVAL_DEBUG_ENDPOINTS: bool = False

    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    DATABASE_POOL_TIMEOUT_SECONDS: int = 30

    STORAGE_PROVIDER: str = "local"

    DATABASE_URL: str

    ASYNC_DATABASE_URL: str = (
        "postgresql+asyncpg://personal_policy_rag_assistant:"
        "personal_policy_rag_assistant@postgres:5432/personal_policy_rag_assistant"
    )

    RAW_DOCUMENTS_DIR: str = "/app/data/raw_documents"
    REDACTED_DOCUMENTS_DIR: str = "/app/data/redacted_documents"
    EXTRACTED_TEXT_DIR: str = "/app/data/extracted_text"
    PROCESSED_CHUNKS_DIR: str = "/app/data/processed_chunks"

    EMBEDDING_PROVIDER: str = "local"
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    RERANKER_PROVIDER: str = "local"
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    RERANKER_TOP_K: int = 8
    INFERENCE_MAX_CONCURRENCY: int = 2
    PRELOAD_LOCAL_MODELS: bool = True

    LLM_PROVIDER: str = "openai"
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL_NAME: str = "gpt-4o-mini"
    OPENAI_AUX_MODEL_NAME: str = "gpt-4o-mini"
    OPENAI_MAX_CONCURRENCY: int = 3
    OPENAI_MAX_RETRIES: int = 4
    OPENAI_TIMEOUT_SECONDS: float = 60.0
    OPENAI_ANSWER_MAX_TOKENS: int = 600
    OPENAI_AUX_MAX_TOKENS: int = 200
    PROMPT_MAX_EVIDENCE_CHARS_PER_CHUNK: int = 2500
    PROMPT_MAX_EVIDENCE_CHARS: int = 10000
    ANSWER_TOP_K: int = 5
    RAG_MAX_RETRIEVAL_ATTEMPTS: int = 2
    RAG_MAX_GENERATION_ATTEMPTS: int = 2

    JWT_SECRET_KEY: str = "change_me_in_production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440
    AUTH_COOKIE_NAME: str = "rag_session"
    OAUTH_STATE_COOKIE_NAME: str = "rag_oauth_state"
    AUTH_COOKIE_SECURE: bool = False
    FRONTEND_URL: str = "http://localhost:3000"

    GOOGLE_CLIENT_ID: str | None = None
    GOOGLE_CLIENT_SECRET: str | None = None
    GOOGLE_REDIRECT_URI: str | None = None

    MICROSOFT_CLIENT_ID: str | None = None
    MICROSOFT_CLIENT_SECRET: str | None = None
    MICROSOFT_REDIRECT_URI: str | None = None

    MAX_UPLOAD_SIZE_MB: int = 250

    JOB_EXECUTION_MODE: str = "inline"
    JOB_POLL_INTERVAL_SECONDS: float = 1.0
    JOB_STALE_AFTER_MINUTES: int = 30
    API_MAX_CONCURRENT_RAG_REQUESTS: int = 20
    ASYNC_RAG_WORKFLOW: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
