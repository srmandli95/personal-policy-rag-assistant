from functools import lru_cache

from app.config.settings import settings
from app.embeddings.local_embedder import LocalEmbeddingService
from app.utils.logger import get_logger


logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embedding_service() -> LocalEmbeddingService:
    """Return the configured embedding service implementation."""
    provider = settings.EMBEDDING_PROVIDER.lower().strip()
    logger.debug("Resolving embedding service: provider=%s", provider)

    if provider == "local":
        return LocalEmbeddingService(settings.EMBEDDING_MODEL_NAME)

    logger.error("Unsupported embedding provider requested: provider=%s", settings.EMBEDDING_PROVIDER)
    raise ValueError(f"Unsupported embedding provider: {settings.EMBEDDING_PROVIDER}")
