from typing import List
from threading import BoundedSemaphore

from sentence_transformers import SentenceTransformer

from app.utils.logger import get_logger
from app.config.settings import settings


logger = get_logger(__name__)
_inference_slots = BoundedSemaphore(max(1, settings.INFERENCE_MAX_CONCURRENCY))


class LocalEmbeddingService:
    """SentenceTransformer-backed embedding service."""

    def __init__(self, model_name: str) -> None:
        """Load the local embedding model by name."""
        self.model_name = model_name
        logger.info("Loading local embedding model: model_name=%s", model_name)
        self.model = SentenceTransformer(model_name)
        logger.info("Local embedding model loaded: model_name=%s", model_name)

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string into a vector."""
        safe_text = text if text and text.strip() else " "
        logger.debug(
            "Embedding single text: model_name=%s text_length=%s",
            self.model_name,
            len(safe_text),
        )

        with _inference_slots:
            embedding = self.model.encode(
                safe_text,
                normalize_embeddings=True,
            )

        return embedding.astype(float).tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple text strings into vectors."""
        safe_texts = [
            text if text and text.strip() else " "
            for text in texts
        ]
        logger.debug("Embedding text batch: model_name=%s count=%s", self.model_name, len(safe_texts))

        with _inference_slots:
            embeddings = self.model.encode(
                safe_texts,
                normalize_embeddings=True,
            )

        return [
            embedding.astype(float).tolist()
            for embedding in embeddings
        ]
