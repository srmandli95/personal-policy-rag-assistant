import importlib
import asyncio
from abc import ABC, abstractmethod
from functools import lru_cache
from threading import BoundedSemaphore
from typing import Any

from app.config.settings import settings
from app.utils.logger import get_logger

try:
    openai_module = importlib.import_module("openai")
    OpenAI: Any | None = openai_module.OpenAI
    AsyncOpenAI: Any | None = openai_module.AsyncOpenAI
    OpenAIRateLimitError: Any = openai_module.RateLimitError
except ImportError:
    OpenAI = None
    AsyncOpenAI = None
    OpenAIRateLimitError = ()


logger = get_logger(__name__)
_sync_provider_slots = BoundedSemaphore(max(1, settings.OPENAI_MAX_CONCURRENCY))
_async_provider_slots = asyncio.Semaphore(max(1, settings.OPENAI_MAX_CONCURRENCY))


class LLMRateLimitError(RuntimeError):
    """The configured LLM provider exhausted its request/token allowance."""


class LLMClient(ABC):
    """
    Base LLM client interface.

    Future providers like Gemini, local models, or Bedrock can implement
    this same generate() method.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate an answer from a prompt.
        """
        raise NotImplementedError

    async def agenerate(self, prompt: str) -> str:
        """Asynchronously generate text when a provider supports async I/O."""
        raise NotImplementedError


class OpenAILLMClient(LLMClient):
    """
    OpenAI LLM client using chat completions.
    """

    def __init__(self, api_key: str, model_name: str, max_output_tokens: int = 600):
        """Initialize the OpenAI client with API credentials and model name."""
        if not api_key or not api_key.strip():
            logger.error("OpenAI LLM client initialization failed: missing API key")
            raise ValueError(
                "OPENAI_API_KEY is required when LLM_PROVIDER is set to openai"
            )

        if not model_name or not model_name.strip():
            logger.error("OpenAI LLM client initialization failed: missing model name")
            raise ValueError("OPENAI_MODEL_NAME is required")

        if OpenAI is None:
            logger.error("OpenAI LLM client initialization failed: package not installed")
            raise ImportError(
                "openai package is not installed. Run: pipenv install openai"
            )

        self.model_name = model_name
        self.max_output_tokens = max(1, max_output_tokens)
        client_options = {
            "api_key": api_key,
            "max_retries": max(0, settings.OPENAI_MAX_RETRIES),
            "timeout": settings.OPENAI_TIMEOUT_SECONDS,
        }
        self.client = OpenAI(**client_options)
        self.async_client = AsyncOpenAI(**client_options)
        logger.info("OpenAI LLM client initialized: model_name=%s", model_name)

    def generate(self, prompt: str) -> str:
        """Generate a model response for the supplied prompt."""
        if not prompt or not prompt.strip():
            logger.warning("OpenAI generation rejected: empty prompt")
            raise ValueError("prompt is required")

        logger.debug(
            "OpenAI generation started: model_name=%s prompt_length=%s",
            self.model_name,
            len(prompt),
        )
        try:
            with _sync_provider_slots:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a grounded document assistant. "
                                "Answer only from the supplied evidence."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=self.max_output_tokens,
                )
        except OpenAIRateLimitError as exc:
            raise LLMRateLimitError("OpenAI rate limit exhausted") from exc

        answer = response.choices[0].message.content

        if not answer:
            logger.warning(
                "OpenAI generation returned an empty answer: model_name=%s",
                self.model_name,
            )
            return ""

        logger.debug(
            "OpenAI generation completed: model_name=%s answer_length=%s",
            self.model_name,
            len(answer),
        )
        return answer.strip()

    async def agenerate(self, prompt: str) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("prompt is required")
        try:
            async with _async_provider_slots:
                response = await self.async_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a grounded document assistant. Answer only from the supplied evidence."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=self.max_output_tokens,
                )
        except OpenAIRateLimitError as exc:
            raise LLMRateLimitError("OpenAI rate limit exhausted") from exc
        answer = response.choices[0].message.content
        return answer.strip() if answer else ""


@lru_cache(maxsize=4)
def get_llm_client(model_name: str | None = None, max_output_tokens: int | None = None) -> LLMClient:
    """
    Return the configured LLM client.

    Day 11 supports OpenAI first.
    Gemini/local can be added later without changing answer generation logic.
    """
    provider = getattr(settings, "LLM_PROVIDER", "openai").lower().strip()
    logger.debug("Resolving LLM client: provider=%s", provider)

    if provider == "openai":
        return OpenAILLMClient(
            api_key=getattr(settings, "OPENAI_API_KEY", ""),
            model_name=model_name or getattr(settings, "OPENAI_MODEL_NAME", "gpt-4o-mini"),
            max_output_tokens=max_output_tokens or settings.OPENAI_ANSWER_MAX_TOKENS,
        )

    logger.error("Unsupported LLM provider requested: provider=%s", provider)
    raise ValueError(
        f"Unsupported LLM_PROVIDER '{provider}'. Supported providers: openai"
    )


def get_aux_llm_client() -> LLMClient:
    return get_llm_client(settings.OPENAI_AUX_MODEL_NAME, settings.OPENAI_AUX_MAX_TOKENS)
