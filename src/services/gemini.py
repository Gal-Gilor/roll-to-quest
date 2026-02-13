"""Google GenAI client utilities with automatic retry and error handling.

This module provides functions for interacting with Google's `google-genai` SDK,
including content generation and embedding creation, with built-in retry logic
for handling transient API errors (e.g., rate limits, server errors).

Retry Strategy:
    Uses google.api_core.retry_async.AsyncRetry with exponential backoff and jitter.
    - Retryable status codes: 429 (Rate Limit), 500 (Server Error), 503 (Service Unavailable)
    - Non-retryable errors raised immediately (e.g., 400, 401)
    - Default: 120s timeout, 1s initial delay, 2x backoff, 60s max delay

Usage:
    >>> @gemini_async_retry(timeout=120.0)
    ... async def my_api_call():
    ...     return await client.models.generate_content(...)

    >>> response = await generate_content_async("What is Python?")
    >>> embeddings = await generate_embeddings_async("Hello world")
"""

from typing import Optional
from typing import Union

from google import genai
from google.api_core.retry_async import AsyncRetry
from google.genai.errors import APIError

from src.settings import client as default_client
from src.settings import config
from src.settings import logger

RETRYABLE_STATUS_CODES = (429, 500, 503)


def _is_retryable(exception: Exception) -> bool:
    """Return True if the exception is a transient API error worth retrying."""
    return isinstance(exception, APIError) and exception.status_code in RETRYABLE_STATUS_CODES


def gemini_async_retry(
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    timeout: float = 120.0,
) -> AsyncRetry:
    """Create an AsyncRetry decorator for transient API errors.

    Uses exponential backoff with jitter. Only retries on HTTP status codes
    429 (Rate Limit), 500 (Server Error), and 503 (Service Unavailable).

    Args:
        initial_delay: Initial delay in seconds before first retry. Defaults to 1.0.
        backoff_factor: Multiplicative factor for exponential delay growth. Defaults to 2.0.
        max_delay: Maximum delay between retries in seconds. Defaults to 60.0.
        timeout: Total timeout in seconds before giving up. Defaults to 120.0.

    Returns:
        AsyncRetry: Decorator that retries the wrapped async function.

    Raises:
        APIError: Re-raised if timeout exceeded or if status code is not retryable.

    Examples:
        >>> @gemini_async_retry(timeout=120.0)
        ... async def call_gemini():
        ...     return await client.generate_content("Hello")
    """
    return AsyncRetry(
        predicate=_is_retryable,
        initial=initial_delay,
        multiplier=backoff_factor,
        maximum=max_delay,
        timeout=timeout,
    )


@gemini_async_retry()
async def generate_embeddings_async(
    contents: str | list[str],
    model: Optional[str] = None,
    client: Optional[genai.Client] = None,
) -> Optional[genai.types.EmbedContentResponse]:
    """Generate embeddings for given content using Google GenAI.

    Args:
        contents: Text content(s) to embed. Can be a single string or list of strings.
        model: Optional embedding model name. If None, uses config.EMBEDDING_MODEL.
        client: Optional GenAI client. If None, uses default gemini_client.

    Returns:
        EmbedContentResponse or None: Generated embeddings for the content,
            or None if operation fails.

    Examples:
        >>> embeddings = await create_embeddings("Hello world")
        >>> embeddings = await create_embeddings(["Hello", "world"])
    """
    if not contents:
        logger.warning("No content provided for embedding generation")

        return None

    try:
        embedding_model = model or config.EMBEDDING_MODEL
        genai_client = client or default_client

        embeddings = await genai_client.aio.models.embed_content(
            model=embedding_model, contents=contents
        )

        return embeddings

    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}", exc_info=True)

        return None


@gemini_async_retry()
async def generate_content_async(
    contents: str | list[str],
    model: Optional[str] = None,
    client: Optional[genai.Client] = None,
    generation_config: Optional[Union[genai.types.GenerateContentResponse, dict]] = None,
) -> Optional[genai.types.GenerateContentResponse]:
    """Sends a generate content request to a `google-genai` client.

    Args:
        contents: Text prompt(s) to generate content from. Can be a single string or
            list of strings.
        model: Optional generation model name. If None, uses config.GENERATION_MODEL.
        client: Optional GenAI client. If None, uses default gemini_client.
        config: Optional dictionary containing generation parameters
            or a GenerateContentResponse object.

    Returns:
        GenerateContentResponse or None: Generated content response,
            or None if operation fails.

    Examples:
        >>> response = await generate_content("Hello world")
        >>> response = await generate_content(["Hello", "world"])
    """
    if not contents:
        logger.warning("No content provided for generation")

        return None

    try:
        generation_model = model or config.GENERATION_MODEL
        genai_client = client or default_client

        response = await genai_client.aio.models.generate_content(
            model=generation_model, contents=contents, config=generation_config
        )

        return response

    except Exception as e:
        logger.error(f"Failed to generate content: {e}", exc_info=True)

        return None
