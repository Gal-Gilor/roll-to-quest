import functools
import random
import time
from typing import Any
from typing import Callable
from typing import Optional
from typing import TypeVar
from typing import Union

from google import genai
from google.genai.errors import APIError

from src.settings import client as default_client
from src.settings import config
from src.settings import logger

F = TypeVar("F", bound=Callable[..., Any])


def retry_transient_errors(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    use_jitter: bool = True,
) -> Callable[[F], F]:
    """Decorator to retry functions on transient API errors with exponential backoff.

    Implements exponential backoff with optional jitter for API calls that may fail
    with transient errors. Only retries on specific HTTP status codes (500, 503, 429).
    All other errors are raised immediately.

    Args:
        max_retries: Maximum number of retry attempts. Defaults to 3.
        initial_delay: Initial delay in seconds before first retry. Defaults to 1.0.
        backoff_factor: Multiplicative factor for exponential delay growth. Defaults to 2.0.
        use_jitter: Whether to add random jitter (±50%) to delays. Defaults to True.

    Returns:
        Callable[[F], F]: Decorator that preserves the original function signature.

    Raises:
        APIError: Re-raised if max retries exceeded or if status code is not retryable.

    Examples:
        >>> @retry_transient_errors(max_retries=5, initial_delay=2.0)
        ... def call_gemini():
        ...     return client.generate_content("Hello")
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = initial_delay
            last_exception: Optional[APIError] = None

            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except APIError as e:
                    last_exception = e
                    is_retryable = e.status_code in (500, 503, 429)
                    has_retries_left = attempt < max_retries

                    if is_retryable and has_retries_left:
                        wait_time = (
                            delay * (0.5 + random.random() / 2) if use_jitter else delay
                        )
                        logger.warning(
                            f"Transient error (status {e.status_code}) on attempt "
                            f"{attempt}/{max_retries}. Retrying in {wait_time:.2f}s..."
                        )
                        time.sleep(wait_time)
                        delay *= backoff_factor
                    else:
                        raise

            # This line is reached only if all retries exhausted without success
            if last_exception:
                raise last_exception

        return wrapper  # type: ignore[return-value]

    return decorator


@retry_transient_errors(max_retries=5)
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


@retry_transient_errors(max_retries=5)
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
