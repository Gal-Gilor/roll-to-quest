import itertools
import json
from pathlib import Path
from typing import Any
from typing import AsyncIterator
from typing import Iterable
from typing import Iterator
from typing import Optional
from typing import TypeVar

import aiofiles
import jinja2

from src.settings import logger

T = TypeVar("T")


def create_batches(
    iterable: Iterable[T], batch_size: Optional[int] = 100
) -> Iterator[tuple[T, ...]]:
    """Break an iterable into fixed-size chunks.

    Args:
        iterable: The iterable to batch.
        batch_size: Size of each batch. Defaults to 100.

    Returns:
        Iterator yielding tuples of items from the iterable.

    Raises:
        ValueError: If batch_size is less than 1.

    Examples:
        >>> list(create_batches([1, 2, 3, 4, 5], 2))
        [(1, 2), (3, 4), (5,)]
        >>> list(create_batches([], 3))
        []
    """
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))


async def read_chunks_in_batches(
    file_path: str | Path, batch_size: int = 10
) -> AsyncIterator[list[dict[str, Any]]]:
    """Read JSONL file and yield batches of chunk objects.

    Args:
        file_path: Path to the JSONL file containing chunk data.
        batch_size: Number of chunks per batch. Defaults to 10.

    Yields:
        Lists of chunk dictionaries with keys: section_header, section_text,
        header_level, metadata.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        json.JSONDecodeError: If a line contains invalid JSON.

    Examples:
        >>> async for batch in read_chunks_in_batches("data/chunks.jsonl"):
        ...     process_batch(batch)
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    async def chunk_generator() -> AsyncIterator[dict[str, Any]]:
        """Generate individual chunks from the JSONL file."""
        async with aiofiles.open(file_path, mode="r", encoding="utf-8") as f:
            async for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)

                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON line: {e}")
                        continue

    chunks = []
    async for chunk in chunk_generator():
        chunks.append(chunk)

    for batch in create_batches(chunks, batch_size):
        yield list(batch)


async def render_template(env: jinja2.Environment, template_name: str, **context: Any) -> str:
    """Render a Jinja2 template asynchronously with the given context.

    Args:
        env: Jinja2 environment containing the templates.
        template_name: Name of the template file to render.
        **context: Keyword arguments to pass as context variables to the template.

    Returns:
        str: The rendered template as a string.

    Raises:
        jinja2.TemplateNotFound: If the specified template does not exist.
        jinja2.TemplateError: If template rendering fails.

    Example:
        >>> env = jinja2.Environment(loader=jinja2.FileSystemLoader("templates"))
        >>> rendered = await render_template(env, "greeting.html", name="Alice")
    """
    template = env.get_template(template_name)

    return await template.render_async(**context)
