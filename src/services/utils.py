"""Utility functions for batch processing, async file streaming, and template rendering.

This module provides utilities for common operations in the data processing pipeline:
- Batching iterables into fixed-size chunks for efficient processing
- Streaming large JSONL files line-by-line without loading into memory
- Asynchronous Jinja2 template rendering

Key Features:
    - Memory-efficient async file streaming with configurable batch sizes
    - Support for processing file ranges (start_line to end_line)
    - Flexible error handling (strict vs. lenient JSON parsing)
    - Generic batch creation for any iterable type

Typical Use Cases:
    - Processing large JSONL files in manageable batches
    - Streaming document chunks for embedding generation
    - Rendering prompts using Jinja2 templates (asynchronously)
"""

import itertools
import json
from pathlib import Path
from typing import Any
from typing import AsyncIterator
from typing import Iterable
from typing import Iterator
from typing import TypeVar

import aiofiles
import jinja2

from src.settings import logger

T = TypeVar("T")


def create_batches(iterable: Iterable[T], batch_size: int = 20) -> Iterator[tuple[T, ...]]:
    """Break an iterable into fixed-size chunks.

    Args:
        iterable: The iterable to batch.
        batch_size: Size of each batch. Must be >= 1. Defaults to 20.

    Returns:
        Iterator yielding tuples of items from the iterable.

    Raises:
        ValueError: If batch_size is less than 1.
        TypeError: If batch_size is not an integer.

    Examples:
        >>> list(create_batches([1, 2, 3, 4, 5], 2))
        [(1, 2), (3, 4), (5,)]
        >>> list(create_batches([], 3))
        []
        >>> list(create_batches(range(10), 3))
        [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9,)]
    """
    if not isinstance(batch_size, int):
        raise TypeError(f"batch_size must be an integer, got {type(batch_size).__name__}")

    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")

    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))


async def read_chunks_in_batches(
    file_path: str | Path,
    batch_size: int = 10,
    strict: bool = False,
    start_line: int | None = None,
    end_line: int | None = None,
) -> AsyncIterator[list[dict[str, Any]]]:
    """Read JSONL file and yield batches of chunk objects in a streaming manner.

    This function processes the file line-by-line without loading the entire
    file into memory, making it suitable for large files.

    Args:
        file_path: Path to the JSONL file containing chunk data.
        batch_size: Number of chunks per batch. Must be >= 1. Defaults to 10.
        strict: If True, raise exception on JSON parse errors. If False, log
            and skip invalid lines. Defaults to False.
        start_line: Starting line number (1-indexed, inclusive). If None, starts from
            beginning. Defaults to None.
        end_line: Ending line number (1-indexed, inclusive). If None, processes until
            end of file. Defaults to None.

    Yields:
        Lists of chunk dictionaries with keys: section_header, section_text,
        header_level, metadata.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If batch_size is less than 1, or if start_line > end_line.
        json.JSONDecodeError: If strict=True and a line contains invalid JSON.

    Examples:
        >>> async for batch in read_chunks_in_batches("data/chunks.jsonl"):
        ...     process_batch(batch)
        >>> async for batch in read_chunks_in_batches(
        ...     "data/chunks.jsonl", start_line=1, end_line=100
        ... ):
        ...     # Process only first 100 lines
        ...     process_batch(batch)
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")

    if start_line is not None and start_line < 1:
        raise ValueError(f"start_line must be at least 1, got {start_line}")

    if end_line is not None and end_line < 1:
        raise ValueError(f"end_line must be at least 1, got {end_line}")

    if start_line is not None and end_line is not None and start_line > end_line:
        raise ValueError(f"start_line ({start_line}) must be <= end_line ({end_line})")

    current_batch: list[dict[str, Any]] = []
    line_number = 0

    # Open file once and stream line-by-line (async I/O for non-blocking reads)
    async with aiofiles.open(file_path, mode="r", encoding="utf-8") as f:
        async for line in f:
            line_number += 1

            # Skip lines before start_line (allows processing file ranges)
            if start_line is not None and line_number < start_line:
                continue

            # Stop processing after end_line (useful for distributed processing)
            if end_line is not None and line_number > end_line:
                break

            line = line.strip()

            # Skip empty lines (common in hand-edited JSONL files)
            if not line:
                continue

            try:
                # Parse JSON and add to current batch
                chunk = json.loads(line)
                current_batch.append(chunk)

                # Yield complete batch when it reaches the desired size
                # This enables streaming processing without loading entire file
                if len(current_batch) >= batch_size:
                    yield current_batch
                    current_batch = []

            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse JSON at line {line_number} in {file_path}: {e}"

                # In strict mode, fail fast on invalid JSON
                # In lenient mode, log and continue (useful for partially corrupted files)
                if strict:
                    raise json.JSONDecodeError(error_msg, e.doc, e.pos) from e

                logger.error(error_msg)
                continue

    # Yield any remaining chunks in the final partial batch
    # Last batch may be smaller than batch_size
    if current_batch:
        yield current_batch


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
