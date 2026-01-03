"""Generate anchor-positive pairs from document chunks using GenAI.

This module handles the core pair generation logic, transforming document chunks
into anchor-positive pairs suitable for training embedding models.

Pair Structure:
    - anchor: A query or question that relates to the content
    - positive: The original document chunk (relevant to the anchor)

Generation Process:
    1. For each chunk, render a Jinja2 template with the chunk text
    2. Call GenAI API with structured output schema (AnchorOnly list)
    3. Parse the JSON response into AnchorOnly objects
    4. Convert to AnchorPositivePair objects by adding original chunk as positive
    5. Process multiple chunks concurrently using asyncio.gather

Key Features:
    - Concurrent batch processing for efficiency
    - Schema-based generation ensures valid JSON output
    - Automatic error handling and logging
    - Configurable templates via environment variables
    - Returns empty list on failure (fail gracefully)

Environment Variables:
    GENERATE_PAIRS_TEMPLATE: Template file name for anchor-only generation
        (default: "generate_anchor_only.md")

Example:
    >>> chunks = [{"section_text": "Python is a programming language..."}]
    >>> pairs = await generate_pairs_from_chunks(chunks)
    >>> print(pairs[0].anchor)
    'What is Python?'
"""

import asyncio
import os
from typing import Any
from typing import Optional

import jinja2

from src.pair_generation.models import AnchorOnly
from src.pair_generation.models import AnchorPositivePair
from src.services.gemini import generate_content_async
from src.settings import client as default_client
from src.settings import config
from src.settings import jinja2_env_async
from src.settings import logger


async def _generate_pairs_from_chunk(
    chunk: dict[str, Any],
    template: Optional[jinja2.Template] = None,
    client: Optional[Any] = None,
) -> list[AnchorPositivePair]:
    """Generate anchor-positive pairs from a single chunk object.

    This function generates pairs without negatives, using an anchor-only template.

    Args:
        chunk: A dictionary with keys: section_header, section_text,
            header_level, metadata.
        client: Optional language model client. If None, uses default from settings.
        template: Optional Jinja2 template. If None, loads default template.

    Returns:
        List of AnchorPositivePair objects generated from the chunk.
        Returns empty list on failure (graceful degradation).

    Example:
        >>> chunk = {"section_text": "Python is a programming language..."}
        >>> pairs = await _generate_pairs_from_chunk(chunk)
        >>> len(pairs)
        3  # e.g., 3 anchor-positive pairs
    """
    if not client:
        client = default_client

    try:
        # Load template if not provided (allows reuse across batches)
        if not template:
            template = await jinja2_env_async.get_template(
                os.getenv("GENERATE_PAIRS_TEMPLATE", "generate_anchor_only.md")
            )

        # Extract section text from chunk and render template
        section_text = chunk.get("section_text", "")

        # Skip empty or too-short chunks (code-level filter)
        if not section_text or len(section_text.strip()) < 50:
            logger.debug(
                f"Skipping chunk - too short ({len(section_text.strip())} chars): "
                f"{section_text[:50] if section_text else '(empty)'}..."
            )
            return []

        contents = await template.render_async(text=section_text)

        # Generate anchor-only objects from GenAI using structured output
        # Schema-based generation ensures the response is valid JSON matching AnchorOnly
        response = await generate_content_async(
            contents=contents,
            model=config.GENERATION_MODEL,
            client=client,
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": list[AnchorOnly],
            },
        )

        # Convert anchor-only objects to full anchor-positive pairs
        # The original section_text becomes the "positive" example
        anchors = response.parsed or []
        pairs = [
            AnchorPositivePair(
                anchor=anchor_obj.anchor,
                positive=section_text,
            )
            for anchor_obj in anchors
        ]

    except jinja2.TemplateNotFound as e:
        logger.error(f"Template not found: {e}")
        raise e

    except Exception as e:
        logger.error(f"Failed to generate pairs: {e}", exc_info=True)
        return []

    return pairs


async def generate_pairs_from_chunks(
    chunks: list[dict[str, Any]],
    template: Optional[jinja2.Template] = None,
    client: Optional[Any] = None,
) -> list[AnchorPositivePair]:
    """Generate anchor-positive pairs from chunk objects using concurrent processing.

    This function processes multiple chunks in parallel for anchor-positive generation
    without creating negatives, reducing API costs by ~40-50% compared to triplet generation.

    Args:
        chunks: A list of dictionaries, each with keys: section_header, section_text,
            header_level, metadata.
        template: Optional pre-loaded Jinja2 template. If None, loads from environment.
            Passing a template avoids repeated loads when processing batches.
        client: Optional GenAI client. If None, uses default from settings.

    Returns:
        List of AnchorPositivePair objects generated from all chunks. If a chunk fails,
        it returns an empty list for that chunk (graceful degradation).

    Example:
        >>> chunks = [{"section_text": "Python is..."}, {"section_text": "JavaScript is..."}]
        >>> pairs = await generate_pairs_from_chunks(chunks)
        >>> len(pairs)
        10  # e.g., 5 pairs per chunk
    """
    if not client:
        client = default_client

    # Load template once for all chunks (efficiency)
    if not template:
        template = await jinja2_env_async.get_template(
            os.getenv("GENERATE_PAIRS_TEMPLATE", "generate_anchor_only.md")
        )

    # Create tasks for concurrent processing
    # Each chunk is processed independently in parallel
    tasks = [
        _generate_pairs_from_chunk(chunk, template=template, client=client)
        for chunk in chunks
    ]

    # Execute all tasks concurrently and wait for completion
    # return_exceptions=True prevents one failure from canceling others
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Flatten list of lists and filter out exceptions
    # sum() concatenates lists: sum([[1,2], [3,4]], []) -> [1,2,3,4]
    pairs = sum(
        (result for result in results if isinstance(result, list)),
        [],
    )

    return pairs
