"""Generate training triplets from document chunks using GenAI with schema-based generation.

This module handles the core triplet generation logic, transforming document chunks
into anchor-positive-negative triplets suitable for training embedding models.

Triplet Structure:
    - anchor: A query or question that relates to the content
    - positive: The original document chunk (relevant to the anchor)
    - negative: Generated text that is semantically different but plausibly related

Generation Process:
    1. For each chunk, render a Jinja2 template with the chunk text
    2. Call GenAI API with structured output schema (AnchorNegativePair list)
    3. Parse the JSON response into AnchorNegativePair objects
    4. Convert pairs to full Triplet objects by adding original chunk as positive
    5. Process multiple chunks concurrently using asyncio.gather

Key Features:
    - Concurrent batch processing for efficiency
    - Schema-based generation ensures valid JSON output
    - Automatic error handling and logging
    - Configurable templates via environment variables
    - Returns empty list on failure (fail gracefully)

Environment Variables:
    GENERATE_TRIPLETS_TEMPLATE: Template file name for anchor-negative generation
        (default: "generate_anchor_negative.md")

Example:
    >>> chunks = [{"section_text": "Python is a programming language..."}]
    >>> triplets = await generate_triplets_from_chunks(chunks)
    >>> print(triplets[0].anchor)
    'What is Python?'
"""
import asyncio
import os
from typing import Any
from typing import Optional

import jinja2

from src.services.gemini import generate_content_async
from src.settings import client as default_client
from src.settings import config
from src.settings import jinja2_env_async
from src.settings import logger
from src.triplet_generation.models import AnchorNegativePair
from src.triplet_generation.models import Triplet


async def _generate_triplets_from_chunk(
    chunk: dict[str, Any],
    template: Optional[jinja2.Template] = None,
    client: Optional[Any] = None,
) -> list[Triplet]:
    """Generate triplets from a single chunk object.

    Args:
        chunk: A dictionary with keys: section_header, section_text,
            header_level, metadata.
        client: Optional language model client. If None, uses default from settings.
        template: Optional Jinja2 template. If None, loads default template.

    Returns:
        List of Triplet objects generated from the chunk.
    """

    if not client:
        client = default_client

    try:
        # Load template if not provided (allows reuse across batches)
        if not template:
            template = await jinja2_env_async.get_template(
                os.getenv("GENERATE_TRIPLETS_TEMPLATE", "generate_anchor_negative.md")
            )

        # Extract section text from chunk and render template
        section_text = chunk.get("section_text", "")
        contents = await template.render_async(text=section_text)

        # Generate anchor-negative pairs from GenAI using structured output
        # Schema-based generation ensures the response is valid JSON matching AnchorNegativePair
        response = await generate_content_async(
            contents=contents,
            model=config.GENERATION_MODEL,
            client=client,
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": list[AnchorNegativePair],
            },
        )

        # Convert anchor-negative pairs to full triplets
        # The original section_text becomes the "positive" example
        pairs = response.parsed or []
        triplets = [
            Triplet(
                anchor=pair.anchor,
                positive=section_text,
                negative=pair.negative,
            )
            for pair in pairs
        ]

    except jinja2.TemplateNotFound as e:
        logger.error(f"Template not found: {e}")
        raise e

    except Exception as e:
        logger.error(f"Failed to generate triplets: {e}", exc_info=True)
        return []

    return triplets


async def generate_triplets_from_chunks(
    chunks: list[dict[str, Any]],
    template: Optional[jinja2.Template] = None,
    client: Optional[Any] = None,
) -> list[Triplet]:
    """Generate triplets from a list of chunk objects using concurrent processing.

    This function processes multiple chunks in parallel, maximizing throughput
    when calling the GenAI API. Each chunk is processed independently, and
    failures are handled gracefully (logged but don't stop other chunks).

    Args:
        chunks: A list of dictionaries, each with keys: section_header, section_text,
            header_level, metadata.
        template: Optional pre-loaded Jinja2 template. If None, loads from environment.
            Passing a template avoids repeated loads when processing batches.
        client: Optional GenAI client. If None, uses default from settings.

    Returns:
        List of Triplet objects generated from all chunks. If a chunk fails,
        it returns an empty list for that chunk (graceful degradation).

    Example:
        >>> chunks = [
        ...     {"section_text": "Python is..."},
        ...     {"section_text": "JavaScript is..."}
        ... ]
        >>> triplets = await generate_triplets_from_chunks(chunks)
        >>> len(triplets)
        6  # e.g., 3 triplets per chunk
    """

    if not client:
        client = default_client

    # Load template once for all chunks (efficiency)
    if not template:
        template = await jinja2_env_async.get_template(
            os.getenv("GENERATE_TRIPLETS_TEMPLATE", "generate_anchor_negative.md")
        )

    # Create tasks for concurrent processing
    # Each chunk is processed independently in parallel
    tasks = [
        _generate_triplets_from_chunk(chunk, template=template, client=client)
        for chunk in chunks
    ]

    # Execute all tasks concurrently and wait for completion
    # return_exceptions=True prevents one failure from canceling others
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Flatten list of lists and filter out exceptions
    # sum() concatenates lists: sum([[1,2], [3,4]], []) -> [1,2,3,4]
    triplets = sum(
        (result for result in results if isinstance(result, list)),
        [],
    )

    return triplets
