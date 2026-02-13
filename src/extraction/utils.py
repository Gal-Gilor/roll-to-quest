"""Extract structured D&D 5E entities from document chunks using GenAI.

This module handles the core entity extraction logic, transforming SRD document
chunks into structured entity dictionaries using 5 focused Pydantic models that
run in parallel per chunk.

Extraction Process:
    1. Format each chunk's header + text into Markdown
    2. Render the extraction prompt template with the formatted text
    3. Run all 5 focused extraction models concurrently per chunk
    4. Merge results into a single flat dict with non-overlapping keys
    5. Filter out empty entity lists

Key Features:
    - 5 focused models run in parallel per chunk (up to 50 API calls per batch)
    - Graceful degradation: individual model/chunk failures don't stop the batch
    - Schema-based generation ensures valid JSON output
    - Configurable thinking budget for Gemini's reasoning

Focused Extraction Models:
    - GameplayEntities: spells, monsters, classes, races, subclasses
    - EquipmentEntities: weapons, armor, magic items, tools, vehicles, items
    - CharacterEntities: backgrounds, feats, features, actions
    - DescriptorEntities: conditions, skills, damage types, schools, sizes, creature types
    - WorldEntities: planes, deities, languages, environments, senses, movement types
"""

import asyncio
import json
from typing import Any
from typing import Callable
from typing import Optional

import jinja2
from aiolimiter import AsyncLimiter
from google import genai
from google.genai.errors import APIError
from pydantic import BaseModel

from src.extraction.models import CharacterEntities
from src.extraction.models import DescriptorEntities
from src.extraction.models import EquipmentEntities
from src.extraction.models import GameplayEntities
from src.extraction.models import WorldEntities
from src.services.gemini import gemini_async_retry
from src.settings import logger

EXTRACTION_MODELS: list[type[BaseModel]] = [
    GameplayEntities,
    EquipmentEntities,
    CharacterEntities,
    DescriptorEntities,
    WorldEntities,
]


def _format_chunk_text(chunk: dict[str, Any]) -> str:
    """Format a chunk's header and text into Markdown.

    Combines the section header with its text content, prefixing the header
    with the appropriate number of '#' characters based on header_level.

    Args:
        chunk: A dictionary with keys: section_header, section_text,
            header_level, metadata.

    Returns:
        Formatted Markdown string with header and text.

    Examples:
        >>> _format_chunk_text(
        ...     {
        ...         "header_level": 2,
        ...         "section_header": "Fireball",
        ...         "section_text": "A bright streak...",
        ...     }
        ... )
        '## Fireball\\nA bright streak...'
    """
    header = chunk.get("section_header", "")
    text = chunk.get("section_text", "")
    header_level = chunk.get("header_level")

    if header_level is not None and header:
        return f"{'#' * header_level} {header}\n{text}"

    return text


async def _extract_with_model(
    client: genai.Client,
    model_id: str,
    contents: str,
    extraction_model: type[BaseModel],
    thinking_budget: int,
) -> tuple[dict[str, list], dict[str, int]]:
    """Extract entities from content using a single focused Pydantic model.

    Calls Gemini's structured output API with the model's JSON schema and
    validates the response with Pydantic. APIErrors are re-raised so that
    upstream retry logic can handle transient failures (429, 500, 503).

    Args:
        client: Google GenAI client instance.
        model_id: Gemini model identifier (e.g., 'gemini-2.5-flash').
        contents: Rendered prompt with the chunk text.
        extraction_model: One of the 5 focused Pydantic extraction models.
        thinking_budget: Token budget for Gemini's thinking/reasoning.

    Returns:
        Tuple of (entities_dict, usage_dict). entities_dict maps entity names
        to lists. usage_dict has 'input_tokens' and 'output_tokens' counts.
        Returns ({}, zero usage) on non-API failure (graceful degradation).

    Raises:
        APIError: Re-raised to allow upstream retry logic to handle transient
            API errors.
    """
    empty_usage = {"input_tokens": 0, "output_tokens": 0}
    try:
        response = await client.aio.models.generate_content(
            model=model_id,
            contents=contents,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": extraction_model.model_json_schema(),
                "thinking_config": {"thinking_budget": thinking_budget},
            },
        )

        usage = dict(empty_usage)
        if response.usage_metadata:
            usage["input_tokens"] = response.usage_metadata.prompt_token_count or 0
            usage["output_tokens"] = (
                (response.usage_metadata.candidates_token_count or 0)
                + (response.usage_metadata.thoughts_token_count or 0)
            )

        data = json.loads(response.text)
        model_instance = extraction_model.model_validate(data)

        return model_instance.model_dump(exclude_none=True), usage

    except APIError:
        raise

    except Exception as e:
        logger.error(
            f"Extraction failed with {extraction_model.__name__}: {e}",
            exc_info=True,
        )
        return {}, dict(empty_usage)


async def _extract_chunk(
    client: genai.Client,
    model_id: str,
    contents: str,
    thinking_budget: int,
    limiter: AsyncLimiter,
) -> tuple[dict[str, list], dict[str, int]]:
    """Extract entities from a single chunk using all 5 focused models in parallel.

    Runs all extraction models concurrently via asyncio.gather, then merges
    results into a single flat dict. Models have non-overlapping keys, so
    merging is safe. Each API call is gated by the rate limiter.

    Args:
        client: Google GenAI client instance.
        model_id: Gemini model identifier.
        contents: Rendered prompt with the chunk text.
        thinking_budget: Token budget for Gemini's thinking/reasoning.
        limiter: Async rate limiter to throttle API calls.

    Returns:
        Tuple of (merged_entities, chunk_usage). merged_entities is a flat
        dict of entity lists with empty lists filtered out. chunk_usage
        sums input/output tokens across all 5 models.
    """

    async def _limited_extract(
        model: type[BaseModel],
    ) -> tuple[dict[str, list], dict[str, int]]:
        retry = gemini_async_retry()

        async def _attempt():
            async with limiter:
                return await _extract_with_model(
                    client, model_id, contents, model, thinking_budget
                )

        try:
            return await retry(_attempt)()
        except Exception as e:
            logger.error(
                f"Extraction failed for {model.__name__} after retries: {e}"
            )
            return {}, {"input_tokens": 0, "output_tokens": 0}

    tasks = [_limited_extract(model) for model in EXTRACTION_MODELS]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    merged: dict[str, list] = {}
    chunk_usage = {"input_tokens": 0, "output_tokens": 0}
    for result in results:
        if isinstance(result, tuple):
            entities, usage = result
            for key, value in entities.items():
                if isinstance(value, list) and len(value) > 0:
                    merged[key] = value
            chunk_usage["input_tokens"] += usage["input_tokens"]
            chunk_usage["output_tokens"] += usage["output_tokens"]

    return merged, chunk_usage


async def extract_entities_from_chunks(
    chunks: list[dict[str, Any]],
    client: genai.Client,
    template: jinja2.Template,
    model_id: str,
    thinking_budget: int,
    template_var: Optional[str] = None,
    limiter: Optional[AsyncLimiter] = None,
    on_chunk_done: Optional[Callable[[dict[str, int]], None]] = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Extract entities from multiple chunks concurrently.

    All chunks are processed concurrently with the rate limiter as the sole
    concurrency control. An optional callback fires after each chunk completes,
    enabling real-time progress tracking.

    Args:
        chunks: List of chunk dictionaries with keys: section_header,
            section_text, header_level, metadata.
        client: Google GenAI client instance.
        template: Pre-loaded Jinja2 template for the extraction prompt.
        model_id: Gemini model identifier.
        thinking_budget: Token budget for Gemini's thinking/reasoning.
        template_var: Optional template variable name. If None, uses 'text'.
        limiter: Optional async rate limiter to throttle API calls.
        on_chunk_done: Optional callback invoked after each chunk completes,
            receiving the chunk's usage dict for progress tracking.

    Returns:
        Tuple of (results, total_usage). results is a list of dicts with
        'section_header' and 'entities' keys (chunks with zero entities
        are omitted). total_usage sums input/output tokens across all chunks.
    """
    if limiter is None:
        limiter = AsyncLimiter(30, 60)

    var_name = template_var or "text"
    total_usage = {"input_tokens": 0, "output_tokens": 0}

    async def _process_chunk(
        chunk: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        formatted_text = _format_chunk_text(chunk)
        contents = await template.render_async(**{var_name: formatted_text})
        entities, usage = await _extract_chunk(
            client, model_id, contents, thinking_budget, limiter
        )

        total_usage["input_tokens"] += usage["input_tokens"]
        total_usage["output_tokens"] += usage["output_tokens"]

        if on_chunk_done:
            on_chunk_done(usage)

        if not entities:
            return None

        return {
            "section_header": chunk.get("section_header", ""),
            "entities": entities,
        }

    tasks = [_process_chunk(chunk) for chunk in chunks]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    filtered = [
        result
        for result in results
        if isinstance(result, dict) and result is not None
    ]
    return filtered, total_usage
