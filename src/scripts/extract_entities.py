"""Extract structured D&D 5E entities from SRD document chunks.

This script processes document chunks from a JSONL file and extracts structured
entities (spells, monsters, classes, items, etc.) using Gemini with 5 focused
extraction models running in parallel.

Each chunk is processed by all 5 models concurrently, and results are merged
into a single entity dictionary per chunk. All chunks are processed through a
single rate limiter — no artificial batch boundaries.

Workflow:
    1. Read all chunks from JSONL file into memory
    2. Process all chunks concurrently (rate limiter controls throughput)
    3. For each chunk, run 5 focused extraction models in parallel via Gemini
    4. Merge and filter entity results (remove empty lists)
    5. Write results to output JSONL file

Dependencies:
    - Google Gemini credentials (configured in settings)
    - Jinja2 templates: system prompt + 5 per-model extraction prompts
    - Input chunks in JSONL format

Example Usage:
    # Process a small range for testing
    python -m src.scripts.extract_entities SRD_CC_v5.2.1_extracted_chunks.jsonl \\
        --start-line 1 --end-line 20

    # Full run with default settings
    python -m src.scripts.extract_entities SRD_CC_v5.2.1_extracted_chunks.jsonl

    # Custom rate limit and thinking budget
    python -m src.scripts.extract_entities SRD_CC_v5.2.1_extracted_chunks.jsonl \\
        --max-rate 60 --thinking-budget 1000

Output Format:
    JSONL file where each line contains a JSON object with:
    - section_header: str (original chunk header)
    - entities: dict (extracted entities with non-empty lists only)
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from time import time

import aiofiles
from aiolimiter import AsyncLimiter
from tqdm import tqdm

from src.extraction.utils import MODEL_TEMPLATES
from src.extraction.utils import extract_entities_from_chunks
from src.services.utils import read_chunks_in_batches
from src.settings import client
from src.settings import config
from src.settings import jinja2_env_async
from src.settings import logger

INPUT_COST_PER_M = 0.30
OUTPUT_COST_PER_M = 2.50


def _estimate_cost(usage: dict[str, int]) -> float:
    """Estimate the dollar cost based on token usage.

    Uses Gemini Flash 2.5 pricing: $0.30/M input, $2.50/M output.

    Args:
        usage: Dict with 'input_tokens' and 'output_tokens' counts.

    Returns:
        Estimated cost in USD.
    """
    return (
        usage["input_tokens"] * INPUT_COST_PER_M + usage["output_tokens"] * OUTPUT_COST_PER_M
    ) / 1_000_000


async def main(
    input_file: str,
    start_line: int | None = None,
    end_line: int | None = None,
    thinking_budget: int = 600,
    model_id: str | None = None,
    max_rate: int = 30,
) -> str:
    """Extract entities from chunks in the input file.

    Reads all chunks into memory, then processes them concurrently with the
    rate limiter as the sole throughput control. Progress updates per chunk.

    Output File Naming:
        - Full file: src/extraction/data/entities.jsonl
        - Partial range: src/extraction/data/entities_lines_{start}_to_{end}.jsonl

    Args:
        input_file: Name of the JSONL file in src/extraction/data/ containing
            document chunks.
        start_line: Starting line number (1-indexed, inclusive). None processes
            from the beginning.
        end_line: Ending line number (1-indexed, inclusive). None processes
            until end of file.
        thinking_budget: Token budget for Gemini's thinking/reasoning.
            Defaults to 600.
        model_id: Gemini model identifier. If None, uses config.GENERATION_MODEL.
        max_rate: Maximum API calls per 60-second window. Defaults to 30.

    Returns:
        str: Success message with total processing time in seconds.

    Raises:
        SystemExit: If input file is not found.
    """
    generation_model = model_id or config.GENERATION_MODEL
    data_dir = Path(__file__).parent.parent / "extraction" / "data"
    file_path = data_dir / input_file

    if start_line is not None or end_line is not None:
        range_suffix = f"_lines_{start_line or 1}_to_{end_line or 'end'}"
        output_path = data_dir / f"entities{range_suffix}.jsonl"
    else:
        output_path = data_dir / "entities.jsonl"

    if not file_path.exists():
        logger.error(f"Input file not found: {file_path}")
        sys.exit(1)

    log_msg = f"Reading chunks from: {file_path}"
    if start_line is not None or end_line is not None:
        log_msg += f" (lines {start_line or 1} to {end_line or 'end'})"
    logger.info(log_msg)

    # Read all chunks into memory (JSONL parsing is fast; API calls are the bottleneck)
    chunks = []
    async for batch in read_chunks_in_batches(
        file_path, batch_size=10_000, start_line=start_line, end_line=end_line
    ):
        chunks.extend(batch)

    logger.info(
        f"Loaded {len(chunks)} chunks | model: {generation_model}, "
        f"max_rate: {max_rate}/min, thinking_budget: {thinking_budget}"
    )

    system_instruction = await jinja2_env_async.get_template(
        "extract_system.md"
    ).render_async()
    model_templates = {
        model: jinja2_env_async.get_template(template_name)
        for model, template_name in MODEL_TEMPLATES.items()
    }
    limiter = AsyncLimiter(max_rate, 60)

    process_start_time = time()
    total_usage = {"input_tokens": 0, "output_tokens": 0}

    try:
        with tqdm(
            total=len(chunks),
            desc="Extracting entities",
            unit="chunk",
        ) as pbar:

            def _on_chunk_done(usage: dict[str, int]) -> None:
                total_usage["input_tokens"] += usage["input_tokens"]
                total_usage["output_tokens"] += usage["output_tokens"]
                pbar.set_postfix(cost=f"${_estimate_cost(total_usage):.4f}")
                pbar.update(1)

            results, _ = await extract_entities_from_chunks(
                chunks,
                client=client,
                system_instruction=system_instruction,
                model_templates=model_templates,
                model_id=generation_model,
                thinking_budget=thinking_budget,
                limiter=limiter,
                on_chunk_done=_on_chunk_done,
            )

        async with aiofiles.open(output_path, mode="w", encoding="utf-8") as f:
            for result in results:
                await f.write(json.dumps(result) + "\n")

        total_chunks_with_entities = len(results)
        logger.info(
            f"Total: {len(chunks)} chunks processed, "
            f"{total_chunks_with_entities} with entities"
        )
        logger.info(f"Entities saved to: {output_path}")
        process_end_time = time()

        return f"Processing completed in {process_end_time - process_start_time:.2f} seconds."

    except (IOError, OSError, PermissionError) as e:
        logger.error(f"File operation failed: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in input file: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during processing: {e}", exc_info=True)
        sys.exit(1)
    finally:
        estimated_cost = _estimate_cost(total_usage)
        logger.info(
            f"Total tokens — input: {total_usage['input_tokens']:,}, "
            f"output: {total_usage['output_tokens']:,}"
        )
        logger.info(f"Estimated cost: ${estimated_cost:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Extract structured D&D 5E entities from chunked SRD documents.")
    )
    parser.add_argument(
        "filename",
        type=str,
        help=(
            "Name of the JSONL file in src/extraction/data/ "
            "(e.g., SRD_CC_v5.2.1_extracted_chunks.jsonl)"
        ),
    )
    parser.add_argument(
        "--start-line",
        type=int,
        default=None,
        help="Starting line number (1-indexed, inclusive). Defaults to file start.",
    )
    parser.add_argument(
        "--end-line",
        type=int,
        default=None,
        help="Ending line number (1-indexed, inclusive). Defaults to file end.",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=0,
        help="Token budget for Gemini's thinking/reasoning (default: 0).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(f"Gemini model identifier (default: {config.GENERATION_MODEL})."),
    )
    parser.add_argument(
        "--max-rate",
        type=int,
        default=30,
        help="Max API calls per 60-second window for rate limiting (default: 30).",
    )

    args = parser.parse_args()

    if "/" in args.filename or "\\" in args.filename or ".." in args.filename:
        parser.error(
            f"Invalid filename: {args.filename}. "
            "Filename must not contain path separators or '..'."
        )

    if args.start_line is not None and args.start_line < 1:
        parser.error("--start-line must be at least 1")

    if args.end_line is not None and args.end_line < 1:
        parser.error("--end-line must be at least 1")

    if (
        args.start_line is not None
        and args.end_line is not None
        and args.start_line > args.end_line
    ):
        parser.error("--start-line must be <= --end-line")

    if args.thinking_budget < 0:
        parser.error("--thinking-budget must be non-negative")

    if args.max_rate < 1:
        parser.error("--max-rate must be at least 1")

    asyncio.run(
        main(
            args.filename,
            start_line=args.start_line,
            end_line=args.end_line,
            thinking_budget=args.thinking_budget,
            model_id=args.model,
            max_rate=args.max_rate,
        )
    )
