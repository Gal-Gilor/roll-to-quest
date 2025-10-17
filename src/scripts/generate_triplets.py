"""Generate training triplets (anchor, positive, negative) from chunked documents.

This script processes document chunks from JSONL files and generates triplets for
training embedding models. Each triplet consists of:
- anchor: A query or question related to the content
- positive: The actual document chunk (relevant to the anchor)
- negative: Generated text that is semantically different from the positive

The triplets are generated using Gemini 2.5 (Flash by default).
The prompt templates for generating (anchor, negative) pairs,
and (anchor, positive, negative) triplets are tracked in "src/templates/" directory.
The projects uses Jinja2 for loading and rendering templates.

Workflow:
    1. Read chunked documents from JSONL file (created by chunk_documents.py)
    2. Process chunks in batches for efficiency
    3. For each chunk, use Gemini to generate anchor-negative pairs
    4. Combine pairs with original chunk text to form complete triplets
    5. Stream results to output JSONL file

Dependencies:
    - Google Gemini credentials (configured in settings)
    - Jinja2 template for triplet generation (default: generate_anchor_negative.md)
    - Input chunks in JSONL format from chunk_documents.py

Environment Variables:
    GENERATE_TRIPLETS_TEMPLATE: Name of the Jinja2 template to use for
        generating anchor-negative pairs. Defaults to "generate_anchor_negative.md".
        Generating the anchor and negative without the positive, which is the chunk itself,
        is more efficient and cost effective.

Example Usage:
    # Process entire file
    python -m src.scripts.generate_triplets document_chunks.jsonl

    # Process specific line range
    python -m src.scripts.generate_triplets document_chunks.jsonl \\
        --start-line 1 --end-line 100

    # With poetry
    poetry run python -m src.scripts.generate_triplets document_chunks.jsonl

Output Format:
    JSONL file where each line contains a JSON object with:
    - anchor: str
    - positive: str
    - negative: str
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from time import time

import aiofiles

from src.services.utils import read_chunks_in_batches
from src.settings import client
from src.settings import jinja2_env_async
from src.settings import logger
from src.triplet_generation.utils import generate_triplets_from_chunks


async def main(
    input_file: str, start_line: int | None = None, end_line: int | None = None
) -> str:
    """Generate triplets from chunks in the input file.

    This function orchestrates the entire triplet generation process:
    1. Locates the input JSONL file in the data/ directory
    2. Loads the Jinja2 template for anchor-negative generation
    3. Reads chunks in batches using async streaming
    4. For each batch, generates triplets concurrently via GenAI
    5. Writes triplets to output file immediately (streaming)
    6. Logs progress every 10 batches

    Output File Naming:
        - Full file: data/triplets.jsonl
        - Partial range: data/triplets_lines_{start}_to_{end}.jsonl

    Args:
        input_file: Name of the JSONL file in the data/ folder containing document chunks.
            Expected format: one JSON object per line with keys section_header,
            section_text, header_level, metadata.
        start_line: Starting line number (1-indexed, inclusive). None processes from
            the beginning of the file.
        end_line: Ending line number (1-indexed, inclusive). None processes until
            the end of file.

    Returns:
        str: Success message with total processing time in seconds.

    Raises:
        SystemExit: If input file is not found in the data/ directory.

    Example:
        >>> await main("document_chunks.jsonl")
        'Processing completed in 45.32 seconds.'
        >>> await main("document_chunks.jsonl", start_line=1, end_line=50)
        'Processing completed in 8.21 seconds.'
    """
    # Construct the full path to the input file in the data/ directory
    data_dir = Path(__file__).parent.parent.parent / "data"
    file_path = data_dir / input_file

    # Create output filename: add range suffix if processing partial file
    # This allows multiple partial runs without overwriting results
    if start_line is not None or end_line is not None:
        range_suffix = f"_lines_{start_line or 1}_to_{end_line or 'end'}"
        output_path = data_dir / f"triplets{range_suffix}.jsonl"
    else:
        output_path = data_dir / "triplets.jsonl"

    if not file_path.exists():
        logger.error(f"Input file not found: {file_path}")
        sys.exit(1)

    log_msg = f"Reading chunks from: {file_path}"
    if start_line is not None or end_line is not None:
        log_msg += f" (lines {start_line or 1} to {end_line or 'end'})"
    logger.info(log_msg)

    # Load the Jinja2 template once (reused for all batches)
    # Template defines the prompt structure for GenAI anchor-negative generation
    template = jinja2_env_async.get_template(
        os.getenv("GENERATE_TRIPLETS_TEMPLATE", "generate_anchor_negative.md")
    )

    process_start_time = time()
    batch_count = 0
    total_triplets = 0

    # Stream processing: Open output file once and write results as they're generated
    # This avoids loading all triplets into memory before writing
    async with aiofiles.open(output_path, mode="w", encoding="utf-8") as f:
        # Read chunks in batches using async streaming (memory efficient)
        async for batch in read_chunks_in_batches(
            file_path, start_line=start_line, end_line=end_line
        ):
            batch_count += 1
            # Generate triplets for all chunks in batch concurrently
            triplets = await generate_triplets_from_chunks(
                batch, template=template, client=client
            )

            # Write each triplet as a separate line in JSONL format
            for triplet in triplets:
                await f.write(json.dumps(triplet.model_dump()) + "\n")

            total_triplets += len(triplets)

            # Log progress every 10 batches to track long-running jobs
            if batch_count % 10 == 0:
                logger.info(
                    f"Processed {batch_count} batches, "
                    f"{total_triplets} triplets generated so far"
                )

    logger.info(
        f"Total: {batch_count} batches processed, {total_triplets} triplets generated"
    )

    logger.info(f"Triplets saved to: {output_path}")
    process_end_time = time()

    return f"Processing completed in {process_end_time - process_start_time:.2f} seconds."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate triplets from chunk data in JSONL format."
    )
    parser.add_argument(
        "filename",
        type=str,
        help="Name of the JSONL file in the data/ folder (e.g., chunks.jsonl)",
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

    args = parser.parse_args()

    # Validate line range arguments
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

    # Run the async main function
    asyncio.run(main(args.filename, args.start_line, args.end_line))
