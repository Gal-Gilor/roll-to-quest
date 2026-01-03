"""Generate training pairs (anchor, positive) from chunked documents.

This script processes document chunks from JSONL files and generates anchor-positive pairs for
training embedding models. Each pair consists of:
- anchor: A query or question related to the content
- positive: The actual document chunk (relevant to the anchor)

This script is optimized for cost efficiency by generating only anchors (not negatives),
reducing API costs by ~40-50% compared to full triplet generation.

The pairs are generated using Gemini 2.5 (Flash by default).
The prompt template for generating anchors is tracked in "src/templates/" directory.
The project uses Jinja2 for loading and rendering templates.

Workflow:
    1. Read chunked documents from JSONL file (created by chunk_documents.py)
    2. Process chunks in batches for efficiency
    3. For each chunk, use Gemini to generate anchor queries
    4. Combine anchors with original chunk text to form complete pairs
    5. Stream results to output JSONL file

Dependencies:
    - Google Gemini credentials (configured in settings)
    - Jinja2 template for anchor generation (default: generate_anchor_only.md)
    - Input chunks in JSONL format from chunk_documents.py

Environment Variables:
    GENERATE_PAIRS_TEMPLATE: Name of the Jinja2 template to use for
        generating anchors. Defaults to "generate_anchor_only.md".

Example Usage:
    # Process entire file
    python -m src.scripts.generate_pairs document_chunks.jsonl

    # Process specific line range
    python -m src.scripts.generate_pairs document_chunks.jsonl \\
        --start-line 1 --end-line 100

    # With poetry
    poetry run python -m src.scripts.generate_pairs document_chunks.jsonl

Output Format:
    JSONL file where each line contains a JSON object with:
    - anchor: str
    - positive: str

Cost Savings:
    Compared to generate_triplets.py, this script reduces API costs by ~40-50%
    by not generating hard negatives. Use this for most training scenarios unless
    you specifically need triplets with hard negatives.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from time import time

import aiofiles

from src.pair_generation.utils import generate_pairs_from_chunks
from src.services.utils import read_chunks_in_batches
from src.settings import client
from src.settings import jinja2_env_async
from src.settings import logger


async def main(
    input_file: str, start_line: int | None = None, end_line: int | None = None
) -> str:
    """Generate anchor-positive pairs from chunks in the input file.

    This function orchestrates the entire pair generation process:
    1. Locates the input JSONL file in the data/ directory
    2. Loads the Jinja2 template for anchor generation
    3. Reads chunks in batches using async streaming
    4. For each batch, generates pairs concurrently via GenAI
    5. Writes pairs to output file immediately (streaming)
    6. Logs progress every 10 batches

    Output File Naming:
        - Full file: data/pairs.jsonl
        - Partial range: data/pairs_lines_{start}_to_{end}.jsonl

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
        output_path = data_dir / f"pairs{range_suffix}.jsonl"
    else:
        output_path = data_dir / "pairs.jsonl"

    if not file_path.exists():
        logger.error(f"Input file not found: {file_path}")
        sys.exit(1)

    log_msg = f"Reading chunks from: {file_path}"
    if start_line is not None or end_line is not None:
        log_msg += f" (lines {start_line or 1} to {end_line or 'end'})"
    logger.info(log_msg)

    # Load the Jinja2 template once (reused for all batches)
    # Template defines the prompt structure for GenAI anchor generation
    template = jinja2_env_async.get_template(
        os.getenv("GENERATE_PAIRS_TEMPLATE", "generate_anchor_only.md")
    )

    process_start_time = time()
    batch_count = 0
    total_pairs = 0

    # Stream processing: Open output file once and write results as they're generated
    # This avoids loading all pairs into memory before writing
    async with aiofiles.open(output_path, mode="w", encoding="utf-8") as f:
        # Read chunks in batches using async streaming (memory efficient)
        async for batch in read_chunks_in_batches(
            file_path, start_line=start_line, end_line=end_line
        ):
            batch_count += 1
            # Generate pairs for all chunks in batch concurrently
            pairs = await generate_pairs_from_chunks(batch, template=template, client=client)

            # Write each pair as a separate line in JSONL format
            for pair in pairs:
                await f.write(json.dumps(pair.model_dump()) + "\n")

            total_pairs += len(pairs)

            # Log progress every 10 batches to track long-running jobs
            if batch_count % 10 == 0:
                logger.info(
                    f"Processed {batch_count} batches, {total_pairs} pairs generated so far"
                )

    logger.info(f"Total: {batch_count} batches processed, {total_pairs} pairs generated")

    logger.info(f"Pairs saved to: {output_path}")
    process_end_time = time()

    return f"Processing completed in {process_end_time - process_start_time:.2f} seconds."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate anchor-positive pairs from chunk data in JSONL format."
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
