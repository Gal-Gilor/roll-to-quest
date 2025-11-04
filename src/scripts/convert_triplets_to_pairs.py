"""Convert triplet training data to anchor-positive pair format.

This script reads JSONL files containing training triplets (anchor, positive, negative)
and extracts only the anchor-positive pairs for creating embedding training datasets.

Each triplet is validated against the Triplet Pydantic model before conversion.
Invalid triplets are logged and skipped, allowing processing to continue.

Workflow:
    1. Read triplets from input JSONL file
    2. Validate each triplet against Triplet model schema
    3. Extract (anchor, positive) pairs from valid triplets
    4. Stream results to output JSONL file
    5. Generate summary statistics

Dependencies:
    - Input JSONL file with triplet data (from generate_triplets.py)
    - Triplet Pydantic model for validation

Example Usage:
    # Process entire file with default output name
    python -m src.scripts.convert_triplets_to_pairs triplets.jsonl

    # Process with custom output path
    python -m src.scripts.convert_triplets_to_pairs triplets.jsonl \\
        --output data/my_pairs.jsonl

    # Process specific line range
    python -m src.scripts.convert_triplets_to_pairs triplets.jsonl \\
        --start-line 1 --end-line 1000

    # With poetry
    poetry run python -m src.scripts.convert_triplets_to_pairs triplets.jsonl

Output Format:
    JSONL file where each line contains:
    {"anchor": "...", "positive": "..."}
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from time import time

import aiofiles
from pydantic import ValidationError

from src.services.utils import read_chunks_in_batches
from src.settings import logger
from src.triplet_generation.models import Triplet


async def convert_triplets_to_pairs(
    input_file: str,
    output_file: str | None = None,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    """Convert triplets to anchor-positive pairs.

    Reads a JSONL file containing triplets, validates each entry against the
    Triplet model, and extracts anchor-positive pairs. Invalid triplets are
    logged and skipped.

    Output File Naming:
        - If output_file is provided: Use the specified path
        - Default: {input_stem}_anchor_positive_dataset.jsonl in data/ directory

    Args:
        input_file: Name of the JSONL file in the data/ folder containing triplets.
            Expected format: one JSON object per line with keys: anchor, positive, negative.
        output_file: Optional custom output file path. If None, uses default naming
            convention: {input_stem}_anchor_positive_dataset.jsonl
        start_line: Starting line number (1-indexed, inclusive). None processes from
            the beginning of the file.
        end_line: Ending line number (1-indexed, inclusive). None processes until
            the end of file.

    Returns:
        str: Success message with processing statistics and elapsed time.

    Raises:
        SystemExit: If input file is not found in the data/ directory.

    Example:
        >>> await convert_triplets_to_pairs("triplets.jsonl")
        'Converted 1000/1050 triplets (50 invalid) in 2.34 seconds.'
        >>> await convert_triplets_to_pairs(
        ...     "triplets.jsonl",
        ...     output_file="custom_pairs.jsonl",
        ...     start_line=1,
        ...     end_line=100
        ... )
        'Converted 95/100 triplets (5 invalid) in 0.45 seconds.'
    """
    # Construct the full path to the input file in the data/ directory
    data_dir = Path(__file__).parent.parent.parent / "data"
    input_path = data_dir / input_file

    # Determine output file path
    if output_file:
        output_path = Path(output_file)
        # If relative path, resolve relative to data/ directory
        if not output_path.is_absolute():
            output_path = data_dir / output_file
    else:
        # Default naming: {input_stem}_anchor_positive_dataset.jsonl
        input_stem = input_path.stem
        output_path = data_dir / f"{input_stem}_anchor_positive_dataset.jsonl"

    # Validate input file exists
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    log_msg = f"Reading triplets from: {input_path}"
    if start_line is not None or end_line is not None:
        log_msg += f" (lines {start_line or 1} to {end_line or 'end'})"
    logger.info(log_msg)
    logger.info(f"Output will be written to: {output_path}")

    process_start_time = time()
    batch_count = 0
    batch_size = 1000
    total_triplets = 0
    valid_triplets = 0
    invalid_triplets = 0

    # Stream processing: Open output file once and write results as they're generated
    async with aiofiles.open(output_path, mode="w", encoding="utf-8") as f:
        # Read triplets in batches using async streaming (memory efficient)
        async for batch in read_chunks_in_batches(
            input_path, batch_size=batch_size, start_line=start_line, end_line=end_line
        ):
            batch_count += 1

            # Process each triplet in the batch
            for idx, triplet_data in enumerate(batch, start=1):
                total_triplets += 1
                line_number = (batch_count - 1) * batch_size + idx

                try:
                    # Validate triplet against Pydantic model
                    triplet = Triplet.model_validate(triplet_data)

                    # Extract anchor-positive pair
                    pair = {"anchor": triplet.anchor, "positive": triplet.positive}

                    # Write to output file
                    await f.write(json.dumps(pair) + "\n")

                    valid_triplets += 1

                except ValidationError as e:
                    # Log validation errors with line number
                    invalid_triplets += 1
                    logger.warning(
                        f"Invalid triplet at line ~{line_number}: {e.error_count()} "
                        f"validation error(s). Skipping."
                    )
                    logger.debug(f"Validation errors: {e.errors()}")

                except Exception as e:
                    # Catch any unexpected errors
                    invalid_triplets += 1
                    logger.error(
                        f"Unexpected error processing triplet at line ~{line_number}: {e}"
                    )

            # Log progress every 10 batches
            if batch_count % 10 == 0:
                logger.info(
                    f"Processed {batch_count} batches: "
                    f"{valid_triplets} valid, {invalid_triplets} invalid"
                )

    # Final summary
    logger.info(
        f"Conversion complete: {valid_triplets}/{total_triplets} triplets converted "
        f"({invalid_triplets} invalid/skipped)"
    )
    logger.info(f"Anchor-positive pairs saved to: {output_path}")

    process_end_time = time()
    elapsed_time = process_end_time - process_start_time

    return (
        f"Converted {valid_triplets}/{total_triplets} triplets "
        f"({invalid_triplets} invalid) in {elapsed_time:.2f} seconds."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert triplets to anchor-positive pairs in JSONL format."
    )
    parser.add_argument(
        "filename",
        type=str,
        help="Name of the JSONL file in the data/ folder (e.g., triplets.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Custom output file path. Defaults to "
            "{input_stem}_anchor_positive_dataset.jsonl in data/ directory."
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

    # Run the async conversion function
    result = asyncio.run(
        convert_triplets_to_pairs(
            args.filename,
            output_file=args.output,
            start_line=args.start_line,
            end_line=args.end_line,
        )
    )

    print(result)
