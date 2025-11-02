"""Merge multiple triplet JSONL files into a single consolidated file.

This script processes and merges triplet JSONL files from the data/ directory
into a single consolidated output file. It validates JSON structure and checks
for required fields (anchor, positive, negative).

Example Usage:
    # Merge all triplet_lines files (default pattern)
    python -m src.scripts.merge_triplets

    # Use custom glob pattern
    python -m src.scripts.merge_triplets --pattern "triplets_*.jsonl"

    # Specify custom output filename
    python -m src.scripts.merge_triplets --output merged.jsonl

    # With poetry
    poetry run python -m src.scripts.merge_triplets --pattern "*.jsonl"
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from time import time

import aiofiles

from src.services.utils import read_chunks_in_batches
from src.settings import logger


def validate_triplet(triplet: dict) -> bool:
    """Validate that a triplet contains required fields.

    Args:
        triplet: Dictionary representing a single triplet.

    Returns:
        bool: True if triplet has anchor, positive, and negative fields.
    """
    required_fields = {"anchor", "positive", "negative"}
    return required_fields.issubset(triplet.keys())


async def merge_files(
    input_files: list[Path], output_path: Path, batch_size: int = 100
) -> tuple[int, int]:
    """Merge multiple JSONL files into a single output file.

    Args:
        input_files: List of input file paths to merge.
        output_path: Path to the output file.
        batch_size: Number of triplets to read per batch.

    Returns:
        tuple[int, int]: (total_triplets_written, total_invalid_lines)
    """
    total_triplets = 0
    total_invalid = 0

    async with aiofiles.open(output_path, mode="w", encoding="utf-8") as outfile:
        for file_path in input_files:
            logger.info(f"Processing: {file_path.name}")
            valid_count = 0

            async for batch in read_chunks_in_batches(file_path, batch_size=batch_size):
                for triplet in batch:
                    if not validate_triplet(triplet):
                        logger.warning(
                            f"Skipping invalid triplet in {file_path.name}: "
                            "Missing required fields"
                        )
                        total_invalid += 1
                        continue

                    await outfile.write(json.dumps(triplet) + "\n")
                    valid_count += 1

            logger.info(f"  Processed {valid_count} triplets from {file_path.name}")
            total_triplets += valid_count

    return total_triplets, total_invalid


async def main(pattern: str, output_filename: str) -> str:
    """Merge all matching JSONL files in data/ directory.

    Args:
        pattern: Glob pattern to match input files (e.g., "triplet_lines_*.jsonl").
        output_filename: Name of the output file in data/ directory.

    Returns:
        str: Success message with processing statistics.

    Raises:
        SystemExit: If no matching files are found.
    """
    data_dir = Path(__file__).parent.parent.parent / "data"
    input_files = sorted(data_dir.glob(pattern))

    if not input_files:
        logger.error(f"No files found matching pattern '{pattern}' in {data_dir}")
        sys.exit(1)

    logger.info(f"Found {len(input_files)} file(s) to merge:")
    for file in input_files:
        logger.info(f"  - {file.name}")

    output_path = data_dir / output_filename
    logger.info(f"Output: {output_path}")

    start_time = time()
    total_triplets, total_invalid = await merge_files(input_files, output_path)
    elapsed_time = time() - start_time

    logger.info("=" * 60)
    logger.info("Merge completed!")
    logger.info(f"Files processed: {len(input_files)}")
    logger.info(f"Total triplets written: {total_triplets}")
    if total_invalid > 0:
        logger.info(f"Invalid lines skipped: {total_invalid}")
    logger.info(f"Processing time: {elapsed_time:.2f} seconds")
    logger.info("=" * 60)

    return f"Processing completed in {elapsed_time:.2f} seconds."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge multiple triplet JSONL files into a single file."
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="triplet_lines_*.jsonl",
        help="Glob pattern to match input files (default: triplet_lines_*.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="merged_triplets.jsonl",
        help="Output filename in data/ directory (default: merged_triplets.jsonl)",
    )

    args = parser.parse_args()
    asyncio.run(main(args.pattern, args.output))
