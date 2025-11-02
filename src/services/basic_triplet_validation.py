"""Basic validation service for triplet datasets.

This module provides utilities for validating triplet (anchor, positive, negative)
datasets used for fine-tuning embedding models.

Performs basic quality checks:
- Format validation: Ensures triplets have required fields with correct types
- Completeness: Detects null/empty values
- Duplicate detection: Identifies exact duplicate triplets
"""

import json
from pathlib import Path

from pydantic import ValidationError

from src.services.utils import read_chunks_in_batches
from src.settings import logger
from src.triplet_generation.models import Triplet


async def validate_triplets_file(
    file_path: str | Path, batch_size: int = 100
) -> dict:
    """Validate a JSONL file containing triplets.

    Performs basic validation checks:
    - Format: All required fields present with correct types
    - Completeness: No null or empty values
    - Duplicates: Exact duplicate triplets

    Args:
        file_path: Path to JSONL file with triplet data.
        batch_size: Number of triplets to process per batch. Defaults to 100.

    Returns:
        Dictionary with validation results:
            - total_triplets: Total number of triplets processed
            - valid_count: Number of valid triplets
            - invalid_count: Number of invalid triplets
            - duplicate_count: Number of exact duplicate triplets
            - invalid_indices: List of line numbers with invalid triplets
            - duplicate_indices: List of line numbers with duplicate triplets

    Raises:
        FileNotFoundError: If the specified file does not exist.

    Examples:
        >>> summary = await validate_triplets_file("data/triplets.jsonl")
        >>> print(f"Valid: {summary['valid_count']}/{summary['total_triplets']}")
        Valid: 950/1000
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"Starting validation of {file_path}")

    # Initialize counters
    total_triplets = 0
    valid_count = 0
    invalid_indices = []
    duplicate_indices = []

    # Track seen triplets for duplicate detection
    seen_triplets = set()

    # Stream file in batches
    async for batch in read_chunks_in_batches(file_path, batch_size=batch_size):
        for triplet_dict in batch:
            total_triplets += 1

            # Validate format and completeness using Pydantic model
            try:
                triplet = Triplet(**triplet_dict)

                # Check for empty/whitespace-only fields
                empty_fields = [
                    field_name
                    for field_name in Triplet.model_fields.keys()
                    if not getattr(triplet, field_name).strip()
                ]
                if empty_fields:
                    raise ValueError(
                        f"Empty or whitespace-only fields: {', '.join(empty_fields)}"
                    )

                # Check for duplicates
                triplet_str = json.dumps(triplet_dict, sort_keys=True)
                if triplet_str in seen_triplets:
                    duplicate_indices.append(total_triplets)
                    raise ValueError("Duplicate triplet")

                seen_triplets.add(triplet_str)
                valid_count += 1

            except (ValidationError, ValueError) as e:
                invalid_indices.append(total_triplets)
                logger.warning(f"Line {total_triplets}: {e}")

    invalid_count = total_triplets - valid_count
    duplicate_count = len(duplicate_indices)

    logger.info(
        f"Processed {total_triplets} triplets: "
        f"{valid_count} valid, {invalid_count} invalid, "
        f"{duplicate_count} duplicates"
    )

    return {
        "total_triplets": total_triplets,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "duplicate_count": duplicate_count,
        "invalid_indices": invalid_indices,
        "duplicate_indices": duplicate_indices,
    }
