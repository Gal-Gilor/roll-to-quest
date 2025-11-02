"""Tests for basic triplet validation service."""

import json

import pytest

from src.services.basic_triplet_validation import validate_triplets_file


@pytest.mark.asyncio
async def test_validate_valid_triplet(temp_triplets_jsonl, sample_valid_triplet):
    """Test validation of a single valid triplet."""
    # Write valid triplet to file
    with open(temp_triplets_jsonl, "w") as f:
        f.write(json.dumps(sample_valid_triplet) + "\n")

    result = await validate_triplets_file(temp_triplets_jsonl)

    assert result["total_triplets"] == 1
    assert result["valid_count"] == 1
    assert result["invalid_count"] == 0
    assert result["duplicate_count"] == 0
    assert result["invalid_indices"] == []
    assert result["duplicate_indices"] == []


@pytest.mark.asyncio
async def test_validate_invalid_triplets(temp_triplets_jsonl, sample_invalid_triplets):
    """Test validation catches various invalid triplet formats."""
    # Write invalid triplets to file
    with open(temp_triplets_jsonl, "w") as f:
        for triplet in sample_invalid_triplets:
            f.write(json.dumps(triplet) + "\n")

    result = await validate_triplets_file(temp_triplets_jsonl)

    assert result["total_triplets"] == len(sample_invalid_triplets)
    assert result["valid_count"] == 0
    assert result["invalid_count"] == len(sample_invalid_triplets)
    assert len(result["invalid_indices"]) == len(sample_invalid_triplets)


@pytest.mark.asyncio
async def test_validate_duplicates(temp_triplets_jsonl, sample_triplets_with_duplicates):
    """Test duplicate detection identifies exact duplicate triplets."""
    # Write triplets with duplicates to file
    with open(temp_triplets_jsonl, "w") as f:
        for triplet in sample_triplets_with_duplicates:
            f.write(json.dumps(triplet) + "\n")

    result = await validate_triplets_file(temp_triplets_jsonl)

    assert result["total_triplets"] == 4
    assert result["valid_count"] == 3  # Only 3 unique valid triplets
    assert result["invalid_count"] == 1  # 1 duplicate
    assert result["duplicate_count"] == 1
    assert 3 in result["duplicate_indices"]  # Third triplet is duplicate


@pytest.mark.asyncio
async def test_validate_mixed_triplets(
    temp_triplets_jsonl, sample_valid_triplet, sample_invalid_triplets
):
    """Test validation of file with both valid and invalid triplets."""
    # Write mix of valid and invalid triplets
    with open(temp_triplets_jsonl, "w") as f:
        f.write(json.dumps(sample_valid_triplet) + "\n")
        for triplet in sample_invalid_triplets[:2]:  # Add 2 invalid
            f.write(json.dumps(triplet) + "\n")

    result = await validate_triplets_file(temp_triplets_jsonl)

    assert result["total_triplets"] == 3
    assert result["valid_count"] == 1
    assert result["invalid_count"] == 2
    assert result["invalid_indices"] == [2, 3]


@pytest.mark.asyncio
async def test_validate_empty_file(temp_triplets_jsonl):
    """Test validation of empty JSONL file."""
    # Create empty file
    temp_triplets_jsonl.write_text("")

    result = await validate_triplets_file(temp_triplets_jsonl)

    assert result["total_triplets"] == 0
    assert result["valid_count"] == 0
    assert result["invalid_count"] == 0
    assert result["duplicate_count"] == 0


@pytest.mark.asyncio
async def test_validate_nonexistent_file(tmp_path):
    """Test validation raises error for nonexistent file."""
    nonexistent_file = tmp_path / "does_not_exist.jsonl"

    with pytest.raises(FileNotFoundError):
        await validate_triplets_file(nonexistent_file)


@pytest.mark.asyncio
async def test_validate_batch_processing(temp_triplets_jsonl, sample_valid_triplet):
    """Test validation works correctly with batch processing."""
    # Write 10 valid triplets
    with open(temp_triplets_jsonl, "w") as f:
        for i in range(10):
            triplet = sample_valid_triplet.copy()
            triplet["anchor"] = f"Question {i}"
            f.write(json.dumps(triplet) + "\n")

    result = await validate_triplets_file(temp_triplets_jsonl, batch_size=3)

    assert result["total_triplets"] == 10
    assert result["valid_count"] == 10
    assert result["invalid_count"] == 0
