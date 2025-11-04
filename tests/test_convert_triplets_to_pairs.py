"""Unit tests for the convert_triplets_to_pairs script.

Tests cover:
- Valid triplet conversion
- Invalid triplet handling (validation errors)
- Line range processing
- Output file naming (default and custom)
- Empty input file handling
"""

import json

import pytest

from src.scripts.convert_triplets_to_pairs import convert_triplets_to_pairs


@pytest.mark.asyncio
async def test_convert_valid_triplets(valid_triplets_file, temp_data_dir, monkeypatch):
    """Test conversion of all valid triplets."""
    import src.scripts.convert_triplets_to_pairs as convert_module

    fake_path = str(temp_data_dir / "scripts" / "convert_triplets_to_pairs.py")
    monkeypatch.setattr(convert_module, "__file__", fake_path)

    result = await convert_triplets_to_pairs(valid_triplets_file.name)

    assert "Converted 3/3 triplets" in result
    assert "0 invalid" in result

    output_file = temp_data_dir / "valid_triplets_anchor_positive_dataset.jsonl"
    assert output_file.exists()

    with open(output_file, "r", encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f]

    assert len(pairs) == 3
    assert pairs[0]["anchor"] == "What is Python?"
    assert "negative" not in pairs[0]


@pytest.mark.asyncio
async def test_skip_invalid_triplets(mixed_validity_file, temp_data_dir, monkeypatch):
    """Test that invalid triplets are skipped with logging."""
    import src.scripts.convert_triplets_to_pairs as convert_module

    fake_path = str(temp_data_dir / "scripts" / "convert_triplets_to_pairs.py")
    monkeypatch.setattr(convert_module, "__file__", fake_path)

    result = await convert_triplets_to_pairs(mixed_validity_file.name)

    assert "Converted 4/5 triplets" in result
    assert "1 invalid" in result

    output_file = temp_data_dir / "mixed_triplets_anchor_positive_dataset.jsonl"
    with open(output_file, "r", encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f]

    assert len(pairs) == 4


@pytest.mark.asyncio
async def test_custom_output_path(valid_triplets_file, temp_data_dir, monkeypatch):
    """Test conversion with custom output file path."""
    import src.scripts.convert_triplets_to_pairs as convert_module

    fake_path = str(temp_data_dir / "scripts" / "convert_triplets_to_pairs.py")
    monkeypatch.setattr(convert_module, "__file__", fake_path)

    result = await convert_triplets_to_pairs(
        valid_triplets_file.name, output_file="custom_pairs.jsonl"
    )

    assert "Converted 3/3 triplets" in result
    assert (temp_data_dir / "custom_pairs.jsonl").exists()


@pytest.mark.asyncio
async def test_line_range_processing(valid_triplets_file, temp_data_dir, monkeypatch):
    """Test conversion with line range limits."""
    import src.scripts.convert_triplets_to_pairs as convert_module

    fake_path = str(temp_data_dir / "scripts" / "convert_triplets_to_pairs.py")
    monkeypatch.setattr(convert_module, "__file__", fake_path)

    result = await convert_triplets_to_pairs(
        valid_triplets_file.name, start_line=1, end_line=2
    )

    assert "Converted 2/2 triplets" in result

    output_file = temp_data_dir / "valid_triplets_anchor_positive_dataset.jsonl"
    with open(output_file, "r", encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f]

    assert len(pairs) == 2


@pytest.mark.asyncio
async def test_empty_input_file(empty_file, temp_data_dir, monkeypatch):
    """Test handling of empty input file."""
    import src.scripts.convert_triplets_to_pairs as convert_module

    fake_path = str(temp_data_dir / "scripts" / "convert_triplets_to_pairs.py")
    monkeypatch.setattr(convert_module, "__file__", fake_path)

    result = await convert_triplets_to_pairs(empty_file.name)

    assert "Converted 0/0 triplets" in result
    assert (temp_data_dir / "empty_anchor_positive_dataset.jsonl").exists()


@pytest.mark.asyncio
async def test_output_schema(valid_triplets_file, temp_data_dir, monkeypatch):
    """Test that output contains only anchor and positive fields."""
    import src.scripts.convert_triplets_to_pairs as convert_module

    fake_path = str(temp_data_dir / "scripts" / "convert_triplets_to_pairs.py")
    monkeypatch.setattr(convert_module, "__file__", fake_path)

    await convert_triplets_to_pairs(valid_triplets_file.name)

    output_file = temp_data_dir / "valid_triplets_anchor_positive_dataset.jsonl"

    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            pair = json.loads(line)
            assert set(pair.keys()) == {"anchor", "positive"}
            assert isinstance(pair["anchor"], str)
            assert isinstance(pair["positive"], str)
