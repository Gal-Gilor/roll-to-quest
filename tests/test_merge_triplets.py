import json
from pathlib import Path

import pytest

from src.scripts.merge_triplets import main
from src.scripts.merge_triplets import merge_files
from src.scripts.merge_triplets import validate_triplet


def test_validate_triplet_valid():
    """Test validation of a valid triplet."""
    # Arrange
    triplet = {
        "anchor": "What is Python?",
        "positive": "Python is a programming language.",
        "negative": "Java is a programming language.",
    }

    # Act
    result = validate_triplet(triplet)

    # Assert
    assert result is True


def test_validate_triplet_missing_field():
    """Test validation fails when a required field is missing."""
    # Arrange
    triplet = {
        "anchor": "What is Python?",
        "positive": "Python is a programming language.",
    }

    # Act
    result = validate_triplet(triplet)

    # Assert
    assert result is False


def test_validate_triplet_missing_multiple_fields():
    """Test validation fails when multiple fields are missing."""
    # Arrange
    triplet = {"anchor": "What is Python?"}

    # Act
    result = validate_triplet(triplet)

    # Assert
    assert result is False


def test_validate_triplet_empty():
    """Test validation fails for empty dictionary."""
    # Arrange
    triplet = {}

    # Act
    result = validate_triplet(triplet)

    # Assert
    assert result is False


def test_validate_triplet_extra_fields():
    """Test validation succeeds even with extra fields."""
    # Arrange
    triplet = {
        "anchor": "What is Python?",
        "positive": "Python is a programming language.",
        "negative": "Java is a programming language.",
        "metadata": {"source": "wikipedia"},
    }

    # Act
    result = validate_triplet(triplet)

    # Assert
    assert result is True


@pytest.mark.asyncio
async def test_merge_files_single_file(tmp_path):
    """Test merging a single triplet file."""
    # Arrange
    input_file = tmp_path / "triplets_1.jsonl"
    output_file = tmp_path / "merged.jsonl"

    triplets = [
        {
            "anchor": "Question 1?",
            "positive": "Answer 1",
            "negative": "Wrong answer 1",
        },
        {
            "anchor": "Question 2?",
            "positive": "Answer 2",
            "negative": "Wrong answer 2",
        },
    ]
    input_file.write_text("\n".join(json.dumps(t) for t in triplets))

    # Act
    total, invalid = await merge_files([input_file], output_file)

    # Assert
    assert total == 2
    assert invalid == 0
    assert output_file.exists()

    # Verify output content
    output_lines = output_file.read_text().strip().split("\n")
    assert len(output_lines) == 2
    assert json.loads(output_lines[0]) == triplets[0]
    assert json.loads(output_lines[1]) == triplets[1]


@pytest.mark.asyncio
async def test_merge_files_multiple_files(tmp_path):
    """Test merging multiple triplet files."""
    # Arrange
    file1 = tmp_path / "triplets_1.jsonl"
    file2 = tmp_path / "triplets_2.jsonl"
    output_file = tmp_path / "merged.jsonl"

    triplets1 = [
        {"anchor": "Q1?", "positive": "A1", "negative": "N1"},
        {"anchor": "Q2?", "positive": "A2", "negative": "N2"},
    ]
    triplets2 = [
        {"anchor": "Q3?", "positive": "A3", "negative": "N3"},
        {"anchor": "Q4?", "positive": "A4", "negative": "N4"},
    ]

    file1.write_text("\n".join(json.dumps(t) for t in triplets1))
    file2.write_text("\n".join(json.dumps(t) for t in triplets2))

    # Act
    total, invalid = await merge_files([file1, file2], output_file)

    # Assert
    assert total == 4
    assert invalid == 0

    # Verify output content
    output_lines = output_file.read_text().strip().split("\n")
    assert len(output_lines) == 4
    assert json.loads(output_lines[0]) == triplets1[0]
    assert json.loads(output_lines[1]) == triplets1[1]
    assert json.loads(output_lines[2]) == triplets2[0]
    assert json.loads(output_lines[3]) == triplets2[1]


@pytest.mark.asyncio
async def test_merge_files_with_invalid_triplets(tmp_path, caplog):
    """Test merging files with some invalid triplets."""
    # Arrange
    input_file = tmp_path / "triplets.jsonl"
    output_file = tmp_path / "merged.jsonl"

    content = (
        '{"anchor": "Q1?", "positive": "A1", "negative": "N1"}\n'
        '{"anchor": "Q2?", "positive": "A2"}\n'  # Missing 'negative'
        '{"anchor": "Q3?", "positive": "A3", "negative": "N3"}\n'
        '{"positive": "A4", "negative": "N4"}\n'  # Missing 'anchor'
    )
    input_file.write_text(content)

    # Act
    total, invalid = await merge_files([input_file], output_file)

    # Assert
    assert total == 2
    assert invalid == 2

    # Verify only valid triplets were written
    output_lines = output_file.read_text().strip().split("\n")
    assert len(output_lines) == 2
    assert json.loads(output_lines[0])["anchor"] == "Q1?"
    assert json.loads(output_lines[1])["anchor"] == "Q3?"

    # Verify warnings were logged
    assert any("Missing required fields" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_merge_files_with_invalid_json(tmp_path, caplog):
    """Test merging files with invalid JSON lines."""
    # Arrange
    input_file = tmp_path / "triplets.jsonl"
    output_file = tmp_path / "merged.jsonl"

    content = (
        '{"anchor": "Q1?", "positive": "A1", "negative": "N1"}\n'
        '{invalid json}\n'
        '{"anchor": "Q2?", "positive": "A2", "negative": "N2"}\n'
    )
    input_file.write_text(content)

    # Act
    total, invalid = await merge_files([input_file], output_file)

    # Assert - invalid JSON is handled by read_chunks_in_batches (logged but not counted)
    assert total == 2
    # Invalid count only tracks validation failures, not JSON parse errors
    assert invalid == 0

    # Verify only valid triplets were written
    output_lines = output_file.read_text().strip().split("\n")
    assert len(output_lines) == 2

    # Verify error was logged by read_chunks_in_batches
    assert any("Failed to parse JSON" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_merge_files_empty_file(tmp_path):
    """Test merging an empty file."""
    # Arrange
    input_file = tmp_path / "empty.jsonl"
    output_file = tmp_path / "merged.jsonl"
    input_file.write_text("")

    # Act
    total, invalid = await merge_files([input_file], output_file)

    # Assert
    assert total == 0
    assert invalid == 0
    assert output_file.exists()
    assert output_file.read_text() == ""


@pytest.mark.asyncio
async def test_merge_files_empty_lines(tmp_path):
    """Test merging files with empty lines."""
    # Arrange
    input_file = tmp_path / "triplets.jsonl"
    output_file = tmp_path / "merged.jsonl"

    content = (
        '{"anchor": "Q1?", "positive": "A1", "negative": "N1"}\n'
        "\n"
        "\n"
        '{"anchor": "Q2?", "positive": "A2", "negative": "N2"}\n'
        "\n"
    )
    input_file.write_text(content)

    # Act
    total, invalid = await merge_files([input_file], output_file)

    # Assert
    assert total == 2
    assert invalid == 0

    # Verify empty lines were skipped
    output_lines = output_file.read_text().strip().split("\n")
    assert len(output_lines) == 2


@pytest.mark.asyncio
async def test_merge_files_custom_batch_size(tmp_path):
    """Test merging with custom batch size."""
    # Arrange
    input_file = tmp_path / "triplets.jsonl"
    output_file = tmp_path / "merged.jsonl"

    triplets = [
        {"anchor": f"Q{i}?", "positive": f"A{i}", "negative": f"N{i}"}
        for i in range(50)
    ]
    input_file.write_text("\n".join(json.dumps(t) for t in triplets))

    # Act - use small batch size
    total, invalid = await merge_files([input_file], output_file, batch_size=5)

    # Assert
    assert total == 50
    assert invalid == 0

    # Verify all triplets were written
    output_lines = output_file.read_text().strip().split("\n")
    assert len(output_lines) == 50


@pytest.mark.asyncio
async def test_main_no_matching_files(tmp_path, monkeypatch, caplog):
    """Test main function when no matching files are found."""
    # Arrange - create a directory with no matching files
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Import the module to get __file__
    import src.scripts.merge_triplets as merge_module

    # Mock __file__ in the module
    original_file = merge_module.__file__
    script_path = tmp_path / "src" / "scripts" / "merge_triplets.py"
    script_path.parent.mkdir(parents=True)
    script_path.touch()
    monkeypatch.setattr(merge_module, "__file__", str(script_path))

    # Act & Assert
    with pytest.raises(SystemExit) as exc_info:
        await main("nonexistent_*.jsonl", "output.jsonl")

    assert exc_info.value.code == 1
    assert any("No files found" in record.message for record in caplog.records)

    # Restore
    monkeypatch.setattr(merge_module, "__file__", original_file)


@pytest.mark.asyncio
async def test_main_successful_merge(tmp_path, monkeypatch):
    """Test successful merge using main function."""
    # Arrange
    # Create data directory structure
    script_path = tmp_path / "src" / "scripts" / "merge_triplets.py"
    script_path.parent.mkdir(parents=True)
    script_path.touch()

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create test files
    file1 = data_dir / "triplet_lines_1_to_10.jsonl"
    file2 = data_dir / "triplet_lines_11_to_20.jsonl"

    triplets1 = [{"anchor": f"Q{i}?", "positive": f"A{i}", "negative": f"N{i}"} for i in range(3)]
    triplets2 = [{"anchor": f"Q{i}?", "positive": f"A{i}", "negative": f"N{i}"} for i in range(3, 6)]

    file1.write_text("\n".join(json.dumps(t) for t in triplets1))
    file2.write_text("\n".join(json.dumps(t) for t in triplets2))

    # Import and mock __file__ in the module
    import src.scripts.merge_triplets as merge_module

    original_file = merge_module.__file__
    monkeypatch.setattr(merge_module, "__file__", str(script_path))

    # Act
    result = await main("triplet_lines_*.jsonl", "merged_output.jsonl")

    # Assert
    assert "Processing completed" in result
    output_file = data_dir / "merged_output.jsonl"
    assert output_file.exists()

    # Verify merged content
    output_lines = output_file.read_text().strip().split("\n")
    assert len(output_lines) == 6

    # Restore
    monkeypatch.setattr(merge_module, "__file__", original_file)


@pytest.mark.asyncio
async def test_merge_files_order_preserved(tmp_path):
    """Test that order of triplets is preserved when merging."""
    # Arrange
    file1 = tmp_path / "file1.jsonl"
    file2 = tmp_path / "file2.jsonl"
    output_file = tmp_path / "merged.jsonl"

    triplets1 = [
        {"anchor": "First", "positive": "1", "negative": "N1"},
        {"anchor": "Second", "positive": "2", "negative": "N2"},
    ]
    triplets2 = [
        {"anchor": "Third", "positive": "3", "negative": "N3"},
        {"anchor": "Fourth", "positive": "4", "negative": "N4"},
    ]

    file1.write_text("\n".join(json.dumps(t) for t in triplets1))
    file2.write_text("\n".join(json.dumps(t) for t in triplets2))

    # Act
    await merge_files([file1, file2], output_file)

    # Assert - verify order
    output_lines = output_file.read_text().strip().split("\n")
    parsed_triplets = [json.loads(line) for line in output_lines]

    assert parsed_triplets[0]["anchor"] == "First"
    assert parsed_triplets[1]["anchor"] == "Second"
    assert parsed_triplets[2]["anchor"] == "Third"
    assert parsed_triplets[3]["anchor"] == "Fourth"


@pytest.mark.asyncio
async def test_merge_files_complex_triplets(tmp_path):
    """Test merging files with complex, realistic triplet content."""
    # Arrange
    input_file = tmp_path / "triplets.jsonl"
    output_file = tmp_path / "merged.jsonl"

    triplets = [
        {
            "anchor": "Which character species are listed in this manual?",
            "positive": "Character Species: Dragonborn, Dwarf, Elf, Gnome, Goliath...",
            "negative": "This manual lists character species such as the Elf, Gnome...",
        },
        {
            "anchor": "Are there rules for multiclassing in this guide?",
            "positive": "Character Creation... Multiclassing......24",
            "negative": "This guide does not contain rules for multiclassing...",
        },
    ]
    input_file.write_text("\n".join(json.dumps(t) for t in triplets))

    # Act
    total, invalid = await merge_files([input_file], output_file)

    # Assert
    assert total == 2
    assert invalid == 0

    # Verify content preservation
    output_lines = output_file.read_text().strip().split("\n")
    parsed_triplets = [json.loads(line) for line in output_lines]

    assert parsed_triplets[0]["anchor"] == triplets[0]["anchor"]
    assert parsed_triplets[0]["positive"] == triplets[0]["positive"]
    assert parsed_triplets[0]["negative"] == triplets[0]["negative"]
