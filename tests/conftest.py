from textwrap import dedent
from unittest.mock import AsyncMock

import pytest

from src.services.cloud_storage import AsyncStorageBucket
from src.services.cloud_storage import Storage
from src.services.splitter import MarkdownSplitter


@pytest.fixture
def splitter():
    return MarkdownSplitter()


@pytest.fixture
def sample_markdown():
    return dedent(
        """
        # Header 1
        Content 1
        ## Header 1.1
        Content 1.1
        # Header 2
        Content 2
        ## Header 2.1
        Content 2.1
        ```python
        # Code block comment
        ```
        ### Header 2.1.1
        Content 2.1.1
        ## Header 2.2
        Content 2.2"""
    )


@pytest.fixture
def nested_markdown():
    return dedent(
        """
        # Main
        Content
        ## Sub
        Sub content
        ### Deep
        Deep content
        ## Sub 2
        Sub content 2
        """
    )


@pytest.fixture
def mock_storage_client():
    """Create a mock for the async Google Cloud Storage client."""
    mock_client = AsyncMock(spec=Storage)

    # Configure common mock responses
    mock_client.list_objects = AsyncMock(return_value={"items": []})
    mock_client.upload = AsyncMock(return_value={"name": "test-blob"})
    mock_client.download = AsyncMock(return_value=b"test file content")
    mock_client.delete = AsyncMock()
    mock_client.download_metadata = AsyncMock(return_value={"size": "1024"})

    return mock_client


@pytest.fixture
def mock_storage_bucket():
    """Create a proper mock for the AsyncStorageBucket."""
    mock_bucket = AsyncMock(spec=AsyncStorageBucket)
    mock_bucket.name = "test-bucket"
    mock_bucket.exists = AsyncMock(return_value=True)
    mock_bucket.list_blobs = AsyncMock(return_value=[])
    mock_bucket.upload_blob = AsyncMock(return_value={"name": "test-blob"})
    mock_bucket.download_blob = AsyncMock(return_value=None)
    mock_bucket.delete_blob = AsyncMock(return_value=None)
    mock_bucket.get_blob_metadata = AsyncMock(return_value={"size": "1024"})

    return mock_bucket


@pytest.fixture
def temp_file(tmp_path):
    """Create a temporary file for upload/download testing."""
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("This is test content")
    return file_path


@pytest.fixture
def sample_valid_triplet():
    """Single valid triplet for unit testing."""
    return {
        "anchor": "What is a fireball spell?",
        "positive": "Fireball is a 3rd-level evocation spell that deals fire damage.",
        "negative": "Lightning bolt is a 3rd-level evocation spell that deals lightning damage.",
    }


@pytest.fixture
def sample_invalid_triplets():
    """Collection of invalid triplets for validation testing."""
    return [
        {"anchor": "test", "positive": "text"},  # Missing negative field
        {"anchor": "", "positive": "text", "negative": "text"},  # Empty anchor
        {"anchor": "text", "positive": "   ", "negative": "text"},  # Whitespace positive
        {"positive": "text", "negative": "text"},  # Missing anchor field
        {"anchor": None, "positive": "text", "negative": "text"},  # Null anchor
    ]


@pytest.fixture
def sample_triplets_with_duplicates():
    """Triplets with intentional duplicates for duplicate detection testing."""
    return [
        {"anchor": "Q1", "positive": "P1", "negative": "N1"},
        {"anchor": "Q2", "positive": "P2", "negative": "N2"},
        {"anchor": "Q1", "positive": "P1", "negative": "N1"},  # Exact duplicate of first
        {"anchor": "Q3", "positive": "P3", "negative": "N3"},
    ]


@pytest.fixture
def temp_triplets_jsonl(tmp_path):
    """Create temporary JSONL file with sample triplets."""
    file_path = tmp_path / "triplets.jsonl"
    return file_path
