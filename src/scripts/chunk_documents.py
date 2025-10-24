"""Chunk text documents into sections and save as JSONL for downstream processing.

This script processes text files (.txt, .md) and splits them into semantic sections
based on Markdown header hierarchy. It's designed as a preprocessing step for
document indexing, embedding generation, or other NLP tasks.

Features:
    - Automatic discovery of .txt and .md files in a directory
    - Markdown header-based splitting (# through ##### H5)
    - Preserves hierarchical relationships (parent headers, sibling sections)
    - Filters out empty sections automatically
    - Outputs to JSONL format for easy streaming and processing

Workflow:
    1. Discover text files in the specified directory
    2. For each file, split into sections using MarkdownSplitter
    3. Convert sections to dictionaries with metadata
    4. Filter out sections with empty text content
    5. Write to JSONL file (one section per line)

Output Format:
    JSONL files named {original_filename}_chunks.jsonl with structure:
    - section_header: str (header text)
    - section_text: str (content under the header)
    - header_level: int (1-5)
    - metadata: dict containing:
        - token_count: int
        - model_version: str
        - normalized: bool
        - error: str | None
        - original_content: dict | None
        - parents: dict (e.g., {"h1": "Introduction"})
        - siblings: list[str]
        - source: str (original filename)

Example Usage:
    # Process all .txt and .md files in data/ directory
    python -m src.scripts.chunk_documents

    # Process files in custom directory
    python -m src.scripts.chunk_documents --filepath /path/to/files

    # Process single file with custom output location
    python -m src.scripts.chunk_documents \
        --filepath data --filename doc.txt \
        --output data/output

    # With poetry
    poetry run python -m src.scripts.chunk_documents

Typical Pipeline:
    1. chunk_documents.py -> Creates {filename}_chunks.jsonl
    2. generate_triplets.py -> Processes chunks to create training triplets
"""

import argparse
import json
import sys
from pathlib import Path
from textwrap import dedent
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from src.services.splitter import MarkdownSplitter
from src.settings import logger


def section_to_dict(section: Any, source_filename: str) -> Dict[str, Any]:
    """Convert a Section object to a dictionary for JSON serialization.

    Args:
        section: Section object containing document section data and metadata.
        source_filename: Name of the source file being processed.

    Returns:
        dict: Serialized section data with the following structure:
            - section_header: Header text of the section
            - section_text: Content text of the section
            - header_level: Hierarchical level of the header
            - metadata: Dictionary containing:
                - token_count: Number of tokens in the section
                - model_version: Version of the model used
                - normalized: Whether content was normalized
                - error: Any error message, if applicable
                - original_content: Original section data before processing
                - parents: Parent section references
                - siblings: Sibling section references
                - source: Name of the source file

    Examples:
        >>> section = Section(...)
        >>> result = section_to_dict(section, "document.pdf")
        >>> result["metadata"]["source"]
        'document.pdf'
    """
    return {
        "section_header": section.section_header,
        "section_text": section.section_text,
        "header_level": section.header_level,
        "metadata": {
            "token_count": section.metadata.token_count,
            "model_version": section.metadata.model_version,
            "normalized": section.metadata.normalized,
            "error": section.metadata.error,
            "original_content": (
                {
                    "section_header": section.metadata.original_content.section_header,
                    "section_text": section.metadata.original_content.section_text,
                }
                if section.metadata.original_content
                else None
            ),
            "parents": section.metadata.parents,
            "siblings": section.metadata.siblings,
            "source": source_filename,
        },
    }


def discover_text_files(
    source_directory: Path, target_filename: Optional[str] = None
) -> List[Path]:
    """Discover text files (.txt, .md) to process based on input parameters.

    Args:
        source_directory: Directory path containing text files.
        target_filename: Optional specific filename to process.
            If None, all .txt and .md files are discovered.

    Returns:
        list: List of Path objects for files to process. Empty list if no files found.

    Examples:
        >>> discover_text_files(Path("data"), "document.txt")
        [PosixPath('data/document.txt')]
        >>> discover_text_files(Path("data"))
        [PosixPath('data/doc1.txt'), PosixPath('data/doc2.md')]
    """
    if not source_directory.exists():
        logger.error(f"Directory does not exist: {source_directory}")
        return []

    if not source_directory.is_dir():
        logger.error(f"Path is not a directory: {source_directory}")
        return []

    if target_filename:
        # Process specific file
        target_file_path = source_directory / target_filename
        if not target_file_path.exists():
            logger.error(f"File does not exist: {target_file_path}")
            return []
        logger.info(f"Found target file: {target_filename}")
        return [target_file_path]

    # Discover all .txt and .md files
    txt_files = list(source_directory.glob("*.txt"))
    md_files = list(source_directory.glob("*.md"))
    discovered_text_files = sorted(txt_files + md_files)

    if not discovered_text_files:
        logger.warning(f"No .txt or .md files found in directory: {source_directory}")
        return []

    logger.info(f"Discovered {len(discovered_text_files)} text file(s) in {source_directory}")
    return discovered_text_files


def process_text_file(source_file_path: Path, output_directory: Path) -> bool:
    """Process a single text file and save chunked sections to JSON.

    Args:
        source_file_path: Path to the text file to process.
        output_directory: Directory where output JSON files will be saved.

    Returns:
        bool: True if processing succeeded, False otherwise.
    """
    source_filename = source_file_path.name
    logger.info(f"Processing: {source_filename}")

    try:
        # Use MarkdownSplitter.from_file to chunk the file based on headers
        document_sections = MarkdownSplitter.from_file(source_file_path)

        # Convert sections to dictionaries with source metadata
        # Filter: Only include sections with non-empty section_text to avoid
        # storing headers without content (common in sparse documents)
        serialized_sections = [
            section_to_dict(section, source_filename)
            for section in document_sections
            if section.section_text and section.section_text.strip()
        ]

        # Create output filename: {original_name}_chunks.jsonl
        output_file_path = output_directory / f"{source_file_path.stem}_chunks.jsonl"

        # Write to JSONL format (one JSON object per line)
        # JSONL allows for efficient streaming in downstream processing
        with open(output_file_path, "w", encoding="utf-8") as jsonl_file:
            for section in serialized_sections:
                jsonl_file.write(json.dumps(section, ensure_ascii=False) + "\n")

        logger.info(
            f"Successfully saved {len(document_sections)} sections to {output_file_path}"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to process {source_filename}: {e}", exc_info=True)
        return False


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments containing:
            - filepath: Directory path containing text files
            - filename: Optional specific file to process
            - output: Output directory for chunked JSON files
    """
    parser = argparse.ArgumentParser(
        description=(
            "Chunk text documents (.txt, .md) using MarkdownSplitter and save as JSONL."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent(
            """
            Examples:
              # Process all .txt and .md files in the default 'data' directory
              python -m src.scripts.chunk_documents

              # Process all text files in a custom directory
              python -m src.scripts.chunk_documents --filepath /path/to/files

              # Process a specific file with custom output directory
              python -m src.scripts.chunk_documents --filepath data --filename document.txt \
              --output data/output

              # With poetry
              poetry run python -m src.scripts.chunk_documents
        """
        ),
    )

    parser.add_argument(
        "--filepath",
        type=str,
        default="data",
        help="Directory containing text files to process (default: data)",
    )

    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help=(
            "Optional specific text file to process. "
            "If not provided, all .txt and .md files in filepath will be processed."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/chunks",
        help="Output directory for chunked JSONL files (default: data/chunks)",
    )

    return parser.parse_args()


def main() -> int:
    """Main execution function.

    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
    # Parse command-line arguments
    args = parse_arguments()

    source_directory = Path(args.filepath)
    target_filename = args.filename
    output_directory = Path(args.output)

    logger.info("Starting document chunking process")
    logger.info(f"Source directory: {source_directory}")
    logger.info(f"Output directory: {output_directory}")

    # Create output directory
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create output directory {output_directory}: {e}")
        return 1

    # Discover files to process
    text_files_to_process = discover_text_files(source_directory, target_filename)

    if not text_files_to_process:
        logger.error("No files to process. Exiting.")
        return 1

    # Process each file
    successful_count = 0
    failed_count = 0

    for text_file_path in text_files_to_process:
        processing_succeeded = process_text_file(text_file_path, output_directory)
        if processing_succeeded:
            successful_count += 1
        else:
            failed_count += 1

    # Summary
    logger.info(f"Processing complete: {successful_count} successful, {failed_count} failed")

    if failed_count > 0:
        logger.warning(f"{failed_count} file(s) failed to process")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
