# Roll-to-Quest

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)
![Poetry](https://img.shields.io/badge/poetry-2.0.1-blue)
![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)

A playground for D&D-related applications and custom scripts. Built for personal use, but the utilities may be useful for other projects.

Part of a document processing toolchain with [Gemini Scribe](https://github.com/Gal-Gilor/gemini-scribe) (PDF to Markdown) and [Markdown MCP](https://github.com/Gal-Gilor/markdown-mcp) (MCP server for Markdown chunking).

## What's Inside

Tools for creating D&D 5e training datasets for embedding models. The process can be adapted to other domains.

- **Document Chunking**: Split Markdown documents into semantic sections based on header hierarchy (H1-H5), preserving parent-child relationships and sibling context
- **Pair Generation**: Create anchor-positive training pairs from chunks using Google Gemini, where anchors are natural language queries and positives are the source text
- **Cloud Storage**: Async utilities for reading and writing to Google Cloud Storage

## Quick Start

### Prerequisites

- Python 3.12+
- Poetry for dependency management
- Google Cloud Project (for Gemini API)
- Pinecone account (for vector storage)

### Installation

```bash
# Clone the repository
git clone https://github.com/Gal-Gilor/roll-to-quest.git
cd roll-to-quest

# Install dependencies
poetry install
```

### Environment Configuration

Copy the provided `.env.example` template and configure it with your credentials:

```bash
cp .env.example .env
```

Then edit `.env` with the following variables:

```env
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_GENAI_USE_VERTEXAI=True
GOOGLE_CLOUD_BUCKET=your-bucket-name
GENERATION_MODEL=gemini-2.5-flash
EMBEDDING_MODEL=gemini-embedding-001
PINECONE_API_KEY=your-pinecone-key
PINECONE_INDEX_HOST=your-index-host
```

## Usage

### 1. Chunk Your Documents

Process Markdown or text files into semantic sections:

```bash
# Process all .txt and .md files in data/ directory
poetry run python -m src.scripts.chunk_documents

# Process a specific file
poetry run python -m src.scripts.chunk_documents --filepath data --filename my_document.md
```

Output: JSONL files with headers, content, hierarchy metadata, and parent/sibling relationships.

### 2. Generate Training Pairs

Create anchor-positive pairs from chunked documents:

```bash
# Generate pairs from chunked documents
poetry run python -m src.scripts.generate_pairs document_chunks.jsonl

# Process a specific range of lines (useful for large files)
poetry run python -m src.scripts.generate_pairs document_chunks.jsonl --start-line 1 --end-line 100
```

Output: JSONL file with anchor-positive pairs for model training.

## Development

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src

# Run specific test file
poetry run pytest tests/test_splitters.py
```

### Code Quality

```bash
# Lint and auto-fix
poetry run ruff check src/ tests/ --fix

# Format code
poetry run ruff format src/ tests/
```

## Architecture

Pipeline:

```
PDF Source Materials (use Gemini Scribe for conversion)
    ↓
Markdown Documents (.txt, .md)
    ↓
chunk_documents.py → Semantic sections with metadata
    ↓
generate_pairs.py → Anchor-positive pairs (JSONL)
```

### Key Components

- `src/settings.py` - Configuration using pydantic-settings
- `src/services/splitter.py` - Markdown splitter with hierarchy tracking
- `src/pair_generation/` - Async pair generation using Gemini API
- `src/services/cloud_storage.py` - Async Google Cloud Storage wrapper
- `src/templates/` - Jinja2 templates for prompts

## Use Cases

These utilities work with any Markdown content:

- Processing documentation with header hierarchies
- Creating training data for embedding models
- Building document navigation systems
- Splitting content for RAG applications

## Related Projects

### [Gemini Scribe](https://github.com/Gal-Gilor/gemini-scribe)

FastAPI service for converting PDFs to Markdown using Google Gemini. Use it as a preprocessing step:

```
PDF Source Materials
    ↓
Gemini Scribe → Clean Markdown files
    ↓
Roll-to-Quest → Chunked sections → Anchor-positive pairs
```

Features: PDF to Markdown conversion, Google Cloud Storage integration, async processing, Docker support, Cloud Run ready.

### [Markdown MCP](https://github.com/Gal-Gilor/markdown-mcp)

MCP server that gives LLMs tools to chunk Markdown into hierarchical sections.

MCP (Model Context Protocol) lets LLMs access external tools and data sources. This server implements the protocol for document processing.

Features: Hierarchical splitting (H1-H5), parent-child relationships, sibling tracking, code-aware parsing, Pydantic models, FastAPI async server.

Roll-to-Quest provides the same chunking as a Python library. Markdown MCP exposes it via MCP protocol for LLM integration.

## Contributing

Personal project. Fork and adapt as needed. Issues and PRs welcome.

## Disclaimer

Provided as-is for personal use. Adapt at your own discretion.