# Roll-to-Quest

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)
![Poetry](https://img.shields.io/badge/poetry-2.0.1-blue)

A playground for experimenting with D&D-related applications and developing custom scripts for personal use. While built primarily for my own D&D projects, this repository contains utilities that other developers may find useful and easily adaptable for their own needs.

This project is part of a suite of tools for document processing and embedding model development, alongside [Gemini Scribe](https://github.com/Gal-Gilor/gemini-scribe) (PDF to Markdown conversion) and [Markdown MCP](https://github.com/Gal-Gilor/markdown-mcp) (production-grade MCP server for Markdown chunking).

## What's Inside

This repository includes tools for processing tabletop RPG source materials into structured, machine-learning-ready formats:

- **Document Chunking**: Split Markdown documents into semantic sections based on header hierarchy, with code-aware parsing that preserves parent-child relationships and sibling headers
- **Triplet Generation**: Generate training triplets (anchor, positive, negative) for fine-tuning embedding models using Google Gemini
- **Async Cloud Storage**: Utilities for working with Google Cloud Storage

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

**Output**: JSONL files with sections containing headers, content, hierarchy metadata, and parent/sibling relationships.

### 2. Generate Training Triplets

Create anchor-positive-negative triplets for embedding model fine-tuning:

```bash
# Generate triplets from chunked documents
poetry run python -m src.scripts.generate_triplets document_chunks.jsonl

# Process a specific range of lines (useful for large files)
poetry run python -m src.scripts.generate_triplets document_chunks.jsonl --start-line 1 --end-line 100
```

**Output**: JSONL file with triplets ready for model training.

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
# Format code
poetry run black src/ tests/

# Lint and auto-fix
poetry run ruff check src/ tests/

# Sort imports
poetry run isort src/ tests/
```

## Architecture

The project follows a simple pipeline architecture:

```
PDF Source Materials (use Gemini Scribe for conversion)
    ↓
Markdown Documents (.txt, .md)
    ↓
chunk_documents.py → Semantic sections with metadata
    ↓
generate_triplets.py → Training triplets (anchor, positive, negative)
    ↓
Fine-tune embedding models
```

### Key Components

- **`src/settings.py`**: Centralized configuration using pydantic-settings
- **`src/services/splitter.py`**: Markdown document splitter with hierarchy tracking
- **`src/triplet_generation/`**: Async triplet generation using Gemini API
- **`src/services/cloud_storage.py`**: Async Google Cloud Storage wrapper
- **`src/templates/`**: Jinja2 templates for prompt engineering

## Use Cases

While designed for D&D content, these utilities are adaptable for:

- Processing any Markdown documentation with header hierarchies
- Creating training data for domain-specific embedding models
- Building hierarchical document navigation systems
- Splitting long-form content for RAG (Retrieval-Augmented Generation) applications

## Related Projects

### [Gemini Scribe](https://github.com/Gal-Gilor/gemini-scribe)

A FastAPI service for converting PDF documents to clean Markdown using Google Gemini. Gemini Scribe works seamlessly with Roll-to-Quest as a preprocessing step:

**Workflow:**
```
PDF Source Materials
    ↓
Gemini Scribe → Clean Markdown files
    ↓
Roll-to-Quest → Chunked sections → Training triplets
```

**Key Features:**
- PDF to Markdown conversion using Gemini
- Google Cloud Storage integration
- High-performance async processing
- Docker containerization for easy deployment
- Google Cloud Run ready

If you're working with PDF source materials (like D&D rulebooks), use Gemini Scribe to convert them to Markdown first, then process them with Roll-to-Quest's chunking and triplet generation utilities.

### [Markdown MCP](https://github.com/Gal-Gilor/markdown-mcp)

A production-grade Model Context Protocol (MCP) server that provides LLMs with tools to chunk Markdown documents into hierarchical sections. This server demonstrates how to extend language model capabilities through the MCP standard.

**What is MCP?**

The Model Context Protocol enables LLMs to securely access external tools and data sources. Markdown MCP implements this protocol to give language models sophisticated document processing capabilities.

**Key Features:**
- Hierarchical Markdown splitting (H1 → H2 → H3 → H4 → H5)
- Preserves parent-child header relationships
- Detects and tracks sibling headers
- Code-aware processing (ignores `#` in code blocks)
- Type-safe Pydantic models
- FastAPI-based high-performance async server

**Use Case:**

While Roll-to-Quest includes standalone Markdown chunking utilities for direct Python use, Markdown MCP packages similar functionality as an MCP server, allowing any MCP-compatible language model to perform document chunking through a standardized protocol interface.

**Comparison:**
- **Roll-to-Quest**: Python library approach - import and use directly in your code
- **Markdown MCP**: Server approach - expose chunking capabilities to LLMs via MCP protocol

## Contributing

This is primarily a personal project, but feel free to fork, modify, and adapt the code for your own needs. If you find bugs or have suggestions, issues and pull requests are welcome!

## Disclaimer

This repository is for personal use and experimentation. The code is provided as-is, and while I've included tests and documentation, it's designed primarily for my own workflow. Adapt and use at your own discretion.