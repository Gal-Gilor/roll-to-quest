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

### 3. Chat with the SRD

A Gradio chatbot that answers rules questions with GraphRAG. On startup it builds a knowledge graph of `srd_extraction.md` in memory, which takes a few seconds. There is no graph database and no LLM call to build the graph.

The graph is built in `src/graphrag/graph.py`:

1. [markdown-chunkify](https://github.com/Gal-Gilor/markdown-chunkify) splits the SRD into sections.
2. Each section gets an entity label from its place in the document: a section under "Spell Descriptions" is a Spell, a stat block under "Monsters A-Z" is a Monster. An entity absorbs its subsections, so a Monster node holds the whole stat block.
3. Damage types, schools of magic, sizes and the other enums in `src/extraction/enums.py` become nodes of their own.
4. Rules read the fixed-format lines of each entity. A spell's first line gives `BELONGS_TO_SCHOOL` and `CAST_BY`, a monster's Immunities line gives `IMMUNE_TO_DAMAGE` and `IMMUNE_TO_CONDITION`.
5. A node whose text names another node gets an edge typed by the two labels, such as Monster `CAN_CAST` Spell, or `REFERENCES` when the pair has no specific meaning.

Labels and edge types follow `src/extraction/entities.json` and `relationships.json`. The rules are heuristics, so a small share of edges is wrong.

For each question, seed nodes are picked, their graph neighbors are ranked and added, and Gemini answers from the nodes' text and relationships. The retrieved nodes are listed under "Sources" in each reply.

Seeds are picked in one of two ways:

- In memory (default): BM25 over each node's header, breadcrumb and text.
- With `--neo4j`: hybrid. The BM25 ranking is fused with a cosine-similarity ranking over `gemini-embedding-001` embeddings by reciprocal rank fusion. BM25 finds exact names like "Fireball", embeddings find paraphrases like "paralyze" for "Paralyzed".

The chatbot calls `gemini-3.5-flash-lite` on Vertex AI in project `agent-dnd`, location `us-central1`. These are constants at the top of `src/graphrag/vertex.py`. Authenticate with Application Default Credentials:

```bash
gcloud auth application-default login
poetry run python -m src.scripts.chat
```

### 4. Store the Graph in Neo4j

`build_graph_db` builds the graph, embeds each node with `gemini-embedding-001` (one Vertex AI request per node, about 2,300) and writes nodes, relationships and embeddings to Neo4j. Every node has the label `SrdNode` plus its entity label, and edges become relationships of their type. A rerun replaces the previous `SrdNode` graph.

```bash
# A local Neo4j
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/<password> neo4j:5

# NEO4J_URI, NEO4J_USERNAME and NEO4J_PASSWORD come from the environment or .env
poetry run python -m src.scripts.build_graph_db

# The chatbot then loads the graph from Neo4j and uses hybrid seeds
poetry run python -m src.scripts.chat --neo4j
```

The graph can also be queried directly:

```cypher
MATCH (m:Monster)-[:IMMUNE_TO_DAMAGE]->(:DamageType {header: "Fire"}) RETURN m.header
```

### 5. Measure Retrieval

`evaluate_retrieval` scores seed selection on the [anchor-positive pairs dataset](https://huggingface.co/datasets/galgilor/dnd-5e-srd-anchor-positive-pairs). The dataset was generated from a different chunking, so each passage is first matched to the graph nodes that contain it. Pairs whose passage matches no node are skipped.

```bash
# BM25 on the in-memory graph, 2,000 sampled pairs
poetry run python -m src.scripts.evaluate_retrieval

# BM25, dense and hybrid on the Neo4j graph. Embeds each sampled question once.
poetry run python -m src.scripts.evaluate_retrieval --neo4j --sample 1000
```

BM25 on all pairs (`--sample 0`), 10,881 of 11,267 matched to a node:

| Mode | recall@1 | recall@4 | recall@10 | MRR | Context recall |
|------|----------|----------|-----------|-----|----------------|
| bm25 | 0.471 | 0.700 | 0.801 | 0.579 | 0.730 |

Context recall is the share of questions whose answer node is among the 4 seeds plus 8 graph neighbors sent to Gemini. Many questions in the dataset don't name their subject ("What level is this enchantment spell?"), which caps the score for any retriever.

### 6. Build an MNLR Dataset from the Graph

A question-passage dataset for fine-tuning an embedder with MultipleNegativesRankingLoss, built from the graph nodes.

```bash
# 1. Split nodes into passages and write batch files to data/mnlr/batches/
poetry run python -m src.scripts.export_passages

# 2. Generate questions for each batch with a model, following
#    src/templates/generate_mnlr_questions.md, into data/mnlr/generated/

# 2b. Optional second pass for harder questions on named entities
poetry run python -m src.scripts.export_passages --entities-only --batch-size 40
#     Generate from data/mnlr/hard_batches/ with
#     src/templates/generate_mnlr_hard_questions.md into data/mnlr/generated/hard_*.jsonl

# 3. Apply the scripted checks and write data/mnlr/validated.jsonl. Questions BM25
#    ranks outside the top 4 are also written to data/mnlr/to_judge/
poetry run python -m src.scripts.validate_questions

# 4. Judge the hard questions with a stronger model, following
#    src/templates/judge_mnlr_pairs.md, into data/mnlr/judged/

# 5. Keep easy pairs and accepted hard pairs, split by node into
#    data/mnlr/dataset/{train,validation,test}.jsonl
poetry run python -m src.scripts.assemble_mnlr_dataset

# 6. Score retrieval on the new test split
poetry run python -m src.scripts.evaluate_retrieval --pairs data/mnlr/dataset/test.jsonl --sample 0
```

Each passage gets a `lookup`, a `paraphrase` and a `descriptive` question, plus a `table` question when it contains a table. The scripted checks drop questions that copy the passage's wording or refer to "this spell" instead of naming it. A question that BM25 gets wrong is either a good hard question or an ambiguous one, so a judge decides with both passages in view, and only accepted ones are kept. The dataset card is `src/mnlr/DATASET_CARD.md`.

`notebooks/srd_graph.ipynb` lists every relationship in the graph and draws it: the full graph colored by entity label, and the neighborhood of a single entity with labeled edges.

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
- `src/graphrag/` - Knowledge graph and graph retrieval for the chatbot
- `src/scripts/chat.py` - Gradio chatbot

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