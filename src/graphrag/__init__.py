"""GraphRAG over the SRD: build the knowledge graph, store it, and retrieve from it."""

from functools import lru_cache
from pathlib import Path

from google import genai

from src.graphrag.graph import build_graph
from src.graphrag.retrieval import GraphRetriever
from src.graphrag.vertex import embed_query

DEFAULT_MARKDOWN = Path(__file__).parents[2] / "srd_extraction.md"


def create_retriever(
    client: genai.Client, markdown_path: Path = DEFAULT_MARKDOWN, from_neo4j: bool = False
) -> GraphRetriever:
    """Create a retriever over the SRD graph.

    Args:
        client: GenAI client used to embed questions.
        markdown_path: SRD Markdown to build the graph from when not loading from Neo4j.
        from_neo4j: Load the graph and its node embeddings from Neo4j, which enables
            hybrid seeds. Otherwise the graph is built in memory and seeds use BM25 only.
    """
    if not from_neo4j:
        return GraphRetriever(build_graph(markdown_path))

    # Imported here so the in-memory chatbot runs without Neo4j settings.
    from src.graphrag.store import connect
    from src.graphrag.store import load_graph

    with connect() as driver:
        graph = load_graph(driver)

    return GraphRetriever(graph, lru_cache(maxsize=4096)(lambda q: embed_query(client, q)))
