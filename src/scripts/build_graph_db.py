"""Build the SRD knowledge graph and load it into Neo4j.

Pipeline:
    1. Split the SRD Markdown into sections with `markdown-chunkify`.
    2. Label the sections as D&D entities and link them (see `src.graphrag.graph`).
    3. Embed each node's breadcrumb and text with Gemini on Vertex AI.
    4. Replace the SRD graph in Neo4j with the new nodes, relationships and embeddings.

The run is repeatable: it deletes the previous `SrdNode` graph before writing.

Dependencies:
    - A running Neo4j, with NEO4J_URI, NEO4J_USERNAME and NEO4J_PASSWORD in the
      environment or `.env`. For a local one:
        docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/<password> neo4j:5
    - Application Default Credentials for Vertex AI, unless --no-embeddings is set.

Example Usage:
    python -m src.scripts.build_graph_db
    python -m src.scripts.build_graph_db --no-embeddings
"""

import argparse
from collections import Counter
from pathlib import Path

from src.graphrag import DEFAULT_MARKDOWN
from src.graphrag.graph import build_graph
from src.graphrag.store import connect
from src.graphrag.store import save_graph
from src.graphrag.vertex import create_client
from src.graphrag.vertex import embed_documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Load the SRD knowledge graph into Neo4j.")
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip embedding. The chatbot then picks seeds with BM25 only.",
    )
    args = parser.parse_args()

    graph = build_graph(args.markdown)
    print(f"Built {graph.number_of_nodes():,} nodes and {graph.number_of_edges():,} edges.")

    if not args.no_embeddings:
        # Enum nodes like DamageType "Fire" have no text to embed.
        nodes = [node for node, data in graph.nodes(data=True) if data["content"]]
        texts = [
            f"{graph.nodes[node]['breadcrumb']}\n\n{graph.nodes[node]['content']}"
            for node in nodes
        ]
        for node, embedding in zip(nodes, embed_documents(create_client(), texts)):
            graph.nodes[node]["embedding"] = embedding

    with connect() as driver:
        save_graph(graph, driver)

    labels = Counter(data["label"] for _, data in graph.nodes(data=True))
    print("Saved to Neo4j. Nodes by label:")
    for label, count in labels.most_common():
        print(f"  {label}: {count:,}")


if __name__ == "__main__":
    main()
