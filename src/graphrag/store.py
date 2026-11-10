"""Save the knowledge graph to Neo4j and load it back.

Every node gets the shared label `SrdNode` plus its entity label (`Spell`, `Monster`),
and every edge becomes a relationship of its type (`CAST_BY`, `CONTAINS`). Embeddings
are stored on the nodes, so they are computed once and not on every chatbot start.

Connection settings come from the environment or `.env`:
    NEO4J_URI (default bolt://localhost:7687), NEO4J_USERNAME, NEO4J_PASSWORD
"""

import re
from collections import defaultdict

import networkx as nx
from neo4j import Driver
from neo4j import GraphDatabase
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict

NODE_LABEL = "SrdNode"
BATCH_SIZE = 500
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class Neo4jSettings(BaseSettings):
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USERNAME: str = "neo4j"
    NEO4J_PASSWORD: str

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


def connect() -> Driver:
    """Open a Neo4j driver from the environment settings. Close it when done."""
    settings = Neo4jSettings()

    return GraphDatabase.driver(
        settings.NEO4J_URI, auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
    )


def _identifier(name: str) -> str:
    """Return a label or relationship type that is safe to put in a Cypher query.

    Cypher can't take labels or relationship types as parameters.
    """
    if not _IDENTIFIER.fullmatch(name):
        raise ValueError(f"Not a valid Neo4j label or relationship type: {name!r}")

    return name


def _batches(rows: list[dict]):
    for start in range(0, len(rows), BATCH_SIZE):
        yield rows[start : start + BATCH_SIZE]


def save_graph(graph: nx.DiGraph, driver: Driver, node_label: str = NODE_LABEL) -> None:
    """Replace the graph stored under `node_label` in Neo4j with this one.

    Existing nodes with that label and their relationships are deleted first, so
    sections removed from the Markdown don't linger. Other data in the database is
    untouched.
    """
    node_label = _identifier(node_label)
    nodes_by_label: dict[str, list[dict]] = defaultdict(list)
    for node, data in graph.nodes(data=True):
        properties = {key: value for key, value in data.items() if key != "label"}
        nodes_by_label[_identifier(data["label"])].append({"id": node, **properties})

    edges_by_type: dict[str, list[dict]] = defaultdict(list)
    for source, target, data in graph.edges(data=True):
        edges_by_type[_identifier(data["type"])].append({"source": source, "target": target})

    with driver.session() as session:
        session.run(
            f"CREATE CONSTRAINT {node_label}_id IF NOT EXISTS "
            f"FOR (n:{node_label}) REQUIRE n.id IS UNIQUE"
        )
        session.run(
            f"MATCH (n:{node_label}) CALL (n) {{ DETACH DELETE n }} "
            f"IN TRANSACTIONS OF 1000 ROWS"
        )
        for label, rows in nodes_by_label.items():
            for batch in _batches(rows):
                session.run(
                    f"UNWIND $rows AS row CREATE (n:{node_label}:{label}) SET n = row",
                    rows=batch,
                )
        for edge_type, rows in edges_by_type.items():
            for batch in _batches(rows):
                session.run(
                    f"UNWIND $rows AS row "
                    f"MATCH (a:{node_label} {{id: row.source}}) "
                    f"MATCH (b:{node_label} {{id: row.target}}) "
                    f"CREATE (a)-[:{edge_type}]->(b)",
                    rows=batch,
                )


def load_graph(driver: Driver, node_label: str = NODE_LABEL) -> nx.DiGraph:
    """Read the graph stored under `node_label`, in the shape `build_graph` returns.

    Nodes that were saved with embeddings have an `embedding` attribute.
    """
    node_label = _identifier(node_label)
    graph = nx.DiGraph()
    with driver.session() as session:
        for record in session.run(
            f"MATCH (n:{node_label}) RETURN properties(n) AS properties, labels(n) AS labels"
        ):
            properties = dict(record["properties"])
            label = next(label for label in record["labels"] if label != node_label)
            graph.add_node(properties.pop("id"), label=label, **properties)

        for record in session.run(
            f"MATCH (a:{node_label})-[r]->(b:{node_label}) "
            f"RETURN a.id AS source, type(r) AS type, b.id AS target"
        ):
            graph.add_edge(record["source"], record["target"], type=record["type"])

    return graph
