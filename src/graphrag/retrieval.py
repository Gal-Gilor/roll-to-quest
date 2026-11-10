"""Retrieve graph nodes for a question: ranked seeds, then one hop over the graph."""

import math
import re
from collections import defaultdict
from collections.abc import Callable
from typing import Literal

import networkx as nx
import numpy as np
from rank_bm25 import BM25Okapi

from src.graphrag.graph import CONTAINS
from src.graphrag.graph import REFERENCES

Mode = Literal["bm25", "dense", "hybrid"]

MAX_SECTION_CHARS = 6000  # Keeps one oversized section from crowding out the rest
HEADER_WEIGHT = 3  # A section named after a query term outranks one that mentions it
RRF_K = 60  # Reciprocal rank fusion constant from the original paper

# Question words would otherwise pick the seeds for short questions.
STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i if in is it its me my of on or "
    "s that the to what when which who with work works you your".split()
)


def _tokenize(text: str) -> list[str]:
    return [
        token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in STOPWORDS
    ]


class GraphRetriever:
    """Find the nodes that answer a question.

    Seeds are picked by BM25 over each node's header, breadcrumb and text. When the
    nodes carry an `embedding` attribute and `embed_query` is given, the BM25 ranking is
    fused with the cosine-similarity ranking (reciprocal rank fusion). BM25 finds exact
    names like "Fireball"; embeddings find paraphrases like "paralyze" for "Paralyzed".

    The seeds' graph neighbors are then ranked and the best ones are added, which
    brings in rules a seed refers to but the question doesn't name.

    Args:
        graph: The knowledge graph from `build_graph` or `load_graph`.
        embed_query: Embeds a question with the model that embedded the nodes.
    """

    def __init__(
        self, graph: nx.DiGraph, embed_query: Callable[[str], list[float]] | None = None
    ):
        self.graph = graph
        self._ids = list(graph.nodes)
        self._bm25 = BM25Okapi(
            [
                _tokenize(data["header"]) * HEADER_WEIGHT
                + _tokenize(f"{data['breadcrumb']} {data['content']}")
                for _, data in graph.nodes(data=True)
            ]
        )

        self._embed_query = embed_query
        self._embedded_ids = [
            node for node in self._ids if graph.nodes[node].get("embedding")
        ]
        self._embeddings = np.array(
            [graph.nodes[node]["embedding"] for node in self._embedded_ids], dtype=np.float32
        )
        if len(self._embeddings):
            self._embeddings /= np.linalg.norm(self._embeddings, axis=1, keepdims=True)

    @property
    def default_mode(self) -> Mode:
        """Hybrid when node embeddings and a query embedder are available, else BM25."""
        return "hybrid" if self._embed_query and len(self._embedded_ids) else "bm25"

    def _bm25_ranking(self, query: str) -> list[str]:
        scores = self._bm25.get_scores(_tokenize(query))
        order = np.argsort(-scores, kind="stable")

        return [self._ids[i] for i in order if scores[i] > 0]

    def _dense_ranking(self, query: str) -> list[str]:
        if not self._embed_query or not len(self._embedded_ids):
            raise ValueError("Dense retrieval needs node embeddings and `embed_query`.")

        similarities = self._embeddings @ np.asarray(self._embed_query(query), np.float32)

        return [self._embedded_ids[i] for i in np.argsort(-similarities, kind="stable")]

    def rank(self, query: str, mode: Mode | None = None) -> dict[str, float]:
        """Score the nodes for a query, best first. Scores are relative to the best one.

        Each ranking gives a node 1 / (RRF_K + rank). Hybrid sums the two, so a node
        ranked well by both beats a node ranked first by one.
        """
        mode = mode or self.default_mode
        rankings = []
        if mode in ("bm25", "hybrid"):
            rankings.append(self._bm25_ranking(query))
        if mode in ("dense", "hybrid"):
            rankings.append(self._dense_ranking(query))

        fused: dict[str, float] = defaultdict(float)
        for ranking in rankings:
            for rank, node in enumerate(ranking, 1):
                fused[node] += 1 / (RRF_K + rank)

        best = max(fused.values(), default=1.0)

        return {
            node: score / best
            for node, score in sorted(fused.items(), key=lambda item: -item[1])
        }

    def retrieve(
        self,
        query: str,
        num_seeds: int = 4,
        num_neighbors: int = 8,
        mode: Mode | None = None,
    ) -> list[str]:
        """Return node ids relevant to the query, seeds first.

        Args:
            query: The user's question.
            num_seeds: Number of top-ranked nodes to start from.
            num_neighbors: Number of graph neighbors of the seeds to add.
            mode: How to rank seeds. Defaults to `default_mode`.

        Returns:
            Node ids, or an empty list if nothing matches the query.
        """
        scores = self.rank(query, mode)
        seeds = list(scores)[:num_seeds]

        # A link to a rarely linked section says more than a link to a hub like
        # "Attack Roll", so each link is weighted by the neighbor's inverse log degree.
        link_weights: dict[str, float] = defaultdict(float)
        for seed in seeds:
            for neighbor in nx.all_neighbors(self.graph, seed):
                if neighbor not in seeds:
                    link_weights[neighbor] += 1 / math.log(1 + self.graph.degree(neighbor))

        neighbors = sorted(
            link_weights,
            key=lambda node: link_weights[node] * (1 + scores.get(node, 0.0)),
            reverse=True,
        )

        return seeds + neighbors[:num_neighbors]

    def _relationships(self, node_id: str) -> str:
        """List a node's typed edges, one line per type. "<-" marks incoming edges."""
        names: dict[str, list[str]] = defaultdict(list)
        for _, target, data in self.graph.out_edges(node_id, data=True):
            names[data["type"]].append(self.graph.nodes[target]["header"])
        for source, _, data in self.graph.in_edges(node_id, data=True):
            names[f"<- {data['type']}"].append(self.graph.nodes[source]["header"])

        return "\n".join(
            f"- {edge_type}: {', '.join(sorted(headers))}"
            for edge_type, headers in sorted(names.items())
            if edge_type.removeprefix("<- ") not in (CONTAINS, REFERENCES)
        )

    def format_context(self, node_ids: list[str]) -> str:
        """Render nodes as Markdown: label and breadcrumb, text, then typed relationships.

        Nodes with neither text nor relationships are skipped.
        """
        blocks = []
        for node_id in node_ids:
            data = self.graph.nodes[node_id]
            parts = [data["content"][:MAX_SECTION_CHARS]]
            if relationships := self._relationships(node_id)[:MAX_SECTION_CHARS]:
                parts.append(f"Relationships:\n{relationships}")
            if body := "\n\n".join(part for part in parts if part):
                blocks.append(f"## [{data['label']}] {data['breadcrumb']}\n{body}")

        return "\n\n".join(blocks)
