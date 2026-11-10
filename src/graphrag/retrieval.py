"""Retrieve graph sections for a question: lexical seeds, then one hop over the graph."""

import math
import re
from collections import defaultdict

import networkx as nx
from rank_bm25 import BM25Okapi

MAX_SECTION_CHARS = 6000  # Keeps one oversized section from crowding out the rest
HEADER_WEIGHT = 3  # A section named after a query term outranks one that mentions it

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
    """Find the sections that answer a question.

    BM25 over each section's header, breadcrumb and text picks the seed sections. Their graph
    neighbors (parents, subsections, mentioned and mentioning sections) are then ranked
    and the best ones are added, which brings in rules a seed refers to but the
    question doesn't name.
    """

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self._ids = list(graph.nodes)
        self._bm25 = BM25Okapi(
            [
                _tokenize(data["header"]) * HEADER_WEIGHT
                + _tokenize(f"{data['breadcrumb']} {data['content']}")
                for _, data in graph.nodes(data=True)
            ]
        )

    def retrieve(self, query: str, num_seeds: int = 4, num_neighbors: int = 8) -> list[int]:
        """Return section ids relevant to the query, seeds first.

        Args:
            query: The user's question.
            num_seeds: Number of sections picked by lexical match.
            num_neighbors: Number of graph neighbors of the seeds to add.

        Returns:
            Section ids, or an empty list if nothing matches the query.
        """
        scores = dict(zip(self._ids, self._bm25.get_scores(_tokenize(query))))
        ranked = sorted(scores, key=scores.get, reverse=True)
        seeds = [node for node in ranked[:num_seeds] if scores[node] > 0]
        if not seeds:
            return []

        # A link to a rarely linked section says more than a link to a hub like
        # "Attack Roll", so each link is weighted by the neighbor's inverse log degree.
        top_score = scores[seeds[0]]
        link_weights: dict[int, float] = defaultdict(float)
        for seed in seeds:
            for neighbor in nx.all_neighbors(self.graph, seed):
                if neighbor not in seeds:
                    link_weights[neighbor] += 1 / math.log(1 + self.graph.degree(neighbor))

        neighbors = sorted(
            link_weights,
            key=lambda node: link_weights[node] * (1 + scores[node] / top_score),
            reverse=True,
        )

        return seeds + neighbors[:num_neighbors]

    def format_context(self, node_ids: list[int]) -> str:
        """Render sections as Markdown, each under its breadcrumb. Skips empty sections."""
        blocks = []
        for node_id in node_ids:
            data = self.graph.nodes[node_id]
            if data["content"]:
                blocks.append(
                    f"## {data['breadcrumb']}\n{data['content'][:MAX_SECTION_CHARS]}"
                )

        return "\n\n".join(blocks)
