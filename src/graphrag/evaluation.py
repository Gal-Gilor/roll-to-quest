"""Score retrieval against question-passage pairs.

The pairs were generated from a different chunking of the SRD, so a passage has no node
id. `GoldIndex` finds the nodes whose text holds the passage; a question counts as
answered when one of them is retrieved.
"""

import re
from collections import Counter
from collections import defaultdict

import networkx as nx

SHINGLE_SIZE = 5
MIN_OVERLAP = 0.6  # Share of a passage's shingles a node must hold to count as its source


def _words(text: str) -> list[str]:
    """Lowercase words without markup, so "**Range:** 90 feet" matches "Range: 90 feet"."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _shingles(words: list[str]) -> set[tuple[str, ...]]:
    if len(words) <= SHINGLE_SIZE:
        return {tuple(words)} if words else set()

    return {tuple(words[i : i + SHINGLE_SIZE]) for i in range(len(words) - SHINGLE_SIZE + 1)}


class GoldIndex:
    """Map a passage to the graph nodes that contain it."""

    def __init__(self, graph: nx.DiGraph):
        self._nodes_by_shingle: dict[tuple[str, ...], set[str]] = defaultdict(set)
        for node, data in graph.nodes(data=True):
            for shingle in _shingles(_words(data["content"])):
                self._nodes_by_shingle[shingle].add(node)

    def find(self, passage: str) -> set[str]:
        """Return the nodes sharing the most word 5-grams with the passage.

        Ties are all returned: a passage like a spell's "Casting Time: Action" header
        block can sit in several nodes. Returns an empty set when no node holds at
        least MIN_OVERLAP of the passage, or the passage is under five words.
        """
        shingles = _shingles(_words(passage))
        if not shingles or len(next(iter(shingles))) < SHINGLE_SIZE:
            return set()

        counts = Counter(
            node for shingle in shingles for node in self._nodes_by_shingle.get(shingle, ())
        )
        if not counts:
            return set()

        best = max(counts.values())
        if best / len(shingles) < MIN_OVERLAP:
            return set()

        return {node for node, count in counts.items() if count == best}


def score(ranked: list[str], context: list[str], gold: set[str], cutoffs: list[int]) -> dict:
    """Score one question.

    Args:
        ranked: Seed candidates in rank order.
        context: The nodes the chatbot would send to the model (seeds and neighbors).
        gold: The nodes that hold the answer passage.
        cutoffs: The k values for recall@k over `ranked`.

    Returns:
        `recall@k` per cutoff, `mrr` over `ranked`, and `context_recall`, each 0 or 1
        except `mrr`.
    """
    first_hit = next((rank for rank, node in enumerate(ranked, 1) if node in gold), None)
    result = {f"recall@{k}": float(first_hit is not None and first_hit <= k) for k in cutoffs}
    result["mrr"] = 1 / first_hit if first_hit else 0.0
    result["context_recall"] = float(any(node in gold for node in context))

    return result
