"""Apply the judge's verdicts and split the pairs into train, validation and test."""

import random
from collections import defaultdict


def apply_verdicts(rows: list[dict], verdicts: dict[int, str]) -> list[dict]:
    """Keep the rows the judge accepted. A row without a verdict is dropped."""
    return [row for index, row in enumerate(rows) if verdicts.get(index) == "accept"]


def split_by_node(
    rows: list[dict], validation_share: float = 0.05, test_share: float = 0.1, seed: int = 0
) -> dict[str, list[dict]]:
    """Split rows so that all pairs of a node land in the same split.

    A model is then tested on entities it never saw a question about.
    """
    by_node: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_node[row["node_id"]].append(row)

    nodes = sorted(by_node)
    random.Random(seed).shuffle(nodes)
    num_test = round(len(nodes) * test_share)
    num_validation = round(len(nodes) * validation_share)
    node_splits = {
        "test": nodes[:num_test],
        "validation": nodes[num_test : num_test + num_validation],
        "train": nodes[num_test + num_validation :],
    }

    return {
        split: [row for node in split_nodes for row in by_node[node]]
        for split, split_nodes in node_splits.items()
    }
