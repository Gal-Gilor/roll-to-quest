"""Build a knowledge graph of a Markdown document from its section structure.

Nodes are the sections returned by `markdown-chunkify`. Edges are:
    - CONTAINS: a section to each of its direct subsections.
    - MENTIONS: a section to another section whose header appears in its text.

The graph is small enough to rebuild in memory on startup, so nothing is persisted.
"""

import re
from collections import Counter
from pathlib import Path

import networkx as nx
from markdown_chunkify import MarkdownSplitter
from markdown_chunkify import Section

CONTAINS = "CONTAINS"
MENTIONS = "MENTIONS"

MIN_ALIAS_LENGTH = 4
_TAG_SUFFIX = re.compile(r"\s*\[[^\]]*\]$")  # "Hide [Action]" -> "Hide"


def _alias(header: str) -> str:
    """Return the name other sections use to refer to a header."""
    return _TAG_SUFFIX.sub("", header).strip()


def _mention_targets(sections: list[Section]) -> dict[str, int]:
    """Map each unambiguous alias to its section id.

    An alias shared by several sections ("Actions", "Traits") can't be resolved to one
    node, so it is dropped. Short or lowercase aliases match too much prose.
    """
    counts = Counter(_alias(section.header) for section in sections)

    return {
        alias: section.metadata.id
        for section in sections
        if counts[alias := _alias(section.header)] == 1
        and len(alias) >= MIN_ALIAS_LENGTH
        and alias[0].isupper()
    }


def build_graph(markdown_path: str | Path) -> nx.DiGraph:
    """Split a Markdown file into sections and link them into a directed graph.

    Args:
        markdown_path: Path to the Markdown file.

    Returns:
        A graph whose nodes are section ids with `header`, `breadcrumb`, `category`
        (the top-level header), `level` and `content` attributes. Each edge has a
        `type` attribute, either CONTAINS or MENTIONS.
    """
    sections = MarkdownSplitter().split_file(markdown_path)
    graph = nx.DiGraph()

    for section in sections:
        graph.add_node(
            section.metadata.id,
            header=section.header,
            breadcrumb=section.breadcrumb(),
            category=section.metadata.parents.get("h1", section.header),
            level=section.level,
            content=section.content,
        )
        if section.metadata.parent_id is not None:
            graph.add_edge(section.metadata.parent_id, section.metadata.id, type=CONTAINS)

    targets = _mention_targets(sections)
    if not targets:
        return graph

    # Longest alias first, so "Attack Roll" wins over "Attack". Matching is case-sensitive
    # because the SRD capitalizes game terms.
    aliases = sorted(targets, key=len, reverse=True)
    pattern = re.compile(r"\b(?:" + "|".join(map(re.escape, aliases)) + r")\b")

    for section in sections:
        source = section.metadata.id
        for alias in set(pattern.findall(section.content)):
            target = targets[alias]
            if target == source or graph.has_edge(source, target):
                continue
            if graph.has_edge(target, source):  # Don't duplicate a CONTAINS link
                continue
            graph.add_edge(source, target, type=MENTIONS)

    return graph
