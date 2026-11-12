"""Turn graph nodes into passages for an anchor-positive (MNLR) dataset.

A passage is the positive of a training pair. Long nodes are split so a passage fits an
embedding model's input, and near-identical passages are dropped: MNLR treats every
other positive in a batch as a negative, so two copies of the same text would be
trained apart.
"""

import re
from collections import Counter
from collections import defaultdict

import networkx as nx
from pydantic import BaseModel

MAX_PASSAGE_CHARS = 1600  # About 400 tokens
MIN_PASSAGE_WORDS = 8
SHINGLE_SIZE = 5
DUPLICATE_OVERLAP = 0.8  # Share of the shorter passage's shingles found in the other

_ENTITY_PREFIX = re.compile(r"^Level \d+: ")
_ENTITY_SUFFIX = re.compile(r"\s*(?:\[[^\]]*\]|\([^)]*\))$")


class Passage(BaseModel):
    """One positive, with the node it came from."""

    passage_id: str
    node_id: str
    label: str
    entity: str
    breadcrumb: str
    text: str

    @property
    def positive(self) -> str:
        """The passage under its breadcrumb, the same shape the node embeddings use."""
        return f"{self.breadcrumb}\n\n{self.text}"


def entity_name(header: str) -> str:
    """Return the name a question would use: "Level 14: Overchannel" -> "Overchannel"."""
    return _ENTITY_SUFFIX.sub("", _ENTITY_PREFIX.sub("", header)).strip()


def _pack(pieces: list[str], separator: str, limit: int) -> list[str]:
    """Join consecutive pieces into chunks of at most `limit` characters."""
    chunks, current = [], ""
    for piece in pieces:
        candidate = f"{current}{separator}{piece}" if current else piece
        if len(candidate) <= limit or not current:
            current = candidate
        else:
            chunks.append(current)
            current = piece
    if current:
        chunks.append(current)

    return chunks


def split_text(text: str, limit: int = MAX_PASSAGE_CHARS) -> list[str]:
    """Split text into chunks of about `limit` characters, on paragraph boundaries.

    A paragraph longer than the limit is split on line boundaries. When it is a table,
    every chunk repeats the header rows, so the columns stay identifiable. A single
    line longer than the limit is kept whole.
    """
    paragraphs = []
    for paragraph in re.split(r"\n\s*\n", text.strip()):
        lines = paragraph.split("\n")
        if len(paragraph) <= limit:
            paragraphs.append(paragraph)
        elif len(lines) > 2 and all(line.lstrip().startswith("|") for line in lines):
            header = "\n".join(lines[:2])
            rows = _pack(lines[2:], "\n", limit - len(header) - 1)
            paragraphs.extend(f"{header}\n{chunk}" for chunk in rows)
        else:
            paragraphs.extend(_pack(lines, "\n", limit))

    return _pack(paragraphs, "\n\n", limit)


def _shingles(text: str) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    if len(words) <= SHINGLE_SIZE:
        return {tuple(words)}

    return {tuple(words[i : i + SHINGLE_SIZE]) for i in range(len(words) - SHINGLE_SIZE + 1)}


def drop_near_duplicates(passages: list[Passage]) -> list[Passage]:
    """Keep the first passage of each group of near-identical passages."""
    kept: list[Passage] = []
    kept_sizes: list[int] = []
    kept_by_shingle: dict[tuple[str, ...], list[int]] = defaultdict(list)

    for passage in passages:
        shingles = _shingles(passage.text)
        shared = Counter(i for shingle in shingles for i in kept_by_shingle.get(shingle, ()))
        if any(
            count / min(len(shingles), kept_sizes[i]) >= DUPLICATE_OVERLAP
            for i, count in shared.items()
        ):
            continue

        for shingle in shingles:
            kept_by_shingle[shingle].append(len(kept))
        kept.append(passage)
        kept_sizes.append(len(shingles))

    return kept


def extract_passages(graph: nx.DiGraph) -> list[Passage]:
    """Return the deduplicated passages of every node that has text, in graph order."""
    passages = []
    for node, data in graph.nodes(data=True):
        chunks = [
            chunk
            for chunk in split_text(data["content"])
            if len(chunk.split()) >= MIN_PASSAGE_WORDS
        ]
        for index, chunk in enumerate(chunks):
            passages.append(
                Passage(
                    passage_id=f"{node}-{index}",
                    node_id=node,
                    label=data["label"],
                    entity=entity_name(data["header"]),
                    breadcrumb=data["breadcrumb"],
                    text=chunk,
                )
            )

    return drop_near_duplicates(passages)
