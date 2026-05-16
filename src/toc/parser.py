from pathlib import Path

import yaml

from src.toc.models import ChunkStrategy
from src.toc.models import TocDocument
from src.toc.models import TocNode


def load_toc(path: Path) -> TocDocument:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return TocDocument.model_validate(data)


def _build_index(
    sections: list[TocNode],
    parent_id: str | None = None,
) -> tuple[dict[str, TocNode], dict[str, str | None]]:
    nodes: dict[str, TocNode] = {}
    parents: dict[str, str | None] = {}
    for node in sections:
        nodes[node.id] = node
        parents[node.id] = parent_id
        if node.children:
            child_nodes, child_parents = _build_index(node.children, node.id)
            nodes.update(child_nodes)
            parents.update(child_parents)
    return nodes, parents


def build_index(
    document: TocDocument,
) -> tuple[dict[str, TocNode], dict[str, str | None]]:
    return _build_index(document.sections)


def find_node_by_id(node_id: str, document: TocDocument) -> TocNode | None:
    nodes, _ = _build_index(document.sections)
    return nodes.get(node_id)


def find_node_by_title(title: str, document: TocDocument) -> TocNode | None:
    nodes, _ = _build_index(document.sections)
    return next((n for n in nodes.values() if n.title == title), None)


def resolve_chunk_strategy(node_id: str, document: TocDocument) -> ChunkStrategy | None:
    nodes, parents = _build_index(document.sections)
    current_id: str | None = node_id
    while current_id is not None:
        node = nodes.get(current_id)
        if node is None:
            break
        if node.chunk_strategy is not None:
            return node.chunk_strategy
        current_id = parents.get(current_id)
    return None


def resolve_entity_type(node_id: str, document: TocDocument) -> str | None:
    nodes, parents = _build_index(document.sections)
    current_id: str | None = node_id
    while current_id is not None:
        node = nodes.get(current_id)
        if node is None:
            break
        if node.entity_type is not None:
            return node.entity_type
        current_id = parents.get(current_id)
    return None
