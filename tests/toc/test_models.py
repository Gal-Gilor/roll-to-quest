import pytest
from pydantic import ValidationError

from src.toc.models import ChunkStrategy
from src.toc.models import NodeType
from src.toc.models import TocDocument
from src.toc.models import TocNode


def test_toc_node_requires_id():
    with pytest.raises(ValidationError):
        TocNode(title="Spells", node_type="chapter")


def test_toc_node_requires_title():
    with pytest.raises(ValidationError):
        TocNode(id="spells", node_type="chapter")


def test_toc_node_requires_node_type():
    with pytest.raises(ValidationError):
        TocNode(id="spells", title="Spells")


def test_toc_node_optional_fields_default_to_none():
    node = TocNode(id="spells", title="Spells", node_type="chapter")
    assert node.chunk_strategy is None
    assert node.entity_type is None
    assert node.page is None
    assert node.children == []


def test_toc_node_with_all_fields():
    node = TocNode(
        id="spells/spell-descriptions",
        title="Spell Descriptions",
        node_type="section",
        chunk_strategy="entry",
        entity_type="Spell",
        page=107,
    )
    assert node.id == "spells/spell-descriptions"
    assert node.chunk_strategy == ChunkStrategy.ENTRY
    assert node.entity_type == "Spell"
    assert node.page == 107


def test_toc_node_with_children():
    child = TocNode(id="spells/acid-arrow", title="Acid Arrow", node_type="entry")
    parent = TocNode(
        id="spells",
        title="Spells",
        node_type="chapter",
        children=[child],
    )
    assert len(parent.children) == 1
    assert parent.children[0].title == "Acid Arrow"


def test_toc_document_valid():
    node = TocNode(id="spells", title="Spells", node_type="chapter")
    doc = TocDocument(document="D&D 5e SRD", sections=[node])
    assert doc.document == "D&D 5e SRD"
    assert len(doc.sections) == 1


def test_toc_document_empty_sections():
    doc = TocDocument(document="D&D 5e SRD", sections=[])
    assert doc.sections == []


def test_node_type_enum():
    assert NodeType.CHAPTER == "chapter"
    assert NodeType.SECTION == "section"
    assert NodeType.SUBSECTION == "subsection"
    assert NodeType.ENTRY == "entry"


def test_chunk_strategy_enum():
    assert ChunkStrategy.ENTRY == "entry"
    assert ChunkStrategy.ROW == "row"
    assert ChunkStrategy.SECTION == "section"


def test_invalid_node_type_rejected():
    with pytest.raises(ValidationError):
        TocNode(id="foo", title="Foo", node_type="invalid_type")


def test_invalid_chunk_strategy_rejected():
    with pytest.raises(ValidationError):
        TocNode(id="foo", title="Foo", node_type="chapter", chunk_strategy="invalid")
