import textwrap

import pytest

from src.toc.models import ChunkStrategy
from src.toc.models import TocDocument
from src.toc.parser import find_node_by_id
from src.toc.parser import find_node_by_title
from src.toc.parser import load_toc
from src.toc.parser import resolve_chunk_strategy
from src.toc.parser import resolve_entity_type

FIXTURE_YAML = textwrap.dedent("""
    document: "Test SRD"
    sections:
      - id: spells
        title: "Spells"
        node_type: chapter
        chunk_strategy: entry
        children:
          - id: spells/spell-descriptions
            title: "Spell Descriptions"
            node_type: section
            entity_type: Spell
            children:
              - id: spells/spell-descriptions/acid-arrow
                title: "Acid Arrow"
                node_type: entry
      - id: equipment
        title: "Equipment"
        node_type: chapter
        children:
          - id: equipment/weapons
            title: "Weapons"
            node_type: section
            entity_type: Weapon
            chunk_strategy: row
""")


@pytest.fixture
def fixture_doc(tmp_path):
    yaml_file = tmp_path / "toc.yaml"
    yaml_file.write_text(FIXTURE_YAML)
    return load_toc(yaml_file)


def test_load_toc_returns_document(tmp_path):
    yaml_file = tmp_path / "toc.yaml"
    yaml_file.write_text(FIXTURE_YAML)
    doc = load_toc(yaml_file)
    assert isinstance(doc, TocDocument)
    assert doc.document == "Test SRD"
    assert len(doc.sections) == 2


def test_load_toc_parses_nested_children(fixture_doc):
    spells = fixture_doc.sections[0]
    assert len(spells.children) == 1
    assert spells.children[0].id == "spells/spell-descriptions"
    assert len(spells.children[0].children) == 1


def test_find_node_by_id_top_level(fixture_doc):
    node = find_node_by_id("spells", fixture_doc)
    assert node is not None
    assert node.title == "Spells"


def test_find_node_by_id_nested(fixture_doc):
    node = find_node_by_id("spells/spell-descriptions/acid-arrow", fixture_doc)
    assert node is not None
    assert node.title == "Acid Arrow"


def test_find_node_by_id_missing(fixture_doc):
    assert find_node_by_id("nonexistent/id", fixture_doc) is None


def test_find_node_by_title_found(fixture_doc):
    node = find_node_by_title("Spell Descriptions", fixture_doc)
    assert node is not None
    assert node.id == "spells/spell-descriptions"


def test_find_node_by_title_missing(fixture_doc):
    assert find_node_by_title("Nonexistent Section", fixture_doc) is None


def test_resolve_chunk_strategy_direct(fixture_doc):
    strategy = resolve_chunk_strategy("equipment/weapons", fixture_doc)
    assert strategy == ChunkStrategy.ROW


def test_resolve_chunk_strategy_inherited_from_grandparent(fixture_doc):
    # acid-arrow → spell-descriptions (no strategy) → spells (entry)
    strategy = resolve_chunk_strategy("spells/spell-descriptions/acid-arrow", fixture_doc)
    assert strategy == ChunkStrategy.ENTRY


def test_resolve_chunk_strategy_unset_returns_none(fixture_doc):
    # equipment has no chunk_strategy
    strategy = resolve_chunk_strategy("equipment", fixture_doc)
    assert strategy is None


def test_resolve_entity_type_direct(fixture_doc):
    entity = resolve_entity_type("spells/spell-descriptions", fixture_doc)
    assert entity == "Spell"


def test_resolve_entity_type_inherited(fixture_doc):
    # acid-arrow has no entity_type, inherits from spell-descriptions
    entity = resolve_entity_type("spells/spell-descriptions/acid-arrow", fixture_doc)
    assert entity == "Spell"


def test_resolve_entity_type_none_when_unset(fixture_doc):
    entity = resolve_entity_type("spells", fixture_doc)
    assert entity is None


def test_resolve_entity_type_missing_node(fixture_doc):
    entity = resolve_entity_type("nonexistent/id", fixture_doc)
    assert entity is None
