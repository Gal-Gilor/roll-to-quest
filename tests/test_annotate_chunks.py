import pytest

from src.scripts.annotate_chunks import annotate_chunk
from src.scripts.annotate_chunks import load_annotator
from src.toc.models import ChunkStrategy
from src.toc.models import NodeType
from src.toc.models import TocDocument
from src.toc.models import TocNode


@pytest.fixture
def fixture_doc():
    trinkets = TocNode(
        id="character-creation/trinkets",
        title="Trinkets",
        node_type=NodeType.SECTION,
        entity_type="Trinket",
        chunk_strategy=ChunkStrategy.ROW,
        page=26,
    )
    char_creation = TocNode(
        id="character-creation",
        title="Character Creation",
        node_type=NodeType.CHAPTER,
        chunk_strategy=ChunkStrategy.SECTION,
        page=19,
        children=[trinkets],
    )
    spell_descriptions = TocNode(
        id="spells/spell-descriptions",
        title="Spell Descriptions",
        node_type=NodeType.SECTION,
        entity_type="Spell",
        chunk_strategy=ChunkStrategy.ENTRY,
        page=107,
    )
    spells = TocNode(
        id="spells",
        title="Spells",
        node_type=NodeType.CHAPTER,
        page=104,
        children=[spell_descriptions],
    )
    return TocDocument(document="Test SRD", sections=[char_creation, spells])


def test_annotate_chunk_adds_entity_type(fixture_doc):
    chunk = {"section_header": "Trinkets", "section_text": "A white glove"}
    result = annotate_chunk(chunk, fixture_doc)
    assert result["entity_type"] == "Trinket"


def test_annotate_chunk_adds_context_path(fixture_doc):
    chunk = {"section_header": "Trinkets", "section_text": "A white glove"}
    result = annotate_chunk(chunk, fixture_doc)
    assert result["context_path"] == "character-creation/trinkets"


def test_annotate_chunk_adds_chunk_strategy(fixture_doc):
    chunk = {"section_header": "Trinkets", "section_text": "A white glove"}
    result = annotate_chunk(chunk, fixture_doc)
    assert result["chunk_strategy"] == "row"


def test_annotate_chunk_unknown_header_returns_unchanged(fixture_doc):
    chunk = {"section_header": "Unknown Section", "section_text": "text"}
    result = annotate_chunk(chunk, fixture_doc)
    assert result == chunk


def test_annotate_chunk_no_entity_type_when_unset(fixture_doc):
    # Character Creation has chunk_strategy=section but no entity_type
    chunk = {"section_header": "Character Creation", "section_text": "..."}
    result = annotate_chunk(chunk, fixture_doc)
    assert result.get("entity_type") is None
    assert result["chunk_strategy"] == "section"


def test_annotate_chunk_does_not_mutate_original(fixture_doc):
    original = {"section_header": "Trinkets", "section_text": "A white glove"}
    chunk = dict(original)
    annotate_chunk(chunk, fixture_doc)
    assert chunk == original


def test_annotate_chunk_spell_descriptions(fixture_doc):
    chunk = {"section_header": "Spell Descriptions", "section_text": "..."}
    result = annotate_chunk(chunk, fixture_doc)
    assert result["entity_type"] == "Spell"
    assert result["context_path"] == "spells/spell-descriptions"
    assert result["chunk_strategy"] == "entry"


def test_load_annotator_returns_document(tmp_path):
    yaml_content = (
        'document: "Test"\n'
        "sections:\n"
        "  - id: spells\n"
        "    title: Spells\n"
        "    node_type: chapter\n"
        "    chunk_strategy: section\n"
    )
    toc_file = tmp_path / "toc.yaml"
    toc_file.write_text(yaml_content)
    doc = load_annotator(toc_file)
    assert doc.document == "Test"
