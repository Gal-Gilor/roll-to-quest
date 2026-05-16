# SRD TOC Schema Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert `srd_toc_contents.md` into a validated YAML schema (`data/srd_toc.yaml`) with Pydantic models, a parser with inheritance resolution, a Neo4j structure seeder, and a chunk annotator.

**Architecture:** Seven tasks built TDD: (1) add pyyaml + neo4j dependencies, (2) Pydantic models/enums for the YAML schema, (3) a parser that loads the YAML and resolves `chunk_strategy`/`entity_type` inheritance by walking ancestor chains, (4) the YAML data file itself, (5) Neo4j connection settings, (6) a Neo4j structure-pass seeder creating `:Section` nodes and `CONTAINS` edges, and (7) a chunk annotator that stamps existing chunk JSONL with `entity_type` and `context_path`.

**Tech Stack:** Python 3.12, Pydantic v2, PyYAML, neo4j driver, pytest

---

### Task 1: Add Dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add pyyaml and neo4j**

Run: `poetry add pyyaml neo4j`
Expected: Both packages installed, `pyproject.toml` and `poetry.lock` updated.

- [ ] **Step 2: Verify imports**

Run: `python -c "import yaml; import neo4j; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml poetry.lock
git commit -m "chore: add pyyaml and neo4j dependencies"
```

---

### Task 2: TOC Schema Models

**Files:**
- Create: `src/toc/__init__.py`
- Create: `src/toc/models.py`
- Create: `tests/toc/__init__.py`
- Create: `tests/toc/test_models.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/toc/__init__.py` (empty).

Create `tests/toc/test_models.py`:

```python
import pytest
from pydantic import ValidationError
from src.toc.models import ChunkStrategy, NodeType, TocDocument, TocNode


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
```

- [ ] **Step 2: Run to verify failure**

Run: `poetry run pytest tests/toc/test_models.py -v`
Expected: `ModuleNotFoundError: No module named 'src.toc'`

- [ ] **Step 3: Create `src/toc/__init__.py`** (empty file)

- [ ] **Step 4: Create `src/toc/models.py`**

```python
from enum import StrEnum

from pydantic import BaseModel
from pydantic import Field


class NodeType(StrEnum):
    CHAPTER = "chapter"
    SECTION = "section"
    SUBSECTION = "subsection"
    ENTRY = "entry"


class ChunkStrategy(StrEnum):
    ENTRY = "entry"
    ROW = "row"
    SECTION = "section"


class TocNode(BaseModel):
    id: str
    title: str
    node_type: NodeType
    chunk_strategy: ChunkStrategy | None = None
    entity_type: str | None = None
    page: int | None = None
    children: list["TocNode"] = Field(default_factory=list)


class TocDocument(BaseModel):
    document: str
    sections: list[TocNode]
```

- [ ] **Step 5: Run tests to verify pass**

Run: `poetry run pytest tests/toc/test_models.py -v`
Expected: All 11 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/toc/__init__.py src/toc/models.py tests/toc/__init__.py tests/toc/test_models.py
git commit -m "feat: add TOC schema Pydantic models and enums"
```

---

### Task 3: TOC YAML Parser

**Files:**
- Create: `src/toc/parser.py`
- Create: `tests/toc/test_parser.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/toc/test_parser.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `poetry run pytest tests/toc/test_parser.py -v`
Expected: `ModuleNotFoundError: No module named 'src.toc.parser'`

- [ ] **Step 3: Create `src/toc/parser.py`**

```python
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


def find_node_by_id(node_id: str, document: TocDocument) -> TocNode | None:
    nodes, _ = _build_index(document.sections)
    return nodes.get(node_id)


def find_node_by_title(title: str, document: TocDocument) -> TocNode | None:
    nodes, _ = _build_index(document.sections)
    return next((n for n in nodes.values() if n.title == title), None)


def resolve_chunk_strategy(
    node_id: str, document: TocDocument
) -> ChunkStrategy | None:
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
```

- [ ] **Step 4: Run tests to verify pass**

Run: `poetry run pytest tests/toc/test_parser.py -v`
Expected: All 14 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/toc/parser.py tests/toc/test_parser.py
git commit -m "feat: add TOC YAML parser with inheritance resolution"
```

---

### Task 4: Create `data/srd_toc.yaml`

**Files:**
- Create: `data/srd_toc.yaml`

- [ ] **Step 1: Add validation test to `tests/toc/test_parser.py`**

Append to the bottom of `tests/toc/test_parser.py`:

```python
def test_load_real_srd_toc():
    from pathlib import Path
    toc_path = Path("data/srd_toc.yaml")
    assert toc_path.exists(), "data/srd_toc.yaml not yet created"
    doc = load_toc(toc_path)
    assert doc.document == "D&D 5e 2024 SRD"
    assert len(doc.sections) == 12
    assert find_node_by_id("spells/spell-descriptions", doc) is not None
    assert find_node_by_id("monsters/monsters-a-z", doc) is not None
    assert find_node_by_id("magic-items/magic-items-a-z", doc) is not None
    assert find_node_by_id("classes/barbarian", doc) is not None
    assert resolve_entity_type("spells/spell-descriptions", doc) == "Spell"
    assert resolve_chunk_strategy("spells/spell-descriptions", doc) == ChunkStrategy.ENTRY
    assert resolve_entity_type("monsters/monsters-a-z", doc) == "Monster"
    assert resolve_chunk_strategy("character-creation/trinkets", doc) == ChunkStrategy.ROW
```

- [ ] **Step 2: Run to verify failure**

Run: `poetry run pytest tests/toc/test_parser.py::test_load_real_srd_toc -v`
Expected: FAIL — `data/srd_toc.yaml not yet created`

- [ ] **Step 3: Create `data/srd_toc.yaml`**

Create the file with the complete structural skeleton derived from `srd_toc_contents.md`.
Annotate each node with `node_type`, `chunk_strategy`, and `entity_type` per the rules below.

**Annotation rules:**
- `node_type`: top-level entries → `chapter`; first indent → `section`; second indent → `subsection`; named leaf items (classes, species, backgrounds, feats) → `entry`
- `chunk_strategy`: sections that are tables → `row`; sections split by named items → `entry`; prose sections → `section`; omit when inherited from parent
- `entity_type`: set on the section that contains a given entity type; omit when inherited

```yaml
document: "D&D 5e 2024 SRD"
sections:
  - id: legal-information
    title: "Legal Information"
    page: 1
    node_type: chapter
    chunk_strategy: section

  - id: playing-the-game
    title: "Playing the Game"
    page: 5
    node_type: chapter
    chunk_strategy: section
    children:
      - id: playing-the-game/rhythm-of-play
        title: "Rhythm of Play"
        page: 5
        node_type: section
      - id: playing-the-game/the-six-abilities
        title: "The Six Abilities"
        page: 5
        node_type: section
      - id: playing-the-game/d20-tests
        title: "D20 Tests"
        page: 6
        node_type: section
        children:
          - id: playing-the-game/d20-tests/ability-checks
            title: "Ability Checks"
            page: 6
            node_type: subsection
          - id: playing-the-game/d20-tests/saving-throws
            title: "Saving Throws"
            page: 7
            node_type: subsection
          - id: playing-the-game/d20-tests/attack-rolls
            title: "Attack Rolls"
            page: 7
            node_type: subsection
      - id: playing-the-game/advantage-disadvantage
        title: "Advantage/Disadvantage"
        page: 7
        node_type: section
      - id: playing-the-game/proficiency
        title: "Proficiency"
        page: 8
        node_type: section
      - id: playing-the-game/actions
        title: "Actions"
        page: 9
        node_type: section
        children:
          - id: playing-the-game/actions/bonus-actions
            title: "Bonus Actions"
            page: 10
            node_type: subsection
          - id: playing-the-game/actions/reactions
            title: "Reactions"
            page: 10
            node_type: subsection
      - id: playing-the-game/social-interaction
        title: "Social Interaction"
        page: 10
        node_type: section
      - id: playing-the-game/exploration
        title: "Exploration"
        page: 11
        node_type: section
        children:
          - id: playing-the-game/exploration/vision-and-light
            title: "Vision and Light"
            page: 11
            node_type: subsection
          - id: playing-the-game/exploration/hiding
            title: "Hiding"
            page: 11
            node_type: subsection
          - id: playing-the-game/exploration/interacting-with-objects
            title: "Interacting with Objects"
            page: 11
            node_type: subsection
          - id: playing-the-game/exploration/hazards
            title: "Hazards"
            page: 12
            node_type: subsection
          - id: playing-the-game/exploration/travel
            title: "Travel"
            page: 12
            node_type: subsection
      - id: playing-the-game/combat
        title: "Combat"
        page: 13
        node_type: section
        children:
          - id: playing-the-game/combat/the-order-of-combat
            title: "The Order of Combat"
            page: 13
            node_type: subsection
          - id: playing-the-game/combat/movement-and-position
            title: "Movement and Position"
            page: 14
            node_type: subsection
          - id: playing-the-game/combat/making-an-attack
            title: "Making an Attack"
            page: 14
            node_type: subsection
          - id: playing-the-game/combat/ranged-attacks
            title: "Ranged Attacks"
            page: 15
            node_type: subsection
          - id: playing-the-game/combat/melee-attacks
            title: "Melee Attacks"
            page: 15
            node_type: subsection
          - id: playing-the-game/combat/mounted-combat
            title: "Mounted Combat"
            page: 15
            node_type: subsection
          - id: playing-the-game/combat/underwater-combat
            title: "Underwater Combat"
            page: 16
            node_type: subsection
      - id: playing-the-game/damage-and-healing
        title: "Damage and Healing"
        page: 16
        node_type: section
        children:
          - id: playing-the-game/damage-and-healing/hit-points
            title: "Hit Points"
            page: 16
            node_type: subsection
          - id: playing-the-game/damage-and-healing/damage-rolls
            title: "Damage Rolls"
            page: 16
            node_type: subsection
          - id: playing-the-game/damage-and-healing/critical-hits
            title: "Critical Hits"
            page: 16
            node_type: subsection
          - id: playing-the-game/damage-and-healing/saving-throws-and-damage
            title: "Saving Throws and Damage"
            page: 16
            node_type: subsection
          - id: playing-the-game/damage-and-healing/damage-types
            title: "Damage Types"
            page: 16
            node_type: subsection
          - id: playing-the-game/damage-and-healing/resistance-and-vulnerability
            title: "Resistance and Vulnerability"
            page: 17
            node_type: subsection
          - id: playing-the-game/damage-and-healing/immunity
            title: "Immunity"
            page: 17
            node_type: subsection
          - id: playing-the-game/damage-and-healing/healing
            title: "Healing"
            page: 17
            node_type: subsection
          - id: playing-the-game/damage-and-healing/dropping-to-0-hit-points
            title: "Dropping to 0 Hit Points"
            page: 17
            node_type: subsection
          - id: playing-the-game/damage-and-healing/temporary-hit-points
            title: "Temporary Hit Points"
            page: 18
            node_type: subsection

  - id: character-creation
    title: "Character Creation"
    page: 19
    node_type: chapter
    chunk_strategy: section
    children:
      - id: character-creation/choose-a-character-sheet
        title: "Choose a Character Sheet"
        page: 19
        node_type: section
      - id: character-creation/create-your-character
        title: "Create Your Character"
        page: 19
        node_type: section
      - id: character-creation/level-advancement
        title: "Level Advancement"
        page: 23
        node_type: section
      - id: character-creation/starting-at-higher-levels
        title: "Starting at Higher Levels"
        page: 24
        node_type: section
      - id: character-creation/multiclassing
        title: "Multiclassing"
        page: 24
        node_type: section
      - id: character-creation/trinkets
        title: "Trinkets"
        page: 26
        node_type: section
        entity_type: Trinket
        chunk_strategy: row

  - id: classes
    title: "Classes"
    page: 28
    node_type: chapter
    chunk_strategy: entry
    children:
      - id: classes/barbarian
        title: "Barbarian"
        page: 28
        node_type: entry
        entity_type: Class
        children:
          - id: classes/barbarian/path-of-the-berserker
            title: "Barbarian Subclass: Path of the Berserker"
            page: 30
            node_type: entry
            entity_type: Subclass
      - id: classes/bard
        title: "Bard"
        page: 31
        node_type: entry
        entity_type: Class
        children:
          - id: classes/bard/spell-list
            title: "Bard Spell List"
            page: 33
            node_type: section
            entity_type: Spell
            chunk_strategy: entry
          - id: classes/bard/college-of-lore
            title: "Bard Subclass: College of Lore"
            page: 35
            node_type: entry
            entity_type: Subclass
      - id: classes/cleric
        title: "Cleric"
        page: 36
        node_type: entry
        entity_type: Class
        children:
          - id: classes/cleric/spell-list
            title: "Cleric Spell List"
            page: 38
            node_type: section
            entity_type: Spell
            chunk_strategy: entry
          - id: classes/cleric/life-domain
            title: "Cleric Subclass: Life Domain"
            page: 40
            node_type: entry
            entity_type: Subclass
      - id: classes/druid
        title: "Druid"
        page: 41
        node_type: entry
        entity_type: Class
        children:
          - id: classes/druid/spell-list
            title: "Druid Spell List"
            page: 44
            node_type: section
            entity_type: Spell
            chunk_strategy: entry
          - id: classes/druid/circle-of-the-land
            title: "Druid Subclass: Circle of the Land"
            page: 46
            node_type: entry
            entity_type: Subclass
      - id: classes/fighter
        title: "Fighter"
        page: 47
        node_type: entry
        entity_type: Class
        children:
          - id: classes/fighter/champion
            title: "Fighter Subclass: Champion"
            page: 49
            node_type: entry
            entity_type: Subclass
      - id: classes/monk
        title: "Monk"
        page: 49
        node_type: entry
        entity_type: Class
        children:
          - id: classes/monk/warrior-of-the-open-hand
            title: "Monk Subclass: Warrior of the Open Hand"
            page: 52
            node_type: entry
            entity_type: Subclass
      - id: classes/paladin
        title: "Paladin"
        page: 53
        node_type: entry
        entity_type: Class
        children:
          - id: classes/paladin/spell-list
            title: "Paladin Spell List"
            page: 55
            node_type: section
            entity_type: Spell
            chunk_strategy: entry
          - id: classes/paladin/oath-of-devotion
            title: "Paladin Subclass: Oath of Devotion"
            page: 56
            node_type: entry
            entity_type: Subclass
      - id: classes/ranger
        title: "Ranger"
        page: 57
        node_type: entry
        entity_type: Class
        children:
          - id: classes/ranger/spell-list
            title: "Ranger Spell List"
            page: 60
            node_type: section
            entity_type: Spell
            chunk_strategy: entry
          - id: classes/ranger/hunter
            title: "Ranger Subclass: Hunter"
            page: 61
            node_type: entry
            entity_type: Subclass
      - id: classes/rogue
        title: "Rogue"
        page: 61
        node_type: entry
        entity_type: Class
        children:
          - id: classes/rogue/thief
            title: "Rogue Subclass: Thief"
            page: 64
            node_type: entry
            entity_type: Subclass
      - id: classes/sorcerer
        title: "Sorcerer"
        page: 64
        node_type: entry
        entity_type: Class
        children:
          - id: classes/sorcerer/metamagic-options
            title: "Metamagic Options"
            page: 66
            node_type: section
          - id: classes/sorcerer/spell-list
            title: "Sorcerer Spell List"
            page: 67
            node_type: section
            entity_type: Spell
            chunk_strategy: entry
          - id: classes/sorcerer/draconic-sorcery
            title: "Sorcerer Subclass: Draconic Sorcery"
            page: 69
            node_type: entry
            entity_type: Subclass
      - id: classes/warlock
        title: "Warlock"
        page: 70
        node_type: entry
        entity_type: Class
        children:
          - id: classes/warlock/eldritch-invocation-options
            title: "Eldritch Invocation Options"
            page: 72
            node_type: section
          - id: classes/warlock/spell-list
            title: "Warlock Spell List"
            page: 74
            node_type: section
            entity_type: Spell
            chunk_strategy: entry
          - id: classes/warlock/fiend-patron
            title: "Warlock Subclass: Fiend Patron"
            page: 76
            node_type: entry
            entity_type: Subclass
      - id: classes/wizard
        title: "Wizard"
        page: 77
        node_type: entry
        entity_type: Class
        children:
          - id: classes/wizard/spell-list
            title: "Wizard Spell List"
            page: 79
            node_type: section
            entity_type: Spell
            chunk_strategy: entry
          - id: classes/wizard/evoker
            title: "Wizard Subclass: Evoker"
            page: 82
            node_type: entry
            entity_type: Subclass

  - id: character-origins
    title: "Character Origins"
    page: 83
    node_type: chapter
    children:
      - id: character-origins/character-backgrounds
        title: "Character Backgrounds"
        page: 83
        node_type: section
        entity_type: Background
        chunk_strategy: entry
        children:
          - id: character-origins/character-backgrounds/acolyte
            title: "Acolyte"
            page: 83
            node_type: entry
            entity_type: Background
          - id: character-origins/character-backgrounds/criminal
            title: "Criminal"
            page: 83
            node_type: entry
            entity_type: Background
          - id: character-origins/character-backgrounds/sage
            title: "Sage"
            page: 83
            node_type: entry
            entity_type: Background
          - id: character-origins/character-backgrounds/soldier
            title: "Soldier"
            page: 83
            node_type: entry
            entity_type: Background
      - id: character-origins/character-species
        title: "Character Species"
        page: 83
        node_type: section
        entity_type: Race
        chunk_strategy: entry
        children:
          - id: character-origins/character-species/dragonborn
            title: "Dragonborn"
            page: 84
            node_type: entry
            entity_type: Race
          - id: character-origins/character-species/dwarf
            title: "Dwarf"
            page: 84
            node_type: entry
            entity_type: Race
          - id: character-origins/character-species/elf
            title: "Elf"
            page: 84
            node_type: entry
            entity_type: Race
          - id: character-origins/character-species/gnome
            title: "Gnome"
            page: 85
            node_type: entry
            entity_type: Race
          - id: character-origins/character-species/goliath
            title: "Goliath"
            page: 85
            node_type: entry
            entity_type: Race
          - id: character-origins/character-species/halfling
            title: "Halfling"
            page: 86
            node_type: entry
            entity_type: Race
          - id: character-origins/character-species/human
            title: "Human"
            page: 86
            node_type: entry
            entity_type: Race
          - id: character-origins/character-species/orc
            title: "Orc"
            page: 86
            node_type: entry
            entity_type: Race
          - id: character-origins/character-species/tiefling
            title: "Tiefling"
            page: 86
            node_type: entry
            entity_type: Race

  - id: feats
    title: "Feats"
    page: 87
    node_type: chapter
    entity_type: Feat
    chunk_strategy: entry
    children:
      - id: feats/feat-descriptions
        title: "Feat Descriptions"
        page: 87
        node_type: section
      - id: feats/origin-feats
        title: "Origin Feats"
        page: 87
        node_type: section
      - id: feats/general-feats
        title: "General Feats"
        page: 87
        node_type: section
      - id: feats/fighting-style-feats
        title: "Fighting Style Feats"
        page: 87
        node_type: section
      - id: feats/epic-boon-feats
        title: "Epic Boon Feats"
        page: 88
        node_type: section

  - id: equipment
    title: "Equipment"
    page: 89
    node_type: chapter
    children:
      - id: equipment/coins
        title: "Coins"
        page: 89
        node_type: section
        chunk_strategy: section
      - id: equipment/weapons
        title: "Weapons"
        page: 89
        node_type: section
        entity_type: Weapon
        chunk_strategy: row
      - id: equipment/properties
        title: "Properties"
        page: 89
        node_type: section
        chunk_strategy: section
      - id: equipment/mastery-properties
        title: "Mastery Properties"
        page: 90
        node_type: section
        chunk_strategy: section
      - id: equipment/armor
        title: "Armor"
        page: 92
        node_type: section
        entity_type: Armor
        chunk_strategy: row
      - id: equipment/tools
        title: "Tools"
        page: 93
        node_type: section
        entity_type: Tool
        chunk_strategy: row
      - id: equipment/adventuring-gear
        title: "Adventuring Gear"
        page: 94
        node_type: section
        entity_type: Item
        chunk_strategy: row
      - id: equipment/mounts-and-vehicles
        title: "Mounts and Vehicles"
        page: 100
        node_type: section
        entity_type: Vehicle
        chunk_strategy: row
      - id: equipment/lifestyle-expenses
        title: "Lifestyle Expenses"
        page: 101
        node_type: section
        chunk_strategy: section
      - id: equipment/food-drink-and-lodging
        title: "Food, Drink, and Lodging"
        page: 101
        node_type: section
        chunk_strategy: section
      - id: equipment/hirelings
        title: "Hirelings"
        page: 102
        node_type: section
        chunk_strategy: section
      - id: equipment/spellcasting
        title: "Spellcasting"
        page: 102
        node_type: section
        chunk_strategy: section
      - id: equipment/magic-items
        title: "Magic Items"
        page: 102
        node_type: section
        chunk_strategy: section
      - id: equipment/crafting-nonmagical-items
        title: "Crafting Nonmagical Items"
        page: 103
        node_type: section
        chunk_strategy: section
      - id: equipment/brewing-potions-of-healing
        title: "Brewing Potions of Healing"
        page: 103
        node_type: section
        chunk_strategy: section
      - id: equipment/scribing-spell-scrolls
        title: "Scribing Spell Scrolls"
        page: 103
        node_type: section
        chunk_strategy: section

  - id: spells
    title: "Spells"
    page: 104
    node_type: chapter
    children:
      - id: spells/gaining-spells
        title: "Gaining Spells"
        page: 104
        node_type: section
        chunk_strategy: section
      - id: spells/casting-spells
        title: "Casting Spells"
        page: 104
        node_type: section
        chunk_strategy: section
      - id: spells/spell-descriptions
        title: "Spell Descriptions"
        page: 107
        node_type: section
        entity_type: Spell
        chunk_strategy: entry

  - id: rules-glossary
    title: "Rules Glossary"
    page: 176
    node_type: chapter
    chunk_strategy: section

  - id: gameplay-toolbox
    title: "Gameplay Toolbox"
    page: 192
    node_type: chapter
    chunk_strategy: section
    children:
      - id: gameplay-toolbox/travel-pace
        title: "Travel Pace"
        page: 192
        node_type: section
      - id: gameplay-toolbox/creating-a-background
        title: "Creating a Background"
        page: 192
        node_type: section
      - id: gameplay-toolbox/curses-and-magical-contagions
        title: "Curses and Magical Contagions"
        page: 193
        node_type: section
      - id: gameplay-toolbox/environmental-effects
        title: "Environmental Effects"
        page: 195
        node_type: section
      - id: gameplay-toolbox/fear-and-mental-stress
        title: "Fear and Mental Stress"
        page: 196
        node_type: section
      - id: gameplay-toolbox/poison
        title: "Poison"
        page: 197
        node_type: section
      - id: gameplay-toolbox/traps
        title: "Traps"
        page: 199
        node_type: section
      - id: gameplay-toolbox/combat-encounters
        title: "Combat Encounters"
        page: 202
        node_type: section

  - id: magic-items
    title: "Magic Items"
    page: 204
    node_type: chapter
    children:
      - id: magic-items/magic-item-categories
        title: "Magic Item Categories"
        page: 204
        node_type: section
        chunk_strategy: section
      - id: magic-items/magic-item-rarity
        title: "Magic Item Rarity"
        page: 205
        node_type: section
        chunk_strategy: section
      - id: magic-items/activating-a-magic-item
        title: "Activating a Magic Item"
        page: 206
        node_type: section
        chunk_strategy: section
      - id: magic-items/the-next-dawn
        title: "The Next Dawn"
        page: 206
        node_type: section
        chunk_strategy: section
      - id: magic-items/cursed-items
        title: "Cursed Items"
        page: 206
        node_type: section
        chunk_strategy: section
      - id: magic-items/magic-item-resilience
        title: "Magic Item Resilience"
        page: 206
        node_type: section
        chunk_strategy: section
      - id: magic-items/crafting-magic-items
        title: "Crafting Magic Items"
        page: 206
        node_type: section
        chunk_strategy: section
      - id: magic-items/sentient-magic-items
        title: "Sentient Magic Items"
        page: 207
        node_type: section
        chunk_strategy: section
      - id: magic-items/magic-items-a-z
        title: "Magic Items A-Z"
        page: 209
        node_type: section
        entity_type: MagicItem
        chunk_strategy: entry

  - id: monsters
    title: "Monsters"
    page: 254
    node_type: chapter
    children:
      - id: monsters/stat-block-overview
        title: "Stat Block Overview"
        page: 254
        node_type: section
        chunk_strategy: section
      - id: monsters/parts-of-a-stat-block
        title: "Parts of a Stat Block"
        page: 254
        node_type: section
        chunk_strategy: section
      - id: monsters/running-a-monster
        title: "Running a Monster"
        page: 255
        node_type: section
        chunk_strategy: section
      - id: monsters/monsters-a-z
        title: "Monsters A-Z"
        page: 258
        node_type: section
        entity_type: Monster
        chunk_strategy: entry
      - id: monsters/animals
        title: "Animals"
        page: 344
        node_type: section
        entity_type: Monster
        chunk_strategy: entry
```

- [ ] **Step 4: Run the validation test**

Run: `poetry run pytest tests/toc/test_parser.py::test_load_real_srd_toc -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add data/srd_toc.yaml tests/toc/test_parser.py
git commit -m "feat: add SRD TOC YAML structural skeleton (12 chapters)"
```

---

### Task 5: Neo4j Connection Settings

**Files:**
- Modify: `src/settings.py`

- [ ] **Step 1: Add Neo4j fields to Config**

In `src/settings.py`, add the following three lines inside the `Config` class body, after the `HF_TOKEN` field:

```python
NEO4J_URI: str = "bolt://localhost:7687"
NEO4J_USERNAME: str = "neo4j"
NEO4J_PASSWORD: str | None = None
```

- [ ] **Step 2: Verify Config still loads**

Run: `python -c "from src.settings import config; print(config.NEO4J_URI)"`
Expected: `bolt://localhost:7687`

- [ ] **Step 3: Commit**

```bash
git add src/settings.py
git commit -m "feat: add Neo4j connection settings to Config"
```

---

### Task 6: Neo4j Structure Seeder

**Files:**
- Create: `src/toc/neo4j_seeder.py`
- Create: `src/scripts/seed_neo4j.py`
- Create: `tests/toc/test_neo4j_seeder.py`

- [ ] **Step 1: Write failing tests**

Create `tests/toc/test_neo4j_seeder.py`:

```python
from unittest.mock import MagicMock

import pytest

from src.toc.models import ChunkStrategy
from src.toc.models import NodeType
from src.toc.models import TocDocument
from src.toc.models import TocNode
from src.toc.neo4j_seeder import seed_structure


@pytest.fixture
def simple_doc():
    child = TocNode(
        id="spells/spell-descriptions",
        title="Spell Descriptions",
        node_type=NodeType.SECTION,
        entity_type="Spell",
        chunk_strategy=ChunkStrategy.ENTRY,
        page=107,
    )
    root = TocNode(
        id="spells",
        title="Spells",
        node_type=NodeType.CHAPTER,
        page=104,
        children=[child],
    )
    return TocDocument(document="Test SRD", sections=[root])


def test_seed_structure_merges_section_nodes(simple_doc):
    mock_session = MagicMock()
    seed_structure(simple_doc, mock_session)
    # Two nodes → at least 2 session.run() calls for MERGE
    assert mock_session.run.call_count >= 2


def test_seed_structure_creates_contains_relationship(simple_doc):
    mock_session = MagicMock()
    seed_structure(simple_doc, mock_session)
    all_cypher = " ".join(str(c) for c in mock_session.run.call_args_list)
    assert "CONTAINS" in all_cypher


def test_seed_structure_applies_entity_label(simple_doc):
    mock_session = MagicMock()
    seed_structure(simple_doc, mock_session)
    all_cypher = " ".join(str(c) for c in mock_session.run.call_args_list)
    assert "Spell" in all_cypher


def test_seed_structure_no_contains_for_root_nodes(simple_doc):
    mock_session = MagicMock()
    seed_structure(simple_doc, mock_session)
    # The root node ("spells") has no parent, so CONTAINS is only created once
    contains_calls = [
        c for c in mock_session.run.call_args_list if "CONTAINS" in str(c)
    ]
    assert len(contains_calls) == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `poetry run pytest tests/toc/test_neo4j_seeder.py -v`
Expected: `ModuleNotFoundError: No module named 'src.toc.neo4j_seeder'`

- [ ] **Step 3: Create `src/toc/neo4j_seeder.py`**

```python
from neo4j import Session

from src.toc.models import TocDocument
from src.toc.models import TocNode


def _merge_node(node: TocNode, session: Session) -> None:
    props = {
        "title": node.title,
        "node_type": node.node_type.value,
        "page": node.page,
        "chunk_strategy": node.chunk_strategy.value if node.chunk_strategy else None,
        "entity_type": node.entity_type,
    }
    if node.entity_type:
        session.run(
            f"MERGE (s:Section:{node.entity_type} {{id: $id}}) SET s += $props",
            id=node.id,
            props=props,
        )
    else:
        session.run(
            "MERGE (s:Section {id: $id}) SET s += $props",
            id=node.id,
            props=props,
        )


def _merge_contains(parent_id: str, child_id: str, session: Session) -> None:
    session.run(
        "MATCH (p:Section {id: $parent_id}) "
        "MATCH (c:Section {id: $child_id}) "
        "MERGE (p)-[:CONTAINS]->(c)",
        parent_id=parent_id,
        child_id=child_id,
    )


def _seed_nodes(
    nodes: list[TocNode],
    session: Session,
    parent_id: str | None = None,
) -> None:
    for node in nodes:
        _merge_node(node, session)
        if parent_id is not None:
            _merge_contains(parent_id, node.id, session)
        if node.children:
            _seed_nodes(node.children, session, node.id)


def seed_structure(document: TocDocument, session: Session) -> None:
    _seed_nodes(document.sections, session)
```

- [ ] **Step 4: Run tests**

Run: `poetry run pytest tests/toc/test_neo4j_seeder.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Create `src/scripts/seed_neo4j.py`**

```python
"""Seed Neo4j with the SRD TOC structural skeleton (structure pass).

Reads data/srd_toc.yaml, MERGEs :Section nodes, and creates CONTAINS edges.
Set NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD in .env before running.

Usage:
    poetry run python -m src.scripts.seed_neo4j
    poetry run python -m src.scripts.seed_neo4j --toc-path data/srd_toc.yaml
"""

import argparse
import sys
from pathlib import Path

from neo4j import GraphDatabase

from src.settings import config
from src.settings import logger
from src.toc.neo4j_seeder import seed_structure
from src.toc.parser import load_toc


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed Neo4j with SRD TOC structure.")
    parser.add_argument(
        "--toc-path",
        type=str,
        default="data/srd_toc.yaml",
        help="Path to the SRD TOC YAML file (default: data/srd_toc.yaml)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    toc_path = Path(args.toc_path)

    if not toc_path.exists():
        logger.error(f"TOC file not found: {toc_path}")
        return 1

    if config.NEO4J_PASSWORD is None:
        logger.error("NEO4J_PASSWORD is not set in .env")
        return 1

    logger.info(f"Loading TOC from {toc_path}")
    document = load_toc(toc_path)
    logger.info(f"Loaded {len(document.sections)} top-level chapters")

    logger.info(f"Connecting to Neo4j at {config.NEO4J_URI}")
    driver = GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD),
    )

    try:
        with driver.session() as session:
            seed_structure(document, session)
        logger.info("Structure seeding complete")
    except Exception as e:
        logger.error(f"Seeding failed: {e}", exc_info=True)
        return 1
    finally:
        driver.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Commit**

```bash
git add src/toc/neo4j_seeder.py src/scripts/seed_neo4j.py tests/toc/test_neo4j_seeder.py
git commit -m "feat: add Neo4j structure seeder for SRD TOC"
```

---

### Task 7: Chunk Annotator

**Files:**
- Create: `src/scripts/annotate_chunks.py`
- Create: `tests/test_annotate_chunks.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_annotate_chunks.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `poetry run pytest tests/test_annotate_chunks.py -v`
Expected: `ModuleNotFoundError: No module named 'src.scripts.annotate_chunks'`

- [ ] **Step 3: Create `src/scripts/annotate_chunks.py`**

```python
"""Annotate existing chunk JSONL files with TOC-derived metadata.

Reads a chunks JSONL file (output of chunk_documents.py), looks up each
section_header in data/srd_toc.yaml, and stamps entity_type, context_path,
and chunk_strategy onto each matched chunk.

Usage:
    poetry run python -m src.scripts.annotate_chunks data/chunks/SRD_CC_v5.2.1_chunks.jsonl
    poetry run python -m src.scripts.annotate_chunks <input> --toc data/srd_toc.yaml
    poetry run python -m src.scripts.annotate_chunks <input> --output <out.jsonl>
"""

import argparse
import json
import sys
from pathlib import Path

from src.settings import logger
from src.toc.models import TocDocument
from src.toc.parser import find_node_by_title
from src.toc.parser import load_toc
from src.toc.parser import resolve_chunk_strategy
from src.toc.parser import resolve_entity_type


def load_annotator(toc_path: Path) -> TocDocument:
    return load_toc(toc_path)


def annotate_chunk(chunk: dict, document: TocDocument) -> dict:
    title = chunk.get("section_header", "")
    node = find_node_by_title(title, document)
    if node is None:
        return chunk
    result = dict(chunk)
    entity_type = resolve_entity_type(node.id, document)
    chunk_strategy = resolve_chunk_strategy(node.id, document)
    if entity_type is not None:
        result["entity_type"] = entity_type
    result["context_path"] = node.id
    if chunk_strategy is not None:
        result["chunk_strategy"] = chunk_strategy.value
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stamp chunk JSONL with entity_type and context_path from the TOC."
    )
    parser.add_argument("input", type=str, help="Path to input chunks JSONL file")
    parser.add_argument(
        "--toc",
        type=str,
        default="data/srd_toc.yaml",
        help="Path to the SRD TOC YAML file (default: data/srd_toc.yaml)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output path for annotated JSONL. "
            "Defaults to <input_stem>_annotated.jsonl in the same directory."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    input_path = Path(args.input)
    toc_path = Path(args.toc)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    if not toc_path.exists():
        logger.error(f"TOC file not found: {toc_path}")
        return 1

    output_path = (
        Path(args.output)
        if args.output
        else input_path.parent / f"{input_path.stem}_annotated.jsonl"
    )

    logger.info(f"Loading TOC from {toc_path}")
    document = load_annotator(toc_path)

    annotated_count = 0
    total_count = 0

    with (
        open(input_path, encoding="utf-8") as infile,
        open(output_path, "w", encoding="utf-8") as outfile,
    ):
        for line in infile:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            result = annotate_chunk(chunk, document)
            outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
            total_count += 1
            if "context_path" in result:
                annotated_count += 1

    logger.info(
        f"Annotated {annotated_count}/{total_count} chunks. Output: {output_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests**

Run: `poetry run pytest tests/test_annotate_chunks.py -v`
Expected: All 8 tests PASS.

- [ ] **Step 5: Run full TOC test suite**

Run: `poetry run pytest tests/toc/ tests/test_annotate_chunks.py -v`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/scripts/annotate_chunks.py tests/test_annotate_chunks.py
git commit -m "feat: add chunk annotator using SRD TOC YAML"
```
