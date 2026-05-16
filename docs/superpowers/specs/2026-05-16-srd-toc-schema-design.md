# SRD TOC Schema Design

**Date:** 2026-05-16
**Status:** Approved

## Problem

`srd_toc_contents.md` is a human-readable Markdown nested list. It is not machine-readable, carries no type information, and provides no signal to a chunking pipeline about how to split or label content. The Graph database (Neo4j) needs the TOC to serve three roles simultaneously:

- **Navigation backbone** — structural hierarchy of the SRD document
- **Chunk metadata** — tells the chunker where to split and what entity type to stamp on each chunk
- **Taxonomy** — provides semantic labels so the graph can answer "what kind of thing is this?"

## Solution

Convert the TOC to a YAML file at `data/srd_toc.yaml`. The original `srd_toc_contents.md` is preserved untouched. The stat block index at the bottom of the original is dropped (redundant once entity nodes exist in the graph).

## YAML Schema

Each node in the YAML represents a section of the SRD. Fields:

| Field | Type | Required | Used by |
|-------|------|----------|---------|
| `id` | string (slug path) | yes | Chunker, Graph (node key) |
| `title` | string | yes | Chunker (header matching), Graph, User |
| `node_type` | enum | yes | Graph |
| `chunk_strategy` | enum | no | Chunker (inherited from nearest ancestor) |
| `entity_type` | string | no | Chunker (chunk label), Graph (entity linkage) |
| `page` | int | no | Graph (user-facing page reference only) |
| `children` | list | no | Both |

### `node_type` values

- `chapter` — top-level SRD section (Playing the Game, Classes, Spells, Monsters, etc.)
- `section` — major subdivision within a chapter
- `subsection` — further subdivision
- `entry` — a named leaf item (a specific spell, monster, class, magic item, etc.)

### `chunk_strategy` values

- `entry` — each named sub-item becomes its own chunk (e.g., each spell, each magic item)
- `row` — each table row becomes its own chunk (e.g., each weapon, each trinket)
- `section` — the entire section prose is one chunk

A node without `chunk_strategy` inherits the nearest ancestor that has one set.

### `entity_type` values

Matches entity class names: `Spell`, `Monster`, `Class`, `Subclass`, `Race`, `Background`,
`Feat`, `MagicItem`, `Weapon`, `Armor`, `Tool`, `Item`, `Trinket`, `Vehicle`, `Feature`,
`Condition`, `Skill`, `DamageType`, `Action`.

`entity_type` propagates to children: rows/entries under a section inherit the parent's
`entity_type` when they don't set their own.

## Example YAML

```yaml
document: "D&D 5e 2024 SRD"
sections:
  - id: playing-the-game
    title: "Playing the Game"
    page: 5
    node_type: chapter
    chunk_strategy: section
    children:
      - id: playing-the-game/d20-tests
        title: "D20 Tests"
        page: 6
        node_type: section
        children:
          - id: playing-the-game/d20-tests/ability-checks
            title: "Ability Checks"
            page: 6
            node_type: subsection

  - id: character-creation
    title: "Character Creation"
    page: 19
    node_type: chapter
    children:
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
    children:
      - id: classes/barbarian
        title: "Barbarian"
        page: 28
        node_type: entry
        entity_type: Class
        chunk_strategy: entry
        children:
          - id: classes/barbarian/path-of-the-berserker
            title: "Path of the Berserker"
            page: 30
            node_type: entry
            entity_type: Subclass

  - id: equipment
    title: "Equipment"
    page: 89
    node_type: chapter
    children:
      - id: equipment/weapons
        title: "Weapons"
        page: 89
        node_type: section
        entity_type: Weapon
        chunk_strategy: row

  - id: spells
    title: "Spells"
    page: 104
    node_type: chapter
    children:
      - id: spells/spell-descriptions
        title: "Spell Descriptions"
        page: 107
        node_type: section
        entity_type: Spell
        chunk_strategy: entry
        children:
          - id: spells/spell-descriptions/acid-arrow
            title: "Acid Arrow"
            page: 107
            node_type: entry
            entity_type: Spell

  - id: magic-items
    title: "Magic Items"
    page: 204
    node_type: chapter
    children:
      - id: magic-items/magic-items-a-z
        title: "Magic Items A–Z"
        page: 209
        node_type: section
        entity_type: MagicItem
        chunk_strategy: entry
        children:
          - id: magic-items/magic-items-a-z/sun-blade
            title: "Sun Blade"
            page: 247
            node_type: entry
            entity_type: MagicItem

  - id: monsters
    title: "Monsters"
    page: 254
    node_type: chapter
    children:
      - id: monsters/monsters-a-z
        title: "Monsters A–Z"
        page: 258
        node_type: section
        entity_type: Monster
        chunk_strategy: entry
```

## Neo4j Graph Model

### Node Labels

- **`:Section`** — every YAML entry becomes a Section node
- **`:Section:<EntityType>`** — entry nodes with `entity_type` get a second label (e.g., `:Section:Spell`, `:Section:Monster`). Entity properties from the extraction pipeline are merged onto these nodes in a second pass.

### Relationships

- **`(:Section)-[:CONTAINS]->(:Section)`** — derived from YAML nesting
- No separate entity nodes needed; entity data merges directly onto `:Section:<EntityType>` nodes

### Example Cypher queries

```cypher
-- All spells in the Spells chapter
MATCH (:Section {id: "spells"})-[:CONTAINS*]->(s:Spell)
RETURN s.title, s.level, s.school

-- What kind of thing is "Sun Blade"?
MATCH (n:Section {title: "Sun Blade"}) RETURN labels(n), n.page

-- Navigate structure: what does Combat contain?
MATCH (:Section {id: "playing-the-game/combat"})-[:CONTAINS]->(s:Section)
RETURN s.title
```

## Seeding Pipeline (two passes)

1. **Structure pass** — parse `data/srd_toc.yaml`, MERGE all `:Section` nodes, create `CONTAINS` edges
2. **Entity pass** — for each extracted entity from the chunking/extraction pipeline, MERGE its properties onto the matching `:Section` node and add the entity label

## Chunking Integration

The chunker uses the YAML as a lookup table:

1. Match the current Markdown heading to a YAML node by `title` (or `id` path)
2. Read `chunk_strategy` (walk up ancestors until found)
3. Read `entity_type` (walk up ancestors until found)
4. Stamp the chunk with `entity_type` and `context_path` (the slash-separated `id`)

Example chunk metadata for a row in the Trinkets table:

```json
{
  "text": "A white, sequined glove sized for a human",
  "entity_type": "Trinket",
  "context_path": "character-creation/trinkets",
  "section_title": "Trinkets"
}
```
