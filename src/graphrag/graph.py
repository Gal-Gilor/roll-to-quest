"""Build a D&D knowledge graph from the SRD Markdown, without an LLM.

The SRD is regular enough to read entities and relationships from its structure:

1. `markdown-chunkify` splits the document into sections.
2. Each section gets an entity label from where it sits in the document. A section
   under "Spell Descriptions" is a Spell, a stat block under "Monsters A-Z" is a
   Monster. Sections that are no entity keep the label "Section".
3. An entity absorbs its subsections, so a Monster node holds the whole stat block.
4. Closed vocabularies that have no section of their own (damage types, schools of
   magic, sizes) become nodes from the enums in `src.extraction.enums`.
5. FIELD_RULES read fixed-format lines, such as a monster's Immunities line.
6. A section that names another section's header gets an edge typed by the two labels
   (REFERENCE_TYPES), or REFERENCES when the pair has no specific meaning.

Labels and edge types follow `src/extraction/entities.json` and `relationships.json`.
The graph is rebuilt in memory on startup, so nothing is persisted.
"""

import re
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import networkx as nx
from markdown_chunkify import MarkdownSplitter
from markdown_chunkify import Section

from src.extraction import enums

SECTION = "Section"
CONTAINS = "CONTAINS"
REFERENCES = "REFERENCES"

# Entities whose subsections are other entities, so they don't absorb them.
CONTAINER_LABELS = frozenset({SECTION, "Class", "Subclass"})

CONCEPTS = {
    "AbilityScore": enums.AbilityScore,
    "Alignment": enums.Alignment,
    "CreatureType": enums.CreatureType,
    "DamageType": enums.DamageType,
    "Rarity": enums.Rarity,
    "SchoolOfMagic": enums.SchoolOfMagic,
    "Size": enums.Size,
    "Skill": enums.DndSkill,
}

STRUCTURE_TYPES = {
    ("Class", "Subclass"): "HAS_SUBCLASS",
    ("Class", "Feature"): "HAS_FEATURE",
    ("Subclass", "Feature"): "HAS_FEATURE",
}

# A spell name in a monster or item is only a spell it casts when the line says so or
# lists it. "Darkvision 60 ft." under Senses is not the Darkvision spell.
REFERENCE_TYPES = {
    ("Monster", "Spell"): "CAN_CAST",
    ("MagicItem", "Spell"): "GRANTS_SPELL",
    ("Feature", "Spell"): "GRANTS_SPELL",
    ("Feat", "Spell"): "GRANTS_SPELL",
}


class FieldRule(NamedTuple):
    """Link `source` entities to the `target` names found in the text `pattern` captures."""

    source: str
    pattern: re.Pattern
    target: str
    edge_type: str


def _line(prefix: str) -> re.Pattern:
    return re.compile(rf"^{prefix}(.+)$", re.MULTILINE)


_FIRST_LINE = re.compile(r"\A(.+)$", re.MULTILINE)
_INFLICTS = re.compile(r"(?<!doesn't )(?<!don't )(?<!longer )ha(?:s|ve) the (\w+) condition")
_IMMUNITIES = _line(r"\*\*Immunities\*\* ")
_RESISTANCES = _line(r"\*\*Resistances\*\* ")
_VULNERABILITIES = _line(r"\*\*Vulnerabilities\*\* ")

FIELD_RULES = [
    # "Level 3 Evocation (Sorcerer, Wizard)"
    FieldRule("Spell", _FIRST_LINE, "SchoolOfMagic", "BELONGS_TO_SCHOOL"),
    FieldRule("Spell", _FIRST_LINE, "Class", "CAST_BY"),
    FieldRule("Spell", re.compile(r"(\w+) damage"), "DamageType", "DEALS_DAMAGE"),
    FieldRule("Spell", re.compile(r"(\w+) saving throw"), "AbilityScore", "REQUIRES_SAVE"),
    FieldRule("Spell", _INFLICTS, "Condition", "INFLICTS_CONDITION"),
    # "*Huge Dragon (Chromatic), Chaotic Evil*"
    FieldRule("Monster", _FIRST_LINE, "Size", "HAS_SIZE"),
    FieldRule("Monster", _FIRST_LINE, "CreatureType", "HAS_TYPE"),
    FieldRule("Monster", _FIRST_LINE, "Alignment", "HAS_ALIGNMENT"),
    # "**Immunities** Poison; Charmed, Exhaustion, Poisoned"
    FieldRule("Monster", _IMMUNITIES, "DamageType", "IMMUNE_TO_DAMAGE"),
    FieldRule("Monster", _IMMUNITIES, "Condition", "IMMUNE_TO_CONDITION"),
    FieldRule("Monster", _RESISTANCES, "DamageType", "RESISTANT_TO"),
    FieldRule("Monster", _VULNERABILITIES, "DamageType", "VULNERABLE_TO"),
    FieldRule("Monster", _line(r"\*\*Skills\*\* "), "Skill", "PROFICIENT_IN"),
    FieldRule("Monster", _INFLICTS, "Condition", "INFLICTS_CONDITION"),
    # "*Staff, Very Rare (Requires Attunement by a Druid, Sorcerer, Warlock, or Wizard)*"
    FieldRule("MagicItem", _FIRST_LINE, "Rarity", "HAS_RARITY"),
    FieldRule(
        "MagicItem",
        re.compile(r"Resistance to (\w+) damage"),
        "DamageType",
        "PROVIDES_RESISTANCE",
    ),
    # "| Primary Ability | Intelligence |"
    FieldRule("Class", _line(r"\| Primary Ability\s*\|"), "AbilityScore", "USES_ABILITY"),
    # "**Feat:** Magic Initiate (Cleric) (see 'Feats')"
    FieldRule("Background", _line(r"\*\*Feat:\*\* "), "Feat", "GRANTS_FEAT"),
    FieldRule("Background", _line(r"\*\*Skill Proficiencies:\*\* "), "Skill", "GRANTS_SKILL"),
    FieldRule("Background", _line(r"\*\*Tool Proficiency:\*\* "), "Tool", "GRANTS_TOOL"),
    FieldRule("Race", _line(r"\*\*Size:\*\* "), "Size", "HAS_SIZE"),
]

_SPELL_CONTEXT = re.compile(r"cast|spell|^\s*[-|]", re.IGNORECASE)
MIN_ALIAS_LENGTH = 4
# "Hide [Action]" -> "Hide", "Cook's Utensils (1 GP)" -> "Cook's Utensils"
_HEADER_SUFFIX = re.compile(r"\s*(?:\[[^\]]*\]|\((?:[\d,]+ [CSG]P|Varies)\))$")
_STAT_LINE = re.compile(r"\A\*[A-Z][^*\n]*, [^*\n]*\*$", re.MULTILINE)


def _alias(header: str) -> str:
    """Return the name other sections use to refer to a header."""
    return _HEADER_SUFFIX.sub("", header).strip()


def _label(section: Section) -> str:
    """Return the entity label of a section, from its place in the document."""
    parents = section.metadata.parents
    chapter, part, group = (parents.get(key) for key in ("h1", "h2", "h3"))
    header, level = section.header, section.level

    if header.endswith("[Condition]"):
        return "Condition"
    if header.endswith("[Action]"):
        return "Action"
    if chapter == "Classes":
        if level == 2:
            return "Class"
        if " Subclass: " in header:
            return "Subclass"
        if re.match(r"Level \d+: ", header):
            return "Feature"
    if part == "Spell Descriptions" and level == 3:
        return "Spell"
    if part == "Magic Items A-Z" and level == 3:
        return "MagicItem"
    if part in ("Monsters A-Z", "Animals") and _STAT_LINE.match(section.content):
        return "Monster"
    if (
        chapter == "Feats"
        and level == 3
        and part not in ("Feat Descriptions", "Parts of a Feat")
    ):
        return "Feat"
    if group == "Background Descriptions":
        return "Background"
    if group == "Species Descriptions":
        return "Race"
    if part == "Tools" and level == 4:
        return "Tool"

    return SECTION


def _name_pattern(names: list[str]) -> re.Pattern:
    """Match any of the names as whole words. Longest first, so "Very Rare" beats "Rare"."""
    ordered = sorted(names, key=len, reverse=True)

    return re.compile(r"\b(?:" + "|".join(map(re.escape, ordered)) + r")\b")


def _add_section_nodes(graph: nx.DiGraph, sections: list[Section]) -> None:
    """Add one node per section, folding the subsections of an entity into its node."""
    owner: dict[int, int] = {}  # Absorbed section id -> the entity node that holds it

    for section in sections:
        section_id, parent_id = section.metadata.id, section.metadata.parent_id
        parent = owner.get(parent_id, parent_id)

        if parent is not None and graph.nodes[parent]["label"] not in CONTAINER_LABELS:
            owner[section_id] = parent
            graph.nodes[parent]["content"] += f"\n\n{section.to_markdown()}"
            continue

        label = _label(section)
        graph.add_node(
            section_id,
            label=label,
            header=section.header,
            breadcrumb=section.breadcrumb(),
            category=section.metadata.parents.get("h1", section.header),
            content=section.content,
        )
        if parent is not None:
            edge_type = STRUCTURE_TYPES.get((graph.nodes[parent]["label"], label), CONTAINS)
            graph.add_edge(parent, section_id, type=edge_type)


def _add_concept_nodes(graph: nx.DiGraph) -> None:
    """Add a node per enum member, such as DamageType "Fire". They carry no text."""
    for label, members in CONCEPTS.items():
        for member in members:
            graph.add_node(
                f"{label}:{member.value}",
                label=label,
                header=member.value,
                breadcrumb=f"{label} > {member.value}",
                category=label,
                content="",
            )


def _names_by_label(graph: nx.DiGraph) -> dict[str, dict[str, int | str]]:
    """Map each entity label to its nodes by name. Later duplicates of a name are dropped."""
    names: dict[str, dict[str, int | str]] = defaultdict(dict)
    for node, data in graph.nodes(data=True):
        if data["label"] != SECTION:
            names[data["label"]].setdefault(_alias(data["header"]), node)

    return names


def _add_field_edges(graph: nx.DiGraph) -> None:
    """Add the edges FIELD_RULES read from fixed-format text."""
    names = _names_by_label(graph)
    patterns = {label: _name_pattern(list(by_name)) for label, by_name in names.items()}

    for node, data in list(graph.nodes(data=True)):
        for rule in FIELD_RULES:
            if rule.source != data["label"] or rule.target not in patterns:
                continue
            for field in rule.pattern.findall(data["content"]):
                for name in patterns[rule.target].findall(field):
                    target = names[rule.target][name]
                    if target != node and not graph.has_edge(node, target):
                        graph.add_edge(node, target, type=rule.edge_type)


def _reference_targets(graph: nx.DiGraph) -> dict[str, int]:
    """Map each unambiguous header alias to its node.

    A name shared by an entity and plain sections ("Ghoul" the stat block and "Ghoul"
    the heading above it) resolves to the entity. A name still shared after that can't
    be resolved to one node and is dropped. Short or lowercase aliases match too much
    prose. Concept nodes are left to FIELD_RULES: "Fire" or "Medium" in running text
    is rarely the game term.
    """
    sections = [
        (node, _alias(data["header"]), data["label"] != SECTION)
        for node, data in graph.nodes(data=True)
        if data["label"] not in CONCEPTS
    ]
    entity_counts = Counter(alias for _, alias, is_entity in sections if is_entity)
    counts = Counter(alias for _, alias, _ in sections)

    return {
        alias: node
        for node, alias, is_entity in sections
        if len(alias) >= MIN_ALIAS_LENGTH
        and alias[0].isupper()
        and (
            (is_entity and entity_counts[alias] == 1)
            or (counts[alias] == 1 and not entity_counts[alias])
        )
    }


def _add_reference_edges(graph: nx.DiGraph) -> None:
    """Link each node to the nodes whose header its text names."""
    targets = _reference_targets(graph)
    if not targets:
        return

    # Matching is case-sensitive because the SRD capitalizes game terms.
    pattern = _name_pattern(list(targets))
    for source, data in list(graph.nodes(data=True)):
        content = data["content"]
        for match in pattern.finditer(content):
            target = targets[match.group()]
            if target == source or graph.has_edge(source, target):
                continue
            if graph.has_edge(target, source):  # Don't mirror a structural link
                continue

            edge_type = REFERENCE_TYPES.get((data["label"], graph.nodes[target]["label"]))
            if edge_type is None:
                graph.add_edge(source, target, type=REFERENCES)
                continue

            line_start = content.rfind("\n", 0, match.start()) + 1
            line_end = content.find("\n", match.end())
            line = content[line_start : line_end if line_end != -1 else len(content)]
            if _SPELL_CONTEXT.search(line):
                graph.add_edge(source, target, type=edge_type)


def build_graph(markdown_path: str | Path) -> nx.DiGraph:
    """Split a Markdown file into sections and link them into a directed graph.

    Args:
        markdown_path: Path to the SRD Markdown file.

    Returns:
        A graph whose nodes have `label` (entity type), `header`, `breadcrumb`,
        `category` (the top-level header) and `content` attributes. Each edge has a
        `type` attribute.
    """
    graph = nx.DiGraph()
    _add_section_nodes(graph, MarkdownSplitter().split_file(markdown_path))
    _add_concept_nodes(graph)
    _add_field_edges(graph)
    _add_reference_edges(graph)

    return graph
