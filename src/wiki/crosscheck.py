"""Compare an extracted WikiMonster with the stat block it came from.

The source prints every field on a fixed line shape, so all fields except `size`,
`type`, `creatureType` and `alignment` can be read back with a regex. Those four
are constrained by the model's Literal and enum types instead.
"""

import re
from typing import Any

from src.wiki.splitter import StatBlock

_LABEL = re.compile(r"^\*\*(\w+)\*\* (.+)$", re.MULTILINE)
_ABILITY_ROW = re.compile(
    r"^\| (STR|DEX|CON|INT|WIS|CHA) +\| (\d+) +\| [+-]\d+ +\| ([+-]\d+) +\|$", re.MULTILINE
)
_CR = re.compile(r"^(\S+) \(XP ([\d,]+)(?:, or ([\d,]+) in lair)?; PB \+(\d+)\)$")

TEXT_FIELDS = {
    "Initiative": "initiative",
    "HP": "hp",
    "Speed": "speed",
    "Skills": "skills",
    "Gear": "gear",
    "Resistances": "resistances",
    "Vulnerabilities": "vulnerabilities",
    "Immunities": "immunities",
    "Senses": "senses",
    "Languages": "languages",
}


def _number(value: str | None) -> int | None:
    return None if value is None else int(value.replace(",", ""))


def expected_fields(block: StatBlock) -> dict[str, Any]:
    """Read the checkable fields from a stat block, keyed by WikiMonster alias."""
    header, sep, sections = block.text.partition("\n##### ")
    if not sep:
        raise ValueError(f"{block.name}: no '##### ' section found")

    labels = dict(_LABEL.findall(header))
    cr = _CR.match(labels["CR"])
    if not cr:
        raise ValueError(f"{block.name}: unexpected CR line {labels['CR']!r}")

    expected: dict[str, Any] = {
        "name": block.name,
        "category": block.category,
        "group": block.group,
        "ac": int(labels["AC"]),
        **{key: labels.get(label) for label, key in TEXT_FIELDS.items()},
        "cr": cr.group(1),
        "xp": _number(cr.group(2)),
        "xpInLair": _number(cr.group(3)),
        "proficiencyBonus": int(cr.group(4)),
        "saves": {},
        "body": re.sub(r"^##### ", "## ", "##### " + sections, flags=re.MULTILINE).strip(),
    }
    for ability, score, save in _ABILITY_ROW.findall(header):
        key = ability.lower()
        expected[key] = int(score)
        expected["saves"][key] = int(save)
    if len(expected["saves"]) != 6:
        raise ValueError(f"{block.name}: ability table has {len(expected['saves'])} rows")
    return expected


def mismatches(record: dict[str, Any], block: StatBlock) -> list[str]:
    """Return one message per field where `record` (a WikiMonster dump) differs."""
    return [
        f"{block.name}: {key} is {record.get(key)!r}, source has {value!r}"
        for key, value in expected_fields(block).items()
        if record.get(key) != value
    ]
