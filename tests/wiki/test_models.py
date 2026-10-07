import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.wiki.models import WikiMonster
from src.wiki.models import proficiency_bonus
from src.wiki.models import slugify

SAMPLE = json.loads((Path(__file__).parent / "monsters_sample.json").read_text())

# Keys of `Monster` in fabled-campaigns types/wiki.ts.
WIKI_MONSTER_KEYS = {
    "slug", "name", "category", "group", "size", "type", "creatureType",
    "alignment", "ac", "initiative", "hp", "speed",
    "str", "dex", "con", "int", "wis", "cha", "saves",
    "skills", "gear", "resistances", "vulnerabilities", "immunities",
    "senses", "languages", "cr", "xp", "xpInLair", "proficiencyBonus", "body",
}  # fmt: skip


@pytest.mark.parametrize("record", SAMPLE, ids=[r["slug"] for r in SAMPLE])
def test_sample_round_trips_to_wiki_shape(record):
    dumped = WikiMonster.model_validate(record).model_dump()
    assert dumped == record
    assert set(dumped) == WIKI_MONSTER_KEYS


@pytest.mark.parametrize(
    ("name", "slug"),
    [
        ("Adult White Dragon", "adult-white-dragon"),
        ("Saber-Toothed Tiger", "saber-toothed-tiger"),
        ("Ammunition, +1", "ammunition-1"),
    ],
)
def test_slugify(name, slug):
    assert slugify(name) == slug


@pytest.mark.parametrize(
    ("cr", "pb"),
    [("0", 2), ("1/8", 2), ("4", 2), ("5", 3), ("13", 5), ("21", 7), ("30", 9)],
)
def test_proficiency_bonus(cr, pb):
    assert proficiency_bonus(cr) == pb


def test_llm_schema_uses_wiki_keys_without_slug():
    properties = WikiMonster.model_json_schema()["properties"]
    assert set(properties) == WIKI_MONSTER_KEYS - {"slug"}


@pytest.mark.parametrize(
    "override",
    [
        {"body": "## Actions\n\n##### Bite"},
        {"str": 0},
        {"cr": "1/3"},
        {"hp": "150"},
        {"alignment": "Any"},
        {"creatureType": "Swarm"},
        {"proficiencyBonus": 3},
        {"saves": {**SAMPLE[0]["saves"], "dex": -5}},
    ],
)
def test_rejects_invalid_values(override):
    record = {k: v for k, v in SAMPLE[0].items() if k != "slug"} | override
    with pytest.raises(ValidationError):
        WikiMonster.model_validate(record)
