"""Structured-output models for the fabled-campaigns Wiki datasets.

Field aliases are the camelCase JSON keys the Wiki's `types/wiki.ts` reads, so a
dumped list drops into that repo's `data/*.json`.
"""

import re
from typing import Annotated
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import computed_field
from pydantic import field_validator
from pydantic import model_validator
from pydantic.alias_generators import to_camel

from src.extraction.enums import CreatureType

MonsterSize = Literal[
    "Tiny", "Small", "Medium", "Large", "Huge", "Gargantuan", "Medium or Small"
]

MonsterAlignment = Literal[
    "Lawful Good",
    "Neutral Good",
    "Chaotic Good",
    "Lawful Neutral",
    "Neutral",
    "Chaotic Neutral",
    "Lawful Evil",
    "Neutral Evil",
    "Chaotic Evil",
    "Unaligned",
]

ChallengeRating = Literal[
    "0", "1/8", "1/4", "1/2",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
    "21", "22", "23", "24", "25", "26", "27", "28", "29", "30",
]  # fmt: skip

AbilityScore = Annotated[int, Field(ge=1, le=30)]
SaveBonus = Annotated[int, Field(ge=-5, le=20)]

_HEADING = re.compile(r"^(#+) ", re.MULTILINE)


def slugify(name: str) -> str:
    """'Adult White Dragon' -> 'adult-white-dragon' (matches the magic-items slugs)."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def proficiency_bonus(cr: str) -> int:
    """SRD proficiency bonus by challenge rating: +2 up to CR 4, +1 per 4 CRs after."""
    value = 0 if "/" in cr else int(cr)
    return 2 + max(value - 1, 0) // 4


class _WikiModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel, validate_by_name=True, serialize_by_alias=True
    )


class AbilitySaves(_WikiModel):
    """The SAVE column of the stat block's ability table."""

    strength: SaveBonus = Field(alias="str")
    dexterity: SaveBonus = Field(alias="dex")
    constitution: SaveBonus = Field(alias="con")
    intelligence: SaveBonus = Field(alias="int")
    wisdom: SaveBonus = Field(alias="wis")
    charisma: SaveBonus = Field(alias="cha")


class WikiMonster(_WikiModel):
    """One SRD 5.2 stat block, shaped as the Wiki's `Monster` record.

    `slug` is derived from `name`, so it is left out of the validation JSON
    schema sent to the model and added back on `model_dump()`. The ability
    modifier (MOD column) is not stored: it is always floor((score - 10) / 2).
    """

    name: str = Field(description="Stat block name, e.g. 'Adult White Dragon'.")
    category: Literal["Monster", "Animal"] = Field(
        description="'Animal' for entries in the SRD Animals chapter, else 'Monster'."
    )
    group: str = Field(
        description=(
            "The SRD heading the stat block sits under, shared by its variants, "
            "e.g. 'White Dragons' or 'Bandits'. Equals `name` for single entries."
        )
    )
    size: MonsterSize = Field(description="Size from the italic line under the name.")
    type: str = Field(
        description=(
            "Creature type as printed on the italic line, including any tag, "
            "e.g. 'Beast', 'Fiend (Demon)', 'Swarm of Tiny Beasts'."
        )
    )
    creature_type: CreatureType = Field(
        description=(
            "Base creature type for filtering, without tags or swarm wording, "
            "e.g. 'Fiend' for 'Fiend (Demon)', 'Beast' for 'Swarm of Tiny Beasts'."
        )
    )
    alignment: MonsterAlignment = Field(description="Alignment from the italic line.")
    ac: int = Field(ge=1, le=30, description="Armor Class from the **AC** line.")
    initiative: str = Field(
        pattern=r"^[+-]\d+ \(\d+\)$",
        description="The **Initiative** value as printed, e.g. '+7 (17)'.",
    )
    hp: str = Field(
        pattern=r"^\d+ \(\d+d\d+(?: [+-] \d+)?\)$",
        description="The **HP** value with hit dice, e.g. '150 (20d10 + 40)'.",
    )
    speed: str = Field(
        description="The **Speed** value as printed, e.g. '10 ft., Swim 40 ft.'."
    )
    strength: AbilityScore = Field(alias="str")
    dexterity: AbilityScore = Field(alias="dex")
    constitution: AbilityScore = Field(alias="con")
    intelligence: AbilityScore = Field(alias="int")
    wisdom: AbilityScore = Field(alias="wis")
    charisma: AbilityScore = Field(alias="cha")
    saves: AbilitySaves = Field(description="Saving throw bonuses from the SAVE column.")
    skills: str | None = Field(
        default=None, description="The **Skills** value as printed, or null if absent."
    )
    gear: str | None = Field(
        default=None, description="The **Gear** value as printed, or null if absent."
    )
    resistances: str | None = Field(
        default=None,
        description="The **Resistances** value as printed, or null if absent.",
    )
    vulnerabilities: str | None = Field(
        default=None,
        description="The **Vulnerabilities** value as printed, or null if absent.",
    )
    immunities: str | None = Field(
        default=None,
        description=(
            "The **Immunities** value as printed, or null if absent. Damage and "
            "condition immunities stay separated by ';', e.g. 'Fire, Poison; Poisoned'."
        ),
    )
    senses: str = Field(
        description=(
            "The **Senses** value as printed, e.g. "
            "'Darkvision 120 ft.; Passive Perception 20'."
        )
    )
    languages: str = Field(description="The **Languages** value as printed.")
    cr: ChallengeRating = Field(description="Challenge rating from the **CR** line.")
    xp: int = Field(ge=0, description="XP from the **CR** line, e.g. 5900.")
    xp_in_lair: int | None = Field(
        default=None,
        ge=0,
        description="XP when encountered in its lair, e.g. 7200, or null if not listed.",
    )
    proficiency_bonus: int = Field(
        ge=2, le=9, description="PB from the **CR** line, e.g. 4 for 'PB +4'."
    )
    body: str = Field(
        min_length=1,
        description=(
            "Markdown of the stat block's sections in source order (Traits, Actions, "
            "Bonus Actions, Reactions, Legendary Actions; skip any it lacks). Each "
            "section is a '## ' heading followed by its entries verbatim."
        ),
    )

    @computed_field
    @property
    def slug(self) -> str:
        return slugify(self.name)

    @field_validator("body")
    @classmethod
    def headings_are_h2(cls, body: str) -> str:
        # The Wiki page title is the H1 and MarkdownBody styles h2 for sections.
        levels = {len(hashes) for hashes in _HEADING.findall(body)}
        if levels - {2}:
            raise ValueError("body section headings must be '## ' (H2)")
        return body

    @model_validator(mode="after")
    def stats_are_consistent(self) -> "WikiMonster":
        if self.proficiency_bonus != proficiency_bonus(self.cr):
            raise ValueError(f"PB +{self.proficiency_bonus} does not match CR {self.cr}")
        for field in AbilitySaves.model_fields:
            modifier = (getattr(self, field) - 10) // 2
            if getattr(self.saves, field) < modifier:
                raise ValueError(f"{field} save is below its ability modifier")
        return self
