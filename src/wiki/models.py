"""Structured-output models for the fabled-campaigns Wiki datasets.

Each model mirrors a type in fabled-campaigns `types/wiki.ts`, so a dumped list
drops into that repo's `data/*.json` without frontend changes.
"""

import re
from typing import Annotated
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import computed_field
from pydantic import field_validator

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

_HEADING = re.compile(r"^(#+) ", re.MULTILINE)


def slugify(name: str) -> str:
    """'Adult White Dragon' -> 'adult-white-dragon' (matches the magic-items slugs)."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


class WikiMonster(BaseModel):
    """One SRD 5.2 stat block, shaped as the Wiki's `Monster` record.

    `slug` is derived from `name`, so it is left out of the validation JSON
    schema sent to the model and added back on `model_dump()`.
    """

    model_config = ConfigDict(validate_by_name=True, serialize_by_alias=True)

    name: str = Field(description="Stat block name, e.g. 'Adult White Dragon'.")
    size: MonsterSize = Field(description="Size from the italic line under the name.")
    type: str = Field(
        description=(
            "Creature type from the italic line, including any tag in parentheses, "
            "e.g. 'Beast', 'Fiend (Demon)', 'Swarm of Tiny Beasts'."
        )
    )
    alignment: MonsterAlignment = Field(description="Alignment from the italic line.")
    ac: str = Field(pattern=r"^\d+", description="The **AC** value as written, e.g. '17'.")
    hp: str = Field(
        pattern=r"^\d+ \(\d+d\d+(?: [+-] \d+)?\)$",
        description="The **HP** value with hit dice, e.g. '150 (20d10 + 40)'.",
    )
    speed: str = Field(
        description="The **Speed** value as written, e.g. '10 ft., Swim 40 ft.'."
    )
    strength: AbilityScore = Field(alias="str")
    dexterity: AbilityScore = Field(alias="dex")
    constitution: AbilityScore = Field(alias="con")
    intelligence: AbilityScore = Field(alias="int")
    wisdom: AbilityScore = Field(alias="wis")
    charisma: AbilityScore = Field(alias="cha")
    cr: ChallengeRating = Field(description="Challenge rating from the **CR** line.")
    body: str = Field(
        min_length=1,
        description=(
            "Markdown for everything the header fields above do not hold. Start "
            "with one paragraph per line, separated by blank lines: **Initiative**, "
            "then **Saving Throws** listing only abilities whose SAVE differs from "
            "MOD (e.g. 'Dex +5, Wis +6'; omit the line if none), then the source "
            "**Skills**, **Gear**, **Resistances**, **Vulnerabilities**, "
            "**Immunities**, **Senses**, **Languages** and **CR** lines verbatim, "
            "skipping any the stat block lacks. Then each source section (Traits, "
            "Actions, Bonus Actions, Reactions, Legendary Actions) as a '## ' "
            "heading followed by its entries verbatim."
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
