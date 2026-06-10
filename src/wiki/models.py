"""Structured-output model for magic items rendered on the Fabled Campaigns Wiki.

The Wiki's TypeScript contract (``types/wiki.ts`` in the ``fabled-campaigns``
repo) is::

    export type MagicItem = {
      slug: string;
      name: string;
      itemType: string;
      rarity: MagicItemRarity;
      requiresAttunement: boolean;
      attunementBy?: string;
      body: string;
    };

``MagicItem`` below mirrors that contract field-for-field. Because the model uses
a camelCase alias generator, ``item.model_dump(by_alias=True)`` yields exactly the
JSON the Wiki consumes, and ``MagicItem.model_json_schema(by_alias=True)`` yields a
schema suitable for ``google-genai`` structured output.
"""

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic.alias_generators import to_camel

from src.extraction.enums import Rarity


class MagicItem(BaseModel):
    """A single magic item, shaped for Wiki rendering and LLM structured output.

    Serialize with ``by_alias=True`` to obtain the camelCase JSON the Fabled
    Campaigns Wiki expects (``slug``, ``name``, ``itemType``, ``rarity``,
    ``rarities``, ``requiresAttunement``, optional ``attunementBy``, ``body``).

    ``rarity`` is a verbatim display string; ``rarities`` is the normalized list
    of tiers used for faceted filtering (see the field docs below).
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        use_enum_values=True,
    )

    slug: str = Field(
        description=(
            "URL-safe kebab-case identifier derived from the name, used as the Wiki "
            "route segment (e.g. 'staff-of-the-magi')."
        ),
    )
    name: str = Field(
        description="Display name of the magic item (e.g. 'Staff of the Magi')."
    )
    item_type: str = Field(
        description=(
            "Item category exactly as it appears before the rarity in the SRD type "
            "line, including any parenthetical qualifier "
            "(e.g. 'Wondrous Item', 'Staff', 'Weapon (Mace)', "
            "'Armor (Any Light, Medium, or Heavy)')."
        ),
    )
    rarity: str = Field(
        description=(
            "Rarity clause exactly as written in the SRD type line, for display only. "
            "Usually a single tier ('Common', 'Uncommon', 'Rare', 'Very Rare', "
            "'Legendary', 'Artifact'), but kept as free text to stay faithful to items "
            "that scale by bonus ('Uncommon (+1), Rare (+2), or Very Rare (+3)') or are "
            "marked 'Rarity Varies'."
        ),
    )
    rarities: list[Rarity] = Field(
        description=(
            "Normalized rarity tiers for faceted filtering, in ascending order. A "
            "single-rarity item has one entry; an entry sold at several tiers has one "
            "per tier. A '+1/+2/+3' entry is really three items (the +1 is Uncommon, "
            "the +2 Rare, the +3 Very Rare), so it lists ['Uncommon', 'Rare', "
            "'Very Rare']. A 'Rarity Varies' entry lists every tier its variants span. "
            "Filtering by a tier should return any entry whose rarities include it."
        ),
    )
    requires_attunement: bool = Field(
        description="True if the item requires attunement before its magic can be used.",
    )
    attunement_by: str | None = Field(
        default=None,
        description=(
            "Attunement restriction phrase when one exists, written as it follows "
            "'Requires Attunement by ' in the SRD (e.g. 'a Sorcerer, Warlock, or "
            "Wizard', 'a Spellcaster'). Null when the item has no restriction or no "
            "attunement at all."
        ),
    )
    body: str = Field(
        description=(
            "Full item description as Markdown, excluding the name heading and the "
            "italic type/rarity line. Preserve bold property labels (e.g. "
            "'**Regaining Charges.**') and any Markdown tables."
        ),
    )
