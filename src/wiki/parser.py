"""Parse ``magic_items.md`` (SRD magic items) into :class:`MagicItem` models.

The SRD markdown lays each item out as::

    ### Staff of the Magi

    *Staff, Legendary (Requires Attunement by a Sorcerer, Warlock, or Wizard)*

    This staff has 50 charges ...

This module splits that into structured :class:`MagicItem` records so the shape
can be validated against the Wiki contract and sampled for review.
"""

import re

from src.extraction.enums import Rarity
from src.wiki.models import MagicItem

# Top-level item headings are exactly three hashes ("### Name"); deeper headings
# ("#### ...", "##### ...") are sub-sections inside an item body and must not split.
_ITEM_SPLIT = re.compile(r"\n### (?!#)")
_TYPE_LINE = re.compile(r"\*([^*]+)\*")

# Longest rarity strings first so "Very Rare" is matched before "Rare".
_RARITIES = [
    Rarity.VERY_RARE,
    Rarity.UNCOMMON,
    Rarity.COMMON,
    Rarity.LEGENDARY,
    Rarity.ARTIFACT,
    Rarity.RARE,
]


def slugify(name: str) -> str:
    """Convert an item name into a kebab-case slug (e.g. 'Staff of the Magi')."""
    slug = re.sub(r"[^\w\s-]", "", name.lower())
    slug = re.sub(r"[\s_]+", "-", slug.strip())

    return re.sub(r"-+", "-", slug)


def _parse_type_line(type_line: str) -> tuple[str, Rarity, bool, str | None]:
    """Split an SRD italic type line into (item_type, rarity, attunement, by).

    The lowest rarity present is used so that bonus-scaling items
    (e.g. 'Uncommon (+1), Rare (+2), or Very Rare (+3)') resolve to a single
    Wiki-compatible rarity.
    """
    requires_attunement = "Requires Attunement" in type_line

    attunement_by = None
    by_match = re.search(r"Requires Attunement by ([^)]+)", type_line)
    if by_match:
        attunement_by = by_match.group(1).strip()

    # Everything before the first top-level comma is the item type. Parenthetical
    # qualifiers can themselves contain commas, so respect parenthesis depth.
    depth = 0
    split_at = len(type_line)
    for i, char in enumerate(type_line):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            split_at = i
            break
    item_type = type_line[:split_at].strip()

    # The remainder holds the rarity. Bonus-scaling items list several tiers
    # ("Uncommon (+1), Rare (+2), ..."); "Rarity Varies" items list none.
    remainder = re.sub(r"\([^)]*\)", "", type_line[split_at:])
    if "Varies" in remainder:
        rarity = Rarity.VARIES
    else:
        present = [r for r in _RARITIES if r in remainder]
        if not present:
            raise ValueError(f"No rarity found in type line: {type_line!r}")
        rarity = min(present, key=lambda r: list(Rarity).index(r))

    return item_type, rarity, requires_attunement, attunement_by


def parse_magic_item(block: str) -> MagicItem:
    """Parse a single ``### Name`` block (heading already stripped) into a model."""
    name, _, rest = block.partition("\n")
    name = name.strip()

    type_match = _TYPE_LINE.search(rest)
    if not type_match:
        raise ValueError(f"No italic type line found for item: {name!r}")
    item_type, rarity, requires_attunement, attunement_by = _parse_type_line(
        type_match.group(1).strip()
    )

    # Body is everything after the type line, trimmed of surrounding whitespace.
    body = rest[type_match.end() :].strip()

    return MagicItem(
        slug=slugify(name),
        name=name,
        item_type=item_type,
        rarity=rarity,
        requires_attunement=requires_attunement,
        attunement_by=attunement_by,
        body=body,
    )


def parse_magic_items(markdown: str) -> list[MagicItem]:
    """Parse the full ``magic_items.md`` document into a list of models."""
    blocks = _ITEM_SPLIT.split(markdown)
    # blocks[0] is the document intro before the first item heading.
    return [parse_magic_item(block) for block in blocks[1:]]
