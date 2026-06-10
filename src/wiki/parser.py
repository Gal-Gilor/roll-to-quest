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
# Strips only the trailing "(Requires Attunement ...)" clause, leaving variant
# qualifiers like "(+1)" or "(Bronze)" that are part of the rarity intact.
_ATTUNEMENT_CLAUSE = re.compile(r"\s*\(Requires Attunement[^)]*\)")
# A rarity clause segment shaped like "Tier (label)", e.g. "Uncommon (+1)" or
# "Rare (Silver or Brass)". 'Very Rare' must precede 'Rare' in the alternation.
_RARITY_TIERS = "Very Rare|Uncommon|Common|Legendary|Artifact|Rare"
_TIER_VARIANT = re.compile(rf"({_RARITY_TIERS})\s*\(([^)]+)\)")
_LONE_TIER = re.compile(rf"\b({_RARITY_TIERS})\b")


def slugify(name: str) -> str:
    """Convert an item name into a kebab-case slug (e.g. 'Staff of the Magi')."""
    slug = re.sub(r"[^\w\s-]", "", name.lower())
    slug = re.sub(r"[\s_]+", "-", slug.strip())

    return re.sub(r"-+", "-", slug)


def _parse_type_line(type_line: str) -> tuple[str, str, bool, str | None]:
    """Split an SRD italic type line into (item_type, rarity_clause, attunement, by).

    ``rarity_clause`` is kept verbatim (minus the attunement note) so a caller can
    decide whether it names a single tier, several variants, or 'Rarity Varies'.
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

    # Everything after that comma is the rarity clause; drop only the attunement note.
    rarity_clause = _ATTUNEMENT_CLAUSE.sub("", type_line[split_at + 1 :]).strip()
    if not rarity_clause:
        raise ValueError(f"No rarity found in type line: {type_line!r}")

    return item_type, rarity_clause, requires_attunement, attunement_by


def _variant_name(base_name: str, label: str) -> str:
    """Build a variant's display name from the base name and its label.

    Bonus labels keep the SRD's comma form ('Wand of the War Mage, +1'); other
    labels (metal types, etc.) are parenthesized ('Horn of Valhalla (Bronze)').
    """
    if label.startswith("+"):
        return f"{base_name}, {label}"

    return f"{base_name} ({label})"


def _resolve_rarities(name: str, rarity_clause: str) -> list[tuple[str, Rarity]]:
    """Resolve a rarity clause into a list of (display_name, rarity) variants.

    - Multi-variant clauses ('Uncommon (+1), Rare (+2), ...', 'Rare (Silver or
      Brass), ...') expand into one variant per 'Tier (label)' segment.
    - 'Rarity Varies' catalog entries collapse to a single ``Varies`` variant.
    - A plain single tier yields one variant with the original name.
    """
    variants = _TIER_VARIANT.findall(rarity_clause)
    if len(variants) >= 2:
        # The base name drops the SRD's bundled bonus suffix, e.g.
        # 'Wand of the War Mage, +1, +2, or +3' -> 'Wand of the War Mage'.
        base_name = name.split(", +")[0]
        return [
            (_variant_name(base_name, label.strip()), Rarity(tier))
            for tier, label in variants
        ]

    if "Varies" in rarity_clause:
        return [(name, Rarity.VARIES)]

    lone = _LONE_TIER.search(rarity_clause)
    if not lone:
        raise ValueError(f"No rarity tier resolved from {rarity_clause!r} for {name!r}")

    return [(name, Rarity(lone.group(1)))]


def parse_magic_item(block: str) -> list[MagicItem]:
    """Parse a ``### Name`` block (heading stripped) into one model per variant."""
    name, _, rest = block.partition("\n")
    name = name.strip()

    type_match = _TYPE_LINE.search(rest)
    if not type_match:
        raise ValueError(f"No italic type line found for item: {name!r}")
    item_type, rarity_clause, requires_attunement, attunement_by = _parse_type_line(
        type_match.group(1).strip()
    )

    # Body is everything after the type line, trimmed of surrounding whitespace.
    body = rest[type_match.end() :].strip()

    return [
        MagicItem(
            slug=slugify(variant_name),
            name=variant_name,
            item_type=item_type,
            rarity=rarity,
            requires_attunement=requires_attunement,
            attunement_by=attunement_by,
            body=body,
        )
        for variant_name, rarity in _resolve_rarities(name, rarity_clause)
    ]


def parse_magic_items(markdown: str) -> list[MagicItem]:
    """Parse the full ``magic_items.md`` document into a flat list of models."""
    blocks = _ITEM_SPLIT.split(markdown)
    # blocks[0] is the document intro before the first item heading.
    return [item for block in blocks[1:] for item in parse_magic_item(block)]
