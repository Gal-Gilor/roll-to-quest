"""Build the Wiki magic-items dataset from the SRD markdown.

Parses ``magic_items.md`` into :class:`~src.wiki.models.MagicItem` records and
writes them to ``data/magic_items.json`` as camelCase JSON. The output is a
drop-in for the Fabled Campaigns Wiki's ``data/magic-items.json``.

Bundled SRD entries (the +1/+2/+3 weapons and armor, Horn of Valhalla's metal
variants) are expanded into one row per variant by the parser; catalog entries
(Ioun Stone, Spell Scroll, ...) stay as a single row with rarity "Varies".

Example Usage:
    poetry run python -m src.scripts.build_magic_items
"""

import json
from pathlib import Path

from src.wiki.parser import parse_magic_items

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCE = _REPO_ROOT / "magic_items.md"
_OUTPUT = _REPO_ROOT / "data" / "magic_items.json"


def main() -> None:
    """Parse the SRD markdown and write the magic-items JSON dataset."""
    items = parse_magic_items(_SOURCE.read_text(encoding="utf-8"))

    # ``by_alias`` emits camelCase keys; ``exclude_none`` omits ``attunementBy``
    # when absent, matching the Wiki's optional field convention.
    rows = [item.model_dump(by_alias=True, exclude_none=True) for item in items]
    _OUTPUT.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"Wrote {len(rows)} magic items to {_OUTPUT.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
