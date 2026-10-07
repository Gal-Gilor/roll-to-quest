"""Split the SRD Monsters A-Z / Animals markdown into one record per stat block.

The source (`monster_az.md`) nests headings as:
    ## Monsters A-Z | ## Animals   -> category
    ### White Dragons              -> group
    #### Adult White Dragon        -> one stat block
    ##### Traits / Actions / ...   -> sections inside the stat block
"""

import re
from dataclasses import dataclass
from typing import Literal

CATEGORIES: dict[str, Literal["Monster", "Animal"]] = {
    "Monsters A-Z": "Monster",
    "Animals": "Animal",
}

_HEADING = re.compile(r"^(#{2,4}) (.+)$")


@dataclass(frozen=True)
class StatBlock:
    name: str
    category: Literal["Monster", "Animal"]
    group: str
    text: str


def split_stat_blocks(markdown: str) -> list[StatBlock]:
    """Return the stat blocks in source order.

    Each block's `text` runs from its `#### ` line up to the next `#### `, `### `
    or `## ` line.
    """
    blocks: list[StatBlock] = []
    category = group = name = None
    lines: list[str] = []

    def close() -> None:
        if name is not None:
            blocks.append(StatBlock(name, category, group, "\n".join(lines).strip()))

    for line in markdown.splitlines():
        match = _HEADING.match(line)
        if not match:
            if name is not None:
                lines.append(line)
            continue

        level, title = len(match.group(1)), match.group(2).strip()
        close()
        name, lines = None, []
        if level == 2:
            if title not in CATEGORIES:
                raise ValueError(f"unexpected chapter heading: {title!r}")
            category, group = CATEGORIES[title], None
        elif level == 3:
            group = title
        else:
            if category is None or group is None:
                raise ValueError(f"stat block {title!r} is outside a chapter or group")
            name, lines = title, [line]

    close()
    return blocks
