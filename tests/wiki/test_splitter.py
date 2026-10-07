from collections import Counter
from pathlib import Path

import pytest

from src.wiki.models import slugify
from src.wiki.splitter import split_stat_blocks

SOURCE = Path(__file__).parents[2] / "monster_az.md"
BLOCKS = split_stat_blocks(SOURCE.read_text())
BY_NAME = {block.name: block for block in BLOCKS}


def test_counts_every_stat_block():
    assert len(BLOCKS) == 330
    assert Counter(block.category for block in BLOCKS) == {"Monster": 235, "Animal": 95}


def test_slugs_are_unique():
    assert len({slugify(block.name) for block in BLOCKS}) == len(BLOCKS)


@pytest.mark.parametrize(
    ("name", "category", "group"),
    [
        ("Aboleth", "Monster", "Aboleth"),
        ("Adult White Dragon", "Monster", "White Dragons"),
        ("Bandit Captain", "Monster", "Bandits"),
        ("Allosaurus", "Animal", "Allosaurus"),
    ],
)
def test_assigns_category_and_group(name, category, group):
    block = BY_NAME[name]
    assert (block.category, block.group) == (category, group)


def test_block_text_stops_at_next_stat_block():
    text = BY_NAME["Bandit"].text
    assert text.startswith("#### Bandit\n")
    assert "#### Bandit Captain" not in text


def test_rejects_unknown_chapter():
    with pytest.raises(ValueError):
        split_stat_blocks("## Spells\n\n### Fireball\n\n#### Fireball\n")
